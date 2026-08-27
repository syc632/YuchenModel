from train.train_util import *
import math
import json
from contextlib import nullcontext
from model.model import Config, YuchenModelCausalLLM
try:
    import wandb  #日志
except ImportError:
    wandb = None
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer
from pathlib import Path

@dataclass
class TrainConfig:
    project_dir: Path = Path(r"/")

    tokenizer_dir: str = "BPEmodel"
    data_file: str = "..."
    checkpoint_dir: str = "..."
    save_path: str = "..."

    # None 表示使用全部数据；调试时可以设为 1000
    max_samples: int | None = 20_000

    max_length: int = 256
    batch_size: int = 2
    accumulation_steps: int = 16
    epochs: int = 1

    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Windows 建议先用 0
    num_workers: int = 2
    seed: int = 42

    save_interval: int = 100
    log_interval: int = 10

    # "bfloat16"、"float16" 或 "float32"
    dtype: str = "bfloat16"

    # 是否加载 checkpoint
    resume: bool = True

    use_wandb: bool = True
    project_name:str = "YuchenModel"
    use_moe: bool = True
    hidden_size: int = 512
    use_compile: bool = True

    #字段存原料,属性存成品
    @property
    def tokenizer_path(self):
        return self.project_dir/self.tokenizer_dir

    @property
    def data_path(self):
        return self.project_dir/self.data_file

    @property
    def max_len(self):
        return self.max_length


class PretrainData(Dataset):
    def __init__(self,data_path,tokenzier,max_len,max_samples=None):
        self.max_len = max_len
        self.tokenizer = tokenzier
        #优先使用datasets读取json数据；没有安装datasets时退回到逐行读取jsonl
        if load_dataset is not None:
            self.sample = load_dataset("json",data_files=str(data_path),split="train")
            if max_samples is not None:
                self.sample = self.sample.select(
                    range(min(max_samples,len(self.sample)))
                )
        else:
            self.sample = []
            with open(data_path,"r",encoding="utf-8") as file:
                for line in file:
                    if max_samples is not None and len(self.sample)>=max_samples:
                        break
                    line = line.strip()
                    if line:
                        self.sample.append(json.loads(line))

    def __len__(self):
        #返回数据集的样本总量
        return len(self.sample)

    def __getitem__(self, index):
        #根据索引提取出文本数据."text"字段的内容
        sample = self.sample[index]
        text_content = str(sample["text"])
        #分词
        token = self.tokenizer(
            text_content,
            add_special_tokens=False,
            max_length=self.max_len-2,  #预留两个位置给BOS和EOS
            truncation=True,            #保证文本长度不超过max_len-2
        ).input_ids

        #加入bos和eos
        token = [self.tokenizer.bos_token_id] + token + [self.tokenizer.eos_token_id]

        #填充padding
        input_ids = token + [self.tokenizer.pad_token_id] * (self.max_len - len(token))
        input_ids = torch.tensor(input_ids,dtype = torch.long)
        #复制一份input_ids作为训练的标签
        label = input_ids.clone()
        #屏蔽padding标签
        label[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids,label

def build_model(tokenizer,train_config):
    #在统一的模型Config上覆盖本次预训练需要的配置
    model_config = Config()
    model_config.d_model = train_config.hidden_size
    model_config.embd = train_config.hidden_size
    model_config.d_head = model_config.d_model//model_config.n_head
    model_config.vocab_size = len(tokenizer)
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.bos_token_id = tokenizer.bos_token_id
    model_config.eos_token_id = tokenizer.eos_token_id
    model_config.use_moe = train_config.use_moe
    model_config.use_attn_res = True
    return YuchenModelCausalLLM(model_config)


def train_epoch(
    epoch,loader,current_step,model,optimizer,scaler,device,auto_cast,
    config,total_update_steps,start_step=0
):
    start_time = time.time()
    model.train()
    #开始一个epoch前把梯度清空,后续按照accumulation_steps累计
    optimizer.zero_grad(set_to_none=True)
    for step,(input_ids,labels) in enumerate(loader):
        if step < start_step:
            continue

        #把输入和标签真正移动到训练设备;Tensor.to不会原地修改,所以必须接收返回值
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        #根据当前优化器更新次数计算本步学习率
        lr = get_lr(current_step,total_update_steps,config.lr,config.warmup_ratio,config.min_lr_ratio)
        #手动更新优化器中所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        #进入自动混合精度上下文
        with auto_cast:
            res = model(input_ids=input_ids,labels=labels)
            #模型内部已经把aux_loss按系数加入loss,这里不再重复相加
            loss = res.loss/config.accumulation_steps

        #Scaler在FP16下放大loss再反向传播,防止小梯度下溢
        scaler.scale(loss).backward()

        #累计够指定步数或已经到达最后一个batch时执行一次参数更新
        should_update = (
            (step+1) % config.accumulation_steps == 0
            or step+1 == len(loader)
        )
        if should_update:
            #先还原梯度尺度,再做梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip)
            #真正执行optimizer.step;原来的代码缺少这一步时参数不会更新
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            current_step += 1

        #每隔log_interval步记录一次控制台和WandB日志
        if step%config.log_interval == 0:
            spend_time = time.time() - start_time
            current_loss = res.loss.item()
            current_aux_loss = res.aux_loss.item()
            current_logits_loss = res.lm_loss.item()
            #预计本轮剩余训练时间（分钟）
            processed_steps = max(1,step-start_step+1)
            remaining_steps = max(0,len(loader)-step-1)
            eta_time = spend_time/processed_steps*remaining_steps/60
            Logger(
                f'Epoch:[{epoch + 1}/{config.epochs}]({step}/{len(loader)}),'
                f' loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, '
                f'lr: {lr:.8f}, epoch_time: {eta_time:.1f}min'
            )
            if config.use_wandb and wandb is not None and wandb.run is not None:
                wandb.log({
                    "loss":current_loss,
                    "logits_loss":current_logits_loss,
                    "aux_loss":current_aux_loss,
                    "learning_rate":lr,
                    "epoch_time":eta_time,
                },step=current_step)

        #只在完成参数更新后保存,并且仅由主进程写checkpoint
        if should_update and current_step%config.save_interval==0 and is_main_process():
            lm_check_point(
                config,weight="pretrain_weight",model=model,optimizer=optimizer,
                epoch=epoch,step=step+1,scaler=scaler,current_step=current_step
            )

    return current_step


def main():
    config = TrainConfig()
    set_seed(config.seed)
    #优先使用CUDA,没有GPU时退回CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    #CPU训练统一使用FP32;GPU根据配置选择FP32、FP16或BF16
    model_dtype = dtype if device =="cuda" else torch.float32
    #创建自动混合精度上下文;CPU或FP32训练不需要autocast
    auto_cast = nullcontext() if device == "cpu" else torch.amp.autocast(dtype=dtype,device_type=device)

    ckp_data = None
    #尝试恢复上次训练保存的完整状态
    if config.resume:
        ckp_data = lm_check_point(model=None,weight="pretrain_weight",lm_config=config)

    #从本地目录加载已经训练好的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config.tokenizer_path))
    #确保三个特殊token都存在
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer 没有 pad_token")
    if tokenizer.bos_token_id is None:
        raise ValueError("Tokenizer 没有 bos_token")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer 没有 eos_token")

    #创建预训练数据集
    train_dataset = PretrainData(
        config.data_path,tokenizer,config.max_len,config.max_samples
    )
    #构建模型并一次性移动到目标设备和精度
    model = build_model(tokenizer,config).to(device=device,dtype=model_dtype)
    #优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),lr=config.lr,eps=1e-8,
        weight_decay=config.weight_decay,betas=(0.9,0.95)
    )
    #仅FP16需要启用GradScaler;BF16通常不需要
    scaler = torch.amp.GradScaler(
        "cuda",enabled=(device  =="cuda" and config.dtype=="float16")
    )

    start_epoch,start_step,current_step = 0,0,0
    if ckp_data:
        #恢复模型、优化器、Scaler以及训练位置
        model.load_state_dict(ckp_data['model'])
        if ckp_data.get('optimizer') is not None:
            optimizer.load_state_dict(ckp_data['optimizer'])
        if ckp_data.get('scaler') is not None:
            scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data.get('epoch',0)
        start_step = ckp_data.get('step',0)
        current_step = ckp_data.get('current_step',0)

    #torch.compile会优化计算图;MoE动态路由调试阶段默认关闭
    if config.use_compile:
        model = torch.compile(model)

    #创建DataLoader;Windows环境下num_workers默认使用0更稳定
    dataloader = DataLoader(
        train_dataset,batch_size=config.batch_size,shuffle=True,
        num_workers=config.num_workers,pin_memory=device =="cuda",drop_last=True
    )
    #梯度累计后,每个epoch真正执行多少次optimizer更新
    update_per_epochs = math.ceil(len(dataloader)/config.accumulation_steps)
    total_update_steps = config.epochs*update_per_epochs

    Logger(f"设备:{device}")
    Logger(f"预训练数据集大小:{len(train_dataset)}")
    raw_model = model._orig_mod if hasattr(model,"_orig_mod") else model
    get_model_params(raw_model,raw_model.config)

    #只有主进程初始化WandB,避免多卡时生成重复run
    if config.use_wandb and wandb is not None and is_main_process():
        wandb.init(project=config.project_name)

    #从断点记录的epoch和batch继续训练
    for epoch in range(start_epoch,config.epochs):
        set_seed(config.seed+epoch)
        current_step = train_epoch(
            epoch,dataloader,current_step,model,optimizer,scaler,device,
            auto_cast,config,total_update_steps,
            start_step=start_step if epoch==start_epoch else 0,
        )
        start_step = 0

    #训练结束后再保存一次最终权重和完整训练状态
    if is_main_process():
        lm_check_point(
            config,weight="pretrain_weight",model=model,optimizer=optimizer,
            epoch=config.epochs,step=0,scaler=scaler,current_step=current_step
        )

    if config.use_wandb and wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
