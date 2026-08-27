from contextlib import nullcontext

import torch
from model.model import Config
from pathlib import Path
from datasets import Features, List, Value
import json
from transformers import AutoTokenizer
from dataclasses import dataclass
from model.model import YuchenModelCausalLLM
from train.train_util import *
@dataclass
class SFTConfig:
    project_dir: Path = Path(r"/")

    tokenizer_dir: str = "BPEmodel" #分词器
    data_file: str = "..." #训练数据
    checkpoint_dir: str = "..."  #检查点目录
    save_path: str = "..."  #保存路径

    # None 表示使用全部数据；调试时可以设为 1000
    max_samples: int | None = 90000

    max_length: int = 512
    batch_size: int = 16
    accumulation_steps: int = 2
    epochs: int = 1

    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    num_workers: int = 8
    seed: int = 42

    save_interval: int = 2000
    log_interval: int = 10

    dtype: torch.dtype = torch.bfloat16

    # 是否加载 checkpoint
    resume: bool = True

    use_wandb: bool = True
    project_name:str = "YuchenModel_sft"
    use_moe: bool = True
    hidden_size: int = 512
    use_compile: bool = True



    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

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



SFT_FEATURES = Features({
    "conversations": List({
        "role": Value("string"),
        "content": Value("string"),
        "reasoning_content": Value("string"),
        "tools": Value("string"),
        "tool_calls": Value("string"),
    })
})


class SFTDataSet(Dataset):
    def __init__(self,json_path,tokenizer,max_len=512):
        """

        :param json_path: 数据集路径
        :param tokenizer: 分词器AutoTokenizer
        :param max_len: 最大长度
        """

        super().__init__()
        self.tokenizer = tokenizer
        self.json_path = json_path
        self.max_len = max_len
        #使用HuggingFace的load_dataset加载数据
        #他使用的是懒加载,即需要哪条加载哪条
        self.sample = load_dataset("json",data_files=json_path,split="train",features=SFT_FEATURES)
        #获取tokenizer已经编码好的特殊编码
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n",add_special_tokens = False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}",add_special_tokens = False).input_ids
    def __len__(self):
        return len(self.sample)

    def create_chat_prompt(self, cs):
        def parse_json_field(value):
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value

        message = []
        for item in cs:
            item = dict(item)

            if item.get("tools") is not None:
                item["tools"] = parse_json_field(item["tools"])

            if item.get("tool_calls") is not None:
                item["tool_calls"] = parse_json_field(item["tool_calls"])

            message.append(item)

        tools = None
        if message and message[0]["role"] == "system":
            tools = message[0].pop("tools", None)

        return self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )
    def create_labels(self,input_ids):
        """
        双指针去找labels
        """

        # 创建标签，初始化全为-100,后续用input_ids的值覆盖
        labels = [-100] * len(input_ids)
        i = 0
        while i <len(input_ids):
            #当找到bos_id的时候(回答开始)
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                start = i+len(self.bos_id)
                end = start
                #去找回答结束的标志(eos_id)
                while end < len(input_ids):
                    if input_ids[end:end+len(self.eos_id)] == self.eos_id:
                        break
                    else:
                        end+=1
                #替换label中的值
                for j in range(start,min(end+len(self.eos_id),self.max_len)):
                    labels[j] = input_ids[j]
                i = end+len(self.eos_id) if i < len(input_ids) else len(input_ids)
            #没找到bos就继续往后找
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        """

        :param index: json数据集中的样本索引
        :return:
        """
        #获取原始数据
        sample = self.sample[index]
        #渲染prompt字符串,字符串
        prompt = self.create_chat_prompt(sample["conversations"])
        #分词并且截取最大长度的token,列表
        input_ids = self.tokenizer(prompt).input_ids[:self.max_len]



        #padding
        input_ids += [self.tokenizer.pad_token_id] * (self.max_len-len(input_ids))

        #生成标签
        labels = self.create_labels(input_ids)
        # # === 调试代码 在正式训练前取消注释跑一次 ===
        # # 作用：人工肉眼检查 Mask 是否正确。
        # # 也就是看 User 的部分对应的 Label 是否真的是 -100。
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     # 打印 Input Token 和 对应的 Label (注意这里模拟了 Next Token Prediction 的错位)
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")

        return torch.tensor(input_ids,dtype=torch.long),torch.tensor(labels,dtype=torch.long)




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




def train_epoch(epoch,iter,sft_config,wandb=None):
    """

    :param epoch: 当前训练轮数
    :param iter: 一个epoch的步数
    :param start_step: 开始步数(用于skip)
    :param wandb:
    :return:
    """
    start_time = time.time()
    for step,(input_idx,labels) in enumerate(loader):
        #1搬运数据到GPU
        #Embedding查表只需要Long/Int(因为行号只能用整数),所以这里不需要转换dtype
        #non_blocking=True,使用异步数据搬运,当GPU在做数据运算的时候,CPU可以继续做数据搬运
        input_idx,labels = input_idx.to(device = sft_config.device,non_blocking=True),labels.to(device = sft_config.device,non_blocking=True)


        #2.动态调整学习率
        lr = get_lr(current_step=epoch*iter+step,
                    total_step=sft_config.epochs*iter,
                    lr = sft_config.lr,
                    warmup_ratio=sft_config.warmup_ratio,
                    min_lr_ratio=sft_config.min_lr_ratio)
        for param in optimizer.param_groups:
            param['lr'] = lr



        #3.前向传播(混合精度上下文)
        with auto_cast:
            res = model(input_ids=input_idx,labels=labels)
            loss = res.loss
            #梯度累加,不会导致梯度过大
            loss = loss/sft_config.accumulation_steps



        #4.反向传播
        loss.backward()

        #5.梯度更新
        if(step+1)%sft_config.accumulation_steps == 0 or step == iter-1:
            #梯度剪裁防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), sft_config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        #6.打印日志
        if step%sft_config.log_interval == 0 or step == iter-1:
            spend_time = time.time() - start_time
            #梯度剪裁会使得当前的loss变小，所以需要乘以accumulation_steps
            current_loss = loss.item()*sft_config.accumulation_steps
            current_aux_loss = res.aux_loss.item() if sft_config.use_moe else ""
            #纯语言模型loss
            current_logits_loss = res.lm_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_time =  spend_time/(step+1)*iter//60 - spend_time//60
            Logger(f"轮数:{epoch}/{sft_config.epochs},当前步数:{step},损失:{current_loss},辅助损失:{current_aux_loss},纯语言模型损失:{current_logits_loss},学习率:{current_lr},剩余时间:{eta_time}分钟")
            if wandb is not None:
                wandb.log({
                    "loss": current_loss,
                    "aux_loss": current_aux_loss,
                    "logits_loss": current_logits_loss,
                    "lr": current_lr,
                    "eta_time": eta_time
                })
        #7.保存模型
        if (step%sft_config.save_interval == 0 or step == iter-1) and is_main_process():

            lm_check_point(sft_config, weight="sft_weight", model=model, optimizer=optimizer,
                          epoch=epoch, step=step)

        del input_idx,labels,res,loss





if __name__ == "__main__":

    #cfg
    model_config = Config()
    sft_config = SFTConfig()


    set_seed(sft_config.seed)

    #tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(sft_config.tokenizer_path))
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer 没有 pad_token")
    if tokenizer.bos_token_id is None:
        raise ValueError("Tokenizer 没有 bos_token")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer 没有 eos_token")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前设备:{device}")

    dataset = SFTDataSet(str(sft_config.data_path), tokenizer,max_len=sft_config.max_len)
    #调试代码
    # for i in range(10):
    #     input_ids,labels = dataset[i]
    Logger(f"监督微调数据集大小:{len(dataset)-1}")


    #模型
    model = build_model(tokenizer, sft_config).to(device=device,dtype=sft_config.dtype)
    get_model_params(model,model_config)
    # 加载预训练权重：这里只加载模型参数，不加载预训练的优化器状态
    pretrained_path = Path(r"/train/weight/pretrain_weight\pretrain_weight_512_moe.pth")

    state_dict = torch.load(pretrained_path,map_location="cpu",weights_only=True,)

    model.load_state_dict(state_dict, strict=True)


    #训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=sft_config.lr, weight_decay=sft_config.weight_decay)
    auto_cast = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda",dtype=sft_config.dtype)

    ckp_data = lm_check_point(lm_config=sft_config,weight="sft_weight",optimizer=optimizer) if sft_config.resume else None

    #wandb
    wandb_run = None
    if sft_config.use_wandb:
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "allow" if wandb_id else None
        wandb_run_name = f"YuchenModel-Full-SFT-Epoch-{sft_config.epochs}-BatchSize-{sft_config.batch_size}-LearningRate-{sft_config.lr}"
        wandb_run = wandb.init(project = sft_config.project_name,id = wandb_id,name = wandb_run_name,resume = resume)



    #resume
    start_epoch,start_step = 0,0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step",0)
        Logger(f"从第 {start_epoch}轮重新加载, step:  {start_step}")



    if sft_config.use_compile:
        model = torch.compile(model)#图编译加速器
        Logger("使用compile")


    for epoch in range(start_epoch,sft_config.epochs):
        #重新生成索引,用于SkipSampler
        set_seed(sft_config.seed+epoch)

        #如果断点续训需要跳过的步数
        skip = start_step if (epoch== start_epoch and start_step>0) else 0

        generator = torch.Generator()
        generator.manual_seed(sft_config.seed + epoch)
        sampler = torch.utils.data.RandomSampler(dataset,generator=generator)
        batch_sampler = SkipBatchSimple(sampler,sft_config.batch_size, skip)
        #persistent_worker=True,dataLoader的子进程在一个epoch之后结束之后不会被销毁,下一轮继续复用
        #prefectch_factor = 4,每个worker预先准备4个batch,放进队列等待GPU消费
        loader = DataLoader(dataset, batch_sampler=batch_sampler,num_workers=sft_config.num_workers,
                            pin_memory=True,persistent_workers=True,prefetch_factor=4)

        steps_per_epoch = (len(dataset) + sft_config.batch_size - 1) // sft_config.batch_size


        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{sft_config.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch=epoch,iter=steps_per_epoch,sft_config=sft_config,wandb=wandb_run)
        else:
            train_epoch(epoch=epoch,iter=steps_per_epoch, sft_config=sft_config,wandb=wandb_run)
