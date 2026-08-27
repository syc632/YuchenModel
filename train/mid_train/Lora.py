from contextlib import nullcontext
from dataclasses import dataclass

import torch.optim

from model.model import Config
from model.model import YuchenModelCausalLLM
from train.train_util import *
import torch.nn as nn
from SFT import SFTDataSet
from transformers import AutoTokenizer
from pathlib import Path
@dataclass
class LoraConfig:
    project_dir: Path = Path(r"/")

    tokenizer_dir: str = "BPEmodel"  # 分词器
    data_file: str = "..."  # 训练数据
    checkpoint_dir: str = "..."  # 检查点目录
    save_path: str = "..."  # 保存路径

    # None 表示使用全部数据；调试时可以设为 1000
    max_samples: int | None = 90000

    max_length: int = 512
    batch_size: int = 8
    accumulation_steps: int = 4
    epochs: int = 1

    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    num_workers: int = 2
    seed: int = 42

    save_interval: int = 2000
    log_interval: int = 10

    dtype: torch.dtype = torch.bfloat16

    # 是否加载 checkpoint
    resume: bool = True

    use_wandb: bool = True
    project_name: str = "YuchenModel_Lora"
    use_moe: bool = True
    hidden_size: int = 512
    use_compile: bool = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16


    #Lora
    rank :int = 8
    alpha:int = 32
    dropout:float = 0.5
    d_model:int = 512

    # 字段存原料,属性存成品
    @property
    def tokenizer_path(self):
        return self.project_dir / self.tokenizer_dir

    @property
    def data_path(self):
        return self.project_dir / self.data_file

    @property
    def max_len(self):
        return self.max_length

class LoraLinear(nn.Module):
    def __init__(self,cfg:LoraConfig,basic_layer:nn.Linear):
        super().__init__()
        self.basic_layer = basic_layer
        self.W_a = nn.Linear(cfg.d_model,cfg.rank,bias=False)
        self.W_b = nn.Linear(cfg.rank,cfg.d_model,bias=False)
        self.scale = cfg.alpha/cfg.rank
        self.dropout = nn.Dropout(cfg.dropout)
        for p in basic_layer.parameters():
            p.requires_grad = False
    def forward(self,x):
        return self.basic_layer(x) + self.scale*self.W_b(self.W_a(self.dropout(x)))


def find_lora_model(model:YuchenModelCausalLLM):
    target_model = []
    for name,module in model.named_modules():
        if isinstance(module,nn.Linear):
            target_model.append(name)
    return target_model



def replace_lora(module,cfg:LoraConfig):
    target_module = find_lora_model(module)

    replace = 0
    for name,module in module.named_children():
        if name in target_module and isinstance(module,nn.Linear):
            setattr(module,name,LoraLinear(cfg,module))
            replace +=1
        else:
            replace += replace_lora(module,cfg)
    return replace
def make_lora_trainable(module:nn.Module):
    for param in module.parameters():
        param.requires_grad = False
    for mod in module.modules():
        if isinstance(mod,LoraLinear):
            mod.W_a.weight.requires_grad = True
            mod.W_b.weight.requires_grad = True

def train_epoch(epoch,auto_cast,model,loader,optimizer:torch.optim.Optimizer,iter,lora_config:LoraConfig,wandb=None):
    start_time = time.time()
    for step,(input_ids,labels) in enumerate(loader):

        input_ids,labels = input_ids.to(device=lora_config.device,non_blocking=True),labels.to(device = lora_config.device,non_blocking = True)


        lr = get_lr(current_step=iter*epoch +step,
                    total_step=lora_config.epochs*iter,
                    lr=lora_config.lr,
                    warmup_ratio=lora_config.warmup_ratio,
                    min_lr_ratio=lora_config.min_lr_ratio)
        for param in optimizer.param_groups:
            #param_groups:一个列表,每个元素都是一个字典
            param["lr"] = lr

        with auto_cast:
            res = model(input_ids=input_ids,labels=labels)
            loss = res.loss
            loss = loss/lora_config.accumulation_steps

        loss.backward()


        if (step+1)%lora_config.accumulation_steps == 0 or step == iter-1:
            torch.nn.utils.clip_grad_norm_(model.parameters(),lora_config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


            if step % lora_config.log_interval == 0 or step == iter - 1:
                spend_time = time.time() - start_time
                # 梯度剪裁会使得当前的loss变小，所以需要乘以accumulation_steps
                current_loss = loss.item() * lora_config.accumulation_steps
                current_aux_loss = res.aux_loss.item() if lora_config.use_moe else ""
                # 纯语言模型loss
                current_logits_loss = res.lm_loss.item()
                current_lr = optimizer.param_groups[-1]['lr']
                processed_steps = max(1, step - start_step + 1)
                remaining_steps = max(0, len(loader) - step - 1)
                eta_time = spend_time / processed_steps * remaining_steps / 60
                Logger(
                    f"轮数:{epoch}/{lora_config.epochs},当前步数:{step},损失:{current_loss},辅助损失:{current_aux_loss}"
                    f",纯语言模型损失:{current_logits_loss},学习率:{current_lr},剩余时间:{eta_time}分钟")
                if wandb is not None:
                    wandb.log({
                        "loss": current_loss,
                        "aux_loss": current_aux_loss,
                        "logits_loss": current_logits_loss,
                        "lr": current_lr,
                        "eta_time": eta_time
                    })

            if (step % lora_config.save_interval == 0 or step == iter - 1) and is_main_process():
                lm_check_point(lora_config, weight="sft_weight", model=model, optimizer=optimizer,
                               epoch=epoch, step=step)

            del input_ids, labels, res, loss


if __name__ == "__main__":
    model_config = Config()
    lora_config = LoraConfig()


    set_seed(lora_config.seed)

    tokenizer = AutoTokenizer.from_pretrained(str(lora_config.tokenizer_path))
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer 没有 pad_token")
    if tokenizer.bos_token_id is None:
        raise ValueError("Tokenizer 没有 bos_token")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer 没有 eos_token")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Logger(f"当前设备:{device}")

    dataset = SFTDataSet(str(lora_config.data_path), tokenizer, max_len=lora_config.max_len)
    # 调试代码
    # for i in range(10):
    #     input_ids,labels = dataset[i]
    Logger(f"Lora数据集大小:{len(dataset) - 1}")

    model = YuchenModelCausalLLM(model_config).to(device = lora_config.device,dtype= lora_config.dtype)

    get_model_params(model, model_config)

    sft_path = Path(
        r"/train/weight/pretrain_weight\pretrain_weight_512_moe.pth"
    )

    state_dict = torch.load(sft_path,map_location="cpu")

    model.load_state_dict(state_dict,strict=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lora_config.lr, weight_decay=lora_config.weight_decay)
    auto_cast = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda",dtype=lora_config.dtype)


    ckp_data = lm_check_point(lm_config=lora_config,weight=lora_config.save_path,optimizer=optimizer) if lora_config.resume else None

    wandb_run = None
    if lora_config.use_wandb:
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"YuchenModel-Full-SFT-Epoch-{lora_config.epochs}-BatchSize-{lora_config.batch_size}-LearningRate-{lora_config.lr}"
        wandb_run = wandb.init(project=lora_config.project_name, id=wandb_id, name=wandb_run_name, resume=resume)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)
        Logger(f"从第 {start_epoch}轮重新加载, step:  {start_step}")


    if lora_config.use_compile:
        model = torch.compile(model)#图编译加速器
        Logger("使用compile")

    for epoch in range(lora_config.epochs):
        set_seed(lora_config.seed+epoch)

        skip = start_step if (epoch== start_epoch and start_step>0) else 0


        generator = torch.Generator()
        generator.manual_seed(lora_config.seed + epoch)
        sampler = torch.utils.data.RandomSampler(dataset, generator=generator)

        batch_sampler = SkipBatchSimple(sampler, lora_config.batch_size, skip)

        loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=lora_config.num_workers,
                            pin_memory=True, persistent_workers=True, prefetch_factor=4)

        steps_per_epoch = (
                                  len(dataset) + lora_config.batch_size - 1
                          ) // lora_config.batch_size

        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{lora_config.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch=epoch, auto_cast=auto_cast, model=model, loader=loader,
                        optimizer=optimizer, iter=steps_per_epoch, lora_config=lora_config,
                        wandb=wandb_run)
        else:
            train_epoch(epoch=epoch, auto_cast=auto_cast, model=model, loader=loader,
                        optimizer=optimizer, iter=steps_per_epoch, lora_config=lora_config,
                        wandb=wandb_run)
