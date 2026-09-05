from contextlib import nullcontext
from dataclasses import dataclass

import torch.optim

from model.model import Config
from model.model import YuchenModelCausalLLM
from train.train_util import *
import torch.nn as nn
from .SFT import SFTDataSet
from transformers import AutoTokenizer
from pathlib import Path


@dataclass
class LoraConfig:
    project_dir: Path = Path(r"D:\Kimi")

    tokenizer_dir: str = "BPEmodel"  # 分词器
    data_file: str = "data/lora_exam.jsonl"  # 训练数据
    save_dir: str = "train/mid_train/weight/lora_weight"  # 保存目录/检查点

    # None 表示使用全部数据；调试时可以设为 1000
    max_samples: int | None = 90000

    max_length: int = 512
    batch_size: int = 8
    accumulation_steps: int = 4
    epochs: int = 1

    lr: float = 1e-4
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
    # 首次编译自定义 MoE 的前向/反向可能耗时很久；确认训练正常后再开启。
    use_compile: bool = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16


    #Lora
    rank :int = 8
    alpha:int = 32
    dropout:float = 0.5
    # model.q 会被残差逻辑直接读取 .weight，不能替换成包装模块。
    exclude_modules: tuple[str, ...] = ("model.q",)

    # 字段存原料,属性存成品
    @property
    def tokenizer_path(self):
        return self.project_dir / self.tokenizer_dir

    @property
    def data_path(self):
        return self.project_dir / self.data_file

    @property
    def save_path(self):
        return self.project_dir / self.save_dir

    @property
    def max_len(self):
        return self.max_length

class LoraLinear(nn.Module):
    def __init__(self,cfg:LoraConfig,basic_layer:nn.Linear):
        super().__init__()
        self.basic_layer = basic_layer
        factory_kwargs = {
            "device": basic_layer.weight.device,
            "dtype": basic_layer.weight.dtype,
        }
        self.W_a = nn.Linear(basic_layer.in_features, cfg.rank, bias=False, **factory_kwargs)
        self.W_b = nn.Linear(cfg.rank, basic_layer.out_features, bias=False, **factory_kwargs)
        self.scale = cfg.alpha/cfg.rank
        self.dropout = nn.Dropout(cfg.dropout)

        # B 从 0 开始，确保刚注入 LoRA 时模型输出与 SFT 权重完全一致。
        nn.init.kaiming_uniform_(self.W_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.W_b.weight)
        for p in basic_layer.parameters():
            p.requires_grad = False
    def forward(self,x):
        return self.basic_layer(x) + self.scale*self.W_b(self.W_a(self.dropout(x)))


def replace_lora(module, cfg: LoraConfig, prefix=""):
    replace = 0
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, LoraLinear):
            continue
        if isinstance(child, nn.Linear) and full_name not in cfg.exclude_modules:
            setattr(module, name, LoraLinear(cfg, child))
            replace += 1
        else:
            replace += replace_lora(child, cfg, full_name)
    return replace


def make_lora_trainable(module:nn.Module):
    for param in module.parameters():
        param.requires_grad = False
    for mod in module.modules():
        if isinstance(mod,LoraLinear):
            mod.W_a.weight.requires_grad = True
            mod.W_b.weight.requires_grad = True


def train_epoch(epoch, auto_cast, model, loader, optimizer: torch.optim.Optimizer,
                iter, lora_config: LoraConfig, start_step=0, wandb=None):
    start_time = time.time()
    for local_step, (input_ids, labels) in enumerate(loader):
        # DataLoader 在断点续训后会从 0 重新计数，这里恢复为 epoch 内真实步数。
        step = start_step + local_step

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


        should_update = (step + 1) % lora_config.accumulation_steps == 0 or step == iter - 1
        if should_update:
            torch.nn.utils.clip_grad_norm_(model.parameters(),lora_config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if ((step + 1) % lora_config.save_interval == 0 or step == iter - 1) and is_main_process():
                lm_check_point(lora_config, weight="lora_weight", model=model, optimizer=optimizer,
                               epoch=epoch, step=step + 1)

        # 日志不能放在 should_update 内，否则 accumulation_steps=4、log_interval=10
        # 时两个条件永远错开，直到 epoch 最后一步都不会打印或上传。
        if step % lora_config.log_interval == 0 or step == iter - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * lora_config.accumulation_steps
            current_aux_loss = res.aux_loss.item() if lora_config.use_moe else 0.0
            current_logits_loss = res.lm_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            processed_steps = max(1, local_step + 1)
            remaining_steps = max(0, iter - step - 1)
            eta_time = spend_time / processed_steps * remaining_steps / 60
            Logger(
                f"轮数:{epoch + 1}/{lora_config.epochs},当前步数:{step},损失:{current_loss},辅助损失:{current_aux_loss}"
                f",纯语言模型损失:{current_logits_loss},学习率:{current_lr},剩余时间:{eta_time}分钟")
            if wandb is not None:
                wandb.log({
                    "loss": current_loss,
                    "aux_loss": current_aux_loss,
                    "logits_loss": current_logits_loss,
                    "lr": current_lr,
                    "eta_time": eta_time
                }, step=epoch * iter + step)

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

    model = YuchenModelCausalLLM(model_config).to(device=lora_config.device, dtype=lora_config.dtype)

    get_model_params(model, model_config)

    sft_path = lora_config.project_dir/"train"/"mid_train"/"weight"/"sft_weight"/"sft_weight_512_moe.pth"

    state_dict = torch.load(sft_path,map_location="cpu")

    model.load_state_dict(state_dict,strict=True)

    replaced_count = replace_lora(model, lora_config)
    make_lora_trainable(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    Logger(
        f"已注入 {replaced_count} 个 LoRA 层，可训练参数: {trainable_params:,}/"
        f"{total_params:,} ({trainable_params / total_params:.2%})"
    )

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lora_config.lr,weight_decay=lora_config.weight_decay)
    auto_cast = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda",dtype=lora_config.dtype)


    ckp_data = lm_check_point(
        lm_config=lora_config, weight="lora_weight", optimizer=optimizer
    ) if lora_config.resume else None

    wandb_run = None
    if lora_config.use_wandb:
        if wandb is None:
            raise RuntimeError("use_wandb=True，但当前环境没有安装 wandb")
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "allow" if wandb_id else None
        wandb_run_name = f"YuchenModel-LoRA-Epoch-{lora_config.epochs}-BatchSize-{lora_config.batch_size}-LearningRate-{lora_config.lr}"
        wandb_run = wandb.init(
            project=lora_config.project_name,
            id=wandb_id,
            name=wandb_run_name,
            resume=resume,
            config={
                "rank": lora_config.rank,
                "alpha": lora_config.alpha,
                "dropout": lora_config.dropout,
                "batch_size": lora_config.batch_size,
                "accumulation_steps": lora_config.accumulation_steps,
                "learning_rate": lora_config.lr,
                "trainable_params": trainable_params,
                "lora_layers": replaced_count,
            }
        )

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

    try:
        for epoch in range(start_epoch, lora_config.epochs):
            set_seed(lora_config.seed+epoch)

            skip = start_step if (epoch == start_epoch and start_step > 0) else 0


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
                Logger(f'Epoch [{epoch + 1}/{lora_config.epochs}]: 跳过前{skip}个step，从step {skip + 1}开始')
            train_epoch(epoch=epoch, auto_cast=auto_cast, model=model, loader=loader,
                        optimizer=optimizer, iter=steps_per_epoch, lora_config=lora_config,
                        start_step=skip, wandb=wandb_run)
            start_step = 0
    finally:
        if wandb_run is not None:
            wandb_run.finish()
