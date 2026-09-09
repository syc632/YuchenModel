from train.train_util import *
import math
import json
from contextlib import nullcontext
from model.model import Config, YuchenModelCausalLLM
try:
    import wandb  #日志
except ImportError:
    wandb = None
from dataclasses import asdict, dataclass, field
import torch
from torch.utils.data import DataLoader,Dataset,random_split
from transformers import AutoTokenizer
from pathlib import Path

@dataclass
class TrainConfig:
    project_dir: Path = Path(r"D:\Kimi")

    tokenizer_dir: str = ""
    data_file: str = ""
    val_file: str = ""  #可选独立验证JSONL；留空时从输入样本固定划分1%
    val_ratio: float = 0.01
    val_seed: int = 2026
    save_path: str = "weight/pretrain_gibc"  #新配置单独保存，避免覆盖旧模型

    # None 表示使用全部数据；调试时可以设为 1000
    max_samples: int | None = None

    max_length: int = 512
    batch_size: int = 2
    accumulation_steps: int = 16  #每次更新最多2*512*16=16384个输入token，含padding
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
    eval_interval: int = 100  #按成功的optimizer更新次数评估，epoch末尾也评估
    log_interval: int = 10

    # "bfloat16"、"float16" 或 "float32"
    dtype: str = "bfloat16"

    # 新配置从零开始训练，不加载旧架构checkpoint
    resume: bool = False

    use_wandb: bool = True
    project_name:str = "YuchenModel"
    use_moe: bool = True
    hidden_size: int = 512
    use_compile: bool = False
    model_config: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype必须为float32、float16或bfloat16")
        if min(self.batch_size, self.accumulation_steps, self.epochs,
               self.save_interval, self.eval_interval, self.log_interval) < 1 or self.max_length < 3:
            raise ValueError("batch、累积步数、epoch、日志间隔必须为正，max_length至少为3")
        if not self.val_file and not 0 < self.val_ratio < 1:
            raise ValueError("未指定val_file时，val_ratio必须在0和1之间")

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
    values = dict(d_model=train_config.hidden_size, use_moe=train_config.use_moe)
    #.update把tran_config.model_cofig这个字典里的所有键值对都合并覆盖到values中
    values.update(train_config.model_config)
    values.update(vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
                  bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    return YuchenModelCausalLLM(Config(**values))



def precision_context(device, dtype):
    #参数始终保留FP32；只有前向计算使用AMP，CPU和显式FP32不启用autocast。
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("未知训练精度")
    if torch.device(device).type != "cuda" or dtype == "float32":
        return nullcontext()
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("当前GPU不支持BF16，请将dtype改为float16或float32")
    return torch.autocast(device_type="cuda", dtype=getattr(torch, dtype))


def build_datasets(tokenizer, config):
    dataset = PretrainData(config.data_path, tokenizer, config.max_len, config.max_samples)
    if config.val_file:
        val_path = config.project_dir / config.val_file
        if val_path.resolve() == config.data_path.resolve():
            raise ValueError("训练文件和验证文件不能是同一个文件")
        validation = PretrainData(val_path, tokenizer, config.max_len)
        if not len(dataset) or not len(validation):
            raise ValueError("训练集和验证集都不能为空")
        return dataset, validation
    if len(dataset) < 2:
        raise ValueError("自动划分验证集至少需要两条样本")
    val_size = min(len(dataset)-1, max(1, int(len(dataset)*config.val_ratio)))
    #先按样本固定划分，再通过Dataset分词；训练和验证不共享样本索引。
    return random_split(dataset, [len(dataset)-val_size, val_size],
                        generator=torch.Generator().manual_seed(config.val_seed))


@torch.no_grad()
def evaluate(model, loader, device, dtype):
    was_training = model.training
    model.eval()
    total_nll, total_targets = 0.0, 0
    try:
        for input_ids, labels in loader:
            n = int(labels[:, 1:].ne(-100).sum())
            if not n:
                continue
            with precision_context(device, dtype):
                result = model(input_ids=input_ids.to(device), labels=labels.to(device))
            #只计算语言模型损失，不能把MoE辅助项加入PPL。
            ce = float(result.lm_loss)
            if not math.isfinite(ce):
                raise FloatingPointError("验证集出现非有限CE")
            total_nll += ce*n
            total_targets += n
    finally:
        model.train(was_training)
    if not total_targets:
        raise ValueError("验证集没有有效预测目标")
    ce = total_nll/total_targets
    return {"ce": ce, "ppl": math.exp(ce) if ce < 700 else None,
            "targets": total_targets}


def save_training_checkpoint(config, model, optimizer, scaler, epoch, step,
                             current_step, state, weight="pretrain_weight"):
    raw_model = getattr(model, "_orig_mod", model)
    lm_check_point(
        config, weight=weight, model=model, optimizer=optimizer,
        epoch=epoch, step=step, scaler=scaler, current_step=current_step,
        training_state=dict(state), model_config=asdict(raw_model.config),
        torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def validate_and_save(model, loader, optimizer, scaler, device, config,
                      epoch, step, current_step, state):
    #间隔评估恰好落在epoch末尾时，不重复评估同一份参数。
    if state.get("last_eval_step") == current_step:
        return
    metrics = evaluate(model, loader, device, config.dtype)
    state["last_eval_step"] = current_step
    improved = metrics["ce"] < state.get("best_val_ce", float("inf"))
    if improved:
        state["best_val_ce"] = metrics["ce"]
    Logger(f'Validation step:{current_step}, CE:{metrics["ce"]:.4f}, '
           f'PPL:{metrics["ppl"]}, targets:{metrics["targets"]}')
    if is_main_process():
        output = Path(config.save_path)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "validation.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"epoch": epoch, "update": current_step,
                                   "best": improved, **metrics}, allow_nan=False)+"\n")
        if improved:
            save_training_checkpoint(config, model, optimizer, scaler, epoch, step,
                                     current_step, state, weight="pretrain_best")
        if config.use_wandb and wandb is not None and wandb.run is not None:
            wandb.log({"val/ce": metrics["ce"], "val/ppl": metrics["ppl"],
                       "val/targets": metrics["targets"]}, step=current_step)


def train_epoch(
    epoch,loader,current_step,model,optimizer,scaler,device,auto_cast,
    config,total_update_steps,start_step=0,val_loader=None,training_state=None
):
    state = training_state if training_state is not None else {}
    start_time = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    group_targets = 0
    #先除以固定参考量，避免FP16缩放时反传一个很大的loss和；更新前再按实际token数校正。
    reference_targets = config.batch_size*(config.max_len-1)*config.accumulation_steps
    for step,(input_ids,labels) in enumerate(loader):
        if step < start_step:
            continue
        n = int(labels[:, 1:].ne(-100).sum())
        input_ids, labels = input_ids.to(device), labels.to(device)
        lr = get_lr(current_step,total_update_steps,config.lr,config.warmup_ratio,config.min_lr_ratio)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        res = None
        if n:
            with auto_cast:
                res = model(input_ids=input_ids,labels=labels)
                #res.loss已包含一次aux系数；按该微批有效预测目标数加权。
                loss = res.loss*(n/reference_targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("训练出现非有限loss")
            scaler.scale(loss).backward()
            group_targets += n

        boundary = (step+1) % config.accumulation_steps == 0 or step+1 == len(loader)
        updated = False
        if boundary and group_targets:
            scaler.unscale_(optimizer)
            #尾部不足一个累积组、不同长度/padding比例都使用实际分母。
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(reference_targets/group_targets)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                           error_if_nonfinite=not scaler.is_enabled())
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updated = not scaler.is_enabled() or scaler.get_scale() >= old_scale
            if updated:
                current_step += 1
                state["trained_targets"] = state.get("trained_targets", 0)+group_targets
            else:
                state["skipped_updates"] = state.get("skipped_updates", 0)+1
            optimizer.zero_grad(set_to_none=True)
            group_targets = 0

        if res is not None and step % config.log_interval == 0:
            elapsed = time.time()-start_time
            eta_time = elapsed/max(1,step-start_step+1)*max(0,len(loader)-step-1)/60
            Logger(f'Epoch:[{epoch+1}/{config.epochs}]({step}/{len(loader)}), '
                   f'loss:{res.loss.item():.4f}, logits_loss:{res.lm_loss.item():.4f}, '
                   f'aux_loss:{res.aux_loss.item():.4f}, lr:{lr:.8f}, epoch_time:{eta_time:.1f}min')
            if config.use_wandb and wandb is not None and wandb.run is not None:
                wandb.log({"loss":res.loss.item(), "logits_loss":res.lm_loss.item(),
                           "aux_loss":res.aux_loss.item(), "learning_rate":lr,
                           "epoch_time":eta_time}, step=current_step)

        if updated and val_loader is not None and current_step % config.eval_interval == 0:
            validate_and_save(model, val_loader, optimizer, scaler, device, config,
                              epoch, step+1, current_step, state)
        if updated and current_step % config.save_interval == 0 and is_main_process():
            save_training_checkpoint(config, model, optimizer, scaler, epoch, step+1,
                                     current_step, state)

    if val_loader is not None:
        validate_and_save(model, val_loader, optimizer, scaler, device, config,
                          epoch+1, 0, current_step, state)
    return current_step


def main():
    config = TrainConfig()
    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    auto_cast = precision_context(device, config.dtype)

    ckp_data = None
    if config.resume:
        #保存和恢复使用完全相同的文件名前缀。
        ckp_data = lm_check_point(model=None, weight="pretrain_weight", lm_config=config)
        if ckp_data is None:
            raise FileNotFoundError(f"未找到恢复checkpoint: {config.save_path}/pretrain_weight；首次训练请设resume=False")

    tokenizer = AutoTokenizer.from_pretrained(str(config.tokenizer_path))
    for name in ("pad", "bos", "eos"):
        if getattr(tokenizer, f"{name}_token_id") is None:
            raise ValueError(f"Tokenizer 没有 {name}_token")
    train_dataset, val_dataset = build_datasets(tokenizer, config)
    #AdamW维护FP32参数及状态，前向由autocast选择计算精度。
    model = build_model(tokenizer, config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, eps=1e-8,
                                  weight_decay=config.weight_decay, betas=(0.9,0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and config.dtype == "float16")

    start_epoch, start_step, current_step = 0, 0, 0
    state = {}
    if ckp_data:
        if ckp_data.get("model_config") is not None and ckp_data["model_config"] != asdict(model.config):
            raise ValueError("checkpoint模型配置与当前配置不一致")
        model.load_state_dict(ckp_data['model'])
        if ckp_data.get('optimizer') is not None:
            optimizer.load_state_dict(ckp_data['optimizer'])
        if ckp_data.get('scaler') is not None:
            scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data.get('epoch',0)
        start_step = ckp_data.get('step',0)
        current_step = ckp_data.get('current_step',0)
        state = ckp_data.get('training_state', {})

    if config.use_compile:
        model = torch.compile(model)
    #独立generator保证shuffle及验证加载器不消耗模型/dropout使用的随机数。
    train_generator = torch.Generator()
    dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                            generator=train_generator, num_workers=config.num_workers,
                            pin_memory=device == "cuda", drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            generator=torch.Generator().manual_seed(config.val_seed),
                            num_workers=config.num_workers, pin_memory=device == "cuda")
    update_per_epochs = math.ceil(len(dataloader)/config.accumulation_steps)
    total_update_steps = config.epochs*update_per_epochs

    Logger(f"设备:{device}, 主权重:FP32, 前向精度:{config.dtype if device == 'cuda' else 'float32'}")
    Logger(f"训练样本:{len(train_dataset)}, 验证样本:{len(val_dataset)}")
    raw_model = getattr(model, "_orig_mod", model)
    get_model_params(raw_model, raw_model.config)
    if is_main_process():
        Path(config.save_path).mkdir(parents=True, exist_ok=True)
        (Path(config.save_path)/"train_config.json").write_text(
            json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")
    if config.use_wandb and wandb is not None and is_main_process():
        wandb.init(project=config.project_name, id=ckp_data.get("wandb_id") if ckp_data else None,
                   resume="allow" if ckp_data and ckp_data.get("wandb_id") else None)

    for epoch in range(start_epoch, config.epochs):
        train_generator.manual_seed(config.seed+epoch)
        if epoch == start_epoch and start_step and ckp_data and "torch_rng" in ckp_data:
            torch.set_rng_state(ckp_data["torch_rng"])
            random.setstate(ckp_data["python_rng"])
            if device == "cuda" and ckp_data.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(ckp_data["cuda_rng"])
        else:
            set_seed(config.seed+epoch)
        current_step = train_epoch(
            epoch, dataloader, current_step, model, optimizer, scaler, device, auto_cast,
            config, total_update_steps, start_step=start_step if epoch == start_epoch else 0,
            val_loader=val_loader, training_state=state,
        )
        start_step = 0
        if is_main_process():
            save_training_checkpoint(config, model, optimizer, scaler, epoch+1, 0,
                                     current_step, state)

    if config.use_wandb and wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
