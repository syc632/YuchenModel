"""Token-weighted training, resumable runs and held-out evaluation."""
from contextlib import nullcontext
from dataclasses import asdict
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from model.Stable_Latent_Moe import MoEGate
from .config import (digest, environment, initialized_model, parameter_counts, read_json,
                     source_identity, write_json)
from .data import Blocks, BatchStream, batches, cap_targets, target_count, verify_data


def runtime(settings):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if settings.device == "auto" else settings.device)
    dtype = settings.dtype
    if dtype == "auto":
        dtype = "float32" if device.type != "cuda" else ("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
    if device.type != "cuda" and dtype != "float32":
        raise ValueError("CPU实验使用float32；CUDA使用统一AMP精度")
    if dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("当前CUDA设备不支持BF16")
    return device, dtype


def autocast(device, dtype):
    return nullcontext() if dtype == "float32" else torch.autocast(device_type=device.type, dtype=getattr(torch, dtype))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, dataset, settings, device, dtype, token_limit=None):
    was_training = model.training
    model.eval()
    total_nll, total = 0.0, 0
    try:
        for batch in batches(dataset, settings.micro_batch, model.config.pad_token_id):
            if token_limit is not None:
                batch = cap_targets(batch, token_limit-total)
            n = target_count(batch)
            if n:
                with autocast(device, dtype):
                    out = model(**to_device(batch, device))
                loss = float(out.lm_loss)
                if not math.isfinite(loss):
                    raise FloatingPointError("评估出现非有限CE")
                total_nll += loss*n
                total += n
            if token_limit is not None and total >= token_limit:
                break
    finally:
        model.train(was_training)
    if not total:
        return {"ce": None, "ppl": None, "targets": 0, "status": "no_eligible_documents"}
    ce = total_nll/total
    return {"ce": ce, "ppl": math.exp(ce) if ce < 700 else None, "targets": total, "nll_sum": total_nll, "status": "ok"}


class RoutingMonitor:
    def __init__(self, model):
        self.counts, self.handles = {}, []
        for name, module in model.named_modules():
            if isinstance(module, MoEGate):
                self.handles.append(module.register_forward_hook(self._hook(name, module.n_route)))

    def _hook(self, name, n_experts):
        def record(module, inputs, output):
            if not module.training:
                return
            indices, weights, _ = output
            count = torch.bincount(indices.detach()[weights.detach() > 0], minlength=n_experts)
            self.counts[name] = self.counts.get(name, torch.zeros_like(count)) + count
        return record

    def snapshot(self):
        result = {}
        for name, counts in self.counts.items():
            counts = counts.cpu().tolist()
            result[name] = {"assignments": counts,
                            "fractions": [x/max(1, sum(counts)) for x in counts],
                            "idle_fraction": counts.count(0)/len(counts)}
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()


def learning_rate(tokens, settings):
    warmup = settings.max_tokens*settings.warmup_ratio
    if tokens < warmup:
        return settings.lr*max(1, tokens)/max(1, warmup)
    progress = min(1, (tokens-warmup)/max(1, settings.max_tokens-warmup))
    return settings.lr*(settings.min_lr_ratio + (1-settings.min_lr_ratio)*0.5*(1+math.cos(math.pi*progress)))


def save_checkpoint(path, model, optimizer, scaler, state, identity):
    # 保留FP32主权重与完整优化器/RNG，不使用旧训练脚本的FP16权重快照。
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scaler": scaler.state_dict(), "state": state, "identity": identity,
               "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
               "numpy_rng": np.random.get_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_training(data_path, run_dir, cfg, base_cfg, settings, name="B", stage="screen", resume=False, smoke=False):
    metadata = verify_data(data_path)
    for key in ("vocab_size", "pad_token_id", "bos_token_id", "eos_token_id"):
        if getattr(cfg, key) != metadata[key]:
            raise ValueError(f"模型与数据的{key}不匹配")
    device, dtype = runtime(settings)
    identity = {"name": name, "stage": stage, "smoke": smoke, "model": asdict(cfg), "base_model": asdict(base_cfg),
                "training": asdict(settings), "effective_dtype": dtype,
                "data_fingerprint": metadata["fingerprint"], "source": source_identity(),
                "environment": environment(device)}
    identity["fingerprint"] = digest(identity)
    run_dir = Path(run_dir)
    checkpoint = run_dir/"checkpoint.pt"
    if run_dir.exists() and not resume:
        raise FileExistsError(f"运行目录已存在；使用新目录或显式--resume: {run_dir}")
    if resume:
        saved = read_json(run_dir/"run.json")
        if saved != identity:
            raise ValueError("拒绝恢复：配置、代码、环境或数据已改变")
        if not checkpoint.exists():
            raise FileNotFoundError("没有可恢复的checkpoint")
    else:
        run_dir.mkdir(parents=True)
        write_json(run_dir/"run.json", identity)
    seed_all(settings.seed)
    model = initialized_model(cfg, settings.seed, base_cfg).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=settings.lr, betas=(0.9, 0.95), weight_decay=settings.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == "float16")
    monitor = RoutingMonitor(model)
    state = {"tokens": 0, "updates": 0, "skipped_updates": 0, "train_seconds": 0.0,
             "wall_seconds": 0.0, "history": [], "stream": {"epoch": 0, "position": 0}}
    if resume:
        loaded = torch.load(checkpoint, map_location=device, weights_only=False)
        if loaded["identity"] != identity:
            raise ValueError("checkpoint身份与manifest不匹配")
        model.load_state_dict(loaded["model"])
        optimizer.load_state_dict(loaded["optimizer"])
        scaler.load_state_dict(loaded["scaler"])
        state = loaded["state"]
        random.setstate(loaded["python_rng"])
        np.random.set_state(loaded["numpy_rng"])
        torch.set_rng_state(loaded["torch_rng"].cpu())
        if loaded["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in loaded["cuda_rng"]])
        monitor.counts = {k: torch.tensor(v["assignments"], device=device) for k, v in state.get("routing", {}).items()}
    train = Blocks(data_path, "train", settings.seq_len)
    validation = Blocks(data_path, "validation", settings.seq_len)
    stream = BatchStream(train, settings.micro_batch, cfg.pad_token_id, settings.data_seed, **state["stream"])
    thresholds = sorted(set(math.ceil(settings.max_tokens*i/settings.eval_intervals) for i in range(1, settings.eval_intervals+1)))
    next_eval = next((x for x in thresholds if x > state["tokens"]), settings.max_tokens)
    session_start, previous_wall = time.perf_counter(), state["wall_seconds"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    try:
        while state["tokens"] < settings.max_tokens:
            synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            n_update, lm_sum, aux_sum = 0, 0.0, 0.0
            # 不跨验证边界，因而各组在相同累计target处比较。
            limit = min(settings.max_tokens, next_eval)-state["tokens"]
            update_target = min(settings.tokens_per_update, limit)
            lr = learning_rate(state["tokens"]+min(update_target, limit), settings)
            for group in optimizer.param_groups:
                group["lr"] = lr
            while n_update < update_target:
                batch = cap_targets(stream.next(), limit-n_update)
                n = target_count(batch)
                if not n:
                    continue
                with autocast(device, dtype):
                    out = model(**to_device(batch, device))
                    # 先按目标数加权；step前按实际累计目标数归一化。
                    loss = out.loss * (n/settings.tokens_per_update)
                if not torch.isfinite(loss):
                    raise FloatingPointError("训练loss出现NaN/Inf")
                scaler.scale(loss).backward()
                lm_sum += float(out.lm_loss.detach())*n
                aux_sum += float(out.aux_loss.detach())*n
                n_update += n
            scaler.unscale_(optimizer)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(settings.tokens_per_update/n_update)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            scale_before = scaler.get_scale()
            if not torch.isfinite(norm) and not scaler.is_enabled():
                raise FloatingPointError("训练梯度出现NaN/Inf")
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < scale_before
            state["skipped_updates"] += int(skipped)
            state["updates"] += 1
            state["tokens"] += n_update
            synchronize(device)
            state["train_seconds"] += time.perf_counter()-start
            state["stream"] = stream.state_dict()
            if state["tokens"] >= next_eval:
                metrics = evaluate(model, validation, settings, device, dtype, settings.max_eval_tokens)
                state["wall_seconds"] = previous_wall + time.perf_counter()-session_start
                row = {"tokens": state["tokens"], "updates": state["updates"], "learning_rate": lr,
                       "train_lm_loss": lm_sum/n_update, "aux_loss": aux_sum/n_update,
                       "grad_norm": float(norm) if torch.isfinite(norm) else None,
                       "skipped_updates": state["skipped_updates"], "validation": metrics,
                       "train_seconds": state["train_seconds"], "wall_seconds": state["wall_seconds"],
                       "effective_tokens_per_second": state["tokens"]/max(state["train_seconds"], 1e-9),
                       "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
                state["history"].append(row)
                state["routing"] = monitor.snapshot()
                save_checkpoint(checkpoint, model, optimizer, scaler, state, identity)
                write_json(run_dir/"metrics.json", state)
                print(f'{name} seed={settings.seed} tokens={state["tokens"]}/{settings.max_tokens} val_ce={metrics["ce"]:.5f}', flush=True)
                next_eval = next((x for x in thresholds if x > state["tokens"]), settings.max_tokens)
        summary = {"name": name, "seed": settings.seed, "stage": stage, "status": "complete",
                   "identity": identity["fingerprint"], "tokens": state["tokens"],
                   "validation": state["history"][-1]["validation"], **parameter_counts(model),
                   "train_seconds": state["train_seconds"], "skipped_updates": state["skipped_updates"],
                   "peak_memory_bytes": state["history"][-1]["peak_memory_bytes"]}
        write_json(run_dir/"summary.json", summary)
        (run_dir/"failure.json").unlink(missing_ok=True)
        return summary
    except Exception as exc:
        write_json(run_dir/"failure.json", {"type": type(exc).__name__, "message": str(exc), "tokens": state["tokens"]})
        raise
    finally:
        monitor.close()
