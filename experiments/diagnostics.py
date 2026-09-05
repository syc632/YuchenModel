"""Correctness gates, actual training throughput, and cache-aware latency."""
from dataclasses import replace
import math
import statistics
import time

import torch
from model.model import YuchenModelCausalLLM
from .config import initialized_model, parameter_counts
from .data import BatchStream, Blocks, target_count
from .engine import autocast, runtime, synchronize, to_device, RoutingMonitor, seed_all


def check_contract(cfg, device="cpu", dtype="float32", fit_steps=100):
    device = torch.device(device)
    seed_all(17)
    model = initialized_model(cfg, 17).to(device)
    ids = torch.randint(3, cfg.vocab_size, (3, 7), device=device)
    mask = torch.tensor([[1]*7, [1]*4+[0]*3, [0]*7], device=device, dtype=torch.bool)
    labels = ids.masked_fill(~mask, -100)
    atol, rtol = ((2e-5, 3e-4) if dtype == "float32" else (0.08, 0.08))
    model.eval()
    with torch.no_grad(), autocast(device, dtype):
        full = model(input_ids=ids, attention_mask=mask)
        cache, outputs = None, []
        for i in range(ids.size(1)):
            result = model(input_ids=ids[:, i:i+1], attention_mask=mask[:, i:i+1], cache=cache)
            cache = result.past_key_values
            outputs.append(result.logits)
        torch.testing.assert_close(full.logits, torch.cat(outputs, dim=1), atol=atol, rtol=rtol)
        # 改变未来内容，同时改变padding内容，前缀必须保持不变。
        altered = ids.clone()
        altered[:, 4:] = torch.randint(3, cfg.vocab_size, altered[:, 4:].shape, device=device)
        changed = model(input_ids=altered, attention_mask=mask).logits
        torch.testing.assert_close(changed[:, :4], full.logits[:, :4], atol=atol, rtol=rtol)
        individual = model(input_ids=ids[1:2, :4]).logits
        torch.testing.assert_close(individual, full.logits[1:2, :4], atol=atol, rtol=rtol)
        torch.testing.assert_close(full.logits[~mask], torch.zeros_like(full.logits[~mask]), atol=0, rtol=0)
        # 整个batch无有效目标时损失为0，不为NaN。
        empty = model(input_ids=ids, attention_mask=torch.zeros_like(mask), labels=torch.full_like(ids, -100))
        assert torch.isfinite(empty.loss) and empty.lm_loss.item() == 0
    model.train()
    before = model.embd.weight.detach().clone()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == "float16")
    losses = []
    for step in range(max(1, fit_steps)):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, dtype):
            result = model(input_ids=ids, attention_mask=mask, labels=labels)
        if not torch.isfinite(result.loss):
            raise AssertionError("非有限loss")
        scaler.scale(result.loss).backward()
        scaler.unscale_(optimizer)
        for name, p in model.named_parameters():
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                raise AssertionError(f"无梯度或非有限梯度: {name}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(result.lm_loss.detach()))
    assert not torch.equal(before, model.embd.weight), "参数未更新"
    if fit_steps >= 10 and statistics.mean(losses[-5:]) >= statistics.mean(losses[:5]):
        raise AssertionError("小样本CE没有下降")
    return {"status": "ready", "fit_steps": fit_steps, "first_ce": losses[0], "last_ce": losses[-1],
            "cache_atol": atol, "cache_rtol": rtol, **parameter_counts(model)}


def training_throughput(cfg, base, settings, data, warmup=3, steps=10):
    if warmup < 1 or steps < 1:
        raise ValueError("warmup/steps必须为正")
    device, dtype = runtime(settings)
    seed_all(settings.seed)
    model = initialized_model(cfg, settings.seed, base).to(device).train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=settings.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == "float16")
    monitor = RoutingMonitor(model)
    stream = BatchStream(Blocks(data, "train", settings.seq_len), settings.micro_batch, cfg.pad_token_id, settings.data_seed)
    total, elapsed = 0, 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for i in range(warmup+steps):
            synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            count = 0
            while count < settings.tokens_per_update:
                batch = stream.next()
                n = target_count(batch)
                with autocast(device, dtype):
                    output = model(**to_device(batch, device))
                    loss = output.loss*(n/settings.tokens_per_update)
                if not torch.isfinite(loss):
                    raise FloatingPointError("测速loss非有限")
                scaler.scale(loss).backward()
                count += n
            scaler.unscale_(optimizer)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(settings.tokens_per_update/count)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            if not torch.isfinite(norm):
                raise FloatingPointError("测速梯度非有限")
            scaler.step(optimizer)
            scaler.update()
            synchronize(device)
            if i >= warmup:
                elapsed += time.perf_counter()-start
                total += count
        return {"effective_tokens_per_second": total/elapsed, "targets": total, "seconds": elapsed,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                **parameter_counts(model)}
    finally:
        monitor.close()


def cache_bytes(value):
    if torch.is_tensor(value):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(cache_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(cache_bytes(v) for v in value)
    return 0


@torch.no_grad()
def inference_benchmark(model, settings, lengths=(128, 512, 2048), decode_steps=128, repeats=5, warmup=2):
    device, dtype = runtime(settings)
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(9102)
    results = []
    for length in lengths:
        # 相同词表、相同输入；固定后续token，避免生成EOS提前停止造成计时偏差。
        prompt = torch.randint(3, model.config.vocab_size, (1, length), generator=generator).to(device)
        continuation = torch.randint(3, model.config.vocab_size, (1, decode_steps), generator=generator).to(device)
        prefill, decode = [], []
        peak = None
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for i in range(warmup+repeats):
                synchronize(device)
                start = time.perf_counter()
                with autocast(device, dtype):
                    out = model(input_ids=prompt, logits_to_keep=1)
                synchronize(device)
                prefill_ms = (time.perf_counter()-start)*1000
                prefill_cache = cache_bytes(out.past_key_values)
                cache = out.past_key_values
                start = time.perf_counter()
                with autocast(device, dtype):
                    for j in range(decode_steps):
                        out = model(input_ids=continuation[:, j:j+1], cache=cache, logits_to_keep=1)
                        cache = out.past_key_values
                synchronize(device)
                decode_ms = (time.perf_counter()-start)*1000/decode_steps
                final_cache_size = cache_bytes(cache)
                if i >= warmup:
                    prefill.append(prefill_ms)
                    decode.append(decode_ms)
                del out, cache
            if device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(device)
            results.append({"prompt_length": length, "status": "ok", "decode_steps": decode_steps,
                            "prefill_ms": statistics.median(prefill), "decode_ms_per_token": statistics.median(decode),
                            "prefill_samples_ms": prefill, "decode_samples_ms_per_token": decode,
                            "prefill_cache_bytes": prefill_cache, "final_cache_bytes": final_cache_size,
                            "peak_memory_bytes": peak})
        except torch.OutOfMemoryError:
            results.append({"prompt_length": length, "status": "out_of_memory"})
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return results


def matched_dense_config(base, ffn_type="situglu"):
    def count(cfg):
        with torch.device("meta"):
            model = YuchenModelCausalLLM(cfg)
        return sum(p.numel() for p in model.parameters())
    target = count(base)
    dense = replace(base, use_moe=False, ffn_type=ffn_type)
    a, b = count(replace(dense, d_inner=16)), count(replace(dense, d_inner=32))
    slope = (b-a)/16
    ideal = 16+(target-a)/slope
    widths = {max(16, 16*math.floor(ideal/16)), max(16, 16*math.ceil(ideal/16))}
    width = min(widths, key=lambda x: abs(count(replace(dense, d_inner=x))-target))
    if abs(count(replace(dense, d_inner=width))-target)/target > 0.02:
        # 小模型16对齐过粗时允许整数宽度，优先保证参数量公平性。
        width = max(1, round(ideal))
    result = replace(dense, d_inner=width)
    actual = count(result)
    return result, {"target_parameters": target, "actual_parameters": actual,
                    "relative_error": abs(actual-target)/target,
                    "status": "matched" if abs(actual-target)/target <= 0.02 else "outside_tolerance"}
