"""Experiment matrix, immutable provenance and semantic shared initialization."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess

import torch
from model.model import Config, YuchenModelCausalLLM

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "B": {}, "M": {"mixer_type": "mamba2"},
    "A1": {"attention_type": "nope"}, "A2": {"attention_type": "embedding_gate"},
    "F1": {"use_moe": False, "ffn_type": "swiglu"},
    "F2": {"use_moe": False, "ffn_type": "situglu"},
    "F3": {"expert_ffn_type": "swiglu"}, "R": {"use_attn_res": False},
}
GROUPS = {"M": "mixer", "A1": "attention", "A2": "attention",
          "F1": "ffn", "F2": "ffn", "F3": "ffn", "R": "residual"}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_identity():
    files = sorted(p for folder in ("model", "experiments", "train")
                   for p in (ROOT/folder).rglob("*.py"))
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in files}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {"sha256": digest(hashes), "files": hashes, "git_commit": commit}


def environment(device):
    import transformers
    import importlib.metadata
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "rotary_embedding_torch": importlib.metadata.version("rotary-embedding-torch"),
            "numpy": importlib.metadata.version("numpy"), "tokenizers": importlib.metadata.version("tokenizers"),
            "cpu_threads": torch.get_num_threads(),
            "device": str(device), "hardware": torch.cuda.get_device_name(device)
            if torch.device(device).type == "cuda" else platform.machine()}


@dataclass
class TrainSettings:
    max_tokens: int = 100_000
    seq_len: int = 512
    micro_batch: int = 1
    tokens_per_update: int = 16_384
    seed: int = 42
    data_seed: int = 2026
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    dtype: str = "auto"
    device: str = "auto"
    eval_intervals: int = 10
    max_eval_tokens: int | None = None

    def __post_init__(self):
        if min(self.max_tokens, self.micro_batch, self.tokens_per_update, self.eval_intervals) < 1 or self.seq_len < 2:
            raise ValueError("token/batch/interval budgets必须为正，seq_len>=2")
        if self.dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("未知dtype")
        if self.max_eval_tokens is not None and self.max_eval_tokens < 1:
            raise ValueError("max_eval_tokens必须为正")
        if not (0 <= self.warmup_ratio < 1 and 0 <= self.min_lr_ratio <= 1) or self.lr <= 0:
            raise ValueError("无效学习率配置")


def tiny_config():
    return Config(d_model=16, n_head=4, head_dim=8, n_layer=4, ratio=1,
                  conv_size=3, chunk_size=4, qk_nope=4, qk_rope=4, qk_head_dim=8,
                  v_head_dim=4, kv_latent=8, q_latent=8, kv_embd=8,
                  n_shared_expert=1, n_route_expert=3, n_expert_per_token=2,
                  d_latent=8, d_inner=24, mamba_d_state=4, vocab_size=37,
                  pad_token_id=0, bos_token_id=1, eos_token_id=2)


def variant_config(base, variant, overrides=None):
    changes = dict(VARIANTS[variant]) if variant in VARIANTS else {}
    if variant not in VARIANTS and overrides is None:
        raise ValueError(f"未知实验{variant}")
    changes.update(overrides or {})
    # 避免replace保留由旧d_model推导的d_head。
    if "d_model" in changes and "d_head" not in changes:
        changes["d_head"] = None
    return replace(base, **changes)


def initialized_model(cfg, seed, reference_cfg=None):
    """Copy matching *semantic modules*, not merely matching tensor shapes.

    Variant-specific modules retain their seeded initialization; common module
    classes and paths receive the baseline initialization even when their parent
    mixer differs (e.g. MLA and EG_MLA projections). FFN variants never cross-copy.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = YuchenModelCausalLLM(cfg)
        if reference_cfg is None or asdict(reference_cfg) == asdict(cfg):
            return model
        torch.manual_seed(seed)
        reference = YuchenModelCausalLLM(reference_cfg)
        src = dict(reference.named_parameters())
        src_modules = dict(reference.named_modules())
        dst_modules = dict(model.named_modules())
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name not in src or src[name].shape != parameter.shape:
                    continue
                parts = name.split(".")
                if parts[:2] == ["model", "layers"]:
                    layer_path = ".".join(parts[:3])
                    component = parts[3]
                    if component in {"mixer", "ffn"}:
                        parent = layer_path+"."+component
                        # MLA and EG share the same Q/K/V projection semantics.
                        src_type = type(src_modules[parent]).__name__
                        dst_type = type(dst_modules[parent]).__name__
                        compatible = src_type == dst_type or {src_type, dst_type} <= {"MLA", "EG_MLA"}
                        if not compatible:
                            continue
                        if component == "ffn" and cfg.expert_ffn_type != reference_cfg.expert_ffn_type:
                            if "shared_expert" in parts or "route_expert" in parts:
                                continue
                parent = name.rsplit(".", 1)[0]
                if type(src_modules.get(parent)) is type(dst_modules.get(parent)):
                    parameter.copy_(src[name])
    return model


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    active = total
    for layer in model.model.layers:
        if layer.use_moe:
            experts = layer.ffn.route_expert
            counts = [sum(p.numel() for p in expert.parameters()) for expert in experts]
            active -= sum(counts) - sum(counts[:layer.ffn.top_k])
    return {"parameters": total, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "active_parameters_estimate": active}


def common_token_budget(rates, hours):
    if not rates or hours <= 0 or any(not math.isfinite(x) or x <= 0 for x in rates):
        raise ValueError("需要正数吞吐和GPU小时")
    return int(0.85*hours*3600 / sum(1/x for x in rates))


def select_candidates(scores):
    if "B" not in scores or not math.isfinite(scores["B"]):
        raise ValueError("缺少有效基线")
    ranked = sorted((k for k in scores if k in GROUPS and math.isfinite(scores[k])), key=lambda k: (scores[k], k))
    if len(ranked) < 2:
        raise ValueError("至少需要两个有效候选")
    best = ranked[0]
    beneficial = [k for k in ranked if scores[k] < scores["B"]]
    pair = next(((a, b) for i, a in enumerate(beneficial) for b in beneficial[i+1:]
                 if GROUPS[a] != GROUPS[b]), None)
    candidates = [{"name": "B", "overrides": {}}, {"name": best, "overrides": VARIANTS[best]}]
    if pair:
        candidates.append({"name": "+".join(pair), "overrides": {**VARIANTS[pair[0]], **VARIANTS[pair[1]]},
                           "components": list(pair)})
    else:
        candidates.append({"name": ranked[1], "overrides": VARIANTS[ranked[1]]})
    return candidates
