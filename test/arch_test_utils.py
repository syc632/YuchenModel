"""Shared helpers for architecture smoke and correctness tests.

These helpers deliberately build a very small configuration so the tests can
run on CPU with random data.  They do not load checkpoints or touch training
artifacts.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

#__file__当前脚本自身的路径
#Path(...).resolve()转为绝对路径
#.parents[1]取上一级目录的父目录D;\Kimi\test\
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"

# Stable_Latent_Moe.py currently imports ``ffn`` as a top-level module.  Adding this path is
# test-process-only compatibility glue; it does not modify the model package.
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

# Do not put MODEL_DIR on sys.path: without model/__init__.py it would shadow
# the namespace package as model.py.  Stable_Latent_Moe.py needs a legacy ``from ffn``
# import, so register only that module for this test process.
if "ffn" not in sys.modules:
    ffn_spec = importlib.util.spec_from_file_location("ffn", MODEL_DIR / "ffn.py")
    assert ffn_spec is not None and ffn_spec.loader is not None
    ffn_module = importlib.util.module_from_spec(ffn_spec)
    sys.modules["ffn"] = ffn_module
    ffn_spec.loader.exec_module(ffn_module)


def make_tiny_config(*, use_moe: bool = True, use_attn_res: bool = True):
    """Return a CPU-friendly Config that exercises both DeltaNet and MLA."""
    from model.model import Config

    return Config(
        d_model=16,
        n_head=4,
        n_layer=3,
        ratio=1,
        conv_size=3,
        chunk_size=4,
        qk_nope=4,
        # rotary-embedding-torch requires a RoPE dimension greater than 2.
        qk_rope=4,
        v_head_dim=4,
        kv_latent=8,
        q_latent=8,
        n_shared_expert=1,
        n_route_expert=3,
        n_expert_per_token=2,
        d_latent=8,
        d_inner=24,
        aux_loss_alpha=0.01,
        use_moe=use_moe,
        use_attn_res=use_attn_res,
        vocab_size=37,
        pad_token_id=0,
        dropout=0.0,
    )


def assert_finite(testcase, value: torch.Tensor, name: str) -> None:
    testcase.assertTrue(torch.isfinite(value).all().item(), f"{name} contains NaN or Inf")


def unique_parameter_count(module: torch.nn.Module) -> int:
    """Count each shared Parameter once (the embedding/LM head are tied)."""
    seen: set[int] = set()
    total = 0
    for parameter in module.parameters():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            total += parameter.numel()
    return total


def parameter_count_by_top_level(module: torch.nn.Module) -> dict[str, int]:
    """Return unique parameter counts grouped by the first module path part."""
    seen: set[int] = set()
    counts: dict[str, int] = {}
    for name, parameter in module.named_parameters():
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        group = name.split(".", 1)[0]
        counts[group] = counts.get(group, 0) + parameter.numel()
    return counts
