"""Mamba2的SSD、缓存、padding和主模型接入测试。

Run from the repository root:
    python test/test_mamba2.py
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch

# 允许按照文件顶部的命令直接运行测试脚本。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.Mamba2 import Mamba2, Mamba2Config


def make_mamba_config() -> Mamba2Config:
    """返回一个可以在CPU上快速执行的Mamba2配置。"""
    return Mamba2Config(
        d_model=8,
        head_dim=4,
        conv_size=3,
        chunk_size=4,
        norm_eps=1e-5,
        mamba_d_state=3,
        mamba_expand=1,
        mamba_n_groups=1,
    )


class TestMamba2Ssd(unittest.TestCase):
    def test_chunk_ssd_matches_explicit_recurrence(self) -> None:
        torch.manual_seed(41)
        module = Mamba2(make_mamba_config())
        batch, seq_len = 2, 7  # 故意不让长度被chunk_size整除
        x = torch.randn(batch, seq_len, module.n_heads, module.head_dim)
        A = -torch.rand(batch, seq_len, module.n_heads)
        B = torch.randn(batch, seq_len, module.n_heads, module.d_state)
        C = torch.randn(batch, seq_len, module.n_heads, module.d_state)
        initial_state = torch.randn(
            batch,
            module.n_heads,
            module.head_dim,
            module.d_state,
        )

        actual, actual_state = module.ssd(x, A, B, C, initial_state)

        state = initial_state
        expected_outputs = []
        for index in range(seq_len):
            decay = torch.exp(A[:, index]).unsqueeze(-1).unsqueeze(-1)
            write = torch.einsum(
                "bhp,bhn->bhpn",
                x[:, index],
                B[:, index],
            )
            state = state * decay + write
            output = torch.einsum("bhpn,bhn->bhp", state, C[:, index])
            expected_outputs.append(output)
        expected = torch.stack(expected_outputs, dim=1)

        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(actual_state, state, rtol=2e-5, atol=2e-5)


class TestMamba2Block(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(43)
        self.cfg = make_mamba_config()
        self.module = Mamba2(self.cfg)

    def test_full_sequence_matches_token_cache(self) -> None:
        self.module.eval()
        x = torch.randn(2, 7, self.cfg.d_model)

        with torch.no_grad():
            full_output, full_cache = self.module(x)
            cache = None
            streamed = []
            for index in range(x.size(1)):
                output, cache = self.module(
                    x[:, index : index + 1],
                    cache=cache,
                )
                streamed.append(output)
            streamed_output = torch.cat(streamed, dim=1)

        torch.testing.assert_close(streamed_output, full_output, rtol=2e-4, atol=2e-5)
        self.assertEqual(set(full_cache), {"conv_state", "ssm_state"})
        self.assertEqual(
            tuple(full_cache["conv_state"].shape),
            (2, self.module.conv_dim, self.cfg.conv_size - 1),
        )
        self.assertEqual(
            tuple(full_cache["ssm_state"].shape),
            (
                2,
                self.module.n_heads,
                self.cfg.head_dim,
                self.cfg.mamba_d_state,
            ),
        )
        torch.testing.assert_close(cache["conv_state"], full_cache["conv_state"])
        torch.testing.assert_close(
            cache["ssm_state"],
            full_cache["ssm_state"],
            rtol=2e-4,
            atol=2e-5,
        )

    def test_right_padding_matches_individual_sequences_and_cache(self) -> None:
        self.module.eval()
        x = torch.randn(2, 6, self.cfg.d_model)
        padding_mask = torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, True, True, False, False],
            ]
        )

        with torch.no_grad():
            batch_output, batch_cache = self.module(x, padding_mask=padding_mask)
            first_output, first_cache = self.module(x[:1])
            second_output, second_cache = self.module(x[1:2, :4])

        torch.testing.assert_close(batch_output[:1], first_output, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(
            batch_output[1:2, :4],
            second_output,
            rtol=2e-4,
            atol=2e-5,
        )
        torch.testing.assert_close(
            batch_output[1:2, 4:],
            torch.zeros_like(batch_output[1:2, 4:]),
        )
        torch.testing.assert_close(
            batch_cache["conv_state"][:1],
            first_cache["conv_state"],
        )
        torch.testing.assert_close(
            batch_cache["conv_state"][1:2],
            second_cache["conv_state"],
        )
        torch.testing.assert_close(
            batch_cache["ssm_state"][:1],
            first_cache["ssm_state"],
            rtol=2e-4,
            atol=2e-5,
        )
        torch.testing.assert_close(
            batch_cache["ssm_state"][1:2],
            second_cache["ssm_state"],
            rtol=2e-4,
            atol=2e-5,
        )

    def test_output_and_all_parameter_gradients_are_finite(self) -> None:
        self.module.train()
        x = (torch.randn(2, 5, self.cfg.d_model) * 10).requires_grad_(True)
        output, _ = self.module(x)
        self.assertEqual(tuple(output.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(output).all().item())

        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        for name, parameter in self.module.named_parameters():
            self.assertIsNotNone(parameter.grad, f"{name}没有梯度")
            self.assertTrue(
                torch.isfinite(parameter.grad).all().item(),
                f"{name}的梯度包含NaN或Inf",
            )

    def test_rejects_invalid_cache(self) -> None:
        x = torch.randn(1, 1, self.cfg.d_model)
        with self.assertRaisesRegex(ValueError, "cache必须包含"):
            self.module(x, cache={})


class TestMamba2ModelIntegration(unittest.TestCase):
    def test_config_selects_mamba2_without_changing_default(self) -> None:
        # 当前测试环境缺少MLA已有的rotary_embedding_torch依赖。这里只为导入
        # model.Config提供最小占位类，不执行MLA，也不会改变生产代码的依赖关系。
        fake_rotary = types.ModuleType("rotary_embedding_torch")

        class RotaryEmbedding:
            def __init__(self, *_args, **_kwargs):
                pass

        fake_rotary.RotaryEmbedding = RotaryEmbedding
        sys.modules.pop("model.model", None)
        sys.modules.pop("model.MLA", None)
        previous_rotary = sys.modules.get("rotary_embedding_torch")
        sys.modules["rotary_embedding_torch"] = fake_rotary
        try:
            from model.model import Config, ModelLayer

            self.assertEqual(Config().mixer_type, "gdn")
            cfg = Config(
                d_model=8,
                n_head=2,
                head_dim=4,
                conv_size=3,
                chunk_size=4,
                mixer_type="mamba2",
                mamba_d_state=3,
                mamba_expand=1,
                mamba_n_groups=1,
                use_moe=False,
                d_inner=12,
            )
            layer = ModelLayer(cfg, mla=False)
            self.assertIsInstance(layer.mixer, Mamba2)

            with self.assertRaisesRegex(ValueError, "mixer_type"):
                Config(mixer_type="unknown")
            with self.assertRaisesRegex(ValueError, "head_dim"):
                Config(
                    d_model=8,
                    n_head=2,
                    head_dim=3,
                    mixer_type="mamba2",
                    mamba_expand=1,
                )
        finally:
            # 避免占位模块泄漏到同一进程中的其他测试。
            sys.modules.pop("model.model", None)
            sys.modules.pop("model.MLA", None)
            if previous_rotary is None:
                sys.modules.pop("rotary_embedding_torch", None)
            else:
                sys.modules["rotary_embedding_torch"] = previous_rotary


if __name__ == "__main__":
    unittest.main(verbosity=2)
