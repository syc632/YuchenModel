"""Streaming-cache equivalence and finite-value stress tests for MLA/DeltaNet.

Run from the repository root:
    python test/test_mixer_cache_stability.py
"""

from __future__ import annotations

import unittest

import torch

from arch_test_utils import assert_finite, make_tiny_config


class TestMixerCacheAndStability(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(29)
        self.cfg = make_tiny_config(use_moe=False)

    def _assert_cached_matches_full(self, module, x: torch.Tensor, *, rtol: float, atol: float) -> None:
        module.eval()
        with torch.no_grad():
            full_output, _ = module(x)
            cache = None
            streamed = []
            for index in range(x.size(1)):
                output, cache = module(x[:, index : index + 1], cache=cache)
                streamed.append(output)
            streamed_output = torch.cat(streamed, dim=1)
        torch.testing.assert_close(streamed_output, full_output, rtol=rtol, atol=atol)

    def test_mla_cache_matches_full_sequence(self) -> None:
        from model.MLA import MLA

        x = torch.randn(2, 6, self.cfg.d_model)
        self._assert_cached_matches_full(MLA(self.cfg), x, rtol=1e-4, atol=1e-5)

    def test_gated_deltanet_cache_matches_full_sequence(self) -> None:
        from model.GatedDeltaNet import GatedDeltaNet

        x = torch.randn(2, 6, self.cfg.d_model)
        self._assert_cached_matches_full(GatedDeltaNet(self.cfg), x, rtol=2e-4, atol=2e-5)

    def test_mixer_outputs_and_gradients_stay_finite_across_input_scales(self) -> None:
        from model.GatedDeltaNet import GatedDeltaNet
        from model.MLA import MLA

        for module_name, module in (("MLA", MLA(self.cfg)), ("GatedDeltaNet", GatedDeltaNet(self.cfg))):
            module.train()
            for scale in (1e-3, 1.0, 1e3):
                x = (torch.randn(2, 5, self.cfg.d_model) * scale).requires_grad_(True)
                output, _ = module(x)
                loss = output.square().mean()
                assert_finite(self, output, f"{module_name} output at scale {scale}")
                assert_finite(self, loss, f"{module_name} loss at scale {scale}")
                loss.backward()
                assert_finite(self, x.grad, f"{module_name} input gradient at scale {scale}")
                for name, parameter in module.named_parameters():
                    if parameter.grad is not None:
                        assert_finite(self, parameter.grad, f"{module_name}.{name} gradient at scale {scale}")
                module.zero_grad(set_to_none=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
