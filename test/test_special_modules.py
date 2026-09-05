"""Contract tests for AttnRes, Stable Latent MoE, and optional GDN modules.

Run from the repository root:
    python test/test_special_modules.py
"""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn as nn

from arch_test_utils import ROOT, assert_finite, make_tiny_config


class TestAttnRes(unittest.TestCase):
    def test_matches_explicit_weighted_sum_and_backpropagates(self) -> None:
        from model.model import attn_res

        torch.manual_seed(17)
        sources = [torch.randn(2, 3, 8, requires_grad=True) for _ in range(2)]
        partial = torch.randn(2, 3, 8, requires_grad=True)
        norm = nn.RMSNorm(8)
        query = nn.Linear(8, 1, bias=False)

        actual = attn_res(sources, partial, norm, query)
        stacked = torch.stack([*sources, partial], dim=0)
        weights = torch.softmax(
            torch.einsum("d,nbld->nbl", query.weight.view(-1), norm(stacked)),
            dim=0,
        )
        expected = torch.einsum("nbl,nbld->bld", weights, stacked)

        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        for index, source in enumerate([*sources, partial]):
            self.assertIsNotNone(source.grad, f"source {index} has no gradient")
            assert_finite(self, source.grad, f"AttnRes source {index} gradient")
        assert_finite(self, query.weight.grad, "AttnRes query gradient")

    def test_zero_query_produces_uniform_source_average(self) -> None:
        from model.model import attn_res

        sources = [torch.randn(1, 2, 4), torch.randn(1, 2, 4)]
        query = nn.Linear(4, 1, bias=False)
        nn.init.zeros_(query.weight)
        actual = attn_res(sources, None, nn.RMSNorm(4), query)
        torch.testing.assert_close(actual, torch.stack(sources).mean(dim=0))


class TestStableLatentMoe(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        from model.Stable_Latent_Moe import MoE

        self.cfg = make_tiny_config(use_moe=True)
        self.moe = MoE(self.cfg)
        self.x = torch.randn(2, 5, self.cfg.d_model)
        self.padding_mask = torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]]
        )

    def test_gate_respects_topk_padding_and_probability_normalization(self) -> None:
        latent = self.moe.W_down(self.x)
        indices, weights, aux_loss = self.moe.gate(latent, self.padding_mask)
        valid = self.padding_mask.reshape(-1)

        self.assertEqual(tuple(indices.shape), (self.x.size(0) * self.x.size(1), self.cfg.n_expert_per_token))
        self.assertTrue(((0 <= indices) & (indices < self.cfg.n_route_expert)).all().item())
        torch.testing.assert_close(weights[valid].sum(dim=-1), torch.ones(valid.sum()))
        torch.testing.assert_close(weights[~valid], torch.zeros_like(weights[~valid]))
        assert_finite(self, aux_loss, "MoE load-balance loss")

    def test_training_path_has_finite_output_and_gradients(self) -> None:
        self.moe.train()
        x = self.x.detach().clone().requires_grad_(True)
        output, aux_loss = self.moe(x, self.padding_mask)
        self.assertEqual(tuple(output.shape), tuple(x.shape))
        torch.testing.assert_close(output[~self.padding_mask], torch.zeros_like(output[~self.padding_mask]))
        loss = output.square().mean() + aux_loss
        assert_finite(self, loss, "Stable Latent MoE test loss")
        loss.backward()
        assert_finite(self, x.grad, "Stable Latent MoE input gradient")
        for name, parameter in self.moe.named_parameters():
            self.assertIsNotNone(parameter.grad, f"{name} has no gradient")
            assert_finite(self, parameter.grad, f"Stable Latent MoE gradient for {name}")

    def test_inference_matches_reference_routing_pipeline(self) -> None:
        """The optimized eval path must equal the documented routing pipeline."""
        self.moe.eval()
        with torch.no_grad():
            latent = self.moe.W_down(self.x)
            indices, weights, _ = self.moe.gate(latent, self.padding_mask)
            routed = self.moe.moe_infer(latent.reshape(-1, latent.size(-1)), indices, weights)
            routed = routed.view_as(latent)
            reference = self.moe.W_up(self.moe.up_norm(routed))
            identity = self.x * self.padding_mask.unsqueeze(-1)
            for expert in self.moe.shared_expert:
                reference = reference + expert(identity)
            reference = reference * self.padding_mask.unsqueeze(-1)
            actual, _ = self.moe(self.x, self.padding_mask)

        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)


class TestGdnAvailability(unittest.TestCase):
    def test_gdn_has_an_implementation_to_test(self) -> None:
        """Skip transparently until a GDN implementation is added to model/."""
        candidates = [path for path in (ROOT / "model").glob("*.py") if "gdn" in path.stem.lower()]
        if not candidates:
            self.skipTest("No GDN implementation exists under model/; GDN numerical tests are not applicable yet.")
        self.fail(f"GDN source found but no contract adapter is registered: {candidates}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
