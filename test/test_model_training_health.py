"""End-to-end forward/backward, gradient, and parameter-count tests.

Run from the repository root:
    python test/test_model_training_health.py
"""

from __future__ import annotations

import unittest

import torch

from arch_test_utils import (
    assert_finite,
    make_tiny_config,
    parameter_count_by_top_level,
    unique_parameter_count,
)


class TestModelTrainingHealth(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260822)
        from model.model import YuchenModelCausalLLM

        self.cfg = make_tiny_config(use_moe=True, use_attn_res=True)
        self.model = YuchenModelCausalLLM(self.cfg).train()

        # The final two positions are right-padding in the second sample.
        self.input_ids = torch.randint(1, self.cfg.vocab_size, (2, 6))
        self.input_ids[1, -2:] = self.cfg.pad_token_id
        self.attention_mask = self.input_ids.ne(self.cfg.pad_token_id)
        self.labels = self.input_ids.clone()
        self.labels[~self.attention_mask] = -100

    def test_forward_backward_gradients_and_optimizer_step(self) -> None:
        output = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
        )

        self.assertEqual(tuple(output.logits.shape), (2, 6, self.cfg.vocab_size))
        self.assertEqual(output.loss.ndim, 0)
        self.assertEqual(output.aux_loss.ndim, 0)
        assert_finite(self, output.logits, "logits")
        assert_finite(self, output.loss, "total loss")
        assert_finite(self, output.lm_loss, "language-model loss")
        assert_finite(self, output.aux_loss, "MoE auxiliary loss")

        before = self.model.lm_head.weight.detach().clone()
        output.loss.backward()

        gradient_norms: dict[str, float] = {}
        no_gradient: list[str] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                no_gradient.append(name)
                continue
            assert_finite(self, parameter.grad, f"gradient for {name}")
            gradient_norms[name] = parameter.grad.norm().item()

        self.assertFalse(no_gradient, f"trainable parameters without gradients: {no_gradient}")
        self.assertTrue(any(norm > 0.0 for norm in gradient_norms.values()))

        torch.optim.SGD(self.model.parameters(), lr=1e-3).step()
        self.assertFalse(torch.equal(before, self.model.lm_head.weight.detach()))

    def test_reports_unique_parameter_count(self) -> None:
        total = unique_parameter_count(self.model)
        grouped = parameter_count_by_top_level(self.model)

        self.assertGreater(total, 0)
        self.assertEqual(total, sum(grouped.values()))
        # This print is intentional: it gives a directly usable parameter total
        # for the exact configuration exercised by this test.
        print(f"tiny-config unique parameters: {total:,}")
        print("tiny-config parameter groups:", grouped)


if __name__ == "__main__":
    unittest.main(verbosity=2)
