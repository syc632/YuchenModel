"""CPU unit tests for the Post_train OPD helpers."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

import torch


ROOT = Path(__file__).resolve().parents[1]
OPD_PATH = ROOT / "train" / "Post_train" / "OPD.py"


def load_post_train_opd() -> ModuleType:
    """Load pure OPD helpers without constructing local teacher/student models."""

    module_name = "post_train_opd_under_test"
    model_package = ModuleType("model")
    model_module = ModuleType("model.model")
    model_module.Config = type("Config", (), {})
    model_module.YuchenModelCausalLLM = type("YuchenModelCausalLLM", (), {})

    train_package = ModuleType("train")
    train_util_module = ModuleType("train.train_util")
    train_util_module.lm_check_point = lambda *args, **kwargs: None

    injected_modules = {
        "model": model_package,
        "model.model": model_module,
        "train": train_package,
        "train.train_util": train_util_module,
    }
    previous_modules = {name: sys.modules.get(name) for name in injected_modules}
    try:
        sys.modules.update(injected_modules)
        spec = importlib.util.spec_from_file_location(module_name, OPD_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


POST_TRAIN_OPD = load_post_train_opd()


class TestPostTrainOPD(unittest.TestCase):
    def test_top_p_sampling_keeps_highest_probability_token(self) -> None:
        """A tiny top-p should leave only the highest-logit token eligible."""

        logits = torch.tensor([[9.0, 0.0, -3.0]])
        sampled = POST_TRAIN_OPD.next_token_id(logits, temperature=1.0, top_p=0.01)

        self.assertEqual(sampled.shape, (1, 1))
        self.assertEqual(sampled.item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
