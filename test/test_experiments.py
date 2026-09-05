"""Numerical and protocol regressions for controlled architecture experiments."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from arch_test_utils import ROOT
from experiments.config import (VARIANTS, TrainSettings, common_token_budget,
                                initialized_model, select_candidates, tiny_config, variant_config)
from experiments.data import Blocks, BatchStream, batches, cap_targets, prepare, target_count, verify_data
from experiments.diagnostics import check_contract, matched_dense_config
from experiments.engine import evaluate, run_training
from model.model import Config, attn_res


def make_fixture(root):
    root = Path(root)
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
             **{f"t{i}": i+4 for i in range(128)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                       bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")
    tokenizer.save_pretrained(root/"tokenizer")
    source = root/"source.jsonl"
    texts = [" ".join([f"t{i}"] + [f"t{j%10}" for j in range(40)]) for i in range(110)]
    source.write_text("\n".join(json.dumps({"text": x}) for x in texts + [texts[0], ""]))
    metadata = prepare(source, root/"tokenizer", root/"data")
    cfg = replace(tiny_config(), **{k: metadata[k] for k in ("vocab_size", "pad_token_id", "bos_token_id", "eos_token_id")})
    return root/"data", cfg


class TestArchitectureExperiments(unittest.TestCase):
    def test_all_variants_causal_cache_padding_gradients(self):
        for name in VARIANTS:
            with self.subTest(name=name):
                result = check_contract(variant_config(tiny_config(), name), fit_steps=1)
                self.assertEqual(result["status"], "ready")

    def test_attnres_fuses_only_at_layer_cycle_and_no_duplicate_source(self):
        cfg = replace(tiny_config(), n_layer=5, ratio=1)
        model = initialized_model(cfg, 1)
        captured = []
        def record(sources, partial, norm, query):
            captured.append((len(sources), partial))
            self.assertEqual(len({id(x) for x in sources}), len(sources))
            return attn_res(sources, partial, norm, query)
        with patch("model.model.attn_res", side_effect=record):
            model(input_ids=torch.tensor([[3, 4, 5]]))
        self.assertEqual(captured, [(3, None), (3, None), (2, None)])

    def test_common_weights_and_nonshared_expert_semantics(self):
        base = tiny_config()
        baseline = initialized_model(base, 42)
        mamba = initialized_model(variant_config(base, "M"), 42, base)
        self.assertTrue(torch.equal(baseline.embd.weight, mamba.embd.weight))
        self.assertTrue(torch.equal(baseline.model.layers[1].mixer.q_down.weight,
                                    mamba.model.layers[1].mixer.q_down.weight))
        other = initialized_model(variant_config(base, "F3"), 42, base)
        raw_other = initialized_model(variant_config(base, "F3"), 42)
        self.assertTrue(torch.equal(raw_other.model.layers[0].ffn.shared_expert[0].W_down.weight,
                                    other.model.layers[0].ffn.shared_expert[0].W_down.weight))
        self.assertFalse(torch.equal(baseline.model.layers[0].ffn.shared_expert[0].W_down.weight,
                                     other.model.layers[0].ffn.shared_expert[0].W_down.weight))
        self.assertTrue(torch.equal(baseline.model.layers[0].ffn.gate.weight,
                                    other.model.layers[0].ffn.gate.weight))

    def test_matrix_changes_and_selection(self):
        base = asdict(tiny_config())
        for name, changes in VARIANTS.items():
            cfg = asdict(variant_config(tiny_config(), name))
            self.assertEqual({k: v for k, v in cfg.items() if base[k] != v}, {k: v for k, v in changes.items() if base[k] != v})
        candidates = select_candidates({"B": 3, "A1": 2, "A2": 2.1, "M": 2.2, "F1": 3.2})
        self.assertEqual([x["name"] for x in candidates], ["B", "A1", "A1+M"])
        candidates = select_candidates({"B": 2, "M": 3, "A1": 4})
        self.assertEqual([x["name"] for x in candidates], ["B", "M", "A1"])
        self.assertEqual(common_token_budget([100, 200], 1), 204000)
        cfg, result = matched_dense_config(tiny_config())
        self.assertFalse(cfg.use_moe)
        self.assertLessEqual(result["relative_error"], 0.02)


class TestDataAndTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data, cls.cfg = make_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_document_split_dedup_and_target_coverage(self):
        metadata = verify_data(self.data)
        self.assertEqual(sum(metadata["document_counts"].values()), 110)
        sets = [set(x["hash"] for x in json.loads((self.data/f"{split}.index.json").read_text()))
                for split in ("train", "validation", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        blocks = Blocks(self.data, "train", 8)
        actual = sum(len(blocks[i])-1 for i in range(len(blocks)))
        self.assertEqual(actual, metadata["token_counts"]["train"]-metadata["document_counts"]["train"])
        a = BatchStream(blocks, 2, 0, 2026)
        torch.rand(50)
        b = BatchStream(blocks, 2, 0, 2026)
        for _ in range(3):
            self.assertTrue(torch.equal(a.next()["input_ids"], b.next()["input_ids"]))
        c = BatchStream(blocks, 2, 0, 2026, **a.state_dict())
        self.assertTrue(torch.equal(a.next()["input_ids"], c.next()["input_ids"]))

    def test_calibration_oom_restarts_all_variants_at_shared_length(self):
        from experiments.cli import main, provenance
        from experiments.config import write_json, read_json
        settings = TrainSettings(seq_len=512, tokens_per_update=16, device="cpu", dtype="float32")
        preflight = provenance(self.data, self.cfg, settings)
        preflight.update(formal_fit_check=True, variants={name: {"status": "ready"} for name in VARIANTS})
        preflight_path, output = self.root/"oom-preflight.json", self.root/"oom-schedule.json"
        write_json(preflight_path, preflight)
        calls = []
        def measure(cfg, base, settings, *args):
            calls.append((cfg.mixer_type, settings.seq_len))
            if cfg.mixer_type == "mamba2" and settings.seq_len == 512:
                raise torch.OutOfMemoryError("simulated GPU OOM")
            return {"effective_tokens_per_second": 100, "peak_memory_bytes": 1024}
        with patch("experiments.cli.training_throughput", side_effect=measure):
            main(["calibrate", "--preflight", str(preflight_path), "--output", str(output), "--token-cap", "20"])
        schedule = read_json(output)
        self.assertEqual(schedule["training"]["seq_len"], 256)
        self.assertEqual(len(schedule["runs"]), 8)
        self.assertEqual(sum(length == 256 for _, length in calls), 8)
        self.assertFalse(schedule["rejected"])
        schedule["training"]["max_tokens"] += 1
        write_json(output, schedule)
        with self.assertRaisesRegex(ValueError, "指纹"):
            main(["run", "--schedule", str(output), "--output", str(self.root/"must-not-run")])
        self.assertFalse((self.root/"must-not-run").exists())

    def test_freeze_rejects_missing_runs_and_evaluate_rejects_tampering(self):
        from experiments.cli import main, provenance
        from experiments.config import write_json, digest
        settings = TrainSettings(max_tokens=20, seq_len=8, device="cpu", dtype="float32")
        document = provenance(self.data, self.cfg, settings)
        document.update(stage="confirm", smoke=True, runs=[{"name": "B", "seed": 42, "overrides": {}}])
        document["schedule_id"] = digest(document)
        schedule_path = self.root/"incomplete-schedule.json"
        write_json(schedule_path, document)
        with self.assertRaises(FileNotFoundError):
            main(["freeze", "--schedule", str(schedule_path), "--runs", str(self.root/"missing"),
                  "--output", str(self.root/"must-not-freeze.json")])
        self.assertFalse((self.root/"must-not-freeze.json").exists())
        corrupted = {"schedule": document, "runs": [], "test_protocol": {}}
        corrupted["freeze_id"] = digest(corrupted)
        corrupted["test_protocol"]["decode_steps"] = 1
        path = self.root/"tampered-freeze.json"
        write_json(path, corrupted)
        with self.assertRaisesRegex(ValueError, "冻结评估协议"):
            main(["evaluate", "--freeze", str(path)])

    def test_evaluation_is_token_weighted_and_partition_invariant(self):
        model = initialized_model(self.cfg, 11)
        data = Blocks(self.data, "validation", 8)
        settings = TrainSettings(seq_len=8, device="cpu", dtype="float32", micro_batch=1)
        one = evaluate(model, data, settings, torch.device("cpu"), "float32")
        two = evaluate(model, data, replace(settings, micro_batch=2), torch.device("cpu"), "float32")
        self.assertAlmostEqual(one["ce"], two["ce"], places=5)
        limited = evaluate(model, data, settings, torch.device("cpu"), "float32", token_limit=5)
        self.assertEqual(limited["targets"], 5)
        batch = next(batches(data, 2, 0))
        self.assertEqual(target_count(cap_targets(batch, 3)), 3)

    def test_gradient_accumulation_matches_concatenated_tokens(self):
        # 无MoE辅助项时，变长microbatch的加权梯度必须等价于合并batch。
        cfg = variant_config(self.cfg, "F2")
        a = initialized_model(cfg, 5)
        b = copy.deepcopy(a)
        ids = torch.tensor([[3, 4, 5, 6], [3, 4, 0, 0]])
        mask = ids.ne(0)
        labels = ids.masked_fill(~mask, -100)
        a(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()
        total = int(labels[:, 1:].ne(-100).sum())
        for i in range(2):
            n = int(labels[i:i+1, 1:].ne(-100).sum())
            (b(input_ids=ids[i:i+1], attention_mask=mask[i:i+1], labels=labels[i:i+1]).loss*n/total).backward()
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=3e-4, msg=name)

    def test_exact_budget_resume_and_mismatched_config_rejection(self):
        settings = TrainSettings(max_tokens=23, seq_len=8, tokens_per_update=7,
                                 device="cpu", dtype="float32", eval_intervals=2)
        run_dir = self.root/"resume_run"
        from experiments import engine
        original_save = engine.save_checkpoint
        def save_then_stop(*args, **kwargs):
            original_save(*args, **kwargs)
            raise RuntimeError("simulated interruption")
        with patch.object(engine, "save_checkpoint", side_effect=save_then_stop):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                run_training(self.data, run_dir, self.cfg, self.cfg, settings)
        resumed = run_training(self.data, run_dir, self.cfg, self.cfg, settings, resume=True)
        full_dir = self.root/"full_run"
        full = run_training(self.data, full_dir, self.cfg, self.cfg, settings)
        self.assertEqual(resumed["tokens"], 23)
        self.assertAlmostEqual(resumed["validation"]["ce"], full["validation"]["ce"], places=7)
        left = torch.load(run_dir/"checkpoint.pt", weights_only=False)["model"]
        right = torch.load(full_dir/"checkpoint.pt", weights_only=False)["model"]
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "拒绝恢复"):
            run_training(self.data, run_dir, self.cfg, self.cfg, replace(settings, max_tokens=24), resume=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
