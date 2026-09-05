"""Full offline workflow on synthetic documents; writes only to the chosen output.

OMP_NUM_THREADS=1 python test/smoke_experiments.py --output /tmp/yuchen-smoke
"""
import argparse
from pathlib import Path
import sys
import torch
from test_experiments import make_fixture
from experiments.cli import main


def run(output):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    data, _ = make_fixture(root)
    def command(*args):
        print("\nCLI:", " ".join(map(str, args)), flush=True)
        main(list(map(str, args)))
    command("preflight", "--data", data, "--tiny", "--seq-len", 8,
            "--tokens-per-update", 16, "--fit-steps", 1, "--device", "cpu", "--output", root/"preflight.json")
    command("calibrate", "--preflight", root/"preflight.json", "--screen-hours", 1, "--confirm-hours", 1,
            "--warmup", 1, "--steps", 1, "--smoke", "--token-cap", 20, "--output", root/"screen.json")
    command("run", "--schedule", root/"screen.json", "--output", root/"runs")
    command("select", "--schedule", root/"screen.json", "--runs", root/"runs", "--warmup", 1,
            "--steps", 1, "--token-cap", 80, "--output", root/"confirm.json")
    command("run", "--schedule", root/"confirm.json", "--output", root/"runs")
    command("freeze", "--schedule", root/"confirm.json", "--runs", root/"runs", "--output", root/"freeze.json")
    command("evaluate", "--freeze", root/"freeze.json")
    command("match", "--schedule", root/"confirm.json", "--token-cap", 20, "--warmup", 1, "--steps", 1,
            "--output", root/"matched.json")
    command("run", "--schedule", root/"matched.json", "--output", root/"runs")
    command("freeze", "--schedule", root/"matched.json", "--runs", root/"runs", "--output", root/"matched-freeze.json")
    command("evaluate", "--freeze", root/"matched-freeze.json")
    command("report", "--runs", root/"runs", "--output", root/"report")
    print(f"\nSmoke workflow complete: {root/'report'/'report.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
