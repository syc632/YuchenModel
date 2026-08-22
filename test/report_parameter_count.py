"""Report the exact parameter count of the current default architecture.

Run from the repository root:
    py -3.14 test/report_parameter_count.py
"""

from __future__ import annotations

from collections import defaultdict

from arch_test_utils import unique_parameter_count


def main() -> None:
    from model.model import Config, YuchenModelCausalLLM

    config = Config()
    model = YuchenModelCausalLLM(config)

    # Parameter tying (embedding == LM head) is counted once by identity.
    total = unique_parameter_count(model)
    layer_counts: dict[str, int] = defaultdict(int)
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        parts = name.split(".")
        group = ".".join(parts[:3]) if parts[:2] == ["model", "layers"] else parts[0]
        layer_counts[group] += parameter.numel()

    print("Default Config parameter report")
    print(f"  unique/trainable parameters: {total:,}")
    print(f"  approximate fp32 weight memory: {total * 4 / 1024**2:.2f} MiB")
    print("  by component:")
    for group, count in sorted(layer_counts.items()):
        print(f"    {group:<18} {count:>12,}")


if __name__ == "__main__":
    main()
