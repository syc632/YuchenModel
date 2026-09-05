"""Export standalone publication-friendly plots and cautious seed summaries."""
import csv
from collections import defaultdict
from pathlib import Path
import statistics
import os
import tempfile

from .config import read_json, write_json


def mean_sd(values):
    values = [x for x in values if x is not None]
    return (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None) if values else (None, None)


def build_report(runs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"yuchen-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, groups = [], defaultdict(list)
    for path in sorted(Path(runs).rglob("summary.json")):
        summary = read_json(path)
        if summary.get("status") != "complete":
            continue
        directory = path.parent
        identity = read_json(directory/"run.json")
        if summary["identity"] != identity["fingerprint"]:
            raise ValueError(f"summary身份不一致: {directory}")
        evaluation = read_json(directory/"evaluation.json") if (directory/"evaluation.json").exists() else None
        record = {"summary": summary, "identity": identity,
                  "metrics": read_json(directory/"metrics.json"), "evaluation": evaluation}
        records.append(record)
        # 不混合不同数据、预算、主干或硬件的实验。
        from .config import digest
        cohort = digest({k: identity[k] for k in ("base_model", "source", "data_fingerprint", "environment", "effective_dtype", "smoke")}
                        | {"training": {k: v for k, v in identity["training"].items() if k != "seed"}})
        group_key = (cohort, summary["stage"], summary["name"])
        if groups[group_key] and groups[group_key][0]["identity"]["model"] != identity["model"]:
            raise ValueError("同名实验的模型配置不同，不能合并种子")
        groups[group_key].append(record)
    if not records:
        raise ValueError("没有完成的运行")
    rows = []
    for (cohort, stage, name), group in groups.items():
        summaries = [x["summary"] for x in group]
        seeds = [x["seed"] for x in summaries]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"同一实验存在重复seed: {name}/{stage}")
        test = [x["evaluation"]["test"]["ce"] for x in group if x["evaluation"]]
        ppl = [x["evaluation"]["test"]["ppl"] for x in group if x["evaluation"]]
        val_mean, val_sd = mean_sd([x["validation"]["ce"] for x in summaries])
        test_mean, test_sd = mean_sd(test)
        ppl_mean, ppl_sd = mean_sd(ppl)
        inference = [b["decode_ms_per_token"] for x in group if x["evaluation"]
                     for b in x["evaluation"]["inference"] if b["status"] == "ok" and b["prompt_length"] == 512]
        train_mean, train_sd = mean_sd([x["train_seconds"] for x in summaries])
        peak, _ = mean_sd([x["peak_memory_bytes"] for x in summaries])
        latency, latency_sd = mean_sd(inference)
        status = "single_seed_exploratory" if len(seeds) < 3 else "multi_seed_descriptive"
        delta_mean = delta_sd = None
        baseline = groups.get((cohort, stage, "B"), [])
        baseline_by_seed = {x["summary"]["seed"]: x for x in baseline}
        pairs = [(x, baseline_by_seed[x["summary"]["seed"]]) for x in group
                 if x["summary"]["seed"] in baseline_by_seed and x["evaluation"]
                 and baseline_by_seed[x["summary"]["seed"]]["evaluation"]]
        if name != "B" and len(pairs) >= 3:
            deltas = [a["evaluation"]["test"]["ce"]-b["evaluation"]["test"]["ce"] for a, b in pairs]
            delta_mean, delta_sd = mean_sd(deltas)
            status = ("consistent_gain_at_this_budget" if all(x < 0 for x in deltas) and abs(delta_mean) > delta_sd
                      else "insufficient_evidence" if not all(x > 0 for x in deltas) or abs(delta_mean) <= delta_sd
                      else "consistent_regression_at_this_budget")
        if any(x["identity"].get("smoke") for x in group):
            status = "workflow_smoke_no_research_conclusion"
        rows.append({"cohort": cohort[:12], "stage": stage, "name": name, "seeds": ",".join(map(str, sorted(seeds))),
                     "n_seeds": len(seeds), "n_test_seeds": len(test), "tokens": summaries[0]["tokens"],
                     "validation_ce_mean": val_mean, "validation_ce_sd": val_sd,
                     "test_ce_mean": test_mean, "test_ce_sd": test_sd, "test_ppl_mean": ppl_mean, "test_ppl_sd": ppl_sd,
                     "parameters": summaries[0]["parameters"], "active_parameters_estimate": summaries[0]["active_parameters_estimate"],
                     "train_seconds_mean": train_mean, "train_seconds_sd": train_sd, "peak_memory_bytes_mean": peak,
                     "decode_ms_512_mean": latency, "decode_ms_512_sd": latency_sd,
                     "paired_ce_delta_mean": delta_mean, "paired_ce_delta_sd": delta_sd,
                     "skipped_updates": sum(s["skipped_updates"] for s in summaries), "interpretation": status})
    with (output/"summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(output/"summary.json", rows)
    panels = sorted({(cohort, stage) for cohort, stage, _ in groups})
    for xfield, xlabel, filename, factor in (("tokens", "Effective training targets", "ce_vs_tokens", 1),
                                             ("wall_seconds", "Elapsed training + validation hours", "ce_vs_hours", 3600)):
        fig, axes = plt.subplots(len(panels), 1, figsize=(9, 4*len(panels)), squeeze=False)
        for ax, (cohort, stage) in zip(axes[:, 0], panels):
            for (group_cohort, group_stage, name), group in groups.items():
                if (group_cohort, group_stage) != (cohort, stage):
                    continue
                histories = [r["metrics"]["history"] for r in group]
                points = sorted(set.intersection(*(set(h["tokens"] for h in history) for history in histories)))
                maps = [{h["tokens"]: h for h in history} for history in histories]
                xs, ys, sds = [], [], []
                for point in points:
                    xs.append(statistics.mean(m[point][xfield] for m in maps)/factor)
                    mean, sd = mean_sd([m[point]["validation"]["ce"] for m in maps])
                    ys.append(mean); sds.append(sd or 0)
                line, = ax.plot(xs, ys, label=f"{name} (n={len(group)})")
                if len(group) > 1:
                    ax.fill_between(xs, [m-s for m, s in zip(ys, sds)], [m+s for m, s in zip(ys, sds)],
                                    color=line.get_color(), alpha=0.15)
            ax.set(title=f"{stage} / {cohort[:8]}", xlabel=xlabel, ylabel="Validation cross-entropy")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, ncol=4)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(output/f"{filename}.{ext}", dpi=180)
        plt.close(fig)
    latency_panels = [(cohort, stage) for cohort, stage in panels
                      if any(r["evaluation"] for (c, st, _), group in groups.items()
                             if (c, st) == (cohort, stage) for r in group)]
    fig, axes = plt.subplots(max(1, len(latency_panels)), 1, figsize=(9, 4*max(1, len(latency_panels))), squeeze=False)
    if not latency_panels:
        axes[0, 0].text(0.5, 0.5, "No frozen test / latency evaluation yet", ha="center", transform=axes[0, 0].transAxes)
    for ax, (cohort, stage) in zip(axes[:, 0], latency_panels):
        panel_groups = [(name, group) for (c, st, name), group in groups.items() if (c, st) == (cohort, stage)]
        smoke = any(r["identity"].get("smoke") for _, group in panel_groups for r in group)
        prompt_length = 16 if smoke else 512
        for index, (name, group) in enumerate(panel_groups):
            pairs = [(r["evaluation"]["test"]["ce"], b["decode_ms_per_token"]) for r in group if r["evaluation"]
                     for b in r["evaluation"]["inference"] if b["status"] == "ok" and b["prompt_length"] == prompt_length]
            if not pairs:
                continue
            ce, ce_sd = mean_sd([p[0] for p in pairs])
            latency, latency_sd = mean_sd([p[1] for p in pairs])
            params = group[0]["summary"]["parameters"]
            ax.errorbar(latency, ce, xerr=latency_sd or 0, yerr=ce_sd or 0, fmt="o", capsize=3,
                        label=f"{name}: {params/1e6:.2f}M params (n={len(pairs)})")
        ax.set(title=f"{stage} / prompt length {prompt_length}", xlabel="Decode milliseconds per token", ylabel="Test cross-entropy")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output/f"ce_vs_latency.{ext}", dpi=180)
    plt.close(fig)
    lines = ["# 模块对比实验报告", "", "测试集仅用于冻结后的最终评估。单种子结果为探索性证据；三种子的方向/标准差检查不等价于显著性检验。", "",
             "| 阶段 | 模型 | 种子数 | 验证 CE | 测试 CE | 参数量 | 判断 |",
             "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        def fmt(mean, sd):
            return "尚未评估" if mean is None else f"{mean:.4f}"+(f" ± {sd:.4f}" if sd is not None else "")
        lines.append(f'| {row["stage"]} | {row["name"]} | {row["n_seeds"]} | {fmt(row["validation_ce_mean"], row["validation_ce_sd"])} | {fmt(row["test_ce_mean"], row["test_ce_sd"])} | {row["parameters"]:,} | {row["interpretation"]} |')
    if any(x["identity"].get("smoke") for x in records):
        lines.extend(["", "注意：本报告包含smoke流程测试。其小数据/短训练结果不能作为架构研究结论。", ""])
    lines.extend(["", "同token和同宽度不等于同参数量或同计算量；激活参数估计不是FLOPs。",
                  "GatedMLA比较包含位置编码和门控变化，FFN变体比较包含完整计算形式与bias差异。",
                  "未重复的筛选单项不提供组合收益的确定归因。没有对照的交互效应仍需后续消融。", "",
                  "![CE versus tokens](ce_vs_tokens.png)", "![CE versus hours](ce_vs_hours.png)", "![CE versus latency](ce_vs_latency.png)"])
    (output/"report.md").write_text("\n".join(lines)+"\n")
    return rows
