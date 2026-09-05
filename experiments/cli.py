"""python -m experiments.cli --help"""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import math

import torch
from model.model import Config
from .config import (VARIANTS, TrainSettings, common_token_budget, digest, environment,
                     file_hash, initialized_model, read_json, select_candidates,
                     source_identity, tiny_config, variant_config, write_json)
from .data import Blocks, prepare, verify_data
from .diagnostics import check_contract, inference_benchmark, matched_dense_config, training_throughput
from .engine import evaluate, run_training, runtime


def base_for_data(data, model_json=None, tiny=False):
    metadata = verify_data(data)
    values = asdict(tiny_config()) if tiny else {}
    if model_json:
        values.update(read_json(model_json))
    values.update({k: metadata[k] for k in ("vocab_size", "pad_token_id", "bos_token_id", "eos_token_id")})
    return Config(**values), metadata


def settings_for_args(args):
    return TrainSettings(seq_len=args.seq_len, micro_batch=args.micro_batch,
                         tokens_per_update=args.tokens_per_update, device=args.device, dtype=args.dtype)


def provenance(data, base, settings):
    device, dtype = runtime(settings)
    return {"data": str(Path(data).resolve()), "data_fingerprint": verify_data(data)["fingerprint"],
            "base_model": asdict(base), "training": asdict(settings), "source": source_identity(),
            "environment": environment(device), "effective_dtype": dtype}


def check_provenance(document):
    if "schedule_id" in document and digest({k: v for k, v in document.items() if k != "schedule_id"}) != document["schedule_id"]:
        raise ValueError("schedule指纹不匹配；请勿手改冻结计划")
    if document["source"] != source_identity():
        raise ValueError("源码已改变；请重新执行预检/测速并冻结新实验")
    if verify_data(document["data"])["fingerprint"] != document["data_fingerprint"]:
        raise ValueError("数据已改变")
    settings = TrainSettings(**document["training"])
    device, dtype = runtime(settings)
    if document["environment"] != environment(device) or document["effective_dtype"] != dtype:
        raise ValueError("实验环境或有效精度已改变")
    return Config(**document["base_model"]), settings


def require_new(path):
    if Path(path).exists():
        raise FileExistsError(f"拒绝覆盖已冻结的配置: {path}")


def run_path(root, stage, name, seed):
    return Path(root)/stage/f"{name}-seed{seed}"


def cmd_preflight(args):
    require_new(args.output)
    base, _ = base_for_data(args.data, args.model_json, args.tiny)
    settings = settings_for_args(args)
    document = provenance(args.data, base, settings)
    device, dtype = runtime(settings)
    document["variants"] = {}
    for name in VARIANTS:
        print(f"Checking {name} ...", flush=True)
        try:
            result = check_contract(variant_config(base, name), device, dtype, args.fit_steps)
        except Exception as exc:
            result = {"status": "not_ready", "error": f"{type(exc).__name__}: {exc}"}
        document["variants"][name] = result
    document["formal_fit_check"] = args.fit_steps >= 100
    write_json(args.output, document)
    print({k: v["status"] for k, v in document["variants"].items()})
    if document["variants"]["B"]["status"] != "ready":
        raise RuntimeError("基线未通过；报告已保存")


def cmd_calibrate(args):
    require_new(args.output)
    preflight = read_json(args.preflight)
    base, settings = check_provenance(preflight)
    if not preflight["formal_fit_check"] and not args.smoke:
        raise ValueError("正式实验需要至少100步的小样本拟合检查；快速测试请显式--smoke")
    ready = [k for k, v in preflight["variants"].items() if v["status"] == "ready"]
    rejected = {k: v for k, v in preflight["variants"].items() if v["status"] != "ready"}
    while True:
        results, oom = {}, False
        for name in ready:
            print(f"Timing {name}, seq_len={settings.seq_len} ...", flush=True)
            try:
                results[name] = training_throughput(variant_config(base, name), base, settings,
                                                    preflight["data"], args.warmup, args.steps)
            except torch.OutOfMemoryError:
                oom = True
                rejected[name] = {"status": "out_of_memory", "seq_len": settings.seq_len}
            except Exception as exc:
                rejected[name] = {"status": "not_ready", "error": f"{type(exc).__name__}: {exc}"}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if oom and settings.seq_len == 512:
            # 撤销本轮所有测速结果，对全部候选统一使用256。
            settings = replace(settings, seq_len=256)
            continue
        break
    if "B" not in results or len(results) < 3:
        raise RuntimeError("需要基线和至少两个可运行候选")
    for name in results:
        rejected.pop(name, None)
    rates = [v["effective_tokens_per_second"] for v in results.values()]
    screen_tokens = common_token_budget(rates, args.screen_hours)
    # 先以两倍最慢单项耗时保守预留9个确认运行；组合选出后再次实测。
    provisional_confirm = common_token_budget([min(rates)/2]*9, args.confirm_hours)
    screen_tokens = min(screen_tokens, provisional_confirm//4)
    if args.token_cap is not None:
        screen_tokens = min(screen_tokens, args.token_cap)
    if screen_tokens < settings.eval_intervals:
        raise ValueError("预算过小；每个验证区间至少需要1个target")
    settings = replace(settings, max_tokens=screen_tokens)
    document = provenance(preflight["data"], base, settings)
    document.update(stage="screen", smoke=args.smoke, measurements=results, rejected=rejected,
                    screen_hours=args.screen_hours, confirm_hours=args.confirm_hours,
                    preflight_sha256=file_hash(args.preflight),
                    runs=[{"name": name, "seed": 42, "overrides": VARIANTS[name]} for name in results])
    document["schedule_id"] = digest(document)
    write_json(args.output, document)
    print(f"Frozen screen budget: {screen_tokens:,} effective targets per run")


def cmd_run(args):
    schedule = read_json(args.schedule)
    if digest({k: v for k, v in schedule.items() if k != "schedule_id"}) != schedule["schedule_id"]:
        raise ValueError("schedule指纹不匹配；请勿手改冻结计划")
    base, settings = check_provenance(schedule)
    failed = []
    for run in schedule["runs"]:
        name, seed = run["name"], run["seed"]
        cfg = variant_config(base, name, run["overrides"])
        directory = run_path(args.output, schedule["stage"], name, seed)
        try:
            run_training(schedule["data"], directory, cfg, base, replace(settings, seed=seed),
                         name, schedule["stage"], args.resume and directory.exists(), smoke=schedule["smoke"])
        except Exception as exc:
            failed.append({"name": name, "seed": seed, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {name}/{seed}: {exc}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if failed:
        write_json(Path(args.output)/f'{schedule["stage"]}-failures.json', failed)
        raise RuntimeError("部分运行失败；已记录，失败结果不参与排名")


def verified_summary(directory, schedule, run):
    summary = read_json(directory/"summary.json")
    identity = read_json(directory/"run.json")
    expected_model = asdict(variant_config(Config(**schedule["base_model"]), run["name"], run["overrides"]))
    expected_settings = {**schedule["training"], "seed": run["seed"]}
    expected = {"model": expected_model, "base_model": schedule["base_model"], "training": expected_settings,
                "data_fingerprint": schedule["data_fingerprint"], "source": schedule["source"],
                "environment": schedule["environment"], "effective_dtype": schedule["effective_dtype"],
                "stage": schedule["stage"], "name": run["name"], "smoke": schedule["smoke"]}
    if any(identity.get(k) != v for k, v in expected.items()) or summary["identity"] != identity["fingerprint"]:
        raise ValueError(f"运行不属于本实验计划: {directory}")
    if digest({k: v for k, v in identity.items() if k != "fingerprint"}) != identity["fingerprint"]:
        raise ValueError("run.json指纹损坏")
    if summary["status"] != "complete" or summary["tokens"] != expected_settings["max_tokens"]:
        raise ValueError(f"运行未完成统一token预算: {directory}")
    if not math.isfinite(summary["validation"]["ce"]):
        raise ValueError("无效验证CE")
    return summary, identity


def cmd_select(args):
    require_new(args.output)
    screen = read_json(args.schedule)
    if screen["stage"] != "screen":
        raise ValueError("select仅接受筛选计划")
    base, settings = check_provenance(screen)
    scores = {}
    for run in screen["runs"]:
        directory = run_path(args.runs, "screen", run["name"], run["seed"])
        if (directory/"summary.json").exists():
            summary, _ = verified_summary(directory, screen, run)
            scores[run["name"]] = summary["validation"]["ce"]
    candidates = select_candidates(scores)
    measurements = {}
    device, dtype = runtime(settings)
    for candidate in candidates:
        cfg = variant_config(base, candidate["name"], candidate["overrides"])
        # 新组合必须独立通过正确性与拟合门槛，不从单项结果推断其可用性。
        check_contract(cfg, device, dtype, 1 if screen["smoke"] else 100)
        measurements[candidate["name"]] = training_throughput(cfg, base, settings, screen["data"], args.warmup, args.steps)
    hours = args.hours if args.hours is not None else screen["confirm_hours"]
    tokens = common_token_budget([m["effective_tokens_per_second"] for m in measurements.values() for _ in range(3)], hours)
    if args.token_cap is not None:
        tokens = min(tokens, args.token_cap)
    if tokens < settings.max_tokens*4:
        required = settings.max_tokens*4*sum(3/m["effective_tokens_per_second"] for m in measurements.values())/(0.85*3600)
        raise ValueError(f"确认预算不足4倍筛选tokens，需要至少{required:.3f}小时；增加--hours，或重做更短筛选")
    document = provenance(screen["data"], base, replace(settings, max_tokens=tokens))
    document.update(stage="confirm", smoke=screen["smoke"], candidates=candidates, screen_scores=scores,
                    screen_schedule_sha256=file_hash(args.schedule), measurements=measurements, confirm_hours=hours,
                    runs=[{**candidate, "seed": seed} for candidate in candidates for seed in (42, 43, 44)])
    document["schedule_id"] = digest(document)
    write_json(args.output, document)
    print(f"Candidates: {[c['name'] for c in candidates]}; {tokens:,} targets/run")


def cmd_match(args):
    require_new(args.output)
    reference = read_json(args.schedule)
    base, settings = check_provenance(reference)
    cfg, match = matched_dense_config(base)
    if match["status"] != "matched":
        raise ValueError(f"无法达到2%容差: {match}")
    device, dtype = runtime(settings)
    check_contract(cfg, device, dtype, 1 if reference["smoke"] else 100)
    measured = training_throughput(cfg, base, settings, reference["data"], args.warmup, args.steps)
    reference_rate = reference["measurements"]["B"]["effective_tokens_per_second"]
    tokens = common_token_budget([reference_rate, measured["effective_tokens_per_second"]], args.hours)
    if args.token_cap is not None:
        tokens = min(tokens, args.token_cap)
    if tokens < settings.eval_intervals:
        raise ValueError("参数对齐预算过小")
    document = provenance(reference["data"], base, replace(settings, max_tokens=tokens))
    document.update(stage="matched", smoke=reference["smoke"], match=match, measurements={"dense_matched": measured},
                    interpretation="Single-seed exploratory capacity control; does not equalize FLOPs.",
                    runs=[{"name": "B", "seed": 42, "overrides": {}},
                          {"name": "dense_matched", "seed": 42,
                           "overrides": {"use_moe": False, "ffn_type": "situglu", "d_inner": cfg.d_inner}}])
    document["schedule_id"] = digest(document)
    write_json(args.output, document)


def cmd_freeze(args):
    require_new(args.output)
    schedule = read_json(args.schedule)
    check_provenance(schedule)
    if schedule["stage"] not in {"confirm", "matched"}:
        raise ValueError("只允许冻结确认/参数对齐阶段，筛选阶段不能查看测试集")
    runs = []
    for run in schedule["runs"]:
        directory = run_path(args.runs, schedule["stage"], run["name"], run["seed"]).resolve()
        summary, identity = verified_summary(directory, schedule, run)
        runs.append({"directory": str(directory), "identity": identity["fingerprint"],
                     "checkpoint_sha256": file_hash(directory/"checkpoint.pt"),
                     "summary_sha256": file_hash(directory/"summary.json")})
    document = {"schedule": schedule, "runs": runs,
                "test_protocol": {"lengths": [512, 1024, 2048], "prompt_lengths": [128, 512, 2048],
                                  "decode_steps": 128, "repeats": 5, "warmup": 2,
                                  "main_metric": "token_weighted_test_ce", "selection_metric": "endpoint_validation_ce"}}
    if schedule["smoke"]:
        document["test_protocol"].update(lengths=[8, 16, 32], prompt_lengths=[8, 16, 32], decode_steps=4, repeats=2, warmup=1)
    document["freeze_id"] = digest(document)
    write_json(args.output, document)


def cmd_evaluate(args):
    frozen = read_json(args.freeze)
    if digest({k: v for k, v in frozen.items() if k != "freeze_id"}) != frozen["freeze_id"]:
        raise ValueError("冻结评估协议被修改")
    schedule, protocol = frozen["schedule"], frozen["test_protocol"]
    base, settings = check_provenance(schedule)
    device, dtype = runtime(settings)
    for entry in frozen["runs"]:
        directory = Path(entry["directory"])
        if file_hash(directory/"checkpoint.pt") != entry["checkpoint_sha256"] or file_hash(directory/"summary.json") != entry["summary_sha256"]:
            raise ValueError("冻结后权重或summary被修改")
        identity = read_json(directory/"run.json")
        if identity["fingerprint"] != entry["identity"]:
            raise ValueError("冻结运行身份不匹配")
        output_path = directory/"evaluation.json"
        if output_path.exists():
            if read_json(output_path)["freeze_id"] != frozen["freeze_id"]:
                raise ValueError("已有不同冻结协议的评估结果")
            print(f"Already evaluated: {directory.name}")
            continue
        model = initialized_model(Config(**identity["model"]), identity["training"]["seed"]).to(device)
        checkpoint = torch.load(directory/"checkpoint.pt", map_location="cpu", weights_only=False)
        if checkpoint["identity"] != identity:
            raise ValueError("checkpoint与run.json不一致")
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        # 主测试指标使用训练长度与完整保留集，不以长度外推结果重新选型。
        main = evaluate(model, Blocks(schedule["data"], "test", settings.seq_len), settings, device, dtype)
        lengths = {}
        for length in protocol["lengths"]:
            dataset = Blocks(schedule["data"], "test", length, long_only=True)
            try:
                lengths[str(length)] = {**evaluate(model, dataset, settings, device, dtype),
                                        "extrapolation": length > settings.seq_len}
            except torch.OutOfMemoryError:
                lengths[str(length)] = {"status": "out_of_memory", "extrapolation": length > settings.seq_len}
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        benchmark = inference_benchmark(model, settings, protocol["prompt_lengths"], protocol["decode_steps"], protocol["repeats"], protocol["warmup"])
        write_json(output_path, {"freeze_id": frozen["freeze_id"], "test": main,
                                 "length_evaluation": lengths, "inference": benchmark})
        print(f"Test CE {directory.name}: {main['ce']:.5f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description="YuchenModel受控模块消融实验")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="去重、文档切分、冻结tokenizer并生成mmap token文件")
    p.add_argument("--source", required=True); p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True); p.add_argument("--seed", type=int, default=2026)
    p.set_defaults(func=lambda a: prepare(a.source, a.tokenizer, a.output, a.seed))
    p = sub.add_parser("preflight", help="8组的因果性/cache/padding/梯度和小样本拟合预检")
    p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--model-json"); p.add_argument("--tiny", action="store_true")
    p.add_argument("--fit-steps", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=512); p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--tokens-per-update", type=int, default=16384)
    p.add_argument("--device", default="auto"); p.add_argument("--dtype", default="auto")
    p.set_defaults(func=cmd_preflight)
    p = sub.add_parser("calibrate", help="实测吞吐、统一长度回退，冻结筛选预算")
    p.add_argument("--preflight", required=True); p.add_argument("--output", required=True)
    p.add_argument("--screen-hours", type=float, default=12); p.add_argument("--confirm-hours", type=float, default=48)
    p.add_argument("--warmup", type=int, default=3); p.add_argument("--steps", type=int, default=10)
    p.add_argument("--token-cap", type=int, help="冻结前对每组训练target预算设置上限")
    p.add_argument("--smoke", action="store_true", help="仅流程测试，结果不作研究证据")
    p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("run", help="依冻结计划顺序训练所有模型和种子")
    p.add_argument("--schedule", required=True); p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true"); p.set_defaults(func=cmd_run)
    p = sub.add_parser("select", help="按验证CE选单项/组合，测速并冻结3种子确认计划")
    p.add_argument("--schedule", required=True); p.add_argument("--runs", required=True)
    p.add_argument("--output", required=True); p.add_argument("--hours", type=float)
    p.add_argument("--token-cap", type=int, help="确认训练预算上限，仍须>=4倍筛选预算")
    p.add_argument("--warmup", type=int, default=3); p.add_argument("--steps", type=int, default=10)
    p.set_defaults(func=cmd_select)
    p = sub.add_parser("match", help="生成同总参数量稠密对照和同token基线")
    p.add_argument("--schedule", required=True); p.add_argument("--output", required=True)
    p.add_argument("--hours", type=float, default=6)
    p.add_argument("--token-cap", type=int, help="参数对齐训练target上限")
    p.add_argument("--warmup", type=int, default=3); p.add_argument("--steps", type=int, default=10)
    p.set_defaults(func=cmd_match)
    p = sub.add_parser("freeze", help="锁定权重及测试协议，完成后才可访问测试评估")
    p.add_argument("--schedule", required=True); p.add_argument("--runs", required=True)
    p.add_argument("--output", required=True); p.set_defaults(func=cmd_freeze)
    p = sub.add_parser("evaluate", help="执行冻结后的测试集/长度外推/推理评估")
    p.add_argument("--freeze", required=True); p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("report", help="生成CE曲线、延迟散点图和多种子汇总")
    p.add_argument("--runs", required=True); p.add_argument("--output", required=True)
    def report(args):
        from .report import build_report
        build_report(args.runs, args.output)
    p.set_defaults(func=report)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
