from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rotary_target.adapters.base import CommandSpec
from rotary_target.algorithms import list_algorithms
from rotary_target.config import ConfigError, ExperimentConfig
from rotary_target.data.dota import validate_dota_root
from rotary_target.doctor import run_doctor
from rotary_target.registry import get_adapter
from rotary_target.report import collect_runs, render_markdown


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rt", description="Rotary Target training harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check local scaffold, optional deps, CUDA, and data")
    doctor.add_argument("--data-root", default="~/autodl-tmp/DOTA")
    doctor.add_argument("--strict", action="store_true")
    doctor.add_argument("--json", action="store_true", dest="as_json")
    doctor.set_defaults(func=cmd_doctor)

    algorithms = subparsers.add_parser("algorithms", help="List supported algorithm specs")
    algorithms.set_defaults(func=cmd_algorithms)

    data = subparsers.add_parser("data", help="Dataset helpers")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    validate = data_subparsers.add_parser("validate", help="Validate a DOTA v1.0 root")
    validate.add_argument("--root", default="~/autodl-tmp/DOTA")
    validate.add_argument("--max-files", type=int, default=50)
    validate.add_argument("--json", action="store_true", dest="as_json")
    validate.set_defaults(func=cmd_data_validate)

    split = data_subparsers.add_parser("split-dota", help="Prepare a DOTA split command")
    split.add_argument("--raw-root", default="~/autodl-tmp/DOTA")
    split.add_argument("--out-root", default="data/processed/DOTA-v1.0/split_ss_1024_200")
    split.add_argument("--tile-size", type=int, default=1024)
    split.add_argument("--gap", type=int, default=200)
    split.add_argument("--nproc", type=int, default=10)
    split.add_argument("--config-dir")
    split.add_argument("--backend-root")
    split.add_argument("--dry-run", action="store_true")
    split.set_defaults(func=cmd_data_split)

    train = subparsers.add_parser("train", help="Train an experiment")
    train.add_argument("--experiment", required=True)
    train.add_argument("--backend-root")
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("eval", help="Evaluate an experiment checkpoint")
    evaluate.add_argument("--experiment", required=True)
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--backend-root")
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.set_defaults(func=cmd_eval)

    report = subparsers.add_parser("report", help="Collect metrics.json files into Markdown")
    report.add_argument("--outputs", default="outputs")
    report.add_argument("--output", default="reports/experiment_summary.md")
    report.add_argument("--dry-run", action="store_true")
    report.set_defaults(func=cmd_report)

    ensemble = subparsers.add_parser("ensemble", help="Fuse predictions from multiple models (OWBF)")
    ensemble.add_argument("--predictions", nargs="+", required=True,
                          help="Paths to prediction JSON files from different models")
    ensemble.add_argument("--weights", nargs="*", type=float,
                          help="Model weights (uniform if omitted)")
    ensemble.add_argument("--iou-threshold", type=float, default=0.5)
    ensemble.add_argument("--score-threshold", type=float, default=0.05)
    ensemble.add_argument("--output", default="outputs/ensemble/fused_predictions.json")
    ensemble.add_argument("--dry-run", action="store_true")
    ensemble.set_defaults(func=cmd_ensemble)

    return parser


def cmd_doctor(args: argparse.Namespace) -> int:
    data_root = str(Path(args.data_root).expanduser())
    report = run_doctor(data_root, strict=args.strict)
    if args.as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        for check in report.checks:
            print(f"[{check.status}] {check.name}: {check.detail}")
    return 0 if report.ok else 1


def cmd_algorithms(args: argparse.Namespace) -> int:
    for spec in list_algorithms():
        print(
            f"{spec.key}\t{spec.display_name}\t{spec.family}\t"
            f"{spec.angle_version}\t{spec.upstream_config}"
        )
    return 0


def cmd_data_validate(args: argparse.Namespace) -> int:
    root = str(Path(args.root).expanduser())
    report = validate_dota_root(root, max_label_files=args.max_files)
    if args.as_json:
        payload = {
            "root": str(report.root),
            "exists": report.exists,
            "ok": report.ok,
            "checked_label_files": report.checked_label_files,
            "objects": report.objects,
            "errors": report.errors,
            "warnings": report.warnings,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"DOTA root: {report.root}")
        print(f"Exists: {report.exists}")
        print(f"Checked label files: {report.checked_label_files}")
        print(f"Objects seen: {report.objects}")
        for warning in report.warnings:
            print(f"warning: {warning}")
        for error in report.errors:
            print(f"error: {error}")
    return 0 if report.ok else 1


def cmd_data_split(args: argparse.Namespace) -> int:
    from rotary_target.adapters.mmrotate import MMRotateAdapter
    from rotary_target.data.split_config import build_mmrotate_split_configs, write_mmrotate_split_configs

    config_dir = (Path(args.config_dir) if args.config_dir else Path(args.out_root) / "split_configs").resolve()
    configs = build_mmrotate_split_configs(
        args.raw_root,
        args.out_root,
        tile_size=args.tile_size,
        gap=args.gap,
        nproc=args.nproc,
    )
    config_paths = [config_dir / "trainval.json", config_dir / "test.json"]
    commands = MMRotateAdapter().split_dota_commands(config_paths, backend_root=args.backend_root)
    print(f"split config dir: {config_dir}")
    for command in commands:
        print_command(command)
    if args.dry_run:
        print("dry-run: split configs were not written and images were not split.")
        return 0
    written = write_mmrotate_split_configs(configs, config_dir)
    print("wrote split configs: " + ", ".join(str(path) for path in written.values()))
    for command in commands:
        exit_code = run_command(command)
        if exit_code != 0:
            return exit_code
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    experiment = ExperimentConfig.from_file(args.experiment)
    adapter = get_adapter(experiment.adapter)
    command = adapter.train_command(experiment, backend_root=args.backend_root)
    print_command(command)
    if args.dry_run:
        print("dry-run: training was not started.")
        return 0
    return run_command(command)


def cmd_eval(args: argparse.Namespace) -> int:
    experiment = ExperimentConfig.from_file(args.experiment)
    adapter = get_adapter(experiment.adapter)
    command = adapter.eval_command(
        experiment,
        checkpoint=args.checkpoint,
        backend_root=args.backend_root,
    )
    print_command(command)
    if args.dry_run:
        print("dry-run: evaluation was not started.")
        return 0
    return run_command(command)


def cmd_report(args: argparse.Namespace) -> int:
    markdown = render_markdown(collect_runs(args.outputs))
    if args.dry_run:
        print(markdown)
        return 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {output}")
    return 0


def cmd_ensemble(args: argparse.Namespace) -> int:
    from rotary_target.ensemble import FusionConfig, OrientedBoxFusion, load_predictions, save_predictions
    from rotary_target.ensemble.io import Prediction

    if args.dry_run:
        print(f"ensemble: would fuse {len(args.predictions)} prediction files")
        for p in args.predictions:
            print(f"  - {p}")
        print(f"iou_threshold: {args.iou_threshold}")
        print(f"score_threshold: {args.score_threshold}")
        print(f"output: {args.output}")
        print("dry-run: no fusion performed.")
        return 0

    model_preds: list[list[Prediction]] = []
    for pred_path in args.predictions:
        path = Path(pred_path)
        if not path.exists():
            print(f"error: prediction file not found: {path}", file=sys.stderr)
            return 2
        model_preds.append(load_predictions(path))

    config = FusionConfig(
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
    )
    fusion = OrientedBoxFusion(config)
    fused = fusion.fuse(model_preds, model_weights=args.weights)

    from rotary_target.ensemble.io import Prediction as Pred
    out_preds = [
        Pred(cx=f.cx, cy=f.cy, w=f.w, h=f.h, theta=f.theta, score=f.score, class_id=f.class_id)
        for f in fused
    ]
    out_path = save_predictions(out_preds, args.output)
    total_input = sum(len(p) for p in model_preds)
    print(f"Fused {total_input} predictions from {len(model_preds)} models -> {len(fused)} boxes")
    print(f"Wrote {out_path}")
    return 0


def print_command(command: CommandSpec) -> None:
    print(f"# {command.description}")
    if command.cwd:
        print(f"cwd: {command.cwd}")
    print(command.shell_line())


def run_command(command: CommandSpec) -> int:
    if any("<MMROTATE_ROOT>" in part for part in command.argv):
        print("error: set --backend-root or MMROTATE_ROOT before running a non-dry-run command", file=sys.stderr)
        return 2
    completed = subprocess.run(command.argv, cwd=command.cwd if command.cwd else None, check=False)
    return int(completed.returncode)
