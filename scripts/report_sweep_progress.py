#!/usr/bin/env python3
"""Report completion progress for a DNA sweep run.

The script reads a run's ``manifest.json`` and ``status.jsonl`` and reports:

- planned task count
- number of unique models with a final recorded status
- completion percentage
- any models still missing from the status journal

It accepts either an explicit run directory or a temperature/top-p/repeat
specification and maps it to the expected run directory name.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_RUN_ROOT = Path("out") / "rand_chinese_temp_top_p_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report completion progress for a DNA sweep run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--run-dir", type=Path, help="Exact run directory containing manifest.json and status.jsonl")
    target.add_argument("--suffix", type=str, help="Run suffix such as _t00_p08_r1")
    parser.add_argument("--temperature", type=float, help="Temperature value used to build the suffix")
    parser.add_argument("--top-p", dest="top_p", type=float, help="Top-p value used to build the suffix")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat index used to build the suffix")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Root directory that contains per-setting run directories",
    )
    args = parser.parse_args()
    if args.run_dir is None and args.suffix is None and (args.temperature is None or args.top_p is None):
        parser.error("provide --run-dir, --suffix, or both --temperature and --top-p")
    return args


def format_code(value: float) -> str:
    return f"{int(round(value * 10)):02d}"


def build_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir
    if args.suffix is not None:
        suffix = args.suffix.strip()
        if not suffix.startswith("_"):
            suffix = f"_{suffix}"
        return args.run_root.with_name(f"{args.run_root.name}{suffix}")
    if args.temperature is None or args.top_p is None:
        raise ValueError("Provide either --run-dir, --suffix, or both --temperature and --top-p.")
    suffix = f"_t{format_code(args.temperature)}_p{format_code(args.top_p)}_r{args.repeat}"
    return args.run_root.with_name(f"{args.run_root.name}{suffix}")


def load_latest_statuses(status_path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    with status_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            model_id = str(row.get("model_id", "")).strip()
            if model_id:
                latest[model_id] = row
    return latest


def main() -> int:
    args = parse_args()
    run_dir = build_run_dir(args)
    manifest_path = run_dir / "manifest.json"
    status_path = run_dir / "status.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not status_path.exists():
        raise FileNotFoundError(f"status journal not found: {status_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    latest = load_latest_statuses(status_path)
    planned = int(manifest.get("task_count", 0))
    task_models = [str(task.get("model_id", "")).strip() for task in manifest.get("tasks", []) if str(task.get("model_id", "")).strip()]

    finished = sum(
        1
        for row in latest.values()
        if str(row.get("status", "")).startswith("failed") or str(row.get("status", "")) == "success"
    )
    success = sum(1 for row in latest.values() if str(row.get("status", "")) == "success")
    failed = finished - success
    failure_reasons = Counter(
        str(row.get("status", "unknown"))
        for row in latest.values()
        if str(row.get("status", "")).startswith("failed")
    )
    missing = [model_id for model_id in task_models if model_id not in latest]
    percent = (finished / planned * 100.0) if planned else 0.0

    print(f"run_dir={run_dir}")
    print(f"planned={planned}")
    print(f"unique_models_recorded={len(latest)}")
    print(f"finished={finished}")
    print(f"success={success}")
    print(f"failed={failed}")
    print(f"missing={len(missing)}")
    print(f"percent={percent:.1f}%")
    print("failure_status_counts=" + json.dumps(dict(sorted(failure_reasons.items())), ensure_ascii=False))
    if missing:
        print("missing_models=" + json.dumps(missing, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
