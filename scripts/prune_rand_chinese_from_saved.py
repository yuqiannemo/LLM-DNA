#!/usr/bin/env python3
"""Remove already-saved rand_chinese model outputs that also exist in out-saved/rand."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "out-saved" / "rand"
DEFAULT_TARGET_DIR = ROOT / "out" / "rand_chinese"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove target model output directories when the same model exists in out-saved/rand.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="Reference output tree")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR, help="Directory to prune")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview deletions without changing anything")
    mode.add_argument("--apply", action="store_true", help="Actually delete the matching directories")
    return parser.parse_args()


def collect_model_dirs(base_dir: Path) -> set[str]:
    if not base_dir.exists():
        return set()
    return {entry.name for entry in base_dir.iterdir() if entry.is_dir()}


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir
    target_dir = args.target_dir

    if not source_dir.exists():
        raise FileNotFoundError(f"source directory not found: {source_dir}")
    if not target_dir.exists():
        print(f"[WARN] target directory not found, nothing to do: {target_dir}")
        return 0

    source_models = collect_model_dirs(source_dir)
    target_models = collect_model_dirs(target_dir)
    matched_models = sorted(source_models & target_models)
    missing_in_target = sorted(source_models - target_models)

    print(f"[INFO] source: {source_dir}")
    print(f"[INFO] target: {target_dir}")
    print(f"[INFO] source model dirs: {len(source_models)}")
    print(f"[INFO] target model dirs: {len(target_models)}")
    print(f"[INFO] matched model dirs: {len(matched_models)}")

    if matched_models:
        action = "would remove" if args.dry_run or not args.apply else "removing"
        print(f"[INFO] {action} matched target directories:")
        for model_name in matched_models:
            print(f"  - {target_dir / model_name}")

    if missing_in_target:
        print(f"[INFO] source models not present under target: {len(missing_in_target)}")

    if args.dry_run or not args.apply:
        print("[DONE] dry-run only, no files changed")
        return 0

    removed = 0
    for model_name in matched_models:
        path = target_dir / model_name
        shutil.rmtree(path)
        removed += 1

    print(f"[DONE] removed {removed} directory(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())