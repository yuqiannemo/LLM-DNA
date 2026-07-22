#!/usr/bin/env python3
"""Remove model entries from a JSONL list using latest statuses from a status journal."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune model list JSONL by status.jsonl latest records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--status-jsonl", type=Path, required=True, help="Path to status.jsonl")
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Model list JSONL to prune")
    parser.add_argument("--provider", type=str, default=None, help="Only use statuses from this provider")
    parser.add_argument(
        "--status",
        type=str,
        default="success",
        help="Status value to remove from input JSONL (e.g., success, failed_oom)",
    )
    parser.add_argument("--backup", action="store_true", help="Create timestamped backup before writing")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing file")
    return parser.parse_args()


def load_latest_status_map(status_path: Path, provider: str | None) -> dict[str, str]:
    latest: dict[str, dict] = {}
    with status_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            model_id = str(row.get("model_id", "")).strip()
            row_provider = str(row.get("provider", "")).strip()
            if not model_id:
                continue
            if provider and row_provider != provider:
                continue
            latest[model_id] = row

    status_map: dict[str, str] = {}
    for model_id, row in latest.items():
        status_map[model_id] = str(row.get("status", "")).strip()
    return status_map


def load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{idx}: {exc}") from exc
    return rows


def write_jsonl_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()

    if not args.status_jsonl.exists():
        raise FileNotFoundError(f"status file not found: {args.status_jsonl}")
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"input file not found: {args.input_jsonl}")

    status_map = load_latest_status_map(args.status_jsonl, args.provider)
    rows = load_jsonl_rows(args.input_jsonl)

    kept: list[dict] = []
    removed: list[str] = []
    target_status = args.status.strip()

    for row in rows:
        model_id = str(row.get("model_id", "")).strip()
        if not model_id:
            kept.append(row)
            continue
        if status_map.get(model_id) == target_status:
            removed.append(model_id)
            continue
        kept.append(row)

    print(f"[INFO] status file: {args.status_jsonl}")
    print(f"[INFO] input file:  {args.input_jsonl}")
    print(f"[INFO] target status: {target_status}")
    print(f"[INFO] provider filter: {args.provider or 'ALL'}")
    print(f"[INFO] rows: {len(rows)} -> {len(kept)} (removed {len(removed)})")

    if removed:
        print("[INFO] removed model_id entries:")
        for model_id in removed:
            print(f"  - {model_id}")

    if args.dry_run:
        print("[DONE] dry-run only, file unchanged")
        return 0

    if args.backup:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = args.input_jsonl.with_name(f"{args.input_jsonl.name}.bak.{stamp}.status-{target_status}")
        backup.write_text(args.input_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[INFO] backup: {backup}")

    write_jsonl_rows(args.input_jsonl, kept)
    print("[DONE] file updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
