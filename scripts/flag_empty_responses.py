#!/usr/bin/env python3
"""Scan response caches and report runs with empty responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flag responses.json files that contain empty responses.")
    parser.add_argument("--data-dir", type=Path, default=Path("out/rand_chinese"), help="Directory to scan recursively.")
    parser.add_argument("--output-file", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--max-show", type=int, default=5, help="Maximum empty indices to display per run.")
    parser.add_argument("--show-all", action="store_true", help="Print every run, including clean ones.")
    return parser.parse_args()


def inspect_response_file(path: Path, max_show: int) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    empty_indices: list[int] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            empty_indices.append(idx)
            continue
        response = str(item.get("response", "")).strip()
        if not response:
            empty_indices.append(idx)

    return {
        "path": str(path),
        "run_name": path.parent.name,
        "item_count": len(items),
        "empty_count": len(empty_indices),
        "empty_indices": empty_indices[:max_show],
        "truncated": len(empty_indices) > max_show,
    }


def main() -> int:
    args = parse_args()
    records = []

    for response_path in sorted(args.data_dir.rglob("responses.json")):
        try:
            record = inspect_response_file(response_path, args.max_show)
        except Exception as exc:
            record = {
                "path": str(response_path),
                "run_name": response_path.parent.name,
                "error": str(exc),
            }
            records.append(record)
            continue

        if record["empty_count"] > 0 or args.show_all:
            records.append(record)

    for record in records:
        if "error" in record:
            print(f"ERROR {record['path']}: {record['error']}")
            continue
        indices = ", ".join(str(idx) for idx in record["empty_indices"])
        suffix = " ..." if record["truncated"] else ""
        print(
            f"{record['empty_count']:4d} empty / {record['item_count']:4d} items  "
            f"{record['run_name']}: [{indices}{suffix}]"
        )

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nmatched {len(records)} problematic run(s)" if records else "\nno empty responses found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
