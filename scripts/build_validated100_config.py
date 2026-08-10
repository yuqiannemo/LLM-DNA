#!/usr/bin/env python3
"""Reconstruct the latest validated-100 JSONL from its canonical manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [
        {
            "model_id": str(task["model_id"]),
            "provider": str(task.get("provider", "huggingface")),
        }
        for task in manifest.get("tasks", [])
        if task.get("model_id")
    ]
    model_ids = [row["model_id"] for row in rows]
    if len(rows) != 100 or len(set(model_ids)) != 100:
        raise ValueError(
            f"Expected 100 unique manifest models, found "
            f"{len(rows)} rows / {len(set(model_ids))} unique IDs"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"Wrote {len(rows)} models to {args.output}")


if __name__ == "__main__":
    main()
