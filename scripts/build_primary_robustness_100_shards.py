#!/usr/bin/env python3
"""Split the fixed 100-model primary cohort across RTX 3090 and A100 workers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model_ids = [str(row.get("model_id", "")).strip() for row in rows]
    if len(rows) != 100:
        raise ValueError(f"Expected exactly 100 cohort rows, found {len(rows)}.")
    if any(not model_id for model_id in model_ids):
        raise ValueError("Every cohort row must contain a non-empty model_id.")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("The cohort contains duplicate model IDs.")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign the largest primary models to A100s and the remainder to "
            "RTX 3090s."
        )
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("configs/rand_chinese_stratified_100.jsonl"),
    )
    parser.add_argument("--a100-models", type=int, default=23)
    parser.add_argument(
        "--rtx3090-output",
        type=Path,
        default=Path("configs/rand_chinese_stratified_100_rtx3090.jsonl"),
    )
    parser.add_argument(
        "--a100-output",
        type=Path,
        default=Path("configs/rand_chinese_stratified_100_a100.jsonl"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("configs/rand_chinese_stratified_100_shards_audit.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.cohort)
    if not 0 < args.a100_models < len(rows):
        raise ValueError("--a100-models must be between 1 and 99.")

    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row.get("parameter_count_billions", 0.0) or 0.0),
            str(row["model_id"]),
        ),
    )
    a100_ids = {str(row["model_id"]) for row in ranked[: args.a100_models]}
    a100_rows = [row for row in rows if str(row["model_id"]) in a100_ids]
    rtx3090_rows = [row for row in rows if str(row["model_id"]) not in a100_ids]

    rtx3090_sha = write_jsonl(args.rtx3090_output, rtx3090_rows)
    a100_sha = write_jsonl(args.a100_output, a100_rows)
    cohort_sha = hashlib.sha256(args.cohort.read_bytes()).hexdigest()
    audit = {
        "cohort": str(args.cohort),
        "cohort_sha256": cohort_sha,
        "assignment_rule": (
            f"the {args.a100_models} largest models by recorded "
            "parameter_count_billions go to the two A100 GPUs; all remaining "
            "models go to the four-GPU RTX 3090 fleet"
        ),
        "rtx3090": {
            "path": str(args.rtx3090_output),
            "count": len(rtx3090_rows),
            "sha256": rtx3090_sha,
            "models": [row["model_id"] for row in rtx3090_rows],
        },
        "a100": {
            "path": str(args.a100_output),
            "count": len(a100_rows),
            "sha256": a100_sha,
            "models": [row["model_id"] for row in a100_rows],
        },
    }
    args.audit_output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
