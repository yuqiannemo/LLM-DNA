#!/usr/bin/env python3
"""Build the missing-model queue for one explicit DistDNA pilot seed.

An artifact is reusable only when its run manifest explicitly records the
requested generation seed and decoding setting, its latest per-model status is
successful, and both the DNA and summary files still exist. Historical runs
whose seed is absent from the manifest are deliberately ignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolved_artifact(path_value: object) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def explicit_successes(
    output_root: Path,
    model_ids: set[str],
    seed: int,
    temperature: float,
    top_p: float,
) -> dict[str, dict[str, str]]:
    successes: dict[str, dict[str, str]] = {}
    for manifest_path in output_root.glob("**/manifest.json"):
        try:
            manifest = read_json(manifest_path)
            if int(manifest.get("generation_seed")) != seed:
                continue
            if float(manifest.get("temperature")) != temperature:
                continue
            if float(manifest.get("top_p")) != top_p:
                continue
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue

        status_path = manifest_path.with_name("status.jsonl")
        if not status_path.is_file():
            continue

        latest: dict[str, dict[str, Any]] = {}
        try:
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                model_id = record.get("model_id")
                if model_id in model_ids:
                    latest[model_id] = record
        except (OSError, json.JSONDecodeError):
            continue

        for model_id, record in latest.items():
            if record.get("status") != "success":
                continue
            dna_path = resolved_artifact(record.get("output_path"))
            summary_path = resolved_artifact(record.get("summary_path"))
            if not dna_path or not summary_path:
                continue
            if not dna_path.is_file() or dna_path.stat().st_size == 0:
                continue
            if not summary_path.is_file() or summary_path.stat().st_size == 0:
                continue
            successes[model_id] = {
                "manifest": str(manifest_path.relative_to(ROOT)),
                "status": str(status_path.relative_to(ROOT)),
                "dna": str(dna_path.relative_to(ROOT)),
                "summary": str(summary_path.relative_to(ROOT)),
            }
    return successes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--search-root", type=Path, default=ROOT / "out")
    parser.add_argument("--missing-jsonl", type=Path, required=True)
    parser.add_argument("--inventory-json", type=Path, required=True)
    args = parser.parse_args()

    cohort_path = args.cohort if args.cohort.is_absolute() else ROOT / args.cohort
    rows = [json.loads(line) for line in cohort_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    model_ids = {str(row["model_id"]) for row in rows}
    if len(rows) != 10 or len(model_ids) != 10:
        raise SystemExit(f"pilot cohort must contain exactly 10 unique models: {cohort_path}")

    search_root = args.search_root if args.search_root.is_absolute() else ROOT / args.search_root
    successes = explicit_successes(search_root, model_ids, args.seed, args.temperature, args.top_p)
    missing = [row for row in rows if row["model_id"] not in successes]

    args.missing_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.inventory_json.parent.mkdir(parents=True, exist_ok=True)
    args.missing_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in missing),
        encoding="utf-8",
    )
    args.inventory_json.write_text(
        json.dumps(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "generation_seed": args.seed,
                "cohort_size": len(rows),
                "reusable_explicit_success_count": len(successes),
                "missing_count": len(missing),
                "reusable_explicit_successes": successes,
                "missing_models": [row["model_id"] for row in missing],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"seed={args.seed} reusable={len(successes)}/10 "
        f"missing={len(missing)}/10 queue={args.missing_jsonl}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
