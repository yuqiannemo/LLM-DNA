#!/usr/bin/env python3
"""Validate a failed-slot plan and build one Hugging Face list per grid cell."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALLOWED_FAILURES = {"failed_empty_response", "failed_missing_file"}


def cell_id(row: dict[str, Any]) -> str:
    temperature = int(round(float(row["temperature"]) * 10))
    top_p = int(round(float(row["top_p"]) * 10))
    return f"t{temperature:02d}_p{top_p:02d}_r{int(row['repeat'])}"


def load_plan(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            required = {
                "model_id",
                "provider",
                "temperature",
                "top_p",
                "repeat",
                "generation_seed",
                "prior_status",
            }
            missing = required - set(row)
            if missing:
                raise ValueError(
                    f"Missing fields on {path}:{line_number}: {sorted(missing)}"
                )
            if row["provider"] != "huggingface":
                raise ValueError(f"Unsupported provider on line {line_number}")
            if row["prior_status"] not in ALLOWED_FAILURES:
                raise ValueError(f"Unexpected prior status on line {line_number}")
            rows.append(row)
    return rows


def validate(rows: list[dict[str, Any]], expected_slots: int) -> None:
    if len(rows) != expected_slots:
        raise ValueError(f"Expected {expected_slots} failed slots, found {len(rows)}")

    keys = [
        (
            str(row["model_id"]),
            float(row["temperature"]),
            float(row["top_p"]),
            int(row["repeat"]),
        )
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        raise ValueError(f"Duplicate failed-slot keys: {duplicates}")

    for row in rows:
        temperature = float(row["temperature"])
        repeat = int(row["repeat"])
        expected_seed = 42 if temperature == 0.5 else 42000 + repeat
        if int(row["generation_seed"]) != expected_seed:
            raise ValueError(
                f"Wrong generation seed for {row['model_id']} {cell_id(row)}: "
                f"expected {expected_seed}, found {row['generation_seed']}"
            )


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-slots", type=int, default=72)
    args = parser.parse_args()

    rows = load_plan(args.plan)
    validate(rows, args.expected_slots)

    grouped: dict[tuple[float, float, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            float(row["temperature"]),
            float(row["top_p"]),
            int(row["repeat"]),
            int(row["generation_seed"]),
        )
        grouped[key].append(row)

    index_lines = ["cell_id\ttemperature\ttop_p\trepeat\tgeneration_seed\ttask_count\tconfig"]
    for key in sorted(grouped):
        temperature, top_p, repeat, generation_seed = key
        cell_rows = sorted(grouped[key], key=lambda row: str(row["model_id"]))
        identifier = cell_id(cell_rows[0])
        config_path = args.output_dir / f"{identifier}.jsonl"
        content = "".join(
            json.dumps(
                {"model_id": row["model_id"], "provider": "huggingface"},
                ensure_ascii=False,
            )
            + "\n"
            for row in cell_rows
        )
        atomic_write(config_path, content)
        index_lines.append(
            "\t".join(
                [
                    identifier,
                    str(temperature),
                    str(top_p),
                    str(repeat),
                    str(generation_seed),
                    str(len(cell_rows)),
                    str(config_path),
                ]
            )
        )

    atomic_write(args.output_dir / "cells.tsv", "\n".join(index_lines) + "\n")
    failures = Counter(str(row["prior_status"]) for row in rows)
    print(
        f"Prepared {len(rows)} failed slots across {len(grouped)} cells: "
        + ", ".join(f"{status}={count}" for status, count in sorted(failures.items()))
    )


if __name__ == "__main__":
    main()
