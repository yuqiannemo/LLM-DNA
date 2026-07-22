#!/usr/bin/env python3
"""Filter already-seen models (present in configs/rand) from model list JSONL files."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAND_DIR = ROOT / "configs" / "rand"
HF_INPUT = ROOT / "configs" / "huggingface_llm_list.jsonl"
OR_INPUT = ROOT / "configs" / "openrouter_llm_list.jsonl"
HF_OUTPUT = ROOT / "configs" / "huggingface_llm_list_left.jsonl"
OR_OUTPUT = ROOT / "configs" / "openrouter_llm_list_left.jsonl"


def safe_model_name(model_id: str) -> str:
    """Mirror scripts/run_hf_dna_pipeline.py naming for model output folders."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.strip("/"))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                print(f"[WARN] Skip invalid JSON at {path}:{idx}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_seen_names(rand_dir: Path) -> set[str]:
    if not rand_dir.exists():
        return set()
    return {p.name for p in rand_dir.iterdir() if p.is_dir()}


def filter_rows(rows: list[dict], seen_safe_names: set[str]) -> tuple[list[dict], list[str]]:
    kept: list[dict] = []
    dropped_model_ids: list[str] = []
    for row in rows:
        model_id = str(row.get("model_id", "")).strip()
        if not model_id:
            kept.append(row)
            continue

        model_safe_name = safe_model_name(model_id)
        if model_safe_name in seen_safe_names:
            dropped_model_ids.append(model_id)
        else:
            kept.append(row)
    return kept, dropped_model_ids


def main() -> int:
    seen_safe_names = collect_seen_names(RAND_DIR)
    if not seen_safe_names:
        print(f"[WARN] No directories found under: {RAND_DIR}")

    hf_rows = read_jsonl(HF_INPUT)
    or_rows = read_jsonl(OR_INPUT)

    hf_kept, hf_dropped = filter_rows(hf_rows, seen_safe_names)
    or_kept, or_dropped = filter_rows(or_rows, seen_safe_names)

    write_jsonl(HF_OUTPUT, hf_kept)
    write_jsonl(OR_OUTPUT, or_kept)

    print(f"[DONE] seen dirs in rand: {len(seen_safe_names)}")
    print(f"[DONE] huggingface: {len(hf_rows)} -> {len(hf_kept)} (dropped {len(hf_dropped)})")
    print(f"[DONE] openrouter: {len(or_rows)} -> {len(or_kept)} (dropped {len(or_dropped)})")
    print(f"[OUT] {HF_OUTPUT}")
    print(f"[OUT] {OR_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
