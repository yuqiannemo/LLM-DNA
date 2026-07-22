#!/usr/bin/env python3
"""Build a larger, reproducible model cohort from existing reference artifacts.

Selection never uses evaluation accuracy.  Eligible models must have a valid
unsuffixed reference DNA and recorded parameter count, then deterministic
round-robin sampling balances architecture and logarithmic size strata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_RELIABILITY_REPLACEMENTS = {
    "tiiuae/Falcon3-3B-Instruct": "princeton-nlp/Sheared-LLaMA-2.7B",
    "sarvamai/sarvam-30b": "tiiuae/Falcon3-10B-Base",
    "allenai/Olmo-3-7B-Instruct-DPO": "allenai/Llama-3.1-Tulu-3-8B",
}


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip("/"))


def load_catalog(path: Path) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        model = str(row.get("model_id", "")).strip()
        if model:
            output[model] = row
    return output


def load_eligible_references(
    dataset_dir: Path, minimum_billions: float, maximum_billions: float
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    eligible: dict[str, dict[str, object]] = {}
    excluded: dict[str, str] = {}
    for path in sorted(dataset_dir.glob("*/*_dna.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata", {})
            model = str(metadata.get("model_name", "")).strip()
            vector = np.asarray(payload.get("signature", []), dtype=np.float32)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if not model or path.parent.name != safe_model_name(model):
            continue
        size = metadata.get("model_metadata", {}).get("size", {}).get("parameter_count_billions")
        if not isinstance(size, (int, float)):
            excluded[model] = "missing_parameter_count"
            continue
        if not minimum_billions <= float(size) <= maximum_billions:
            excluded[model] = "outside_size_range"
            continue
        if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
            excluded[model] = "invalid_reference_dna"
            continue
        architecture = str(
            metadata.get("model_metadata", {}).get("architecture", {}).get("model_type") or "unknown"
        )
        eligible[model] = {
            "parameter_count_billions": float(size),
            "architecture": architecture,
            "reference_path": str(path),
        }
    return eligible, excluded


def size_bucket(size_billions: float) -> str:
    lower = 2 ** math.floor(math.log2(max(size_billions, 0.125)))
    upper = lower * 2
    return f"{lower:g}-{upper:g}B"


def select_stratified(
    eligible: dict[str, dict[str, object]], target: int, seed: int
) -> list[str]:
    if target > len(eligible):
        raise ValueError(f"Requested {target} models but only {len(eligible)} are eligible.")
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for model, metadata in eligible.items():
        key = (str(metadata["architecture"]), size_bucket(float(metadata["parameter_count_billions"])))
        strata[key].append(model)
    rng = np.random.default_rng(seed)
    for values in strata.values():
        rng.shuffle(values)
    ordered_keys = sorted(strata, key=lambda key: (key[0], key[1]))
    selected: list[str] = []
    while len(selected) < target:
        made_progress = False
        for key in ordered_keys:
            if strata[key] and len(selected) < target:
                selected.append(strata[key].pop())
                made_progress = True
        if not made_progress:
            break
    return selected


def apply_reliability_replacements(
    selected: list[str],
    eligible: dict[str, dict[str, object]],
    replacements: dict[str, str],
) -> tuple[list[str], list[dict[str, object]]]:
    """Replace known generation failures without using evaluation accuracy."""
    output = list(selected)
    audit_rows: list[dict[str, object]] = []
    for removed, replacement in replacements.items():
        if removed not in output:
            continue
        if replacement not in eligible:
            raise ValueError(f"Reliability replacement is not eligible: {replacement}")
        if replacement in output:
            raise ValueError(f"Reliability replacement is already selected: {replacement}")
        index = output.index(removed)
        output[index] = replacement
        audit_rows.append(
            {
                "removed_model": removed,
                "removed_size_billions": eligible[removed]["parameter_count_billions"],
                "replacement_model": replacement,
                "replacement_size_billions": eligible[replacement]["parameter_count_billions"],
                "reason": "repeated_failed_empty_response",
                "validation": "cached baseline contains 100 non-empty responses",
            }
        )
    return output, audit_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stratified DNA experiment cohort.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("out/rand_chinese"))
    parser.add_argument("--catalog", type=Path, default=Path("configs/huggingface_llm_list.jsonl"))
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--minimum-billions", type=float, default=0.3)
    parser.add_argument("--maximum-billions", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("configs/rand_chinese_stratified_300.jsonl"))
    parser.add_argument("--audit-output", type=Path, default=Path("configs/rand_chinese_stratified_300_audit.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_catalog(args.catalog)
    eligible, excluded = load_eligible_references(
        args.dataset_dir, args.minimum_billions, args.maximum_billions
    )
    try:
        selected = select_stratified(eligible, args.target, args.seed)
        selected, reliability_replacements = apply_reliability_replacements(
            selected, eligible, DEFAULT_RELIABILITY_REPLACEMENTS
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for model in selected:
            source = catalog.get(model, {})
            row = {
                "model_id": model,
                "provider": source.get("provider", "huggingface"),
                "task": source.get("task", "text-generation"),
                "downloads": int(source.get("downloads", 0) or 0),
                "parameter_count_billions": eligible[model]["parameter_count_billions"],
                "architecture": eligible[model]["architecture"],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    audit = {
        "selection_uses_performance": False,
        "selection_method": "seeded round-robin over architecture x logarithmic parameter-size strata, followed by documented reliability replacements that do not use evaluation accuracy",
        "seed": args.seed,
        "target": args.target,
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "minimum_billions": args.minimum_billions,
        "maximum_billions": args.maximum_billions,
        "output_sha256": digest,
        "post_selection_replacements": reliability_replacements,
        "selected_models": selected,
        "selected_metadata": {model: eligible[model] for model in selected},
        "exclusion_reason_counts": {
            reason: sum(value == reason for value in excluded.values()) for reason in sorted(set(excluded.values()))
        },
    }
    args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "audit": str(args.audit_output), **{key: audit[key] for key in ["eligible_count", "selected_count", "output_sha256"]}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
