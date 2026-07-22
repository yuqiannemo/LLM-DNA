#!/usr/bin/env python3
"""Generate sliding 50-response subset DNAs from cached responses."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_dna.api import (
    DNAExtractionConfig,
    _default_model_metadata,
    _extract_signature_from_text_responses,
    _save_signature_outputs,
)

DATA_DIR = ROOT / "dna-out-chat" / "rand_chinese"
OUTPUT_ROOT = ROOT / "dna-out-chat" / "rand_chinese_subsets"
ENCODER_NAME = "all-mpnet-base-v2"
WINDOW_SIZE = 50
WINDOW_STEP = 5


@dataclass(frozen=True)
class ValidModel:
    key: str
    model_name: str
    responses: list[str]
    payload: dict


def load_valid_models(data_dir: Path) -> list[ValidModel]:
    """Load complete artifacts with at least one non-empty cached response."""
    records: list[ValidModel] = []
    if not data_dir.exists():
        return records
    for model_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        response_path = model_dir / "responses.json"
        summary_files = list(model_dir.glob("*_summary.json"))
        dna_files = list(model_dir.glob("*_dna.json"))
        if not response_path.exists() or not summary_files or not dna_files:
            continue
        try:
            payload = json.loads(response_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        responses = [
            str(item.get("response", "")).strip()
            for item in payload.get("items", [])
            if isinstance(item, dict) and str(item.get("response", "")).strip()
        ]
        if responses:
            records.append(
                ValidModel(
                    key=model_dir.name,
                    model_name=str(payload.get("model") or model_dir.name),
                    responses=responses,
                    payload=payload,
                )
            )
    return records


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def default_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_ROOT / f"run_{stamp}"


def log(message: str, log_file: Path) -> None:
    line = f"[{timestamp()}] {message}"
    print(line, flush=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_models(limit: int | None) -> list[tuple[str, dict]]:
    models: list[tuple[str, dict]] = []
    for record in load_valid_models(DATA_DIR):
        payload = dict(record.payload)
        items = [item for item in payload.get("items", []) if isinstance(item, dict) and str(item.get("response", "")).strip()]
        if len(items) < WINDOW_SIZE:
            continue
        payload["items"] = items
        models.append((record.key, payload))
        if limit is not None and len(models) >= limit:
            break
    return models


def build_windows(item_count: int) -> list[tuple[str, int, int]]:
    windows = []
    index = 1
    for start in range(0, item_count - WINDOW_SIZE + 1, WINDOW_STEP):
        end = start + WINDOW_SIZE
        windows.append((f"50_{index:03d}", start, end))
        index += 1
    return windows


def subset_payload(base_payload: dict, subset_name: str, start: int, end: int) -> dict:
    subset_items = base_payload["items"][start:end]
    return {
        "model": base_payload.get("model", ""),
        "dataset": base_payload.get("dataset", "rand_chinese"),
        "count": len(subset_items),
        "subset": subset_name,
        "subset_start": start,
        "subset_end": end,
        "items": subset_items,
    }


def save_subset_responses(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_config(model_name: str, output_dir: Path, output_path: Path, sample_count: int) -> DNAExtractionConfig:
    return DNAExtractionConfig(
        model_name=model_name,
        model_type="openrouter",
        dataset="rand_chinese",
        probe_set="rand",
        max_samples=sample_count,
        extractor_type="embedding",
        dna_dim=128,
        reduction_method="random_projection",
        embedding_merge="concat",
        output_dir=output_dir,
        output_path=output_path,
        save=True,
        device="cpu",
        random_seed=42,
        max_length=1024,
    )


def main() -> int:
    limit = None if len(sys.argv) <= 1 or sys.argv[1] == "all" else int(sys.argv[1])
    run_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.log"
    distance_file = run_dir / "distance_summary.json"
    log_file.write_text("", encoding="utf-8")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    logging.getLogger().setLevel(logging.WARNING)

    models = load_models(limit)
    if not models:
        raise SystemExit(f"no models with at least {WINDOW_SIZE} non-empty responses found in {DATA_DIR}")

    log(f"loaded {len(models)} model(s)", log_file)
    all_distances: dict[str, dict[str, object]] = {}

    for index, (model_key, payload) in enumerate(models, 1):
        log(f"processing {index}/{len(models)}: {model_key}", log_file)
        signatures = {}
        subset_dirs = {}
        windows = build_windows(len(payload["items"]))
        log(f"  windows={len(windows)}", log_file)

        for subset_name, start, end in windows:
            tagged_key = f"{model_key}__subset_{subset_name}"
            subset_dir = run_dir / tagged_key
            subset_dirs[subset_name] = subset_dir
            subset_data = subset_payload(payload, subset_name, start, end)
            save_subset_responses(subset_dir / "responses.json", subset_data)

            output_path = subset_dir / f"{tagged_key}_dna.json"
            config = build_config(
                model_name=payload.get("model", model_key),
                output_dir=run_dir,
                output_path=output_path,
                sample_count=len(subset_data["items"]),
            )
            signature, _, elapsed = _extract_signature_from_text_responses(
                model_name=payload.get("model", model_key),
                responses=[item["response"] for item in subset_data["items"]],
                config=config,
                model_meta=_default_model_metadata(payload.get("model", model_key)),
                generation_device="cpu",
                sentence_encoder=ENCODER_NAME,
                encoder_device="cpu",
            )
            _save_signature_outputs(
                signature=signature,
                config=config,
                output_path=output_path,
                summary_path=output_path.with_name(f"{output_path.stem}_summary.json"),
                elapsed_seconds=elapsed,
            )
            signatures[subset_name] = signature

        pairwise = {}
        subset_names = list(signatures)
        cosine_values = []
        euclidean_values = []
        for i in range(len(subset_names)):
            for j in range(i + 1, len(subset_names)):
                a = subset_names[i]
                b = subset_names[j]
                cosine = signatures[a].distance_to(signatures[b], metric="cosine")
                euclidean = signatures[a].distance_to(signatures[b], metric="euclidean")
                pairwise[f"{a}__{b}"] = {"cosine": cosine, "euclidean": euclidean}
                cosine_values.append(cosine)
                euclidean_values.append(euclidean)

        all_distances[model_key] = {
            "subset_dirs": {name: str(path) for name, path in subset_dirs.items()},
            "windows": [
                {"name": subset_name, "start": start, "end": end}
                for subset_name, start, end in windows
            ],
            "pairwise": pairwise,
            "summary": {
                "window_size": WINDOW_SIZE,
                "window_step": WINDOW_STEP,
                "window_count": len(windows),
                "mean_cosine": sum(cosine_values) / len(cosine_values) if cosine_values else 0.0,
                "max_cosine": max(cosine_values) if cosine_values else 0.0,
                "mean_euclidean": sum(euclidean_values) / len(euclidean_values) if euclidean_values else 0.0,
                "max_euclidean": max(euclidean_values) if euclidean_values else 0.0,
            },
        }
        log(
            f"finished {model_key}: "
            f"window_count={len(windows)}, "
            f"mean_cos={all_distances[model_key]['summary']['mean_cosine']:.4f}, "
            f"max_cos={all_distances[model_key]['summary']['max_cosine']:.4f}",
            log_file,
        )

    distance_file.write_text(json.dumps(all_distances, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved distance summary to {distance_file}", log_file)
    log(f"saved run log to {log_file}", log_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
