#!/usr/bin/env python3
"""Evaluate DistDNA/RFFTrace with matched-sample retrieval baselines.

This is the executable counterpart of ``docs/main (2).pdf``. It uses a real
sentence encoder, calibrates the RBF bandwidth on disjoint reference prompts,
shares one RFF map (and optional JL projection) across every model/setting, and
compares single cosine, matched-K mean cosine, exact RBF-MMD, and RFFTrace on
identical held-out prompts and response samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from llm_dna.distributional import (
    RandomFourierMap,
    compact_rfftrace_vector,
    gaussian_projection_matrix,
    median_bandwidth,
    prompt_averaged_mmd2,
    rfftrace_vector,
)


@dataclass(frozen=True, order=True)
class Setting:
    temperature: float
    top_p: float

    @property
    def code(self) -> str:
        return f"t{int(round(self.temperature * 10)):02d}_p{int(round(self.top_p * 10)):02d}"


@dataclass(frozen=True)
class ResponseRun:
    model_id: str
    setting: Setting
    repeat: int
    path: Path
    prompts: tuple[str, ...]
    responses: tuple[str, ...]


RUN_SUFFIX = re.compile(r"_t(?P<t>\d{2})_p(?P<p>\d{2})_r(?P<repeat>\d+)$")


def parse_setting(raw: str) -> Setting:
    try:
        temperature, top_p = raw.split(":", maxsplit=1)
        return Setting(float(temperature), float(top_p))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid setting {raw!r}; expected temperature:top_p"
        ) from exc


def parse_repeats(raw: str) -> tuple[int, ...]:
    try:
        repeats = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Repeats must be comma-separated integers") from exc
    if not repeats or len(set(repeats)) != len(repeats):
        raise argparse.ArgumentTypeError("At least one unique repeat is required")
    return repeats


def parse_run_path(path: Path) -> tuple[Setting, int] | None:
    match = RUN_SUFFIX.search(path.parent.name)
    if match is None:
        return None
    return (
        Setting(int(match.group("t")) / 10.0, int(match.group("p")) / 10.0),
        int(match.group("repeat")),
    )


def load_response_run(path: Path) -> ResponseRun:
    parsed = parse_run_path(path)
    if parsed is None:
        raise ValueError(f"Run directory has no decoding suffix: {path.parent}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    prompts = tuple(str(item.get("prompt", "")) for item in items)
    responses = tuple(str(item.get("response", "")) for item in items)
    if not prompts or len(prompts) != len(responses):
        raise ValueError(f"Invalid prompt/response items in {path}")
    if any(not prompt.strip() for prompt in prompts):
        raise ValueError(f"Empty prompt in {path}")
    if any(not response.strip() for response in responses):
        raise ValueError(f"Empty response in {path}")
    model_id = str(payload.get("model", "")).strip()
    if not model_id:
        # Old payloads may omit the model field; the suffix is unambiguous but
        # the sanitized directory name is then retained as the identity label.
        model_id = RUN_SUFFIX.sub("", path.parent.name)
    setting, repeat = parsed
    return ResponseRun(model_id, setting, repeat, path, prompts, responses)


def discover_runs(
    data_dirs: Iterable[Path],
    settings: set[Setting],
) -> dict[tuple[str, Setting, int], ResponseRun]:
    records: dict[tuple[str, Setting, int], ResponseRun] = {}
    mtimes: dict[tuple[str, Setting, int], float] = {}
    for data_dir in data_dirs:
        for path in sorted(data_dir.rglob("responses.json")):
            parsed = parse_run_path(path)
            if parsed is None or parsed[0] not in settings:
                continue
            try:
                record = load_response_run(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            key = (record.model_id, record.setting, record.repeat)
            modified = path.stat().st_mtime
            if key not in records or modified > mtimes[key]:
                records[key] = record
                mtimes[key] = modified
    return records


def select_models(
    records: dict[tuple[str, Setting, int], ResponseRun],
    reference_setting: Setting,
    query_settings: list[Setting],
    reference_repeats: tuple[int, ...],
    query_repeats: tuple[int, ...],
    samples_per_side: int,
    cohort_file: Path | None,
    model_limit: int,
) -> list[str]:
    required = [
        (reference_setting, reference_repeats[:samples_per_side]),
        *[(setting, query_repeats[:samples_per_side]) for setting in query_settings],
    ]
    available = {model for model, _setting, _repeat in records}
    models = [
        model
        for model in available
        if all(
            all((model, setting, repeat) in records for repeat in repeats)
            for setting, repeats in required
        )
    ]
    if cohort_file is not None:
        order = [
            str(json.loads(line)["model_id"])
            for line in cohort_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected = [model for model in order if model in set(models)]
    else:
        selected = sorted(models)
    return selected[:model_limit] if model_limit > 0 else selected


def common_prompts(runs: Iterable[ResponseRun]) -> list[str]:
    records = list(runs)
    if not records:
        return []
    common = set(records[0].prompts)
    for record in records[1:]:
        common.intersection_update(record.prompts)
    return [prompt for prompt in records[0].prompts if prompt in common]


def split_prompts(
    prompts: list[str], calibration_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    if len(prompts) < 2:
        raise ValueError("At least two common prompts are required")
    if not 0.0 < calibration_ratio < 1.0:
        raise ValueError("calibration_ratio must lie strictly between zero and one")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(prompts))
    count = min(len(prompts) - 1, max(1, round(len(prompts) * calibration_ratio)))
    calibration_indices = set(int(index) for index in order[:count])
    calibration = [prompt for index, prompt in enumerate(prompts) if index in calibration_indices]
    evaluation = [prompt for index, prompt in enumerate(prompts) if index not in calibration_indices]
    return calibration, evaluation


def _cache_key(record: ResponseRun, encoder_name: str, normalize: bool) -> str:
    digest = hashlib.sha256()
    digest.update(encoder_name.encode("utf-8"))
    digest.update(str(normalize).encode("ascii"))
    for prompt, response in zip(record.prompts, record.responses):
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\0")
        digest.update(response.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class SentenceEmbeddingStore:
    def __init__(
        self,
        encoder_name: str,
        device: str,
        batch_size: int,
        normalize: bool,
        cache_dir: Path,
    ) -> None:
        self.encoder_name = encoder_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._encoder = None

    def encode(self, record: ResponseRun) -> np.ndarray:
        cache_path = self.cache_dir / f"{_cache_key(record, self.encoder_name, self.normalize)}.npz"
        if cache_path.exists():
            with np.load(cache_path) as payload:
                return np.asarray(payload["embeddings"], dtype=np.float32)
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.encoder_name, device=self.device)
        matrix = self._encoder.encode(
            list(record.responses),
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
        )
        matrix = np.asarray(matrix, dtype=np.float32)
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, embeddings=matrix)
        temporary.replace(cache_path)
        return matrix


def prompt_samples(
    model_id: str,
    setting: Setting,
    repeats: tuple[int, ...],
    prompts: list[str],
    records: dict[tuple[str, Setting, int], ResponseRun],
    store: SentenceEmbeddingStore,
) -> np.ndarray:
    per_repeat: list[np.ndarray] = []
    for repeat in repeats:
        record = records[(model_id, setting, repeat)]
        embeddings = store.encode(record)
        prompt_to_index = {prompt: index for index, prompt in enumerate(record.prompts)}
        per_repeat.append(np.stack([embeddings[prompt_to_index[prompt]] for prompt in prompts]))
    # (repeats, prompts, features) -> (prompts, repeats, features)
    return np.stack(per_repeat, axis=0).transpose(1, 0, 2).astype(np.float32)


def cosine_distance_matrix(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    reference = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    return (1.0 - query @ reference.T).astype(np.float32)


def squared_distance_matrix(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(query * query, axis=1, keepdims=True)
        + np.sum(reference * reference, axis=1, keepdims=True).T
        - 2.0 * query @ reference.T
    )
    return np.maximum(squared, 0.0).astype(np.float32)


def ranking_rows(
    comparison: str,
    method: str,
    models: list[str],
    distances: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    ranks: list[int] = []
    predictions: list[dict[str, object]] = []
    for row, model_id in enumerate(models):
        order = np.argsort(distances[row], kind="stable")
        rank = int(np.flatnonzero(order == row)[0]) + 1
        ranks.append(rank)
        predicted = models[int(order[0])]
        predictions.append(
            {
                "comparison": comparison,
                "method": method,
                "model_id": model_id,
                "predicted_model": predicted,
                "rank": rank,
                "correct": int(rank == 1),
                "self_distance": float(distances[row, row]),
                "nearest_distance": float(distances[row, order[0]]),
            }
        )
    metrics: dict[str, object] = {
        "comparison": comparison,
        "method": method,
        "model_count": len(models),
        "top1": float(np.mean([rank <= 1 for rank in ranks])),
        "top3": float(np.mean([rank <= 3 for rank in ranks])),
        "top5": float(np.mean([rank <= 5 for rank in ranks])),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])),
        "mean_self_distance": float(np.mean(np.diag(distances))),
    }
    return metrics, predictions


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched-budget exact RBF-MMD and DistDNA/RFFTrace evaluation"
    )
    parser.add_argument("--data-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("out/rfftrace_experiment"))
    parser.add_argument("--reference-setting", type=parse_setting, default=parse_setting("0.2:0.8"))
    parser.add_argument("--query-setting", type=parse_setting, action="append", required=True)
    parser.add_argument("--reference-repeats", type=parse_repeats, default=parse_repeats("1"))
    parser.add_argument("--query-repeats", type=parse_repeats, default=parse_repeats("2"))
    parser.add_argument("--samples-per-side", type=int, default=1)
    parser.add_argument("--cohort-file", type=Path)
    parser.add_argument("--minimum-models", type=int, default=80)
    parser.add_argument("--model-limit", type=int, default=0)
    parser.add_argument("--prompt-limit", type=int, default=0)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--encoder", default="all-mpnet-base-v2")
    parser.add_argument("--encoder-device", default="cpu")
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--normalize-embeddings", action="store_true")
    parser.add_argument("--embedding-cache-dir", type=Path, default=Path("out/response_embedding_cache"))
    parser.add_argument("--rff-dimension", type=int, default=1024)
    parser.add_argument("--compact-dimension", type=int, default=128)
    parser.add_argument("--bandwidth-multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_side <= 0:
        raise SystemExit("--samples-per-side must be positive")
    if len(args.reference_repeats) < args.samples_per_side:
        raise SystemExit("Not enough --reference-repeats for --samples-per-side")
    if len(args.query_repeats) < args.samples_per_side:
        raise SystemExit("Not enough --query-repeats for --samples-per-side")
    if args.rff_dimension <= 0 or args.compact_dimension < 0:
        raise SystemExit("RFF dimension must be positive and compact dimension non-negative")
    if not np.isfinite(args.bandwidth_multiplier) or args.bandwidth_multiplier <= 0:
        raise SystemExit("--bandwidth-multiplier must be finite and positive")

    started = time.time()
    settings = {args.reference_setting, *args.query_setting}
    records = discover_runs(args.data_dir, settings)
    models = select_models(
        records,
        args.reference_setting,
        args.query_setting,
        args.reference_repeats,
        args.query_repeats,
        args.samples_per_side,
        args.cohort_file,
        args.model_limit,
    )
    if len(models) < args.minimum_models:
        raise SystemExit(
            f"Only {len(models)} models have the required runs; minimum is {args.minimum_models}"
        )

    selected_runs = []
    for model in models:
        selected_runs.extend(
            records[(model, args.reference_setting, repeat)]
            for repeat in args.reference_repeats[: args.samples_per_side]
        )
        for setting in args.query_setting:
            selected_runs.extend(
                records[(model, setting, repeat)]
                for repeat in args.query_repeats[: args.samples_per_side]
            )
    prompts = common_prompts(selected_runs)
    if args.prompt_limit > 0:
        prompts = prompts[: args.prompt_limit]
    calibration_prompts, evaluation_prompts = split_prompts(
        prompts, args.calibration_ratio, args.seed
    )

    store = SentenceEmbeddingStore(
        args.encoder,
        args.encoder_device,
        args.encoder_batch_size,
        args.normalize_embeddings,
        args.embedding_cache_dir,
    )
    reference_repeats = args.reference_repeats[: args.samples_per_side]
    query_repeats = args.query_repeats[: args.samples_per_side]

    calibration = []
    for model in models:
        samples = prompt_samples(
            model,
            args.reference_setting,
            reference_repeats,
            calibration_prompts,
            records,
            store,
        )
        calibration.append(samples.reshape(-1, samples.shape[-1]))
    bandwidth = median_bandwidth(np.concatenate(calibration), seed=args.seed)
    bandwidth *= args.bandwidth_multiplier
    if bandwidth <= 0:
        raise SystemExit("Calibrated bandwidth is not positive")

    first_record = records[(models[0], args.reference_setting, reference_repeats[0])]
    embedding_dimension = store.encode(first_record).shape[1]
    feature_map = RandomFourierMap.sample(
        embedding_dimension, args.rff_dimension, bandwidth, seed=args.seed
    )
    raw_dimension = len(evaluation_prompts) * args.rff_dimension
    projection = (
        gaussian_projection_matrix(raw_dimension, args.compact_dimension, seed=args.seed)
        if args.compact_dimension > 0
        else None
    )

    reference_samples = {
        model: prompt_samples(
            model,
            args.reference_setting,
            reference_repeats,
            evaluation_prompts,
            records,
            store,
        )
        for model in models
    }
    reference_single = np.stack(
        [reference_samples[model][:, 0, :].reshape(-1) for model in models]
    )
    reference_mean = np.stack(
        [reference_samples[model].mean(axis=1).reshape(-1) for model in models]
    )
    reference_rff = np.stack(
        [rfftrace_vector(reference_samples[model], feature_map) for model in models]
    )
    reference_compact = (
        np.stack(
            [
                compact_rfftrace_vector(reference_samples[model], feature_map, projection)
                for model in models
            ]
        )
        if projection is not None
        else None
    )

    summaries: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    vector_payload: dict[str, np.ndarray] = {"models": np.asarray(models)}
    vector_payload[f"reference_{args.reference_setting.code}_rfftrace"] = reference_rff
    if reference_compact is not None:
        vector_payload[f"reference_{args.reference_setting.code}_compact"] = reference_compact

    for setting in args.query_setting:
        comparison = f"{args.reference_setting.code}_to_{setting.code}_K{args.samples_per_side}"
        query_samples = {
            model: prompt_samples(
                model,
                setting,
                query_repeats,
                evaluation_prompts,
                records,
                store,
            )
            for model in models
        }
        query_single = np.stack(
            [query_samples[model][:, 0, :].reshape(-1) for model in models]
        )
        query_mean = np.stack(
            [query_samples[model].mean(axis=1).reshape(-1) for model in models]
        )
        query_rff = np.stack(
            [rfftrace_vector(query_samples[model], feature_map) for model in models]
        )
        query_compact = (
            np.stack(
                [
                    compact_rfftrace_vector(query_samples[model], feature_map, projection)
                    for model in models
                ]
            )
            if projection is not None
            else None
        )

        matrices: list[tuple[str, np.ndarray]] = [
            ("single_cosine", cosine_distance_matrix(query_single, reference_single)),
            ("mean_cosine", cosine_distance_matrix(query_mean, reference_mean)),
            ("rfftrace", squared_distance_matrix(query_rff, reference_rff)),
        ]
        exact = np.empty((len(models), len(models)), dtype=np.float32)
        for row, query_model in enumerate(models):
            for column, reference_model in enumerate(models):
                exact[row, column] = prompt_averaged_mmd2(
                    query_samples[query_model], reference_samples[reference_model], bandwidth
                )
        matrices.append(("exact_rbf_mmd", exact))
        if query_compact is not None and reference_compact is not None:
            matrices.append(
                ("compact_rfftrace", squared_distance_matrix(query_compact, reference_compact))
            )

        for method, matrix in matrices:
            metrics, method_predictions = ranking_rows(comparison, method, models, matrix)
            metrics.update(
                {
                    "samples_per_side": args.samples_per_side,
                    "prompt_count": len(evaluation_prompts),
                    "bandwidth": bandwidth,
                    "rff_dimension": args.rff_dimension,
                    "compact_dimension": args.compact_dimension,
                }
            )
            summaries.append(metrics)
            predictions.extend(method_predictions)
        vector_payload[f"query_{setting.code}_rfftrace"] = query_rff
        if query_compact is not None:
            vector_payload[f"query_{setting.code}_compact"] = query_compact

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "predictions.csv", predictions)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "selected_models.txt").write_text(
        "\n".join(models) + "\n", encoding="utf-8"
    )
    (args.output_dir / "calibration_prompts.txt").write_text(
        "\n".join(calibration_prompts) + "\n", encoding="utf-8"
    )
    (args.output_dir / "evaluation_prompts.txt").write_text(
        "\n".join(evaluation_prompts) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output_dir / "rfftrace_vectors.npz", **vector_payload)
    config = {
        "data_dirs": [str(path) for path in args.data_dir],
        "reference_setting": asdict(args.reference_setting),
        "query_settings": [asdict(setting) for setting in args.query_setting],
        "reference_repeats": list(reference_repeats),
        "query_repeats": list(query_repeats),
        "samples_per_side": args.samples_per_side,
        "model_count": len(models),
        "calibration_prompt_count": len(calibration_prompts),
        "evaluation_prompt_count": len(evaluation_prompts),
        "encoder": args.encoder,
        "normalize_embeddings": args.normalize_embeddings,
        "embedding_dimension": embedding_dimension,
        "bandwidth": bandwidth,
        "bandwidth_multiplier": args.bandwidth_multiplier,
        "rff_dimension": args.rff_dimension,
        "compact_dimension": args.compact_dimension,
        "seed": args.seed,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **config}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
