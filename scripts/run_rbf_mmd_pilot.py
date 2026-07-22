#!/usr/bin/env python3
"""Small pilot for distributional model retrieval under stochastic decoding.

This script implements the first draft-stage experiment:
- compare single-sample cosine retrieval against exact RBF-MMD
- optionally compare a mean-response cosine baseline
- keep the experiment small, deterministic, and easy to inspect

The pilot uses repeated response runs as samples from a prompt-conditional
response distribution. For same-setting comparisons, the repeated runs are split
into reference/query halves. For cross-setting comparisons, the reference and
query settings are specified explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SettingSpec:
    temperature: float
    top_p: float

    @property
    def code(self) -> str:
        return f"t{int(round(self.temperature * 10)):02d}_p{int(round(self.top_p * 10)):02d}"


@dataclass(frozen=True)
class RunRecord:
    model_name: str
    setting: SettingSpec
    repeat: int
    path: Path
    prompts: list[str]
    responses: list[str]


def parse_setting(raw: str) -> SettingSpec:
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid setting {raw!r}; expected temperature:top_p, e.g. 0.2:0.8")
    return SettingSpec(temperature=float(parts[0]), top_p=float(parts[1]))


def format_setting(setting: SettingSpec) -> str:
    return f"{setting.temperature:g}:{setting.top_p:g}"


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip("/"))


def parse_run_dir_name(name: str) -> tuple[str, SettingSpec, int] | None:
    match = re.match(r"^(?P<model>.+)_(?P<t>t\d{2})_(?P<p>p\d{2})_r(?P<repeat>\d+)$", name)
    if not match:
        return None
    temperature = int(match.group("t")[1:]) / 10.0
    top_p = int(match.group("p")[1:]) / 10.0
    return match.group("model"), SettingSpec(temperature=temperature, top_p=top_p), int(match.group("repeat"))


def load_run_record(path: Path) -> RunRecord:
    payload = json.loads((path / "responses.json").read_text(encoding="utf-8"))
    items = payload.get("items", [])
    prompts = [str(item.get("prompt", "")) for item in items]
    responses = [str(item.get("response", "")) for item in items]
    parsed = parse_run_dir_name(path.name)
    if parsed is None:
        raise ValueError(f"Unrecognized run directory name: {path.name}")
    model_name, setting, repeat = parsed
    return RunRecord(
        model_name=model_name,
        setting=setting,
        repeat=repeat,
        path=path,
        prompts=prompts,
        responses=responses,
    )


def load_run_records(data_dir: Path, settings: set[SettingSpec]) -> dict[tuple[str, str, int], RunRecord]:
    records: dict[tuple[str, str, int], RunRecord] = {}
    code_to_setting = {setting.code: setting for setting in settings}
    for run_dir in sorted(data_dir.glob("*")):
        if not run_dir.is_dir() or not (run_dir / "responses.json").exists():
            continue
        parsed = parse_run_dir_name(run_dir.name)
        if parsed is None:
            continue
        model_name, setting, repeat = parsed
        if setting.code not in code_to_setting:
            continue
        records[(model_name, setting.code, repeat)] = load_run_record(run_dir)
    return records


def select_models(records: dict[tuple[str, str, int], RunRecord], settings: list[SettingSpec], model_limit: int) -> list[str]:
    per_setting: list[set[str]] = []
    for setting in settings:
        model_to_repeats: dict[str, set[int]] = {}
        for (model_name, code, repeat), record in records.items():
            if code != setting.code:
                continue
            model_to_repeats.setdefault(model_name, set()).add(repeat)
        candidates = {model for model, repeats in model_to_repeats.items() if len(repeats) >= 2}
        per_setting.append(candidates)

    common = sorted(set.intersection(*per_setting)) if per_setting else []
    return common if model_limit <= 0 else common[:model_limit]


def select_prompts(records: dict[tuple[str, str, int], RunRecord], selected_models: list[str], settings: list[SettingSpec], prompt_limit: int) -> list[str]:
    prompt_sets: list[list[str]] = []
    for setting in settings:
        reference = None
        for model_name in selected_models:
            for repeat in sorted({repeat for (m, code, repeat) in records if m == model_name and code == setting.code}):
                reference = records[(model_name, setting.code, repeat)]
                break
            if reference is not None:
                break
        if reference is None:
            continue
        prompt_sets.append(reference.prompts)

    if not prompt_sets:
        return []

    common = list(prompt_sets[0])
    for prompts in prompt_sets[1:]:
        common = [prompt for prompt in common if prompt in prompts]
    return common if prompt_limit <= 0 else common[:prompt_limit]


def chunk_runs_for_same_setting(runs: list[RunRecord]) -> tuple[list[RunRecord], list[RunRecord]]:
    runs = sorted(runs, key=lambda record: record.repeat)
    if len(runs) < 2:
        raise ValueError("Need at least two repeats for same-setting comparison.")
    split = max(1, len(runs) // 2)
    ref = runs[:split]
    qry = runs[split:]
    if not qry:
        qry = runs[-1:]
        ref = runs[:-1]
    return ref, qry


def load_embeddings(texts: Iterable[str], cache: dict[str, np.ndarray], features: int = 4096) -> np.ndarray:
    missing = [text for text in texts if text not in cache]
    if missing:
        from sklearn.feature_extraction.text import HashingVectorizer

        vectorizer = HashingVectorizer(
            n_features=features,
            analyzer="char_wb",
            ngram_range=(3, 5),
            alternate_sign=False,
            norm=None,
            lowercase=False,
        )
        matrix = vectorizer.transform(missing).astype(np.float32).toarray()
        for text, row in zip(missing, matrix):
            norm = float(np.linalg.norm(row))
            cache[text] = row if norm == 0.0 else row / norm
    return np.stack([cache[text] for text in texts], axis=0).astype(np.float32)


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def concat_prompt_vectors(prompt_embeddings: dict[str, np.ndarray], prompts: list[str]) -> np.ndarray:
    return np.concatenate([prompt_embeddings[prompt] for prompt in prompts], axis=0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def cosine_distance_matrix(query: np.ndarray, base: np.ndarray) -> np.ndarray:
    query_norm = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    base_norm = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-12)
    return 1.0 - query_norm @ base_norm.T


def pairwise_sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=2)


def median_bandwidth(samples: np.ndarray) -> float:
    if len(samples) < 2:
        return 1.0
    sq = pairwise_sqdist(samples, samples)
    dist = np.sqrt(np.maximum(sq, 0.0))
    vals = dist[np.triu_indices_from(dist, k=1)]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 1.0
    return float(np.median(vals))


def rbf_kernel(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-pairwise_sqdist(a, b) / (2.0 * sigma * sigma))


def exact_mmd2(a: np.ndarray, b: np.ndarray, sigma: float) -> float:
    kaa = rbf_kernel(a, a, sigma).mean()
    kbb = rbf_kernel(b, b, sigma).mean()
    kab = rbf_kernel(a, b, sigma).mean()
    return float(kaa + kbb - 2.0 * kab)


def build_prompt_samples(
    runs: list[RunRecord],
    prompts: list[str],
    cache: dict[str, np.ndarray],
    features: int,
) -> dict[str, dict[str, np.ndarray]]:
    samples: dict[str, dict[str, np.ndarray]] = {}
    for run in runs:
        prompt_to_text = {prompt: response for prompt, response in zip(run.prompts, run.responses)}
        ordered_texts = [prompt_to_text[prompt] for prompt in prompts]
        embeddings = load_embeddings(ordered_texts, cache, features=features)
        samples[run.path.name] = {prompt: embeddings[idx] for idx, prompt in enumerate(prompts)}
    return samples


def gather_prompt_sample_sets(
    run_group: list[RunRecord],
    prompts: list[str],
    cache: dict[str, np.ndarray],
    features: int,
) -> dict[str, dict[str, np.ndarray]]:
    grouped = build_prompt_samples(run_group, prompts, cache, features)
    by_model: dict[str, dict[str, list[np.ndarray]]] = {}
    for run in run_group:
        prompt_embeddings = grouped[run.path.name]
        model_bucket = by_model.setdefault(run.model_name, {prompt: [] for prompt in prompts})
        for prompt in prompts:
            model_bucket[prompt].append(prompt_embeddings[prompt])
    return {
        model: {prompt: np.stack(samples, axis=0).astype(np.float32) for prompt, samples in prompt_map.items()}
        for model, prompt_map in by_model.items()
    }


def build_point_vectors(sample_sets: dict[str, dict[str, np.ndarray]], prompts: list[str], mode: str) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for model, prompt_map in sample_sets.items():
        per_prompt: list[np.ndarray] = []
        for prompt in prompts:
            samples = prompt_map[prompt]
            if mode == "single":
                per_prompt.append(samples[0])
            elif mode == "mean":
                per_prompt.append(np.mean(samples, axis=0))
            else:
                raise ValueError(f"Unknown point mode: {mode}")
        vectors[model] = np.concatenate(per_prompt, axis=0)
    return vectors


def prompt_distance_matrix(
    ref_sets: dict[str, dict[str, np.ndarray]],
    qry_sets: dict[str, dict[str, np.ndarray]],
    prompts: list[str],
    method: str,
    sigma: float,
) -> tuple[list[str], np.ndarray]:
    models = sorted(set(ref_sets) & set(qry_sets))
    matrix = np.zeros((len(models), len(models)), dtype=np.float32)
    for row, qry_model in enumerate(models):
        for col, ref_model in enumerate(models):
            prompt_values = []
            for prompt in prompts:
                ref_samples = ref_sets[ref_model][prompt]
                qry_samples = qry_sets[qry_model][prompt]
                if method == "mmd":
                    prompt_values.append(exact_mmd2(qry_samples, ref_samples, sigma=sigma))
                else:
                    raise ValueError(f"Unsupported prompt method: {method}")
            matrix[row, col] = float(np.mean(prompt_values))
    return models, matrix


def retrieval_metrics(labels: list[str], matrix: np.ndarray, bootstrap_samples: int = 2000, seed: int = 42) -> dict[str, float]:
    preds = list(np.argmin(matrix, axis=1))
    y_true = labels
    y_pred = [labels[idx] for idx in preds]
    correct = [t == p for t, p in zip(y_true, y_pred)]
    ranks = []
    for row_idx, label in enumerate(labels):
        order = list(np.argsort(matrix[row_idx]))
        ranks.append(order.index(row_idx) + 1)

    top_hits = {1: 0, 3: 0, 5: 0}
    for rank in ranks:
        for k in top_hits:
            if rank <= k:
                top_hits[k] += 1

    rng = np.random.default_rng(seed)

    def interval(values: list[float]) -> tuple[float, float]:
        array = np.asarray(values, dtype=np.float64)
        if len(array) <= 1 or bootstrap_samples <= 0:
            value = float(np.mean(array)) if len(array) else float("nan")
            return value, value
        means = np.mean(array[rng.integers(0, len(array), size=(bootstrap_samples, len(array)))], axis=1)
        return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))

    top1_low, top1_high = interval([float(value) for value in correct])
    mrr_low, mrr_high = interval([1.0 / rank for rank in ranks])
    return {
        "model_count": len(labels),
        "top1": float(np.mean(correct)) if correct else 0.0,
        "top3": top_hits[3] / len(labels) if labels else 0.0,
        "top5": top_hits[5] / len(labels) if labels else 0.0,
        "mrr": float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0,
        "top1_ci95_low": top1_low,
        "top1_ci95_high": top1_high,
        "mrr_ci95_low": mrr_low,
        "mrr_ci95_high": mrr_high,
        "mean_self_distance": float(np.mean([matrix[i, i] for i in range(len(labels))])) if labels else float("nan"),
        "mean_nearest_distance": float(np.mean([np.min(matrix[i]) for i in range(len(labels))])) if labels else float("nan"),
    }


def evaluate_comparison(
    name: str,
    reference_runs: list[RunRecord],
    query_runs: list[RunRecord],
    prompts: list[str],
    cache: dict[str, np.ndarray],
    features: int,
    sigma: float,
) -> list[dict[str, object]]:
    ref_sets = gather_prompt_sample_sets(reference_runs, prompts, cache, features)
    qry_sets = gather_prompt_sample_sets(query_runs, prompts, cache, features)
    labels = sorted(set(ref_sets) & set(qry_sets))
    ref_sets = {label: ref_sets[label] for label in labels}
    qry_sets = {label: qry_sets[label] for label in labels}

    rows: list[dict[str, object]] = []

    single_ref = build_point_vectors(ref_sets, prompts, mode="single")
    single_qry = build_point_vectors(qry_sets, prompts, mode="single")
    mean_ref = build_point_vectors(ref_sets, prompts, mode="mean")
    mean_qry = build_point_vectors(qry_sets, prompts, mode="mean")

    def eval_point(method: str, ref_vectors: dict[str, np.ndarray], qry_vectors: dict[str, np.ndarray]) -> dict[str, object]:
        ordered = sorted(ref_vectors)
        ref_matrix = np.stack([ref_vectors[label] for label in ordered], axis=0)
        qry_matrix = np.stack([qry_vectors[label] for label in ordered], axis=0)
        matrix = cosine_distance_matrix(qry_matrix, ref_matrix)
        metrics = retrieval_metrics(ordered, matrix)
        return {
            "comparison": name,
            "method": method,
            "sigma": sigma,
            "prompt_count": len(prompts),
            **metrics,
        }

    rows.append(eval_point("single_cosine", single_ref, single_qry))
    rows.append(eval_point("mean_cosine", mean_ref, mean_qry))

    mmd_labels, mmd_matrix = prompt_distance_matrix(ref_sets, qry_sets, prompts, method="mmd", sigma=sigma)
    rows.append(
        {
            "comparison": name,
            "method": "exact_rbf_mmd",
            "sigma": sigma,
            "prompt_count": len(prompts),
            **retrieval_metrics(mmd_labels, mmd_matrix),
        }
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Budget-matched exact RBF-MMD vs cosine retrieval experiment.")
    parser.add_argument("--data-dir", type=Path, default=Path("out/rand_chinese"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/rbf_mmd_pilot"))
    parser.add_argument("--reference-setting", type=parse_setting, default=parse_setting("0.2:0.8"))
    parser.add_argument("--comparison-setting", type=parse_setting, action="append", default=[parse_setting("0.2:0.8"), parse_setting("0.3:0.8"), parse_setting("0.3:0.9")])
    parser.add_argument("--model-limit", type=int, default=0, help="Maximum common models; 0 uses all.")
    parser.add_argument("--minimum-models", type=int, default=100, help="Fail below this common-cohort size.")
    parser.add_argument("--prompt-limit", type=int, default=0, help="Maximum common prompts; 0 uses all.")
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--embedding-features", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-file", type=Path, default=Path("docs/rbf_mmd_pilot.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    selected_settings = [args.reference_setting] + [setting for setting in args.comparison_setting if setting != args.reference_setting]
    records = load_run_records(args.data_dir, set(selected_settings))
    selected_models = select_models(records, selected_settings, args.model_limit)
    if not selected_models:
        raise SystemExit("No models satisfy the requested settings and repeat coverage.")
    if len(selected_models) < args.minimum_models:
        raise SystemExit(
            f"Only {len(selected_models)} models have the required repeat coverage; "
            f"minimum is {args.minimum_models}. Use --minimum-models only to label an explicit pilot."
        )

    prompts = select_prompts(records, selected_models, selected_settings, args.prompt_limit)
    if not prompts:
        raise SystemExit("No common prompts found for the selected models/settings.")

    calibration_count = max(1, int(round(len(prompts) * args.calibration_ratio)))
    calibration_prompts = prompts[:calibration_count]
    eval_prompts = prompts[calibration_count:] or prompts

    cache: dict[str, np.ndarray] = {}
    all_rows: list[dict[str, object]] = []
    comparison_notes: list[str] = []

    for setting in selected_settings:
        per_model = {}
        for model_name in selected_models:
            run_records = [record for (model, code, _repeat), record in records.items() if model == model_name and code == setting.code]
            run_records = sorted(run_records, key=lambda record: record.repeat)
            if setting == args.reference_setting:
                ref_runs, qry_runs = chunk_runs_for_same_setting(run_records)
                comparison_name = f"same_{setting.code}"
            else:
                ref_runs = [record for (model, code, _repeat), record in records.items() if model == model_name and code == args.reference_setting.code]
                ref_runs = sorted(ref_runs, key=lambda record: record.repeat)
                qry_runs = run_records
                comparison_name = f"cross_{args.reference_setting.code}_to_{setting.code}"
            per_model[model_name] = (ref_runs, qry_runs)

        ref_for_sigma = [
            record
            for (model, code, _repeat), record in records.items()
            if code == args.reference_setting.code and model in selected_models
        ]
        ref_for_sigma = sorted(ref_for_sigma, key=lambda record: (record.model_name, record.repeat))
        if not ref_for_sigma:
            raise SystemExit("Unable to find runs for bandwidth calibration.")

        # Estimate bandwidth on calibration prompts using the reference setting only.
        calibration_sets = gather_prompt_sample_sets(ref_for_sigma, calibration_prompts, cache, args.embedding_features)
        calibration_embeddings = []
        for model in selected_models:
            if model not in calibration_sets:
                continue
            for prompt in calibration_prompts:
                calibration_embeddings.extend(list(calibration_sets[model][prompt]))
        sigma = median_bandwidth(np.stack(calibration_embeddings, axis=0))

        for model_name, (ref_runs, qry_runs) in per_model.items():
            if not ref_runs or not qry_runs:
                comparison_notes.append(f"skip {model_name} {setting.code} due to insufficient repeats")

        if setting == args.reference_setting:
            ref_runs_all = [record for (model, code, _repeat), record in records.items() if code == setting.code and model in selected_models]
            ref_runs_all = sorted(ref_runs_all, key=lambda record: (record.model_name, record.repeat))
            ref_runs, qry_runs = chunk_runs_for_same_setting(ref_runs_all)
            comparison_name = f"same_{setting.code}"
            all_rows.extend(evaluate_comparison(comparison_name, ref_runs, qry_runs, eval_prompts, cache, args.embedding_features, sigma))
        else:
            ref_runs = [record for (model, code, _repeat), record in records.items() if code == args.reference_setting.code and model in selected_models]
            qry_runs = [record for (model, code, _repeat), record in records.items() if code == setting.code and model in selected_models]
            ref_runs = sorted(ref_runs, key=lambda record: (record.model_name, record.repeat))
            qry_runs = sorted(qry_runs, key=lambda record: (record.model_name, record.repeat))
            comparison_name = f"cross_{args.reference_setting.code}_to_{setting.code}"
            all_rows.extend(evaluate_comparison(comparison_name, ref_runs, qry_runs, eval_prompts, cache, args.embedding_features, sigma))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "summary.csv"
    summary_json = args.output_dir / "summary.json"
    models_file = args.output_dir / "selected_models.txt"
    prompts_file = args.output_dir / "selected_prompts.txt"
    report_path = args.report_file

    write_csv(summary_csv, all_rows)
    summary_json.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    models_file.write_text("\n".join(selected_models) + "\n", encoding="utf-8")
    prompts_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RBF-MMD Pilot",
        "",
        "## Setup",
        f"- Data dir: `{args.data_dir}`",
        f"- Reference setting: `{format_setting(args.reference_setting)}`",
        f"- Comparison settings: {', '.join(f'`{format_setting(s)}`' for s in selected_settings)}",
        f"- Selected models: `{len(selected_models)}`",
        f"- Selected prompts: `{len(prompts)}`",
        f"- Calibration prompts: `{len(calibration_prompts)}`",
        f"- Evaluation prompts: `{len(eval_prompts)}`",
        f"- Embedding features: `{args.embedding_features}`",
        "",
        "## Notes",
        "- `single_cosine` uses one response sample per prompt from the selected run split.",
        "- `mean_cosine` averages repeated samples per prompt before concatenation.",
        "- `exact_rbf_mmd` computes prompt-wise squared RBF-MMD and averages across prompts.",
        "- `same_*` compares two repeat splits from the same decoding setting.",
        "- `cross_*` compares the reference setting against another setting.",
        "- NA in intermediate figures means the model or prompt was not present in the required split, or the run lacked enough repeats.",
        "",
        "## Results",
    ]
    for row in all_rows:
        lines.append(
            f"- `{row['comparison']}` / `{row['method']}`: top1={row['top1']:.3f}, top3={row['top3']:.3f}, top5={row['top5']:.3f}, mrr={row['mrr']:.3f}"
        )
    if comparison_notes:
        lines.extend(["", "## Skips"] + [f"- {note}" for note in comparison_notes])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv),
                "summary_json": str(summary_json),
                "models_file": str(models_file),
                "prompts_file": str(prompts_file),
                "report": str(report_path),
                "selected_models": selected_models,
                "selected_prompts": prompts,
                "rows": all_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
