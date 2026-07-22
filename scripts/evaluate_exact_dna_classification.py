#!/usr/bin/env python3
"""Evaluate exact-model DNA classification under known and unknown decoding settings.

Unlike the legacy pairwise verification experiment, every method here solves
the same closed-set task: rank the exact model among a fixed gallery.  Training
and test generation repeats are supplied separately, and temperature/top-p are
never passed to the deployment classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


@dataclass(frozen=True)
class RunSpec:
    name: str
    suffix: str
    temperature: float
    top_p: float
    repeat: int

    @property
    def setting(self) -> tuple[float, float]:
        return self.temperature, self.top_p


@dataclass(frozen=True)
class DnaRecord:
    model_name: str
    vector: np.ndarray


def parse_run_spec(raw: str) -> RunSpec:
    parts = raw.split(":")
    if len(parts) != 5:
        raise ValueError(
            f"Invalid run {raw!r}; expected name:suffix:temperature:top_p:repeat"
        )
    name, suffix, temperature, top_p, repeat = parts
    if not name or not suffix:
        raise ValueError("Run name and suffix must be non-empty.")
    return RunSpec(name, suffix, float(temperature), float(top_p), int(repeat))


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip("/"))


def load_records_for_suffix(dataset_dir: Path, suffix: str) -> dict[str, DnaRecord]:
    records: dict[str, DnaRecord] = {}
    for model_dir in sorted(dataset_dir.glob(f"*{suffix}")):
        if not model_dir.is_dir() or not model_dir.name.endswith(suffix):
            continue
        dna_files = sorted(model_dir.glob("*_dna.json"))
        if not dna_files:
            continue
        payload = json.loads(dna_files[0].read_text(encoding="utf-8"))
        model_name = str(payload.get("metadata", {}).get("model_name") or model_dir.name)
        vector = np.asarray(payload["signature"], dtype=np.float32)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            continue
        records[model_name] = DnaRecord(model_name, vector)
    return records


def validate_split(train_runs: list[RunSpec], test_runs: list[RunSpec]) -> None:
    train_suffixes = {run.suffix for run in train_runs}
    test_suffixes = {run.suffix for run in test_runs}
    overlap = sorted(train_suffixes & test_suffixes)
    if overlap:
        raise ValueError(f"Train/test response leakage: suffixes occur in both splits: {overlap}")
    duplicate_names = len({run.name for run in train_runs + test_runs}) != len(train_runs + test_runs)
    if duplicate_names:
        raise ValueError("Every run name must be unique.")


def select_common_cohort(
    records: dict[str, dict[str, DnaRecord]], run_names: Iterable[str]
) -> list[str]:
    selected = list(run_names)
    if not selected:
        return []
    common = set(records[selected[0]])
    for name in selected[1:]:
        common &= set(records[name])
    return sorted(common)


def cosine_scores(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    gallery = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
    return query @ gallery.T


def bootstrap_mean_interval(
    values: Iterable[float], samples: int = 2000, seed: int = 42
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1 or samples <= 0:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    means = np.mean(array[rng.integers(0, len(array), size=(samples, len(array)))], axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def ranking_metrics(
    labels: list[str], scores: np.ndarray, classes: list[str], bootstrap_samples: int
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    class_index = {label: index for index, label in enumerate(classes)}
    rows: list[dict[str, object]] = []
    reciprocal_ranks: list[float] = []
    hits = {1: [], 3: [], 5: []}
    for row_index, label in enumerate(labels):
        order = np.argsort(-scores[row_index], kind="stable")
        truth_index = class_index[label]
        rank = int(np.flatnonzero(order == truth_index)[0]) + 1
        prediction = classes[int(order[0])]
        reciprocal_ranks.append(1.0 / rank)
        for k in hits:
            hits[k].append(float(rank <= k))
        rows.append(
            {
                "model": label,
                "predicted_model": prediction,
                "correct_top1": prediction == label,
                "self_rank": rank,
                "reciprocal_rank": 1.0 / rank,
            }
        )
    metrics: dict[str, float | int] = {"model_count": len(labels)}
    for k, values in hits.items():
        low, high = bootstrap_mean_interval(values, bootstrap_samples)
        metrics[f"top{k}"] = float(np.mean(values))
        metrics[f"top{k}_ci95_low"] = low
        metrics[f"top{k}_ci95_high"] = high
    mrr_low, mrr_high = bootstrap_mean_interval(reciprocal_ranks, bootstrap_samples)
    metrics.update(
        {
            "mrr": float(np.mean(reciprocal_ranks)),
            "mrr_ci95_low": mrr_low,
            "mrr_ci95_high": mrr_high,
            "chance_top1": 1.0 / len(classes),
            "chance_corrected_top1": (float(np.mean(hits[1])) - 1.0 / len(classes))
            / (1.0 - 1.0 / len(classes)),
        }
    )
    return metrics, rows


def training_arrays(
    train_runs: list[RunSpec], records: dict[str, dict[str, DnaRecord]], cohort: list[str]
) -> tuple[np.ndarray, list[str]]:
    vectors: list[np.ndarray] = []
    targets: list[str] = []
    for run in train_runs:
        for model in cohort:
            vectors.append(records[run.name][model].vector)
            targets.append(model)
    return np.stack(vectors), targets


def centroid_gallery(
    train_runs: list[RunSpec], records: dict[str, dict[str, DnaRecord]], cohort: list[str]
) -> np.ndarray:
    return np.stack(
        [np.mean([records[run.name][model].vector for run in train_runs], axis=0) for model in cohort]
    )


def fit_classifiers(train_x: np.ndarray, train_y: list[str], methods: list[str]) -> dict[str, object]:
    classifiers: dict[str, object] = {}
    if "linear_svm" in methods:
        classifiers["linear_svm"] = make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=30000, random_state=42),
        )
    if "rbf_svm" in methods:
        classifiers["rbf_svm"] = make_pipeline(
            StandardScaler(),
            SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", decision_function_shape="ovr"),
        )
    for classifier in classifiers.values():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            classifier.fit(train_x, train_y)
    return classifiers


def classifier_scores(classifier: object, query: np.ndarray, cohort: list[str]) -> np.ndarray:
    raw = np.asarray(classifier.decision_function(query))
    learned_classes = [str(value) for value in classifier.classes_]
    if raw.ndim == 1:
        raw = np.stack([-raw, raw], axis=1)
    indices = [learned_classes.index(model) for model in cohort]
    return raw[:, indices]


def evaluate_bound(
    bound: float,
    train_runs: list[RunSpec],
    test_runs: list[RunSpec],
    records: dict[str, dict[str, DnaRecord]],
    methods: list[str],
    minimum_models: int,
    bootstrap_samples: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    eligible_train = [run for run in train_runs if run.temperature <= bound]
    eligible_test = [run for run in test_runs if run.temperature <= bound]
    if not eligible_train or not eligible_test:
        raise ValueError(f"No train/test runs are available at temperature bound {bound:g}.")
    cohort = select_common_cohort(records, [run.name for run in eligible_train + eligible_test])
    if len(cohort) < minimum_models:
        raise ValueError(
            f"Temperature bound {bound:g} has only {len(cohort)} common models; "
            f"minimum is {minimum_models}. Complete/backfill the sweep before drawing conclusions."
        )

    train_x, train_y = training_arrays(eligible_train, records, cohort)
    gallery = centroid_gallery(eligible_train, records, cohort)
    classifiers = fit_classifiers(train_x, train_y, methods)
    summaries: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for run in eligible_test:
        query = np.stack([records[run.name][model].vector for model in cohort])
        score_sets: dict[str, np.ndarray] = {"cosine_centroid": cosine_scores(query, gallery)}
        score_sets.update(
            {method: classifier_scores(classifier, query, cohort) for method, classifier in classifiers.items()}
        )
        for method, scores in score_sets.items():
            metrics, rows = ranking_metrics(cohort, scores, cohort, bootstrap_samples)
            summaries.append(
                {
                    "temperature_bound": bound,
                    "method": method,
                    "run": run.name,
                    "temperature": run.temperature,
                    "top_p": run.top_p,
                    "repeat": run.repeat,
                    "training_run_count": len(eligible_train),
                    **metrics,
                }
            )
            predictions.extend(
                {
                    "temperature_bound": bound,
                    "method": method,
                    "run": run.name,
                    "temperature": run.temperature,
                    "top_p": run.top_p,
                    "repeat": run.repeat,
                    **row,
                }
                for row in rows
            )
    metadata = {
        "temperature_bound": bound,
        "cohort_size": len(cohort),
        "cohort": cohort,
        "train_runs": [run.name for run in eligible_train],
        "test_runs": [run.name for run in eligible_test],
    }
    return summaries, predictions, metadata


def aggregate_deployment(
    summaries: list[dict[str, object]], predictions: list[dict[str, object]], bootstrap_samples: int
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    keys = sorted({(float(row["temperature_bound"]), str(row["method"])) for row in summaries})
    for bound, method in keys:
        selected = [row for row in summaries if float(row["temperature_bound"]) == bound and row["method"] == method]
        # First average repeats within a cell, then weight each temperature/top-p cell equally.
        cells: dict[tuple[float, float], list[dict[str, object]]] = {}
        for row in selected:
            cells.setdefault((float(row["temperature"]), float(row["top_p"])), []).append(row)
        cell_accuracies = [float(np.mean([float(row["top1"]) for row in rows])) for rows in cells.values()]

        pred = [row for row in predictions if float(row["temperature_bound"]) == bound and row["method"] == method]
        model_names = sorted({str(row["model"]) for row in pred})
        model_values = []
        for model in model_names:
            per_model = [float(bool(row["correct_top1"])) for row in pred if row["model"] == model]
            model_values.append(float(np.mean(per_model)))
        low, high = bootstrap_mean_interval(model_values, bootstrap_samples)
        output.append(
            {
                "temperature_bound": bound,
                "method": method,
                "cell_count": len(cells),
                "model_count": len(model_names),
                "mixed_accuracy_equal_cell": float(np.mean(cell_accuracies)),
                "mixed_accuracy_ci95_low_model_cluster": low,
                "mixed_accuracy_ci95_high_model_cluster": high,
                "worst_setting_accuracy": float(np.min(cell_accuracies)),
                "best_setting_accuracy": float(np.max(cell_accuracies)),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact-model comparison of cosine centroids and multiclass SVMs."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("out/rand_chinese"))
    parser.add_argument("--train-run", action="append", default=[], help="name:suffix:temperature:top_p:repeat")
    parser.add_argument("--test-run", action="append", default=[], help="name:suffix:temperature:top_p:repeat")
    parser.add_argument("--temperature-bound", type=float, action="append", default=[])
    parser.add_argument("--method", choices=["linear_svm", "rbf_svm"], action="append", default=[])
    parser.add_argument("--minimum-models", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("out/exact_dna_classification"))
    parser.add_argument("--prefix", default="exact_dna")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_runs = [parse_run_spec(value) for value in args.train_run]
    test_runs = [parse_run_spec(value) for value in args.test_run]
    if not train_runs or not test_runs:
        raise SystemExit("Supply at least one --train-run and one independent --test-run.")
    try:
        validate_split(train_runs, test_runs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    methods = list(dict.fromkeys(args.method or ["linear_svm", "rbf_svm"]))
    bounds = sorted(set(args.temperature_bound or [run.temperature for run in test_runs]))
    records = {
        run.name: load_records_for_suffix(args.dataset_dir, run.suffix)
        for run in train_runs + test_runs
    }
    empty = [name for name, values in records.items() if not values]
    if empty:
        raise SystemExit(f"No DNA records found for runs: {empty}")

    summaries: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    cohorts: list[dict[str, object]] = []
    for bound in bounds:
        try:
            bound_summaries, bound_predictions, metadata = evaluate_bound(
                bound, train_runs, test_runs, records, methods, args.minimum_models, args.bootstrap_samples
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        summaries.extend(bound_summaries)
        predictions.extend(bound_predictions)
        cohorts.append(metadata)
    deployment = aggregate_deployment(summaries, predictions, args.bootstrap_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_setting_csv": args.output_dir / f"{args.prefix}_per_setting.csv",
        "predictions_csv": args.output_dir / f"{args.prefix}_predictions.csv",
        "deployment_csv": args.output_dir / f"{args.prefix}_bounded_deployment.csv",
        "summary_json": args.output_dir / f"{args.prefix}_summary.json",
    }
    write_csv(paths["per_setting_csv"], summaries)
    write_csv(paths["predictions_csv"], predictions)
    write_csv(paths["deployment_csv"], deployment)
    payload = {
        "protocol": "temperature-agnostic exact-model classification",
        "train_test_repeats_disjoint": True,
        "temperature_and_top_p_hidden_from_classifier": True,
        "cohorts": cohorts,
        "per_setting": summaries,
        "bounded_deployment": deployment,
    }
    paths["summary_json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
