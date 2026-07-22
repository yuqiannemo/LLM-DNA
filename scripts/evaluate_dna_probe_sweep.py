#!/usr/bin/env python3
"""Evaluate DNA identification robustness across generation settings.

Each rerun suffix is treated as a query set. Query DNAs are matched against the
original/base DNAs by nearest distance, and optionally by a pairwise LinearSVC
probe trained on balanced positive/negative pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


@dataclass(frozen=True)
class RunSpec:
    name: str
    suffix: str
    temperature: float | None
    top_p: float | None


@dataclass(frozen=True)
class DnaRecord:
    model_name: str
    safe_name: str
    path: Path
    vector: np.ndarray


@dataclass(frozen=True)
class PairExample:
    left_label: str
    right_label: str
    distance: float
    target: int


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip("/"))


def parse_float_or_none(value: str) -> float | None:
    value = value.strip()
    if value in {"", "-", "none", "None", "null"}:
        return None
    return float(value)


def parse_run_spec(raw: str) -> RunSpec:
    """Parse name:suffix:temperature:top_p."""
    parts = raw.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Invalid --run {raw!r}. Expected format name:suffix:temperature:top_p, "
            "for example temp07:_temp07_run:0.7:0.9"
        )
    name, suffix, temperature, top_p = parts
    if not name or not suffix:
        raise ValueError(f"Invalid --run {raw!r}: name and suffix are required.")
    return RunSpec(
        name=name,
        suffix=suffix,
        temperature=parse_float_or_none(temperature),
        top_p=parse_float_or_none(top_p),
    )


def load_dna(path: Path) -> DnaRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    model_name = str(metadata.get("model_name") or path.parent.name)
    vector = np.asarray(payload["signature"], dtype=np.float32)
    return DnaRecord(
        model_name=model_name,
        safe_name=safe_model_name(model_name),
        path=path,
        vector=vector,
    )


def load_records_for_suffix(dataset_dir: Path, suffix: str) -> dict[str, DnaRecord]:
    records: dict[str, DnaRecord] = {}
    pattern = "*" if suffix == "" else f"*{suffix}"
    for model_dir in sorted(dataset_dir.glob(pattern)):
        if not model_dir.is_dir():
            continue
        if suffix == "" and re.search(r"_[0-9]+_run$|_temp.*_run$", model_dir.name):
            continue
        if suffix and not model_dir.name.endswith(suffix):
            continue
        dna_files = sorted(model_dir.glob("*_dna.json"))
        if not dna_files:
            continue
        record = load_dna(dna_files[0])
        if suffix == "" and model_dir.name != record.safe_name:
            # Base records must remain in the unsuffixed directory.
            # Rerun directories carry the same metadata.model_name and would
            # otherwise overwrite the real base record.
            continue
        records[record.model_name] = record
    return records


def select_common_model_cohort(
    base_records: dict[str, DnaRecord],
    run_records: dict[str, dict[str, DnaRecord]],
    minimum_run_models: int,
) -> tuple[dict[str, DnaRecord], dict[str, dict[str, DnaRecord]], dict[str, object]]:
    """Keep sufficiently populated runs and restrict them to one shared cohort."""
    available_counts = {
        run_name: len(set(base_records) & set(records))
        for run_name, records in run_records.items()
    }
    included_names = [
        run_name
        for run_name in run_records
        if available_counts[run_name] >= minimum_run_models
    ]
    excluded_names = [run_name for run_name in run_records if run_name not in included_names]
    if not included_names:
        raise ValueError(
            f"No run has at least {minimum_run_models} models in the reference gallery."
        )

    common_models = set(base_records)
    for run_name in included_names:
        common_models &= set(run_records[run_name])
    if len(common_models) < 2:
        raise ValueError(
            "The selected runs have fewer than two common models; increase "
            "--minimum-run-models or wait for more sweep outputs."
        )

    ordered_models = sorted(common_models)
    filtered_base = {model: base_records[model] for model in ordered_models}
    filtered_runs = {
        run_name: {model: run_records[run_name][model] for model in ordered_models}
        for run_name in included_names
    }
    metadata: dict[str, object] = {
        "minimum_run_models": int(minimum_run_models),
        "common_model_count": len(ordered_models),
        "common_models": ordered_models,
        "available_model_counts": available_counts,
        "included_runs": included_names,
        "excluded_runs": excluded_names,
    }
    return filtered_base, filtered_runs, metadata


def cosine_distance_matrix(query: np.ndarray, base: np.ndarray) -> np.ndarray:
    query_norm = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    base_norm = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-12)
    return 1.0 - query_norm @ base_norm.T


def euclidean_distance_matrix(query: np.ndarray, base: np.ndarray) -> np.ndarray:
    diff = query[:, None, :] - base[None, :, :]
    return np.linalg.norm(diff, axis=2)


def macro_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, float]:
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(np.mean(precisions)) if precisions else 0.0,
        "recall_macro": float(np.mean(recalls)) if recalls else 0.0,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "precision_micro": float(accuracy),
        "recall_micro": float(accuracy),
        "f1_micro": float(accuracy),
    }


def is_same_or_related(left: str, right: str, relation_lookup: dict[str, set[str]]) -> bool:
    return left == right or right in relation_lookup.get(left, set()) or left in relation_lookup.get(right, set())


def _split_relation_values(raw: str) -> list[str]:
    return [item.strip() for item in re.split(r"[|,;]", raw) if item.strip()]


def load_relation_groups(path: Path | None) -> list[set[str]]:
    """Load positive relation groups from JSON/JSONL/CSV/TSV.

    Supported forms:
    - JSONL or JSON object with {"models": [...]} or {"source": ..., "target": ...}
    - CSV/TSV rows with columns source,target or models
    """
    if path is None:
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    groups: list[set[str]] = []

    def add_group(values: Iterable[str]) -> None:
        group = {value.strip() for value in values if value.strip()}
        if len(group) >= 2:
            groups.append(group)

    def process_record(record: dict[str, object]) -> None:
        if "models" in record:
            models = record["models"]
            if isinstance(models, str):
                add_group(_split_relation_values(models))
            elif isinstance(models, list):
                add_group(str(item) for item in models)
        elif "source" in record and "target" in record:
            add_group([str(record["source"]), str(record["target"])])
        elif "model" in record and "group" in record:
            add_group([str(record["model"]), str(record["group"])])

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
        for row in reader:
            process_record({key: value for key, value in row.items() if value is not None})
        return groups

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        process_record(payload)
        return groups
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                process_record(item)
        return groups

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            process_record(payload)
    return groups


def build_relation_lookup(groups: list[set[str]]) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = {}
    for group in groups:
        for model in group:
            lookup.setdefault(model, set()).update(group - {model})
    return lookup


def build_pair_examples(
    base_records: dict[str, DnaRecord],
    query_records: dict[str, DnaRecord],
    metric: str,
    relation_lookup: dict[str, set[str]],
    positive_mode: str = "same_or_related",
) -> list[PairExample]:
    labels = sorted(set(base_records) & set(query_records))
    base_matrix = np.stack([base_records[label].vector for label in labels], axis=0)
    query_matrix = np.stack([query_records[label].vector for label in labels], axis=0)
    distances = cosine_distance_matrix(query_matrix, base_matrix) if metric == "cosine" else euclidean_distance_matrix(query_matrix, base_matrix)

    examples: list[PairExample] = []
    for query_idx, query_label in enumerate(labels):
        for base_idx, base_label in enumerate(labels):
            same = query_label == base_label
            related = not same and is_same_or_related(query_label, base_label, relation_lookup)
            if positive_mode == "exact_only":
                if related:
                    continue
                target = int(same)
            elif positive_mode == "lineage_only":
                if same:
                    continue
                target = int(related)
            elif positive_mode == "same_or_related":
                target = int(same or related)
            else:
                raise ValueError(f"Unknown positive mode: {positive_mode}")
            examples.append(
                PairExample(
                    left_label=query_label,
                    right_label=base_label,
                    distance=float(distances[query_idx, base_idx]),
                    target=target,
                )
            )
    return examples


def balance_examples(examples: list[PairExample], negative_ratio: float, seed: int = 42) -> tuple[list[PairExample], dict[str, int]]:
    positives = [example for example in examples if example.target == 1]
    negatives = [example for example in examples if example.target == 0]
    if not positives or not negatives:
        return examples, {
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "sampled_negative_count": len(negatives),
        }

    rng = np.random.default_rng(seed)
    target_negative_count = int(round(len(positives) * negative_ratio))
    target_negative_count = max(1, min(len(negatives), target_negative_count))
    negative_indices = rng.choice(len(negatives), size=target_negative_count, replace=False)
    sampled_negatives = [negatives[int(idx)] for idx in sorted(negative_indices)]
    balanced = positives + sampled_negatives
    rng.shuffle(balanced)
    return balanced, {
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "sampled_negative_count": len(sampled_negatives),
    }


def binary_metrics(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict[str, float]:
    if not y_true:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "specificity": 0.0,
            "mcc": 0.0,
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "tp": 0.0,
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    specificity = 0.0 if tn + fp == 0 else tn / (tn + fp)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    balanced = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = float("nan")
    try:
        average_precision = average_precision_score(y_true, y_score)
    except ValueError:
        average_precision = float("nan")

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "mcc": float(mcc),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def bootstrap_mean_interval(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap CI for a model-level mean."""
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1 or samples <= 0:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, array.size, size=(samples, array.size))
    means = np.mean(array[draw_indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval without requiring scipy."""
    if total <= 0:
        return float("nan"), float("nan")
    # 1.959963984540054 is the two-sided 95% standard-normal quantile.  The
    # CLI currently exposes 95% intervals; reject silent misuse for other CIs.
    if not math.isclose(confidence, 0.95):
        raise ValueError("wilson_interval currently supports confidence=0.95 only")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return float(max(0.0, centre - radius)), float(min(1.0, centre + radius))


def evaluate_nearest(
    base_records: dict[str, DnaRecord],
    query_records: dict[str, DnaRecord],
    metric: str,
    top_ks: Iterable[int],
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    labels = sorted(set(base_records) & set(query_records))
    base_matrix = np.stack([base_records[label].vector for label in labels], axis=0)
    query_matrix = np.stack([query_records[label].vector for label in labels], axis=0)
    distances = cosine_distance_matrix(query_matrix, base_matrix) if metric == "cosine" else euclidean_distance_matrix(query_matrix, base_matrix)

    y_true: list[str] = []
    y_pred: list[str] = []
    rows: list[dict[str, object]] = []
    top_hits = {int(k): 0 for k in top_ks}
    reciprocal_ranks: list[float] = []
    self_distances: list[float] = []
    nearest_distances: list[float] = []
    margins: list[float] = []

    for idx, label in enumerate(labels):
        order = list(np.argsort(distances[idx]))
        pred = labels[order[0]]
        rank = order.index(idx) + 1
        sorted_distances = distances[idx][order]
        self_distance = float(distances[idx, idx])
        nearest_distance = float(sorted_distances[0])
        second_distance = float(sorted_distances[1]) if len(sorted_distances) > 1 else float("nan")
        margin = second_distance - nearest_distance if np.isfinite(second_distance) else float("nan")

        y_true.append(label)
        y_pred.append(pred)
        reciprocal_ranks.append(1.0 / rank)
        self_distances.append(self_distance)
        nearest_distances.append(nearest_distance)
        margins.append(margin)
        for k in top_hits:
            if rank <= k:
                top_hits[k] += 1
        rows.append(
            {
                "model": label,
                "predicted_model": pred,
                "correct_top1": pred == label,
                "self_rank": rank,
                "self_distance": self_distance,
                "nearest_distance": nearest_distance,
                "nearest_margin": margin,
                "top3_hit": rank <= 3,
                "top5_hit": rank <= 5,
            }
        )

    metrics = macro_metrics(y_true, y_pred, labels)
    metrics.update(
        {
            "model_count": len(labels),
            "top1": metrics["accuracy"],
            "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
            "mean_self_distance": float(np.mean(self_distances)) if self_distances else float("nan"),
            "median_self_distance": float(np.median(self_distances)) if self_distances else float("nan"),
            "mean_nearest_distance": float(np.mean(nearest_distances)) if nearest_distances else float("nan"),
            "mean_nearest_margin": float(np.nanmean(margins)) if margins else float("nan"),
        }
    )
    mrr_low, mrr_high = bootstrap_mean_interval(
        reciprocal_ranks,
        samples=bootstrap_samples,
        seed=seed,
    )
    metrics["mrr_ci95_low"] = mrr_low
    metrics["mrr_ci95_high"] = mrr_high
    for k, count in top_hits.items():
        metrics[f"top{k}"] = count / len(labels) if labels else 0.0
        metrics[f"top{k}_count"] = count
        low, high = wilson_interval(count, len(labels))
        metrics[f"top{k}_ci95_low"] = low
        metrics[f"top{k}_ci95_high"] = high
    return metrics, rows


def evaluate_pairwise_svm(
    base_records: dict[str, DnaRecord],
    run_records: dict[str, dict[str, DnaRecord]],
    test_run_name: str,
    metric: str,
    relation_lookup: dict[str, set[str]],
    negative_ratio: float,
    test_negative_ratio: float | None = 1.0,
    positive_mode: str = "same_or_related",
) -> dict[str, float | int | str]:
    test_records = run_records[test_run_name]
    labels = sorted(set(base_records) & set(test_records))
    if len(labels) < 2:
        return {"linear_status": "skipped_too_few_classes"}

    train_examples: list[PairExample] = []
    for run_name, records in run_records.items():
        if run_name == test_run_name:
            continue
        examples = build_pair_examples(base_records, records, metric, relation_lookup, positive_mode)
        train_examples.extend(examples)

    test_examples = build_pair_examples(base_records, test_records, metric, relation_lookup, positive_mode)
    if not train_examples or not test_examples:
        return {"linear_status": "skipped_empty_pairs"}

    balanced_train, train_stats = balance_examples(train_examples, negative_ratio)
    if test_negative_ratio is None:
        balanced_test = list(test_examples)
        test_stats = {
            "positive_count": sum(example.target == 1 for example in test_examples),
            "negative_count": sum(example.target == 0 for example in test_examples),
            "sampled_negative_count": sum(example.target == 0 for example in test_examples),
        }
    else:
        balanced_test, test_stats = balance_examples(test_examples, negative_ratio=test_negative_ratio)

    train_targets = [example.target for example in balanced_train]
    test_targets = [example.target for example in balanced_test]
    if len(set(train_targets)) < 2 or len(set(test_targets)) < 2:
        return {"linear_status": "skipped_too_few_binary_classes"}

    train_x = np.array([[-example.distance] for example in balanced_train], dtype=np.float32)
    test_x = np.array([[-example.distance] for example in balanced_test], dtype=np.float32)

    clf = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=20000, random_state=42),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        clf.fit(train_x, train_targets)

    y_pred = list(clf.predict(test_x))
    decision_scores = list(clf.decision_function(test_x))
    metrics = {f"linear_{key}": value for key, value in binary_metrics(test_targets, y_pred, decision_scores).items()}
    metrics.update(
        {
            "linear_status": "ok",
            "linear_train_samples": len(balanced_train),
            "linear_train_positive_samples": train_stats["positive_count"],
            "linear_train_negative_samples": train_stats["negative_count"],
            "linear_train_sampled_negative_samples": train_stats["sampled_negative_count"],
            "linear_test_samples": len(balanced_test),
            "linear_test_positive_samples": test_stats["positive_count"],
            "linear_test_negative_samples": test_stats["negative_count"],
            "linear_test_sampled_negative_samples": test_stats["sampled_negative_count"],
            "linear_positive_rate_train": float(train_stats["positive_count"] / max(train_stats["positive_count"] + train_stats["sampled_negative_count"], 1)),
            "linear_positive_rate_test": float(test_stats["positive_count"] / max(test_stats["positive_count"] + test_stats["sampled_negative_count"], 1)),
            "linear_min_train_per_class": min(train_targets.count(0), train_targets.count(1)),
            "linear_mean_train_per_class": float(np.mean([train_targets.count(0), train_targets.count(1)])),
            "linear_feature": metric,
            "linear_negative_ratio": float(negative_ratio),
            "linear_test_negative_ratio": "all" if test_negative_ratio is None else float(test_negative_ratio),
            "linear_positive_mode": positive_mode,
        }
    )
    return metrics


def evaluate_pairwise_kernel_svm(
    base_records: dict[str, DnaRecord],
    run_records: dict[str, dict[str, DnaRecord]],
    test_run_name: str,
    metric: str,
    relation_lookup: dict[str, set[str]],
    negative_ratio: float,
    test_negative_ratio: float | None = 1.0,
    kernel: str = "rbf",
    gamma: float | str | None = "scale",
    c: float = 1.0,
    positive_mode: str = "same_or_related",
) -> dict[str, float | int | str]:
    test_records = run_records[test_run_name]
    labels = sorted(set(base_records) & set(test_records))
    if len(labels) < 2:
        return {"kernel_status": "skipped_too_few_classes"}

    train_examples: list[PairExample] = []
    for run_name, records in run_records.items():
        if run_name == test_run_name:
            continue
        examples = build_pair_examples(base_records, records, metric, relation_lookup, positive_mode)
        train_examples.extend(examples)

    test_examples = build_pair_examples(base_records, test_records, metric, relation_lookup, positive_mode)
    if not train_examples or not test_examples:
        return {"kernel_status": "skipped_empty_pairs"}

    balanced_train, train_stats = balance_examples(train_examples, negative_ratio)
    if test_negative_ratio is None:
        balanced_test = list(test_examples)
        test_stats = {
            "positive_count": sum(example.target == 1 for example in test_examples),
            "negative_count": sum(example.target == 0 for example in test_examples),
            "sampled_negative_count": sum(example.target == 0 for example in test_examples),
        }
    else:
        balanced_test, test_stats = balance_examples(test_examples, negative_ratio=test_negative_ratio)

    train_targets = [example.target for example in balanced_train]
    test_targets = [example.target for example in balanced_test]
    if len(set(train_targets)) < 2 or len(set(test_targets)) < 2:
        return {"kernel_status": "skipped_too_few_binary_classes"}

    train_x = np.array([[-example.distance] for example in balanced_train], dtype=np.float32)
    test_x = np.array([[-example.distance] for example in balanced_test], dtype=np.float32)

    clf = SVC(C=c, kernel=kernel, gamma=gamma, class_weight="balanced", probability=False, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        clf.fit(train_x, train_targets)

    y_pred = list(clf.predict(test_x))
    decision_scores = list(clf.decision_function(test_x))
    metrics = {f"kernel_{key}": value for key, value in binary_metrics(test_targets, y_pred, decision_scores).items()}
    metrics.update(
        {
            "kernel_status": "ok",
            "kernel_train_samples": len(balanced_train),
            "kernel_train_positive_samples": train_stats["positive_count"],
            "kernel_train_negative_samples": train_stats["negative_count"],
            "kernel_train_sampled_negative_samples": train_stats["sampled_negative_count"],
            "kernel_test_samples": len(balanced_test),
            "kernel_test_positive_samples": test_stats["positive_count"],
            "kernel_test_negative_samples": test_stats["negative_count"],
            "kernel_test_sampled_negative_samples": test_stats["sampled_negative_count"],
            "kernel_positive_rate_train": float(train_stats["positive_count"] / max(train_stats["positive_count"] + train_stats["sampled_negative_count"], 1)),
            "kernel_positive_rate_test": float(test_stats["positive_count"] / max(test_stats["positive_count"] + test_stats["sampled_negative_count"], 1)),
            "kernel_min_train_per_class": min(train_targets.count(0), train_targets.count(1)),
            "kernel_mean_train_per_class": float(np.mean([train_targets.count(0), train_targets.count(1)])),
            "kernel_feature": metric,
            "kernel_negative_ratio": float(negative_ratio),
            "kernel_test_negative_ratio": "all" if test_negative_ratio is None else float(test_negative_ratio),
            "kernel_positive_mode": positive_mode,
            "kernel": kernel,
            "kernel_gamma": float(gamma) if isinstance(gamma, (int, float)) else str(gamma),
            "kernel_c": float(c),
        }
    )
    return metrics


def load_font(size: int):
    from PIL import ImageFont

    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def metric_display_name(metric: str) -> str:
    return {
        "top1_mean": "TOP1",
        "top3_mean": "TOP3",
        "top5_mean": "TOP5",
        "mrr_mean": "MRR",
        "linear_accuracy_mean": "LIN ACC",
        "linear_balanced_accuracy_mean": "LIN BAL",
        "kernel_accuracy_mean": "KER ACC",
        "kernel_balanced_accuracy_mean": "KER BAL",
    }.get(metric, metric)


def build_setting_grid(
    rows: list[dict[str, object]],
    metric: str,
) -> tuple[list[float], list[float], np.ndarray]:
    temperatures = sorted({float(row["temperature"]) for row in rows if row.get("temperature") is not None})
    top_ps = sorted({float(row["top_p"]) for row in rows if row.get("top_p") is not None})
    values = np.full((len(temperatures), len(top_ps)), np.nan, dtype=np.float32)
    temp_index = {temperature: idx for idx, temperature in enumerate(temperatures)}
    top_p_index = {top_p: idx for idx, top_p in enumerate(top_ps)}
    metric_key = metric if metric.endswith("_mean") else f"{metric}_mean"
    for row in rows:
        if row.get("temperature") is None or row.get("top_p") is None:
            continue
        if metric_key not in row or row[metric_key] is None:
            continue
        values[temp_index[float(row["temperature"])], top_p_index[float(row["top_p"])]] = float(row[metric_key])
    return temperatures, top_ps, values


def plot_setting_heatmap_grid(
    rows: list[dict[str, object]],
    metrics: list[str],
    output_path: Path,
    title: str,
) -> None:
    import html
    import subprocess

    metric_rows = []
    for metric in metrics:
        metric_key = metric if metric.endswith("_mean") else f"{metric}_mean"
        if any(metric_key in row for row in rows):
            metric_rows.append(metric)
    if not metric_rows:
        return

    temperatures = sorted({float(row["temperature"]) for row in rows if row.get("temperature") is not None})
    top_ps = sorted({float(row["top_p"]) for row in rows if row.get("top_p") is not None})
    if not temperatures or not top_ps:
        return

    cell = max(58, min(78, 360 // max(len(temperatures), len(top_ps), 1)))
    panel_w = 54 + len(top_ps) * cell + 24
    panel_h = 42 + len(temperatures) * cell + 48
    # A single landscape row makes cross-metric comparison easier in reports.
    cols = min(4, len(metric_rows))
    rows_count = int(math.ceil(len(metric_rows) / cols))
    left_margin = 48
    top_margin = 88
    gap_x = 28
    gap_y = 30
    width = left_margin * 2 + cols * panel_w + (cols - 1) * gap_x
    height = top_margin + rows_count * panel_h + (rows_count - 1) * gap_y + 50

    def esc(text: object) -> str:
        return html.escape(str(text), quote=True)

    def rgb(color: tuple[int, int, int]) -> str:
        return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

    def text(x: float, y: float, value: object, size: int = 14, fill: tuple[int, int, int] = (0, 0, 0), weight: str = "normal", anchor: str = "start") -> str:
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans, Liberation Sans, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{rgb(fill)}" text-anchor="{anchor}">{esc(value)}</text>'
        )

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        text(left_margin, 34, title, size=24, fill=(20, 20, 20), weight="bold"),
        text(left_margin, 58, "GREEN = HIGHER SCORE; EACH CELL IS ANNOTATED.", size=12, fill=(70, 70, 70)),
    ]

    for idx, metric in enumerate(metric_rows):
        row_idx = idx // cols
        col_idx = idx % cols
        left = left_margin + col_idx * (panel_w + gap_x)
        top = top_margin + row_idx * (panel_h + gap_y)
        right = left + panel_w
        bottom = top + panel_h
        _, _, values = build_setting_grid(rows, metric)
        parts.append(f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="#ffffff" stroke="#dcdcdc" stroke-width="2"/>')
        parts.append(text(left + 12, top + 22, metric_display_name(metric), size=16, fill=(25, 25, 25), weight="bold"))
        parts.append(text(left + 10, top + 36, "SCORE", size=12, fill=(70, 70, 70)))
        parts.append(text(right - 10, top + 36, "TOP_P", size=12, fill=(70, 70, 70), anchor="end"))
        parts.append(text(left + 10, bottom - 12, "TEMPERATURE", size=12, fill=(70, 70, 70)))

        inner_left = left + 54
        inner_top = top + 42
        inner_right = right - 24
        inner_bottom = bottom - 48
        grid_w = max(inner_right - inner_left, 1)
        grid_h = max(inner_bottom - inner_top, 1)
        cell_w = grid_w / max(len(top_ps), 1)
        cell_h = grid_h / max(len(temperatures), 1)

        for col, top_p in enumerate(top_ps):
            x0 = inner_left + col * cell_w
            x1 = inner_left + (col + 1) * cell_w
            parts.append(text((x0 + x1) / 2, top + 40, f"{top_p:g}", size=12, fill=(60, 60, 60), anchor="middle"))
        for row, temperature in enumerate(temperatures):
            y0 = inner_top + row * cell_h
            y1 = inner_top + (row + 1) * cell_h
            parts.append(text(left + 42, (y0 + y1) / 2 + 4, f"{temperature:g}", size=12, fill=(60, 60, 60), anchor="end"))

        finite = values[np.isfinite(values)]
        vmax = 1.0
        if finite.size:
            vmax = max(1.0, float(np.max(finite)))
        best_idx: tuple[int, int] | None = None
        if finite.size:
            best_idx = tuple(int(x) for x in np.unravel_index(int(np.nanargmax(values)), values.shape))

        for row_idx2, _temperature in enumerate(temperatures):
            for col_idx2, _top_p in enumerate(top_ps):
                value = float(values[row_idx2, col_idx2])
                x0 = inner_left + col_idx2 * cell_w
                y0 = inner_top + row_idx2 * cell_h
                x1 = inner_left + (col_idx2 + 1) * cell_w
                y1 = inner_top + (row_idx2 + 1) * cell_h
                fill = rgb(color_for_score(value, vmax=vmax))
                parts.append(
                    f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{(x1 - x0):.1f}" height="{(y1 - y0):.1f}" '
                    f'fill="{fill}" stroke="#ffffff" stroke-width="1"/>'
                )
                if best_idx is not None and (row_idx2, col_idx2) == best_idx:
                    parts.append(
                        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{(x1 - x0):.1f}" height="{(y1 - y0):.1f}" '
                        f'fill="none" stroke="#111111" stroke-width="3"/>'
                    )
                text_fill = (255, 255, 255) if sum(color_for_score(value, vmax=vmax)) < 360 else (20, 20, 20)
                cell_text = "NA" if not np.isfinite(value) else f"{value:.3f}"
                parts.append(text((x0 + x1) / 2, (y0 + y1) / 2 + 4, cell_text, size=12, fill=text_fill, anchor="middle"))

    parts.append("</svg>")
    svg_text = "".join(parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_path.with_suffix(".svg")
    svg_path.write_text(svg_text, encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(svg_path), str(output_path)],
        check=True,
    )
    try:
        svg_path.unlink()
    except FileNotFoundError:
        pass


def aggregate_by_temperature(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[float, list[dict[str, object]]] = {}
    for row in summary_rows:
        if row.get("temperature") is None:
            continue
        key = float(row["temperature"])
        groups.setdefault(key, []).append(row)

    metric_names = [
        "top1",
        "top3",
        "top5",
        "mrr",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "mean_self_distance",
        "mean_nearest_distance",
        "mean_nearest_margin",
        "linear_accuracy",
        "linear_balanced_accuracy",
        "kernel_accuracy",
        "kernel_balanced_accuracy",
        "linear_precision_macro",
        "linear_recall_macro",
        "linear_f1_macro",
        "linear_precision",
        "linear_recall",
        "linear_f1",
        "linear_specificity",
        "linear_mcc",
        "linear_roc_auc",
        "linear_average_precision",
    ]

    aggregate_rows: list[dict[str, object]] = []
    for temperature, rows in sorted(groups.items()):
        out: dict[str, object] = {
            "temperature": temperature,
            "repeat_count": len(rows),
            "mean_model_count": float(np.mean([float(row.get("model_count", 0.0)) for row in rows])),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in rows if metric in row and row[metric] is not None]
            if values:
                out[f"{metric}_mean"] = float(np.mean(values))
                out[f"{metric}_std"] = float(np.std(values))
        aggregate_rows.append(out)
    return aggregate_rows


def plot_metric_lines_by_top_p(
    rows: list[dict[str, object]],
    metrics: list[str],
    output_path: Path,
    title: str,
) -> None:
    """Plot temperature trends with one panel per top-p and labeled metric lines."""
    import html
    import subprocess

    if not rows:
        return
    top_ps = sorted({float(row["top_p"]) for row in rows if row.get("top_p") is not None})
    temperatures = sorted({float(row["temperature"]) for row in rows if row.get("temperature") is not None})
    if not top_ps or not temperatures:
        return

    labels = {
        "top1_mean": "Top-1",
        "top3_mean": "Top-3",
        "top5_mean": "Top-5",
        "mrr_mean": "MRR",
        "linear_accuracy_mean": "Linear SVM",
        "kernel_accuracy_mean": "RBF SVM",
    }
    colors = ["#b63a2b", "#2468a9", "#23864b", "#d28b16"]
    dashes = ["", "8 4", "3 3", "12 4 3 4"]
    width = 1420
    height = 570
    outer_left = 76
    outer_right = 34
    top = 118
    bottom = 72
    gap = 28
    panel_w = (width - outer_left - outer_right - gap * (len(top_ps) - 1)) / len(top_ps)
    panel_h = height - top - bottom
    x_min = min(temperatures)
    x_max = max(temperatures)
    x_span = max(x_max - x_min, 1e-9)

    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    def text(x: float, y: float, value: object, size: int = 14, fill: str = "#202020", weight: str = "normal", anchor: str = "start") -> str:
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans, Liberation Sans, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(value)}</text>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fbfaf7"/>',
        text(outer_left, 34, title, size=25, weight="bold"),
        text(outer_left, 58, "Common-model cohort; x = temperature; y = score", size=13, fill="#5b5b57"),
    ]

    legend_x = outer_left
    for idx, metric in enumerate(metrics):
        color = colors[idx % len(colors)]
        dash = dashes[idx % len(dashes)]
        x0 = legend_x + idx * 190
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{x0}" y1="86" x2="{x0 + 34}" y2="86" stroke="{color}" stroke-width="3"{dash_attr}/>')
        parts.append(f'<circle cx="{x0 + 17}" cy="86" r="4" fill="{color}"/>')
        parts.append(text(x0 + 44, 91, labels.get(metric, metric), size=13, weight="bold"))

    lookup = {
        (float(row["temperature"]), float(row["top_p"])): row
        for row in rows
        if row.get("temperature") is not None and row.get("top_p") is not None
    }
    for panel_idx, top_p in enumerate(top_ps):
        left = outer_left + panel_idx * (panel_w + gap)
        right = left + panel_w
        panel_bottom = top + panel_h
        parts.append(f'<rect x="{left:.1f}" y="{top}" width="{panel_w:.1f}" height="{panel_h}" fill="#ffffff" stroke="#d8d5ce" stroke-width="1.5"/>')
        parts.append(text(left + 12, top + 24, f"top_p = {top_p:g}", size=16, weight="bold"))
        plot_left = left + 48
        plot_right = right - 18
        plot_top = top + 38
        plot_bottom = panel_bottom - 42
        for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = plot_bottom - tick * (plot_bottom - plot_top)
            parts.append(f'<line x1="{plot_left:.1f}" y1="{y:.1f}" x2="{plot_right:.1f}" y2="{y:.1f}" stroke="#e8e5df" stroke-width="1"/>')
            if panel_idx == 0:
                parts.append(text(plot_left - 9, y + 5, f"{tick:.2f}", size=12, fill="#66645f", anchor="end"))
        for temperature in temperatures:
            x = plot_left + (temperature - x_min) / x_span * (plot_right - plot_left)
            parts.append(f'<line x1="{x:.1f}" y1="{plot_bottom:.1f}" x2="{x:.1f}" y2="{plot_bottom + 5:.1f}" stroke="#77736c"/>')
            parts.append(text(x, plot_bottom + 22, f"{temperature:g}", size=12, fill="#66645f", anchor="middle"))

        for metric_idx, metric in enumerate(metrics):
            color = colors[metric_idx % len(colors)]
            dash = dashes[metric_idx % len(dashes)]
            segments: list[list[tuple[float, float]]] = []
            current_segment: list[tuple[float, float]] = []
            for temperature in temperatures:
                row = lookup.get((temperature, top_p))
                value = row.get(metric) if row else None
                if value is None:
                    if current_segment:
                        segments.append(current_segment)
                        current_segment = []
                    continue
                score = min(max(float(value), 0.0), 1.0)
                x = plot_left + (temperature - x_min) / x_span * (plot_right - plot_left)
                y = plot_bottom - score * (plot_bottom - plot_top)
                current_segment.append((x, y))
            if current_segment:
                segments.append(current_segment)
            if not segments:
                continue
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            for segment in segments:
                if len(segment) > 1:
                    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in segment)
                    parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"{dash_attr}/>')
                for x, y in segment:
                    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#ffffff" stroke="{color}" stroke-width="3"/>')

        parts.append(text((plot_left + plot_right) / 2, panel_bottom - 8, "temperature", size=12, fill="#55534f", anchor="middle"))

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_path.with_suffix(".svg")
    svg_path.write_text("".join(parts), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(svg_path), str(output_path)],
        check=True,
    )
    svg_path.unlink(missing_ok=True)


def plot_faceted_response_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    plot_metric_lines_by_top_p(
        rows,
        ["top1_mean", "top3_mean", "top5_mean", "mrr_mean"],
        output_path,
        "Exact-model retrieval across decoding settings",
    )


def plot_probe_metric_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    plot_metric_lines_by_top_p(
        rows,
        ["linear_accuracy_mean", "kernel_accuracy_mean"],
        output_path,
        "Pairwise probe accuracy across decoding settings",
    )


def plot_metric_curves(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    if not summary_rows:
        return

    has_pre_aggregated_metrics = any(
        any(key in row for key in ["top1_mean", "top3_mean", "top5_mean", "mrr_mean"]) for row in summary_rows
    )
    plot_rows = summary_rows if has_pre_aggregated_metrics else aggregate_by_setting(summary_rows)

    plot_faceted_response_summary(plot_rows, output_path)
    probe_summary_path = output_path.with_name(output_path.stem + "_probe_metrics.png")
    plot_probe_metric_summary(plot_rows, probe_summary_path)


def aggregate_by_setting(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[float, float], list[dict[str, object]]] = {}
    for row in summary_rows:
        if row.get("temperature") is None or row.get("top_p") is None:
            continue
        key = (float(row["temperature"]), float(row["top_p"]))
        groups.setdefault(key, []).append(row)

    metric_names = [
        "top1",
        "top3",
        "top5",
        "mrr",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "mean_self_distance",
        "mean_nearest_distance",
        "mean_nearest_margin",
        "linear_accuracy",
        "linear_balanced_accuracy",
        "kernel_accuracy",
        "kernel_balanced_accuracy",
        "linear_precision",
        "linear_recall",
        "linear_f1",
        "linear_specificity",
        "linear_mcc",
        "linear_roc_auc",
        "linear_average_precision",
    ]
    aggregate_rows: list[dict[str, object]] = []
    for (temperature, top_p), rows in sorted(groups.items()):
        out: dict[str, object] = {
            "temperature": temperature,
            "top_p": top_p,
            "repeat_count": len(rows),
            "mean_model_count": float(np.mean([float(row.get("model_count", 0.0)) for row in rows])),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in rows if metric in row and row[metric] is not None]
            if values:
                out[f"{metric}_mean"] = float(np.mean(values))
                out[f"{metric}_std"] = float(np.std(values))
        aggregate_rows.append(out)
    return aggregate_rows


def build_best_setting_summary(aggregate_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not aggregate_rows:
        return []

    def score(row: dict[str, object]) -> tuple[float, float, float, float]:
        return (
            float(row.get("top1_mean", -1.0)),
            float(row.get("mrr_mean", -1.0)),
            float(row.get("linear_accuracy_mean", -1.0)),
            float(row.get("top3_mean", -1.0)),
        )

    best = max(aggregate_rows, key=score)
    return [
        {
            "temperature": float(best["temperature"]),
            "top_p": float(best["top_p"]),
            "repeat_count": int(best.get("repeat_count", 0)),
            "mean_model_count": float(best.get("mean_model_count", float("nan"))),
            "top1_mean": float(best.get("top1_mean", float("nan"))),
            "top3_mean": float(best.get("top3_mean", float("nan"))),
            "top5_mean": float(best.get("top5_mean", float("nan"))),
            "mrr_mean": float(best.get("mrr_mean", float("nan"))),
            "linear_accuracy_mean": float(best.get("linear_accuracy_mean", float("nan"))),
            "linear_balanced_accuracy_mean": float(best.get("linear_balanced_accuracy_mean", float("nan"))),
            "kernel_accuracy_mean": float(best.get("kernel_accuracy_mean", float("nan"))),
            "kernel_balanced_accuracy_mean": float(best.get("kernel_balanced_accuracy_mean", float("nan"))),
        }
    ]


def color_for_score(value: float, vmin: float = 0.0, vmax: float = 1.0) -> tuple[int, int, int]:
    if not np.isfinite(value):
        return (220, 220, 220)
    t = max(0.0, min(1.0, (value - vmin) / max(vmax - vmin, 1e-12)))
    # Red -> yellow -> green.
    if t < 0.5:
        u = t / 0.5
        return (int(190 + u * 55), int(45 + u * 175), 55)
    u = (t - 0.5) / 0.5
    return (int(245 - u * 185), int(220 - u * 60), int(55 + u * 55))


def plot_setting_heatmap(
    aggregate_rows: list[dict[str, object]],
    metric: str,
    output_path: Path,
) -> None:
    plot_setting_heatmap_grid(
        aggregate_rows,
        [metric],
        output_path,
        f"{metric_display_name(metric)} MEAN BY TEMPERATURE AND TOP_P",
    )


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
    parser = argparse.ArgumentParser(description="Evaluate DNA model-identification robustness across run suffixes.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("out/rand_chinese"))
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec name:suffix:temperature:top_p. Repeat for each generation setting.",
    )
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--top-k", type=int, action="append", default=[1, 3, 5])
    parser.add_argument("--output-dir", type=Path, default=Path("out/dna_probe_sweep"))
    parser.add_argument("--prefix", type=str, default="rand_chinese_probe_sweep")
    parser.add_argument(
        "--linear-svm",
        action="store_true",
        help="Also evaluate a balanced pairwise LinearSVC probe on positive/negative model pairs.",
    )
    parser.add_argument(
        "--kernel-svm",
        action="store_true",
        help="Also evaluate a balanced pairwise SVC probe with an RBF kernel on positive/negative model pairs.",
    )
    parser.add_argument(
        "--positive-relations-file",
        type=Path,
        default=None,
        help="Optional JSON/JSONL/CSV/TSV file listing related models that should count as positive pairs.",
    )
    parser.add_argument(
        "--positive-mode",
        choices=["same_or_related", "exact_only", "lineage_only"],
        default="same_or_related",
        help="Binary verification target. Related pairs are excluded from exact_only; diagonals are excluded from lineage_only.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Negative samples to keep per positive pair when training the pairwise SVM.",
    )
    parser.add_argument(
        "--test-negative-ratio",
        type=float,
        default=0.0,
        help="Negatives per positive in SVM testing; 0 (default) evaluates every natural-prevalence pair.",
    )
    parser.add_argument(
        "--common-models-only",
        action="store_true",
        help="Restrict every included run and the reference gallery to one shared model cohort.",
    )
    parser.add_argument(
        "--minimum-run-models",
        type=int,
        default=2,
        help="With --common-models-only, exclude runs below this reference-intersection size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_specs = [parse_run_spec(raw) for raw in args.run]
    if not run_specs:
        raise SystemExit("At least one --run is required.")

    base_records = load_records_for_suffix(args.dataset_dir, "")
    if not base_records:
        raise SystemExit(f"No base DNA records found under {args.dataset_dir}.")

    relation_lookup = build_relation_lookup(load_relation_groups(args.positive_relations_file))
    run_records = {spec.name: load_records_for_suffix(args.dataset_dir, spec.suffix) for spec in run_specs}
    available_model_counts = {
        run_name: len(set(base_records) & set(records))
        for run_name, records in run_records.items()
    }
    cohort_metadata: dict[str, object] | None = None
    if args.common_models_only:
        try:
            base_records, run_records, cohort_metadata = select_common_model_cohort(
                base_records,
                run_records,
                args.minimum_run_models,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        included_runs = set(run_records)
        run_specs = [spec for spec in run_specs if spec.name in included_runs]
    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for spec in run_specs:
        records = run_records[spec.name]
        metrics, rows = evaluate_nearest(base_records, records, args.metric, args.top_k)
        summary = {
            "run": spec.name,
            "suffix": spec.suffix,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "metric": args.metric,
            "available_model_count": available_model_counts[spec.name],
            "common_cohort": bool(args.common_models_only),
            **metrics,
        }
        if args.linear_svm:
            linear = evaluate_pairwise_svm(
                base_records,
                run_records,
                spec.name,
                args.metric,
                relation_lookup,
                args.negative_ratio,
                None if args.test_negative_ratio == 0 else args.test_negative_ratio,
                args.positive_mode,
            )
            summary.update(linear)
        if args.kernel_svm:
            kernel = evaluate_pairwise_kernel_svm(
                base_records,
                run_records,
                spec.name,
                args.metric,
                relation_lookup,
                args.negative_ratio,
                None if args.test_negative_ratio == 0 else args.test_negative_ratio,
                positive_mode=args.positive_mode,
            )
            summary.update(kernel)
        summary_rows.append(summary)
        for row in rows:
            prediction_rows.append({"run": spec.name, "suffix": spec.suffix, "temperature": spec.temperature, "top_p": spec.top_p, **row})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / f"{args.prefix}_summary.csv"
    predictions_csv = args.output_dir / f"{args.prefix}_predictions.csv"
    summary_json = args.output_dir / f"{args.prefix}_summary.json"
    temperature_csv = args.output_dir / f"{args.prefix}_aggregate_by_temperature.csv"
    temperature_json = args.output_dir / f"{args.prefix}_aggregate_by_temperature.json"
    aggregate_csv = args.output_dir / f"{args.prefix}_aggregate_by_setting.csv"
    aggregate_json = args.output_dir / f"{args.prefix}_aggregate_by_setting.json"
    curve_png = args.output_dir / f"{args.prefix}_curves.png"
    best_setting_csv = args.output_dir / f"{args.prefix}_best_setting.csv"
    best_setting_json = args.output_dir / f"{args.prefix}_best_setting.json"
    common_models_json = args.output_dir / f"{args.prefix}_common_models.json"
    common_models_csv = args.output_dir / f"{args.prefix}_common_models.csv"

    temperature_rows = aggregate_by_temperature(summary_rows)
    aggregate_rows = aggregate_by_setting(summary_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(predictions_csv, prediction_rows)
    write_csv(temperature_csv, temperature_rows)
    write_csv(aggregate_csv, aggregate_rows)
    best_setting_rows = build_best_setting_summary(aggregate_rows)
    write_csv(best_setting_csv, best_setting_rows)
    summary_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    temperature_json.write_text(json.dumps(temperature_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    aggregate_json.write_text(json.dumps(aggregate_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    best_setting_json.write_text(json.dumps(best_setting_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if cohort_metadata is not None:
        common_models_json.write_text(json.dumps(cohort_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        write_csv(
            common_models_csv,
            [{"model_name": model} for model in cohort_metadata["common_models"]],
        )
    line_paths: dict[str, str] = {}
    try:
        plot_metric_curves(summary_rows, curve_png)
        for metric in ["top1", "top3", "top5", "mrr", "linear_accuracy", "linear_balanced_accuracy", "kernel_accuracy", "kernel_balanced_accuracy"]:
            metric_output = args.output_dir / f"{args.prefix}_{metric}_setting_lines.png"
            metric_key = f"{metric}_mean"
            plot_metric_lines_by_top_p(
                aggregate_rows,
                [metric_key],
                metric_output,
                f"{metric_display_name(metric_key)} across decoding settings",
            )
            if metric_output.exists():
                line_paths[metric] = str(metric_output)
    except ImportError as exc:
        print(f"[warn] plotting skipped because an image dependency is unavailable: {exc}")

    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv),
                "predictions_csv": str(predictions_csv),
                "summary_json": str(summary_json),
                "temperature_csv": str(temperature_csv),
                "temperature_json": str(temperature_json),
                "aggregate_csv": str(aggregate_csv),
                "aggregate_json": str(aggregate_json),
                "best_setting_csv": str(best_setting_csv),
                "best_setting_json": str(best_setting_json),
                "common_models_json": str(common_models_json) if cohort_metadata is not None else None,
                "common_models_csv": str(common_models_csv) if cohort_metadata is not None else None,
                "curve_png": str(curve_png),
                "setting_line_plots": line_paths,
                "relations_file": str(args.positive_relations_file) if args.positive_relations_file else None,
                "cohort": cohort_metadata,
                "runs": summary_rows,
                "aggregate_by_setting": aggregate_rows,
                "best_setting": best_setting_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
