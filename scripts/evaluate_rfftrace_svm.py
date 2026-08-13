#!/usr/bin/env python3
"""Evaluate exported RFFTrace/DistDNA vectors with held-out multiclass SVMs.

The caller must choose disjoint train and test matrix keys from
``rfftrace_vectors.npz``. At least two independent training vectors per model
are required by default; this prevents a one-example-per-class probe from being
mistaken for a validated DistDNA+SVM result.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_matrices(
    path: Path, train_keys: list[str], test_keys: list[str]
) -> tuple[list[str], dict[str, np.ndarray]]:
    if set(train_keys) & set(test_keys):
        raise ValueError("Train and test vector keys must be disjoint")
    with np.load(path) as payload:
        if "models" not in payload:
            raise ValueError(f"{path} has no models array")
        models = [str(value) for value in payload["models"].tolist()]
        matrices: dict[str, np.ndarray] = {}
        for key in [*train_keys, *test_keys]:
            if key not in payload:
                raise ValueError(f"Vector key {key!r} is absent from {path}")
            matrix = np.asarray(payload[key], dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != len(models):
                raise ValueError(
                    f"{key!r} must have shape ({len(models)}, features), got {matrix.shape}"
                )
            matrices[key] = matrix
    dimensions = {matrix.shape[1] for matrix in matrices.values()}
    if len(dimensions) != 1:
        raise ValueError(f"Vector dimensions differ across selected keys: {dimensions}")
    return models, matrices


def rank_from_scores(scores: np.ndarray, classes: np.ndarray, true_label: str) -> int:
    values = np.asarray(scores)
    if values.ndim == 0 and len(classes) == 2:
        values = np.asarray([-float(values), float(values)])
    if values.shape != (len(classes),):
        raise ValueError(
            f"Expected one decision score per class, got {values.shape} for {len(classes)} classes"
        )
    order = np.argsort(-values, kind="stable")
    ranked = np.asarray(classes)[order]
    return int(np.flatnonzero(ranked == true_label)[0]) + 1


def evaluate_classifier(
    name: str,
    classifier: object,
    models: list[str],
    matrices: dict[str, np.ndarray],
    train_keys: list[str],
    test_keys: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_x = np.concatenate([matrices[key] for key in train_keys], axis=0)
    train_y = np.tile(np.asarray(models), len(train_keys))
    pipeline = make_pipeline(StandardScaler(), classifier)
    pipeline.fit(train_x, train_y)
    estimator = pipeline[-1]

    predictions: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for key in test_keys:
        matrix = matrices[key]
        predicted = pipeline.predict(matrix)
        scores = pipeline.decision_function(matrix)
        ranks = [
            rank_from_scores(scores[index], estimator.classes_, model)
            for index, model in enumerate(models)
        ]
        for index, model in enumerate(models):
            predictions.append(
                {
                    "classifier": name,
                    "vector_key": key,
                    "model_id": model,
                    "predicted_model": str(predicted[index]),
                    "rank": ranks[index],
                    "correct": int(predicted[index] == model),
                }
            )
        summaries.append(
            {
                "classifier": name,
                "vector_key": key,
                "model_count": len(models),
                "train_vectors_per_model": len(train_keys),
                "top1": float(np.mean(predicted == np.asarray(models))),
                "top3": float(np.mean(np.asarray(ranks) <= 3)),
                "top5": float(np.mean(np.asarray(ranks) <= 5)),
                "mrr": float(np.mean(1.0 / np.asarray(ranks))),
            }
        )
    return summaries, predictions


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Held-out linear/RBF SVM evaluation for exported DistDNA vectors"
    )
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-key", action="append", required=True)
    parser.add_argument("--test-key", action="append", required=True)
    parser.add_argument("--minimum-train-vectors-per-class", type=int, default=2)
    parser.add_argument("--linear-c", type=float, default=1.0)
    parser.add_argument("--rbf-c", type=float, default=1.0)
    parser.add_argument("--rbf-gamma", type=parse_gamma, default="scale")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_gamma(raw: str) -> str | float:
    if raw in {"scale", "auto"}:
        return raw
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("RBF gamma must be scale, auto, or positive") from exc
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("RBF gamma must be scale, auto, or positive")
    return value


def main() -> int:
    args = parse_args()
    if len(args.train_key) < args.minimum_train_vectors_per_class:
        raise SystemExit(
            f"Need at least {args.minimum_train_vectors_per_class} independent train "
            f"vectors per class, but only {len(args.train_key)} --train-key values were given"
        )
    if args.linear_c <= 0 or args.rbf_c <= 0:
        raise SystemExit("SVM C values must be positive")

    from sklearn.svm import LinearSVC, SVC

    models, matrices = load_matrices(args.vectors, args.train_key, args.test_key)
    classifiers = [
        ("linear_svm", LinearSVC(C=args.linear_c, dual="auto", random_state=args.seed)),
        (
            "rbf_svm",
            SVC(
                C=args.rbf_c,
                gamma=args.rbf_gamma,
                kernel="rbf",
                decision_function_shape="ovr",
            ),
        ),
    ]
    summaries: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for name, classifier in classifiers:
        method_summary, method_predictions = evaluate_classifier(
            name,
            classifier,
            models,
            matrices,
            args.train_key,
            args.test_key,
        )
        summaries.extend(method_summary)
        predictions.extend(method_predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "predictions.csv", predictions)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "vectors": str(args.vectors),
                "train_keys": args.train_key,
                "test_keys": args.test_key,
                "minimum_train_vectors_per_class": args.minimum_train_vectors_per_class,
                "linear_c": args.linear_c,
                "rbf_c": args.rbf_c,
                "rbf_gamma": args.rbf_gamma,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
