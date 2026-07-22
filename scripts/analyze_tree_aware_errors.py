#!/usr/bin/env python3
"""Analyze exact-classification errors and DNA distance by model-tree distance."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np

from evaluate_exact_dna_classification import (
    cosine_scores,
    load_records_for_suffix,
    parse_run_spec,
    select_common_cohort,
)


def load_components(path: Path | None) -> dict[str, int]:
    lookup: dict[str, int] = {}
    if path is None:
        return lookup
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        for model in row.get("models", []):
            lookup[str(model)] = index
    return lookup


def load_graph(path: Path | None) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    if path is None or not path.exists():
        return graph
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        left = str(row.get("source", "")).strip()
        right = str(row.get("target", "")).strip()
        if left and right:
            graph.setdefault(left, set()).add(right)
            graph.setdefault(right, set()).add(left)
    return graph


def shortest_paths(graph: dict[str, set[str]], models: list[str]) -> dict[tuple[str, str], int]:
    allowed = set(models)
    distances: dict[tuple[str, str], int] = {}
    for source in models:
        queue = deque([(source, 0)])
        seen = {source}
        while queue:
            node, distance = queue.popleft()
            if node in allowed:
                distances[(source, node)] = distance
            for neighbor in graph.get(node, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, distance + 1))
    return distances


def relation_bucket(
    left: str,
    right: str,
    paths: dict[tuple[str, str], int],
    components: dict[str, int],
) -> str:
    if left == right:
        return "exact"
    distance = paths.get((left, right))
    if distance == 1:
        return "direct_1hop"
    if distance == 2:
        return "indirect_2hop"
    if distance is not None:
        return "indirect_3plus"
    if left in components and components.get(left) == components.get(right):
        return "same_component_unknown_hops"
    return "disconnected"


def analyze_run(
    labels: list[str],
    reference_matrix: np.ndarray,
    query_matrix: np.ndarray,
    paths: dict[tuple[str, str], int],
    components: dict[str, int],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    scores = cosine_scores(query_matrix, reference_matrix)
    distances = 1.0 - scores
    error_counts: Counter[str] = Counter()
    predictions: list[dict[str, object]] = []
    pair_values: dict[str, list[float]] = {}
    for row_index, truth in enumerate(labels):
        prediction = labels[int(np.argmax(scores[row_index]))]
        bucket = relation_bucket(truth, prediction, paths, components)
        error_counts[bucket] += 1
        predictions.append({"model": truth, "predicted_model": prediction, "relation_bucket": bucket})
        for column_index, candidate in enumerate(labels):
            pair_bucket = relation_bucket(truth, candidate, paths, components)
            pair_values.setdefault(pair_bucket, []).append(float(distances[row_index, column_index]))

    count = len(labels)
    exact = error_counts["exact"]
    direct = error_counts["direct_1hop"]
    two_hop = error_counts["indirect_2hop"]
    component = error_counts["indirect_3plus"] + error_counts["same_component_unknown_hops"]
    summary: dict[str, object] = {
        "model_count": count,
        "exact_top1": exact / count,
        "within_1hop_top1": (exact + direct) / count,
        "within_2hop_top1": (exact + direct + two_hop) / count,
        "same_component_top1": (exact + direct + two_hop + component) / count,
        **{f"top1_error_{key}": value for key, value in sorted(error_counts.items())},
    }
    distance_rows = [
        {
            "relation_bucket": bucket,
            "ordered_pair_count": len(values),
            "mean_cosine_distance": float(np.mean(values)),
            "median_cosine_distance": float(np.median(values)),
            "q25_cosine_distance": float(np.quantile(values, 0.25)),
            "q75_cosine_distance": float(np.quantile(values, 0.75)),
        }
        for bucket, values in sorted(pair_values.items())
    ]
    return summary, predictions, distance_rows


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
    parser = argparse.ArgumentParser(description="Tree-distance and hierarchical error analysis.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("out/rand_chinese"))
    parser.add_argument("--reference-suffix", required=True)
    parser.add_argument("--run", action="append", default=[], help="name:suffix:temperature:top_p:repeat")
    parser.add_argument("--components-file", type=Path, default=Path("out/hf_model_tree/model_relations.jsonl"))
    parser.add_argument("--edges-file", type=Path, default=Path("out/hf_model_tree/model_direct_edges.jsonl"))
    parser.add_argument("--minimum-models", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("out/tree_aware_errors"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = [parse_run_spec(value) for value in args.run]
    if not runs:
        raise SystemExit("At least one --run is required.")
    reference = load_records_for_suffix(args.dataset_dir, args.reference_suffix)
    records = {run.name: load_records_for_suffix(args.dataset_dir, run.suffix) for run in runs}
    all_records = {"reference": reference, **records}
    labels = select_common_cohort(all_records, all_records)
    if len(labels) < args.minimum_models:
        raise SystemExit(
            f"Only {len(labels)} common models are available; minimum is {args.minimum_models}."
        )
    components = load_components(args.components_file)
    graph = load_graph(args.edges_file)
    paths = shortest_paths(graph, labels)
    reference_matrix = np.stack([reference[label].vector for label in labels])

    summaries: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    for run in runs:
        query = np.stack([records[run.name][label].vector for label in labels])
        summary, run_predictions, run_distances = analyze_run(
            labels, reference_matrix, query, paths, components
        )
        identity = {
            "run": run.name,
            "temperature": run.temperature,
            "top_p": run.top_p,
            "repeat": run.repeat,
        }
        summaries.append({**identity, **summary})
        predictions.extend({**identity, **row} for row in run_predictions)
        distance_rows.extend({**identity, **row} for row in run_distances)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tree_aware_summary.csv", summaries)
    write_csv(args.output_dir / "tree_aware_predictions.csv", predictions)
    write_csv(args.output_dir / "tree_distance_distributions.csv", distance_rows)
    payload = {
        "cohort_size": len(labels),
        "direct_edges_available": bool(graph),
        "warning": None if graph else "Direct-edge export unavailable; same-component hop distances are unknown.",
        "summaries": summaries,
        "distance_distributions": distance_rows,
    }
    (args.output_dir / "tree_aware_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "cohort_size": len(labels)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
