#!/usr/bin/env python3
"""Visualize base-vs-rerun DNA distances as a heatmap.

The heatmap rows are rerun DNAs and columns are original/base DNAs. The diagonal
cell is the distance between the same model's original DNA and rerun DNA.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class DnaRecord:
    model_name: str
    safe_name: str
    path: Path
    vector: np.ndarray
    params_b: float | None


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip("/"))


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def load_dna(path: Path) -> DnaRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    model_name = metadata.get("model_name") or path.parent.name
    model_meta = metadata.get("model_metadata", {})
    params_b = model_meta.get("size", {}).get("parameter_count_billions")
    try:
        params_b = float(params_b) if params_b is not None else None
    except (TypeError, ValueError):
        params_b = None
    vector = np.asarray(payload["signature"], dtype=np.float32)
    return DnaRecord(
        model_name=str(model_name),
        safe_name=safe_model_name(str(model_name)),
        path=path,
        vector=vector,
        params_b=params_b,
    )


def find_rerun_records(dataset_dir: Path, suffix: str) -> list[DnaRecord]:
    records: list[DnaRecord] = []
    for model_dir in sorted(dataset_dir.glob(f"*{suffix}")):
        if not model_dir.is_dir():
            continue
        dna_files = sorted(model_dir.glob("*_dna.json"))
        if not dna_files:
            continue
        records.append(load_dna(dna_files[0]))
    return records


def find_base_record(dataset_dir: Path, rerun: DnaRecord, suffix: str) -> DnaRecord | None:
    if not rerun.path.parent.name.endswith(suffix):
        return None
    base_dir_name = rerun.path.parent.name[: -len(suffix)]
    base_dir = dataset_dir / base_dir_name
    dna_files = sorted(base_dir.glob("*_dna.json"))
    if not dna_files:
        return None
    return load_dna(dna_files[0])


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def pair_records(dataset_dir: Path, suffix: str) -> tuple[list[DnaRecord], list[DnaRecord]]:
    reruns = find_rerun_records(dataset_dir, suffix)
    pairs: list[tuple[DnaRecord, DnaRecord]] = []
    for rerun in reruns:
        base = find_base_record(dataset_dir, rerun, suffix)
        if base is None:
            continue
        if base.vector.shape != rerun.vector.shape:
            continue
        pairs.append((base, rerun))

    pairs.sort(key=lambda pair: (math.inf if pair[0].params_b is None else pair[0].params_b, pair[0].model_name.lower()))
    bases = [base for base, _ in pairs]
    paired_reruns = [rerun for _, rerun in pairs]
    return bases, paired_reruns


def compute_distance_matrix(
    bases: list[DnaRecord],
    reruns: list[DnaRecord],
    metric: str,
) -> np.ndarray:
    fn = cosine_distance if metric == "cosine" else euclidean_distance
    matrix = np.zeros((len(reruns), len(bases)), dtype=np.float32)
    for row, rerun in enumerate(reruns):
        for col, base in enumerate(bases):
            matrix[row, col] = fn(rerun.vector, base.vector)
    return matrix


def short_label(name: str, max_len: int = 28) -> str:
    label = name.replace("/", "/\n", 1)
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "…"


def color_for_value(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    if not np.isfinite(value):
        return (220, 220, 220)
    t = 0.0 if vmax <= vmin else (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    # Low distance: deep blue. High distance: warm yellow/red.
    stops = [
        (0.00, (36, 74, 143)),
        (0.35, (70, 150, 190)),
        (0.60, (247, 247, 190)),
        (0.80, (245, 150, 85)),
        (1.00, (165, 0, 38)),
    ]
    for idx in range(1, len(stops)):
        left_t, left_c = stops[idx - 1]
        right_t, right_c = stops[idx]
        if t <= right_t:
            local = (t - left_t) / (right_t - left_t)
            return tuple(int(left_c[i] + local * (right_c[i] - left_c[i])) for i in range(3))
    return stops[-1][1]


def draw_rotated_text(base: Image.Image, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
    lines = text.splitlines()
    line_heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines]
    width = max(font.getlength(line) for line in lines) if lines else 1
    height = sum(line_heights) + max(len(lines) - 1, 0) * 2
    patch = Image.new("RGBA", (int(width) + 8, int(height) + 8), (255, 255, 255, 0))
    draw = ImageDraw.Draw(patch)
    y = 4
    for line, line_height in zip(lines, line_heights):
        draw.text((4, y), line, font=font, fill=fill)
        y += line_height + 2
    rotated = patch.rotate(60, expand=True)
    base.alpha_composite(rotated, xy)


def draw_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str,
    metric: str,
    annotate: bool,
) -> None:
    n = len(labels)
    cell = max(18, min(34, 800 // max(n, 1)))
    left = 300
    top = 260
    right = 120
    bottom = 120
    width = left + n * cell + right
    height = top + n * cell + bottom

    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(22)
    label_font = load_font(10 if n > 30 else 12)
    small_font = load_font(9)

    finite = matrix[np.isfinite(matrix)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.percentile(finite, 95)) if finite.size else 1.0
    vmax = max(vmax, vmin + 1e-8)

    draw.text((24, 24), title, font=title_font, fill=(25, 25, 25))
    draw.text((24, 58), f"Rows: rerun DNA. Columns: original DNA. Metric: {metric}. Color clipped at p95={vmax:.4g}.", font=small_font, fill=(70, 70, 70))

    for row in range(n):
        for col in range(n):
            value = float(matrix[row, col])
            x0 = left + col * cell
            y0 = top + row * cell
            fill = color_for_value(value, vmin, vmax)
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=fill)
            if row == col:
                draw.rectangle((x0, y0, x0 + cell, y0 + cell), outline=(0, 0, 0), width=2)
            if annotate and cell >= 24:
                text = f"{value:.2f}" if metric == "cosine" else f"{value:.1f}"
                text_fill = (255, 255, 255) if sum(fill) < 330 else (20, 20, 20)
                bbox = draw.textbbox((0, 0), text, font=small_font)
                draw.text((x0 + (cell - (bbox[2] - bbox[0])) / 2, y0 + (cell - (bbox[3] - bbox[1])) / 2), text, font=small_font, fill=text_fill)

    for idx, label in enumerate(labels):
        y = top + idx * cell + cell / 2
        draw.text((left - 8, y), short_label(label), anchor="rm", font=label_font, fill=(20, 20, 20))
        draw_rotated_text(image, (left + idx * cell - 8, top - 160), short_label(label), label_font, (20, 20, 20))

    # Color bar.
    bar_x = left + n * cell + 30
    bar_y = top
    bar_h = n * cell
    for i in range(bar_h):
        value = vmin + (vmax - vmin) * (i / max(bar_h - 1, 1))
        draw.line((bar_x, bar_y + i, bar_x + 18, bar_y + i), fill=color_for_value(value, vmin, vmax))
    draw.rectangle((bar_x, bar_y, bar_x + 18, bar_y + bar_h), outline=(40, 40, 40))
    draw.text((bar_x + 26, bar_y - 4), f"{vmin:.3g}", font=small_font, fill=(40, 40, 40))
    draw.text((bar_x + 26, bar_y + bar_h - 8), f"{vmax:.3g}", font=small_font, fill=(40, 40, 40))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def write_matrix_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rerun\\base", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{float(value):.8g}" for value in row]])


def write_predictions(path: Path, matrix: np.ndarray, labels: list[str]) -> dict[str, float | int]:
    true = labels
    pred = [labels[int(np.argmin(row))] for row in matrix]
    nearest_distance = [float(np.min(row)) for row in matrix]
    self_distance = [float(row[idx]) for idx, row in enumerate(matrix)]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["true_model", "predicted_model", "correct", "self_distance", "nearest_distance", "nearest_rank"],
        )
        writer.writeheader()
        for idx, label in enumerate(labels):
            order = list(np.argsort(matrix[idx]))
            writer.writerow(
                {
                    "true_model": label,
                    "predicted_model": pred[idx],
                    "correct": pred[idx] == label,
                    "self_distance": f"{self_distance[idx]:.8g}",
                    "nearest_distance": f"{nearest_distance[idx]:.8g}",
                    "nearest_rank": order.index(idx) + 1,
                }
            )

    correct = sum(t == p for t, p in zip(true, pred))
    per_class_precision: list[float] = []
    per_class_recall: list[float] = []
    per_class_f1: list[float] = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(true, pred))
        fp = sum(t != label and p == label for t, p in zip(true, pred))
        fn = sum(t == label and p != label for t, p in zip(true, pred))
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
        per_class_precision.append(float(precision))
        per_class_recall.append(float(recall))
        per_class_f1.append(float(f1))

    accuracy = correct / len(labels) if labels else 0.0
    return {
        "model_count": len(labels),
        "accuracy": float(accuracy),
        "precision_macro": float(np.mean(per_class_precision)),
        "recall_macro": float(np.mean(per_class_recall)),
        "f1_macro": float(np.mean(per_class_f1)),
        "precision_micro": float(accuracy),
        "recall_micro": float(accuracy),
        "f1_micro": float(accuracy),
        "mean_self_distance": float(np.mean(self_distance)),
        "median_self_distance": float(np.median(self_distance)),
        "mean_nearest_distance": float(np.mean(nearest_distance)),
        "self_is_nearest_count": int(correct),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a base-vs-rerun DNA distance heatmap.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("out/rand_chinese"))
    parser.add_argument("--suffix", type=str, default="_temp07_run")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--output-dir", type=Path, default=Path("out/dna_rerun_heatmaps"))
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--annotate", action="store_true", help="Draw numeric distances inside cells when the heatmap is small enough.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bases, reruns = pair_records(args.dataset_dir, args.suffix)
    if not bases:
        raise SystemExit(f"No paired base/rerun DNA files found under {args.dataset_dir} for suffix {args.suffix!r}.")

    labels = [record.model_name for record in bases]
    matrix = compute_distance_matrix(bases, reruns, args.metric)
    prefix = args.prefix or f"{args.dataset_dir.name}{args.suffix}_{args.metric}"

    heatmap_path = args.output_dir / f"{prefix}_heatmap.png"
    matrix_path = args.output_dir / f"{prefix}_matrix.csv"
    predictions_path = args.output_dir / f"{prefix}_nearest_predictions.csv"
    metrics_path = args.output_dir / f"{prefix}_metrics.json"

    write_matrix_csv(matrix_path, matrix, labels)
    metrics = write_predictions(predictions_path, matrix, labels)
    metrics.update(
        {
            "dataset_dir": str(args.dataset_dir),
            "suffix": args.suffix,
            "metric": args.metric,
            "heatmap_path": str(heatmap_path),
            "matrix_path": str(matrix_path),
            "predictions_path": str(predictions_path),
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    draw_heatmap(
        matrix=matrix,
        labels=labels,
        output_path=heatmap_path,
        title=f"DNA distance heatmap: base vs rerun ({args.suffix})",
        metric=args.metric,
        annotate=args.annotate,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
