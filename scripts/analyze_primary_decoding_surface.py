#!/usr/bin/env python3
"""Analyze and plot the live temperature/top-p retrieval surface.

The analysis uses one fixed strict cohort across all 26 planned run keys.  It
reports (1) queries against a deterministic T=0 reference gallery, (2)
same-setting r1-to-r2 repeat stability, and (3) equal-cell mean and worst-cell
accuracy under joint bounds T <= tau and top-p <= pi.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures" / "primary_robustness_20260813"
GRID_SCRIPT = ROOT / "scripts" / "build_primary_run_grid.py"
TEMPERATURES = (0.0, 0.2, 0.3, 0.5, 0.7)
STOCHASTIC_TEMPERATURES = (0.2, 0.3, 0.5, 0.7)
TOP_PS = (0.8, 0.9, 1.0)
REPEATS = (1, 2)
COLORS = {0.8: "#2563eb", 0.9: "#d97706", 1.0: "#059669"}


def load_grid_module():
    spec = importlib.util.spec_from_file_location("primary_grid_surface", GRID_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def status_rows(path: Path) -> dict[str, dict[str, Any]]:
    final: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return final
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        model = row.get("model_id")
        if model:
            final[str(model)] = row
    return final


def run_key(manifest_path: Path, manifest: dict[str, Any]) -> tuple[float, float, int] | None:
    try:
        temperature = float(manifest["temperature"])
        top_p = float(manifest["top_p"])
    except (KeyError, TypeError, ValueError):
        return None
    suffix = str(manifest.get("output_suffix", ""))
    match = re.search(r"_r([12])$", suffix)
    repeat = int(match.group(1)) if match else None
    if (
        repeat is None
        and manifest_path.parent.name == "rand_chinese_0p3b_7b_temp07"
        and temperature == 0.7
        and top_p == 0.9
    ):
        repeat = 1
    if (
        repeat not in REPEATS
        or temperature not in TEMPERATURES
        or top_p not in TOP_PS
        or (temperature == 0.0 and top_p != 1.0)
    ):
        return None
    return temperature, top_p, int(repeat)


def candidate_priority(manifest: dict[str, Any], row: dict[str, Any]) -> tuple[int, str]:
    seed = manifest.get("generation_seed")
    seed_priority = 2 if seed in (42001, 42002) else 1 if seed is not None else 0
    return seed_priority, str(row.get("recorded_at", ""))


def collect_artifacts(cohort: set[str]) -> dict[tuple[str, float, float, int], Path]:
    selected: dict[tuple[str, float, float, int], tuple[tuple[int, str], Path]] = {}
    for manifest_path in sorted((ROOT / "out").rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        key = run_key(manifest_path, manifest)
        if key is None:
            continue
        temperature, top_p, repeat = key
        for model, row in status_rows(manifest_path.parent / "status.jsonl").items():
            if model not in cohort or row.get("status") != "success":
                continue
            raw_path = row.get("output_path")
            if not raw_path:
                continue
            dna_path = Path(str(raw_path))
            if not dna_path.is_absolute():
                dna_path = ROOT / dna_path
            if not dna_path.exists():
                continue
            artifact_key = model, temperature, top_p, repeat
            candidate = candidate_priority(manifest, row), dna_path
            if artifact_key not in selected or candidate[0] > selected[artifact_key][0]:
                selected[artifact_key] = candidate
    return {key: value[1] for key, value in selected.items()}


def planned_run_keys() -> list[tuple[float, float, int]]:
    return [
        (temperature, top_p, repeat)
        for temperature in TEMPERATURES
        for top_p in TOP_PS
        if temperature > 0 or top_p == 1.0
        for repeat in REPEATS
    ]


def load_vector(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    vector = np.asarray(payload["signature"], dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"Invalid DNA vector: {path}")
    return vector


def matrix(
    artifacts: dict[tuple[str, float, float, int], Path],
    cohort: list[str],
    temperature: float,
    top_p: float,
    repeat: int,
) -> np.ndarray:
    return np.stack(
        [load_vector(artifacts[(model, temperature, top_p, repeat)]) for model in cohort]
    )


def normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def retrieval_metrics(gallery: np.ndarray, query: np.ndarray) -> dict[str, float]:
    scores = normalize(query) @ normalize(gallery).T
    order = np.argsort(-scores, axis=1, kind="stable")
    truth = np.arange(len(query))
    ranks = np.asarray([int(np.flatnonzero(order[i] == i)[0]) + 1 for i in truth])
    return {
        "top1": float(np.mean(ranks == 1)),
        "top3": float(np.mean(ranks <= 3)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


def setting_rows(
    artifacts: dict[tuple[str, float, float, int], Path], cohort: list[str]
) -> list[dict[str, Any]]:
    fixed_gallery = matrix(artifacts, cohort, 0.0, 1.0, 1)
    rows: list[dict[str, Any]] = []
    for temperature in TEMPERATURES:
        for top_p in TOP_PS:
            if temperature == 0.0 and top_p != 1.0:
                continue
            query = matrix(artifacts, cohort, temperature, top_p, 2)
            same_gallery = matrix(artifacts, cohort, temperature, top_p, 1)
            for comparison, gallery in (
                ("fixed_t0_gallery", fixed_gallery),
                ("same_setting_repeat", same_gallery),
            ):
                rows.append(
                    {
                        "comparison": comparison,
                        "temperature": temperature,
                        "top_p": top_p,
                        "model_count": len(cohort),
                        **retrieval_metrics(gallery, query),
                    }
                )
    return rows


def bounded_rows(
    artifacts: dict[tuple[str, float, float, int], Path], cohort: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tau in STOCHASTIC_TEMPERATURES:
        for p_bound in TOP_PS:
            settings = [(0.0, 1.0)] + [
                (temperature, top_p)
                for temperature in STOCHASTIC_TEMPERATURES
                for top_p in TOP_PS
                if temperature <= tau and top_p <= p_bound
            ]
            galleries = [matrix(artifacts, cohort, t, p, 1) for t, p in settings]
            centroid = np.mean(np.stack(galleries), axis=0)
            cell_rows = []
            for temperature, top_p in settings:
                query = matrix(artifacts, cohort, temperature, top_p, 2)
                cell_rows.append(retrieval_metrics(centroid, query))
            for metric in ("top1", "mrr"):
                values = [row[metric] for row in cell_rows]
                rows.append(
                    {
                        "temperature_bound": tau,
                        "top_p_bound": p_bound,
                        "metric": metric,
                        "cell_count": len(settings),
                        "model_count": len(cohort),
                        "equal_cell_mean": float(np.mean(values)),
                        "worst_setting": float(np.min(values)),
                        "best_setting": float(np.max(values)),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    draw.rounded_rectangle(box, radius=18, fill="#f8fafc", outline="#cbd5e1", width=3)
    draw.text((box[0] + 25, box[1] + 20), title, font=font(27, True), fill="#0f172a")


def axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    xs: list[float],
    xlabel: str,
    ylabel: str,
) -> tuple[callable, callable]:
    left, top, right, bottom = box
    draw.line((left, bottom, right, bottom), fill="#475569", width=3)
    draw.line((left, top, left, bottom), fill="#475569", width=3)
    for value in np.linspace(0, 1, 6):
        y = int(bottom - value * (bottom - top))
        draw.line((left, y, right, y), fill="#e2e8f0", width=2)
        draw.text((left - 62, y - 11), f"{value:.1f}", font=font(18), fill="#475569")
    for index, value in enumerate(xs):
        x = int(left + index * (right - left) / max(1, len(xs) - 1))
        draw.line((x, bottom, x, bottom + 8), fill="#475569", width=2)
        draw.text((x - 17, bottom + 12), f"{value:g}", font=font(18), fill="#475569")
    draw.text(((left + right) // 2 - 85, bottom + 48), xlabel, font=font(20, True), fill="#334155")
    draw.text((left - 78, top - 30), ylabel, font=font(18, True), fill="#334155")
    return (
        lambda index: int(left + index * (right - left) / max(1, len(xs) - 1)),
        lambda value: int(bottom - value * (bottom - top)),
    )


def draw_line_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    rows: list[dict[str, Any]],
) -> None:
    panel(draw, box, "A. Fixed top-p: exact Top-1 vs temperature")
    plot_box = (box[0] + 95, box[1] + 105, box[2] - 35, box[3] - 85)
    xmap, ymap = axes(draw, plot_box, list(TEMPERATURES), "Temperature", "Top-1")
    lookup = {
        (float(row["temperature"]), float(row["top_p"])): float(row["top1"])
        for row in rows
        if row["comparison"] == "fixed_t0_gallery"
    }
    for top_p in TOP_PS:
        points = []
        for index, temperature in enumerate(TEMPERATURES):
            key = (temperature, top_p)
            if key in lookup:
                points.append((xmap(index), ymap(lookup[key]), lookup[key]))
        if len(points) > 1:
            draw.line([(x, y) for x, y, _ in points], fill=COLORS[top_p], width=6)
        for x, y, value in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=COLORS[top_p], outline="white", width=2)
            draw.text((x - 23, y - 31), f"{value:.2f}", font=font(16, True), fill=COLORS[top_p])
    legend_x = box[0] + 125
    for top_p in TOP_PS:
        draw.line((legend_x, box[1] + 68, legend_x + 38, box[1] + 68), fill=COLORS[top_p], width=6)
        draw.text((legend_x + 46, box[1] + 55), f"top-p={top_p:g}", font=font(18), fill="#334155")
        legend_x += 180


def heat_color(value: float) -> str:
    low = (254, 226, 226)
    high = (22, 163, 74)
    ratio = min(1.0, max(0.0, value))
    rgb = [round(a + (b - a) * ratio) for a, b in zip(low, high)]
    return "#" + "".join(f"{part:02x}" for part in rgb)


def draw_heatmap(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    rows: list[dict[str, Any]],
    comparison: str,
) -> None:
    panel(draw, box, title)
    lookup = {
        (float(row["temperature"]), float(row["top_p"])): float(row["top1"])
        for row in rows
        if row["comparison"] == comparison
    }
    left, top = box[0] + 105, box[1] + 115
    cell_w, cell_h = 145, 80
    for column, top_p in enumerate(TOP_PS):
        draw.text((left + column * cell_w + 35, top - 42), f"p={top_p:g}", font=font(19, True), fill="#334155")
    for row_index, temperature in enumerate(TEMPERATURES):
        draw.text((box[0] + 28, top + row_index * cell_h + 25), f"T={temperature:g}", font=font(19, True), fill="#334155")
        for column, top_p in enumerate(TOP_PS):
            coords = (
                left + column * cell_w,
                top + row_index * cell_h,
                left + (column + 1) * cell_w - 10,
                top + (row_index + 1) * cell_h - 10,
            )
            key = (temperature, top_p)
            if key not in lookup:
                draw.rounded_rectangle(coords, radius=9, fill="#e2e8f0")
                draw.text((coords[0] + 48, coords[1] + 20), "N/A", font=font(18, True), fill="#64748b")
                continue
            value = lookup[key]
            draw.rounded_rectangle(coords, radius=9, fill=heat_color(value), outline="#94a3b8", width=2)
            draw.text((coords[0] + 35, coords[1] + 13), f"{value:.3f}", font=font(21, True), fill="#0f172a")
            draw.text((coords[0] + 43, coords[1] + 42), "Top-1", font=font(15), fill="#334155")


def plot_surface(rows: list[dict[str, Any]], cohort_size: int, generated_at: str, output: Path) -> None:
    image = Image.new("RGB", (2000, 1120), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 38), "Temperature and top-p relationship — live primary cohort", font=font(40, True), fill="#0f172a")
    draw.text((65, 92), f"Fixed strict cohort N={cohort_size}; r1 gallery vs held-out r2 query; generated {generated_at}", font=font(22), fill="#475569")
    draw_line_panel(draw, (45, 145, 860, 1050), rows)
    draw_heatmap(draw, (895, 145, 1430, 1050), "B. Fixed T=0 gallery", rows, "fixed_t0_gallery")
    draw_heatmap(draw, (1465, 145, 1990, 1050), "C. Same-setting repeat", rows, "same_setting_repeat")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)


def plot_bounded(rows: list[dict[str, Any]], cohort_size: int, generated_at: str, output: Path) -> None:
    image = Image.new("RGB", (1900, 1050), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 38), "Joint decoding bounds — cosine centroid retrieval", font=font(40, True), fill="#0f172a")
    draw.text((65, 92), f"T≤tau and top-p≤pi; deterministic control always included; strict cohort N={cohort_size}; {generated_at}", font=font(22), fill="#475569")
    selected = [row for row in rows if row["metric"] == "top1"]
    for panel_index, (value_key, title) in enumerate(
        (("equal_cell_mean", "A. Equal-cell mean Top-1"), ("worst_setting", "B. Worst-setting Top-1"))
    ):
        x0 = 45 + panel_index * 735
        box = (x0, 145, x0 + 700, 980)
        panel(draw, box, title)
        plot_box = (box[0] + 95, box[1] + 110, box[2] - 35, box[3] - 85)
        xmap, ymap = axes(draw, plot_box, list(STOCHASTIC_TEMPERATURES), "Temperature bound tau", "Top-1")
        lookup = {
            (float(row["temperature_bound"]), float(row["top_p_bound"])): float(row[value_key])
            for row in selected
        }
        legend_x = box[0] + 110
        for p_bound in TOP_PS:
            points = [(xmap(i), ymap(lookup[(tau, p_bound)]), lookup[(tau, p_bound)]) for i, tau in enumerate(STOCHASTIC_TEMPERATURES)]
            draw.line([(x, y) for x, y, _ in points], fill=COLORS[p_bound], width=6)
            for x, y, _ in points:
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=COLORS[p_bound], outline="white", width=2)
            draw.line((legend_x, box[1] + 72, legend_x + 32, box[1] + 72), fill=COLORS[p_bound], width=6)
            draw.text((legend_x + 38, box[1] + 59), f"p≤{p_bound:g}", font=font(17), fill="#334155")
            legend_x += 145

    box = (1510, 145, 1885, 980)
    panel(draw, box, "C. Mean Top-1 grid")
    lookup = {
        (float(row["temperature_bound"]), float(row["top_p_bound"])): float(row["equal_cell_mean"])
        for row in selected
    }
    left, top, cell_w, cell_h = box[0] + 90, box[1] + 145, 90, 135
    for column, p_bound in enumerate(TOP_PS):
        draw.text((left + column * cell_w + 8, top - 48), f"p≤{p_bound:g}", font=font(16, True), fill="#334155")
    for row_index, tau in enumerate(STOCHASTIC_TEMPERATURES):
        draw.text((box[0] + 18, top + row_index * cell_h + 43), f"T≤{tau:g}", font=font(17, True), fill="#334155")
        for column, p_bound in enumerate(TOP_PS):
            value = lookup[(tau, p_bound)]
            coords = (left + column * cell_w, top + row_index * cell_h, left + (column + 1) * cell_w - 8, top + (row_index + 1) * cell_h - 10)
            draw.rounded_rectangle(coords, radius=9, fill=heat_color(value), outline="#94a3b8", width=2)
            draw.text((coords[0] + 10, coords[1] + 42), f"{value:.3f}", font=font(17, True), fill="#0f172a")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)


def main() -> int:
    grid = load_grid_module()
    fixed_cohort = grid.load_cohort()
    artifacts = collect_artifacts(fixed_cohort)
    run_keys = planned_run_keys()
    strict = sorted(
        model
        for model in fixed_cohort
        if all((model, temperature, top_p, repeat) in artifacts for temperature, top_p, repeat in run_keys)
    )
    if not strict:
        raise SystemExit("No strict common cohort is available.")
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    per_setting = setting_rows(artifacts, strict)
    bounded = bounded_rows(artifacts, strict)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "primary_decoding_surface.csv", per_setting)
    write_csv(OUT_DIR / "primary_bounded_surface.csv", bounded)
    payload = {
        "generated_at": generated_at,
        "strict_cohort_size": len(strict),
        "strict_cohort": strict,
        "artifact_count": len(artifacts),
        "selection_priority": "explicit 42001/42002 success, then other explicit seed, then historical unknown seed",
        "per_setting": per_setting,
        "bounded": bounded,
    }
    (OUT_DIR / "primary_decoding_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    stamp = generated_at.replace("T", " ")
    plot_surface(per_setting, len(strict), stamp, OUT_DIR / "temperature_top_p_relationship.png")
    plot_bounded(bounded, len(strict), stamp, OUT_DIR / "bounded_decoding_relationship.png")
    print(json.dumps({
        "generated_at": generated_at,
        "artifact_count": len(artifacts),
        "strict_cohort_size": len(strict),
        "outputs": [
            str(OUT_DIR / "temperature_top_p_relationship.png"),
            str(OUT_DIR / "bounded_decoding_relationship.png"),
            str(OUT_DIR / "primary_decoding_surface.csv"),
            str(OUT_DIR / "primary_bounded_surface.csv"),
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
