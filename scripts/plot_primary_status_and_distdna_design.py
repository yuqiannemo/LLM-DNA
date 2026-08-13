#!/usr/bin/env python3
"""Render the August 13 live grid snapshot and DistDNA pilot design with Pillow."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures" / "primary_robustness_20260813"
GRID_SCRIPT = ROOT / "scripts" / "build_primary_run_grid.py"
BASELINE_SUCCESS = 2528


def load_grid_module():
    spec = importlib.util.spec_from_file_location("primary_grid", GRID_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def rounded_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=22, fill="#f8fafc", outline="#cbd5e1", width=3)


def status_map(path: Path) -> dict[str, str]:
    final: dict[str, str] = {}
    if not path.exists():
        return final
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("model_id"):
            final[str(row["model_id"])] = str(row.get("status", ""))
    return final


def seed_sources(cohort: set[str]) -> dict[tuple[str, float, float, int], set[str]]:
    sources: dict[tuple[str, float, float, int], set[str]] = defaultdict(set)
    for manifest_path in (ROOT / "out").rglob("manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            temperature = float(manifest["temperature"])
            top_p = float(manifest["top_p"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
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
            repeat not in (1, 2)
            or temperature not in {0.0, 0.2, 0.3, 0.5, 0.7}
            or top_p not in {0.8, 0.9, 1.0}
            or (temperature == 0.0 and top_p != 1.0)
        ):
            continue
        explicit = manifest.get("generation_seed") is not None
        for model, status in status_map(manifest_path.parent / "status.jsonl").items():
            if model in cohort and status == "success":
                sources[(model, temperature, top_p, repeat)].add(
                    "explicit" if explicit else "unknown"
                )
    return sources


def collect_snapshot() -> dict[str, object]:
    grid = load_grid_module()
    cohort = grid.load_cohort()
    runs, _ignored = grid.collect_runs(cohort)
    rows: list[dict[str, object]] = []
    successes: set[tuple[str, float, float, int]] = set()
    for temperature in grid.TEMPERATURES:
        for top_p in grid.TOP_PS:
            if not grid.planned_condition(temperature, top_p):
                continue
            for repeat in grid.REPEATS:
                summary = grid.summarize_run(runs[(temperature, top_p, repeat)], cohort)
                rows.append(
                    {
                        "temperature": temperature,
                        "top_p": top_p,
                        "repeat": repeat,
                        **summary,
                    }
                )
                successes.update(
                    (model, temperature, top_p, repeat)
                    for model, state in runs[(temperature, top_p, repeat)]["final"].items()
                    if model in cohort and state == "success"
                )
    total = sum(int(row["success"]) for row in rows)
    strict = sum(
        all(
            (model, temperature, top_p, repeat) in successes
            for temperature in grid.TEMPERATURES
            for top_p in grid.TOP_PS
            if grid.planned_condition(temperature, top_p)
            for repeat in grid.REPEATS
        )
        for model in cohort
    )
    per_setting: dict[str, int] = defaultdict(int)
    for row in rows:
        key = f"{float(row['temperature']):g}:{float(row['top_p']):g}"
        per_setting[key] += int(row["success"])
    source_tags = seed_sources(cohort)
    provenance: Counter[str] = Counter()
    for key in successes:
        tags = source_tags.get(key, set())
        provenance["explicit"] += int("explicit" in tags)
        provenance["unknown_only"] += int("explicit" not in tags)
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "successful": total,
        "remaining": 2600 - total,
        "recovered_since_august_5": max(0, total - BASELINE_SUCCESS),
        "strict_common_models": strict,
        "explicit_seed_successes": provenance["explicit"],
        "unknown_seed_only_successes": provenance["unknown_only"],
        "per_setting_successes": dict(per_setting),
    }


def interpolate(start: tuple[int, int, int], end: tuple[int, int, int], ratio: float) -> str:
    values = [round(a + (b - a) * ratio) for a, b in zip(start, end)]
    return "#" + "".join(f"{value:02x}" for value in values)


def draw_progress(snapshot: dict[str, object], output: Path) -> None:
    image = Image.new("RGB", (1800, 1050), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 42), "Primary robustness grid — live recovery snapshot", font=font(42, True), fill="#0f172a")
    stamp = str(snapshot["generated_at"]).replace("T", " ")
    draw.text((70, 96), f"Generated {stamp}; 100 models × 13 settings × 2 rounds", font=font(24), fill="#475569")

    rounded_panel(draw, (55, 155, 970, 975))
    draw.text((90, 185), "A. Successful artifacts by decoding setting", font=font(29, True), fill="#0f172a")
    temperatures = [0.0, 0.2, 0.3, 0.5, 0.7]
    top_ps = [0.8, 0.9, 1.0]
    left, top, cell_w, cell_h = 250, 275, 215, 118
    for column, top_p in enumerate(top_ps):
        draw.text((left + column * cell_w + 63, top - 52), f"p={top_p:g}", font=font(25, True), fill="#334155")
    values = snapshot["per_setting_successes"]
    assert isinstance(values, dict)
    for row, temperature in enumerate(temperatures):
        draw.text((92, top + row * cell_h + 38), f"T={temperature:g}", font=font(25, True), fill="#334155")
        for column, top_p in enumerate(top_ps):
            box = (left + column * cell_w, top + row * cell_h, left + (column + 1) * cell_w - 15, top + (row + 1) * cell_h - 15)
            if temperature == 0.0 and top_p != 1.0:
                draw.rounded_rectangle(box, radius=12, fill="#e2e8f0")
                draw.text((box[0] + 71, box[1] + 31), "N/A", font=font(25, True), fill="#64748b")
                continue
            count = int(values[f"{temperature:g}:{top_p:g}"])
            color = interpolate((254, 226, 226), (22, 163, 74), count / 200.0)
            draw.rounded_rectangle(box, radius=12, fill=color, outline="#94a3b8", width=2)
            draw.text((box[0] + 47, box[1] + 23), f"{count}/200", font=font(27, True), fill="#0f172a")
            draw.text((box[0] + 65, box[1] + 59), f"{count / 2:.1f}%", font=font(20), fill="#334155")

    rounded_panel(draw, (1010, 155, 1745, 555))
    draw.text((1045, 185), "B. Recovery and complete-case progress", font=font(29, True), fill="#0f172a")
    successful = int(snapshot["successful"])
    recovered = int(snapshot["recovered_since_august_5"])
    remaining = int(snapshot["remaining"])
    bar = (1050, 285, 1695, 350)
    base_width = round((BASELINE_SUCCESS / 2600) * (bar[2] - bar[0]))
    recovery_width = round((recovered / 2600) * (bar[2] - bar[0]))
    draw.rounded_rectangle(bar, radius=16, fill="#fee2e2")
    draw.rounded_rectangle((bar[0], bar[1], bar[0] + base_width, bar[3]), radius=16, fill="#3b82f6")
    draw.rectangle((bar[0] + base_width - 10, bar[1], bar[0] + base_width + recovery_width, bar[3]), fill="#22c55e")
    draw.text((1050, 370), f"{successful}/2,600 successful ({100 * successful / 2600:.1f}%)", font=font(28, True), fill="#0f172a")
    draw.text((1050, 415), f"August 5 baseline 2,528  +  recovered {recovered}  |  remaining {remaining}", font=font(21), fill="#334155")
    strict = int(snapshot["strict_common_models"])
    draw.text((1050, 468), f"Strict all-26-run common cohort: {strict}/100 models", font=font(24, True), fill="#7c2d12")

    rounded_panel(draw, (1010, 595, 1745, 975))
    draw.text((1045, 625), "C. Generation-seed provenance", font=font(29, True), fill="#0f172a")
    explicit = int(snapshot["explicit_seed_successes"])
    unknown = int(snapshot["unknown_seed_only_successes"])
    total = max(1, explicit + unknown)
    x0, y0, width = 1050, 735, 645
    explicit_width = round(width * explicit / total)
    draw.rounded_rectangle((x0, y0, x0 + width, y0 + 65), radius=16, fill="#f59e0b")
    draw.rounded_rectangle((x0, y0, x0 + explicit_width, y0 + 65), radius=16, fill="#14b8a6")
    draw.text((1050, 820), f"Explicit generation seed: {explicit}", font=font(24, True), fill="#0f766e")
    draw.text((1050, 860), f"Historical unknown-only seed: {unknown}", font=font(24, True), fill="#b45309")
    draw.text((1050, 915), "Coverage is not the same as seed-clean coverage.", font=font(21), fill="#475569")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = "#334155") -> None:
    draw.line((start, end), fill=color, width=7)
    x, y = end
    draw.polygon([(x, y), (x - 20, y - 13), (x - 20, y + 13)], fill=color)


def draw_pilot_design(output: Path) -> None:
    image = Image.new("RGB", (1800, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 40), "Same-setting K=2 DistDNA pilot: leakage-free, matched-budget comparison", font=font(39, True), fill="#0f172a")
    draw.text((70, 92), "Start with a 10-model debugging pilot; expand to 20–30 models for evidence after the pipeline is stable.", font=font(23), fill="#475569")

    rounded_panel(draw, (55, 155, 855, 565))
    draw.text((90, 185), "A. Four independent response sets", font=font(29, True), fill="#0f172a")
    boxes = [
        (100, 285, 305, 395, "r1", "seed 42001", "#bfdbfe"),
        (350, 285, 555, 395, "r3", "seed 42003", "#bfdbfe"),
        (100, 440, 305, 550, "r2", "seed 42002", "#ddd6fe"),
        (350, 440, 555, 550, "r4", "seed 42004", "#ddd6fe"),
    ]
    for x1, y1, x2, y2, name, seed, color in boxes:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=16, fill=color, outline="#64748b", width=2)
        draw.text((x1 + 78, y1 + 14), name, font=font(29, True), fill="#0f172a")
        draw.text((x1 + 28, y1 + 62), seed, font=font(20), fill="#334155")
    draw.text((610, 305), "Reference\ndistribution\nK=2", font=font(25, True), fill="#1d4ed8", spacing=8)
    draw.text((610, 455), "Held-out query\ndistribution\nK=2", font=font(25, True), fill="#6d28d9", spacing=8)
    arrow(draw, (555, 340), (595, 340), "#1d4ed8")
    arrow(draw, (555, 495), (595, 495), "#6d28d9")

    rounded_panel(draw, (895, 155, 1745, 565))
    draw.text((930, 185), "B. Compare on identical responses", font=font(29, True), fill="#0f172a")
    methods = [
        ("Lower-budget diagnostic", "Single-response cosine", "K=1"),
        ("Strong matched baseline", "Mean-DNA cosine", "K=2"),
        ("Exact distribution distance", "RBF-MMD", "K=2"),
        ("Scalable proposed vector", "RFFTrace Euclidean", "K=2"),
        ("Compact proposed vector", "Projected DistDNA", "K=2"),
    ]
    y = 250
    for label, name, budget in methods:
        draw.rounded_rectangle((935, y, 1695, y + 52), radius=12, fill="#ffffff", outline="#cbd5e1", width=2)
        draw.text((955, y + 10), label, font=font(18), fill="#64748b")
        draw.text((1220, y + 8), name, font=font(20, True), fill="#0f172a")
        draw.text((1620, y + 10), budget, font=font(18, True), fill="#334155")
        y += 61

    rounded_panel(draw, (55, 605, 1745, 940))
    draw.text((90, 635), "C. Decision rule for claiming an improvement", font=font(29, True), fill="#0f172a")
    draw.text((105, 705), "Primary endpoint", font=font(22, True), fill="#475569")
    draw.text((360, 698), "Top-1 exact-model retrieval; report MRR, Top-3/5, time, and storage secondarily.", font=font(24), fill="#0f172a")
    draw.text((105, 770), "Fairness", font=font(22, True), fill="#475569")
    draw.text((360, 763), "Compare DistDNA against matched-K=2 mean cosine, not only the cheaper K=1 baseline.", font=font(24), fill="#0f172a")
    draw.text((105, 835), "Statistical gate", font=font(22, True), fill="#475569")
    draw.text((360, 828), "Paired per-model Top-1 difference; model-cluster bootstrap 95% CI lower bound > 0.", font=font(24), fill="#0f172a")
    draw.text((105, 895), "No leakage", font=font(22, True), fill="#475569")
    draw.text((360, 888), "Never reuse r1+r2 on both gallery and query; that forces self-distance toward zero.", font=font(24), fill="#9a3412")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)


def main() -> int:
    snapshot = collect_snapshot()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_progress(snapshot, OUT_DIR / "live_grid_seed_recovery.png")
    draw_pilot_design(OUT_DIR / "distdna_k2_pilot_design.png")
    (OUT_DIR / "snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
