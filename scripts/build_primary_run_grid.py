#!/usr/bin/env python3
"""Build the live completion grid using the latest validated-100 cohort."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "out"
COHORT_MANIFEST = (
    OUTPUT_ROOT
    / "top_p_control_t05_validated100_20260719"
    / "top_p_control_t05_t05_p06_r2"
    / "manifest.json"
)
REPORT_DIR = ROOT / "reports"

TEMPERATURES = (0.0, 0.2, 0.3, 0.5, 0.7)
TOP_PS = (0.8, 0.9, 1.0)
REPEATS = (1, 2)
GENERATION_SEEDS = {1: 42001, 2: 42002}


def load_cohort() -> set[str]:
    manifest = json.loads(COHORT_MANIFEST.read_text(encoding="utf-8"))
    models = {
        str(task["model_id"])
        for task in manifest.get("tasks", [])
        if task.get("model_id")
    }
    if len(models) != 100:
        raise ValueError(f"Expected 100 fixed models, found {len(models)}")
    return models


def planned_condition(temperature: float, top_p: float) -> bool:
    # At T=0 sampling is disabled, so top-p is inactive and recorded once.
    return temperature > 0 or top_p == 1.0


def load_statuses(path: Path) -> dict[str, str]:
    final: dict[str, str] = {}
    if not path.exists():
        return final
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            model_id = record.get("model_id")
            if model_id:
                final[str(model_id)] = str(record.get("status", ""))
    return final


def classify(status: str) -> str:
    if status == "success":
        return "success"
    if status.startswith("failed") or status == "incompatible":
        return "failed"
    if status.startswith("skipped"):
        return "skipped"
    return "pending"


def collect_runs(
    cohort: set[str],
) -> tuple[dict[tuple[float, float, int], dict[str, Any]], list[str]]:
    runs: dict[tuple[float, float, int], dict[str, Any]] = defaultdict(
        lambda: {
            "manifest_models": set(),
            "final": {},
            "sources": [],
            "manifest_count": 0,
            "generation_seeds": set(),
            "cohort_overlaps": [],
        }
    )
    ignored: list[str] = []
    if not OUTPUT_ROOT.exists():
        return runs, ignored

    for manifest_path in sorted(OUTPUT_ROOT.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            temperature = float(manifest["temperature"])
            top_p = float(manifest["top_p"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            ignored.append(manifest_path.relative_to(ROOT).as_posix())
            continue

        suffix = str(manifest.get("output_suffix", ""))
        repeat = next(
            (repeat for repeat in REPEATS if suffix.endswith(f"_r{repeat}")), None
        )
        # The original T=0.7, top-p=0.9 execution was labelled "prelim"
        # rather than r1. It is still a reusable first execution of this cell.
        if (
            repeat is None
            and manifest_path.parent.name == "rand_chinese_0p3b_7b_temp07"
            and temperature == 0.7
            and top_p == 0.9
        ):
            repeat = 1
        if (
            repeat is None
            or temperature not in TEMPERATURES
            or top_p not in TOP_PS
            or not planned_condition(temperature, top_p)
        ):
            ignored.append(manifest_path.relative_to(ROOT).as_posix())
            continue

        key = (temperature, top_p, repeat)
        run = runs[key]
        run["manifest_count"] += 1
        run["sources"].append(manifest_path.parent.relative_to(ROOT).as_posix())
        manifest_models = {
            str(task["model_id"])
            for task in manifest.get("tasks", [])
            if task.get("model_id")
        }
        overlap = manifest_models & cohort
        run["manifest_models"].update(overlap)
        run["cohort_overlaps"].append(len(overlap))
        seed = manifest.get("generation_seed")
        run["generation_seeds"].add(
            str(seed)
            if seed is not None
            else "not explicitly set or recorded"
        )
        for model_id, status in load_statuses(
            manifest_path.parent / "status.jsonl"
        ).items():
            if model_id in cohort:
                run["final"][model_id] = status
    return runs, ignored


def summarize_run(run: dict[str, Any], cohort: set[str]) -> dict[str, Any]:
    counts = defaultdict(int)
    for model_id, status in run["final"].items():
        if model_id in cohort:
            counts[classify(status)] += 1
    observed = sum(counts.values())
    no_status = len(cohort) - observed
    missing_success = len(cohort) - counts["success"]
    return {
        "success": counts["success"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "pending_status": counts["pending"],
        "no_status": no_status,
        "missing_success": missing_success,
        "started": bool(run["manifest_count"] or observed),
        "manifest_models": len(run["manifest_models"]),
        "cohort_overlap": max(run["cohort_overlaps"], default=0),
        "recorded_seeds": ",".join(sorted(run["generation_seeds"])),
        "sources": ";".join(run["sources"]),
    }


def cell_text(repeat_rows: list[dict[str, Any]]) -> str:
    success = sum(int(row["success"]) for row in repeat_rows)
    target = 100 * len(REPEATS)
    if success == target:
        return f"✅ {success}/{target}"
    if any(bool(row["started"]) for row in repeat_rows):
        return f"♻️ {success}/{target}；还需 {target - success}"
    return f"⬜ 0/{target}；待跑 r1/r2"


def main() -> None:
    cohort = load_cohort()
    runs, ignored = collect_runs(cohort)
    detail_rows: list[dict[str, Any]] = []
    setting_rows: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)

    for temperature in TEMPERATURES:
        for top_p in TOP_PS:
            if not planned_condition(temperature, top_p):
                continue
            for repeat in REPEATS:
                summary = summarize_run(
                    runs[(temperature, top_p, repeat)], cohort
                )
                row = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "repeat": f"r{repeat}",
                    # T=0.5 is shared with the validated top-p control, whose
                    # explicit-seed runs use 42 in both rounds.
                    "proposed_backfill_seed": (
                        42
                        if temperature == 0.5
                        else GENERATION_SEEDS[repeat]
                    ),
                    **summary,
                }
                detail_rows.append(row)
                setting_rows[(temperature, top_p)].append(row)

    REPORT_DIR.mkdir(exist_ok=True)
    with (REPORT_DIR / "primary_robustness_run_grid.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "temperature",
            "top_p",
            "repeat",
            "proposed_backfill_seed",
            "recorded_seeds",
            "success",
            "failed",
            "skipped",
            "pending_status",
            "no_status",
            "missing_success",
            "started",
            "manifest_models",
            "cohort_overlap",
            "sources",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(detail_rows)

    total_success = sum(int(row["success"]) for row in detail_rows)
    target_tasks = len(detail_rows) * 100
    completed_settings = sum(
        sum(int(row["success"]) for row in rows) == 200
        for rows in setting_rows.values()
    )
    started_settings = sum(
        any(bool(row["started"]) for row in rows)
        for rows in setting_rows.values()
    )

    markdown = [
        "# Primary robustness experiment — run grid",
        "",
        f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "Revised plan:",
        "",
        "- Fixed cohort: **the latest validated100 model list** from "
        "`top_p_control_t05_t05_p06_r2/manifest.json`.",
        "- Historical artifacts are reused only when their model ID and decoding "
        "cell match this cohort/grid.",
        "- Two repeats per condition. The shared `temperature=0.5` row uses "
        "**seed 42 for both rounds** so its `top_p=0.8/0.9/1.0` cells can also "
        "serve the validated top-p control. Other proposed grid backfills remain "
        "**r1 / 42001** and **r2 / 42002**.",
        "- The historical sweep did not explicitly set or record a response-generation "
        "seed. Its recorded `random_seed=42` controlled probe selection and DNA "
        "reduction, not the generation sampler.",
        "- Stochastic grid: temperature `{0.2, 0.3, 0.5, 0.7}` × top-p `{0.8, 0.9, 1.0}`.",
        "- Deterministic control: `temperature=0`, `top_p=1.0` (top-p inactive).",
        f"- Total: **13 settings × 2 repeats × 100 models = {target_tasks:,} successful model artifacts required**.",
        "",
        "Important cohort note: the two validated100 variants overlap by 97 models, "
        "but the older temperature/top-p sweep used here overlaps the latest "
        "validated100 cohort by **87 models**. Therefore those grid cells require "
        "13 cohort backfills plus retries for any failed/pending overlapping models.",
        "",
        "Legend: `✅` both repeats have 100 successful models; `♻️` historical "
        "results are reusable but the cell is incomplete; `⬜` no matching run was "
        "found; `—` is not a planned condition.",
        "",
        "| Temperature \\\\ Top-p | 0.8 | 0.9 | 1.0 |",
        "|---:|---:|---:|---:|",
    ]
    for temperature in TEMPERATURES:
        cells = []
        for top_p in TOP_PS:
            if not planned_condition(temperature, top_p):
                cells.append("—")
            else:
                cells.append(cell_text(setting_rows[(temperature, top_p)]))
        markdown.append(
            f"| {temperature:g} | " + " | ".join(cells) + " |"
        )

    markdown += [
        "",
        "## Overall progress",
        "",
        f"- Successful artifacts: **{total_success}/{target_tasks}**.",
        f"- Settings started: **{started_settings}/13**.",
        f"- Settings fully complete: **{completed_settings}/13**.",
        f"- Remaining successful artifacts required: **{target_tasks - total_success}**.",
        "",
        "## Repeat-level detail",
        "",
        "| Temperature | Top-p | Round | Historical generation seed | Proposed backfill seed (not run) | Reusable success | Failed | Pending/no status | Still needed | State |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in detail_rows:
        state = (
            "✅ complete"
            if row["success"] == 100
            else ("♻️ reusable/incomplete" if row["started"] else "⬜ not run")
        )
        historical_seed = row["recorded_seeds"] or "—"
        markdown.append(
            f"| {row['temperature']:g} | {row['top_p']:g} | {row['repeat']} | "
            f"{historical_seed} | {row['proposed_backfill_seed']} | {row['success']} | "
            f"{row['failed']} | "
            f"{row['pending_status'] + row['skipped'] + row['no_status']} | "
            f"{row['missing_success']} | {state} |"
        )
    if ignored:
        markdown += [
            "",
            f"Ignored {len(ignored)} manifest(s) outside the revised primary grid; see the CSV/source tree for audit.",
        ]
    markdown.append("")
    (REPORT_DIR / "primary_robustness_run_grid.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(
        f"Wrote primary grid: {total_success}/{target_tasks} successful artifacts, "
        f"{started_settings}/13 settings started"
    )


if __name__ == "__main__":
    main()
