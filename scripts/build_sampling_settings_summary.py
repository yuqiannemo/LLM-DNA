#!/usr/bin/env python3
"""Summarize temperature/top-p experiments by setting rather than run."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def cohort_for(path: Path) -> tuple[int, str]:
    text = path.as_posix()
    if "top_p_control_t05_validated100_20260719" in text:
        return 3, "Top-p control — validated 100"
    if "top_p_control_t05_20260717" in text:
        return 2, "Top-p control — pilot"
    return 1, "Temperature/top-p sweep"


def repeat_for(manifest: dict[str, Any], path: Path) -> str:
    suffix = str(manifest.get("output_suffix", ""))
    match = re.search(r"_r(\d+)$", suffix)
    if match:
        return f"r{match.group(1)}"
    if path.parent.name == "rand_chinese_0p3b_7b_temp07":
        return "prelim"
    return "run"


def read_final_statuses(path: Path) -> dict[tuple[str, str], str]:
    final: dict[tuple[str, str], str] = {}
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
                final[(str(record.get("provider", "")), str(model_id))] = str(
                    record.get("status", "")
                )
    return final


def classify(status: str) -> str:
    if status == "success":
        return "success"
    if status.startswith("failed") or status == "incompatible":
        return "fail"
    if status.startswith("skipped"):
        return "skipped"
    return "pending"


def outcome_for(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    planned = {
        (str(task.get("provider", "")), str(task["model_id"]))
        for task in manifest.get("tasks", [])
        if task.get("model_id")
    }
    final = read_final_statuses(path.parent / "status.jsonl")
    # Only count the cohort declared by the manifest. Stale status records from
    # models outside the cohort are tracked separately.
    current = {identity: status for identity, status in final.items() if identity in planned}
    counts = defaultdict(int)
    for status in current.values():
        counts[classify(status)] += 1
    unobserved = len(planned - set(current))
    pending = counts["pending"] + unobserved
    terminal = counts["success"] + counts["fail"] + counts["skipped"]
    complete = terminal == len(planned) and pending == 0
    fully_successful = complete and counts["success"] == len(planned)
    return {
        "repeat": repeat_for(manifest, path),
        "planned": len(planned),
        "success": counts["success"],
        "fail": counts["fail"],
        "skipped": counts["skipped"],
        "pending": pending,
        "unobserved": unobserved,
        "extra_status_models": len(set(final) - planned),
        "complete": complete,
        "fully_successful": fully_successful,
        "seed": manifest.get("generation_seed", ""),
        "created_at": manifest.get("created_at", ""),
        "source": path.parent.resolve().relative_to(ROOT).as_posix(),
    }


def format_outcomes(outcomes: list[dict[str, Any]], field: str) -> str:
    return "; ".join(
        f"{outcome['repeat']} {outcome[field]}/{outcome['planned']}"
        for outcome in outcomes
    )


def setting_state(outcomes: list[dict[str, Any]]) -> str:
    if all(outcome["fully_successful"] for outcome in outcomes):
        return "✅ 100% successful"
    if all(outcome["complete"] for outcome in outcomes):
        return "✅ Complete"
    complete = sum(bool(outcome["complete"]) for outcome in outcomes)
    return f"⏳ Incomplete ({complete}/{len(outcomes)} repeats complete)"


def grid_cell(row: dict[str, Any]) -> str:
    outcomes = json.loads(str(row["details_json"]))
    values = "/".join(str(outcome["success"]) for outcome in outcomes)
    return values if all(outcome["complete"] for outcome in outcomes) else f"{values}*"


def append_grid(
    markdown: list[str], title: str, rows: list[dict[str, Any]]
) -> None:
    temperatures = sorted({float(row["temperature"]) for row in rows})
    top_ps = sorted({float(row["top_p"]) for row in rows})
    lookup = {
        (float(row["temperature"]), float(row["top_p"])): grid_cell(row)
        for row in rows
    }
    markdown += [
        f"## {title}",
        "",
        "| Temperature \\\\ Top-p | "
        + " | ".join(f"{top_p:g}" for top_p in top_ps)
        + " |",
        "|---:|" + "|".join("---:" for _ in top_ps) + "|",
    ]
    for temperature in temperatures:
        cells = [
            lookup.get((temperature, top_p), "—")
            for top_p in top_ps
        ]
        markdown.append(
            f"| {temperature:g} | " + " | ".join(cells) + " |"
        )
    markdown.append("")


def main() -> None:
    grouped: dict[
        tuple[int, str, float, float], list[dict[str, Any]]
    ] = defaultdict(list)
    for path in sorted(ROOT.rglob("manifest.json")):
        if ".git" in path.parts:
            continue
        try:
            manifest = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if "temperature" not in manifest or "top_p" not in manifest:
            continue
        order, cohort = cohort_for(path)
        temperature = float(manifest["temperature"])
        top_p = float(manifest["top_p"])
        grouped[(order, cohort, temperature, top_p)].append(
            outcome_for(path, manifest)
        )

    rows: list[dict[str, Any]] = []
    for (order, cohort, temperature, top_p), outcomes in sorted(grouped.items()):
        outcomes.sort(key=lambda item: (item["repeat"], item["created_at"]))
        rows.append(
            {
                "order": order,
                "experiment": cohort,
                "temperature": temperature,
                "top_p": top_p,
                "repeats": len(outcomes),
                "successful_models": format_outcomes(outcomes, "success"),
                "failed_models": format_outcomes(outcomes, "fail"),
                "pending_models": format_outcomes(outcomes, "pending"),
                "status": setting_state(outcomes),
                "seeds": "; ".join(
                    f"{item['repeat']} {item['seed'] if item['seed'] != '' else 'not recorded'}"
                    for item in outcomes
                ),
                "sources": ";".join(item["source"] for item in outcomes),
                "details_json": json.dumps(outcomes, ensure_ascii=False),
            }
        )

    REPORT_DIR.mkdir(exist_ok=True)
    csv_path = REPORT_DIR / "sampling_settings_summary.csv"
    fields = [
        "experiment",
        "temperature",
        "top_p",
        "repeats",
        "successful_models",
        "failed_models",
        "pending_models",
        "status",
        "seeds",
        "sources",
        "details_json",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Temperature and top-p settings",
        "",
        (
            f"{len(rows)} distinct settings are shown. Repeated executions of the "
            "same setting are combined in one row. Counts use each planned model's "
            "latest status; status records outside the manifest cohort are excluded."
        ),
        "",
        (
            "A setting is **Complete** when every planned model in every repeat has "
            "a terminal success/failure result. **Incomplete** means at least one "
            "model is pending, waiting to retry, or has no status record."
        ),
        "",
        "## Success-count grids",
        "",
        (
            "Each cell is the number of successful models out of 100. Two values "
            "mean `r1/r2`; `*` marks an incomplete setting; `—` means not tested."
        ),
        "",
    ]
    for cohort in (
        "Temperature/top-p sweep",
        "Top-p control — pilot",
        "Top-p control — validated 100",
    ):
        append_grid(
            markdown,
            cohort,
            [row for row in rows if row["experiment"] == cohort],
        )
    markdown += ["## Per-setting details", ""]
    last_cohort = ""
    for row in rows:
        if row["experiment"] != last_cohort:
            if last_cohort:
                markdown.append("")
            last_cohort = str(row["experiment"])
            markdown += [
                f"### {last_cohort}",
                "",
                "| Temperature | Top-p | Repeats | Successful models | Failed models | Pending models | Current status |",
                "|---:|---:|---:|---|---|---|---|",
            ]
        markdown.append(
            f"| {row['temperature']:g} | {row['top_p']:g} | {row['repeats']} | "
            f"{row['successful_models']} | {row['failed_models']} | "
            f"{row['pending_models']} | {row['status']} |"
        )
    markdown += [
        "",
        "The companion CSV includes seeds, source directories, unobserved counts, "
        "and the complete per-repeat details.",
        "",
    ]
    (REPORT_DIR / "sampling_settings_summary.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(f"Wrote {len(rows)} distinct sampling settings from {sum(r['repeats'] for r in rows)} repeats")


if __name__ == "__main__":
    main()
