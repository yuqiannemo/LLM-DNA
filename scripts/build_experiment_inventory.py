#!/usr/bin/env python3
"""Build a reproducible inventory of experiment manifests and final statuses."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
MANIFEST_NAME = "manifest.json"
STATUS_NAME = "status.jsonl"

PARAMETER_FIELDS = [
    "dataset",
    "max_samples",
    "providers",
    "max_model_size_b",
    "resume_mode",
    "temperature",
    "top_p",
    "do_sample",
    "generation_seed",
    "ignore_response_cache",
    "output_suffix",
    "model_timeout_seconds",
    "max_retries",
    "oom_retries",
    "retry_delay_seconds",
    "gpu_memory_headroom_gb",
    "gpu_memory_per_billion_gb",
]


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_statuses(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("model_id"):
                records.append(record)
    return records


def task_identity(task: dict[str, Any]) -> str:
    provider = task.get("provider")
    model_id = task.get("model_id", "")
    return f"{provider}:{model_id}" if provider else str(model_id)


def model_label(identity: str) -> str:
    return identity.split(":", 1)[1] if ":" in identity else identity


def final_statuses(
    records: list[dict[str, Any]], planned: set[str]
) -> dict[str, dict[str, Any]]:
    """Keep the last record per model, matching provider-less legacy statuses."""
    by_model_id = {model_label(identity): identity for identity in planned}
    final: dict[str, dict[str, Any]] = {}
    for record in records:
        identity = task_identity(record)
        if identity not in planned:
            identity = by_model_id.get(str(record.get("model_id")), identity)
        final[identity] = record
    return final


def status_bucket(status: str) -> str:
    if status == "success":
        return "success"
    if status.startswith("failed"):
        return "fail"
    if status.startswith("skipped"):
        return "skipped"
    return "other"


def compact_parameters(manifest: dict[str, Any]) -> str:
    parts: list[str] = []
    aliases = {
        "max_samples": "n",
        "providers": "provider",
        "max_model_size_b": "max_B",
        "resume_mode": "resume",
        "temperature": "temp",
        "generation_seed": "seed",
        "ignore_response_cache": "no_cache",
        "model_timeout_seconds": "timeout_s",
        "retry_delay_seconds": "retry_delay_s",
        "gpu_memory_headroom_gb": "gpu_headroom_GB",
        "gpu_memory_per_billion_gb": "gpu_GB_per_B",
    }
    for field in PARAMETER_FIELDS:
        if field in manifest:
            value = manifest[field]
            if isinstance(value, bool):
                value = str(value).lower()
            parts.append(f"{aliases.get(field, field)}={value}")
    return "; ".join(parts)


def diff_summary(added: list[str], removed: list[str]) -> str:
    if not added and not removed:
        return "BASE (same 100)"
    count = len(added) + len(removed)
    if count <= 4:
        pieces = [f"+{model_label(item)}" for item in added]
        pieces += [f"−{model_label(item)}" for item in removed]
        return "; ".join(pieces)
    return f"+{len(added)} / −{len(removed)} (see model diffs CSV)"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(ROOT.rglob(MANIFEST_NAME)):
        if ".git" in path.parts:
            continue
        try:
            manifests.append((path, load_json(path)))
        except (OSError, json.JSONDecodeError):
            continue

    model_sets = [
        frozenset(task_identity(task) for task in data.get("tasks", []))
        for _, data in manifests
    ]
    frequencies = Counter(model_sets)
    baseline = set(frequencies.most_common(1)[0][0])
    baseline_frequency = frequencies.most_common(1)[0][1]

    rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []
    manifest_dirs: set[Path] = set()

    for manifest_path, manifest in manifests:
        run_dir = manifest_path.parent
        manifest_dirs.add(run_dir.resolve())
        planned = {
            task_identity(task)
            for task in manifest.get("tasks", [])
            if task.get("model_id")
        }
        records = load_statuses(run_dir / STATUS_NAME)
        final = final_statuses(records, planned)
        buckets = Counter(
            status_bucket(str(record.get("status", ""))) for record in final.values()
        )
        observed_planned = len(set(final) & planned)
        observed_total = len(final)
        extra_observed = len(set(final) - planned)
        unobserved = max(0, len(planned) - observed_planned)
        added = sorted(planned - baseline)
        removed = sorted(baseline - planned)

        known_fields = set(PARAMETER_FIELDS) | {
            "created_at",
            "tasks",
            "task_count",
        }
        extra_parameters = {
            key: value for key, value in manifest.items() if key not in known_fields
        }
        run_name = relative(run_dir)
        row = {
            "created_at": manifest.get("created_at", ""),
            "run": run_name,
            "models": len(planned),
            "observed": observed_total,
            "extra_observed": extra_observed,
            "success": buckets["success"],
            "fail": buckets["fail"],
            "skipped": buckets["skipped"],
            "other_final_status": buckets["other"],
            "unobserved": unobserved,
            "model_diff": diff_summary(added, removed),
            "parameters": compact_parameters(manifest),
            "manifest": relative(manifest_path),
            "status_log": (
                relative(run_dir / STATUS_NAME)
                if (run_dir / STATUS_NAME).exists()
                else ""
            ),
            "extra_parameters_json": json.dumps(
                extra_parameters, ensure_ascii=False, sort_keys=True
            ),
        }
        for field in PARAMETER_FIELDS:
            row[field] = manifest.get(field, "")
        rows.append(row)
        diff_rows.append(
            {
                "run": run_name,
                "is_baseline": not added and not removed,
                "model_count": len(planned),
                "added_count": len(added),
                "removed_count": len(removed),
                "added_models": ";".join(added),
                "removed_models": ";".join(removed),
            }
        )

    # Include legacy status-only runs so "all runs" is not limited to manifests.
    for status_path in sorted(ROOT.rglob(STATUS_NAME)):
        if ".git" in status_path.parts or status_path.parent.resolve() in manifest_dirs:
            continue
        records = load_statuses(status_path)
        final = final_statuses(records, set())
        models = set(final)
        buckets = Counter(
            status_bucket(str(record.get("status", ""))) for record in final.values()
        )
        summary_path = status_path.parent / "summary.json"
        summary = load_json(summary_path) if summary_path.exists() else {}
        added = sorted(models - baseline)
        removed = sorted(baseline - models)
        row = {
            "created_at": summary.get("timestamp", ""),
            "run": relative(status_path.parent),
            "models": len(models),
            "observed": len(models),
            "extra_observed": 0,
            "success": buckets["success"],
            "fail": buckets["fail"],
            "skipped": buckets["skipped"],
            "other_final_status": buckets["other"],
            "unobserved": 0,
            "model_diff": diff_summary(added, removed),
            "parameters": "; ".join(
                f"{key}={summary[key]}"
                for key in ("dataset", "samples", "gpus")
                if key in summary
            ),
            "manifest": "",
            "status_log": relative(status_path),
            "extra_parameters_json": json.dumps(
                summary, ensure_ascii=False, sort_keys=True
            ),
        }
        for field in PARAMETER_FIELDS:
            row[field] = ""
        rows.append(row)
        diff_rows.append(
            {
                "run": row["run"],
                "is_baseline": False,
                "model_count": len(models),
                "added_count": len(added),
                "removed_count": len(removed),
                "added_models": ";".join(added),
                "removed_models": ";".join(removed),
            }
        )

    rows.sort(key=lambda row: (str(row["created_at"]), str(row["run"])))
    diff_rows.sort(key=lambda row: str(row["run"]))
    REPORT_DIR.mkdir(exist_ok=True)

    csv_fields = [
        "created_at",
        "run",
        *PARAMETER_FIELDS,
        "models",
        "observed",
        "extra_observed",
        "success",
        "fail",
        "skipped",
        "other_final_status",
        "unobserved",
        "model_diff",
        "parameters",
        "manifest",
        "status_log",
        "extra_parameters_json",
    ]
    write_csv(REPORT_DIR / "experiment_inventory.csv", rows, csv_fields)
    write_csv(
        REPORT_DIR / "experiment_model_diffs.csv",
        diff_rows,
        [
            "run",
            "is_baseline",
            "model_count",
            "added_count",
            "removed_count",
            "added_models",
            "removed_models",
        ],
    )
    with (REPORT_DIR / "experiment_baseline_models.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(sorted(baseline)) + "\n")

    markdown = [
        "# Experiment inventory",
        "",
        (
            f"Generated from {len(manifests)} manifests plus "
            f"{len(rows) - len(manifests)} status-only run(s). "
            f"The baseline is the modal exact model cohort: {len(baseline)} models, "
            f"used by {baseline_frequency} manifests."
        ),
        "",
        (
            "`success`, `fail`, `skipped`, and `other` count each model's final "
            "status record (retry records are collapsed to the last record). "
            "`unobserved` means a model was planned in the manifest but has no "
            "status record; `extra` means a status model was not in the manifest. "
            "A missing status log therefore does not become a failure."
        ),
        "",
        "| Date | Run | Parameters | Planned models | Status models | Success | Fail | Skipped | Other | Unobserved | Extra | Model diff vs baseline |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        created = str(row["created_at"])[:19].replace("T", " ")
        markdown.append(
            "| {date} | `{run}` | {params} | {models} | {observed} | {success} | "
            "{fail} | {skipped} | {other} | {unobserved} | {extra} | {diff} |".format(
                date=created or "unknown",
                run=row["run"],
                params=str(row["parameters"]).replace("|", "\\|"),
                models=row["models"],
                observed=row["observed"],
                success=row["success"],
                fail=row["fail"],
                skipped=row["skipped"],
                other=row["other_final_status"],
                unobserved=row["unobserved"],
                extra=row["extra_observed"],
                diff=str(row["model_diff"]).replace("|", "\\|"),
            )
        )
    markdown += [
        "",
        "Companion files:",
        "",
        "- `experiment_inventory.csv`: all manifest parameter columns and source paths.",
        "- `experiment_model_diffs.csv`: full added/removed model lists for every run.",
        "- `experiment_baseline_models.txt`: the exact 100-model baseline cohort.",
        "",
    ]
    (REPORT_DIR / "experiment_inventory.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )

    print(
        f"Wrote {len(rows)} runs; baseline={len(baseline)} models "
        f"across {baseline_frequency} manifests"
    )


if __name__ == "__main__":
    main()
