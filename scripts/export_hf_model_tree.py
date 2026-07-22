#!/usr/bin/env python3
"""Export Hugging Face model-tree relations as connected model groups.

This is a lightweight, reproducible exporter for the first-pass relationship
file used by the DNA experiments. It extracts direct base-model links from Hub
model cards and collapses them into connected components so downstream tools
can treat all linked models as related.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    source_path: Path | None = None


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def safe_model_id(raw: str) -> str:
    return raw.strip().strip('"').strip("'")


def load_model_records(path: Path) -> list[ModelRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Model list not found: {path}")

    records: list[ModelRecord] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return records

    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
        for row in reader:
            model_id = row.get("model_id") or row.get("model") or row.get("id") or ""
            model_id = safe_model_id(model_id)
            if model_id:
                records.append(ModelRecord(model_id=model_id, source_path=path))
        return records

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            payload = json.loads(line)
            model_id = payload.get("model_id") or payload.get("model") or payload.get("id") or ""
            model_id = safe_model_id(str(model_id))
            if model_id:
                records.append(ModelRecord(model_id=model_id, source_path=path))
        else:
            records.append(ModelRecord(model_id=safe_model_id(line), source_path=path))
    return records


def split_models(raw: list[str]) -> list[str]:
    out: list[str] = []
    for item in raw:
        parts = [part.strip() for part in re.split(r"[|,;]", item) if part.strip()]
        out.extend(parts)
    return out


def extract_base_models(info) -> list[str]:
    bases: list[str] = []
    card_data = getattr(info, "card_data", None)
    if card_data is not None:
        base_model = getattr(card_data, "base_model", None)
        if isinstance(base_model, str):
            bases.extend(split_models([base_model]))
        elif isinstance(base_model, (list, tuple, set)):
            bases.extend(split_models([str(item) for item in base_model]))

    config = getattr(info, "config", None)
    if isinstance(config, dict):
        for key in ["base_model_name_or_path", "base_model", "parent_model_name_or_path"]:
            value = config.get(key)
            if isinstance(value, str):
                bases.extend(split_models([value]))
            elif isinstance(value, (list, tuple, set)):
                bases.extend(split_models([str(item) for item in value]))

    cleaned: list[str] = []
    seen: set[str] = set()
    for base in bases:
        base = safe_model_id(base)
        if base and base not in seen:
            cleaned.append(base)
            seen.add(base)
    return cleaned


def build_graph(
    records: list[ModelRecord], token: str | None, include_external: bool, timeout: float | None
) -> tuple[list[list[str]], list[dict[str, str]], dict[str, object]]:
    api = HfApi(token=token)
    uf = UnionFind()
    input_models = {record.model_id for record in records}
    direct_edges: list[dict[str, str]] = []
    skipped = 0

    for record in records:
        model_id = record.model_id
        uf.add(model_id)
        try:
            info = api.model_info(model_id, timeout=timeout)
        except Exception:
            skipped += 1
            continue

        for base_model in extract_base_models(info):
            uf.union(model_id, base_model)
            direct_edges.append({"source": model_id, "target": base_model, "relation": "base_model"})

    components: dict[str, list[str]] = {}
    for node in uf.parent:
        root = uf.find(node)
        components.setdefault(root, []).append(node)

    groups: list[list[str]] = []
    for nodes in components.values():
        if include_external:
            group = sorted(nodes)
        else:
            group = sorted(node for node in nodes if node in input_models)
        if len(group) >= 2:
            groups.append(group)
    groups.sort(key=lambda nodes: (len(nodes), nodes[0].lower()))
    summary = {
        "model_count": len(records),
        "node_count": len(uf.parent),
        "edge_count": len(direct_edges),
        "component_count": len(groups),
        "skipped_models": skipped,
        "include_external": include_external,
    }
    return groups, direct_edges, summary


def build_components(records: list[ModelRecord], token: str | None, include_external: bool, timeout: float | None) -> tuple[list[list[str]], dict[str, object]]:
    """Backward-compatible connected-component API."""
    groups, _edges, summary = build_graph(records, token, include_external, timeout)
    return groups, summary


def write_jsonl(path: Path, groups: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(json.dumps({"models": group}, ensure_ascii=False) + "\n")


def write_edges_jsonl(path: Path, edges: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for edge in edges:
            handle.write(json.dumps(edge, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Hugging Face model-tree relations as JSONL groups.")
    parser.add_argument("--models-file", type=Path, required=True, help="Model list file or JSONL with model_id fields.")
    parser.add_argument("--output-file", type=Path, default=Path("out/hf_model_tree/model_relations.jsonl"))
    parser.add_argument("--edges-file", type=Path, default=Path("out/hf_model_tree/model_direct_edges.jsonl"))
    parser.add_argument("--summary-file", type=Path, default=Path("out/hf_model_tree/model_relations_summary.json"))
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Keep base models even if they are not present in the input list.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of input models to inspect.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_model_records(args.models_file)
    if args.limit is not None:
        records = records[: args.limit]
    groups, edges, summary = build_graph(records, token=args.token, include_external=args.include_external, timeout=args.timeout)
    write_jsonl(args.output_file, groups)
    write_edges_jsonl(args.edges_file, edges)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_file": str(args.output_file), "edges_file": str(args.edges_file), "summary_file": str(args.summary_file), **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
