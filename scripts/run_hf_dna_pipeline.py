#!/usr/bin/env python3
"""
Unified DNA batch runner for HuggingFace and OpenRouter model lists.

Designed for long-running `nohup` execution:
- runs one model per subprocess for isolation
- uses at most 4 GPUs concurrently
- routes HuggingFace and OpenRouter lists through the same scheduler
- chooses `/shared/hdd` or `/shared/ssd` for model cache based on free space
- cleans HuggingFace cache when free space falls below a threshold
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import signal
import shutil
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional, cast

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_JSONL = ROOT / "configs" / "huggingface_llm_list.jsonl"
DEFAULT_OPENROUTER_JSONL = ROOT / "configs" / "openrouter_llm_list.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "out"
DEFAULT_DATA_ROOT = ROOT / "data"
DEFAULT_DATASET = "rand_chinese"
DEFAULT_MAX_SAMPLES = 100
MAX_CONCURRENT_GPUS = 64
CACHE_CANDIDATES = [Path("/shared/hdd"), Path("/shared/ssd")]
DEFAULT_GLOBAL_STATUS_JOURNAL = DEFAULT_OUTPUT_DIR / "dna_global_status.jsonl"
DEFAULT_GLOBAL_STATE = DEFAULT_OUTPUT_DIR / "dna_global_state.json"
PYTHON_BIN = (
    str(ROOT / "venv" / "bin" / "python")
    if (ROOT / "venv" / "bin" / "python").exists()
    else shutil.which("python3") or "python3"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


class ModelTask(NamedTuple):
    model_id: str
    provider: str
    inferred_size_b: Optional[float]


class GPUStat(NamedTuple):
    gpu_id: int
    util: float
    mem_used: float
    mem_total: float

    @property
    def mem_free(self) -> float:
        return max(self.mem_total - self.mem_used, 0.0)

    @property
    def mem_free_gb(self) -> float:
        return self.mem_free / 1024.0

    @property
    def mem_used_pct(self) -> float:
        return (self.mem_used / self.mem_total * 100.0) if self.mem_total else 0.0


class CacheRuntimeState:
    def __init__(
        self,
        pending_hf_models: set[str],
        running_hf_models: set[str],
        state_lock: threading.Lock,
        cleanup_lock: threading.Lock,
    ) -> None:
        self.pending_hf_models = pending_hf_models
        self.running_hf_models = running_hf_models
        self.state_lock = state_lock
        self.cleanup_lock = cleanup_lock


class TaskWork:
    def __init__(self, task: ModelTask) -> None:
        self.task = task
        self.attempts = 0
        self.attempted_gpu_ids: list[int] = []
        self.ready_at = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM-DNA extraction for HuggingFace and OpenRouter model lists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hf-jsonl", type=Path, default=DEFAULT_HF_JSONL)
    parser.add_argument("--openrouter-jsonl", type=Path, default=DEFAULT_OPENROUTER_JSONL)
    parser.add_argument(
        "--providers",
        choices=["all", "huggingface", "openrouter"],
        default="all",
        help="Which provider lists to run.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix appended to each model output directory, e.g. _2 for reruns.",
    )
    parser.add_argument(
        "--ignore-response-cache",
        action="store_true",
        help="Regenerate responses instead of reading existing responses.json; new responses are still saved.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--dna-dim", type=int, default=128)
    parser.add_argument(
        "--reduction-method",
        choices=["pca", "svd", "random_projection"],
        default="random_projection",
    )
    parser.add_argument(
        "--embedding-merge",
        choices=["sum", "max", "mean", "concat"],
        default="concat",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. Defaults to 0.0 to preserve deterministic historical runs.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling value used when sampling is enabled.",
    )
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=42,
        help="Response-sampling seed, recorded separately from the fixed projection seed.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=False,
        help="Enable stochastic sampling during generation.",
    )
    parser.add_argument(
        "--no-do-sample",
        dest="do_sample",
        action="store_false",
        help="Disable stochastic sampling during generation.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to pin scheduling. If omitted, dynamically pick from all visible GPUs.",
    )
    parser.add_argument("--max-concurrent-gpus", type=int, default=MAX_CONCURRENT_GPUS)
    parser.add_argument(
        "--min-gpu-free-gb",
        type=float,
        default=4.0,
        help="Only dispatch a new task onto GPUs with at least this much currently free memory.",
    )
    parser.add_argument(
        "--gpu-poll-seconds",
        type=float,
        default=15.0,
        help="How long to wait before re-checking GPU availability when all selected GPUs are busy.",
    )
    parser.add_argument(
        "--gpu-memory-per-billion-gb",
        type=float,
        default=2.2,
        help="Estimated GPU GiB required per billion parameters when deciding whether to dispatch.",
    )
    parser.add_argument(
        "--gpu-memory-headroom-gb",
        type=float,
        default=2.0,
        help="Additional free GPU memory required beyond the parameter-size estimate.",
    )
    parser.add_argument(
        "--oom-retries",
        type=int,
        default=3,
        help="How many times to retry failed_oom runs (preferably on a different GPU).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum deferred retries for transient non-OOM failures.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=300.0,
        help="Cooldown before a waiting model is tried again after other queued models.",
    )
    parser.add_argument(
        "--model-timeout-seconds",
        type=float,
        default=43200.0,
        help="Per-attempt child-process timeout (12 hours by default; 0 disables it).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit queued models after filtering (0 = no limit).")
    parser.add_argument(
        "--cache-low-free-gb",
        type=float,
        default=110.0,
        help="Start cleaning HF cache when free space falls below this low-watermark threshold.",
    )
    parser.add_argument(
        "--cache-target-free-gb",
        type=float,
        default=140.0,
        help=(
            "Stop cleaning after free space reaches this target watermark. "
            "Set above --cache-low-free-gb to enable hysteresis and reduce repeated cleanups."
        ),
    )
    parser.add_argument(
        "--cache-subdir",
        type=str,
        default="llm-dna-cache",
        help="Subdirectory created under /shared/hdd or /shared/ssd for caches.",
    )
    parser.add_argument(
        "--cache-evict-finished",
        choices=["never", "success", "all"],
        default="success",
        help=(
            "Delete the just-finished HuggingFace model cache directory only after verified success "
            "(non-empty responses + DNA file + summary file): "
            "never=disabled, success=verified-success only, all=verified-success or failed_* runs."
        ),
    )
    parser.add_argument("--metadata-file", type=Path, default=ROOT / "configs" / "llm_metadata.json")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--resume-status",
        type=Path,
        default=None,
        help="Optional existing status.jsonl to resume from.",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from the latest run status.jsonl under the output directory when --resume-status is not provided.",
    )
    parser.add_argument(
        "--resume-mode",
        choices=["new", "continue", "retry-failed", "retry-non-success", "all"],
        default="new",
        help=(
            "`new`: run models not yet globally successful; "
            "`continue`: continue a previous run from its unfinished tail; "
            "`retry-failed`: rerun failed models; "
            "`retry-non-success`: rerun failed and skipped models; "
            "`all`: ignore prior state."
        ),
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Deprecated alias for --resume-mode retry-failed.",
    )
    parser.add_argument("--global-status-journal", type=Path, default=DEFAULT_GLOBAL_STATUS_JOURNAL)
    parser.add_argument("--global-state", type=Path, default=DEFAULT_GLOBAL_STATE)
    parser.add_argument(
        "--try-vllm",
        action="store_true",
        help="Try vLLM first for decoder-only HuggingFace models.",
    )
    parser.add_argument(
        "--stream-subprocess-logs",
        action="store_true",
        help="Stream child model subprocess stdout/stderr into the main pipeline log.",
    )
    parser.add_argument(
        "--no-stream-subprocess-logs",
        dest="stream_subprocess_logs",
        action="store_false",
        help="Disable live streaming of child model subprocess logs.",
    )
    parser.add_argument(
        "--child-log-mode",
        choices=["cache-only", "verbose"],
        default="cache-only",
        help="How much child subprocess output to stream.",
    )
    parser.add_argument(
        "--openrouter-use-gpu",
        action="store_true",
        help="Allow OpenRouter tasks to use GPU scheduling (otherwise they run on CPU).",
    )
    parser.set_defaults(stream_subprocess_logs=True)
    return parser.parse_args()


def infer_param_billion(model_id: str) -> Optional[float]:
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", model_id)
    if not matches:
        return None
    return max(float(value) for value in matches)


def load_tasks(jsonl_path: Path, provider: str) -> list[ModelTask]:
    tasks: list[ModelTask] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            model_id = str(record["model_id"]).strip()
            if not model_id:
                continue
            tasks.append(
                ModelTask(
                    model_id=model_id,
                    provider=provider,
                    inferred_size_b=infer_param_billion(model_id),
                )
            )
    return tasks


def load_all_tasks(args: argparse.Namespace) -> list[ModelTask]:
    tasks: list[ModelTask] = []
    if args.providers in {"all", "huggingface"}:
        tasks.extend(load_tasks(args.hf_jsonl, "huggingface"))
    if args.providers in {"all", "openrouter"}:
        tasks.extend(load_tasks(args.openrouter_jsonl, "openrouter"))

    deduped: list[ModelTask] = []
    seen: set[tuple[str, str]] = set()
    for task in tasks:
        key = (task.provider, task.model_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(task)
    return deduped


def parse_gpu_ids(raw: Optional[str], max_concurrent: int) -> list[int]:
    if raw:
        parsed = [int(part.strip()) for part in raw.split(",") if part.strip()]
        return parsed[:max_concurrent]
    return []


def query_gpu_stats() -> list[GPUStat]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        logger.warning("Failed to query GPU state: %s", exc)
        return []

    if result.returncode != 0:
        logger.warning("nvidia-smi returned non-zero exit code: %s", result.returncode)
        return []

    stats: list[GPUStat] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        stats.append(
            GPUStat(
                gpu_id=int(parts[0]),
                util=float(parts[1]),
                mem_used=float(parts[2]),
                mem_total=float(parts[3]),
            )
        )
    return stats


def _score_gpu(stat: GPUStat) -> tuple[float, float, float, int]:
    return (stat.mem_used_pct, stat.util, -stat.mem_free, stat.gpu_id)


def choose_best_gpu(
    stats: Iterable[GPUStat],
    allowed_gpu_ids: Iterable[int],
    busy_gpu_ids: Iterable[int],
    min_free_gb: float = 0.0,
) -> Optional[int]:
    allowed = set(allowed_gpu_ids)
    busy = set(busy_gpu_ids)
    candidates = [
        stat
        for stat in stats
        if stat.gpu_id in allowed and stat.gpu_id not in busy and stat.mem_free_gb >= min_free_gb
    ]
    if not candidates:
        return None
    candidates.sort(key=_score_gpu)
    return candidates[0].gpu_id


def select_best_gpus(limit: int) -> list[int]:
    stats = query_gpu_stats()
    if not stats:
        return []
    stats.sort(key=_score_gpu)
    return [stat.gpu_id for stat in stats[:limit]]


def required_gpu_free_gb(task: ModelTask, args: argparse.Namespace) -> float:
    configured_floor = max(float(args.min_gpu_free_gb), 0.0)
    if task.inferred_size_b is None:
        return configured_floor
    estimated = (
        task.inferred_size_b * max(float(args.gpu_memory_per_billion_gb), 0.0)
        + max(float(args.gpu_memory_headroom_gb), 0.0)
    )
    return max(configured_floor, estimated)


def safe_model_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.strip("/"))


def build_expected_output_path(output_dir: Path, dataset: str, model_id: str, output_suffix: str = "") -> Path:
    name = f"{safe_model_name(model_id)}{output_suffix}"
    return output_dir / dataset / name / f"{name}_dna.json"


def build_expected_summary_path(output_dir: Path, dataset: str, model_id: str, output_suffix: str = "") -> Path:
    name = f"{safe_model_name(model_id)}{output_suffix}"
    return output_dir / dataset / name / f"{name}_summary.json"


def build_expected_response_cache_path(output_dir: Path, dataset: str, model_id: str, output_suffix: str = "") -> Path:
    name = f"{safe_model_name(model_id)}{output_suffix}"
    return output_dir / dataset / name / "responses.json"


def compute_dna_distance_to_base(
    output_dir: Path,
    dataset: str,
    model_id: str,
    output_suffix: str,
) -> Optional[dict[str, float | str]]:
    if not output_suffix:
        return None

    rerun_path = build_expected_output_path(output_dir, dataset, model_id, output_suffix)
    base_path = build_expected_output_path(output_dir, dataset, model_id, "")
    if not rerun_path.exists() or not base_path.exists():
        return None

    try:
        rerun_payload = json.loads(rerun_path.read_text(encoding="utf-8"))
        base_payload = json.loads(base_path.read_text(encoding="utf-8"))
        rerun_vector = np.asarray(rerun_payload["signature"], dtype=np.float32)
        base_vector = np.asarray(base_payload["signature"], dtype=np.float32)
    except Exception as exc:
        logger.warning("Failed to load DNA vectors for distance comparison: %s", exc)
        return None

    if rerun_vector.shape != base_vector.shape:
        return {
            "base_output_path": str(base_path),
            "rerun_output_path": str(rerun_path),
            "error": f"shape mismatch: base={base_vector.shape} rerun={rerun_vector.shape}",
        }

    norms = float(np.linalg.norm(base_vector) * np.linalg.norm(rerun_vector))
    cosine = 1.0 if norms == 0.0 else float(1.0 - np.dot(base_vector, rerun_vector) / norms)
    return {
        "base_output_path": str(base_path),
        "rerun_output_path": str(rerun_path),
        "cosine": cosine,
        "euclidean": float(np.linalg.norm(base_vector - rerun_vector)),
    }


def choose_cache_root(cache_subdir: str) -> Path:
    candidates: list[tuple[int, Path]] = []
    for base in CACHE_CANDIDATES:
        if not base.exists():
            continue
        try:
            usage = shutil.disk_usage(base)
        except OSError:
            continue
        candidates.append((usage.free, base))
    if not candidates:
        raise RuntimeError("Neither /shared/hdd nor /shared/ssd is available.")
    _, best_base = max(candidates, key=lambda item: item[0])
    return best_base / cache_subdir


def build_cache_env(cache_root: Path) -> dict[str, str]:
    hf_home = cache_root / "hf_home"
    torch_home = cache_root / "torch_home"
    sentence_home = cache_root / "sentence_transformers"
    xdg_cache_home = cache_root / "xdg_cache"
    hf_modules_cache = cache_root / "hf_modules"
    triton_cache = cache_root / "triton"
    tmp_dir = cache_root / "tmp"
    reptrace_cache = cache_root / "reptrace_cache"
    for path in (hf_home, torch_home, sentence_home, xdg_cache_home, hf_modules_cache, triton_cache, tmp_dir, reptrace_cache):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HOME"] = str(hf_home)
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    env["TRANSFORMERS_CACHE"] = str(hf_home / "hub")
    env["TORCH_HOME"] = str(torch_home)
    env["SENTENCE_TRANSFORMERS_HOME"] = str(sentence_home)
    env["XDG_CACHE_HOME"] = str(xdg_cache_home)
    env["HF_MODULES_CACHE"] = str(hf_modules_cache)
    env["TRITON_CACHE_DIR"] = str(triton_cache)
    env["TMPDIR"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)
    env["TMP"] = str(tmp_dir)
    env["REPTRACE_CACHE_DIR"] = str(reptrace_cache)
    return env


def cleanup_cache_if_needed(cache_root: Path, min_free_gb: float, target_free_gb: Optional[float] = None) -> None:
    target_free_gb = min_free_gb if target_free_gb is None else max(target_free_gb, min_free_gb)
    try:
        usage = shutil.disk_usage(cache_root.parent)
    except OSError as exc:
        logger.warning("Failed to inspect cache disk %s: %s", cache_root.parent, exc)
        return

    free_gb = usage.free / (1024 ** 3)
    if free_gb >= min_free_gb:
        return

    hub_dir = cache_root / "hf_home" / "hub"
    if not hub_dir.exists():
        return

    model_dirs = sorted(
        [path for path in hub_dir.iterdir() if path.is_dir() and path.name.startswith("models--")],
        key=lambda path: path.stat().st_mtime,
    )
    logger.info(
        "[cache] free space %.1f GB below low %.1f GB, cleaning to target %.1f GB in %s",
        free_gb,
        min_free_gb,
        target_free_gb,
        hub_dir,
    )

    for model_dir in model_dirs:
        if free_gb >= target_free_gb:
            break
        try:
            size_bytes = sum(file.stat().st_size for file in model_dir.rglob("*") if file.is_file())
            shutil.rmtree(model_dir, ignore_errors=False)
            reclaimed_gb = size_bytes / (1024 ** 3)
            free_gb += reclaimed_gb
            measured_free_gb = free_gb
            try:
                measured_free_gb = shutil.disk_usage(cache_root.parent).free / (1024 ** 3)
                free_gb = measured_free_gb
            except OSError:
                pass
            logger.info(
                "[cache] removed %s, reclaimed_est=%.1f GB, free_now=%.1f GB",
                model_dir.name,
                reclaimed_gb,
                measured_free_gb,
            )
        except Exception as exc:
            logger.warning("[cache] failed to remove %s: %s", model_dir, exc)


def model_id_to_hf_cache_dirname(model_id: str) -> str:
    normalized = model_id.strip("/").replace("/", "--")
    return f"models--{normalized}"


def delete_hf_model_cache_dir(cache_root: Path, model_id: str) -> bool:
    hub_dir = cache_root / "hf_home" / "hub"
    if not hub_dir.exists():
        return False
    model_dir = hub_dir / model_id_to_hf_cache_dirname(model_id)
    if not model_dir.exists():
        return False
    shutil.rmtree(model_dir, ignore_errors=False)
    return True


def has_complete_non_empty_responses(response_path: Path) -> bool:
    if not response_path.exists():
        logger.warning("[cache] keep cache: responses file missing at %s", response_path)
        return False

    try:
        with response_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.warning("[cache] keep cache: failed to parse responses file %s: %s", response_path, exc)
        return False

    if not isinstance(payload, dict):
        logger.warning("[cache] keep cache: invalid responses payload type at %s", response_path)
        return False

    items = payload.get("items")
    if not isinstance(items, list) or not items:
        logger.warning("[cache] keep cache: responses items missing/empty at %s", response_path)
        return False

    count = payload.get("count")
    if isinstance(count, int) and count != len(items):
        logger.warning("[cache] keep cache: responses count mismatch at %s (count=%s items=%s)", response_path, count, len(items))
        return False

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            logger.warning("[cache] keep cache: responses item %d is not an object at %s", idx, response_path)
            return False
        response = item.get("response")
        if not isinstance(response, str) or not response.strip():
            logger.warning("[cache] keep cache: responses item %d is empty at %s", idx, response_path)
            return False

    return True


def should_evict_finished_model_cache(
    task: ModelTask,
    record: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    if task.provider != "huggingface":
        return False
    if args.cache_evict_finished == "never":
        return False
    status = str(record.get("status"))
    if args.cache_evict_finished == "all":
        # Evict after any terminal failure to avoid repeatedly retaining unusable model caches.
        if status.startswith("failed_"):
            return True
    if status != "success":
        logger.info("[cache] keep cache: model=%s status=%s is not success", task.model_id, status)
        return False

    output_path = build_expected_output_path(args.output_dir, args.dataset, task.model_id, args.output_suffix)
    summary_path = build_expected_summary_path(args.output_dir, args.dataset, task.model_id, args.output_suffix)
    response_path = build_expected_response_cache_path(args.output_dir, args.dataset, task.model_id, args.output_suffix)

    if not output_path.exists():
        logger.warning("[cache] keep cache: DNA file missing for %s at %s", task.model_id, output_path)
        return False
    if not summary_path.exists():
        logger.warning("[cache] keep cache: summary file missing for %s at %s", task.model_id, summary_path)
        return False
    if not has_complete_non_empty_responses(response_path):
        logger.warning("[cache] keep cache: response validation failed for %s", task.model_id)
        return False

    return True


def cleanup_cache_if_needed_smart(
    cache_root: Path,
    min_free_gb: float,
    target_free_gb: float,
    protected_dirnames: set[str],
    cleanup_lock: threading.Lock,
) -> None:
    with cleanup_lock:
        try:
            usage = shutil.disk_usage(cache_root.parent)
        except OSError as exc:
            logger.warning("Failed to inspect cache disk %s: %s", cache_root.parent, exc)
            return

        free_gb = usage.free / (1024 ** 3)
        if free_gb >= min_free_gb:
            return

        hub_dir = cache_root / "hf_home" / "hub"
        if not hub_dir.exists():
            return

        model_dirs = sorted(
            [path for path in hub_dir.iterdir() if path.is_dir() and path.name.startswith("models--")],
            key=lambda path: path.stat().st_mtime,
        )
        logger.info(
            "[cache] free space %.1f GB below low %.1f GB, cleaning to target %.1f GB in %s",
            free_gb,
            min_free_gb,
            target_free_gb,
            hub_dir,
        )

        removed_any = False
        for model_dir in model_dirs:
            if free_gb >= target_free_gb:
                break
            if model_dir.name in protected_dirnames:
                continue
            try:
                size_bytes = sum(file.stat().st_size for file in model_dir.rglob("*") if file.is_file())
                shutil.rmtree(model_dir, ignore_errors=False)
                reclaimed_gb = size_bytes / (1024 ** 3)
                free_gb += reclaimed_gb
                measured_free_gb = free_gb
                try:
                    measured_free_gb = shutil.disk_usage(cache_root.parent).free / (1024 ** 3)
                    free_gb = measured_free_gb
                except OSError:
                    pass
                removed_any = True
                logger.info(
                    "[cache] removed %s, reclaimed_est=%.1f GB, free_now=%.1f GB",
                    model_dir.name,
                    reclaimed_gb,
                    measured_free_gb,
                )
            except Exception as exc:
                logger.warning("[cache] failed to remove %s: %s", model_dir, exc)

        if not removed_any and free_gb < target_free_gb:
            logger.warning(
                "[cache] low space persists (%.1f GB < %.1f GB target) but no removable model cache found after protection.",
                free_gb,
                target_free_gb,
            )


def classify_failure(output: str, timed_out: bool, return_code: Optional[int] = None) -> str:
    text = output or ""
    lines = [line.strip() for line in text.replace("\r", "\n").splitlines() if line.strip()]
    filtered_lines = [
        line
        for line in lines
        if "resource_tracker" not in line.lower()
        and "deprecationwarning" not in line.lower()
    ]
    analysis_text = "\n".join(filtered_lines) if filtered_lines else text
    lower_text = analysis_text.lower()

    if timed_out:
        return "failed_timeout"
    if return_code in {-15, 143}:
        return "failed_killed"

    # Explicit terminal conditions first.
    if "returned only empty responses" in lower_text:
        return "failed_empty_response"
    if "access denied" in lower_text or "gated repository" in lower_text or "is gated" in lower_text:
        return "failed_gated"
    if "authentication" in lower_text or "api key" in lower_text or "unauthorized" in lower_text:
        return "failed_auth"
    if "no space left on device" in lower_text or "disk quota exceeded" in lower_text:
        return "failed_disk_full"
    if "cuda out of memory" in lower_text or "outofmemoryerror" in lower_text or "memoryerror" in lower_text:
        return "failed_oom"
    if "unsupported architecture" in lower_text or "requires newer transformers version" in lower_text:
        return "failed_unsupported_arch"
    if "not a string" in lower_text and ("gguf" in lower_text or "mlx" in lower_text):
        return "failed_unsupported_format"
    if "file not found" in lower_text or "filenotfounderror" in lower_text or "no such file or directory" in lower_text:
        return "failed_missing_file"
    if "[errno 5]" in lower_text or "input/output error" in lower_text:
        return "failed_io_error"
    if "resource_tracker" in lower_text:
        return "failed_process_teardown"

    # Then use Python exception class names from traceback lines when available.
    exception_candidates = re.findall(r"([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\s*:", analysis_text)
    exception_name = exception_candidates[-1].split(".")[-1] if exception_candidates else ""
    exception_status_map = {
        "FileNotFoundError": "failed_missing_file",
        "IsADirectoryError": "failed_missing_file",
        "NotADirectoryError": "failed_missing_file",
        "PermissionError": "failed_io_error",
        "OSError": "failed_io_error",
        "IOError": "failed_io_error",
        "TimeoutError": "failed_network",
        "ReadTimeout": "failed_network",
        "ConnectTimeout": "failed_network",
        "ConnectionError": "failed_network",
        "HTTPError": "failed_network",
        "TypeError": "failed_type_error",
        "ValueError": "failed_value_error",
        "MemoryError": "failed_oom",
        "OutOfMemoryError": "failed_oom",
        "CudaOutOfMemoryError": "failed_oom",
    }
    mapped = exception_status_map.get(exception_name)
    if mapped:
        return mapped

    if re.search(r"\b(connection reset|connection refused|name resolution|dns|ssl|certificate verify failed)\b", lower_text):
        return "failed_network"
    if re.search(r"\b(read timeout|connect timeout|timed out|network error)\b", lower_text):
        return "failed_network"

    return "failed"


def error_bucket_from_status(status: str) -> str:
    return status if status.startswith("failed_") else "failed"


RETRYABLE_FAILURES = {
    "failed",
    "failed_timeout",
    "failed_killed",
    "failed_oom",
    "failed_network",
    "failed_disk_full",
    "failed_io_error",
    "failed_process_teardown",
    "failed_missing_file",
    "failed_empty_response",
}


def is_retryable_failure(status: str) -> bool:
    """Return whether another attempt could plausibly succeed without a code/config change."""
    return status in RETRYABLE_FAILURES


def task_key(provider: str, model_id: str) -> str:
    return f"{provider}::{model_id}"


def find_latest_status_file(output_dir: Path) -> Optional[Path]:
    candidates = sorted(
        [path for path in output_dir.glob("*/status.jsonl") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_latest_records(status_path: Optional[Path]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if status_path is None or not status_path.exists():
        return latest
    with status_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = task_key(str(record.get("provider", "")), str(record.get("model_id", "")))
            latest[key] = record
    return latest


def load_global_state(state_path: Path) -> dict[str, dict]:
    if not state_path.exists():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Failed to parse global state file: %s", state_path)
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def write_global_state(state_path: Path, state: dict[str, dict]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def should_enqueue(task: ModelTask, args: argparse.Namespace, run_state: dict[str, dict], global_state: dict[str, dict]) -> bool:
    key = task_key(task.provider, task.model_id)
    run_record = run_state.get(key)
    global_record = global_state.get(key)
    mode = "retry-failed" if args.retry_failures else args.resume_mode

    if mode == "all":
        return True
    if mode == "continue":
        return run_record is None or str(run_record.get("status", "")) == "waiting_retry"
    if mode == "retry-failed":
        scope = run_record or global_record
        if scope is None:
            return False
        status = str(scope.get("status", ""))
        return status.startswith("failed") or (
            status == "waiting_retry" and str(scope.get("last_status", "")).startswith("failed")
        )
    if mode == "retry-non-success":
        scope = run_record or global_record
        return scope is not None and str(scope.get("status", "")) != "success"
    if mode == "new":
        return global_record is None or str(global_record.get("status", "")) != "success"
    return True


def append_status(status_path: Path, record: dict, lock: threading.Lock) -> None:
    with lock:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_global_status(journal_path: Path, record: dict, lock: threading.Lock) -> None:
    with lock:
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        with journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_global_state(
    state_path: Path,
    state: dict[str, dict],
    record: dict,
    lock: threading.Lock,
) -> None:
    key = task_key(str(record["provider"]), str(record["model_id"]))
    with lock:
        state[key] = record
        write_global_state(state_path, state)


def build_command(task: ModelTask, args: argparse.Namespace, gpu_id: Optional[int]) -> list[str]:
    command = [
        PYTHON_BIN,
        "-m",
        "llm_dna.cli",
        "--model-name",
        task.model_id,
        "--model-type",
        task.provider,
        "--dataset",
        args.dataset,
        "--max-samples",
        str(args.max_samples),
        "--data-root",
        str(args.data_root),
        "--output-dir",
        str(args.output_dir),
        "--output-suffix",
        str(getattr(args, "output_suffix", "")),
        "--dna-dim",
        str(args.dna_dim),
        "--reduction-method",
        args.reduction_method,
        "--embedding-merge",
        args.embedding_merge,
        "--max-length",
        str(args.max_length),
        "--temperature",
        str(getattr(args, "temperature", 0.0)),
        "--top-p",
        str(getattr(args, "top_p", 1.0)),
        "--generation-seed",
        str(getattr(args, "generation_seed", 42)),
        "--log-level",
        "INFO",
    ]
    if args.metadata_file.exists():
        command.extend(["--metadata-file", str(args.metadata_file)])
    if bool(getattr(args, "try_vllm", False)):
        command.append("--try-vllm")
    if bool(getattr(args, "ignore_response_cache", False)):
        command.append("--ignore-response-cache")
    if bool(getattr(args, "do_sample", False)):
        command.append("--do-sample")
    else:
        command.append("--no-do-sample")
    if gpu_id is not None:
        command.extend(["--device", "cuda", "--gpus", str(gpu_id)])
    else:
        command.extend(["--device", "cpu"])
    return command


def _coerce_process_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _summarize_process_reason(output: str) -> str:
    if not output:
        return ""

    normalized = output.replace("\r", "\n")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return output.strip()

    # Prefer semantically meaningful error lines over teardown/runtime warnings.
    preferred_patterns = [
        r"\bDNA extraction failed\b",
        r"\bERROR\b",
        r"\bTraceback\b",
        r"\b(\w+(?:\.\w+)*(?:Error|Exception))\b",
    ]
    noise_patterns = [
        r"resource_tracker",
        r"DeprecationWarning",
        r"swigvarlink",
        r"^Loading checkpoint shards:",
    ]

    for pattern in preferred_patterns:
        for line in reversed(lines):
            if any(re.search(noise, line, flags=re.IGNORECASE) for noise in noise_patterns):
                continue
            if re.search(pattern, line, flags=re.IGNORECASE):
                return line

    for line in reversed(lines):
        if any(re.search(noise, line, flags=re.IGNORECASE) for noise in noise_patterns):
            continue
        return line

    return lines[-1]


def _run_subprocess_with_optional_streaming(
    command: list[str],
    cwd: str,
    env: dict[str, str],
    stream_logs: bool,
    log_prefix: str,
    child_log_mode: str,
    timeout_seconds: float,
) -> tuple[int, str, str, bool]:
    timeout = timeout_seconds if timeout_seconds > 0 else None
    if not stream_logs:
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
            return completed.returncode, completed.stdout.strip(), completed.stderr.strip(), False
        except subprocess.TimeoutExpired as exc:
            return (
                -signal.SIGKILL,
                _coerce_process_output(exc.output).strip(),
                _coerce_process_output(exc.stderr).strip(),
                True,
            )

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    stdout_lines: deque[str] = deque(maxlen=8000)
    stderr_lines: deque[str] = deque(maxlen=8000)

    def _reader(pipe: Optional[Any], sink: deque[str], level: int, stream_name: str) -> None:
        if pipe is None:
            return
        try:
            for raw_line in iter(pipe.readline, ""):
                line = raw_line.rstrip("\n")
                sink.append(line)
                if not line.strip():
                    continue
                if child_log_mode == "cache-only":
                    if "[cache-resume]" in line:
                        logging.info("%s [cache] %s", log_prefix, _format_cache_resume_line(line))
                    continue
                logging.log(level, "%s [%s] %s", log_prefix, stream_name, line)
        finally:
            pipe.close()

    out_thread = threading.Thread(
        target=_reader,
        args=(process.stdout, stdout_lines, logging.INFO, "stdout"),
        daemon=True,
    )
    err_thread = threading.Thread(
        target=_reader,
        args=(process.stderr, stderr_lines, logging.WARNING, "stderr"),
        daemon=True,
    )
    out_thread.start()
    err_thread.start()
    timed_out = False
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        logger.warning("%s exceeded %.0fs timeout; terminating process group", log_prefix, timeout_seconds)
        try:
            os.killpg(process.pid, signal.SIGTERM)
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            return_code = process.wait()
        except ProcessLookupError:
            return_code = process.wait()
    out_thread.join()
    err_thread.join()

    stdout = "\n".join(stdout_lines).strip()
    stderr = "\n".join(stderr_lines).strip()
    return return_code, stdout, stderr, timed_out


def _extract_cache_resume_line(output: str) -> str:
    if not output:
        return ""

    normalized = output.replace("\r", "\n")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    cache_lines = [line for line in lines if "[cache-resume]" in line]
    return cache_lines[-1] if cache_lines else ""


def _format_cache_resume_line(line: str) -> str:
    found_match = re.search(r"found=(\d+)", line)
    total_match = re.search(r"total=(\d+)", line)
    action_match = re.search(r"action=([a-zA-Z0-9_-]+)", line)
    missing_match = re.search(r"missing=(\d+)", line)

    found = found_match.group(1) if found_match else "0"
    total = total_match.group(1) if total_match else "0"
    action = action_match.group(1) if action_match else "unknown"
    missing = missing_match.group(1) if missing_match else None

    if missing is not None:
        return f"cache found={found}/{total} action={action} missing={missing}"
    return f"cache found={found}/{total} action={action}"


def is_vllm_available(python_bin: str) -> bool:
    try:
        completed = subprocess.run(
            [python_bin, "-c", "import vllm"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        return completed.returncode == 0
    except Exception:
        return False


def run_task(
    task: ModelTask,
    args: argparse.Namespace,
    gpu_id: Optional[int],
    cache_state: Optional[CacheRuntimeState] = None,
) -> dict:
    start = time.time()
    cache_root = choose_cache_root(args.cache_subdir)
    
    # Add detailed logging for OpenRouter models
    if task.provider == "openrouter":
        logger.info(f"[openrouter-task] Starting OpenRouter model: {task.model_id}")
        logger.info(f"[openrouter-task] Dataset: {args.dataset}, Max samples: {args.max_samples}")
        logger.info(f"[openrouter-task] Cache root: {cache_root}")
    
    if task.provider == "huggingface" and cache_state is not None:
        with cache_state.state_lock:
            protected_model_ids = set(cache_state.pending_hf_models) | set(cache_state.running_hf_models)
        protected_dirnames = {model_id_to_hf_cache_dirname(mid) for mid in protected_model_ids}
        cleanup_cache_if_needed_smart(
            cache_root,
            args.cache_low_free_gb,
            args.cache_target_free_gb,
            protected_dirnames,
            cache_state.cleanup_lock,
        )
    elif task.provider == "huggingface":
        cleanup_cache_if_needed(
            cache_root,
            args.cache_low_free_gb,
            float(getattr(args, "cache_target_free_gb", args.cache_low_free_gb)),
        )
    else:
        # OpenRouter/API providers do not require local HF model-cache cleanup.
        logger.debug("[cache] skip cleanup for provider=%s model=%s", task.provider, task.model_id)
    env = build_cache_env(cache_root)
    command = build_command(task, args, gpu_id)
    output_suffix = str(getattr(args, "output_suffix", ""))
    expected_output = build_expected_output_path(args.output_dir, args.dataset, task.model_id, output_suffix)
    expected_summary = build_expected_summary_path(args.output_dir, args.dataset, task.model_id, output_suffix)
    return_code, stdout, stderr, timed_out = _run_subprocess_with_optional_streaming(
        command,
        cwd=str(ROOT),
        env=env,
        stream_logs=bool(getattr(args, "stream_subprocess_logs", False)),
        log_prefix=f"[child model={task.model_id} gpu={gpu_id}]",
        child_log_mode=str(getattr(args, "child_log_mode", "cache-only")),
        timeout_seconds=float(getattr(args, "model_timeout_seconds", getattr(args, "timeout", 43200.0))),
    )
    combined = "\n".join(part for part in [stdout, stderr] if part).strip()
    cache_resume = _extract_cache_resume_line(combined)
    
    # Log OpenRouter task result
    if task.provider == "openrouter":
        logger.info(f"[openrouter-task] {task.model_id} completed with exit code: {return_code}")
        if return_code != 0:
            logger.warning(f"[openrouter-task] {task.model_id} failed with return code {return_code}")
            # Log first 500 chars of error output for debugging
            error_snippet = combined[:500] if combined else "(no output)"
            logger.warning(f"[openrouter-task] {task.model_id} error snippet: {error_snippet}")

    # Treat missing artifacts or empty responses as terminal failures, even if child exits 0.
    if return_code == 0:
        if not expected_output.exists() or not expected_summary.exists():
            record = {
                "provider": task.provider,
                "model_id": task.model_id,
                "status": "failed_missing_file",
                "reason": "exit=0 missing expected DNA/summary outputs",
                "gpu_id": gpu_id,
                "elapsed_seconds": time.time() - start,
                "output_path": str(expected_output),
                "summary_path": str(expected_summary),
                "cache_root": str(cache_root),
            }
            if cache_resume:
                record["cache_resume"] = cache_resume
            return record

        response_path = build_expected_response_cache_path(args.output_dir, args.dataset, task.model_id, output_suffix)
        if not has_complete_non_empty_responses(response_path):
            record = {
                "provider": task.provider,
                "model_id": task.model_id,
                "status": "failed_empty_response",
                "reason": "exit=0 responses missing/invalid/empty",
                "gpu_id": gpu_id,
                "elapsed_seconds": time.time() - start,
                "output_path": str(expected_output),
                "summary_path": str(expected_summary),
                "cache_root": str(cache_root),
            }
            if cache_resume:
                record["cache_resume"] = cache_resume
            return record

    if return_code == 0:
        record = {
            "provider": task.provider,
            "model_id": task.model_id,
            "status": "success",
            "reason": "ok",
            "gpu_id": gpu_id,
            "elapsed_seconds": time.time() - start,
            "output_path": str(expected_output if expected_output.exists() else expected_output),
            "summary_path": str(expected_summary if expected_summary.exists() else expected_summary),
            "cache_root": str(cache_root),
        }
        if cache_resume:
            record["cache_resume"] = cache_resume
        dna_distance = compute_dna_distance_to_base(
            args.output_dir,
            args.dataset,
            task.model_id,
            output_suffix,
        )
        if dna_distance is not None:
            record["dna_distance_to_base"] = dna_distance
        if task.provider == "openrouter":
            logger.info(f"[openrouter-task] {task.model_id} succeeded in {record['elapsed_seconds']:.1f}s")
        if should_evict_finished_model_cache(task, record, args):
            try:
                if cache_state is not None:
                    with cache_state.cleanup_lock:
                        removed = delete_hf_model_cache_dir(cache_root, task.model_id)
                else:
                    removed = delete_hf_model_cache_dir(cache_root, task.model_id)
                if removed:
                    logger.info("[cache] evicted finished model cache: %s", model_id_to_hf_cache_dirname(task.model_id))
            except Exception as exc:
                logger.warning("[cache] failed to evict finished model cache for %s: %s", task.model_id, exc)
        return record
    status = classify_failure(combined, timed_out=timed_out, return_code=return_code)
    reason = _summarize_process_reason(combined)
    
    # Extra logging for OpenRouter failures
    if task.provider == "openrouter" and status.startswith("failed"):
        logger.error(f"[openrouter-task] {task.model_id} failed with status '{status}': {reason}")
    
    record = {
        "provider": task.provider,
        "model_id": task.model_id,
        "status": status,
        "reason": f"exit={return_code} {reason}".strip(),
        "gpu_id": gpu_id,
        "elapsed_seconds": time.time() - start,
        "output_path": None,
        "summary_path": None,
        "cache_root": str(cache_root),
    }
    if cache_resume:
        record["cache_resume"] = cache_resume
    return record


def make_run_dir(output_dir: Path, run_name: Optional[str]) -> Path:
    run_id = run_name or f"dna_run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def acquire_gpu_id(
    allowed_gpu_ids: Optional[list[int]],
    reserved_gpu_ids: set[int],
    reservation_cond: threading.Condition,
    min_free_gb: float,
    poll_seconds: float,
    excluded_gpu_ids: Optional[set[int]] = None,
) -> int:
    excluded = excluded_gpu_ids or set()
    while True:
        with reservation_cond:
            stats = query_gpu_stats()
            allowed = allowed_gpu_ids if allowed_gpu_ids else [stat.gpu_id for stat in stats]
            gpu_id = choose_best_gpu(
                stats=stats,
                allowed_gpu_ids=[gid for gid in allowed if gid not in excluded],
                busy_gpu_ids=reserved_gpu_ids,
                min_free_gb=min_free_gb,
            )
            if gpu_id is None and excluded:
                # An untried GPU is preferred, but do not wait forever when only a
                # previously attempted card has enough memory for this model.
                gpu_id = choose_best_gpu(
                    stats=stats,
                    allowed_gpu_ids=allowed,
                    busy_gpu_ids=reserved_gpu_ids,
                    min_free_gb=min_free_gb,
                )
            if gpu_id is not None:
                reserved_gpu_ids.add(gpu_id)
                return gpu_id

            logger.info(
                "All selected GPUs are busy or below %.1f GiB free; waiting %.1fs before retry.",
                min_free_gb,
                poll_seconds,
            )
            reservation_cond.wait(timeout=max(poll_seconds, 1.0))


def release_gpu_id(gpu_id: int, reserved_gpu_ids: set[int], reservation_cond: threading.Condition) -> None:
    with reservation_cond:
        reserved_gpu_ids.discard(gpu_id)
        reservation_cond.notify_all()


def save_run_manifest(run_dir: Path, tasks: list[ModelTask], args: argparse.Namespace) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(),
        "providers": args.providers,
        "dataset": args.dataset,
        "output_suffix": args.output_suffix,
        "ignore_response_cache": args.ignore_response_cache,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
        "top_p": args.top_p,
        "generation_seed": args.generation_seed,
        "max_samples": args.max_samples,
        "resume_mode": "retry-failed" if args.retry_failures else args.resume_mode,
        "model_timeout_seconds": args.model_timeout_seconds,
        "max_retries": args.max_retries,
        "oom_retries": args.oom_retries,
        "retry_delay_seconds": args.retry_delay_seconds,
        "gpu_memory_per_billion_gb": args.gpu_memory_per_billion_gb,
        "gpu_memory_headroom_gb": args.gpu_memory_headroom_gb,
        "task_count": len(tasks),
        "tasks": [
            {
                "provider": task.provider,
                "model_id": task.model_id,
                "inferred_size_b": task.inferred_size_b,
            }
            for task in tasks
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def queue_tasks(tasks: Iterable[ModelTask]) -> queue.Queue[Optional[TaskWork]]:
    q: queue.Queue[Optional[TaskWork]] = queue.Queue()
    for task in tasks:
        q.put(TaskWork(task=task))
    return q


def main() -> int:
    args = parse_args()
    if args.cache_target_free_gb < args.cache_low_free_gb:
        logger.warning(
            "cache_target_free_gb (%.1f) < cache_low_free_gb (%.1f); clamping target to low.",
            args.cache_target_free_gb,
            args.cache_low_free_gb,
        )
        args.cache_target_free_gb = args.cache_low_free_gb
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.try_vllm:
        vllm_available = is_vllm_available(PYTHON_BIN)
        logger.info("vLLM availability check: %s", "available" if vllm_available else "unavailable")
        if not vllm_available:
            logger.warning("vLLM is not installed in %s; disabling --try-vllm for this run.", PYTHON_BIN)
            args.try_vllm = False
    mode = "retry-failed" if args.retry_failures else args.resume_mode
    resume_status = args.resume_status
    if resume_status is None and args.resume_latest:
        resume_status = find_latest_status_file(args.output_dir)

    if resume_status is not None:
        run_dir = resume_status.parent
        status_path = resume_status
    else:
        run_dir = make_run_dir(args.output_dir, args.run_name)
        status_path = run_dir / "status.jsonl"

    run_state = load_latest_records(status_path)
    global_state = load_global_state(args.global_state)

    tasks = [task for task in load_all_tasks(args) if should_enqueue(task, args, run_state, global_state)]
    if args.limit > 0:
        tasks = tasks[: args.limit]

    if resume_status is None or not (run_dir / "manifest.json").exists():
        save_run_manifest(run_dir, tasks, args)

    max_concurrent = min(args.max_concurrent_gpus, MAX_CONCURRENT_GPUS)
    fixed_gpu_ids = parse_gpu_ids(args.gpus, max_concurrent)
    dynamic_gpu_pool = not bool(args.gpus)
    if dynamic_gpu_pool:
        detected_gpu_count = len(query_gpu_stats())
        use_gpu_scheduler = detected_gpu_count > 0
        worker_slots = min(max_concurrent, detected_gpu_count) if detected_gpu_count > 0 else 1
    else:
        use_gpu_scheduler = bool(fixed_gpu_ids)
        worker_slots = len(fixed_gpu_ids) if fixed_gpu_ids else 1

    logger.info("run_dir=%s", run_dir)
    logger.info("status_path=%s", status_path)
    logger.info("python=%s", PYTHON_BIN)
    logger.info("providers=%s", args.providers)
    logger.info("resume_mode=%s", mode)
    logger.info("resume_status=%s", resume_status)
    logger.info("dataset=%s", args.dataset)
    logger.info("queued_models=%d", len(tasks))
    logger.info("worker_gpus=%s", "dynamic_all" if dynamic_gpu_pool else fixed_gpu_ids)
    logger.info("min_gpu_free_gb=%.1f", args.min_gpu_free_gb)
    logger.info("gpu_memory_estimate=%.1f GiB/B + %.1f GiB headroom", args.gpu_memory_per_billion_gb, args.gpu_memory_headroom_gb)
    logger.info("cache_watermark_low_gb=%.1f", args.cache_low_free_gb)
    logger.info("cache_watermark_target_gb=%.1f", args.cache_target_free_gb)
    logger.info("oom_retries=%d", max(args.oom_retries, 0))
    logger.info("max_retries=%d", max(args.max_retries, 0))
    logger.info("retry_delay_seconds=%.1f", max(args.retry_delay_seconds, 0.0))
    logger.info("model_timeout_seconds=%.1f", max(args.model_timeout_seconds, 0.0))

    if not tasks:
        logger.info("No models to run.")
        return 0

    task_queue = queue_tasks(tasks)

    status_lock = threading.Lock()
    global_state_lock = threading.Lock()
    failure_count = 0
    failure_lock = threading.Lock()
    reserved_gpu_ids: set[int] = set()
    reservation_cond = threading.Condition()
    error_summary: dict[str, dict[str, Any]] = {}
    cache_state = CacheRuntimeState(
        pending_hf_models={task.model_id for task in tasks if task.provider == "huggingface"},
        running_hf_models=set(),
        state_lock=threading.Lock(),
        cleanup_lock=threading.Lock(),
    )

    def record_error(record: dict) -> None:
        bucket = error_bucket_from_status(str(record.get("status", "failed")))
        model_id = str(record.get("model_id", ""))
        entry = error_summary.setdefault(bucket, {"count": 0, "models": []})
        entry["count"] = int(cast(int, entry["count"])) + 1
        models = cast(list[str], entry["models"])
        if model_id and model_id not in models:
            models.append(model_id)

    def worker(worker_idx: int) -> None:
        nonlocal failure_count
        while True:
            work = task_queue.get()
            if work is None:
                task_queue.task_done()
                return
            if work.ready_at > time.monotonic():
                # Keep delayed work in the queue without holding a worker for the full cooldown.
                task_queue.put(work)
                task_queue.task_done()
                time.sleep(min(max(work.ready_at - time.monotonic(), 0.05), 5.0))
                continue

            task = work.task
            gpu_id: Optional[int] = None
            if task.provider == "huggingface":
                with cache_state.state_lock:
                    cache_state.pending_hf_models.discard(task.model_id)
                    cache_state.running_hf_models.add(task.model_id)
            use_gpu_for_task = use_gpu_scheduler and (
                task.provider == "huggingface" or bool(getattr(args, "openrouter_use_gpu", False))
            )
            if use_gpu_for_task:
                task_min_free_gb = required_gpu_free_gb(task, args)
                excluded_gpu_ids = set(work.attempted_gpu_ids)
                visible_ids = set(fixed_gpu_ids) if fixed_gpu_ids else {stat.gpu_id for stat in query_gpu_stats()}
                if visible_ids and visible_ids.issubset(excluded_gpu_ids):
                    logger.info(
                        "All GPUs have been attempted for model=%s; allowing reuse after cooldown.",
                        task.model_id,
                    )
                    excluded_gpu_ids.clear()
                gpu_id = acquire_gpu_id(
                    allowed_gpu_ids=fixed_gpu_ids if fixed_gpu_ids else None,
                    reserved_gpu_ids=reserved_gpu_ids,
                    reservation_cond=reservation_cond,
                    min_free_gb=task_min_free_gb,
                    poll_seconds=args.gpu_poll_seconds,
                    excluded_gpu_ids=excluded_gpu_ids,
                )

            logger.info(
                "[start] worker=%s provider=%s gpu=%s model=%s attempt=%d",
                worker_idx,
                task.provider,
                gpu_id,
                task.model_id,
                work.attempts + 1,
            )
            attempt_started = time.time()
            try:
                record = run_task(task, args, gpu_id, cache_state)
            except Exception as exc:
                logger.exception("Unhandled task error for model=%s", task.model_id)
                record = {
                    "provider": task.provider,
                    "model_id": task.model_id,
                    "status": "failed",
                    "reason": f"scheduler exception: {type(exc).__name__}: {exc}",
                    "gpu_id": gpu_id,
                    "elapsed_seconds": time.time() - attempt_started,
                    "output_path": None,
                    "summary_path": None,
                }
            finally:
                if task.provider == "huggingface":
                    with cache_state.state_lock:
                        cache_state.running_hf_models.discard(task.model_id)
                if gpu_id is not None:
                    work.attempted_gpu_ids.append(gpu_id)
                    release_gpu_id(gpu_id, reserved_gpu_ids, reservation_cond)

            work.attempts += 1
            status = str(record.get("status", "failed"))
            max_retries = max(args.oom_retries if status == "failed_oom" else args.max_retries, 0)
            can_retry = is_retryable_failure(status) and work.attempts <= max_retries
            if can_retry:
                work.ready_at = time.monotonic() + max(args.retry_delay_seconds, 0.0)
                waiting_record = dict(record)
                waiting_record.update(
                    {
                        "status": "waiting_retry",
                        "last_status": status,
                        "attempt_count": work.attempts,
                        "max_attempts": max_retries + 1,
                        "next_attempt_after": datetime.fromtimestamp(
                            time.time() + max(args.retry_delay_seconds, 0.0)
                        ).isoformat(),
                        "attempted_gpu_ids": list(work.attempted_gpu_ids),
                        "run_dir": str(run_dir),
                        "recorded_at": datetime.now().isoformat(),
                    }
                )
                append_status(status_path, waiting_record, status_lock)
                append_global_status(args.global_status_journal, waiting_record, status_lock)
                update_global_state(args.global_state, global_state, waiting_record, global_state_lock)
                if task.provider == "huggingface":
                    with cache_state.state_lock:
                        cache_state.pending_hf_models.add(task.model_id)
                task_queue.put(work)
                logger.warning(
                    "[wait] model=%s status=%s attempt=%d/%d; deferred %.1fs behind queued work",
                    task.model_id,
                    status,
                    work.attempts,
                    max_retries + 1,
                    max(args.retry_delay_seconds, 0.0),
                )
                task_queue.task_done()
                continue

            if work.attempted_gpu_ids:
                record["attempted_gpu_ids"] = list(work.attempted_gpu_ids)
            record["attempt_count"] = work.attempts
            cache_resume = str(record.get("cache_resume", "")).strip()
            if cache_resume:
                logger.info("[cache] model=%s %s", task.model_id, cache_resume)
            record["run_dir"] = str(run_dir)
            record["recorded_at"] = datetime.now().isoformat()
            append_status(status_path, record, status_lock)
            append_global_status(args.global_status_journal, record, status_lock)
            update_global_state(args.global_state, global_state, record, global_state_lock)
            if record["status"] != "success" and not str(record["status"]).startswith("skipped_"):
                record_error(record)
            logger.info(
                "[done] worker=%s provider=%s gpu=%s model=%s status=%s elapsed=%.1fs",
                worker_idx,
                task.provider,
                gpu_id,
                task.model_id,
                record["status"],
                record["elapsed_seconds"],
            )
            if record["status"] != "success" and not record["status"].startswith("skipped_"):
                with failure_lock:
                    failure_count += 1
            task_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(idx,), daemon=False, name=f"gpu-worker-{idx}")
        for idx in range(worker_slots)
    ]
    for thread in threads:
        thread.start()
    task_queue.join()
    for _ in range(worker_slots):
        task_queue.put(None)
    for thread in threads:
        thread.join()

    error_report = {
        "generated_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "failures": error_summary,
    }
    (run_dir / "error_summary.json").write_text(json.dumps(error_report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Completed run with %d non-skipped failures.", failure_count)
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
