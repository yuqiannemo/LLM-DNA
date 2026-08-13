from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_rfftrace_experiment.py"
_SPEC = importlib.util.spec_from_file_location("run_rfftrace_experiment", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
_SPEC.loader.exec_module(experiment)


def write_run(
    root: Path,
    model: str,
    setting: experiment.Setting,
    repeat: int,
    prompts: list[str],
) -> experiment.ResponseRun:
    safe = model.replace("/", "_")
    run_dir = root / f"{safe}_{setting.code}_r{repeat}"
    run_dir.mkdir(parents=True)
    path = run_dir / "responses.json"
    path.write_text(
        json.dumps(
            {
                "model": model,
                "items": [
                    {"prompt": prompt, "response": f"{model}-{repeat}-{prompt}"}
                    for prompt in prompts
                ],
            }
        ),
        encoding="utf-8",
    )
    return experiment.load_response_run(path)


def test_load_response_run_preserves_unsanitized_model_id(tmp_path: Path) -> None:
    setting = experiment.Setting(0.2, 0.8)
    record = write_run(tmp_path, "org/model", setting, 2, ["p1", "p2"])
    assert record.model_id == "org/model"
    assert record.setting == setting
    assert record.repeat == 2


def test_common_prompts_intersects_every_selected_run(tmp_path: Path) -> None:
    setting = experiment.Setting(0.2, 0.8)
    first = write_run(tmp_path, "org/a", setting, 1, ["shared", "only-a"])
    second = write_run(tmp_path, "org/b", setting, 1, ["shared", "only-b"])
    assert experiment.common_prompts([first, second]) == ["shared"]


def test_select_models_requires_every_requested_repeat(tmp_path: Path) -> None:
    reference = experiment.Setting(0.2, 0.8)
    query = experiment.Setting(0.3, 1.0)
    records = {}
    for model in ["complete", "missing"]:
        for setting, repeats in [(reference, [1, 3]), (query, [2, 4])]:
            for repeat in repeats:
                if model == "missing" and setting == query and repeat == 4:
                    continue
                record = write_run(tmp_path, model, setting, repeat, ["p"])
                records[(model, setting, repeat)] = record
    selected = experiment.select_models(
        records,
        reference,
        [query],
        (1, 3),
        (2, 4),
        2,
        None,
        0,
    )
    assert selected == ["complete"]


def test_split_prompts_is_disjoint_and_reproducible() -> None:
    prompts = [f"p{index}" for index in range(10)]
    first = experiment.split_prompts(prompts, 0.2, 42)
    second = experiment.split_prompts(prompts, 0.2, 42)
    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]) | set(first[1]) == set(prompts)
