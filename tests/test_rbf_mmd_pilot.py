from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_rbf_mmd_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_rbf_mmd_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PILOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PILOT
SPEC.loader.exec_module(PILOT)


def test_exact_mmd_is_zero_for_identical_samples() -> None:
    samples = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert np.isclose(PILOT.exact_mmd2(samples, samples, 1.0), 0.0)


def test_same_setting_split_uses_disjoint_repeat_halves() -> None:
    setting = PILOT.SettingSpec(0.2, 0.8)
    runs = [
        PILOT.RunRecord("m", setting, repeat, Path(f"/tmp/r{repeat}"), ["p"], ["r"])
        for repeat in range(1, 5)
    ]
    reference, query = PILOT.chunk_runs_for_same_setting(runs)
    assert [run.repeat for run in reference] == [1, 2]
    assert [run.repeat for run in query] == [3, 4]


def test_select_prompts_intersects_every_selected_record() -> None:
    setting = PILOT.SettingSpec(0.2, 0.8)
    records = {
        ("a", setting.code, 1): PILOT.RunRecord(
            "a", setting, 1, Path("/tmp/a"), ["shared", "a-only"], ["", ""]
        ),
        ("b", setting.code, 1): PILOT.RunRecord(
            "b", setting, 1, Path("/tmp/b"), ["shared", "b-only"], ["", ""]
        ),
    }
    assert PILOT.select_prompts(records, ["a", "b"], [setting], 0) == ["shared"]
