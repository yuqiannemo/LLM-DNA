from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.svm import LinearSVC


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_rfftrace_svm.py"
SPEC = importlib.util.spec_from_file_location("evaluate_rfftrace_svm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_vectors(path: Path) -> None:
    models = np.asarray(["a", "b", "c"])
    base = np.asarray([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    np.savez_compressed(
        path,
        models=models,
        train_1=base,
        train_2=base + 0.05,
        test=base + 0.02,
    )


def test_load_matrices_rejects_split_overlap(tmp_path: Path) -> None:
    path = tmp_path / "vectors.npz"
    write_vectors(path)
    with pytest.raises(ValueError, match="disjoint"):
        MODULE.load_matrices(path, ["train_1"], ["train_1"])


def test_linear_svm_recovers_separable_held_out_vectors(tmp_path: Path) -> None:
    path = tmp_path / "vectors.npz"
    write_vectors(path)
    models, matrices = MODULE.load_matrices(path, ["train_1", "train_2"], ["test"])
    summaries, predictions = MODULE.evaluate_classifier(
        "linear_svm",
        LinearSVC(C=1.0, dual="auto", random_state=42),
        models,
        matrices,
        ["train_1", "train_2"],
        ["test"],
    )
    assert summaries[0]["top1"] == pytest.approx(1.0)
    assert all(row["correct"] == 1 for row in predictions)


def test_binary_rank_expands_single_decision_score() -> None:
    classes = np.asarray(["a", "b"])
    assert MODULE.rank_from_scores(np.asarray(-2.0), classes, "a") == 1
    assert MODULE.rank_from_scores(np.asarray(2.0), classes, "b") == 1


def test_parse_gamma_accepts_named_and_numeric_values() -> None:
    assert MODULE.parse_gamma("scale") == "scale"
    assert MODULE.parse_gamma("0.25") == pytest.approx(0.25)
