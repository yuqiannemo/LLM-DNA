from __future__ import annotations

import numpy as np
import pytest

import llm_dna
from llm_dna.distributional import (
    RandomFourierMap,
    compact_rfftrace_vector,
    exact_mmd2,
    gaussian_projection_matrix,
    median_bandwidth,
    pairwise_squared_distances,
    prompt_averaged_mmd2,
    rbf_kernel,
    rfftrace_vector,
    squared_euclidean_distance,
)


def test_public_api_preserves_existing_exports() -> None:
    assert "calc_dna" in llm_dna.__all__
    assert "RandomFourierMap" in llm_dna.__all__
    assert llm_dna.RandomFourierMap is RandomFourierMap


def test_pairwise_squared_distances_has_expected_geometry() -> None:
    left = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    right = np.asarray([[0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(
        pairwise_squared_distances(left, right),
        np.asarray([[1.0], [2.0]], dtype=np.float32),
    )


def test_exact_mmd_is_zero_for_identical_samples() -> None:
    samples = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert exact_mmd2(samples, samples, 1.0) == pytest.approx(0.0, abs=1e-7)


def test_multiscale_kernel_is_mean_of_individual_kernels() -> None:
    samples = np.asarray([[0.0], [1.0]], dtype=np.float32)
    expected = (rbf_kernel(samples, samples, 0.5) + rbf_kernel(samples, samples, 2.0)) / 2
    np.testing.assert_allclose(rbf_kernel(samples, samples, [0.5, 2.0]), expected)


def test_median_bandwidth_uses_positive_pairwise_distances() -> None:
    samples = np.asarray([[0.0], [1.0], [3.0]], dtype=np.float32)
    assert median_bandwidth(samples) == pytest.approx(2.0)


def test_rff_map_is_reproducible_and_shared() -> None:
    first = RandomFourierMap.sample(3, 64, 1.5, seed=7)
    second = RandomFourierMap.sample(3, 64, 1.5, seed=7)
    np.testing.assert_array_equal(first.frequencies, second.frequencies)
    np.testing.assert_array_equal(first.phases, second.phases)


def test_rff_inner_products_approximate_rbf_kernel() -> None:
    samples = np.asarray(
        [[0.2, -0.1], [1.1, 0.4], [-0.7, 0.8]],
        dtype=np.float32,
    )
    feature_map = RandomFourierMap.sample(2, 50_000, 1.2, seed=11)
    features = feature_map.transform(samples)
    approximation = features @ features.T
    exact = rbf_kernel(samples, samples, 1.2)
    np.testing.assert_allclose(approximation, exact, atol=0.02)


def test_rfftrace_distance_approximates_prompt_averaged_mmd() -> None:
    query = np.asarray(
        [
            [[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0]],
            [[1.0, 0.9], [0.9, 1.0], [1.1, 1.0]],
        ],
        dtype=np.float32,
    )
    reference = np.asarray(
        [
            [[0.4, 0.0], [0.5, 0.1], [0.3, -0.1]],
            [[0.8, 0.5], [0.9, 0.6], [0.7, 0.4]],
        ],
        dtype=np.float32,
    )
    feature_map = RandomFourierMap.sample(2, 80_000, 0.8, seed=19)
    rff_distance = squared_euclidean_distance(
        rfftrace_vector(query, feature_map),
        rfftrace_vector(reference, feature_map),
    )
    exact_distance = prompt_averaged_mmd2(query, reference, 0.8)
    assert rff_distance == pytest.approx(exact_distance, abs=0.015)


def test_compact_projection_is_reproducible() -> None:
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(4, 2, 3)).astype(np.float32)
    feature_map = RandomFourierMap.sample(3, 16, 1.0, seed=5)
    projection_a = gaussian_projection_matrix(4 * 16, 12, seed=9)
    projection_b = gaussian_projection_matrix(4 * 16, 12, seed=9)
    np.testing.assert_array_equal(projection_a, projection_b)
    np.testing.assert_array_equal(
        compact_rfftrace_vector(samples, feature_map, projection_a),
        compact_rfftrace_vector(samples, feature_map, projection_b),
    )


def test_invalid_bandwidth_is_rejected() -> None:
    samples = np.asarray([[0.0], [1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="positive"):
        rbf_kernel(samples, samples, 0.0)
