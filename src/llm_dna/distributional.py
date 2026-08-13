"""Distributional LLM fingerprints based on RBF-MMD and random Fourier features.

The implementation follows the RFFTrace construction in ``docs/main (2).pdf``:
responses are embedded per prompt, RFF kernel features are averaged over
independent generations, prompt means are concatenated with ``1/sqrt(t)``
scaling, and an optional shared Gaussian projection produces a compact DNA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


Array = np.ndarray


def _as_float_matrix(values: Array, name: str) -> Array:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must have shape (samples, features), got {matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _as_prompt_samples(values: Array, name: str) -> Array:
    samples = np.asarray(values, dtype=np.float32)
    if samples.ndim != 3:
        raise ValueError(
            f"{name} must have shape (prompts, samples, features), got {samples.shape}"
        )
    if any(size == 0 for size in samples.shape):
        raise ValueError(f"{name} must be non-empty, got {samples.shape}")
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{name} contains non-finite values")
    return samples


def pairwise_squared_distances(a: Array, b: Array) -> Array:
    """Return all squared Euclidean distances between two row matrices."""

    left = _as_float_matrix(a, "a")
    right = _as_float_matrix(b, "b")
    if left.shape[1] != right.shape[1]:
        raise ValueError(f"Feature dimensions differ: {left.shape[1]} vs {right.shape[1]}")
    squared = (
        np.sum(left * left, axis=1, keepdims=True)
        + np.sum(right * right, axis=1, keepdims=True).T
        - 2.0 * left @ right.T
    )
    return np.maximum(squared, 0.0).astype(np.float32, copy=False)


def rbf_kernel(
    a: Array,
    b: Array,
    bandwidths: float | Iterable[float],
) -> Array:
    """Evaluate a single- or multi-scale RBF kernel.

    A sequence implements Eq. (9) by averaging equally weighted kernels.
    """

    raw = [bandwidths] if np.isscalar(bandwidths) else list(bandwidths)
    sigmas = np.asarray(raw, dtype=np.float64)
    if sigmas.ndim != 1 or sigmas.size == 0 or np.any(~np.isfinite(sigmas)):
        raise ValueError("At least one finite RBF bandwidth is required")
    if np.any(sigmas <= 0):
        raise ValueError(f"RBF bandwidths must be positive, got {sigmas.tolist()}")

    squared = pairwise_squared_distances(a, b).astype(np.float64, copy=False)
    kernels = [np.exp(-squared / (2.0 * sigma * sigma)) for sigma in sigmas]
    return np.mean(kernels, axis=0).astype(np.float32)


def exact_mmd2(
    a: Array,
    b: Array,
    bandwidths: float | Iterable[float],
) -> float:
    """Biased/V-statistic squared MMD used in Eqs. (6)--(7).

    The diagonal terms are intentionally retained: RFF mean-vector distances
    approximate this estimator exactly as the feature dimension grows.
    """

    left = _as_float_matrix(a, "a")
    right = _as_float_matrix(b, "b")
    value = (
        float(rbf_kernel(left, left, bandwidths).mean())
        + float(rbf_kernel(right, right, bandwidths).mean())
        - 2.0 * float(rbf_kernel(left, right, bandwidths).mean())
    )
    # Floating point cancellation may produce a tiny negative value.
    return max(value, 0.0)


def prompt_averaged_mmd2(
    query_samples: Array,
    reference_samples: Array,
    bandwidths: float | Iterable[float],
) -> float:
    """Average exact MMD over aligned prompts, as in Eq. (7)."""

    query = _as_prompt_samples(query_samples, "query_samples")
    reference = _as_prompt_samples(reference_samples, "reference_samples")
    if query.shape[0] != reference.shape[0] or query.shape[2] != reference.shape[2]:
        raise ValueError(
            "Prompt and feature dimensions must match: "
            f"{query.shape} vs {reference.shape}"
        )
    return float(
        np.mean(
            [
                exact_mmd2(query[index], reference[index], bandwidths)
                for index in range(query.shape[0])
            ]
        )
    )


def median_bandwidth(
    calibration_embeddings: Array,
    *,
    maximum_samples: int = 5_000,
    seed: int = 42,
) -> float:
    """Median pairwise-distance bandwidth on a calibration-only sample."""

    embeddings = _as_float_matrix(calibration_embeddings, "calibration_embeddings")
    if embeddings.shape[0] < 2:
        raise ValueError("Bandwidth calibration needs at least two embeddings")
    if maximum_samples > 0 and embeddings.shape[0] > maximum_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(embeddings.shape[0], size=maximum_samples, replace=False)
        embeddings = embeddings[indices]
    distances = np.sqrt(pairwise_squared_distances(embeddings, embeddings))
    upper = distances[np.triu_indices(distances.shape[0], k=1)]
    usable = upper[np.isfinite(upper) & (upper > 0)]
    if usable.size == 0:
        raise ValueError("Calibration embeddings contain no positive pairwise distance")
    return float(np.median(usable))


@dataclass(frozen=True)
class RandomFourierMap:
    """Shared RFF map for an RBF kernel, corresponding to Eq. (3)."""

    frequencies: Array
    phases: Array
    bandwidth: float
    seed: int

    @classmethod
    def sample(
        cls,
        input_dimension: int,
        feature_dimension: int,
        bandwidth: float,
        *,
        seed: int = 42,
    ) -> "RandomFourierMap":
        if input_dimension <= 0 or feature_dimension <= 0:
            raise ValueError("RFF input and feature dimensions must be positive")
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("RFF bandwidth must be finite and positive")
        rng = np.random.default_rng(seed)
        frequencies = rng.normal(
            loc=0.0,
            scale=1.0 / bandwidth,
            size=(feature_dimension, input_dimension),
        ).astype(np.float32)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=feature_dimension).astype(np.float32)
        return cls(frequencies, phases, float(bandwidth), int(seed))

    @property
    def input_dimension(self) -> int:
        return int(self.frequencies.shape[1])

    @property
    def feature_dimension(self) -> int:
        return int(self.frequencies.shape[0])

    def transform(self, embeddings: Array, *, chunk_size: int = 4096) -> Array:
        matrix = _as_float_matrix(embeddings, "embeddings")
        if matrix.shape[1] != self.input_dimension:
            raise ValueError(
                f"Expected {self.input_dimension} input features, got {matrix.shape[1]}"
            )
        if chunk_size <= 0:
            chunk_size = len(matrix)
        scale = np.sqrt(2.0 / self.feature_dimension)
        output = np.empty((len(matrix), self.feature_dimension), dtype=np.float32)
        for start in range(0, len(matrix), chunk_size):
            stop = min(start + chunk_size, len(matrix))
            angles = matrix[start:stop] @ self.frequencies.T
            angles += self.phases
            output[start:stop] = scale * np.cos(angles)
        return output


def rfftrace_prompt_means(samples: Array, feature_map: RandomFourierMap) -> Array:
    """Compute one empirical kernel mean vector per prompt (Eq. 4)."""

    prompt_samples = _as_prompt_samples(samples, "samples")
    prompts, repeats, dimensions = prompt_samples.shape
    if dimensions != feature_map.input_dimension:
        raise ValueError(
            f"Expected embedding dimension {feature_map.input_dimension}, got {dimensions}"
        )
    features = feature_map.transform(prompt_samples.reshape(prompts * repeats, dimensions))
    return features.reshape(prompts, repeats, -1).mean(axis=1).astype(np.float32)


def rfftrace_vector(samples: Array, feature_map: RandomFourierMap) -> Array:
    """Construct the uncompressed ``1/sqrt(t)`` prompt concatenation (Eq. 4)."""

    means = rfftrace_prompt_means(samples, feature_map)
    return (means.reshape(-1) / np.sqrt(means.shape[0])).astype(np.float32)


def gaussian_projection_matrix(
    input_dimension: int,
    output_dimension: int,
    *,
    seed: int = 42,
) -> Array:
    """Sample the shared compact-DNA projection in Eq. (10)."""

    if input_dimension <= 0 or output_dimension <= 0:
        raise ValueError("Projection dimensions must be positive")
    rng = np.random.default_rng(seed)
    return rng.normal(
        0.0,
        1.0 / np.sqrt(output_dimension),
        size=(output_dimension, input_dimension),
    ).astype(np.float32)


def compact_rfftrace_vector(
    samples: Array,
    feature_map: RandomFourierMap,
    projection: Optional[Array],
) -> Array:
    """Return RFFTrace directly, or its shared JL-projected compact DNA."""

    vector = rfftrace_vector(samples, feature_map)
    if projection is None:
        return vector
    matrix = np.asarray(projection, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != vector.size:
        raise ValueError(
            f"Projection must have shape (L, {vector.size}), got {matrix.shape}"
        )
    return (matrix @ vector).astype(np.float32)


def squared_euclidean_distance(a: Array, b: Array) -> float:
    left = np.asarray(a, dtype=np.float32).reshape(-1)
    right = np.asarray(b, dtype=np.float32).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Vector shapes differ: {left.shape} vs {right.shape}")
    delta = left - right
    return float(delta @ delta)
