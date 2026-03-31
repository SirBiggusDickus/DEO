"""Shared helper utilities for the Differential Evolution Optimizer package."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

Bounds: TypeAlias = tuple[tuple[float, float], ...]
Vector: TypeAlias = np.ndarray


def unit_bounds(dimension: int) -> Bounds:
    """Create unit-cube bounds for an internally normalized search space.

    Args:
        dimension: Number of decision variables.

    Returns:
        A bounds tuple where every dimension spans ``[0.0, 1.0]``.

    Raises:
        ValueError: If ``dimension`` is less than one.
    """

    if dimension < 1:
        raise ValueError("dimension must be at least 1.")

    return tuple((0.0, 1.0) for _ in range(dimension))


def normalize_vector(x: Sequence[float], bounds: Bounds) -> Vector:
    """Map a candidate from original bounds into the unit hypercube.

    Args:
        x: Candidate vector in the original search space.
        bounds: Original optimization bounds.

    Returns:
        A normalized vector in the range ``[0.0, 1.0]`` per dimension.

    Raises:
        ValueError: If the candidate dimensionality does not match the bounds.
    """

    candidate = np.asarray(x, dtype=float)
    if candidate.ndim != 1 or candidate.size != len(bounds):
        raise ValueError(
            f"Candidate vectors must be one-dimensional with {len(bounds)} values."
        )

    lower_bounds = np.array([lower for lower, _ in bounds], dtype=float)
    upper_bounds = np.array([upper for _, upper in bounds], dtype=float)
    return (candidate - lower_bounds) / (upper_bounds - lower_bounds)


def denormalize_vector(x: Sequence[float], bounds: Bounds) -> Vector:
    """Map a normalized candidate from the unit hypercube into original bounds.

    Args:
        x: Candidate vector in normalized coordinates.
        bounds: Original optimization bounds.

    Returns:
        A vector expressed in the original search space.

    Raises:
        ValueError: If the candidate dimensionality does not match the bounds.
    """

    candidate = np.asarray(x, dtype=float)
    if candidate.ndim != 1 or candidate.size != len(bounds):
        raise ValueError(
            f"Candidate vectors must be one-dimensional with {len(bounds)} values."
        )

    lower_bounds = np.array([lower for lower, _ in bounds], dtype=float)
    upper_bounds = np.array([upper for _, upper in bounds], dtype=float)
    return lower_bounds + candidate * (upper_bounds - lower_bounds)


def validate_bounds(bounds: Sequence[tuple[float, float]]) -> Bounds:
    """Normalize and validate the optimization bounds.

    Args:
        bounds: Sequence of ``(lower, upper)`` tuples, one per decision
            variable.

    Returns:
        A normalized immutable tuple of ``(lower, upper)`` float pairs.

    Raises:
        ValueError: If no bounds are supplied or a bound is invalid.
    """

    if not bounds:
        raise ValueError("At least one bound is required.")

    normalized_bounds: list[tuple[float, float]] = []
    for index, bound in enumerate(bounds):
        if len(bound) != 2:
            raise ValueError(
                f"Bound at index {index} must contain exactly two values."
            )

        lower, upper = float(bound[0]), float(bound[1])
        if lower >= upper:
            raise ValueError(
                f"Lower bound must be smaller than upper bound at index {index}."
            )

        normalized_bounds.append((lower, upper))

    return tuple(normalized_bounds)


def random_vector(bounds: Bounds, rng: np.random.Generator) -> Vector:
    """Create a uniformly sampled vector inside the configured bounds.

    Args:
        bounds: Normalized optimization bounds.
        rng: Random number generator used for reproducible sampling.

    Returns:
        A one-dimensional NumPy array with one sampled value per bound.
    """

    lower_bounds = np.array([lower for lower, _ in bounds], dtype=float)
    upper_bounds = np.array([upper for _, upper in bounds], dtype=float)
    return rng.uniform(lower_bounds, upper_bounds)


def clip_to_bounds(x: Sequence[float], bounds: Bounds) -> Vector:
    """Project a candidate vector back into the feasible bound range.

    Args:
        x: Candidate vector to clip.
        bounds: Normalized optimization bounds.

    Returns:
        A clipped copy of ``x`` that respects every lower and upper bound.
    """

    lower_bounds = np.array([lower for lower, _ in bounds], dtype=float)
    upper_bounds = np.array([upper for _, upper in bounds], dtype=float)
    return np.clip(np.asarray(x, dtype=float), lower_bounds, upper_bounds)


def select_distinct_indices(
    pop_size: int,
    exclude: int,
    rng: np.random.Generator,
    count: int = 2,
) -> tuple[int, ...]:
    """Sample unique population indices while excluding a target index.

    Args:
        pop_size: Number of individuals in the population.
        exclude: Target individual index that must not be sampled.
        rng: Random number generator used for sampling.
        count: Number of distinct indices to sample.

    Returns:
        A tuple containing ``count`` unique population indices.

    Raises:
        ValueError: If there are not enough valid indices to sample.
    """

    candidates = [index for index in range(pop_size) if index != exclude]
    if len(candidates) < count:
        raise ValueError(
            "Not enough population members are available to sample distinct indices."
        )

    sampled = rng.choice(candidates, size=count, replace=False)
    return tuple(int(index) for index in sampled)