"""Population initialization helpers for the Differential Evolution Optimizer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .utils import Bounds, Vector, clip_to_bounds, random_vector


@dataclass(frozen=True)
class PopulationInitialization:
    """Container holding the initialized population and its composition counts.

    Attributes:
        population: Initialized population matrix with shape
            ``(pop_size, dimension)``.
        seed_count: Number of deterministic seeds inserted directly.
        clone_count: Number of seed-derived semi-random clones created.
        random_count: Number of purely random individuals generated.
    """

    population: np.ndarray
    seed_count: int
    clone_count: int
    random_count: int


def initialize_population(
    bounds: Bounds,
    pop_size: int,
    seeds: Sequence[Sequence[float] | np.ndarray] | None,
    clone_ratio: float,
    rng: np.random.Generator,
) -> PopulationInitialization:
    """Build the initial population using seeds and semi-random clones.

    The implementation inserts every provided seed directly when possible.
    When seeds are present, every remaining population slot is filled with a
    semi-random clone built from one seed and one random vector:

    ``clone_ratio * seed + (1 - clone_ratio) * random_vector``

    A ``clone_ratio`` of ``1.0`` produces exact seed clones, ``0.7`` keeps
    70 percent of the seed and randomizes 30 percent, and ``0.0`` falls back to
    fully random individuals. When no seeds are supplied, the whole population
    is initialized randomly.

    Args:
        bounds: Normalized optimization bounds.
        pop_size: Number of individuals in the population.
        seeds: Optional sequence of candidate vectors used to bias the start.
        clone_ratio: Seed weight used when blending each semi-random clone.
        rng: Random number generator used for reproducible initialization.

    Returns:
        A ``PopulationInitialization`` instance containing the initialized
        population and composition counts.

    Raises:
        ValueError: If the number of seeds exceeds ``pop_size`` or if a seed has
            the wrong dimensionality.
    """

    normalized_seeds = _normalize_seeds(seeds=seeds, bounds=bounds)
    seed_count = len(normalized_seeds)
    if seed_count > pop_size:
        raise ValueError("The number of seeds cannot exceed the population size.")

    if seed_count == 0:
        clone_count = 0
        random_count = pop_size
    else:
        clone_count = pop_size - seed_count
        random_count = 0

    individuals: list[np.ndarray] = [seed.copy() for seed in normalized_seeds]

    for clone_index in range(clone_count):
        seed = normalized_seeds[clone_index % seed_count]
        blended = clone_ratio * seed + (1.0 - clone_ratio) * random_vector(bounds, rng)
        individuals.append(clip_to_bounds(blended, bounds))

    for _ in range(random_count):
        individuals.append(random_vector(bounds, rng))

    population = np.vstack(individuals)
    return PopulationInitialization(
        population=population,
        seed_count=seed_count,
        clone_count=clone_count,
        random_count=random_count,
    )


def _normalize_seeds(
    seeds: Sequence[Sequence[float] | np.ndarray] | None,
    bounds: Bounds,
) -> list[Vector]:
    """Convert provided seeds to bounded one-dimensional float arrays."""

    if not seeds:
        return []

    dimension = len(bounds)
    normalized: list[Vector] = []
    for index, seed in enumerate(seeds):
        seed_array = np.asarray(seed, dtype=float)
        if seed_array.ndim != 1 or seed_array.size != dimension:
            raise ValueError(
                f"Seed at index {index} must have exactly {dimension} values."
            )

        normalized.append(clip_to_bounds(seed_array, bounds))

    return normalized