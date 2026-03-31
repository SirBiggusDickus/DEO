"""Mutation and crossover strategies used by the optimizer."""

from __future__ import annotations

import numpy as np

from .utils import Vector


def mutate_best1(
    population: np.ndarray,
    best_vector: Vector,
    r1: int,
    r2: int,
    F: float,
) -> Vector:
    """Create a mutant vector using the DE/best/1 strategy.

    Args:
        population: Current population matrix of shape ``(pop_size, dimension)``.
        best_vector: Best individual available for the current generation.
        r1: Index of the first difference-vector parent.
        r2: Index of the second difference-vector parent.
        F: Differential weight controlling the mutation step size.

    Returns:
        The mutated candidate vector.
    """

    return np.asarray(best_vector, dtype=float) + F * (
        population[r1] - population[r2]
    )


def crossover_binomial(
    target: Vector,
    mutant: Vector,
    CR: float,
    rng: np.random.Generator,
) -> Vector:
    """Combine a target and mutant vector with binomial crossover.

    Args:
        target: Original population member being evolved.
        mutant: Mutant vector created by the mutation strategy.
        CR: Crossover probability applied independently per dimension.
        rng: Random number generator used for reproducible crossover masks.

    Returns:
        The trial vector produced by binomial crossover.
    """

    target = np.asarray(target, dtype=float)
    mutant = np.asarray(mutant, dtype=float)

    trial = target.copy()
    crossover_mask = rng.random(target.size) < CR
    forced_index = int(rng.integers(target.size))
    crossover_mask[forced_index] = True
    trial[crossover_mask] = mutant[crossover_mask]
    return trial