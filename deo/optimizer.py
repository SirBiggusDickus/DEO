"""Main optimizer implementation for the custom Differential Evolution module."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .plotting import LiveConvergencePlot
from .population import PopulationInitialization, initialize_population
from .strategies import crossover_binomial, mutate_best1
from .utils import (
    Bounds,
    Vector,
    clip_to_bounds,
    denormalize_vector,
    normalize_vector,
    select_distinct_indices,
    unit_bounds,
    validate_bounds,
)


class DifferentialEvolutionOptimizer:
    """Optimize a bounded objective function with DE/best/1/bin.

    The optimizer supports seed-biased initialization, cached fitness values for
    efficient selection, optional live convergence plotting, reproducible runs
    through a dedicated NumPy random number generator, internal normalization to
    a unit hypercube, and both minimization and maximization modes.
    """

    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: Sequence[tuple[float, float]],
        pop_size: int = 20,
        F: float = 0.5,
        CR: float = 0.9,
        max_generations: int = 100,
        seeds: Sequence[Sequence[float] | np.ndarray] | None = None,
        clone_ratio: float = 0.7,
        random_state: int | None = None,
        live_plot: bool = True,
        mode: str = "minimize",
        live_plot_path: str | Path | None = None,
        live_gif_path: str | Path | None = None,
        live_gif_max_frames: int = 10,
        live_gif_frame_duration_seconds: float = 0.5,
        live_plot_refresh_seconds: float = 2.0,
        show_progress: bool = True,
    ) -> None:
        """Configure the optimizer.

        Args:
            objective_function: Callable that evaluates a candidate vector and
                returns a scalar fitness value to minimize.
            bounds: Sequence of ``(lower, upper)`` tuples, one per dimension.
            pop_size: Number of individuals in the population.
            F: Differential weight used by the mutation strategy.
            CR: Binomial crossover probability in the inclusive range ``[0, 1]``.
            max_generations: Number of generations to evolve.
            seeds: Optional initial candidate vectors used for seed-biased
                population initialization.
            clone_ratio: Seed weight used in the semi-random clone formula
                ``clone_ratio * seed + (1 - clone_ratio) * random_vector`` for
                every non-seed individual created during initialization.
            random_state: Optional seed for the dedicated random number generator.
            live_plot: Whether to attempt live convergence plotting.
            mode: Optimization direction, either ``"minimize"`` or
                ``"maximize"``.
            live_plot_path: Optional HTML file path for browser-based live plot
                publishing during optimization.
            live_gif_path: Optional GIF file path that stores the published
                convergence snapshots as an animation.
            live_gif_max_frames: Maximum number of evenly spaced convergence
                checkpoints to include in the GIF animation.
            live_gif_frame_duration_seconds: Playback duration of each GIF
                frame in seconds.
            live_plot_refresh_seconds: Time between live HTML refreshes.
            show_progress: Whether to display a tqdm progress bar while the
                optimizer runs.

        The optimizer evolves candidates in an internally normalized
        ``[0, 1]^n`` search space, then converts them back to the original
        bounds before evaluating the objective function. This keeps dimensions
        with very different numeric ranges equally explorable during mutation and
        crossover.

        Raises:
            TypeError: If ``objective_function`` is not callable.
            ValueError: If a configuration value is outside its valid range.
        """

        if not callable(objective_function):
            raise TypeError("objective_function must be callable.")
        if pop_size < 3:
            raise ValueError("pop_size must be at least 3 for DE/best/1/bin.")
        if F <= 0:
            raise ValueError("F must be greater than 0.")
        if not 0.0 <= CR <= 1.0:
            raise ValueError("CR must be between 0 and 1 inclusive.")
        if max_generations < 1:
            raise ValueError("max_generations must be at least 1.")
        if not 0.0 <= clone_ratio <= 1.0:
            raise ValueError("clone_ratio must be between 0 and 1 inclusive.")
        if mode not in {"minimize", "maximize"}:
            raise ValueError("mode must be either 'minimize' or 'maximize'.")
        if live_gif_max_frames < 2:
            raise ValueError("live_gif_max_frames must be at least 2.")
        if live_gif_frame_duration_seconds <= 0:
            raise ValueError("live_gif_frame_duration_seconds must be greater than 0.")
        if live_plot_refresh_seconds <= 0:
            raise ValueError("live_plot_refresh_seconds must be greater than 0.")

        self.objective_function = objective_function
        self.bounds: Bounds = validate_bounds(bounds)
        self.search_bounds: Bounds = unit_bounds(len(self.bounds))
        self.dimension = len(self.bounds)
        self.pop_size = int(pop_size)
        self.F = float(F)
        self.CR = float(CR)
        self.max_generations = int(max_generations)
        self.seeds = list(seeds) if seeds is not None else None
        self.clone_ratio = float(clone_ratio)
        self.random_state = random_state
        self.live_plot = bool(live_plot)
        self.mode = mode
        self.live_plot_path = Path(live_plot_path) if live_plot_path is not None else None
        self.live_gif_path = Path(live_gif_path) if live_gif_path is not None else None
        self.live_gif_max_frames = int(live_gif_max_frames)
        self.live_gif_frame_duration_seconds = float(live_gif_frame_duration_seconds)
        self.live_plot_refresh_seconds = float(live_plot_refresh_seconds)
        self.show_progress = bool(show_progress)

        self.rng = np.random.default_rng(random_state)
        self.plot = LiveConvergencePlot(enabled=self.live_plot, mode=self.mode)
        self.plot.set_total_generations(self.max_generations)
        if self.live_plot and self.live_plot_path is not None:
            self.plot.configure_live_updates(
                output_path=self.live_plot_path,
                refresh_interval_seconds=self.live_plot_refresh_seconds,
            )
        if self.live_gif_path is not None:
            self.plot.configure_gif_capture(
                output_path=self.live_gif_path,
                frame_duration_seconds=self.live_gif_frame_duration_seconds,
                max_frames=self.live_gif_max_frames,
            )

        self.population: np.ndarray | None = None
        self.evaluations: np.ndarray | None = None
        self.selection_scores: np.ndarray | None = None
        self.best_vector: Vector | None = None
        self.best_evaluation: float | None = None
        self.best_fitness: float | None = None
        self.mean_evaluation: float | None = None
        self.best_fitness_history: list[float] = []
        self.mean_fitness_history: list[float] = []
        self.best_evaluation_history: list[float] = []
        self.mean_evaluation_history: list[float] = []
        self.generation_history: list[int] = []
        self.initialization: PopulationInitialization | None = None

    def evaluate(self, x: Sequence[float] | np.ndarray) -> float:
        """Evaluate a candidate vector in the original search space.

        Args:
            x: One-dimensional candidate vector to evaluate.

        Returns:
            The scalar fitness value returned by the objective function.

        Raises:
            ValueError: If the candidate dimensionality does not match the bounds.
        """

        candidate = np.asarray(x, dtype=float)
        if candidate.ndim != 1 or candidate.size != self.dimension:
            raise ValueError(
                f"Candidate vectors must be one-dimensional with {self.dimension} values."
            )

        return float(self.objective_function(candidate.copy()))

    def _evaluate_normalized(self, x: Sequence[float] | np.ndarray) -> float:
        """Evaluate a candidate vector expressed in normalized coordinates."""

        candidate = clip_to_bounds(x, self.search_bounds)
        denormalized_candidate = denormalize_vector(candidate, self.bounds)
        return self.evaluate(denormalized_candidate)

    def run(self) -> tuple[Vector, float]:
        """Run the optimizer and return the best solution found.

        Returns:
            A tuple ``(best_vector, best_evaluation)`` describing the best
            solution found during the optimization run.
        """

        self.initialization = initialize_population(
            bounds=self.search_bounds,
            pop_size=self.pop_size,
            seeds=self._normalized_seeds(),
            clone_ratio=self.clone_ratio,
            rng=self.rng,
        )
        self.population = self.initialization.population.copy()
        self.evaluations = np.array(
            [self._evaluate_normalized(candidate) for candidate in self.population],
            dtype=float,
        )
        self.selection_scores = np.array(
            [self._selection_score(value) for value in self.evaluations],
            dtype=float,
        )

        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.best_evaluation_history = []
        self.mean_evaluation_history = []
        self.generation_history = []
        self._record_history(generation=0)
        self.plot.publish_live(force=True)

        with tqdm(
            total=self.max_generations,
            disable=not self.show_progress,
            desc="DEO progress",
            unit="gen",
            dynamic_ncols=True,
        ) as progress_bar:
            for generation in range(1, self.max_generations + 1):
                best_index = self._best_index()
                best_vector = self.population[best_index].copy()

                for target_index in range(self.pop_size):
                    r1, r2 = select_distinct_indices(
                        pop_size=self.pop_size,
                        exclude=target_index,
                        rng=self.rng,
                        count=2,
                    )
                    target = self.population[target_index]
                    mutant = mutate_best1(
                        population=self.population,
                        best_vector=best_vector,
                        r1=r1,
                        r2=r2,
                        F=self.F,
                    )
                    trial = crossover_binomial(
                        target=target,
                        mutant=mutant,
                        CR=self.CR,
                        rng=self.rng,
                    )
                    trial = clip_to_bounds(trial, self.search_bounds)
                    trial_evaluation = self._evaluate_normalized(trial)
                    trial_score = self._selection_score(trial_evaluation)

                    if trial_score < self.selection_scores[target_index]:
                        self.population[target_index] = trial
                        self.evaluations[target_index] = trial_evaluation
                        self.selection_scores[target_index] = trial_score

                self._record_history(generation=generation)
                self.plot.publish_live()
                progress_bar.update(1)
                progress_bar.set_postfix_str(
                    f"generation {generation}/{self.max_generations}",
                    refresh=False,
                )

        self.plot.publish_live(force=True, final=True)

        if self.best_vector is None or self.best_evaluation is None:
            raise RuntimeError("Optimizer completed without a recorded best solution.")

        return self.best_vector.copy(), float(self.best_evaluation)

    def _record_history(self, generation: int) -> None:
        """Update best-state caches and convergence history for a generation."""

        if (
            self.population is None
            or self.evaluations is None
            or self.selection_scores is None
        ):
            raise RuntimeError("Optimizer state is not initialized.")

        best_index = self._best_index()
        self.best_vector = denormalize_vector(self.population[best_index], self.bounds)
        self.best_evaluation = float(self.evaluations[best_index])
        self.best_fitness = self.best_evaluation

        self.mean_evaluation = float(np.mean(self.evaluations))
        self.generation_history.append(int(generation))
        self.best_evaluation_history.append(self.best_evaluation)
        self.mean_evaluation_history.append(self.mean_evaluation)
        self.best_fitness_history = self.best_evaluation_history
        self.mean_fitness_history = self.mean_evaluation_history
        self.plot.update(
            generation=generation,
            best_evaluation=self.best_evaluation,
            mean_evaluation=self.mean_evaluation,
        )

    def _selection_score(self, evaluation: float) -> float:
        """Convert a raw objective evaluation to a minimization score."""

        if self.mode == "minimize":
            return float(evaluation)
        return float(-evaluation)

    def _best_index(self) -> int:
        """Return the population index of the current best candidate."""

        if self.selection_scores is None:
            raise RuntimeError("Selection scores are not initialized.")

        return int(np.argmin(self.selection_scores))

    def _normalized_seeds(self) -> list[np.ndarray] | None:
        """Return normalized seed vectors for internal search-space initialization."""

        if self.seeds is None:
            return None

        normalized: list[np.ndarray] = []
        for seed in self.seeds:
            clipped_seed = clip_to_bounds(seed, self.bounds)
            normalized.append(normalize_vector(clipped_seed, self.bounds))

        return normalized