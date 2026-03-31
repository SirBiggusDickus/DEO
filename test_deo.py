"""Minimal runnable example for the Differential Evolution Optimizer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from deo.optimizer import DifferentialEvolutionOptimizer


STRESS_BOUNDS = [
    (0.1, 0.9),
    (1.0, 3000.0),
    (-50.0, 50.0),
    (100.0, 5000.0),
    (0.001, 0.02),
    (25.0, 800.0),
    (1000.0, 200000.0),
    (-3.0, 3.0),
]

STRESS_TARGET = np.array(
    [0.63, 1800.0, -12.5, 3200.0, 0.0125, 410.0, 145000.0, 1.35],
    dtype=float,
)

STRESS_SEEDS = [
    np.array([0.15, 100.0, -45.0, 300.0, 0.002, 80.0, 5000.0, -2.5]),
    np.array([0.82, 2700.0, 40.0, 4700.0, 0.018, 700.0, 190000.0, 2.3]),
    np.array([0.45, 1200.0, 0.0, 1800.0, 0.008, 250.0, 80000.0, -0.4]),
    np.array([0.7, 2100.0, -20.0, 4100.0, 0.015, 520.0, 120000.0, 1.8]),
]

BALANCE_BOUNDS = [
    (0.1, 0.9),
    (1.0, 3000.0),
    (-40.0, 120.0),
    (100.0, 5000.0),
    (0.001, 0.05),
    (10.0, 1000.0),
    (500.0, 200000.0),
    (-5.0, 5.0),
    (0.0, 1.5),
    (100.0, 10000.0),
]

BALANCE_SEEDS = [
    np.array([0.2, 200.0, -20.0, 400.0, 0.004, 120.0, 5000.0, -3.0, 0.2, 1000.0]),
    np.array([0.8, 2600.0, 90.0, 4400.0, 0.040, 850.0, 180000.0, 3.5, 1.2, 9000.0]),
    np.array([0.45, 1400.0, 20.0, 1800.0, 0.020, 480.0, 70000.0, -0.2, 0.7, 3500.0]),
    np.array([0.62, 900.0, 45.0, 1200.0, 0.012, 260.0, 40000.0, 1.1, 0.5, 2200.0]),
    np.array([0.3, 2200.0, 60.0, 3000.0, 0.030, 620.0, 120000.0, -1.8, 1.0, 6000.0]),
]


def sphere(x: np.ndarray) -> float:
    """Return the sphere-function value for a candidate vector."""

    return float(np.sum(x**2))


def peak(x: np.ndarray) -> float:
    """Return a simple maximization objective with a known peak."""

    return float(10.0 - (x[0] - 1.5) ** 2 - (x[1] + 2.0) ** 2)


def multiscale_stress(x: np.ndarray) -> float:
    """Return a larger multiscale objective with strongly mixed bounds.

    The objective is defined in the original units, but each variable is scaled
    by its bound width so every dimension contributes comparably to the target.
    This makes it a good regression test for the optimizer's internal parameter
    normalization.
    """

    bound_array = np.asarray(STRESS_BOUNDS, dtype=float)
    widths = bound_array[:, 1] - bound_array[:, 0]
    scaled_error = (np.asarray(x, dtype=float) - STRESS_TARGET) / widths
    coupled_term = 0.1 * (
        scaled_error[0] * scaled_error[1]
        - scaled_error[2] * scaled_error[3]
        + scaled_error[4] * scaled_error[5]
        - scaled_error[6] * scaled_error[7]
    )
    return float(np.sum(scaled_error**2) + coupled_term**2)


def multiscale_balance_max(x: np.ndarray) -> float:
    """Return a multidimensional maximize objective with opposing influences.

    This test intentionally avoids a single hidden target point. Instead, the
    score is built from competing rewards and penalties in normalized
    coordinates, so the optimizer must find a balance between growth and
    overshoot across dimensions with very different raw numeric ranges.
    """

    bound_array = np.asarray(BALANCE_BOUNDS, dtype=float)
    lower_bounds = bound_array[:, 0]
    widths = bound_array[:, 1] - bound_array[:, 0]
    normalized = (np.asarray(x, dtype=float) - lower_bounds) / widths

    (
        catalyst,
        feed_rate,
        temperature,
        pressure,
        additive,
        residence_time,
        power,
        control_bias,
        recycle_fraction,
        cooling_rate,
    ) = normalized

    reward = 0.0
    reward += 14.0 * (4.0 * catalyst * (1.0 - catalyst))
    reward += 18.0 * (1.0 - np.exp(-4.0 * feed_rate))
    reward += 10.0 * (1.0 - np.exp(-3.0 * power))
    reward += 8.0 * (1.0 - np.exp(-4.0 * recycle_fraction))
    reward += 9.0 * (1.0 - (2.0 * additive - 1.0) ** 2)
    reward += 7.0 * (1.0 - (2.0 * control_bias - 1.0) ** 4)
    reward += 6.0 * (1.0 - np.abs(2.0 * temperature - 1.0))
    reward += 5.0 * np.sqrt(np.clip(residence_time, 0.0, 1.0))
    reward += 4.0 * (1.0 - np.abs(2.0 * cooling_rate - 1.0))

    reward += 10.0 * feed_rate * (1.0 - pressure)
    reward += 8.0 * recycle_fraction * (1.0 - cooling_rate)
    reward += 6.0 * residence_time * (1.0 - pressure)
    reward += 5.0 * catalyst * feed_rate

    penalty = 0.0
    penalty += 22.0 * np.maximum(0.0, feed_rate + pressure - 1.25) ** 2
    penalty += 16.0 * np.maximum(
        0.0, catalyst + additive + recycle_fraction - 1.7
    ) ** 2
    penalty += 14.0 * np.maximum(0.0, power + temperature - 1.45) ** 2
    penalty += 12.0 * np.maximum(0.0, control_bias + cooling_rate - 1.35) ** 2
    penalty += 10.0 * (pressure - temperature) ** 2
    penalty += 9.0 * np.maximum(
        0.0, residence_time + feed_rate + recycle_fraction - 1.85
    ) ** 2

    return float(reward - penalty)


def run_demo(
    name: str,
    objective_function: Callable[[np.ndarray], float],
    mode: str,
    plot_name: str,
    bounds: list[tuple[float, float]] | None = None,
    seeds: list[np.ndarray] | None = None,
    pop_size: int = 20,
    max_generations: int = 50,
    clone_ratio: float = 0.7,
    random_state: int = 42,
) -> None:
    """Run one optimizer demo and export its convergence plot."""

    bounds = bounds or [(-5.0, 5.0), (-5.0, 5.0)]
    seeds = seeds or [
        np.array([3.0, 3.0]),
        np.array([-3.0, -3.0]),
        np.array([4.0, -4.0]),
        np.array([-4.0, 4.0]),
        np.array([2.0, -2.0]),
    ]

    optimizer = DifferentialEvolutionOptimizer(
        objective_function=objective_function,
        bounds=bounds,
        pop_size=pop_size,
        max_generations=max_generations,
        seeds=seeds,
        clone_ratio=clone_ratio,
        random_state=random_state,
        live_plot=True,
        mode=mode,
    )

    print(f"\n=== {name} Run Configuration ===")
    print("Objective function:", objective_function.__name__)
    print("Optimization mode:", mode)
    print("DE mutation strategy:", "DE/best/1")
    print("DE crossover strategy:", "binomial crossover (bin)")
    print(
        "Population initialization:",
        "direct seeds + semi-random clones for all remaining slots",
    )
    print("Bounds:", bounds)
    print("Population size:", optimizer.pop_size)
    print("Max generations:", optimizer.max_generations)
    print("Differential weight (F):", optimizer.F)
    print("Crossover rate (CR):", optimizer.CR)
    print("Clone ratio:", optimizer.clone_ratio)
    print(
        "Clone ratio meaning:",
        "seed weight in clone = clone_ratio * seed + (1 - clone_ratio) * random",
    )
    print("Random state:", optimizer.random_state)
    print("Live plot enabled:", optimizer.live_plot)
    print("Internal search normalization:", "enabled, unit hypercube [0, 1]^n")
    print("Dimension count:", len(bounds))
    print("Seed count:", len(seeds))

    best_x, best_evaluation = optimizer.run()

    assert optimizer.initialization is not None, (
        "Optimizer initialization summary was not created."
    )
    initialization = optimizer.initialization

    print(f"=== {name} Results ===")
    print(f"{name} best solution:", best_x)
    print(f"{name} best evaluation:", best_evaluation)
    print(
        f"{name} final mean evaluation:",
        optimizer.mean_evaluation_history[-1],
    )
    print(f"{name} initialized clone count:", initialization.clone_count)
    print(f"{name} initialized random count:", initialization.random_count)

    plot_path = Path(__file__).with_name(plot_name)
    if optimizer.plot.save_html(plot_path):
        print(f"{name} convergence plot saved to: {plot_path}")

    if optimizer.plot.show(renderer="browser"):
        print(f"{name} convergence plot opened in your default browser.")
    elif optimizer.plot.disabled_reason:
        print(f"{name} convergence plot was not shown: {optimizer.plot.disabled_reason}")


def main() -> None:
    """Run minimization, maximization, and large multiscale demos."""

    run_demo(
        name="Minimization",
        objective_function=sphere,
        mode="minimize",
        plot_name="convergence_plot_minimize.html",
    )
    run_demo(
        name="Maximization",
        objective_function=peak,
        mode="maximize",
        plot_name="convergence_plot_maximize.html",
    )
    run_demo(
        name="Multiscale Stress Test",
        objective_function=multiscale_stress,
        mode="minimize",
        plot_name="convergence_plot_multiscale_stress.html",
        bounds=STRESS_BOUNDS,
        seeds=STRESS_SEEDS,
        pop_size=36,
        max_generations=120,
        clone_ratio=0.65,
        random_state=123,
    )
    run_demo(
        name="Multiscale Balance Maximization",
        objective_function=multiscale_balance_max,
        mode="maximize",
        plot_name="convergence_plot_multiscale_balance_maximize.html",
        bounds=BALANCE_BOUNDS,
        seeds=BALANCE_SEEDS,
        pop_size=40,
        max_generations=140,
        clone_ratio=0.6,
        random_state=321,
    )


if __name__ == "__main__":
    main()
