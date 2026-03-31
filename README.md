# Differential Evolution Optimizer (DEO)

This project contains a custom Differential Evolution Optimizer implemented in Python with Plotly-based convergence visualization.

![Representative convergence plot](convergence_example.gif)

The optimizer is built around the `DE/best/1/bin` strategy and supports:

- minimization and maximization
- deterministic seeds
- semi-random seed clones
- internal normalization to a unit hypercube for mixed-scale bounds
- Plotly convergence plots with retry and timeout handling
- direct use with external objective functions

## Project Structure

```text
DEO/
|-- deo/
|   |-- __init__.py
|   |-- optimizer.py
|   |-- population.py
|   |-- strategies.py
|   |-- plotting.py
|   |-- utils.py
|-- test_deo.py
|-- requirements.txt
|-- instructions
|-- README.md
```

## Installation

The project is currently configured and tested with Python 3.12.

### 1. Create a virtual environment

```powershell
py -3.12 -m venv .venv312
```

### 2. Activate the environment

```powershell
.\.venv312\Scripts\Activate.ps1
```

### 3. Install the requirements

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Requirements

The required packages are listed in `requirements.txt`:

- `numpy`
- `plotly`

## Running the Demo Script

To run the included demo cases:

```powershell
python .\test_deo.py
```

The demo currently runs:

- a 2D minimization case
- a 2D maximization case
- an 8D multiscale stress test with very different bound ranges
- a 10D balance-based maximization test with opposing influences

Each run:

- prints the full configuration
- prints the best and mean evaluation values
- exports a Plotly HTML convergence plot
- attempts to open the plot in your browser

## Importing and Using the Module

You can import the optimizer directly from the package:

```python
import numpy as np

from deo import DifferentialEvolutionOptimizer


def objective_function(x: np.ndarray) -> float:
    return float(np.sum(x**2))


bounds = [(-5.0, 5.0), (-5.0, 5.0)]
seeds = [
    np.array([3.0, 3.0]),
    np.array([-3.0, -3.0]),
]

optimizer = DifferentialEvolutionOptimizer(
    objective_function=objective_function,
    bounds=bounds,
    pop_size=20,
    F=0.5,
    CR=0.9,
    max_generations=100,
    seeds=seeds,
    clone_ratio=0.7,
    random_state=42,
    live_plot=True,
    mode="minimize",
)

best_x, best_evaluation = optimizer.run()

print("Best solution:", best_x)
print("Best evaluation:", best_evaluation)
```

## Key Behavior

### Strategy

The optimizer uses:

- mutation: `DE/best/1`
- crossover: `binomial`
- selection: greedy one-to-one replacement

### Seed Handling

Provided seeds are inserted directly into the initial population.

All remaining population slots become semi-random clones when seeds are present. The current clone rule is:

```python
clone = clone_ratio * seed + (1 - clone_ratio) * random_vector
```

That means:

- `clone_ratio = 1.0` gives exact clones of the seed
- `clone_ratio = 0.7` gives clones that are 70% seed and 30% random
- `clone_ratio = 0.5` gives a 50/50 blend
- `clone_ratio = 0.0` gives fully random non-seed individuals

### Internal Normalization

The optimizer internally normalizes all decision variables to `[0, 1]` before mutation and crossover, then maps candidates back to the original bounds before evaluation.

This is especially important for mixed-scale problems such as:

- one variable with bounds `(0.1, 0.9)`
- another variable with bounds `(1, 3000)`

Without normalization, the wide-range variables dominate differential mutation. With normalization, all dimensions are explored on a comparable scale.

### Plotly Robustness

Plot rendering is designed to be resilient:

- plots are always exportable to HTML
- browser opening is retried every `0.5` seconds
- if Plotly cannot render within `5` seconds, a timeout warning is returned instead of silently failing

## AI-Assisted Construction

This module was intentionally built using an AI-assisted workflow.

The project started from a planning document stored in the root `instructions` file. That file acted as a structured gameplan for implementation, covering:

- architecture
- file layout
- algorithm steps
- initialization rules
- plotting behavior
- testing strategy

The initial user prompt that kicked off the design was:

> he chat!, please write a plan for your future self so that github-copilot can easely construct the following module: i want to construct a custom Differential Evolution Optimizer (DEO) module today, that uses Plotly for showing live convergence. the standard strategy is Best1bin, and it uses a combination of determined seeds, random seeds and semi-random seeds. so if you give 5 seeds and need 20 pops and have a clone-ratio of 0.7 it creates 15 semi-random clones that are for 70% equal to a random seed, equally distributed. the objective function should be the calling of an external function with the optimization parameters as inputs. at the end, write a little test so we can see if it works properly!

That prompt was translated into a more structured implementation blueprint in `instructions`, and the final code was then developed iteratively with AI support.

## How the Program Was Constructed

The implementation process followed a staged AI-guided workflow:

1. A detailed blueprint was written in `instructions` so GitHub Copilot had a concrete architectural target.
2. The base package structure was created under `deo/`.
3. The DE mutation, crossover, selection, and seed initialization logic were implemented.
4. The optimizer was extended to support both minimization and maximization.
5. Internal normalization was added so mixed-range dimensions are explored fairly.
6. Plotly rendering was hardened with HTML export, browser rendering, retry logic, and timeout warnings.
7. Demo and stress-test scripts were added to validate the optimizer on simple and multidimensional problems.

In short, the project was not generated in one shot. It was constructed through an AI-assisted engineering process based on a clear gameplan, iterative refinement, and repeated runtime verification.

## Current Demo Outputs

Running `test_deo.py` will generate HTML plots such as:

- `convergence_plot_minimize.html`
- `convergence_plot_maximize.html`
- `convergence_plot_multiscale_stress.html`
- `convergence_plot_multiscale_balance_maximize.html`

It also includes a static image preview for the README:

- `convergence.png`

## Notes

- The module expects objective functions that accept a NumPy vector and return a scalar float.
- The package exports `DifferentialEvolutionOptimizer` from `deo.__init__` for convenient importing.
- The demo script is intended as both a smoke test and a usage reference.