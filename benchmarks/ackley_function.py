"""
Generate benchmark plots for the Ackley function.
Saves images to docs/_static/images/
"""

import sys
from pathlib import Path

module_path = Path(__file__).parent.parent
sys.path.insert(0, str(module_path))
import matplotlib.pyplot as plt
import numpy as np
import optuna

from src import bo_optimizer, functions_sampler

plt.style.use("science")


# Ackley function
def ackley(x, a=20, b=0.2, c=2 * np.pi):
    x = np.asarray(x)
    n = x.size
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(c * x)) / n)
    return term1 + term2 + a + np.e


def run_boar_optimization(
    n_dim=5, n_initial=20, n_trials=1000, bounds=None, seed=42, tolerance=1e-6, test_population=100000
):
    """Run in-house Bayesian optimization."""
    if bounds is None:
        bounds = [(-5, 5)] * n_dim

    sampler_lhs = functions_sampler.LatinHypercube(bounds=bounds, precision=1e-9, seed=seed)
    test_samples = sampler_lhs.generate_samples(n_initial)
    test_values = np.array([ackley(sample) for sample in test_samples])
    initial_samples = (test_samples, test_values)

    optimizer = bo_optimizer.BayesianOptimizer(
        initial_samples=initial_samples,
        obj_func=ackley,
        sampler=sampler_lhs,
        opt_args={
            "GPR_iterations": 500,
            "tolerance": tolerance,
            "opt_mem_override": False,
            "n_initial": n_initial,
            "max_no_improvement": n_trials,
            "max_tested_vectors": n_trials,
            "test_population": test_population,
            "seed": seed,
        },
    )
    best_sample, best_value, attempted = optimizer.optimize(return_attempted_points=True)
    return best_sample, best_value, attempted


def run_optuna_optimization(
    n_dim: int = 5,
    n_trials: int = 1000,
    n_initial: int = 20,
    bounds: list = None,
    seed: int = 42,
    tolerance: float = 1e-6,
):
    """Run Optuna optimization and stop early when tolerance is reached."""

    if bounds is None:
        bounds = [(-5, 5)] * n_dim

    def objective(trial: optuna.Trial) -> float:
        vector = np.array(
            [trial.suggest_float(f"x{i}", lb, ub) for i, (lb, ub) in enumerate(bounds)],
            dtype=float,
        )

        return ackley(vector)

    # Pure Optuna GP sampler
    sampler = optuna.samplers.GPSampler(
        seed=seed,
        n_startup_trials=n_initial,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )

    # Early stopping callback
    def early_stop_callback(
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ):
        # Skip until at least one feasible completed trial exists
        if len(study.best_trials) == 0:
            return

        if study.best_value <= tolerance:
            print(f"Stopping early: best value {study.best_value:.3e} <= tolerance {tolerance:.3e}")
            study.stop()

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[early_stop_callback],
    )

    tried_vectors = [
        [trial.params[f"x{i}"] for i in range(n_dim)]
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    tried_values = [trial.value for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]

    return (
        study.best_params,
        study.best_value,
        tried_vectors,
        tried_values,
    )


def save_tried_vectors(vectors, output_path, title="Tried vectors"):
    """Save tried vectors to a text file."""
    with open(output_path, "w") as f:
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n")
        for idx, vec in enumerate(vectors, 1):
            f.write(f"{idx:03d}: {vec}\n")
    print(f"Saved: {output_path}")


def plot_convergence_history(all_results, save_path):
    """Plot convergence history for all dimensions."""
    n_dims = len(all_results)
    cols = 3
    rows = (n_dims + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if n_dims > 1 else [axes]

    for idx, (n_dim, result) in enumerate(all_results.items()):
        boar, optuna = result
        ax = axes[idx]

        # Compute cumulative best for BOAR
        boar_best = []
        current_best = float("inf")
        for val in boar[2]:  # boar[2] is attempted (sample, value)
            current_best = min(current_best, val[1])
            boar_best.append(current_best)

        # Compute cumulative best for Optuna
        optuna_best = []
        current_best = float("inf")
        for val in optuna[3]:  # optuna[3] is tried_values
            current_best = min(current_best, val)
            optuna_best.append(current_best)

        ax.plot(range(len(boar_best)), boar_best, label="BOAR", color="steelblue", linewidth=2)
        ax.plot(range(len(optuna_best)), optuna_best, label="Optuna", color="coral", linewidth=2)

        ax.set_xlabel("Function Evaluations", fontsize=14)
        ax.set_ylabel("Best Value Found", fontsize=14)
        ax.set_title(f"n_dim = {n_dim}", fontsize=16)
        ax.set_yscale("log")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(all_results), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Convergence History - Ackley Function", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_ackley_2d(bounds, save_path):
    """Plot 2D Ackley function contour."""
    x = np.linspace(bounds[0][0], bounds[0][1], 200)
    y = np.linspace(bounds[1][0], bounds[1][1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[ackley([xi, yi]) for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(contour, label="f(x, y)")
    plt.scatter([0], [0], color="red", s=200, marker="*", label="Global minimum (0, 0)", zorder=5)
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$y$", fontsize=14)
    plt.title("Ackley Function Contour Plot", fontsize=18)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_search_history(all_results, bounds, save_path):
    """Plot search history of the optimization for 2D case."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create contour plot background
    x = np.linspace(bounds[0][0], bounds[0][1], 200)
    y = np.linspace(bounds[1][0], bounds[1][1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[ackley([xi, yi]) for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

    contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis", alpha=0.7)
    plt.colorbar(contour, label="f(x, y)")

    # Get 2D results
    if 2 in all_results:
        boar = all_results[2][0]  # best_sample, best_value, attempted
        optuna = all_results[2][1]

        # BOAR search history (2D projection)
        boar_points = np.array([pt[0][:2] for pt in boar[2]])
        ax.scatter(
            boar_points[:, 0],
            boar_points[:, 1],
            label="BOAR",
            color="steelblue",
            alpha=0.6,
            s=30,
            edgecolors="black",
            linewidths=0.3,
        )

        # Optuna search history (2D projection)
        optuna_points = np.array(optuna[2])[:, :2]
        ax.scatter(
            optuna_points[:, 0],
            optuna_points[:, 1],
            label="Optuna",
            color="coral",
            alpha=0.6,
            s=30,
            edgecolors="black",
            linewidths=0.3,
        )

        # Mark global minimum
        ax.scatter([0], [0], color="lime", s=300, marker="*", label="Global minimum", zorder=5, edgecolors="black")

        # Mark final solutions
        ax.scatter(
            [boar[0][0]],
            [boar[0][1]],
            color="blue",
            s=150,
            marker="D",
            label="BOAR Final",
            zorder=6,
            edgecolors="black",
        )
        ax.scatter(
            [optuna[0]["x0"]],
            [optuna[0]["x1"]],
            color="yellow",
            s=150,
            marker="D",
            label="Optuna Final",
            zorder=6,
            edgecolors="black",
        )

    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)
    ax.set_title("Search History Comparison - Ackley Function (2D)", fontsize=18)
    ax.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_comparison(all_results, save_path):
    """Plot comparison between in-house BOAR and Optuna across dimensions."""
    dimensions = list(all_results.keys())
    dimensions.sort()

    boar_best_values = []
    optuna_best_values = []
    boar_func_evals = []
    optuna_func_evals = []

    for n_dim in dimensions:
        boar = all_results[n_dim][0]  # best_sample, best_value, attempted
        optuna = all_results[n_dim][1]

        boar_best_values.append(boar[1])  # best_value
        optuna_best_values.append(optuna[1])  # best_value
        boar_func_evals.append(len(boar[2]))  # number of evaluations
        optuna_func_evals.append(len(optuna[3]))  # number of evaluations

    x = np.arange(len(dimensions))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    # Plot 1: Best values achieved
    ax.bar(x - width / 2, boar_best_values, width, label="BOAR", color="steelblue")
    ax.bar(x + width / 2, optuna_best_values, width, label="Optuna", color="coral")

    ax.set_xlabel("Number of Dimensions", fontsize=14)
    ax.set_ylabel("Best Value Found", fontsize=14)
    ax.set_title("Best Value Comparison", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_dimension_scaling(all_results, save_path):
    """Plot how optimization performance scales with dimension."""
    dimensions = list(all_results.keys())
    dimensions.sort()

    metrics = {
        "best_value": ([], []),
        "evaluations": ([], []),
        "time": ([], []),  # Placeholder - would need actual timing
    }

    for n_dim in dimensions:
        boar = all_results[n_dim][0]
        optuna = all_results[n_dim][1]

        metrics["best_value"][0].append(boar[1])
        metrics["best_value"][1].append(optuna[1])
        metrics["evaluations"][0].append(len(boar[2]))
        metrics["evaluations"][1].append(len(optuna[3]))

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    # Best value vs dimension
    ax.plot(dimensions, metrics["best_value"][0], "o-", label="BOAR", color="steelblue", linewidth=2, markersize=8)
    ax.plot(dimensions, metrics["best_value"][1], "s-", label="Optuna", color="coral", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Dimensions", fontsize=14)
    ax.set_ylabel("Best Value Found", fontsize=14)
    ax.set_title("Loss value vs Dimension", fontsize=18)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.set_xlim(0, max(dimensions) + 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_summary_report(all_results, output_path):
    """Generate a text summary of the results for all dimensions."""
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Ackley Function Benchmark Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Dim':<5} {'BOAR Best':<15} {'Optuna Best':<15} {'BOAR Evals':<12} {'Optuna Evals':<12}\n")
        f.write("-" * 70 + "\n")

        for n_dim in sorted(all_results.keys()):
            boar = all_results[n_dim][0]
            optuna = all_results[n_dim][1]

            f.write(f"{n_dim:<5} {boar[1]:<15.6f} {optuna[1]:<15.6f} {len(boar[2]):<12} {len(optuna[3]):<12}\n")

        f.write("-" * 70 + "\n")
        f.write("\nBest Parameters:\n")
        f.write("-" * 70 + "\n")

        for n_dim in sorted(all_results.keys()):
            boar = all_results[n_dim][0]
            f.write(f"\nn_dim = {n_dim}:\n")
            f.write(f"  BOAR: {boar[0]}\n")
            f.write(f"  Optuna: {[optuna[0][f'x{i}'] for i in range(n_dim)]}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Global Minimum: 0.0\n")
        f.write("Search Domain: [-5, 5]^n\n")
        f.write("=" * 70 + "\n")

    print(f"Saved: {output_path}")


def main():
    """Run all benchmark generations for multiple dimensions."""
    # Define dimensions to benchmark
    dimensions = [1, 2, 3, 4, 5, 10, 15, 20]
    n_initial = 15
    n_trials = 100
    seed = 9
    bounds = [(-32.768, 32.768)] * max(dimensions)
    tolerance = 1e-6

    # Create output directory
    output_dir = module_path / "docs" / "_static" / "images" / "ackley_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store results for all dimensions
    all_results = {}

    for n_dim in dimensions:
        print(f"\n{'=' * 60}")
        print(f"Running benchmarks for n_dim = {n_dim}")
        print(f"{'=' * 60}")

        # Run BOAR optimization
        print(f"Running BOAR optimization (n_dim={n_dim})...")
        boar_result = run_boar_optimization(
            n_dim=n_dim, n_trials=n_trials, n_initial=n_initial, bounds=bounds[:n_dim], seed=seed, tolerance=tolerance
        )
        print(f"  Best sample: {boar_result[0]}")
        print(f"  Best value: {boar_result[1]:.6f}")
        print(f"  Evaluations: {len(boar_result[2])}")

        # Run Optuna optimization
        print(f"Running Optuna optimization (n_dim={n_dim})...")
        optuna_result = run_optuna_optimization(
            n_dim=n_dim, n_trials=n_trials, n_initial=n_initial, bounds=bounds[:n_dim], seed=seed, tolerance=tolerance
        )
        print(f"  Best sample: {optuna_result[0]}")
        print(f"  Best value: {optuna_result[1]:.6f}")
        print(f"  Evaluations: {len(optuna_result[3])}")

        all_results[n_dim] = (boar_result, optuna_result)

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    plot_ackley_2d(bounds, output_dir / "ackley_contour.svg")
    plot_convergence_history(all_results, output_dir / "ackley_convergence.svg")
    plot_search_history(all_results, bounds, output_dir / "ackley_search_history.svg")
    plot_comparison(all_results, output_dir / "ackley_comparison.svg")
    plot_dimension_scaling(all_results, output_dir / "ackley_scaling.svg")

    # Generate summary report
    generate_summary_report(all_results, output_dir / "ackley_results.txt")

    # Save individual results for each dimension
    for n_dim in dimensions:
        dim_dir = output_dir / f"dim_{n_dim}"
        dim_dir.mkdir(exist_ok=True)

        # Save BOAR tried vectors
        boar_tried = all_results[n_dim][0][2]
        boar_tried_vectors = [pt[0].tolist() if hasattr(pt[0], "tolist") else list(pt[0]) for pt in boar_tried]
        save_tried_vectors(
            boar_tried_vectors,
            dim_dir / f"boar_tried_vectors_dim{n_dim}.txt",
            title=f"BOAR Tried Vectors (n_dim={n_dim})",
        )

        # Save Optuna tried vectors
        optuna_tried = all_results[n_dim][1][2]
        save_tried_vectors(
            optuna_tried,
            dim_dir / f"optuna_tried_vectors_dim{n_dim}.txt",
            title=f"Optuna Tried Vectors (n_dim={n_dim})",
        )

    print("\n" + "=" * 60)
    print("All benchmark images generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
