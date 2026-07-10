"""
Generate benchmark plots for the Rosenbrock function.
No constraints - unconstrained optimization.
Saves images to docs/_static/images/
"""

import sys
from pathlib import Path

module_path = Path(__file__).parent.parent
sys.path.insert(0, str(module_path))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import scienceplots  # noqa: F401  — registers matplotlib styles on import

from src import bo_optimizer, functions_sampler

plt.style.use("science")
matplotlib.use("Agg")


def rosenbrock(x, a=1, b=100):
    """Standard Rosenbrock function."""
    x = np.asarray(x)
    return np.sum((a - x[:-1]) ** 2 + b * (x[1:] - x[:-1] ** 2) ** 2)


def run_boar_optimization(
    n_dim: int = 5,
    n_initial: int = 20,
    n_trials: int = 1000,
    bounds: list = None,
    seed: int = 42,
    tolerance: float = 1e-6,
):
    """Run BOAR optimization (unconstrained)."""
    if bounds is None:
        bounds = [(-5, 10)] * n_dim

    # Unconstrained sampler
    sampler = functions_sampler.LatinHypercube(bounds=bounds, precision=1e-9, seed=seed)

    # Generate initial samples
    test_samples = sampler.generate_samples(n_initial)
    test_values = np.array([rosenbrock(sample) for sample in test_samples])
    initial_samples = (test_samples, test_values)

    # Run optimization
    optimizer = bo_optimizer.BayesianOptimizer(
        initial_samples=initial_samples,
        obj_func=rosenbrock,
        sampler=sampler,
        opt_args={
            "GPR_iterations": 500,
            "tolerance": tolerance,
            "opt_mem_override": False,
            "n_initial": n_initial,
            "max_no_improvement": n_trials,
            "max_tested_vectors": n_trials,
            "test_population": 100000,
            "seed": seed,
            "GPR_alpha": 1e-10,
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
    """Run Optuna optimization (unconstrained)."""
    if bounds is None:
        bounds = [(-5, 10)] * n_dim

    def objective(trial: optuna.Trial) -> float:
        vector = np.array(
            [trial.suggest_float(f"x{i}", lb, ub) for i, (lb, ub) in enumerate(bounds)],
            dtype=float,
        )
        return rosenbrock(vector)

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

    sampler = optuna.samplers.GPSampler(
        seed=seed,
        n_startup_trials=n_initial,
    )

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, callbacks=[early_stop_callback])

    # Collect all tried vectors with their values
    tried_vectors = [
        [trial.params[f"x{i}"] for i in range(n_dim)]
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    tried_values = [trial.value for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]

    return study.best_params, study.best_value, tried_vectors, tried_values


def plot_rosenbrock_contour(bounds, save_path):
    """Plot 2D Rosenbrock function contour."""
    x = np.linspace(bounds[0][0], bounds[0][1], 200)
    y = np.linspace(bounds[1][0], bounds[1][1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[rosenbrock([xi, yi]) for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(contour, label="f(x, y)")

    # Global minimum
    plt.scatter([1], [1], color="red", s=300, marker="*", label="Global min (1, 1)", zorder=5, edgecolors="black")

    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$y$", fontsize=14)
    plt.title("Rosenbrock Function (Unconstrained)", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_convergence_history(all_results, save_path):
    """Plot convergence history for all dimensions."""
    n_dims = len(all_results)
    cols = 3
    rows = (n_dims + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if n_dims > 1 else [axes]

    for idx, (n_dim, (boar, optuna)) in enumerate(all_results.items()):
        ax = axes[idx]

        # Compute cumulative best for BOAR
        boar_best = []
        current_best = float("inf")
        for val in boar[2]:
            current_best = min(current_best, val[1])
            boar_best.append(current_best)

        # Compute cumulative best for Optuna
        optuna_best = []
        current_best = float("inf")
        for val in optuna[3]:
            current_best = min(current_best, val)
            optuna_best.append(current_best)

        ax.plot(range(len(boar_best)), boar_best, label="BOAR", color="steelblue", linewidth=2)
        ax.plot(range(len(optuna_best)), optuna_best, label="Optuna", color="coral", linewidth=2)

        ax.set_xlabel("Function Evaluations", fontsize=12)
        ax.set_ylabel("Best Value Found", fontsize=12)
        ax.set_title(f"n_dim = {n_dim}", fontsize=14)
        ax.set_yscale("log")
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(all_results), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Convergence History - Rosenbrock Function", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_search_history(results, bounds, save_path):
    """Plot search history for 2D case."""
    if 2 not in results:
        print("Skipping search history plot (n_dim=2 not found)")
        return

    boar, optuna = results[2]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create contour plot
    x = np.linspace(bounds[0][0], bounds[0][1], 200)
    y = np.linspace(bounds[1][0], bounds[1][1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[rosenbrock([xi, yi]) for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

    contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis", alpha=0.7)
    plt.colorbar(contour, label="f(x, y)")

    # BOAR search history
    boar_points = np.array([pt[0][:2] for pt in boar[2]])
    ax.scatter(
        boar_points[:, 0],
        boar_points[:, 1],
        label="BOAR",
        color="blue",
        alpha=0.6,
        s=30,
        edgecolors="white",
        linewidths=0.3,
    )

    # Optuna search history
    optuna_points = np.array(optuna[2])[:, :2]
    ax.scatter(
        optuna_points[:, 0],
        optuna_points[:, 1],
        label="Optuna",
        color="orange",
        alpha=0.6,
        s=20,
        marker="o",
        edgecolors="none",
    )

    # Mark global minimum
    ax.scatter([1], [1], color="lime", s=300, marker="*", label="Global min (1,1)", zorder=6, edgecolors="black")

    # Mark best solutions
    ax.scatter(
        [boar[0][0]], [boar[0][1]], color="blue", s=150, marker="D", label="BOAR best", zorder=6, edgecolors="black"
    )
    ax.scatter(
        [optuna[0]["x0"]],
        [optuna[0]["x1"]],
        color="orange",
        s=150,
        marker="D",
        label="Optuna best",
        zorder=6,
        edgecolors="black",
        alpha=0.7,
    )

    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)
    ax.set_title("Search History - Rosenbrock Function (2D)", fontsize=16)
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_comparison(all_results, save_path):
    """Plot comparison between BOAR and Optuna across dimensions."""
    dimensions = sorted(all_results.keys())

    boar_values = [all_results[n][0][1] for n in dimensions]
    optuna_values = [all_results[n][1][1] for n in dimensions]

    x = np.arange(len(dimensions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width / 2, boar_values, width, label="BOAR", color="steelblue")
    ax.bar(x + width / 2, optuna_values, width, label="Optuna", color="coral")
    ax.set_xlabel("Number of Dimensions", fontsize=12)
    ax.set_ylabel("Best Value Found", fontsize=12)
    ax.set_title("Best Value Comparison - Rosenbrock Function (Unconstrained)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions)
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_summary_report(all_results, output_path):
    """Generate text summary of results."""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Rosenbrock Function Benchmark Results\n")
        f.write("Unconstrained Optimization\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Dim':<5} {'BOAR Best':<15} {'Optuna Best':<15}\n")
        f.write("-" * 50 + "\n")

        for n_dim in sorted(all_results.keys()):
            boar, optuna = all_results[n_dim]
            f.write(f"{n_dim:<5} {boar[1]:<15.6f} {optuna[1]:<15.6f}\n")

        f.write("-" * 50 + "\n\n")
        f.write("Best Parameters:\n")
        f.write("-" * 50 + "\n")

        for n_dim in sorted(all_results.keys()):
            boar, optuna = all_results[n_dim]
            f.write(f"\nn_dim = {n_dim}:\n")
            f.write("  BOAR:\n")
            f.write(f"    Sample: {boar[0]}\n")
            f.write(f"    Value: {boar[1]:.10f}\n")
            f.write("  Optuna:\n")
            f.write(f"    Params: {optuna[0]}\n")
            f.write(f"    Value: {optuna[1]:.10f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Global Minimum: 0.0 at x = (1, 1, ..., 1)\n")
        f.write("Search Domain: [-5, 10]^n\n")
        f.write("=" * 80 + "\n")

    print(f"Saved: {output_path}")


def main():
    """Run all benchmarks for multiple dimensions."""
    dimensions = [2, 3, 4, 5, 10, 15, 20]
    n_initial = 15
    n_trials = 100
    seed = 9
    bounds = [(-5, 10)] * max(dimensions)
    tolerance = 1e-6

    output_dir = module_path / "docs" / "_static" / "images" / "rosenbrock"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for n_dim in dimensions:
        print(f"\n{'=' * 60}")
        print(f"Running benchmarks for n_dim = {n_dim}")
        print(f"{'=' * 60}")

        # Run BOAR optimization
        print("  Running BOAR optimization...")
        boar = run_boar_optimization(
            n_dim=n_dim, n_trials=n_trials, n_initial=n_initial, bounds=bounds[:n_dim], seed=seed, tolerance=tolerance
        )
        print(f"    Best value: {boar[1]:.6f}")

        # Run Optuna optimization
        print("  Running Optuna optimization...")
        optuna = run_optuna_optimization(
            n_dim=n_dim, n_trials=n_trials, n_initial=n_initial, bounds=bounds[:n_dim], seed=seed, tolerance=tolerance
        )
        print(f"    Best value: {optuna[1]:.6f}")

        all_results[n_dim] = (boar, optuna)

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    plot_rosenbrock_contour(bounds[:2], output_dir / "rosenbrock_contour.svg")
    plot_convergence_history(all_results, output_dir / "rosenbrock_convergence.svg")
    plot_search_history(all_results, bounds, output_dir / "rosenbrock_search_history.svg")
    plot_comparison(all_results, output_dir / "rosenbrock_comparison.svg")
    generate_summary_report(all_results, output_dir / "rosenbrock_results.txt")

    print("\n" + "=" * 60)
    print("All benchmark images generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
