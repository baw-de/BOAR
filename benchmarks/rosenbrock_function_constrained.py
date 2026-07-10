"""
Generate benchmark plots for the constrained Rosenbrock function.
Constraint: sum(x²) > n * 5  (outside hypersphere with radius sqrt(n*5))
Saves images to docs/_static/images/
"""

import sys
from pathlib import Path

module_path = Path(__file__).parent.parent
sys.path.insert(0, str(module_path))

import matplotlib.pyplot as plt
import numpy as np
import optuna

from src import bo_optimizer, functions_sampler, utils

plt.style.use("science")


def rosenbrock(x, a=1, b=100):
    """Standard Rosenbrock function."""
    x = np.asarray(x)
    return np.sum((a - x[:-1]) ** 2 + b * (x[1:] - x[:-1] ** 2) ** 2)


def rosenbrock_constrained(args, penalty_weight=1e6):
    """
    Rosenbrock function with penalty for constraint violations.
    Constraint: sum(x²) > n * 5  (outside hypersphere)
    """
    x = np.asarray(args)
    rosenbrock_value = rosenbrock(x)

    # Infeasible if inside the sphere (sum <= n*5)
    if np.sum(x**2) <= len(x) * 5:
        violation = len(x) * 5 - np.sum(x**2)
        return rosenbrock_value + penalty_weight * violation**2

    return rosenbrock_value


def create_sphere_constraint(n_dim):
    """Create a constraint for sum(x²) > n * 5."""
    variables = [f"x{i}" for i in range(n_dim)]
    terms = " + ".join([f"{var}**2" for var in variables])
    expr = f"{terms} > {n_dim * 5}"

    return utils.create_constraint_function(expr, variables)


def run_boar_optimization(
    n_dim: int = 5,
    n_initial: int = 20,
    n_trials: int = 1000,
    bounds: list = None,
    seed: int = 42,
    tolerance: float = 1e-6,
):
    """Run BOAR optimization with constrained sampling."""
    if bounds is None:
        bounds = [(-5, 10)] * n_dim

    # Create constraint function
    constraint_fn = create_sphere_constraint(n_dim)

    # Constrained sampler
    sampler = functions_sampler.LatinHypercube(bounds=bounds, precision=1e-9, constraint_fns=[constraint_fn], seed=seed)

    # Generate initial samples
    test_samples = sampler.generate_samples(n_initial)
    test_values = np.array([rosenbrock_constrained(sample) for sample in test_samples])
    initial_samples = (test_samples, test_values)

    # Run optimization
    optimizer = bo_optimizer.BayesianOptimizer(
        initial_samples=initial_samples,
        obj_func=rosenbrock_constrained,
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
    """Run Optuna optimization with constrained sampling."""
    if bounds is None:
        bounds = [(-5, 10)] * n_dim

    def objective(trial: optuna.Trial) -> float:
        vector = np.array(
            [trial.suggest_float(f"x{i}", lb, ub) for i, (lb, ub) in enumerate(bounds)],
            dtype=float,
        )
        return rosenbrock_constrained(vector)

    def constraints_func(trial: optuna.trial.FrozenTrial) -> tuple:
        """Constraint: sum(x²) > n * 5 (negated for Optuna)"""
        if not trial.params:
            return (0.0,)
        x = np.array([trial.params[f"x{i}"] for i in range(n_dim)])
        # Infeasible when n*5 - sum(x²) > 0 (i.e., sum(x²) <= n*5)
        return (n_dim * 5 - np.sum(x**2),)

    sampler = optuna.samplers.GPSampler(
        seed=seed,
        n_startup_trials=n_initial,
        constraints_func=constraints_func,
    )

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # Collect all tried vectors with their values
    tried_vectors = [
        [trial.params[f"x{i}"] for i in range(n_dim)]
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    tried_values = [trial.value for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]

    return study.best_params, study.best_value, tried_vectors, tried_values


def plot_rosenbrock_contour(bounds, n_dim, save_path):
    """Plot 2D Rosenbrock function contour - outside sphere is feasible."""
    x = np.linspace(bounds[0][0], bounds[0][1], 200)
    y = np.linspace(bounds[1][0], bounds[1][1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[rosenbrock([xi, yi]) for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(contour, label="f(x, y)")

    # Add constraint boundary (circle x² + y² = n*5)
    radius = np.sqrt(n_dim * 5)
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(
        radius * np.cos(theta), radius * np.sin(theta), "r--", linewidth=3, label=f"Constraint: x² + y² = {n_dim * 5}"
    )

    # Shade infeasible region (INSIDE the sphere)
    infeasible = X**2 + Y**2 <= n_dim * 5
    plt.contourf(X, Y, infeasible.astype(int), levels=[0.5, 1], colors=["red"], alpha=0.3)

    # Global minimum (inside sphere - infeasible!)
    plt.scatter(
        [1],
        [1],
        color="white",
        s=300,
        marker="*",
        label="Global min (infeasible)",
        zorder=5,
        edgecolors="black",
        linewidths=2,
    )

    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$y$", fontsize=14)
    plt.title(rf"Rosenbrock: Outside Sphere $x^2 + y^2 > {n_dim * 5}$", fontsize=16)
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

    plt.suptitle("Convergence History - Constrained Rosenbrock Function", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_search_history(results, bounds, n_dim, save_path):
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

    # Add constraint boundary
    radius = np.sqrt(n_dim * 5)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(
        radius * np.cos(theta), radius * np.sin(theta), "r--", linewidth=3, label=f"Constraint: x² + y² = {n_dim * 5}"
    )

    # Shade infeasible region (INSIDE the sphere)
    infeasible = X**2 + Y**2 <= n_dim * 5
    ax.contourf(X, Y, infeasible.astype(int), levels=[0.5, 1], colors=["red"], alpha=0.3)

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

    # Global minimum (inside sphere - infeasible!)
    ax.scatter(
        [1],
        [1],
        color="white",
        s=300,
        marker="*",
        label="Global min\n(infeasible)",
        zorder=6,
        edgecolors="black",
        linewidths=2,
    )

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
    ax.set_title(rf"Search History - Outside Sphere $x^2 + y^2 > {n_dim * 5}$", fontsize=16)
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Best values comparison
    ax1 = axes[0]
    ax1.bar(x - width / 2, boar_values, width, label="BOAR", color="steelblue")
    ax1.bar(x + width / 2, optuna_values, width, label="Optuna", color="coral")
    ax1.set_xlabel(r"Number of Dimensions", fontsize=12)
    ax1.set_ylabel(r"Best Value Found", fontsize=12)
    ax1.set_title(r"Best Value Comparison", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dimensions)
    ax1.legend(fontsize=10)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3, axis="y")

    # Constraint satisfaction (distance from boundary - positive = outside sphere)
    ax2 = axes[1]
    boar_margins = [np.sum(all_results[n][0][0] ** 2) - n * 5 for n in dimensions]
    optuna_margins = [sum(all_results[n][1][0][f"x{i}"] ** 2 for i in range(n)) - n * 5 for n in dimensions]

    ax2.bar(x - width / 2, boar_margins, width, label="BOAR", color="steelblue")
    ax2.bar(x + width / 2, optuna_margins, width, label="Optuna", color="coral")
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=2, label="Constraint boundary")
    ax2.set_xlabel(r"Number of Dimensions", fontsize=12)
    ax2.set_ylabel(r"Distance to Boundary ($\sum x_i^2 - n \cdot 5$)", fontsize=10)
    ax2.set_title(r"Constraint Margin (positive = feasible)", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dimensions)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_summary_report(all_results, output_path):
    """Generate text summary of results."""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Constrained Rosenbrock Function Benchmark Results\n")
        f.write("Constraint: sum(x²) > n * 5  (outside hypersphere)\n")
        f.write("=" * 80 + "\n\n")

        # Show example constraint
        for n_dim in [2, 5, 10]:
            variables = [f"x{i}" for i in range(n_dim)]
            terms = " + ".join([f"{v}**2" for v in variables])
            expr = f"{terms} > {n_dim * 5}"
            radius = np.sqrt(n_dim * 5)
            f.write(f'Example (n={n_dim}): "{expr}"  (radius = {radius:.2f})\n')

        f.write("\n" + "-" * 80 + "\n")
        f.write(
            f"{'Dim':<5} {'Radius':<8} {'BOAR Best':<15} {'Optuna Best':<15} "
            f"{'BOAR Margin':<15} {'Optuna Margin':<15}\n"
        )
        f.write("-" * 80 + "\n")

        for n_dim in sorted(all_results.keys()):
            boar, optuna = all_results[n_dim]
            radius = np.sqrt(n_dim * 5)
            boar_margin = np.sum(boar[0] ** 2) - n_dim * 5
            optuna_margin = sum(optuna[0][f"x{i}"] ** 2 for i in range(n_dim)) - n_dim * 5
            f.write(
                f"{n_dim:<5} {radius:<8.2f} {boar[1]:<15.6f} {optuna[1]:<15.6f} "
                f"{boar_margin:<15.6f} {optuna_margin:<15.6f}\n"
            )

        f.write("-" * 80 + "\n\n")
        f.write("Best Parameters:\n")
        f.write("-" * 80 + "\n")

        for n_dim in sorted(all_results.keys()):
            boar, optuna = all_results[n_dim]
            boar_feasible = np.sum(boar[0] ** 2) > n_dim * 5
            optuna_feasible = sum(optuna[0][f"x{i}"] ** 2 for i in range(n_dim)) > n_dim * 5
            f.write(f"\nn_dim = {n_dim}:\n")
            f.write("  BOAR:\n")
            f.write(f"    Sample: {boar[0]}\n")
            f.write(f"    Value: {boar[1]:.10f}\n")
            f.write(f"    Sum(x²): {np.sum(boar[0] ** 2):.6f}\n")
            f.write(f"    Feasible (outside sphere): {boar_feasible}\n")
            f.write("  Optuna:\n")
            f.write(f"    Params: {optuna[0]}\n")
            f.write(f"    Value: {optuna[1]:.10f}\n")
            f.write(f"    Sum(x²): {sum(optuna[0][f'x{i}'] ** 2 for i in range(n_dim)):.6f}\n")
            f.write(f"    Feasible (outside sphere): {optuna_feasible}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Global Minimum (unconstrained): 0.0 at x = (1, 1, ..., 1)\n")
        f.write("Note: Global minimum is INSIDE the constraint sphere (infeasible)\n")
        f.write("Constraint: sum(x²) > n * 5  (outside hypersphere with radius sqrt(n*5))\n")
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

    output_dir = module_path / "docs" / "_static" / "images" / "rosenbrock_constrained"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for n_dim in dimensions:
        print(f"\n{'=' * 60}")
        print(f"Running benchmarks for n_dim = {n_dim}")
        print(f"{'=' * 60}")

        # Show constraint format
        variables = [f"x{i}" for i in range(n_dim)]
        terms = " + ".join([f"{v}**2" for v in variables])
        expr = f"{terms} > {n_dim * 5}"
        radius = np.sqrt(n_dim * 5)
        print("Constraint:")
        print(f'  expression: "{expr}"')
        print(f"  radius: sqrt({n_dim * 5}) = {radius:.2f}")
        print("  Feasible: outside sphere | Infeasible: inside sphere")

        # Run BOAR optimization
        print("  Running BOAR optimization...")
        boar = run_boar_optimization(
            n_dim=n_dim, n_trials=n_trials, n_initial=n_initial, bounds=bounds[:n_dim], seed=seed, tolerance=tolerance
        )
        print(f"    Best value: {boar[1]:.6f}")
        print(f"    Sum(x²): {np.sum(boar[0] ** 2):.6f}")
        print(f"    Feasible (outside sphere): {np.sum(boar[0] ** 2) > n_dim * 5}")

        # Run Optuna optimization
        print("  Running Optuna optimization...")
        optuna = run_optuna_optimization(
            n_dim=n_dim, n_trials=n_trials, n_initial=n_initial, bounds=bounds[:n_dim], seed=seed, tolerance=tolerance
        )
        print(f"    Best value: {optuna[1]:.6f}")
        optuna_sum_sq = sum(optuna[0][f"x{i}"] ** 2 for i in range(n_dim))
        print(f"    Sum(x²): {optuna_sum_sq:.6f}")
        print(f"    Feasible (outside sphere): {optuna_sum_sq > n_dim * 5}")

        all_results[n_dim] = (boar, optuna)

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    plot_rosenbrock_contour(bounds[:2], 2, output_dir / "rosenbrock_constrained_contour.svg")
    plot_convergence_history(all_results, output_dir / "rosenbrock_constrained_convergence.svg")
    plot_search_history(all_results, bounds, 2, output_dir / "rosenbrock_search_constrained_history.svg")
    plot_comparison(all_results, output_dir / "rosenbrock_constrained_comparison.svg")
    generate_summary_report(all_results, output_dir / "rosenbrock_constrained_results.txt")

    print("\n" + "=" * 60)
    print("All benchmark images generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
