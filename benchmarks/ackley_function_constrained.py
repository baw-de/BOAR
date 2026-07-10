"""
Generate benchmark plots for the constrained Ackley function.
Constraint: sum(x) > 0
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

from src import bo_optimizer, functions_sampler, utils

plt.style.use("science")
matplotlib.use("Agg")


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    """Standard Ackley function."""
    x = np.asarray(x)
    n = x.size
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(c * x)) / n)
    return term1 + term2 + a + np.e


def ackley_constrained(args, penalty_weight=1e6):
    """
    Ackley function with penalty for constraint violations.
    Constraint: sum(x) > 0
    """
    x = np.asarray(args)
    ackley_value = ackley(x)

    if x.sum() <= 0:
        return ackley_value + penalty_weight * (1 + abs(x.sum()))

    return ackley_value


def create_sum_constraint(n_dim):
    """Create a constraint function for sum(x) > 0 using utils.create_constraint_function."""
    variables = [f"x{i}" for i in range(n_dim)]
    expression = " + ".join(variables) + " > 0"

    return utils.create_constraint_function(expression, variables)


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
        bounds = [(-32.768, 32.768)] * n_dim

    # Create constraint function
    constraint_fn = create_sum_constraint(n_dim)

    # Constrained sampler
    sampler = functions_sampler.LatinHypercube(bounds=bounds, precision=1e-9, constraint_fns=[constraint_fn], seed=seed)

    # Generate initial samples
    test_samples = sampler.generate_samples(n_initial)
    test_values = np.array([ackley_constrained(sample) for sample in test_samples])
    initial_samples = (test_samples, test_values)

    # Run optimization
    optimizer = bo_optimizer.BayesianOptimizer(
        initial_samples=initial_samples,
        obj_func=ackley_constrained,
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
    """Run Optuna optimization with constrained sampling."""
    if bounds is None:
        bounds = [(-32.768, 32.768)] * n_dim

    def objective(trial: optuna.Trial) -> float:
        vector = np.array(
            [trial.suggest_float(f"x{i}", lb, ub) for i, (lb, ub) in enumerate(bounds)],
            dtype=float,
        )
        return ackley_constrained(vector)

    def constraints_func(trial: optuna.trial.FrozenTrial) -> tuple:
        """
        Constraint: sum(x) > 0

        In Optuna, constraints must be <= 0 to be feasible.
        Since we want sum(x) > 0, we return -sum(x) so that:
        - If sum(x) > 0 (feasible): return -sum(x) <= 0
        - If sum(x) <= 0 (infeasible): return -sum(x) > 0
        """
        if not trial.params:
            return (0.0,)
        x = np.array([trial.params[f"x{i}"] for i in range(n_dim)])
        return (-x.sum(),)  # Negate so that sum(x) > 0 means constraint <= 0

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
        constraints_func=constraints_func,
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


def plot_ackley_contour(bounds, save_path):
    """Plot 2D Ackley function contour with constraint boundary."""
    x = np.linspace(bounds[0][0], bounds[0][1], 200)
    y = np.linspace(bounds[1][0], bounds[1][1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[ackley([xi, yi]) for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(contour, label="f(x, y)")

    # Add constraint boundary
    plt.axline((0, 0), slope=-1, color="red", linestyle="--", linewidth=2, label="Constraint: $x + y = 0$")

    # Shade infeasible region
    plt.contourf(X, Y, (X + Y <= 0).astype(int), levels=[0.5, 1], colors=["red"], alpha=0.3)

    plt.scatter(
        [0], [0], color="lime", s=300, marker="*", label="Global min (unconstrained)", zorder=5, edgecolors="black"
    )

    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$y$", fontsize=14)
    plt.title(r"Ackley Function with Constraint $x + y > 0$", fontsize=16)
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

    colors = {"BOAR": "steelblue", "Optuna": "coral"}

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

        ax.plot(range(len(boar_best)), boar_best, label="BOAR", color=colors["BOAR"], linewidth=2)
        ax.plot(range(len(optuna_best)), optuna_best, label="Optuna", color=colors["Optuna"], linewidth=2)

        ax.set_xlabel("Function Evaluations", fontsize=12)
        ax.set_ylabel("Best Value Found", fontsize=12)
        ax.set_title(f"n_dim = {n_dim}", fontsize=14)
        ax.set_yscale("log")
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(all_results), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Convergence History - Constrained Ackley Function", fontsize=16, y=1.02)
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
    Z = np.array([[ackley([xi, yi]) for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

    contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis", alpha=0.7)
    plt.colorbar(contour, label="f(x, y)")

    # Add constraint boundary
    ax.axline((0, 0), slope=-1, color="red", linestyle="--", linewidth=3, label=r"Constraint: $x + y = 0$")

    # Infeasible region
    infeasible = X + Y <= 0
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

    # Mark global minimum
    ax.scatter([0], [0], color="lime", s=300, marker="*", label="Unconstrained min (0,0)", zorder=6, edgecolors="black")

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
    ax.set_title(r"Search History - Constrained Ackley Function (2D)", fontsize=16)
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
    boar_margins = [np.sum(all_results[n][0][0]) for n in dimensions]
    optuna_margins = [sum(all_results[n][1][0].values()) for n in dimensions]

    x = np.arange(len(dimensions))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Best values comparison
    ax1 = axes[0]
    ax1.bar(x - width / 2, boar_values, width, label="BOAR", color="steelblue")
    ax1.bar(x + width / 2, optuna_values, width, label="Optuna", color="coral")
    ax1.set_xlabel("Number of Dimensions", fontsize=12)
    ax1.set_ylabel("Best Value Found", fontsize=12)
    ax1.set_title("Best Value Comparison", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dimensions)
    ax1.legend(fontsize=10)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3, axis="y")

    # Constraint satisfaction
    ax2 = axes[1]
    ax2.bar(x - width / 2, boar_margins, width, label="BOAR", color="steelblue")
    ax2.bar(x + width / 2, optuna_margins, width, label="Optuna", color="coral")
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=2, label="Constraint boundary")
    ax2.set_xlabel("Number of Dimensions", fontsize=12)
    ax2.set_ylabel(r"$\sum(x)$ - Constraint Margin", fontsize=12)
    ax2.set_title(r"Constraint Satisfaction $\left (\sum(x) > 0 \right )$", fontsize=14)
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
    example_vars = [f"x{i}" for i in range(2)]
    example_expr = " + ".join(example_vars) + " > 0"

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Constrained Ackley Function Benchmark Results\n")
        f.write("Constraint: sum(x) > 0\n")
        f.write(f'Example expression: "{example_expr}"\n')
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Dim':<5} {'BOAR Best':<15} {'Optuna Best':<15} {'BOAR Sum(x)':<15} {'Optuna Sum(x)':<15}\n")
        f.write("-" * 80 + "\n")

        for n_dim in sorted(all_results.keys()):
            boar, optuna = all_results[n_dim]
            f.write(
                f"{n_dim:<5} {boar[1]:<15.6f} {optuna[1]:<15.6f} "
                f"{np.sum(boar[0]):<15.6f} {sum(optuna[0].values()):<15.6f}\n"
            )

        f.write("-" * 80 + "\n\n")
        f.write("Best Parameters:\n")
        f.write("-" * 80 + "\n")

        for n_dim in sorted(all_results.keys()):
            boar, optuna = all_results[n_dim]
            f.write(f"\nn_dim = {n_dim}:\n")
            f.write("  BOAR:\n")
            f.write(f"    Sample: {boar[0]}\n")
            f.write(f"    Value: {boar[1]:.10f}\n")
            f.write(f"    Feasible: {np.sum(boar[0]) > 0}\n")
            f.write("  Optuna:\n")
            f.write(f"    Params: {optuna[0]}\n")
            f.write(f"    Value: {optuna[1]:.10f}\n")
            f.write(f"    Feasible: {sum(optuna[0].values()) > 0}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Global Minimum (unconstrained): 0.0 at x = (0, 0, ..., 0)\n")
        f.write("Search Domain: [-32.768, 32.768]^n\n")
        f.write("=" * 80 + "\n")

    print(f"Saved: {output_path}")


def main():
    """Run all benchmarks for multiple dimensions."""
    dimensions = [2, 3, 4, 5, 10, 15, 20]
    n_initial = 15
    n_trials = 100
    seed = 9
    bounds = [(-32.768, 32.768)] * max(dimensions)
    tolerance = 1e-6

    output_dir = module_path / "docs" / "_static" / "images" / "ackley_constrained"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for n_dim in dimensions:
        print(f"\n{'=' * 60}")
        print(f"Running benchmarks for n_dim = {n_dim}")
        print(f"{'=' * 60}")

        variables = [f"x{i}" for i in range(n_dim)]
        expr = " + ".join(variables) + " > 0"
        print(f"Constraint: {expr}")

        # Run BOAR optimization
        print("  Running BOAR optimization...")
        boar = run_boar_optimization(
            n_dim=n_dim, n_trials=n_trials, n_initial=n_initial, bounds=bounds[:n_dim], seed=seed, tolerance=tolerance
        )
        print(f"    Best value: {boar[1]:.6f}")
        print(f"    Constraint satisfied: {np.sum(boar[0]) > 0}")

        # Run Optuna optimization
        print("  Running Optuna optimization...")
        optuna = run_optuna_optimization(
            n_dim=n_dim, n_trials=n_trials, n_initial=n_initial, bounds=bounds[:n_dim], seed=seed, tolerance=tolerance
        )
        print(f"    Best value: {optuna[1]:.6f}")
        print(f"    Constraint satisfied: {sum(optuna[0].values()) > 0}")

        all_results[n_dim] = (boar, optuna)

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    plot_ackley_contour(bounds[:2], output_dir / "ackley_constrained_contour.svg")
    plot_convergence_history(all_results, output_dir / "ackley_constrained_convergence.svg")
    plot_search_history(all_results, bounds, output_dir / "ackley_search_constrained_history.svg")
    plot_comparison(all_results, output_dir / "ackley_constrained_comparison.svg")
    generate_summary_report(all_results, output_dir / "ackley_constrained_results.txt")

    print("\n" + "=" * 60)
    print("All benchmark images generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
