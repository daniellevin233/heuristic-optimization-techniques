from pathlib import Path
from typing import Literal
import json
import matplotlib.pyplot as plt
from datetime import datetime


# Parameter definitions for each algorithm
ALNS_PARAMS = [
    "weight_update_period",
    "reaction_factor",
    "min_removal_pct",
    "max_removal_pct",
    "initial_temp",
    "cooling_rate",
    "score_new_best",
    "score_accepted",
]

SA_PARAMS = [
    "initial_temp",
    "cooling_rate",
    "equilibrium_iter",
    "rcl_length",
]

# Human-readable parameter names
PARAM_LABELS = {
    # ALNS
    "weight_update_period": "Weight Update Period",
    "reaction_factor": "Reaction Factor (γ)",
    "min_removal_pct": "Min Removal %",
    "max_removal_pct": "Max Removal %",
    "initial_temp": "Initial Temperature",
    "cooling_rate": "Cooling Rate",
    "score_new_best": "Score: New Best",
    "score_accepted": "Score: Accepted",
    # SA
    "equilibrium_iter": "Equilibrium Iterations",
    "rcl_length": "RCL Length",
    # Common
    "best_objective": "Best Objective",
}

# Optuna search space ranges (from tuning code)
ALNS_SEARCH_SPACE = {
    "weight_update_period": (50, 500),
    "reaction_factor": (0.01, 0.5),
    "min_removal_pct": (0.05, 0.2),
    "max_removal_pct": (0.3, 0.6),
    "initial_temp": (50.0, 500.0),
    "cooling_rate": (0.95, 0.9999),
    "score_new_best": (5.0, 20.0),
    "score_accepted": (0.5, 2.0),
}

SA_SEARCH_SPACE = {
    "initial_temp": (50.0, 1000.0),
    "cooling_rate": (0.85, 0.9999),
    "equilibrium_iter": (50, 500),
    "rcl_length": (1, 10),
}


def load_tuned_params(
    algorithm: Literal["ALNS", "SA"],
    instance_sizes: list[int],
    tuning_dir: Path = None,
) -> dict[int, dict]:
    """
    Load tuned parameter JSON files for specified instance sizes.

    Returns:
        Dictionary mapping instance_size -> tuning_result
    """
    if tuning_dir is None:
        from src.utils import find_project_root
        project_root = find_project_root()

        if algorithm == "ALNS":
            tuning_dir = project_root / "src" / "algorithms" / "alns" / "tuning"
        elif algorithm == "SA":
            tuning_dir = project_root / "src" / "algorithms" / "sa" / "tuning"
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    param_data = {}
    missing_sizes = []

    for size in instance_sizes:
        json_file = tuning_dir / f"tuned_params_n{size}.json"
        if json_file.exists():
            with open(json_file) as f:
                param_data[size] = json.load(f)
        else:
            missing_sizes.append(size)

    if missing_sizes:
        print(f"Warning: Missing tuned params for sizes: {missing_sizes}")

    return param_data


def plot_tuning_trends(
    algorithm: Literal["ALNS", "SA"],
    instance_sizes: list[int] = None,
    tuning_dir: Path = None,
    save_plot: bool = True,
    output_dir: Path = None,
    figsize: tuple[float, float] = (14, 10),
) -> None:
    """
    Plot tuned parameters as parallel trend lines showing progression across instance sizes.

    Creates a clear visualization where each parameter is a line showing how it changes
    with instance size. Makes scaling patterns immediately obvious.

    Args:
        algorithm: "ALNS" or "SA"
        instance_sizes: List of instance sizes to plot
        tuning_dir: Directory containing tuned_params_n{size}.json files
        save_plot: Whether to save plot to file
        output_dir: Where to save plot (default: plots/)
        figsize: Figure size (width, height)
    """
    import numpy as np

    # Default instance sizes
    if instance_sizes is None:
        instance_sizes = [50, 100, 200, 500]

    # Load parameter data
    print(f"Loading tuned parameters for {algorithm}...")
    param_data = load_tuned_params(algorithm, instance_sizes, tuning_dir)

    if not param_data:
        print(f"Error: No tuned parameters found for {algorithm}")
        return

    # Get parameter list and search space for this algorithm
    if algorithm == "ALNS":
        param_names = ALNS_PARAMS
        search_space = ALNS_SEARCH_SPACE
    elif algorithm == "SA":
        param_names = SA_PARAMS
        search_space = SA_SEARCH_SPACE
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Extract sorted instance sizes
    sorted_sizes = sorted(param_data.keys())
    n_sizes = len(sorted_sizes)
    n_params = len(param_names)

    # Build data matrix: rows=parameters, cols=instance_sizes
    data_matrix = np.zeros((n_params, n_sizes))
    for i, param_name in enumerate(param_names):
        for j, size in enumerate(sorted_sizes):
            data_matrix[i, j] = param_data[size]["best_params"][param_name]

    # Normalize each row based on SEARCH SPACE range, not actual values
    # This shows how tuned values sit within the allowed parameter range
    normalized_matrix = np.zeros_like(data_matrix)
    for i in range(n_params):
        param_name = param_names[i]
        row = data_matrix[i, :]

        # Get search space bounds
        space_min, space_max = search_space[param_name]

        # Normalize to [0, 1] based on search space
        if space_max > space_min:
            normalized_matrix[i, :] = (row - space_min) / (space_max - space_min)
        else:
            normalized_matrix[i, :] = 0.5

    # Create figure with subplots (one per parameter)
    n_cols = 2
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_params > 1 else [axes]

    # Color scheme based on trend
    for idx, param_name in enumerate(param_names):
        ax = axes[idx]

        values = data_matrix[idx, :]
        norm_values = normalized_matrix[idx, :]

        # Determine trend: increasing, decreasing, or stable
        first_val = norm_values[0]
        last_val = norm_values[-1]
        trend_change = last_val - first_val

        if abs(trend_change) < 0.1:  # Stable
            color = "dimgray"
            trend_arrow = "→"
        elif trend_change > 0:  # Increasing
            color = "steelblue"
            trend_arrow = "↑"
        else:  # Decreasing
            color = "darkorange"
            trend_arrow = "↓"

        # Plot line with markers
        ax.plot(
            range(n_sizes),
            norm_values,
            marker="o",
            linewidth=3,
            markersize=10,
            color=color,
            markeredgecolor="white",
            markeredgewidth=2,
            alpha=0.8
        )

        # Fill area under curve
        ax.fill_between(range(n_sizes), 0, norm_values, alpha=0.2, color=color)

        # Annotate with actual values
        for j, (size, value) in enumerate(zip(sorted_sizes, values)):
            if param_name in ["weight_update_period", "equilibrium_iter", "rcl_length"]:
                text = f"{int(value)}"
            else:
                text = f"{value:.2f}"

            ax.annotate(
                text,
                (j, norm_values[j]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=color
            )

        # Add search space bounds as gray shaded region
        ax.axhspan(-0.05, 0, facecolor='lightgray', alpha=0.3)
        ax.axhspan(1.0, 1.05, facecolor='lightgray', alpha=0.3)

        # Formatting
        ax.set_xticks(range(n_sizes))
        ax.set_xticklabels([f"n={size}" for size in sorted_sizes], fontsize=10)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.5, 1.0])

        # Get search space for y-axis labels
        space_min, space_max = search_space[param_name]
        if param_name in ["weight_update_period", "equilibrium_iter", "rcl_length"]:
            ax.set_yticklabels([f"{int(space_min)}", f"{int((space_min + space_max) / 2)}", f"{int(space_max)}"], fontsize=9)
        else:
            ax.set_yticklabels([f"{space_min:.2f}", f"{(space_min + space_max) / 2:.2f}", f"{space_max:.2f}"], fontsize=9)

        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

        # Title with trend indicator and search space range
        param_label = PARAM_LABELS.get(param_name, param_name)

        if param_name in ["weight_update_period", "equilibrium_iter", "rcl_length"]:
            space_text = f"[{int(space_min)}–{int(space_max)}]"
        else:
            space_text = f"[{space_min:.2f}–{space_max:.2f}]"

        ax.set_title(
            f"{param_label} {trend_arrow}  {space_text}",
            fontsize=11,
            fontweight="bold",
            color=color,
            pad=10
        )

    # Hide empty subplots
    for idx in range(n_params, len(axes)):
        axes[idx].axis("off")

    # Overall title
    fig.suptitle(
        f"{algorithm} Parameter Tuning: Scaling Trends Across Instance Sizes",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )

    # Add legend explaining the visualization
    legend_text = "↑ = Increases with size | ↓ = Decreases with size | → = Stable"
    fig.text(
        0.5, 0.01, legend_text,
        ha="center",
        fontsize=10,
        style="italic",
        color="gray"
    )

    plt.tight_layout(rect=(0, 0.02, 1, 0.99))

    # Save plot
    if save_plot:
        if output_dir is None:
            from src.utils import find_project_root
            project_root = find_project_root()
            output_dir = project_root / "plots"

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{algorithm.lower()}_tuning_trends_{timestamp}.png"
        output_path = output_dir / filename

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.show()


if __name__ == "__main__":
    print("=" * 80)
    print("Visualizing ALNS Tuning Progression (Trends)")
    print("=" * 80)
    plot_tuning_trends(
        algorithm="ALNS",
        instance_sizes=[50, 100, 200, 500],
        save_plot=False,
    )

    # print("\n" + "=" * 80)
    # print("Visualizing SA Tuning Progression (Trends)")
    # print("=" * 80)
    # try:
    #     plot_tuning_trends(
    #         algorithm="SA",
    #         instance_sizes=[50, 100, 200, 500],
    #         save_plot=False,
    #     )
    # except Exception as e:
    #     print(f"Could not create SA visualization: {e}")