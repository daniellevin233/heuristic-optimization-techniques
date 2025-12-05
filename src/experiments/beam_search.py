"""
Beam Search Parameter Experimentation Framework.

This module provides functionality for analyzing the sensitivity of beam search
algorithm performance to its two key parameters: beam_width and branching_factor.
"""

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

from src.algorithms.beam_search import SCFPDPBeamSearch
from src.instance import SCFPDPInstance
from src.experiments.construction_heuristics import InstanceType
from src.utils import find_project_root


def evaluate_beam_search_parameters(
    instance_size: str,
    instance_type: InstanceType,
    beam_widths: list[int],
    branching_factors: list[int],
    save_plot: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate beam search performance across different parameter combinations.

    Args:
        instance_size: Size of instance to evaluate (e.g., "100")
        instance_type: Type of instance (COMPETITION, TEST, TRAIN)
        beam_widths: List of beam width values to test
        branching_factors: List of branching factor values to test
        save_plot: Whether to save the plot to file

    Returns:
        Tuple of (objectives_grid, runtimes_grid) as 2D numpy arrays
    """
    # Load instance (first file from directory)
    project_root = find_project_root()
    instance_dir = project_root / "instances" / instance_size / instance_type
    instance_files = sorted(instance_dir.glob("*.txt"))

    if not instance_files:
        raise ValueError(f"No instances found in {instance_dir}")

    instance_file = instance_files[0]
    print(f"Loading instance: {instance_file.name}")
    instance = SCFPDPInstance(str(instance_file))

    print(f"\n{'='*80}")
    print(f"  BEAM SEARCH PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Instance: {instance_file.name} (n={instance.n})")
    print(f"Beam widths: {beam_widths}")
    print(f"Branching factors: {branching_factors}")
    print(f"Total combinations: {len(beam_widths) * len(branching_factors)}\n")

    # Initialize result grids
    objectives = np.zeros((len(beam_widths), len(branching_factors)))
    runtimes = np.zeros((len(beam_widths), len(branching_factors)))

    # Run experiments
    total_runs = len(beam_widths) * len(branching_factors)
    with tqdm(total=total_runs, desc="Parameter fine-tune") as pbar:
        for i, beam_width in enumerate(beam_widths):
            for j, branching_factor in enumerate(branching_factors):
                bs = SCFPDPBeamSearch(instance, beam_width, branching_factor)
                solution, runtime = bs.solve()

                # Store results
                objectives[i, j] = solution.calc_objective()
                runtimes[i, j] = runtime

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'beta': beam_width,
                    'bf': branching_factor,
                    'obj': f'{objectives[i, j]:.1f}',
                    'time': f'{runtime:.2f}s'
                })

    print_summary(instance, beam_widths, branching_factors, objectives, runtimes)

    plot_beam_search_heatmaps(
        instance, beam_widths, branching_factors,
        objectives, runtimes, save_plot
    )

    return objectives, runtimes


def print_summary(
    instance: SCFPDPInstance,
    beam_widths: list[int],
    branching_factors: list[int],
    objectives: np.ndarray,
    runtimes: np.ndarray
) -> None:
    """
    Print formatted summary of beam search parameter sweep results.

    Args:
        instance: The problem instance that was solved
        beam_widths: List of beam width values tested
        branching_factors: List of branching factor values tested
        objectives: 2D array of objective values
        runtimes: 2D array of runtimes
    """
    # Find best and worst combinations
    best_i, best_j = np.unravel_index(np.argmin(objectives), objectives.shape)
    worst_i, worst_j = np.unravel_index(np.argmax(objectives), objectives.shape)

    # Calculate statistics
    obj_min = np.min(objectives)
    obj_max = np.max(objectives)
    obj_mean = np.mean(objectives)
    runtime_min = np.min(runtimes)
    runtime_max = np.max(runtimes)
    runtime_mean = np.mean(runtimes)

    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Instance: {instance} (n={instance.n})")
    print(f"Beam widths tested: {beam_widths}")
    print(f"Branching factors tested: {branching_factors}")
    print(f"Total combinations: {len(beam_widths) * len(branching_factors)}")
    print()
    print(f"BEST RESULT:")
    print(f"  beam_width={beam_widths[best_i]}, branching_factor={branching_factors[best_j]}")
    print(f"  Objective: {objectives[best_i, best_j]:.2f}")
    print(f"  Runtime: {runtimes[best_i, best_j]:.3f}s")
    print()
    print(f"WORST RESULT:")
    print(f"  beam_width={beam_widths[worst_i]}, branching_factor={branching_factors[worst_j]}")
    print(f"  Objective: {objectives[worst_i, worst_j]:.2f}")
    print(f"  Runtime: {runtimes[worst_i, worst_j]:.3f}s")
    print()
    print(f"STATISTICS:")
    print(f"  Objective: min={obj_min:.2f}, max={obj_max:.2f}, mean={obj_mean:.2f}")
    print(f"  Runtime: min={runtime_min:.3f}s, max={runtime_max:.3f}s, mean={runtime_mean:.3f}s")
    print(f"{'='*80}\n")


def plot_beam_search_heatmaps(
    instance: SCFPDPInstance,
    beam_widths: list[int],
    branching_factors: list[int],
    objectives: np.ndarray,
    runtimes: np.ndarray,
    save_plot: bool = False
) -> None:
    """
    Create heatmap visualizations of beam search parameter sensitivity.

    Args:
        instance: The problem instance that was solved
        beam_widths: List of beam width values tested
        branching_factors: List of branching factor values tested
        objectives: 2D array of objective values
        runtimes: 2D array of runtimes
        save_plot: Whether to save the plot to file
    """
    # Create figure with 2 heatmap subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Beam Search Parameter Fine-Tuning (n={instance.n})',
                 fontsize=14, fontweight='bold')

    # Plot 1: Objective Values Heatmap
    im1 = ax1.imshow(objectives, cmap='RdYlGn_r', aspect='auto')
    ax1.set_xticks(range(len(branching_factors)))
    ax1.set_xticklabels(branching_factors)
    ax1.set_yticks(range(len(beam_widths)))
    ax1.set_yticklabels(beam_widths)
    ax1.set_xlabel('Branching Factor', fontsize=12)
    ax1.set_ylabel('Beam Width', fontsize=12)
    ax1.set_title('Objective Value', fontsize=12, fontweight='bold')

    # Annotate cells with objective values
    for i in range(len(beam_widths)):
        for j in range(len(branching_factors)):
            # Choose text color based on background intensity
            value = objectives[i, j]
            # Normalize to 0-1 range for color decision
            norm_value = (value - objectives.min()) / (objectives.max() - objectives.min() + 1e-10)
            text_color = 'white' if norm_value < 0.5 else 'black'

            ax1.text(j, i, f'{objectives[i, j]:.0f}',
                    ha="center", va="center", color=text_color, fontsize=8)

    # Mark best combination with a blue rectangle
    best_i, best_j = np.unravel_index(np.argmin(objectives), objectives.shape)
    rect = Rectangle((best_j-0.5, best_i-0.5), 1, 1,
                    fill=False, edgecolor='blue', linewidth=3)
    ax1.add_patch(rect)

    plt.colorbar(im1, ax=ax1, label='Objective Value')

    # Plot 2: Runtime Heatmap
    im2 = ax2.imshow(runtimes, cmap='RdYlGn_r', aspect='auto')
    ax2.set_xticks(range(len(branching_factors)))
    ax2.set_xticklabels(branching_factors)
    ax2.set_yticks(range(len(beam_widths)))
    ax2.set_yticklabels(beam_widths)
    ax2.set_xlabel('Branching Factor', fontsize=12)
    ax2.set_ylabel('Beam Width', fontsize=12)
    ax2.set_title('Runtime (seconds)', fontsize=12, fontweight='bold')

    # Annotate cells with runtime values
    for i in range(len(beam_widths)):
        for j in range(len(branching_factors)):
            # Choose text color based on background intensity
            value = runtimes[i, j]
            # Normalize to 0-1 range for color decision
            norm_value = (value - runtimes.min()) / (runtimes.max() - runtimes.min() + 1e-10)
            text_color = 'white' if norm_value < 0.5 else 'black'

            ax2.text(j, i, f'{runtimes[i, j]:.2f}',
                    ha="center", va="center", color=text_color, fontsize=8)

    best_i, best_j = np.unravel_index(np.argmin(runtimes), runtimes.shape)
    rect = Rectangle((best_j-0.5, best_i-0.5), 1, 1,
                     fill=False, edgecolor='blue', linewidth=3)
    ax2.add_patch(rect)

    plt.colorbar(im2, ax=ax2, label='Runtime (seconds)')

    plt.tight_layout()

    # Save or show plot
    if save_plot:
        project_root = find_project_root()
        plots_dir = project_root / "plots"
        plots_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = plots_dir / f"beam_search_params_n{instance.n}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")
        plt.close()  # Close figure without showing
    else:
        plt.show()  # Only show interactively when not saving


def evaluate_beam_search_multi_size(
    instance_sizes: list[str],
    instance_type: InstanceType,
    beam_widths: list[int],
    branching_factors: list[int],
    save_plot: bool
) -> None:
    """
    Run parameter sweep across multiple instance sizes.

    Args:
        instance_sizes: List of instance sizes to test (e.g., ["50", "100", "200"])
        instance_type: Type of instances (COMPETITION, TEST, TRAIN)
        beam_widths: List of beam width values to test
        branching_factors: List of branching factor values to test
        save_plot: Whether to save plots to file
    """
    for instance_size in tqdm(instance_sizes, desc="Instance Size"):
        print(f"\n{'#'*80}")
        print(f"# Processing instance size: {instance_size}")
        print(f"{'#'*80}\n")

        evaluate_beam_search_parameters(
            instance_size=instance_size,
            instance_type=instance_type,
            beam_widths=beam_widths,
            branching_factors=branching_factors,
            save_plot=save_plot
        )


if __name__ == "__main__":
    all_instance_sizes = ["50", "100", "200", "500", "1000", "2000", "5000", "10000"]
    instances_to_run = ["100", "200", "500"]

    # Default parameter ranges: smaller values, more of them
    beam_widths = [2, 3, 5, 7, 10, 15, 20, 25, 30]
    branching_factors = [2, 3, 4, 5, 6, 8, 10]
    
    evaluate_beam_search_multi_size(
        instance_sizes=instances_to_run,
        instance_type=InstanceType.COMPETITION,
        beam_widths=beam_widths,
        branching_factors=branching_factors,
        save_plot=True
    )