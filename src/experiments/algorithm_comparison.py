"""
Algorithm Comparison Framework for Construction Heuristics.

This module provides a general framework for comparing construction heuristic algorithms
on the SCF-PDP problem across multiple instance sizes.
"""

import gc
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

from src.instance import SCFPDPInstance
from src.solution import SCFPDPSolution
from src.algorithms.construction_heuristics import GreedyConstructionHeuristic, RandomizedConstructionHeuristic
from src.algorithms.beam_search import SCFPDPBeamSearch
from src.algorithms.sa import SCFPDPSA
from src.neighborhoods import RelocateNeighborhood
from src.experiments.construction_heuristics import InstanceType
from src.utils import find_project_root


# ==================== Data Structures ====================

@dataclass
class AlgorithmConfig:
    """Configuration for a single algorithm to test."""
    name: str  # e.g., "Greedy", "Randomized-k10"
    algorithm_class: type  # The heuristic class
    init_kwargs: dict  # kwargs for __init__


@dataclass
class InstanceResult:
    """Results for a single instance run."""
    instance_name: str
    instance_size: int
    algorithm_name: str
    objective_value: float
    construction_time: float
    total_time: float
    parsing_time: float


@dataclass
class ComparisonResults:
    """Comparison between two algorithms for one instance size."""
    instance_size: str
    algorithm1_name: str
    algorithm2_name: str
    algorithm1_mean_obj: float
    algorithm1_std_obj: float
    algorithm2_mean_obj: float
    algorithm2_std_obj: float
    mean_percent_diff: float  # (alg1 - alg2) / alg1 * 100
    std_percent_diff: float
    algorithm1_wins: int
    algorithm2_wins: int
    ties: int
    n_instances: int


@dataclass
class ExperimentSummary:
    """Complete summary across all instance sizes."""
    comparison_results: list[ComparisonResults]
    algorithm1_total_wins: int
    algorithm2_total_wins: int
    total_ties: int
    total_instances: int


# ==================== Core Functions ====================

def run_algorithm_on_instance(
    instance: SCFPDPInstance,
    algorithm_config: AlgorithmConfig
) -> InstanceResult:
    """
    Run a single algorithm on a single instance and collect results.

    Args:
        instance: SCFPDPInstance
        algorithm_config: Configuration for the algorithm to run

    Returns:
        InstanceResult with timing and objective information
    """
    start_time = time.time()

    # Load instance
    solution = SCFPDPSolution(instance)
    parsing_time = time.time() - start_time

    # Run construction heuristic
    construction_start = time.time()
    heuristic = algorithm_config.algorithm_class(solution, **algorithm_config.init_kwargs)
    heuristic.construct()
    construction_time = time.time() - construction_start

    # Calculate objective
    objective = solution.calc_objective()
    total_time = time.time() - start_time

    return InstanceResult(
        instance_name=str(instance),
        instance_size=instance.n,
        algorithm_name=algorithm_config.name,
        objective_value=objective,
        construction_time=construction_time,
        total_time=total_time,
        parsing_time=parsing_time
    )


def process_single_instance(instance_file, algorithm1_config, algorithm2_config):
    instance = SCFPDPInstance(str(instance_file))
    alg1_result = run_algorithm_on_instance(instance, algorithm1_config)
    alg2_result = run_algorithm_on_instance(instance, algorithm2_config)
    return alg1_result, alg2_result


def compare_on_single_size(
    instance_size: str,
    instance_type: InstanceType,
    algorithm1_config: AlgorithmConfig,
    algorithm2_config: AlgorithmConfig,
) -> ComparisonResults:
    """
    Compare two algorithms on all instances of a given size.

    Args:
        instance_size: Size of instances to test (e.g., "100", "500")
        instance_type: Type of instances (TEST, TRAIN, COMPETITION)
        algorithm1_config: Configuration for first algorithm
        algorithm2_config: Configuration for second algorithm

    Returns:
        ComparisonResults with statistics and win counts
    """
    # Load all instance files
    project_root = find_project_root()
    instance_dir = project_root / "instances" / instance_size / instance_type
    instance_files = sorted(instance_dir.glob("*.txt"))

    if not instance_files:
        raise ValueError(f"No instances found in {instance_dir}")

    # Run both algorithms on each instance in parallel
    # Limit workers for large instances to avoid memory issues
    instance_size_int = int(instance_size)
    if instance_size_int >= 5000:
        n_workers = min(2, len(instance_files))  # Max 2 workers for large instances
    else:
        n_workers = min(cpu_count(), len(instance_files))

    process_func = partial(process_single_instance,
                          algorithm1_config=algorithm1_config,
                          algorithm2_config=algorithm2_config)

    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, instance_files, chunksize=1),
            total=len(instance_files),
            desc=f"Size {instance_size}"
        ))

    alg1_results = [r[0] for r in results]
    alg2_results = [r[1] for r in results]

    # Clean up memory
    del results
    gc.collect()

    # Extract objective values
    alg1_objs = np.array([r.objective_value for r in alg1_results])
    alg2_objs = np.array([r.objective_value for r in alg2_results])

    # Calculate statistics
    alg1_mean = float(np.mean(alg1_objs))
    alg1_std = float(np.std(alg1_objs, ddof=1))
    alg2_mean = float(np.mean(alg2_objs))
    alg2_std = float(np.std(alg2_objs, ddof=1))

    # Calculate pairwise percentage differences
    percent_diffs = (alg1_objs - alg2_objs) / alg1_objs * 100
    mean_percent_diff = float(np.mean(percent_diffs))
    std_percent_diff = float(np.std(percent_diffs, ddof=1))

    # Count wins (lower objective is better for minimization)
    alg1_wins = int(np.sum(alg1_objs < alg2_objs))
    alg2_wins = int(np.sum(alg2_objs < alg1_objs))
    ties = int(np.sum(alg1_objs == alg2_objs))

    print(f"  {algorithm1_config.name} wins: {alg1_wins}, {algorithm2_config.name} wins: {alg2_wins}, Ties: {ties}")

    return ComparisonResults(
        instance_size=instance_size,
        algorithm1_name=algorithm1_config.name,
        algorithm2_name=algorithm2_config.name,
        algorithm1_mean_obj=alg1_mean,
        algorithm1_std_obj=alg1_std,
        algorithm2_mean_obj=alg2_mean,
        algorithm2_std_obj=alg2_std,
        mean_percent_diff=mean_percent_diff,
        std_percent_diff=std_percent_diff,
        algorithm1_wins=alg1_wins,
        algorithm2_wins=alg2_wins,
        ties=ties,
        n_instances=len(instance_files)
    )


def compare_algorithms_across_sizes(
    instance_sizes: list[str],
    instance_type: InstanceType,
    algorithm1_config: AlgorithmConfig,
    algorithm2_config: AlgorithmConfig,
) -> ExperimentSummary:
    """
    Main entry point: compare two algorithms across multiple instance sizes.

    Args:
        instance_sizes: List of instance sizes to test (e.g., ["100", "500", "1000"])
        instance_type: Type of instances (TEST, TRAIN, COMPETITION)
        algorithm1_config: Configuration for first algorithm
        algorithm2_config: Configuration for second algorithm

    Returns:
        ExperimentSummary with complete comparison results
    """
    print("="*80)
    print(f"  CONSTRUCTION HEURISTICS COMPARISON: {algorithm1_config.name} vs {algorithm2_config.name}")
    print("="*80)
    print(f"Instance Type: {instance_type.value}")
    print(f"Instance Sizes: {', '.join(instance_sizes)}")

    # Compare on each instance size
    comparison_results = []
    for instance_size in instance_sizes:
        comp = compare_on_single_size(
            instance_size=instance_size,
            instance_type=instance_type,
            algorithm1_config=algorithm1_config,
            algorithm2_config=algorithm2_config,
        )
        comparison_results.append(comp)

    # Aggregate total wins
    algorithm1_total_wins = sum(comp.algorithm1_wins for comp in comparison_results)
    algorithm2_total_wins = sum(comp.algorithm2_wins for comp in comparison_results)
    total_ties = sum(comp.ties for comp in comparison_results)
    total_instances = sum(comp.n_instances for comp in comparison_results)

    summary = ExperimentSummary(
        comparison_results=comparison_results,
        algorithm1_total_wins=algorithm1_total_wins,
        algorithm2_total_wins=algorithm2_total_wins,
        total_ties=total_ties,
        total_instances=total_instances
    )

    # Print summary table
    print_comparison_table(summary)

    return summary


def print_comparison_table(summary: ExperimentSummary) -> None:
    """
    Print formatted comparison table to console using pandas.

    Args:
        summary: ExperimentSummary containing all comparison results
    """
    # Build table data
    data = []
    for comp in summary.comparison_results:
        row = {
            'Size': comp.instance_size,
            f'{comp.algorithm1_name} Obj': f"{comp.algorithm1_mean_obj:.1f} ± {comp.algorithm1_std_obj:.1f}",
            f'{comp.algorithm2_name} Obj': f"{comp.algorithm2_mean_obj:.1f} ± {comp.algorithm2_std_obj:.1f}",
            '% Diff': f"{comp.mean_percent_diff:+.2f} ± {comp.std_percent_diff:.2f}",
            'Wins (G/R/T)': f"{comp.algorithm1_wins}/{comp.algorithm2_wins}/{comp.ties}"
        }
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Calculate sum of objectives across all instances
    alg1_sum_obj = sum(comp.algorithm1_mean_obj * comp.n_instances for comp in summary.comparison_results)
    alg2_sum_obj = sum(comp.algorithm2_mean_obj * comp.n_instances for comp in summary.comparison_results)

    # Add total row
    total_row = {
        'Size': 'TOTAL',
        f'{summary.comparison_results[0].algorithm1_name} Obj': f'{alg1_sum_obj:.1f}',
        f'{summary.comparison_results[0].algorithm2_name} Obj': f'{alg2_sum_obj:.1f}',
        '% Diff': '',
        'Wins (G/R/T)': f"{summary.algorithm1_total_wins}/{summary.algorithm2_total_wins}/{summary.total_ties}"
    }
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    # Print table
    print("\n" + "="*80)
    print(df.to_string(index=False))
    print("="*80)

    # Print legend
    print("\nLegend:")
    print(f"  - Obj: Mean objective value ± standard deviation")
    print(f"  - % Diff: ({summary.comparison_results[0].algorithm1_name} - {summary.comparison_results[0].algorithm2_name}) / {summary.comparison_results[0].algorithm1_name} * 100")
    print(f"    For minimization: (negative = {summary.comparison_results[0].algorithm1_name} better, positive = {summary.comparison_results[0].algorithm2_name} better)")
    print(f"  - Wins: ({summary.comparison_results[0].algorithm1_name} wins / {summary.comparison_results[0].algorithm2_name} wins / Ties)")
    print(f"  - Total instances tested: {summary.total_instances}")
    print("="*80 + "\n")


def plot_comparison_results(summary: ExperimentSummary, save_plot: bool = False) -> None:
    """
    Plot comparison results showing percentage differences and win counts.

    Args:
        summary: ExperimentSummary containing all comparison results
        save_plot: Whether to save the plot to file
    """
    if not summary.comparison_results:
        print("No results to plot")
        return

    # Extract data
    instance_sizes = [comp.instance_size for comp in summary.comparison_results]
    percent_diffs = [comp.mean_percent_diff for comp in summary.comparison_results]
    alg1_wins = [comp.algorithm1_wins for comp in summary.comparison_results]
    alg2_wins = [comp.algorithm2_wins for comp in summary.comparison_results]
    ties = [comp.ties for comp in summary.comparison_results]

    alg1_name = summary.comparison_results[0].algorithm1_name
    alg2_name = summary.comparison_results[0].algorithm2_name

    # Calculate sum of objectives across all instances
    alg1_sum_obj = sum(comp.algorithm1_mean_obj * comp.n_instances for comp in summary.comparison_results)
    alg2_sum_obj = sum(comp.algorithm2_mean_obj * comp.n_instances for comp in summary.comparison_results)

    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Algorithm Comparison: {alg1_name} vs {alg2_name}', fontsize=14, fontweight='bold')

    # Plot 1: Percentage Difference (Bar Plot)
    # Create bar colors based on sign of percentage difference
    # For minimization: negative % diff = alg1 better (blue), positive % diff = alg2 better (green)
    bar_colors = ['green' if diff > 0 else 'blue' for diff in percent_diffs]

    # Calculate bar width for log scale
    x_positions = np.arange(len(instance_sizes))
    ax1.bar(x_positions, percent_diffs, color=bar_colors, alpha=0.7, width=0.6)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Instance Size (n)', fontsize=12)
    ax1.set_ylabel('% Difference', fontsize=12)
    ax1.set_title(f'Percentage Difference\n(obj({alg1_name}) - obj({alg2_name})) / obj({alg1_name}) * 100', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(instance_sizes)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add legend to first subplot
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label=f'{alg1_name} won'),
        Patch(facecolor='green', alpha=0.7, label=f'{alg2_name} won')
    ]
    ax1.legend(handles=legend_elements, loc='best')

    # Plot 2: Win Distribution
    x = np.arange(len(instance_sizes))
    width = 0.25

    ax2.bar(x - width, alg1_wins, width, label=f'{alg1_name} won', color='blue', alpha=0.7)
    ax2.bar(x, alg2_wins, width, label=f'{alg2_name} won', color='green', alpha=0.7)
    ax2.bar(x + width, ties, width, label='Ties', color='gray', alpha=0.7)

    ax2.set_xlabel('Instance Size (n)', fontsize=12)
    ax2.set_ylabel('Number of Wins', fontsize=12)
    ax2.set_title('Win Distribution by Instance Size', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(instance_sizes)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add summary statistics text box
    # Calculate percentage difference between total objectives
    # For minimization: positive % diff means alg1 is worse (higher objective)
    total_obj_percent_diff = (alg1_sum_obj - alg2_sum_obj) / alg1_sum_obj * 100

    if total_obj_percent_diff > 0:
        # alg1 has higher (worse) objective for minimization
        comparison_text = f"{alg2_name}'s total objective is better than {alg1_name}'s by {total_obj_percent_diff:.4f}%"
    elif total_obj_percent_diff < 0:
        # alg1 has lower (better) objective for minimization
        comparison_text = f"{alg1_name}'s total objective is better than {alg2_name}'s by {abs(total_obj_percent_diff):.4f}%"
    else:
        comparison_text = f"{alg1_name}'s and {alg2_name}'s total objectives are equal"

    summary_text = (
        f'Summary Statistics:\n'
        f'Total instances used: {summary.total_instances}\n'
        f'Total Wins: {alg1_name}={summary.algorithm1_total_wins}, '
        f'{alg2_name}={summary.algorithm2_total_wins}, Ties={summary.total_ties}\n'
        f'{comparison_text}'
    )
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=(0, 0.08, 1, 1))  # Leave space for text box at bottom

    if save_plot:
        project_root = find_project_root()
        plots_dir = project_root / "plots"
        plots_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = plots_dir / f"{alg1_name}_vs_{alg2_name}_{'-'.join(instance_sizes)}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")
        plt.close()  # Close figure without showing
    else:
        plt.show()  # Only show interactively when not saving


# ==================== Convenience Functions ====================

def compare_greedy_vs_randomized(
    instance_sizes: list[str],
    instance_type: InstanceType = InstanceType.TEST,
    randomized_top_k: int = 10,
) -> ExperimentSummary:
    """
    Convenience wrapper for comparing Greedy vs Randomized construction heuristics.

    Args:
        instance_sizes: List of instance sizes to test (e.g., ["100", "500", "1000"])
        instance_type: Type of instances (TEST, TRAIN, COMPETITION)
        randomized_top_k: RCL size for randomized construction

    Returns:
        ExperimentSummary with complete comparison results

    Example:
        >>> summary = compare_greedy_vs_randomized(
        ...     instance_sizes=["100", "500"],
        ...     instance_type=InstanceType.TEST,
        ...     randomized_top_k=10
        ... )
    """
    algorithm1_config = AlgorithmConfig(
        name="Greedy",
        algorithm_class=GreedyConstructionHeuristic,
        init_kwargs={}
    )

    algorithm2_config = AlgorithmConfig(
        name=f"Randomized-k{randomized_top_k}",
        algorithm_class=RandomizedConstructionHeuristic,
        init_kwargs={"top_random_pickups_to_consider": randomized_top_k}
    )

    return compare_algorithms_across_sizes(
        instance_sizes=instance_sizes,
        instance_type=instance_type,
        algorithm1_config=algorithm1_config,
        algorithm2_config=algorithm2_config,
    )


# ==================== Metaheuristic Algorithm Wrappers ====================

class BeamSearchWrapper:
    """Wrapper to make Beam Search compatible with comparison framework."""

    def __init__(self, solution: SCFPDPSolution, beam_width: int, branching_factor: int, use_delta_eval: bool = False):
        self.solution = solution
        self.beam_width = beam_width
        self.branching_factor = branching_factor
        self.use_delta_eval = use_delta_eval

    def construct(self):
        """Run beam search and update solution in place."""
        beam_search = SCFPDPBeamSearch(
            self.solution.inst,
            self.beam_width,
            self.branching_factor,
            self.use_delta_eval
        )
        solution, _ = beam_search.solve()
        if solution:
            self.solution.copy_from(solution)


class SAWrapper:
    """Wrapper to make SA compatible with comparison framework."""

    def __init__(self, solution: SCFPDPSolution, sa_settings: dict, use_delta_eval: bool = False):
        self.solution = solution
        self.sa_settings = sa_settings
        self.use_delta_eval = use_delta_eval

    def construct(self):
        """Run SA and update solution in place."""
        sa_solver = SCFPDPSA(
            self.solution.inst,
            RelocateNeighborhood(),
            self.sa_settings,
            self.use_delta_eval
        )
        best_solution = sa_solver.solve()
        self.solution.copy_from(best_solution)


def compare_sa_vs_beam_search(
    instance_sizes: list[str],
    instance_type: InstanceType = InstanceType.COMPETITION,
    sa_settings: dict = None,
    beam_width: int = 3,
    branching_factor: int = 10,
    use_delta_eval: bool = False
) -> ExperimentSummary:
    """
    Compare Simulated Annealing with Beam Search on competition instances.

    Args:
        instance_sizes: List of instance sizes (e.g., ["50", "100", "200"])
        instance_type: Type of instances (default: COMPETITION)
        sa_settings: SA configuration dict (if None, uses defaults)
        beam_width: Beam width for beam search
        branching_factor: Branching factor for beam search
        use_delta_eval: Whether to use delta evaluation

    Returns:
        ExperimentSummary with comparison results
    """
    # Default SA settings if not provided
    if sa_settings is None:
        sa_settings = {
            'mh_titer': 10000,
            'mh_sa_T_init': 50.0,
            'mh_sa_alpha': 0.90,
            'mh_sa_equi_iter': 1000,
            'mh_checkit': True,
            'mh_tciter': -1,
            'mh_ttime': -1,
            'mh_tctime': -1,
            'mh_tobj': -1,
            'mh_lnewinc': False,
            'mh_lfreq': 0,
            'mh_workers': 1
        }

    algorithm1_config = AlgorithmConfig(
        name=f"SA",
        algorithm_class=SAWrapper,
        init_kwargs={"sa_settings": sa_settings, "use_delta_eval": use_delta_eval}
    )

    algorithm2_config = AlgorithmConfig(
        name=f"BS",
        algorithm_class=BeamSearchWrapper,
        init_kwargs={
            "beam_width": beam_width,
            "branching_factor": branching_factor,
            "use_delta_eval": use_delta_eval
        }
    )

    return compare_algorithms_across_sizes(
        instance_sizes=instance_sizes,
        instance_type=instance_type,
        algorithm1_config=algorithm1_config,
        algorithm2_config=algorithm2_config,
    )


if __name__ == "__main__":
    all_instance_sizes = ["50", "100", "200", "500", "1000", "2000", "5000", "10000"]
    instances_to_run = all_instance_sizes[:4]

    # summary = compare_greedy_vs_randomized(
    #     instance_sizes=instances_to_run,
    #     instance_type=InstanceType.TEST,
    #     randomized_top_k=10
    # )

    summary = compare_sa_vs_beam_search(
        instance_sizes=instances_to_run,
        instance_type=InstanceType.TEST,
    )

    # Plot comparison results
    plot_comparison_results(summary, save_plot=True)