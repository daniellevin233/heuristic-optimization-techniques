import time
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from src.instance import SCFPDPInstance
from src.solution import SCFPDPSolution
from src.algorithms.construction_heuristics import FlexiblePickupAndDropoffConstructionHeuristic
from src.algorithms.alns.alns import ALNS
from src.algorithms.alns.config import ALNSConfig
from src.algorithms.alns.operators import (
    RandomRemovalOperator, WorstCostRemovalOperator, LongestRouteRemovalOperator,
    GreedyRepairOperator, RandomGreedyRepairOperator, ObjectiveAwareRepairOperator
)
from src.utils import find_project_root


@dataclass
class FairnessResult:
    """Results for a single instance with a specific fairness measure."""
    fairness_measure: str
    algorithm: str
    instance_name: str

    # Objective components
    objective: float
    total_distance: float
    fairness_value: float

    # Route statistics
    num_stops_per_route: list[int]  # Number of stops (pickups + dropoffs) per route
    mean_stops: float
    std_stops: float
    min_stops: int
    max_stops: int

    # Route distances
    route_distances: list[float]
    mean_distance: float
    std_distance: float
    min_distance: float
    max_distance: float

    # Performance
    execution_time: float
    num_served_requests: int


def compute_fairness_value(solution: SCFPDPSolution) -> float:
    """Extract just the fairness component (not the full objective)."""
    distances = [route.distance for route in solution.routes]
    total_distance = sum(distances)

    if solution.fairness_measure == "jain":
        sum_of_squares = sum(d**2 for d in distances)
        if sum_of_squares == 0:
            return 1.0
        return total_distance**2 / (solution.inst.n_K * sum_of_squares)

    elif solution.fairness_measure == "max_min":
        non_zero_distances = [d for d in distances if d > 0]
        if not non_zero_distances:
            return 1.0
        min_d = min(non_zero_distances)
        max_d = max(distances)
        return min_d / max_d if max_d > 0 else 1.0

    elif solution.fairness_measure == "gini":
        if total_distance == 0:
            return 1.0
        sum_abs_diff = sum(abs(d1 - d2) for d1 in distances for d2 in distances)
        return 1 - sum_abs_diff / (2 * solution.inst.n_K * total_distance)

    return 0.0


def analyze_solution(
    solution: SCFPDPSolution,
    fairness_measure: str,
    algorithm: str,
    instance_name: str,
    execution_time: float
) -> FairnessResult:
    """Extract all metrics from a solution."""

    # Route statistics
    num_stops = [len(route.route) for route in solution.routes]
    route_distances = [route.distance for route in solution.routes]

    return FairnessResult(
        fairness_measure=fairness_measure,
        algorithm=algorithm,
        instance_name=instance_name,
        objective=solution.calc_objective(),
        total_distance=sum(route_distances),
        fairness_value=compute_fairness_value(solution),
        num_stops_per_route=num_stops,
        mean_stops=np.mean(num_stops),
        std_stops=np.std(num_stops),
        min_stops=min(num_stops) if num_stops else 0,
        max_stops=max(num_stops) if num_stops else 0,
        route_distances=route_distances,
        mean_distance=np.mean(route_distances),
        std_distance=np.std(route_distances),
        min_distance=min(route_distances) if route_distances else 0,
        max_distance=max(route_distances) if route_distances else 0,
        execution_time=execution_time,
        num_served_requests=len(solution.get_all_served_requests())
    )


def run_construction_heuristic(
    instance: SCFPDPInstance,
    fairness_measure: str
) -> FairnessResult:
    """Run construction heuristic with specified fairness measure."""
    start_time = time.time()

    solution = SCFPDPSolution(instance, use_delta_eval=True, fairness_measure=fairness_measure)
    constructor = FlexiblePickupAndDropoffConstructionHeuristic(solution)
    constructor.construct()

    execution_time = time.time() - start_time

    return analyze_solution(
        solution,
        fairness_measure,
        "GreedyConstruction",
        instance.file_name,
        execution_time
    )


def run_alns(
    instance: SCFPDPInstance,
    fairness_measure: str,
    timeout_seconds: float = 60.0,
    algorithm_name: str = "ALNS",
    destroy_operator_names: list[str] = None,
    repair_operator_names: list[str] = None
) -> FairnessResult:
    """
    Run ALNS with specified fairness measure and optional custom operators.

    Args:
        instance: Problem instance
        fairness_measure: Fairness measure to use
        timeout_seconds: ALNS timeout
        algorithm_name: Name for this configuration (shown in plots/tables)
        destroy_operator_names: List of destroy operators to use.
            Available: ["Random", "WorstCost", "LongestRoute"]
            If None, uses all three.
        repair_operator_names: List of repair operators to use.
            Available: ["Greedy", "RandomGreedy", "ObjectiveAware"]
            If None, uses all three.
    """
    start_time = time.time()

    # Initial solution
    initial_solution = SCFPDPSolution(instance, use_delta_eval=True, fairness_measure=fairness_measure)
    constructor = FlexiblePickupAndDropoffConstructionHeuristic(initial_solution)
    constructor.construct()

    # ALNS configuration
    config = ALNSConfig(
        max_time_seconds=timeout_seconds,
        max_iterations=100000,
        max_iterations_without_improvement=1000,
        log_interval=1000
    )

    # Build destroy operators
    if destroy_operator_names is None:
        destroy_operator_names = ["Random", "WorstCost", "LongestRoute"]

    destroy_ops = []
    if "Random" in destroy_operator_names:
        destroy_ops.append(RandomRemovalOperator("Random", config))
    if "WorstCost" in destroy_operator_names:
        destroy_ops.append(WorstCostRemovalOperator("WorstCost", config))
    if "LongestRoute" in destroy_operator_names:
        destroy_ops.append(LongestRouteRemovalOperator("LongestRoute", config))

    # Build repair operators
    if repair_operator_names is None:
        repair_operator_names = ["Greedy", "RandomGreedy", "ObjectiveAware"]

    repair_ops = []
    if "Greedy" in repair_operator_names:
        repair_ops.append(GreedyRepairOperator("Greedy", config))
    if "RandomGreedy" in repair_operator_names:
        repair_ops.append(RandomGreedyRepairOperator("RandomGreedy", config))
    if "ObjectiveAware" in repair_operator_names:
        repair_ops.append(ObjectiveAwareRepairOperator("ObjectiveAware", config))

    # Run ALNS
    alns = ALNS(initial_solution, destroy_ops, repair_ops, config)
    best_solution = alns.run()

    execution_time = time.time() - start_time

    return analyze_solution(
        best_solution,
        fairness_measure,
        algorithm_name,
        instance.file_name,
        execution_time
    )


def plot_results(results: list[FairnessResult], output_dir: Path, experiment_name: str, save_plot: bool = False):
    """Generate comparative plots for fairness measures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group results by algorithm
    algorithms = sorted(set(r.algorithm for r in results))
    fairness_measures = ["jain", "max_min", "gini"]
    instance_name = results[0].instance_name.split('/')[-1].replace('.txt', '')

    # Get instance info for metadata
    from src.instance import SCFPDPInstance
    inst = SCFPDPInstance(results[0].instance_name)

    bar_width = 0.25 if len(fairness_measures) > 1 else 0.6

    # Determine operator type for suptitle and filename prefix
    operator_type = ""
    filename_prefix = ""
    if "objective_aware" in experiment_name.lower():
        operator_type = " - Repair Operator Analysis"
        filename_prefix = "ALNS_Repair_"
    elif "longest_route" in experiment_name.lower():
        operator_type = " - Destroy Operator Analysis"
        filename_prefix = "ALNS_Destroy_"
    elif "fairness_comparison" in experiment_name.lower():
        filename_prefix = "Greedy_vs_ALNS_"
    else:
        filename_prefix = ""

    # ===== PLOT 1: Objective Performance (2 subplots) =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Objective Performance (n={inst.n}, K={inst.n_K}, ρ={inst.rho}){operator_type}',
                 fontsize=14, fontweight='bold', y=0.98)

    # Define consistent colors for fairness measures (will be used in Plot 1)
    measure_colors = plt.cm.tab10(np.linspace(0, 0.9, len(fairness_measures)))
    # Also define algo colors for Plot 2
    algo_colors = plt.cm.tab10(np.linspace(0, 0.9, len(algorithms)))

    # Subplot 1: Objective values - grouped by algorithm
    for measure_idx, measure in enumerate(fairness_measures):
        objectives = []
        for algo in algorithms:
            result = next(r for r in results if r.algorithm == algo and r.fairness_measure == measure)
            objectives.append(result.objective)

        x_pos = np.arange(len(algorithms))
        bars = axes[0].bar(x_pos + measure_idx * bar_width, objectives, width=bar_width,
                          label=measure, color=measure_colors[measure_idx])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    axes[0].set_xlabel('Algorithm', fontsize=11)
    axes[0].set_ylabel('Objective Value', fontsize=11)
    axes[0].set_title('Objective Values', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos + bar_width * (len(fairness_measures) - 1) / 2)
    axes[0].set_xticklabels(algorithms)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Subplot 2: Fairness Penalty (absolute values) - grouped by algorithm
    for measure_idx, measure in enumerate(fairness_measures):
        fairness_penalties = []

        for algo in algorithms:
            result = next(r for r in results if r.algorithm == algo and r.fairness_measure == measure)
            fairness_penalty = result.objective - result.total_distance
            fairness_penalties.append(fairness_penalty)

        x_pos = np.arange(len(algorithms))
        bars = axes[1].bar(x_pos + measure_idx * bar_width, fairness_penalties, width=bar_width,
                          label=measure, color=measure_colors[measure_idx])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only label if penalty is positive
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    axes[1].set_xlabel('Algorithm', fontsize=11)
    axes[1].set_ylabel('Fairness Penalty', fontsize=11)
    axes[1].set_title('Fairness Penalty (Absolute)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos + bar_width * (len(fairness_measures) - 1) / 2)
    axes[1].set_xticklabels(algorithms)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    if save_plot:
        filename = f'{filename_prefix}{instance_name}_{experiment_name}_objective.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot: {output_dir / filename}")
    else:
        plt.show()

    # ===== PLOT 2: Fairness Analysis (3 subplots) =====
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f'Fairness Analysis (n={inst.n}, K={inst.n_K}, ρ={inst.rho})',
                 fontsize=14, fontweight='bold', y=0.98)

    # Subplot 1: Fairness index values (bar chart)
    for algo_idx, algo in enumerate(algorithms):
        algo_results = [r for r in results if r.algorithm == algo]
        fairness_values = [next(r.fairness_value for r in algo_results if r.fairness_measure == m)
                          for m in fairness_measures]

        x_pos = np.arange(len(fairness_measures))
        bars = axes[0].bar(x_pos + algo_idx * bar_width, fairness_values, width=bar_width,
                          label=algo, color=algo_colors[algo_idx])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    axes[0].set_xlabel('Fairness Measure', fontsize=11)
    axes[0].set_ylabel('Fairness Index Value', fontsize=11)
    axes[0].set_title('Fairness Index (Higher = More Fair)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos + bar_width/2 if len(algorithms) > 1 else x_pos)
    axes[0].set_xticklabels(fairness_measures)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Subplot 2: Route distance distribution (boxplot)
    boxplot_data = []
    boxplot_positions = []
    pos_counter = 0

    for measure_idx, measure in enumerate(fairness_measures):
        for algo_idx, algo in enumerate(algorithms):
            result = next(r for r in results if r.algorithm == algo and r.fairness_measure == measure)
            boxplot_data.append(result.route_distances)
            boxplot_positions.append(pos_counter)
            pos_counter += 1
        pos_counter += 0.5  # Gap between fairness measures

    bp = axes[1].boxplot(boxplot_data, positions=boxplot_positions, widths=0.6,
                         patch_artist=True, showmeans=True, showfliers=True)

    # Color boxes by algorithm using consistent colors from algo_colors
    for patch_idx, patch in enumerate(bp['boxes']):
        algo_idx = patch_idx % len(algorithms)
        patch.set_facecolor(algo_colors[algo_idx])
        patch.set_alpha(0.7)

    # Set x-axis to show only fairness measures
    measure_positions = []
    for measure_idx in range(len(fairness_measures)):
        measure_center = np.mean([boxplot_positions[measure_idx * len(algorithms) + i]
                                 for i in range(len(algorithms))])
        measure_positions.append(measure_center)

    axes[1].set_xlabel('Fairness Measure', fontsize=11)
    axes[1].set_ylabel('Route Distance', fontsize=11)
    axes[1].set_title('Route Distance Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xticks(measure_positions)
    axes[1].set_xticklabels(fairness_measures)
    axes[1].grid(axis='y', alpha=0.3)

    # Add legend for algorithms
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=algo_colors[i], alpha=0.7, label=algo)
                      for i, algo in enumerate(algorithms)]
    axes[1].legend(handles=legend_elements, loc='upper right')

    # Subplot 3: Route stops distribution (boxplot)
    boxplot_data = []
    boxplot_positions = []
    pos_counter = 0

    for measure_idx, measure in enumerate(fairness_measures):
        for algo_idx, algo in enumerate(algorithms):
            result = next(r for r in results if r.algorithm == algo and r.fairness_measure == measure)
            boxplot_data.append(result.num_stops_per_route)
            boxplot_positions.append(pos_counter)
            pos_counter += 1
        pos_counter += 0.5  # Gap between fairness measures

    bp = axes[2].boxplot(boxplot_data, positions=boxplot_positions, widths=0.6,
                         patch_artist=True, showmeans=True, showfliers=True)

    # Color boxes by algorithm using consistent colors from algo_colors
    for patch_idx, patch in enumerate(bp['boxes']):
        algo_idx = patch_idx % len(algorithms)
        patch.set_facecolor(algo_colors[algo_idx])
        patch.set_alpha(0.7)

    # Set x-axis to show only fairness measures
    measure_positions = []
    for measure_idx in range(len(fairness_measures)):
        measure_center = np.mean([boxplot_positions[measure_idx * len(algorithms) + i]
                                 for i in range(len(algorithms))])
        measure_positions.append(measure_center)

    axes[2].set_xlabel('Fairness Measure', fontsize=11)
    axes[2].set_ylabel('Number of Stops per Route', fontsize=11)
    axes[2].set_title('Route Stops Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xticks(measure_positions)
    axes[2].set_xticklabels(fairness_measures)
    axes[2].grid(axis='y', alpha=0.3)

    # Add legend for algorithms
    axes[2].legend(handles=legend_elements, loc='upper right')

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    if save_plot:
        filename = f'{filename_prefix}{instance_name}_{experiment_name}_fairness.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_dir / filename}")
    else:
        plt.show()

    plt.close('all')


def print_summary_table(results: list[FairnessResult]):
    """Print a summary table of results."""
    print("\n" + "="*120)
    print(f"{'Algorithm':<20} {'Fairness':<10} {'Objective':>12} {'Distance':>12} {'Fair.Val':>10} {'Mean Stops':>12} {'Std Stops':>12} {'Time(s)':>10}")
    print("="*120)

    for result in results:
        print(f"{result.algorithm:<20} {result.fairness_measure:<10} "
              f"{result.objective:>12.2f} {result.total_distance:>12.2f} "
              f"{result.fairness_value:>10.4f} {result.mean_stops:>12.2f} "
              f"{result.std_stops:>12.2f} {result.execution_time:>10.2f}")

    print("="*120)


# ===== Parallel Processing Helpers =====

def _run_single_fairness_config(args):
    """
    Worker function for parallel execution of a single fairness measure + algorithm combination.

    Args:
        args: Tuple of (instance_path, fairness_measure, algorithm_type, algorithm_name,
                       alns_timeout, destroy_ops, repair_ops)

    Returns:
        FairnessResult
    """
    instance_path, fairness_measure, algorithm_type, algorithm_name, alns_timeout, destroy_ops, repair_ops = args

    instance = SCFPDPInstance(instance_path)

    if algorithm_type == "construction":
        return run_construction_heuristic(instance, fairness_measure)
    elif algorithm_type == "alns":
        return run_alns(
            instance, fairness_measure,
            timeout_seconds=alns_timeout,
            algorithm_name=algorithm_name,
            destroy_operator_names=destroy_ops,
            repair_operator_names=repair_ops
        )
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")


def run_fairness_comparison_experiment(
    instance_path: str,
    output_dir: Path,
    alns_timeout: float = 60.0,
    save_plot: bool = True,
):
    """
    Standard experiment: Compare all three fairness measures using Construction and ALNS.
    """
    print("\n" + "="*80)
    print("EXPERIMENT: Fairness Measure Comparison")
    print("="*80)

    fairness_measures = ["jain", "max_min", "gini"]

    # Parallel execution
    tasks = []
    for measure in fairness_measures:
        # Construction heuristic
        tasks.append((instance_path, measure, "construction", "GreedyConstruction", alns_timeout, None, None))
        # ALNS
        tasks.append((instance_path, measure, "alns", "ALNS", alns_timeout, None, None))

    n_workers = min(cpu_count(), len(tasks))
    print(f"Running {len(tasks)} configurations in parallel with {n_workers} workers...")

    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(_run_single_fairness_config, tasks, chunksize=1),
            total=len(tasks),
            desc="Progress"
        ))

    print_summary_table(results)
    plot_results(results, output_dir, experiment_name="fairness_comparison", save_plot=save_plot)


def run_objective_aware_repair_experiment(
    instance_path: str,
    output_dir: Path,
    alns_timeout: float = 60.0,
    save_plot: bool = True
):
    """
    Test impact of ObjectiveAware repair operator.
    Compares three configurations: Full, Without, Only.
    """
    print("\n" + "="*80)
    print("EXPERIMENT: ObjectiveAware Repair Operator")
    print("="*80)

    fairness_measures = ["jain", "max_min", "gini"]

    # Build task list for parallel execution
    tasks = []

    # Config 1: Full ALNS (all operators)
    for measure in fairness_measures:
        tasks.append((instance_path, measure, "alns", "ALNS-Full", alns_timeout, None, None))

    # Config 2: Without ObjectiveAware
    for measure in fairness_measures:
        tasks.append((instance_path, measure, "alns", "ALNS-NoObjAware", alns_timeout,
                     None, ["Greedy", "RandomGreedy"]))

    # Config 3: Only ObjectiveAware
    for measure in fairness_measures:
        tasks.append((instance_path, measure, "alns", "ALNS-OnlyObjAware", alns_timeout,
                     None, ["ObjectiveAware"]))

    # Parallel execution
    n_workers = min(cpu_count(), len(tasks))
    print(f"\nRunning {len(tasks)} configurations in parallel with {n_workers} workers...")

    with Pool(n_workers) as pool:
        results_repair = list(tqdm(
            pool.imap(_run_single_fairness_config, tasks, chunksize=1),
            total=len(tasks),
            desc="Progress"
        ))

    print_summary_table(results_repair)
    plot_results(results_repair, output_dir, experiment_name="objective_aware_repair", save_plot=save_plot)


def run_longest_route_destroy_experiment(
    instance_path: str,
    output_dir: Path,
    alns_timeout: float = 60.0,
    save_plot: bool = True
):
    """
    Test impact of LongestRoute destroy operator.
    Compares three configurations: Full, Without, Only.
    """
    print("\n" + "="*80)
    print("EXPERIMENT: LongestRoute Destroy Operator")
    print("="*80)

    fairness_measures = ["jain", "max_min", "gini"]

    # Build task list for parallel execution
    tasks = []

    # Config 1: Full ALNS (all operators)
    for measure in fairness_measures:
        tasks.append((instance_path, measure, "alns", "ALNS-Full", alns_timeout, None, None))

    # Config 2: Without LongestRoute
    for measure in fairness_measures:
        tasks.append((instance_path, measure, "alns", "ALNS-NoLongest", alns_timeout,
                     ["Random", "WorstCost"], None))

    # Config 3: Only LongestRoute
    for measure in fairness_measures:
        tasks.append((instance_path, measure, "alns", "ALNS-OnlyLongest", alns_timeout,
                     ["LongestRoute"], None))

    # Parallel execution
    n_workers = min(cpu_count(), len(tasks))
    print(f"\nRunning {len(tasks)} configurations in parallel with {n_workers} workers...")

    with Pool(n_workers) as pool:
        results_destroy = list(tqdm(
            pool.imap(_run_single_fairness_config, tasks, chunksize=1),
            total=len(tasks),
            desc="Progress"
        ))

    print_summary_table(results_destroy)
    plot_results(results_destroy, output_dir, experiment_name="longest_route_destroy", save_plot=save_plot)


if __name__ == "__main__":
    # Test instance
    # instance_path = "100/competition/instance61_nreq100_nveh2_gamma91.txt"
    instance_path = "500/competition/instance61_nreq500_nveh10_gamma430.txt"

    # Output directory (project root / plots)
    output_dir = find_project_root() / "plots"

    # ALNS timeout (seconds)
    alns_timeout = 60.0

    # run_fairness_comparison_experiment(
    #     instance_path,
    #     output_dir,
    #     alns_timeout=alns_timeout,
    #     save_plot=True
    # )
    #
    # run_objective_aware_repair_experiment(
    #     instance_path,
    #     output_dir,
    #     alns_timeout=alns_timeout,
    #     save_plot=True
    # )

    run_longest_route_destroy_experiment(
        instance_path,
        output_dir,
        alns_timeout=alns_timeout,
        save_plot=True
    )