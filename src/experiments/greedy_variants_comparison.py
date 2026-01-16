from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.algorithms.construction_heuristics import (
    GreedyConstructionHeuristic,
    FlexiblePickupAndDropoffConstructionHeuristic,
    ClusterBasedConstructionHeuristic,
    HybridFlexibleClusteredConstructionHeuristic,
)
from src.experiments.algorithm_comparison import (
    AlgorithmConfig,
    compare_algorithms_across_sizes,
    plot_comparison_results,
    ExperimentSummary,
)
from src.experiments.construction_heuristics import InstanceType
from src.utils import find_project_root


def compare_greedy_vs_flexible_pickup_dropoff(
    instance_sizes: list[str],
    instance_type: InstanceType = InstanceType.TRAIN,
):
    """Compare baseline Greedy vs FlexiblePickupDropoff construction."""
    greedy_config = AlgorithmConfig(
        name="Greedy",
        algorithm_class=GreedyConstructionHeuristic,
        init_kwargs={},
    )
    flexible_config = AlgorithmConfig(
        name="FlexPickupDropoff",
        algorithm_class=FlexiblePickupAndDropoffConstructionHeuristic,
        init_kwargs={},
    )

    summary = compare_algorithms_across_sizes(
        instance_sizes=instance_sizes,
        instance_type=instance_type,
        algorithm1_config=greedy_config,
        algorithm2_config=flexible_config,
    )
    plot_comparison_results(summary, save_plot=True)
    return summary


def compare_greedy_vs_clustered(
    instance_sizes: list[str],
    instance_type: InstanceType = InstanceType.TRAIN,
):
    """Compare baseline Greedy vs ClusterBased construction."""
    greedy_config = AlgorithmConfig(
        name="Greedy",
        algorithm_class=GreedyConstructionHeuristic,
        init_kwargs={},
    )
    clustered_config = AlgorithmConfig(
        name="Clustered",
        algorithm_class=ClusterBasedConstructionHeuristic,
        init_kwargs={},
    )

    summary = compare_algorithms_across_sizes(
        instance_sizes=instance_sizes,
        instance_type=instance_type,
        algorithm1_config=greedy_config,
        algorithm2_config=clustered_config,
    )
    plot_comparison_results(summary, save_plot=True)
    return summary


def compare_clustered_vs_flexible(
    instance_sizes: list[str],
    instance_type: InstanceType = InstanceType.TRAIN,
):
    """Compare  ClusterBased vs FlexiblePickupDropoff construction."""
    clustered_config = AlgorithmConfig(
        name="Clustered",
        algorithm_class=ClusterBasedConstructionHeuristic,
        init_kwargs={},
    )
    flexible_config = AlgorithmConfig(
        name="FlexPickupDropoff",
        algorithm_class=FlexiblePickupAndDropoffConstructionHeuristic,
        init_kwargs={},
    )

    summary = compare_algorithms_across_sizes(
        instance_sizes=instance_sizes,
        instance_type=instance_type,
        algorithm1_config=clustered_config,
        algorithm2_config=flexible_config,
    )
    plot_comparison_results(summary, save_plot=True)
    return summary


def compare_flexible_vs_hybrid(
    instance_sizes: list[str],
    instance_type: InstanceType = InstanceType.TRAIN,
):
    """Compare FlexiblePickupDropoff vs Hybrid (Flexible + Clustered) construction."""
    flexible_config = AlgorithmConfig(
        name="FlexPickupDropoff",
        algorithm_class=FlexiblePickupAndDropoffConstructionHeuristic,
        init_kwargs={},
    )
    hybrid_config = AlgorithmConfig(
        name="Hybrid",
        algorithm_class=HybridFlexibleClusteredConstructionHeuristic,
        init_kwargs={},
    )

    summary = compare_algorithms_across_sizes(
        instance_sizes=instance_sizes,
        instance_type=instance_type,
        algorithm1_config=flexible_config,
        algorithm2_config=hybrid_config,
    )
    plot_comparison_results(summary, save_plot=True)
    return summary


@dataclass
class RuntimeComparisonSummary:
    """Summary of runtime comparison across instance sizes."""
    instance_sizes: list[str]
    algorithm1_name: str  # baseline algorithm
    algorithm2_name: str  # comparison algorithm

    # Per-size mean runtime (focus on central tendency only)
    algorithm1_mean_times: list[float]  # seconds
    algorithm2_mean_times: list[float]  # seconds

    # Relative performance (baseline = algorithm1)
    speedup_ratios: list[float]  # algorithm1_time / algorithm2_time (>1.0 means alg2 faster, <1.0 means alg2 slower)

    # Overall performance
    overall_speedup: float  # geometric mean of speedup ratios


def extract_runtime_comparison(summary: ExperimentSummary) -> RuntimeComparisonSummary:
    """
    Extract runtime statistics from ExperimentSummary.

    Uses algorithm1 as the consistent baseline for speedup calculation.
    Speedup = algorithm1_time / algorithm2_time
    - If >1.0: algorithm2 is faster than baseline
    - If <1.0: algorithm2 is slower than baseline

    Args:
        summary: ExperimentSummary from compare_algorithms_across_sizes()

    Returns:
        RuntimeComparisonSummary with mean times and speedup ratios
    """
    instance_sizes = []
    alg1_mean_times = []
    alg2_mean_times = []
    speedup_ratios = []

    alg1_name = summary.comparison_results[0].algorithm1_name
    alg2_name = summary.comparison_results[0].algorithm2_name

    for comp_result in summary.comparison_results:
        instance_sizes.append(comp_result.instance_size)

        # Extract mean construction times from ComparisonResults
        alg1_time = comp_result.algorithm1_mean_time
        alg2_time = comp_result.algorithm2_mean_time

        alg1_mean_times.append(alg1_time)
        alg2_mean_times.append(alg2_time)

        # Calculate speedup ratio relative to baseline (algorithm1)
        # speedup > 1.0 means algorithm2 is faster
        # speedup < 1.0 means algorithm2 is slower
        speedup = alg1_time / alg2_time
        speedup_ratios.append(speedup)

    # Calculate overall speedup (geometric mean)
    overall_speedup = np.exp(np.mean(np.log(speedup_ratios)))

    return RuntimeComparisonSummary(
        instance_sizes=instance_sizes,
        algorithm1_name=alg1_name,
        algorithm2_name=alg2_name,
        algorithm1_mean_times=alg1_mean_times,
        algorithm2_mean_times=alg2_mean_times,
        speedup_ratios=speedup_ratios,
        overall_speedup=overall_speedup,
    )


def plot_runtime_comparison(
    runtime_summary: RuntimeComparisonSummary,
    save_plot: bool = True,
) -> None:
    """
    Create runtime comparison plot with side-by-side subplots.

    Left subplot: Clean line chart (mean runtimes only, log scale)
    Right subplot: Speedup ratio bar chart

    Args:
        runtime_summary: RuntimeComparisonSummary with timing data
        save_plot: Whether to save plot to plots/ directory
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Convert instance sizes to integers for x-axis
    sizes_int = [int(s) for s in runtime_summary.instance_sizes]

    # Left subplot: Mean runtime trends with log scale
    ax1.plot(
        sizes_int, runtime_summary.algorithm1_mean_times,
        label=runtime_summary.algorithm1_name,
        marker='o', linestyle='-', linewidth=2, markersize=8, color='tab:blue'
    )
    ax1.plot(
        sizes_int, runtime_summary.algorithm2_mean_times,
        label=runtime_summary.algorithm2_name,
        marker='s', linestyle='-', linewidth=2, markersize=8, color='tab:orange'
    )
    ax1.set_xlabel('Instance Size (n)', fontsize=12)
    ax1.set_ylabel('Mean Construction Time (seconds)', fontsize=12)
    ax1.set_title('Runtime Scalability', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right subplot: Speedup ratios with consistent colors
    # Use consistent colors: lightgreen when algorithm2 (Hybrid) faster, lightblue when algorithm1 (Flexible) faster
    colors = ['lightgreen' if ratio >= 1.0 else 'lightblue' for ratio in runtime_summary.speedup_ratios]
    bars = ax2.bar(range(len(sizes_int)), runtime_summary.speedup_ratios, color=colors, alpha=0.8)
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xticks(range(len(sizes_int)))
    ax2.set_xticklabels(runtime_summary.instance_sizes)
    ax2.set_xlabel('Instance Size (n)', fontsize=12)
    ax2.set_ylabel('Speedup Factor', fontsize=12)
    ax2.set_title('Relative Speedup', fontsize=14, fontweight='bold')

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', alpha=0.8, label=f'{runtime_summary.algorithm2_name} faster'),
        Patch(facecolor='lightblue', alpha=0.8, label=f'{runtime_summary.algorithm1_name} faster'),
    ]
    ax2.legend(handles=legend_elements, loc='best', fontsize=10)

    # Add speedup values on top of bars
    for i, (bar, ratio) in enumerate(zip(bars, runtime_summary.speedup_ratios)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}x', ha='center', va='bottom', fontsize=10)

    # Add summary textbox
    if runtime_summary.overall_speedup >= 1.0:
        summary_text = f'Overall: {runtime_summary.algorithm2_name} is {runtime_summary.overall_speedup:.2f}x faster'
    else:
        summary_text = f'Overall: {runtime_summary.algorithm1_name} is {1.0/runtime_summary.overall_speedup:.2f}x faster'

    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
             verticalalignment='bottom')

    plt.tight_layout(rect=(0, 0.05, 1, 1))

    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = find_project_root()
        plots_dir = project_root / "plots"
        plots_dir.mkdir(exist_ok=True)
        filename = plots_dir / f'Runtime_{runtime_summary.algorithm1_name}_vs_{runtime_summary.algorithm2_name}_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filename}")

    plt.show()


def print_runtime_comparison_table(runtime_summary: RuntimeComparisonSummary) -> None:
    """
    Print formatted table of runtime statistics.

    Args:
        runtime_summary: RuntimeComparisonSummary with timing data
    """
    # Use pandas DataFrame for nice formatting
    df = pd.DataFrame({
        'Size': runtime_summary.instance_sizes,
        f'{runtime_summary.algorithm1_name} (s)': [f'{t:.2f}' for t in runtime_summary.algorithm1_mean_times],
        f'{runtime_summary.algorithm2_name} (s)': [f'{t:.2f}' for t in runtime_summary.algorithm2_mean_times],
        'Speedup': [f'{r:.2f}x' for r in runtime_summary.speedup_ratios],
    })

    print("\n" + "="*80)
    print(f"RUNTIME COMPARISON: {runtime_summary.algorithm1_name} (baseline) vs {runtime_summary.algorithm2_name}")
    print("="*80)
    print(df.to_string(index=False))
    print(f"\nOverall Speedup: {runtime_summary.overall_speedup:.2f}x")

    # Interpretation note
    if runtime_summary.overall_speedup >= 1.0:
        print(f"→ {runtime_summary.algorithm2_name} is {runtime_summary.overall_speedup:.2f}x faster overall")
    else:
        print(f"→ {runtime_summary.algorithm2_name} is {1.0/runtime_summary.overall_speedup:.2f}x slower overall")
    print("="*80)


def compare_flexible_vs_hybrid_runtime(
    instance_sizes: list[str],
    instance_type: InstanceType = InstanceType.TRAIN,
) -> RuntimeComparisonSummary:
    """
    Compare construction times of FlexiblePickupDropoff vs Hybrid.

    Returns a summary with:
    - Mean runtimes per size (baseline = FlexiblePickupDropoff)
    - Speedup ratios (FlexPickupDropoff_time / Hybrid_time)

    Args:
        instance_sizes: List of instance sizes to compare (e.g., ["50", "100", "200"])
        instance_type: Type of instances (TRAIN, TEST, or COMPETITION)

    Returns:
        RuntimeComparisonSummary with timing data and speedup analysis
    """
    flexible_config = AlgorithmConfig(
        name="FlexPickupDropoff",
        algorithm_class=FlexiblePickupAndDropoffConstructionHeuristic,
        init_kwargs={},
    )
    hybrid_config = AlgorithmConfig(
        name="Hybrid",
        algorithm_class=HybridFlexibleClusteredConstructionHeuristic,
        init_kwargs={},
    )

    # Reuse existing compare_algorithms_across_sizes() framework
    print("Running runtime comparison benchmark...")
    print("This will take some time depending on instance sizes.")
    summary = compare_algorithms_across_sizes(
        instance_sizes=instance_sizes,
        instance_type=instance_type,
        algorithm1_config=flexible_config,
        algorithm2_config=hybrid_config,
    )

    # Extract runtime data from summary
    runtime_summary = extract_runtime_comparison(summary)

    # Print results
    print_runtime_comparison_table(runtime_summary)

    # Plot results
    plot_runtime_comparison(runtime_summary, save_plot=True)

    return runtime_summary


if __name__ == "__main__":
    # Instance sizes to compare
    instance_sizes = ["50", "100", "200", "500", "1000"]

    # print("\n" + "=" * 80)
    # print("COMPARISON 1: Greedy vs FlexiblePickupDropoff")
    # print("=" * 80)
    # compare_greedy_vs_flexible_pickup_dropoff(instance_sizes)
    #
    # print("\n" + "=" * 80)
    # print("COMPARISON 2: Greedy vs Clustered")
    # print("=" * 80)
    # compare_greedy_vs_clustered(instance_sizes)
    #
    # print("\n" + "=" * 80)
    # print("COMPARISON 3: FlexiblePickupDropoff vs Clustered")
    # print("=" * 80)
    # compare_clustered_vs_flexible(instance_sizes)

    # print("\n" + "=" * 80)
    # print("COMPARISON 4: FlexiblePickupDropoff vs Hybrid")
    # print("=" * 80)
    # compare_flexible_vs_hybrid(instance_sizes)

    print("\n" + "=" * 80)
    print("COMPARISON 5: FlexiblePickupDropoff vs Hybrid (Runtime Benchmark)")
    print("=" * 80)
    compare_flexible_vs_hybrid_runtime(instance_sizes)