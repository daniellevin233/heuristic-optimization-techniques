from src.algorithms.construction_heuristics import (
    GreedyConstructionHeuristic,
    FlexiblePickupAndDropoffConstructionHeuristic,
    ClusterBasedConstructionHeuristic,
)
from src.experiments.algorithm_comparison import (
    AlgorithmConfig,
    compare_algorithms_across_sizes,
    plot_comparison_results,
)
from src.experiments.construction_heuristics import InstanceType


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


if __name__ == "__main__":
    # Instance sizes to compare
    instance_sizes = ["50", "100", "200", "500", "1000"]

    print("\n" + "=" * 80)
    print("COMPARISON 1: Greedy vs FlexiblePickupDropoff")
    print("=" * 80)
    compare_greedy_vs_flexible_pickup_dropoff(instance_sizes)

    print("\n" + "=" * 80)
    print("COMPARISON 2: Greedy vs Clustered")
    print("=" * 80)
    compare_greedy_vs_clustered(instance_sizes)

    print("\n" + "=" * 80)
    print("COMPARISON 3: FlexiblePickupDropoff vs Clustered")
    print("=" * 80)
    compare_clustered_vs_flexible(instance_sizes)