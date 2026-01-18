from pathlib import Path

from src.solution import SCFPDPSolution
from src.algorithms.sa.sa import SCFPDPSA
from src.algorithms.sa.config import SAConfig
from src.neighborhoods import RelocateNeighborhood
from src.experiments.algorithm_comparison import (
    AlgorithmConfig,
    compare_algorithms_across_sizes,
    plot_comparison_results, plot_distribution_histograms
)
from src.experiments.alns_parameter_comparison import TunedALNSWrapper
from src.experiments.construction_heuristics import InstanceType
from src.utils import find_project_root


class TunedSAWrapper:
    """SA wrapper that dynamically loads tuned config based on instance size."""

    def __init__(self, solution: SCFPDPSolution, tuning_dir: Path = None, max_time_seconds: float = None):
        self.solution = solution
        instance_size = solution.inst.n

        # Load tuned parameters
        override_params = {}
        if tuning_dir is not None:
            override_params['tuning_dir'] = tuning_dir
        if max_time_seconds is not None:
            override_params['max_time_seconds'] = max_time_seconds

        try:
            self.config = SAConfig.from_tuned_params(instance_size=instance_size, **override_params)
        except FileNotFoundError as e:
            print(f"\n[TunedSAWrapper] Error: {e}")
            raise

    def construct(self):
        """Run SA and update solution in place."""
        sa_solver = SCFPDPSA(
            instance=self.solution.inst,
            neighborhood=RelocateNeighborhood(),
            config=self.config,
            use_delta_eval=True
        )

        best_solution: SCFPDPSolution = sa_solver.solve()
        self.solution.copy_from(best_solution)


def compare_sa_vs_alns(
    instance_sizes: list[str],
    instance_type: InstanceType = InstanceType.TEST,
    timeout: float = 60.0,
    alns_tuning_dir: Path = None,
    sa_tuning_dir: Path = None
):
    """
    Compare tuned SA vs tuned ALNS.

    Args:
        instance_sizes: List of instance sizes to compare (e.g., ["50", "100"])
        instance_type: Type of instances to use (TEST, COMPETITION, etc.)
        timeout: Time limit for each algorithm
        alns_tuning_dir: Directory containing tuned ALNS configs
        sa_tuning_dir: Directory containing tuned SA configs

    Returns:
        Comparison summary
    """
    project_root = find_project_root()

    # Set default tuning directories
    if alns_tuning_dir is None:
        alns_tuning_dir = project_root / "src" / "algorithms" / "alns" / "tuning"
    if sa_tuning_dir is None:
        sa_tuning_dir = project_root / "src" / "algorithms" / "sa" / "tuning"

    # Note: Config loaders will automatically fallback to nearest smaller size if exact match not found

    # Create algorithm configs
    algorithm1_config = AlgorithmConfig(
        name="ALNS-Tuned",
        algorithm_class=TunedALNSWrapper,
        init_kwargs={"tuning_dir": alns_tuning_dir, "max_time_seconds": timeout}
    )

    algorithm2_config = AlgorithmConfig(
        name="SA-Tuned",
        algorithm_class=TunedSAWrapper,
        init_kwargs={"tuning_dir": sa_tuning_dir, "max_time_seconds": timeout}
    )

    # Use the standard comparison framework
    summary = compare_algorithms_across_sizes(
        instance_sizes=instance_sizes,
        instance_type=instance_type,
        algorithm1_config=algorithm1_config,
        algorithm2_config=algorithm2_config,
    )

    return summary


def main():
    """Run SA vs ALNS comparison experiments."""
    print("="*80)
    print("Algorithm Comparison: Tuned ALNS vs Tuned SA")
    print("="*80)

    # Configuration
    instance_sizes = ["50", "100", "200", "500", "1000"]
    instance_type = InstanceType.TEST
    timeout = 60.0  # Match tuning time budget

    print(f"\nConfiguration:")
    print(f"  Instance sizes: {instance_sizes}")
    print(f"  Instance type: {instance_type.name}")
    print(f"  Algorithm timeout: {timeout}s")
    print(f"  Comparing: ALNS-Tuned vs SA-Tuned")
    print()

    # Run comparison
    summary = compare_sa_vs_alns(
        instance_sizes=instance_sizes,
        instance_type=instance_type,
        timeout=timeout
    )

    if summary:
        print("\n" + "="*80)
        print("Comparison completed! Generating plots...")
        print("="*80)
        plot_distribution_histograms(summary, save_plot=True)
        plot_comparison_results(summary, save_plot=True)
        print("\nResults saved to out/ directory")
    else:
        print("\nComparison failed. Check error messages above.")


if __name__ == "__main__":
    main()