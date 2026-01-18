from pathlib import Path
import json

from src.solution import SCFPDPSolution
from src.algorithms.construction_heuristics import FlexiblePickupAndDropoffConstructionHeuristic
from src.algorithms.alns.alns import ALNS
from src.algorithms.alns.config import ALNSConfig
from src.experiments.algorithm_comparison import (
    AlgorithmConfig,
    compare_algorithms_across_sizes,
    plot_comparison_results
)
from src.experiments.construction_heuristics import InstanceType
from src.utils import find_project_root


class ALNSWrapper:
    """Wrapper to make ALNS compatible with algorithm comparison framework."""

    def __init__(self, solution: SCFPDPSolution, config: ALNSConfig = None):
        self.solution = solution
        self.config = config if config is not None else ALNSConfig()

    def construct(self):
        """Run construction heuristic + ALNS and update solution in place."""
        constructor = FlexiblePickupAndDropoffConstructionHeuristic(self.solution)
        constructor.construct()

        alns = ALNS(self.solution, config=self.config)
        best_solution = alns.run()

        self.solution.copy_from(best_solution)


class TunedALNSWrapper(ALNSWrapper):
    """ALNS wrapper that dynamically loads tuned config based on instance size."""

    def __init__(self, solution: SCFPDPSolution, tuning_dir: Path = None, max_time_seconds: float = None):
        instance_size = solution.inst.n

        # Use the new from_tuned_params method with automatic fallback
        override_params = {}
        if tuning_dir is not None:
            override_params['tuning_dir'] = tuning_dir
        if max_time_seconds is not None:
            override_params['max_time_seconds'] = max_time_seconds

        try:
            config = ALNSConfig.from_tuned_params(instance_size=instance_size, **override_params)
        except FileNotFoundError as e:
            print(f"\n[TunedALNSWrapper] Error: {e}")
            raise

        super().__init__(solution, config)


def compare_default_vs_tuned(
    instance_sizes: list[str],
    instance_type: InstanceType = InstanceType.COMPETITION,
    alns_timeout: float = 60.0,
    tuning_dir: Path = None
):
    # Check if tuned configs exist for all sizes
    if tuning_dir is None:
        project_root = find_project_root()

        # Match the default location in ALNSConfig.from_tuned_params()
        tuning_dir = project_root / "src" / "algorithms" / "alns" / "tuning"

    for size in instance_sizes:
        json_file = tuning_dir / f"tuned_params_n{size}.json"
        if not json_file.exists():
            print(f"\nWARNING: Tuned parameters not found for size {size}")
            print(f"Expected: {json_file}")
            print("Run alns_tuning.py first to generate tuned parameters.")
            return None

    # Create algorithm configs
    default_config = ALNSConfig(max_time_seconds=alns_timeout)

    algorithm1_config = AlgorithmConfig(
        name="ALNS-Default",
        algorithm_class=ALNSWrapper,
        init_kwargs={"config": default_config}
    )

    algorithm2_config = AlgorithmConfig(
        name="ALNS-Tuned",
        algorithm_class=TunedALNSWrapper,
        init_kwargs={"tuning_dir": tuning_dir, "max_time_seconds": alns_timeout}
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
    """Run parameter comparison experiments."""
    print("="*80)
    print("ALNS Parameter Comparison: Default vs Tuned")
    print("="*80)

    # Configuration
    instance_sizes = ["50"]  # Add more as tuning completes: ["50", "100", "200"]
    instance_type = InstanceType.TEST
    alns_timeout = 60.0  # Match tuning time budget

    print(f"\nConfiguration:")
    print(f"  Instance sizes: {instance_sizes}")
    print(f"  Instance type: {instance_type.name}")
    print(f"  ALNS timeout: {alns_timeout}s")
    print(f"  Comparing: ALNS-Default vs ALNS-Tuned")
    print()

    # Run comparison
    summary = compare_default_vs_tuned(
        instance_sizes=instance_sizes,
        instance_type=instance_type,
        alns_timeout=alns_timeout
    )

    if summary:
        print("\n" + "="*80)
        print("Comparison completed! Generating plots...")
        print("="*80)
        # Plot comparison results
        plot_comparison_results(summary, save_plot=True)
        print("\nResults saved to out/ directory")
    else:
        print("\nComparison failed. Check error messages above.")


if __name__ == "__main__":
    main()