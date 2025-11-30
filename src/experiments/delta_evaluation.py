"""
Delta evaluation benchmark for Beam Search algorithm.

This module benchmarks the performance impact of delta evaluation (incremental objective
calculation) on the Beam Search algorithm.
"""
import random
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from src.algorithms.beam_search import SCFPDPBeamSearch
from src.scfpdp.instance import SCFPDPInstance
from src.experiments.construction_heuristics import InstanceType
from src.utils import find_project_root, plot_experiment_results, ExperimentPlotConfig, PlotData


@dataclass
class DeltaEvalRunResult:
    """Results for single run with/without delta eval."""
    branching_factor: int
    with_delta_runtime: float
    without_delta_runtime: float
    with_delta_objective: float
    without_delta_objective: float
    speedup_factor: float


@dataclass
class DeltaEvalBranchingFactorResult:
    """Aggregated results for a given branching factor."""
    branching_factor: int
    n_runs: int
    mean_with_delta_runtime: float
    std_with_delta_runtime: float
    mean_without_delta_runtime: float
    std_without_delta_runtime: float
    mean_speedup_factor: float
    std_speedup_factor: float


@dataclass
class DeltaEvalSizeResult:
    """Aggregated results for a given instance size."""
    instance_size: int
    n_instances: int
    mean_with_delta_runtime: float
    std_with_delta_runtime: float
    mean_without_delta_runtime: float
    std_without_delta_runtime: float
    mean_speedup_factor: float
    std_speedup_factor: float


def run_beam_search_with_delta_config(
    instance: SCFPDPInstance,
    beam_width: int,
    branching_factor: int,
    use_delta_eval: bool
) -> tuple[float, float]:
    """Run beam search and return (runtime, objective)."""
    beam_search = SCFPDPBeamSearch(instance, beam_width, branching_factor, use_delta_eval)

    if use_delta_eval:
        beam_search.solution.invalidate()

    solution, runtime = beam_search.solve()
    return runtime, solution.calc_objective()


def process_single_run_delta_eval(
    instance: SCFPDPInstance,
    beam_width: int,
    branching_factor: int
) -> DeltaEvalRunResult:
    """Run beam search both ways with given branching factor."""
    # Run without delta eval
    runtime_no_delta, obj_no_delta = run_beam_search_with_delta_config(
        instance, beam_width, branching_factor, False
    )

    # Run with delta eval
    runtime_with_delta, obj_with_delta = run_beam_search_with_delta_config(
        instance, beam_width, branching_factor, True
    )

    speedup = runtime_no_delta / runtime_with_delta if runtime_with_delta > 0 else 0

    return DeltaEvalRunResult(
        branching_factor=branching_factor,
        with_delta_runtime=runtime_with_delta,
        without_delta_runtime=runtime_no_delta,
        with_delta_objective=obj_with_delta,
        without_delta_objective=obj_no_delta,
        speedup_factor=speedup
    )


def benchmark_delta_eval_for_branching_factor(
    instance: SCFPDPInstance,
    beam_width: int,
    branching_factor: int,
    n_runs: int
) -> DeltaEvalBranchingFactorResult:
    """Benchmark a single instance with given branching factor multiple times."""
    print(f"\nBranching factor {branching_factor}: Running {n_runs} experiments...")

    results = []
    for _ in tqdm(range(n_runs), desc=f"BF={branching_factor}"):
        result = process_single_run_delta_eval(instance, beam_width, branching_factor)
        results.append(result)

    # Aggregate with numpy
    with_delta_runtimes = np.array([r.with_delta_runtime for r in results])
    without_delta_runtimes = np.array([r.without_delta_runtime for r in results])
    speedups = np.array([r.speedup_factor for r in results])

    return DeltaEvalBranchingFactorResult(
        branching_factor=branching_factor,
        n_runs=len(results),
        mean_with_delta_runtime=float(np.mean(with_delta_runtimes)),
        std_with_delta_runtime=float(np.std(with_delta_runtimes)),
        mean_without_delta_runtime=float(np.mean(without_delta_runtimes)),
        std_without_delta_runtime=float(np.std(without_delta_runtimes)),
        mean_speedup_factor=float(np.mean(speedups)),
        std_speedup_factor=float(np.std(speedups)),
    )


def benchmark_delta_eval_for_size(
    instance_size: str,
    instance_sample_size: int,
    instance_type: InstanceType,
    beam_width: int,
    branching_factor: int
) -> DeltaEvalSizeResult:
    """Benchmark instances of given size."""
    project_root = find_project_root()
    instance_dir = project_root / "instances" / instance_size / instance_type.value
    instance_files = list(instance_dir.glob("*.txt"))

    if not instance_files:
        raise ValueError(f"No instances found in {instance_dir}")

    # Sample instances
    if instance_sample_size < len(instance_files):
        instance_files = random.sample(instance_files, instance_sample_size)

    print(f"\nSize {instance_size}: Loading {len(instance_files)} instances...")

    # Load all instances
    instances = []
    for instance_file in tqdm(instance_files, desc=f"Loading size {instance_size}"):
        instances.append(SCFPDPInstance(str(instance_file)))

    print(f"Size {instance_size}: Running beam search experiments...")

    results = []
    for instance in tqdm(instances, desc=f"Benchmarking size {instance_size}"):
        result = process_single_run_delta_eval(instance, beam_width, branching_factor)
        results.append(result)

    # Aggregate with numpy
    with_delta_runtimes = np.array([r.with_delta_runtime for r in results])
    without_delta_runtimes = np.array([r.without_delta_runtime for r in results])
    speedups = np.array([r.speedup_factor for r in results])

    return DeltaEvalSizeResult(
        instance_size=int(instance_size),
        n_instances=len(results),
        mean_with_delta_runtime=float(np.mean(with_delta_runtimes)),
        std_with_delta_runtime=float(np.std(with_delta_runtimes)),
        mean_without_delta_runtime=float(np.mean(without_delta_runtimes)),
        std_without_delta_runtime=float(np.std(without_delta_runtimes)),
        mean_speedup_factor=float(np.mean(speedups)),
        std_speedup_factor=float(np.std(speedups)),
    )


def print_benchmark_summary(results: list[DeltaEvalSizeResult]) -> None:
    """Print summary of benchmark results."""
    print("\n" + "="*80)
    print("DELTA EVALUATION BENCHMARK SUMMARY")
    print("="*80)

    for r in results:
        print(f"\nSize {r.instance_size} ({r.n_instances} instances):")
        print(f"  With Delta:    {r.mean_with_delta_runtime:.3f}s � {r.std_with_delta_runtime:.3f}s")
        print(f"  Without Delta: {r.mean_without_delta_runtime:.3f}s � {r.std_without_delta_runtime:.3f}s")
        print(f"  Speedup:       {r.mean_speedup_factor:.2f}x � {r.std_speedup_factor:.2f}x")

    overall_speedup = np.mean([r.mean_speedup_factor for r in results])
    print(f"\nOverall Average Speedup: {overall_speedup:.2f}x")
    print("="*80 + "\n")


def plot_delta_evaluation_results(
    bf_results: list[DeltaEvalBranchingFactorResult],
    size_results: list[DeltaEvalSizeResult],
    beam_width: int,
    save_plot: bool = True
) -> None:
    """Plot delta evaluation benchmark results."""
    # Left plot: Branching factor results
    branching_factors = [r.branching_factor for r in bf_results]
    bf_with_delta = [r.mean_with_delta_runtime for r in bf_results]
    bf_without_delta = [r.mean_without_delta_runtime for r in bf_results]

    # Right plot: Instance size results
    instance_sizes = [r.instance_size for r in size_results]
    speedups = [r.mean_speedup_factor for r in size_results]
    avg_speedup = np.mean(speedups)

    config = ExperimentPlotConfig(
        algorithm_name="Delta Evaluation Benchmark",
        plot_suptitle=f"Beam Search Delta Evaluation Impact (β={beam_width})",
        plot1=PlotData(
            x_values=branching_factors,
            y_values=[bf_with_delta, bf_without_delta],
            x_label='Branching Factor',
            y_label='Average Runtime (seconds)',
            title='Runtime per Branching Factor (n=200)',
            plot_type='line',
            colors=['blue', 'red'],
            labels=['With Delta Eval', 'Without Delta Eval']
        ),
        plot2=PlotData(
            x_values=instance_sizes,
            y_values=[speedups, [1.0] * len(instance_sizes)],
            x_label='Instance Size (n)',
            y_label='Speedup Factor',
            title=f'Speedup Factor (Avg: {avg_speedup:.2f}x)',
            plot_type='line',
            x_scale='log',
            x_ticks=instance_sizes,
            colors=['green', 'gray'],
            labels=['Speedup', 'Baseline'],
        ),
        save_plot=save_plot
    )

    plot_experiment_results(config)


def run_delta_evaluation_benchmark(
    branching_factors: list[int],
    instance_sizes: list[str],
    bf_test_size: str,
    bf_n_runs: int,
    size_sample_size: int,
    instance_type: InstanceType,
    beam_width: int,
    size_branching_factor: int,
    save_plot: bool = True
):
    """Main orchestration function."""
    print(f"\n{'='*80}")
    print(f"BEAM SEARCH DELTA EVALUATION BENCHMARK")
    print(f"{'='*80}")
    print(f"Branching factor test:")
    print(f"  Instance size: {bf_test_size}")
    print(f"  Branching factors: {branching_factors}")
    print(f"  Runs per branching factor: {bf_n_runs}")
    print(f"\nInstance size test:")
    print(f"  Instance sizes: {', '.join(instance_sizes)}")
    print(f"  Sample size per size: {size_sample_size}")
    print(f"  Branching factor: {size_branching_factor}")
    print(f"\nCommon parameters:")
    print(f"  Instance type: {instance_type.value}")
    print(f"  Beam width: {beam_width}")
    print(f"{'='*80}\n")

    # Experiment 1: Branching factor comparison on size 200
    print("=" * 80)
    print("EXPERIMENT 1: Branching Factor Impact (n=200)")
    print("=" * 80)
    project_root = find_project_root()
    instance_dir = project_root / "instances" / bf_test_size / instance_type
    instance_files = list(instance_dir.glob("*.txt"))
    if not instance_files:
        raise ValueError(f"No instances found in {instance_dir}")

    # Load one instance for branching factor test
    instance_file = random.choice(instance_files)
    print(f"Loading instance: {instance_file.stem}...")
    bf_instance = SCFPDPInstance(str(instance_file))

    bf_results = []
    for bf in branching_factors:
        bf_result = benchmark_delta_eval_for_branching_factor(bf_instance, beam_width, bf, bf_n_runs)
        bf_results.append(bf_result)

    # Experiment 2: Instance size comparison
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Instance Size Impact")
    print("=" * 80)
    size_results = []
    for instance_size in instance_sizes:
        size_result = benchmark_delta_eval_for_size(
            instance_size, size_sample_size, instance_type, beam_width, size_branching_factor
        )
        size_results.append(size_result)

    print_benchmark_summary(size_results)
    plot_delta_evaluation_results(bf_results, size_results, beam_width, save_plot)


if __name__ == "__main__":
    run_delta_evaluation_benchmark(
        branching_factors=[1, 5, 10, 20, 50],
        instance_sizes=["50", "100", "200", "500", "1000", "2000", "5000", "10000"][:4],
        bf_test_size="1000",
        bf_n_runs=3,
        size_sample_size=1,
        instance_type=InstanceType.TRAIN,
        beam_width=3,
        size_branching_factor=5,
        save_plot=True
    )