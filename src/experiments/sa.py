"""
Simulated Annealing cooling schedule experiments.

This module benchmarks different cooling schedule configurations for SA,
testing the impact of cooling rate (alpha), initial temperature (T_init),
and equilibrium iterations (equi_iter) on solution quality and runtime.
"""

import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.algorithms.sa import SCFPDPSA
from src.scfpdp.instance import SCFPDPInstance
from src.scfpdp.neighbourhoods import RelocateNeighborhood
from src.experiments.construction_heuristics import InstanceType
from src.utils import find_project_root, plot_experiment_results, ExperimentPlotConfig, PlotData


@dataclass
class SARunResult:
    """Results for single SA run with specific configuration."""
    config_name: str
    parameter_name: str
    parameter_value: float
    instance_name: str
    instance_size: int
    final_objective: float
    runtime: float
    convergence_trajectory: list[tuple[int, float]]


@dataclass
class SAConfigResult:
    """Aggregated results for one parameter configuration across all instances."""
    parameter_name: str
    parameter_value: float
    instance_size: int
    n_instances: int
    mean_objective: float
    std_objective: float
    mean_runtime: float
    std_runtime: float
    best_objective: float
    worst_objective: float


@dataclass
class SAExperimentResult:
    """Results for one experiment (varying one parameter)."""
    experiment_name: str
    parameter_name: str
    config_results: list[SAConfigResult]


def run_sa_with_config(
    instance: SCFPDPInstance,
    alpha: float,
    T_init: float,
    equi_iter: int,
    track_convergence: bool = True
) -> tuple[Any, float, list[tuple[int, float]]]:
    """Run SA with given configuration and return solution, runtime, and convergence."""
    settings = {
        'mh_titer': 10000,
        'mh_sa_T_init': T_init,
        'mh_sa_alpha': alpha,
        'mh_sa_equi_iter': int(equi_iter),
        'mh_checkit': True,
        'mh_tciter': -1,
        'mh_ttime': -1,
        'mh_tctime': -1,
        'mh_tobj': -1,
        'mh_lnewinc': False,
        'mh_lfreq': 0,
        'mh_workers': 1
    }

    sa_solver = SCFPDPSA(instance, RelocateNeighborhood(), settings, use_delta_eval=False)

    start_time = time.time()
    solution = sa_solver.solve()
    runtime = time.time() - start_time

    # Extract convergence trajectory if available
    convergence = []
    if track_convergence and hasattr(sa_solver, 'convergence_trajectory'):
        convergence = sa_solver.convergence_trajectory
        if convergence:
            print(f"  DEBUG: Completed {convergence[-1][0]} iterations, final obj={convergence[-1][1]:.2f}, runtime={runtime:.2f}s")

    return solution, runtime, convergence


def load_instances(
    instance_size: str,
    instance_type: InstanceType,
    random_sample_size: int | None = None
) -> list[tuple[SCFPDPInstance, str]]:
    """Load instances from disk, optionally sampling randomly.

    Args:
        instance_size: Size folder (e.g., "50", "100", "200")
        instance_type: Type of instances (TRAIN, TEST, COMPETITION)
        random_sample_size: If specified, randomly sample this many instances

    Returns:
        List of (instance, instance_name) tuples
    """
    project_root = find_project_root()
    instance_dir = project_root / "instances" / instance_size / instance_type.value
    instance_files = list(instance_dir.glob("*.txt"))

    if not instance_files:
        raise ValueError(f"No instances found in {instance_dir}")

    # Sample if requested
    if random_sample_size is not None and random_sample_size < len(instance_files):
        instance_files = random.sample(instance_files, random_sample_size)

    print(f"Size {instance_size}: Loading {len(instance_files)} instances...")

    instances = []
    for instance_file in tqdm(instance_files, desc=f"Loading size {instance_size}"):
        instances.append((SCFPDPInstance(str(instance_file)), instance_file.stem))

    return instances


def process_instance_sa(
    instance: SCFPDPInstance,
    instance_name: str,
    parameter_name: str,
    parameter_value: float,
    alpha: float,
    T_init: float,
    equi_iter: int
) -> SARunResult:
    """Run SA on single instance with given configuration."""
    solution, runtime, convergence = run_sa_with_config(
        instance, alpha, T_init, equi_iter, track_convergence=True
    )

    return SARunResult(
        config_name=f"{parameter_name}={parameter_value}",
        parameter_name=parameter_name,
        parameter_value=parameter_value,
        instance_name=instance_name,
        instance_size=instance.n,
        final_objective=solution.calc_objective(),
        runtime=runtime,
        convergence_trajectory=convergence
    )


def benchmark_sa_config(
    instances: list[tuple[SCFPDPInstance, str]],
    parameter_name: str,
    parameter_value: float,
    alpha: float,
    T_init: float,
    equi_iter: int
) -> SAConfigResult:
    """Benchmark SA configuration on pre-loaded instances.

    Args:
        instances: List of (instance, instance_name) tuples
        parameter_name: Name of parameter being varied
        parameter_value: Value of the parameter
        alpha: Cooling rate
        T_init: Initial temperature
        equi_iter: Equilibrium iterations

    Returns:
        Aggregated results for this configuration
    """
    if not instances:
        raise ValueError("No instances provided")

    instance_size = instances[0][0].n

    print(f"\n{parameter_name}={parameter_value}, Size {instance_size}: "
          f"Running SA on {len(instances)} instances...")

    # Run SA on each instance
    run_results = []
    for instance, instance_name in tqdm(instances, desc=f"{parameter_name}={parameter_value}"):
        result = process_instance_sa(
            instance, instance_name, parameter_name, parameter_value,
            alpha, T_init, equi_iter
        )
        run_results.append(result)

    # Aggregate results
    objectives = np.array([r.final_objective for r in run_results])
    runtimes = np.array([r.runtime for r in run_results])

    return SAConfigResult(
        parameter_name=parameter_name,
        parameter_value=parameter_value,
        instance_size=instance_size,
        n_instances=len(run_results),
        mean_objective=float(np.mean(objectives)),
        std_objective=float(np.std(objectives)),
        mean_runtime=float(np.mean(runtimes)),
        std_runtime=float(np.std(runtimes)),
        best_objective=float(np.min(objectives)),
        worst_objective=float(np.max(objectives))
    )


def run_cooling_experiment(
    experiment_name: str,
    parameter_name: str,
    parameter_values: list[float],
    instances_by_size: dict[str, list[tuple[SCFPDPInstance, str]]],
    default_alpha: float = 0.95,
    default_T_init: float = 100.0,
    default_equi_iter: int = 1000
) -> SAExperimentResult:
    """Run experiment varying one parameter across multiple instance sizes.

    Args:
        experiment_name: Name of the experiment
        parameter_name: Name of parameter being varied ("alpha", "T_init", "equi_iter")
        parameter_values: List of values to test for the parameter
        instances_by_size: Dict mapping size -> list of (instance, name) tuples
        default_alpha: Default cooling rate
        default_T_init: Default initial temperature
        default_equi_iter: Default equilibrium iterations

    Returns:
        Experiment results
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*80}")
    print(f"Parameter: {parameter_name}")
    print(f"Values: {parameter_values}")
    print(f"Instance sizes: {', '.join(instances_by_size.keys())}")
    print(f"{'='*80}\n")

    config_results = []

    for param_value in parameter_values:
        # Set parameters based on what's being varied
        if parameter_name == "alpha":
            alpha, T_init, equi_iter = param_value, default_T_init, default_equi_iter
        elif parameter_name == "T_init":
            alpha, T_init, equi_iter = default_alpha, param_value, default_equi_iter
        elif parameter_name == "equi_iter":
            alpha, T_init, equi_iter = default_alpha, default_T_init, int(param_value)
        else:
            raise ValueError(f"Unknown parameter: {parameter_name}")

        for instance_size, instances in instances_by_size.items():
            config_result = benchmark_sa_config(
                instances, parameter_name, param_value,
                alpha, T_init, equi_iter
            )
            config_results.append(config_result)

    return SAExperimentResult(
        experiment_name=experiment_name,
        parameter_name=parameter_name,
        config_results=config_results
    )


def print_experiment_summary(experiment: SAExperimentResult) -> None:
    """Print summary table of experiment results."""
    print("\n" + "="*80)
    print(f"{experiment.experiment_name.upper()} SUMMARY")
    print("="*80)

    # Create DataFrame for pretty printing
    data = []
    for config in experiment.config_results:
        row = {
            'Size': config.instance_size,
            experiment.parameter_name: config.parameter_value,
            'Mean Obj': f"{config.mean_objective:.2f}",
            'Std Obj': f"{config.std_objective:.2f}",
            'Best': f"{config.best_objective:.2f}",
            'Worst': f"{config.worst_objective:.2f}",
            'Runtime': f"{config.mean_runtime:.2f}s"
        }
        data.append(row)

    df = pd.DataFrame(data)
    print("\n" + df.to_string(index=False))
    print("="*80 + "\n")


def plot_sa_experiment(
    experiment: SAExperimentResult,
    save_plot: bool = True
) -> None:
    """Plot experiment results showing parameter impact."""
    # Group results by instance size
    size_groups = {}
    for config in experiment.config_results:
        size = config.instance_size
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append(config)

    # Extract data for each size
    param_values = sorted(set(c.parameter_value for c in experiment.config_results))

    obj_lines = []
    runtime_lines = []
    labels = []

    for size in sorted(size_groups.keys()):
        configs = sorted(size_groups[size], key=lambda c: c.parameter_value)
        obj_line = [c.mean_objective for c in configs]
        runtime_line = [c.mean_runtime for c in configs]
        obj_lines.append(obj_line)
        runtime_lines.append(runtime_line)
        labels.append(f"n={size}")

    config = ExperimentPlotConfig(
        algorithm_name="SA Cooling Schedule",
        plot_suptitle=f"Impact of {experiment.parameter_name} on SA Performance",
        plot1=PlotData(
            x_values=param_values,
            y_values=obj_lines,
            x_label=experiment.parameter_name,
            y_label='Mean Objective Value',
            title='Solution Quality',
            plot_type='line',
            labels=labels,
            colors=['blue', 'red', 'green']
        ),
        plot2=PlotData(
            x_values=param_values,
            y_values=runtime_lines,
            x_label=experiment.parameter_name,
            y_label='Mean Runtime (seconds)',
            title='Execution Time',
            plot_type='line',
            labels=labels,
            colors=['blue', 'red', 'green']
        ),
        save_plot=save_plot
    )

    plot_experiment_results(config)


def run_all_cooling_experiments(
    instance_sizes: list[str],
    instance_type: InstanceType,
    random_sample_size: int | None,
    save_plots: bool
) -> dict[str, SAExperimentResult]:
    """Run all three cooling schedule experiments.

    Args:
        instance_sizes: List of instance sizes to test (e.g., ["50", "100", "200"])
        instance_type: Type of instances (TRAIN, TEST, COMPETITION)
        random_sample_size: Number of instances to randomly sample per size (None = all)
        save_plots: Whether to save plots to disk

    Returns:
        Dict mapping experiment name to results
    """
    # Load all instances once at the beginning
    print("="*80)
    print("LOADING INSTANCES")
    print("="*80)

    instances_by_size = {}
    for size in instance_sizes:
        instances_by_size[size] = load_instances(size, instance_type, random_sample_size)

    print(f"\nLoaded instances for sizes: {list(instances_by_size.keys())}")
    for size, instances in instances_by_size.items():
        print(f"  Size {size}: {len(instances)} instances")
    print("="*80)

    experiment_results = {}

    # Experiment 1: Cooling Rate (alpha)
    experiment_results['alpha'] = run_cooling_experiment(
        experiment_name="Cooling Rate Impact",
        parameter_name="alpha",
        parameter_values=[0.90, 0.95, 0.99],
        instances_by_size=instances_by_size
    )
    print_experiment_summary(experiment_results['alpha'])
    plot_sa_experiment(experiment_results['alpha'], save_plots)

    # Experiment 2: Initial Temperature
    experiment_results['T_init'] = run_cooling_experiment(
        experiment_name="Initial Temperature Impact",
        parameter_name="T_init",
        parameter_values=[50.0, 100.0, 200.0],
        instances_by_size=instances_by_size
    )
    print_experiment_summary(experiment_results['T_init'])
    plot_sa_experiment(experiment_results['T_init'], save_plots)

    # Experiment 3: Equilibrium Iterations
    experiment_results['equi_iter'] = run_cooling_experiment(
        experiment_name="Equilibrium Iterations Impact",
        parameter_name="equi_iter",
        parameter_values=[500.0, 1000.0, 2000.0],
        instances_by_size=instances_by_size
    )
    print_experiment_summary(experiment_results['equi_iter'])
    plot_sa_experiment(experiment_results['equi_iter'], save_plots)

    return experiment_results


if __name__ == "__main__":
    all_instance_sizes = ["50", "100", "200"]
    results = run_all_cooling_experiments(
        instance_sizes=all_instance_sizes,
        instance_type=InstanceType.TRAIN,
        random_sample_size=3,
        save_plots=True
    )