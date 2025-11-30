import time
from enum import StrEnum

from src.scfpdp.instance import SCFPDPInstance
from src.scfpdp.solution import SCFPDPSolution
from src.algorithms.construction_heuristics import GreedyConstructionHeuristic, RandomizedConstructionHeuristic
from src.utils import find_project_root, plot_experiment_results, ExperimentPlotConfig, PlotData


def plot_construction_results(algorithm_name: str, instances: list['SCFPDPInstance'], objective_values: list[float],
                              runtimes: list[float], parsing_times: list[float], save_plot: bool = False):
    """
    Plot construction heuristic results: objective values and runtimes per instance.
    Convenience wrapper around plot_experiment_results.

    Args:
        algorithm_name: Name of the algorithm for legend
        instances: List of SCFPDPInstance objects
        objective_values: List of objective values achieved
        runtimes: List of runtimes in seconds
        parsing_times: List of instance parsing times in seconds
        save_plot: Whether to save the plot to file
    """
    instance_sizes = [inst.n for inst in instances]

    config = ExperimentPlotConfig(
        algorithm_name=algorithm_name,
        plot_suptitle=algorithm_name,
        plot1=PlotData(
            x_values=instance_sizes,
            y_values=[objective_values],  # Single series wrapped in list
            x_label='Instance Size (n)',
            y_label='Objective Value',
            title=f'Objective Value vs Instance Size',
            plot_type='line',
            x_ticks=instance_sizes,
            x_scale='log',
            labels=['Objective Value']
        ),
        plot2=PlotData(
            x_values=instance_sizes,
            y_values=[runtimes, parsing_times],  # Two series: total runtime and parsing time
            x_label='Instance Size (n)',
            y_label='Runtime (seconds)',
            title=f'Runtime vs Instance Size',
            plot_type='line',
            x_ticks=instance_sizes,
            x_scale='log',
            labels=['Total Runtime', 'Instance Parsing']
        ),
        save_plot=save_plot
    )

    plot_experiment_results(config)


def run_greedy_construction_experiments_for_instance_files(instance_files: list, save_plots: bool) -> tuple:
    instances = []
    objective_values = []
    runtimes = []
    parsing_times = []

    print("Running Greedy Construction Heuristic on train instances...\n")

    for instance_file in instance_files:
        instance_name = instance_file.stem
        print(f"Processing {instance_name}...")

        start_time = time.time()
        instance = SCFPDPInstance(str(instance_file))
        solution = SCFPDPSolution(instance)
        parsing_time = time.time() - start_time

        GreedyConstructionHeuristic(solution).construct()
        runtime = time.time() - start_time

        objective = solution.calc_objective()

        instances.append(instance)
        objective_values.append(objective)
        runtimes.append(runtime)
        parsing_times.append(parsing_time)

        print(f"  n={instance.n}, Objective: {objective:.2f}, Total Runtime: {runtime:.4f}s, Parsing time: {parsing_time:.4f}s\n")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Instances processed: {len(instances)}")
    print(f"Average objective: {sum(objective_values)/len(objective_values):.2f}")
    print(f"Average runtime: {sum(runtimes)/len(runtimes):.4f}s")
    print(f"Average parsing time: {sum(parsing_times)/len(parsing_times):.4f}s")
    print(f"Best objective: {min(objective_values):.2f}")
    print(f"Worst objective: {max(objective_values):.2f}")

    plot_construction_results(
        algorithm_name="Greedy Construction",
        instances=instances,
        objective_values=objective_values,
        runtimes=runtimes,
        parsing_times=parsing_times,
        save_plot=save_plots
    )

    return instances, objective_values, runtimes

class InstanceType(StrEnum):
    COMPETITION = "competition"
    TEST = "test"
    TRAIN = "train"

def run_greedy_construction_experiments(instance_sizes: list[str], instances_type: InstanceType, save_plot: bool = False):
    """Run greedy construction heuristic on all competition instances."""

    project_root = find_project_root()
    instance_files = []
    for instance_size in instance_sizes:
        instances_dir = project_root / "instances" / instance_size / instances_type
        instance_files.extend(sorted(instances_dir.glob("*.txt")))

    run_greedy_construction_experiments_for_instance_files(instance_files, save_plot)


def evaluate_randomized_construction_top_k(instance_size: str, instances_type: InstanceType, top_k_values: list[int], n_runs: int, save_plot: bool):
    """
    Evaluate randomized construction heuristic with different top_k values on a single instance.

    Args:
        instance_size: Size of instance to evaluate (e.g., "1000")
        instances_type: Type of instance (competition/test/train)
        top_k_values: List of top_k values to test
        n_runs: Number of runs per top_k value (for averaging)
        save_plot: Whether to save the plot to file
    """
    project_root = find_project_root()
    instance_dir = project_root / "instances" / instance_size / instances_type
    instance_files = sorted(instance_dir.glob("*.txt"))

    if not instance_files:
        print(f"No instances found in {instance_dir}")
        return

    instance_file = instance_files[0]
    print(f"Evaluating Randomized Construction on {instance_file.name}")
    print(f"Running {n_runs} iterations per top_k value...\n")

    # Reuse the same instance for all top_k values
    instance = SCFPDPInstance(str(instance_file))

    avg_objectives = []
    avg_runtimes = []

    for top_k in top_k_values:
        print(f"Testing top_k={top_k}...")

        objectives = []
        runtimes = []

        for run in range(n_runs):
            # Create fresh solution for each run
            solution = SCFPDPSolution(instance)

            start_time = time.time()
            RandomizedConstructionHeuristic(solution, top_random_pickups_to_consider=top_k).construct()
            runtime = time.time() - start_time

            objective = solution.calc_objective()

            objectives.append(objective)
            runtimes.append(runtime)

        avg_obj = sum(objectives) / len(objectives)
        avg_runtime = sum(runtimes) / len(runtimes)

        avg_objectives.append(avg_obj)
        avg_runtimes.append(avg_runtime)

        print(f"  top_k={top_k}: Avg Objective={avg_obj:.2f}, Avg Runtime={avg_runtime:.4f}s")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Instance: {instance_file.name} (n={instance.n})")
    print(f"Top_k values tested: {top_k_values}")
    print(f"Runs per top_k: {n_runs}")

    # Plot results
    config = ExperimentPlotConfig(
        algorithm_name=f"Randomized Construction - RCL Sensitivity",
        plot_suptitle=f"Randomized Construction - RCL Sensitivity (n={instance.n})",
        plot1=PlotData(
            x_values=top_k_values,
            y_values=[avg_objectives],
            x_label='top_k',
            y_label='Objective Value',
            title=f'Objective Value vs top_k',
            plot_type='line',
            x_ticks=top_k_values,
            labels=[f'Avg Objective Value over {n_runs} runs']
        ),
        plot2=PlotData(
            x_values=top_k_values,
            y_values=[avg_runtimes],
            x_label='top_k',
            y_label='Runtime (seconds)',
            title=f'Runtime vs top_k',
            plot_type='line',
            x_ticks=top_k_values,
            labels=[f'Avg Runtime over {n_runs} runs'],
            colors=['red']
        ),
        save_plot=save_plot
    )

    plot_experiment_results(config)


def run_randomized_construction_experiments_multi_size(instance_sizes: list[str], instance_type: InstanceType, top_k_values: list[int], n_runs: int = 10, save_plot=False):
    """
    Evaluate randomized construction heuristic with different top_k values on multiple instance sizes.
    Creates a separate plot for each instance size.

    Args:
        instance_sizes: List of instance sizes to evaluate (e.g., ["100", "1000", "5000"])
        instance_type: Type of instance (competition/test/train)
        top_k_values: List of top_k values to test
        n_runs: Number of runs per top_k value (for averaging)
        save_plot: Whether to save the plot to file
    """
    for instance_size in instance_sizes:
        print(f"\n{'='*60}")
        print(f"Processing instance size: {instance_size}")
        print('='*60)

        evaluate_randomized_construction_top_k(
            instance_size=instance_size,
            instances_type=instance_type,
            top_k_values=top_k_values,
            n_runs=n_runs,
            save_plot=save_plot
        )


if __name__ == "__main__":
    all_instance_sizes = ["50", "100", "200", "500", "1000", "2000", "5000", "10000"]
    instances_to_run = all_instance_sizes
    # run_greedy_construction_experiments(instances_to_run, InstanceType.COMPETITION, save_plot=False)

    top_k_values = [1, 3, 5, 10, 20, 30, 40, 50, 75, 100]

    # Single instance size
    # run_randomized_construction_experiments(["1000"], InstanceType.COMPETITION, top_k_values, n_runs=10, save_plot=True)

    # Multiple instance sizes
    run_randomized_construction_experiments_multi_size(
        instance_sizes=["100", "500", "2000"],
        instance_type=InstanceType.COMPETITION,
        top_k_values=top_k_values,
        n_runs=10,
        save_plot=True
    )
