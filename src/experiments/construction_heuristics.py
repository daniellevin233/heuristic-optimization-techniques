import time
from enum import StrEnum

from src.scfpdp.instance import SCFPDPInstance
from src.scfpdp.solution import SCFPDPSolution
from src.scfpdp.construction_heuristics import GreedyConstructionHeuristic
from src.utils import find_project_root, plot_experiment_results, ExperimentPlotConfig, PlotData


def plot_construction_results(algorithm_name: str, instances: list['SCFPDPInstance'], objective_values: list[float],
                              runtimes: list[float], save_plot: bool = False):
    """
    Plot construction heuristic results: objective values and runtimes per instance.
    Convenience wrapper around plot_experiment_results.

    Args:
        algorithm_name: Name of the algorithm for legend
        instances: List of SCFPDPInstance objects
        objective_values: List of objective values achieved
        runtimes: List of runtimes in seconds
        save_plot: Whether to save the plot to file
    """
    instance_sizes = [inst.n for inst in instances]

    config = ExperimentPlotConfig(
        algorithm_name=algorithm_name,
        plot_suptitle=algorithm_name,
        plot1=PlotData(
            x_values=instance_sizes,
            y_values=objective_values,
            x_label='Instance Size (n)',
            y_label='Objective Value',
            title=f'Objective Value vs Instance Size',
            color='blue',
            plot_type='line',
            bar_width_relative=0.1,
            x_ticks=instance_sizes,
        ),
        plot2=PlotData(
            x_values=instance_sizes,
            y_values=runtimes,
            x_label='Instance Size (n)',
            y_label='Runtime (seconds)',
            title=f'Runtime vs Instance Size',
            color='red',
            plot_type='line',
            bar_width_relative=0.1,
            x_ticks=instance_sizes,
        ),
        save_plot=save_plot
    )

    plot_experiment_results(config)


def run_greedy_construction_experiments_for_instance_files(instance_files: list, save_plots: bool) -> tuple:
    instances = []
    objective_values = []
    runtimes = []

    print("Running Greedy Construction Heuristic on train instances...\n")

    for instance_file in instance_files:
        instance_name = instance_file.stem
        print(f"Processing {instance_name}...")

        instance = SCFPDPInstance(str(instance_file))
        solution = SCFPDPSolution(instance)

        start_time = time.time()
        GreedyConstructionHeuristic(solution).construct()
        runtime = time.time() - start_time

        objective = solution.calc_objective()

        instances.append(instance)
        objective_values.append(objective)
        runtimes.append(runtime)

        print(f"  n={instance.n}, Objective: {objective:.2f}, Runtime: {runtime:.4f}s\n")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Instances processed: {len(instances)}")
    print(f"Average objective: {sum(objective_values)/len(objective_values):.2f}")
    print(f"Average runtime: {sum(runtimes)/len(runtimes):.4f}s")
    print(f"Best objective: {min(objective_values):.2f}")
    print(f"Worst objective: {max(objective_values):.2f}")

    plot_construction_results(
        algorithm_name="Greedy Construction",
        instances=instances,
        objective_values=objective_values,
        runtimes=runtimes,
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


if __name__ == "__main__":
    all_instance_sizes = ["50", "100", "200", "500", "1000", "2000", "5000", "10000"]
    instances_to_run = all_instance_sizes
    run_greedy_construction_experiments(instances_to_run, InstanceType.COMPETITION, save_plot=True)
