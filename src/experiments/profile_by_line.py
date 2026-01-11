from src.algorithms.construction_heuristics import (
    FlexiblePickupAndDropoffConstructionHeuristic,
)
from src.experiments.construction_heuristics import run_randomized_construction_experiments_multi_size, InstanceType
from src.instance import SCFPDPInstance
from src.solution import SCFPDPSolution
from src.utils import find_project_root


def profile_instance_loading():
    """Profile instance loading for different sizes."""
    project_root = find_project_root()

    # Test different instance sizes
    instance_sizes = ["2000", "5000", "10000"]

    for size in instance_sizes:
        print(f"\n{'='*60}")
        print(f"Profiling instance size: {size}")
        print('='*60)

        instance_dir = project_root / "instances" / size / "competition"
        instance_files = sorted(instance_dir.glob("*.txt"))

        if instance_files:
            # Profile just the first instance of each size
            instance_file = instance_files[0]
            print(f"Loading: {instance_file.name}")

            # This will be profiled line-by-line
            instance = SCFPDPInstance(str(instance_file))

            print(f"Loaded instance with n={instance.n}")

def profile_randomized_construction():
    run_randomized_construction_experiments_multi_size(
        instance_sizes=["200", "1000"],#, "2000"],
        instance_type=InstanceType.COMPETITION,
        top_k_values=[1,3,5,10,20,30,40,50,75,100],
        n_runs=10,
        save_plot=False
    )

def profile_flexible_construction():
    """Profile the FlexiblePickupAndDropoffConstructionHeuristic for different instance sizes."""
    project_root = find_project_root()

    # Test different instance sizes
    instance_sizes = ["50", "100"]

    for size in instance_sizes:
        print(f"\n{'='*60}")
        print(f"Profiling FlexiblePickupAndDropoff on instance size: {size}")
        print('='*60)

        instance_dir = project_root / "instances" / size / "train"
        instance_files = sorted(instance_dir.glob("*.txt"))

        if instance_files:
            # Profile just the first instance of each size
            instance_file = instance_files[0]
            print(f"Loading: {instance_file.name}")

            instance = SCFPDPInstance(str(instance_file))
            print(f"Loaded instance with n={instance.n}, gamma={instance.gamma}")

            # Create solution and run FlexiblePickupAndDropoff construction
            initial_solution = SCFPDPSolution(inst=instance, use_delta_eval=True)
            heuristic = FlexiblePickupAndDropoffConstructionHeuristic(initial_solution)

            print("Running FlexiblePickupAndDropoff construction...")
            heuristic.construct()

            print(f"Objective: {initial_solution.calc_objective():.2f}")
            print(f"Solution valid: {initial_solution.check() is None}")


if __name__ == "__main__":
    # profile_instance_loading()
    # profile_randomized_construction()
    profile_flexible_construction()