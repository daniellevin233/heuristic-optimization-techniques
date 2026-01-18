from pathlib import Path
import json
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
import numpy as np
from multiprocessing import cpu_count

from src.instance import SCFPDPInstance
from src.solution import SCFPDPSolution
from src.algorithms.sa.sa import SCFPDPSA
from src.algorithms.sa.config import SAConfig
from src.neighborhoods import RelocateNeighborhood
from src.utils import find_project_root


def run_sa_with_config(instance_path: str, config: SAConfig) -> float:
    """Run SA with given configuration and return objective value."""
    instance = SCFPDPInstance(instance_path)

    sa_solver = SCFPDPSA(
        instance=instance,
        neighborhood=RelocateNeighborhood(),
        config=config,
        use_delta_eval=True
    )

    best_solution: SCFPDPSolution = sa_solver.solve()
    return best_solution.obj()


def create_objective_function(instance_paths: list[str], time_budget: float):
    """Create objective function for Optuna to minimize."""

    def objective(trial: optuna.Trial) -> float:
        # Define parameter search space
        config = SAConfig(
            # Simulated annealing parameters
            initial_temperature=trial.suggest_float('initial_temp', 10.0, 500.0),
            cooling_rate=trial.suggest_float('cooling_rate', 0.85, 0.9999),
            equilibrium_iterations=trial.suggest_int('equilibrium_iter', 100, 1000),

            # Construction heuristic parameter
            rcl_length=trial.suggest_int('rcl_length', 1, 20),

            # Fixed parameters
            max_time_seconds=time_budget,
            max_iterations=50000,
            max_iterations_without_improvement=5000,
            log_interval=0  # Suppress logs during tuning
        )

        # Run on all training instances and return average objective
        objectives = []
        for instance_path in instance_paths:
            obj = run_sa_with_config(instance_path, config)
            objectives.append(obj)

        return np.mean(objectives)

    return objective


def tune_parameters(
    instance_paths: list[str],
    time_budget: float,
    n_trials: int,
    n_jobs: int = None,
    output_dir: Path = None,
    instance_size: int = None
) -> dict:
    """
    Tune SA parameters using Optuna.
    Returns:
        Dictionary with best parameters and study statistics
    """
    if n_jobs is None:
        n_jobs = cpu_count()

    if instance_size is None:
        # Extract from first instance path (e.g., "50/train/instance1..." -> 50)
        instance_size = int(instance_paths[0].split('/')[0])

    print(f"\n{'='*80}")
    print(f"Tuning SA parameters for {len(instance_paths)} instances (size={instance_size})")
    print(f"Time budget per run: {time_budget}s")
    print(f"Number of trials: {n_trials}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"{'='*80}\n")

    # Create objective function
    objective = create_objective_function(instance_paths, time_budget)

    # Create Optuna study with TPE sampler
    # Use timeout parameter to handle SQLite locking with parallel jobs
    study_name = f"sa_tuning_n{instance_size}_trials{n_trials}_budget{int(time_budget)}s"
    study = optuna.create_study(
        storage=f"sqlite:///db.sqlite3?timeout=60",  # 60s timeout for database lock waits
        study_name=study_name,
        direction='minimize',
        sampler=TPESampler(seed=42),
        load_if_exists=True  # Resume if study already exists
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    # Extract results
    best_params = study.best_params
    best_value = study.best_value

    print(f"\n{'='*80}")
    print(f"Tuning completed!")
    print(f"Best objective: {best_value:.2f}")
    print(f"\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"{'='*80}\n")

    # Save results if output directory provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        result = {
            "instance_size": instance_size,
            "n_training_instances": len(instance_paths),
            "n_trials": n_trials,
            "time_budget": time_budget,
            "best_objective": best_value,
            "best_params": best_params,
            "timestamp": timestamp
        }

        # Save with timestamp
        output_file_timestamped = output_dir / f"tuned_params_n{instance_size}_{timestamp}.json"
        with open(output_file_timestamped, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {output_file_timestamped}")

        # Also save as "latest" for easy reference by other scripts
        output_file_latest = output_dir / f"tuned_params_n{instance_size}.json"
        with open(output_file_latest, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Latest results symlink: {output_file_latest}")

    return {
        "best_params": best_params,
        "best_value": best_value,
        "study": study,
        "timestamp": timestamp if output_dir is not None else None
    }


def load_training_instances(instance_size: int, n_instances: int = 5) -> list[str]:
    """
    Load training instance paths for tuning from train/ subfolder.

    Returns relative paths like "50/train/instance1_nreq50_nveh2_gamma43.txt"
    """
    project_root = find_project_root()
    instances_dir = project_root / "instances" / str(instance_size) / "train"

    # Find all instance files
    instance_files = sorted(instances_dir.glob("*.txt"))[:n_instances]

    # Convert to relative paths (relative to instances/ folder)
    relative_paths = []
    for file in instance_files:
        relative_path = str(file.relative_to(project_root / "instances"))
        relative_paths.append(relative_path)

    print(f"Loaded {len(relative_paths)} training instances (size={instance_size})")
    return relative_paths


def main():
    """Run parameter tuning experiments."""
    project_root = find_project_root()
    OUTPUT_DIR = project_root / "src" / "algorithms" / "sa" / "tuning"

    # Tuning settings
    INSTANCE_SIZES = [100, 200, 500, 1000]
    N_TRAINING_INSTANCES = 5
    N_TRIALS = 50
    TIME_BUDGET = 60.0

    # Tune for each instance size
    for size in INSTANCE_SIZES:
        print(f"\n{'#'*80}")
        print(f"# Tuning for instance size: {size}")
        print(f"{'#'*80}\n")

        # Load training instances
        instance_paths = load_training_instances(size, N_TRAINING_INSTANCES)

        # Run tuning
        result = tune_parameters(
            instance_paths=instance_paths,
            time_budget=TIME_BUDGET,
            n_trials=N_TRIALS,
            n_jobs=cpu_count(),
            output_dir=OUTPUT_DIR,
            instance_size=size
        )

        print(f"\nCompleted tuning for size {size}")
        print(f"Best objective: {result['best_value']:.2f}")
        print(f"\nTo view visualizations, run: optuna-dashboard sqlite:///db.sqlite3")


if __name__ == "__main__":
    main()