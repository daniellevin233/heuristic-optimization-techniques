from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class SAConfig:
    """Configuration for Simulated Annealing algorithm with tunable parameters."""

    # ===== STOPPING CRITERIA =====
    max_iterations: int = 10000
    max_time_seconds: float = 300.0  # 5 minutes
    max_iterations_without_improvement: int = 1000

    # ===== SIMULATED ANNEALING PARAMETERS =====
    initial_temperature: float = 100.0  # T_init
    cooling_rate: float = 0.95  # alpha for geometric cooling
    equilibrium_iterations: int = 500  # iterations until equilibrium at each temperature

    # ===== CONSTRUCTION HEURISTIC PARAMETERS =====
    rcl_length: int = 3  # RCL length for randomized construction

    # ===== LOGGING =====
    log_interval: int = 100  # Print progress every N iterations (0 = disabled)

    @staticmethod
    def from_tuned_params(
        instance_size: int,
        tuning_dir: Path = None,
        **override_params
    ) -> 'SAConfig':
        """
        Load tuned parameters for a given instance size from tuning directory.

        If exact size not found, searches for the largest available size smaller than the requested size.
        """
        if tuning_dir is None:
            from src.utils import find_project_root
            project_root = find_project_root()
            tuning_dir = project_root / "src" / "algorithms" / "sa" / "tuning"

        # Try exact size first
        config_file = tuning_dir / f"tuned_params_n{instance_size}.json"

        if config_file.exists():
            print(f"[SA Config] Loading tuned parameters for n={instance_size}")
            print(f"[SA Config] Config file: {config_file}")
        else:
            # Find all available tuned configs
            available_configs = sorted(tuning_dir.glob("tuned_params_n*.json"))

            # Extract sizes from filenames
            available_sizes = []
            for file in available_configs:
                try:
                    filename = file.stem
                    if "_" in filename and not filename.endswith("_n" + str(instance_size)):
                        # Skip timestamped versions if non-timestamped exists
                        base_file = tuning_dir / (filename.split("_")[0] + "_" + filename.split("_")[1] + ".json")
                        if base_file.exists():
                            continue

                    size_str = filename.split("_n")[1].split("_")[0]
                    size = int(size_str)
                    if size < instance_size:
                        available_sizes.append((size, file))
                except (IndexError, ValueError):
                    continue

            if available_sizes:
                # Use largest size smaller than target
                fallback_size, config_file = max(available_sizes, key=lambda x: x[0])
                print(f"[SA Config] No tuned parameters found for n={instance_size}")
                print(f"[SA Config] Falling back to n={fallback_size} (next smaller available size)")
                print(f"[SA Config] Config file: {config_file}")
            else:
                raise FileNotFoundError(
                    f"No tuned parameters found for n={instance_size} or any smaller size in {tuning_dir}"
                )

        # Load the config file
        with open(config_file, 'r') as f:
            tuning_result = json.load(f)

        best_params = tuning_result["best_params"]

        # Create config with tuned parameters
        config = SAConfig(
            initial_temperature=best_params["initial_temp"],
            cooling_rate=best_params["cooling_rate"],
            equilibrium_iterations=best_params["equilibrium_iter"],
            rcl_length=best_params["rcl_length"],
            **override_params
        )

        print(f"[SA Config] Loaded tuned parameters successfully")
        return config