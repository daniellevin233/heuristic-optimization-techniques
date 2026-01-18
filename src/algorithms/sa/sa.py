from pymhlib.sa import SA
from pymhlib.solution import Solution

from src.algorithms.construction_heuristics import RandomizedHybridConstructionHeuristic
from src.algorithms.sa.config import SAConfig
from src.solution import SCFPDPSolution
from src.instance import SCFPDPInstance
from src.neighborhoods import Neighborhood, RelocateNeighborhood


class SCFPDPSA:
    def __init__(
        self,
        instance: SCFPDPInstance,
        neighborhood: Neighborhood,
        config: SAConfig = None,
        use_delta_eval: bool = True
    ):
        self.solution = SCFPDPSolution(instance, use_delta_eval=use_delta_eval)
        self.neighborhood = neighborhood
        self.config = config if config is not None else SAConfig()
        self.use_delta_eval = use_delta_eval
        self.convergence_trajectory = []

        # Convert SAConfig to pymhlib settings
        self.own_settings = {
            'mh_titer': self.config.max_iterations,
            'mh_sa_T_init': self.config.initial_temperature,
            'mh_sa_alpha': self.config.cooling_rate,
            'mh_sa_equi_iter': self.config.equilibrium_iterations,
            'mh_ttime': self.config.max_time_seconds,
            'mh_tciter': self.config.max_iterations_without_improvement,
            'mh_tctime': -1,  # maximum time without improvement (<0: turned off)
            'mh_tobj': -1,  # objective value at which to terminate (<0: turned off)
            'mh_checkit': True,
            'mh_lnewinc': True,
            'mh_lfreq': self.config.log_interval if self.config.log_interval > 0 else 0,
        }

    def solve(self) -> Solution:
        constructor = RandomizedHybridConstructionHeuristic(
            self.solution,
            rcl_size=self.config.rcl_length
        )
        constructor.construct()

        def iteration_callback(iteration: int, sol: Solution, temperature: float, acceptance: bool) -> None:
            """Callback to track convergence every 100 iterations."""
            if iteration % 100 == 0:
                self.convergence_trajectory.append((iteration, sol.obj()))

        sa = SA(
            sol=self.solution,
            meths_ch=[],
            random_move_delta_eval=lambda s: s.random_move_delta_eval(self.neighborhood),
            apply_neighborhood_move=lambda s, m: s.apply_neighborhood_move(m),
            iter_cb=iteration_callback,
            own_settings=self.own_settings,
            consider_initial_sol=True
        )
        sa.run()

        return sa.incumbent


def main():
    """Test SA on a small instance."""
    print("="*80)
    print("Testing Simulated Annealing")
    print("="*80)

    # test_instance = SCFPDPInstance('10/test_instance_small.txt')
    competition_instance = SCFPDPInstance('100/competition/instance61_nreq100_nveh2_gamma91.txt')
    # competition_instance = SCFPDPInstance('1000/competition/instance61_nreq1000_nveh20_gamma879.txt')
    # competition_instance = SCFPDPInstance('2000/competition/instance61_nreq2000_nveh40_gamma1829.txt')

    instance = competition_instance
    print(f"Loaded instance: n={instance.n}, K={instance.n_K}, C={instance.C}, gamma={instance.gamma}")

    # Option 1: Use tuned parameters (automatically loads best config for instance size)
    try:
        config = SAConfig.from_tuned_params(
            instance_size=instance.n,
            max_time_seconds=300.0,  # Override time limit
            log_interval=100
        )
    except FileNotFoundError as e:
        print(f"\n[SA Config] {e}")
        print(f"[SA Config] Using default parameters instead\n")
        # Option 2: Use default/manual parameters
        config = SAConfig(
            max_iterations=100000,
            max_time_seconds=300.0,
            initial_temperature=150.0,
            cooling_rate=0.99,
            equilibrium_iterations=500,
            rcl_length=1,
            log_interval=100
        )

    print(f"\nSA Configuration:")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Max time: {config.max_time_seconds}s")
    print(f"  Initial temperature: {config.initial_temperature}")
    print(f"  Cooling rate: {config.cooling_rate}")
    print(f"  Equilibrium iterations: {config.equilibrium_iterations}")
    print(f"  RCL length: {config.rcl_length}")

    sa_solver = SCFPDPSA(instance, RelocateNeighborhood(), config=config)
    best_solution: SCFPDPSolution = sa_solver.solve()

    print("\n" + "="*80)
    print("Results:")
    print(f"  Final objective: {best_solution.obj():.2f}")
    print(f"  Served requests: {len(best_solution.get_all_served_requests())}/{instance.gamma}")
    print("="*80)

    best_solution.write_to_file("SA")


if __name__ == "__main__":
    main()