import random
from typing import Callable

from src.scfpdp.local_search import VND
from src.scfpdp.neighborhoods_scfpdp import InsertNeighborhood, RelocateNeighborhood, SwapNeighborhood
from src.scfpdp.solution import SCFPDPSolution
from src.scfpdp.step_strategies import FirstImprovement
from src.algorithms.construction_heuristics import RandomizedConstructionHeuristic


class GraspSCFPDP:
    """
    GRASP for SCFPDP.
    - construct: randomized construction heuristic function(alpha) -> SCFPDPSolution
    - local_search: function(solution) -> SCFPDPSolution (could be VND or single neighborhood LS)
    """

    def __init__(
        self,
        construct: Callable[[float], SCFPDPSolution],
        local_search: Callable[[SCFPDPSolution], SCFPDPSolution],
        alpha: float = 0.25,
        max_iter: int = 50
    ):
        self.construct = construct
        self.local_search = local_search
        self.alpha = alpha
        self.max_iter = max_iter

    def run(self, verbose: bool = False) -> SCFPDPSolution:
        best_solution = None
        best_obj = float("inf")

        for iteration in range(1, self.max_iter + 1):
            # --- Construction Phase ---
            solution = self.construct(self.alpha)

            # --- Local Search Phase ---
            solution = self.local_search(solution)

            obj = solution.calc_objective()
            if obj < best_obj:
                best_solution = solution
                best_obj = obj

            if verbose:
                print(f"[GRASP] Iter {iteration}/{self.max_iter}, current best={best_obj:.2f}")

        return best_solution


# ------------------------
# Using VND as local search
# ------------------------
def make_grasp_with_vnd(instance, alpha=0.25, max_iter=50):
    from src.algorithms.construction_heuristics import RandomizedConstructionHeuristic
    from src.scfpdp.local_search import VND

    def construct_fn(a: float) -> SCFPDPSolution:
        sol = SCFPDPSolution(instance)
        RandomizedConstructionHeuristic(sol).construct()
        return sol

    neighborhoods = [InsertNeighborhood(), SwapNeighborhood(), RelocateNeighborhood()]
    vnd_search = VND(neighborhoods=neighborhoods, step_strategy=FirstImprovement())

    def local_search_fn(solution: SCFPDPSolution) -> SCFPDPSolution:
        return vnd_search.run(solution)

    return GraspSCFPDP(
        construct=construct_fn,
        local_search=local_search_fn,
        alpha=alpha,
        max_iter=max_iter
    )
