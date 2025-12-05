from typing import Callable

from src.solution import SCFPDPSolution

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
