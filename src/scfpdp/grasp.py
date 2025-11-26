import time
from src.scfpdp.solution import SCFPDPSolution
from src.scfpdp.local_improvement_scfpdp import vnd_local_search, first_improvement, best_improvement
from src.construction_heuristics import RandomizedConstructionHeuristic

class GraspSCFPDP:
    def __init__(self, max_iter=50, max_time=None, local_search="VND"):
        """
        local_search ∈ {"FIRST", "BEST", "VND"}
        """
        self.max_iter = max_iter
        self.max_time = max_time  # seconds (optional)
        self.local_search = local_search.upper()

    def _apply_local_search(self, solution):
        if self.local_search == "FIRST":
            return first_improvement(solution)
        elif self.local_search == "BEST":
            return best_improvement(solution)
        elif self.local_search == "VND":
            return vnd_local_search(solution)
        else:
            raise ValueError(f"Unknown local search: {self.local_search}")

    def run(self, instance):
        best_solution = None
        best_obj = float("inf")
        t0 = time.time()

        for _ in range(self.max_iter):
            # --------- 1) RANDOMIZED GREEDY CONSTRUCTION  ---------
            sol = SCFPDPSolution(instance)
            RandomizedConstructionHeuristic().construct(sol)

            # --------- 2) SELECTED LOCAL IMPROVEMENT  ---------
            improved = self._apply_local_search(sol)
            improved.invalidate()

            # --------- 3) KEEP BEST ---------
            obj = improved.calc_objective()
            if obj < best_obj:
                best_obj = obj
                best_solution = improved

            # --------- Stop if time reached ---------
            if self.max_time and time.time() - t0 >= self.max_time:
                break

        return best_solution
