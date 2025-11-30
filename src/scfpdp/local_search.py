import time
from typing import Iterable, Optional

from src.scfpdp.solution import SCFPDPSolution
from src.scfpdp.neighborhoods_scfpdp import Neighborhood
from src.scfpdp.step_strategies import FirstImprovement, BestImprovement


class LocalSearch:
    """
    Generic LS core:
    - step_strategy.improve(solution, neighborhood) → (candidate, improved)
    - deterministic neighborhood order
    - restart after each improvement (VND-like)
    """

    def __init__(self, step_strategy, neighborhoods: Iterable[Neighborhood]):
        self.step_strategy = step_strategy
        self.neighborhoods = list(neighborhoods)

    def search_once(self, solution: SCFPDPSolution) -> tuple[SCFPDPSolution, bool]:
        """
        Perform one VND-style sweep over all neighborhoods.
        """
        current = solution
        for nh in self.neighborhoods:
            candidate, improved = self.step_strategy.improve(current, nh)
            if improved:
                return candidate, True
        return current, False

    def run(self, solution: SCFPDPSolution, time_limit: Optional[float] = None) -> SCFPDPSolution:
        """
        Repeated descent until no neighborhood improves the solution
        or time limit expires.
        """
        start = time.time()
        current = solution

        while True:
            if time_limit and (time.time() - start) >= time_limit:
                return current

            new_sol, improved = self.search_once(current)
            if not improved:
                return current
            current = new_sol


class VND:
    """
    Simple VND wrapper using LocalSearch.
    """

    def __init__(self, neighborhoods, step_strategy=None):
        if step_strategy is None:
            step_strategy = FirstImprovement()
        self.ls = LocalSearch(step_strategy, neighborhoods)

    def run(self, solution: SCFPDPSolution, time_limit: Optional[float] = None) -> SCFPDPSolution:
        return self.ls.run(solution, time_limit=time_limit)