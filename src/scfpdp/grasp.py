"""
GRASP for the SCFPDP (Selective Capacitated Fair Pickup and Delivery Problem)

- Construction is randomized greedy (user-provided function)
- Local Search uses injected strategies (e.g., VND)
"""

import random
from typing import Callable

from src.scfpdp.solution import SCFPDPSolution


# ===============================
#         GRASP CLASS
# ===============================

class GraspSCFPDP:

    def __init__(
        self,
        construct: Callable[[float], SCFPDPSolution],
        local_search: Callable[[SCFPDPSolution], SCFPDPSolution],
        alpha: float = 0.25,
        max_iter: int = 50
    ):
        """
        Parameters
        ----------
        construct : function(alpha) -> SCFPDPSolution
            Greedy randomized construction receiving alpha.
        local_search : function(solution) -> SCFPDPSolution
            Example: VND(modes="swap-insert-relocate")
        alpha : float
            0 = pure greedy, 1 = purely random
        max_iter : int
            number of independent GRASP runs
        """
        self.construct = construct
        self.local_search = local_search
        self.alpha = alpha
        self.max_iter = max_iter

    def run(self, verbose: bool = False) -> SCFPDPSolution:
        """
        Executes the GRASP metaheuristic.
        """
        best = None

        for it in range(self.max_iter):
            # --- Construction Phase ---
            sol = self.construct(self.alpha)

            # --- Local Search Phase ---
            sol = self.local_search(sol)

            # --- Update Best ---
            if best is None or sol.cost < best.cost:
                best = sol.copy()

            if verbose:
                print(f"[GRASP] Iter {it+1}/{self.max_iter} best={best.cost:.2f}")

        return best


# ===============================
#  OPTIONAL ALPHA VARIANTS
# ===============================

def alpha_random(min_a=0.05, max_a=0.35):
    """ Dynamic alpha for reactive GRASP """
    return random.uniform(min_a, max_a)


def alpha_constant(a=0.25):
    """ Constant alpha (default greedy/rand mix) """
    return a


# ===============================
#  FACTORY HELPER
# ===============================

def make_grasp(
    construct: Callable[[float], SCFPDPSolution],
    local_search: Callable[[SCFPDPSolution], SCFPDPSolution],
    alpha=0.25,
    iters=50
) -> GraspSCFPDP:
    """ Shortcut for cleaner client code """
    return GraspSCFPDP(
        construct=construct,
        local_search=local_search,
        alpha=alpha,
        max_iter=iters
    )
