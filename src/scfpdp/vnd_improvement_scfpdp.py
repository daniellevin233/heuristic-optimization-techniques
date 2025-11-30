from enum import Enum
from typing import Iterable

from src.scfpdp.solution import SCFPDPSolution
from src.scfpdp.neighborhoods_scfpdp import InsertNeighborhood, SwapNeighborhood, RelocateNeighborhood
from src.scfpdp.step_strategies import FirstImprovement, BestImprovement
from src.scfpdp.local_search import LocalSearch


class StepStrategyType(Enum):
    FIRST = "FIRST"
    BEST = "BEST"


class VND:
    """
    Variable Neighborhood Descent (VND) for SCFPDP.
    Cycles through a list of neighborhoods; restarts from the first neighborhood
    whenever an improvement is found.
    """

    def __init__(self, neighborhoods: Iterable = None, step_strategy: StepStrategyType = StepStrategyType.FIRST):
        # Default fixed neighborhood order
        if neighborhoods is None:
            neighborhoods = [InsertNeighborhood(), SwapNeighborhood(), RelocateNeighborhood()]

        self.neighborhoods = list(neighborhoods)

        if step_strategy == StepStrategyType.FIRST:
            self.step = FirstImprovement()
        elif step_strategy == StepStrategyType.BEST:
            self.step = BestImprovement()
        else:
            raise ValueError(f"Unknown step strategy: {step_strategy}")

        # wrap LocalSearch for reusability
        self.ls = LocalSearch(self.step, self.neighborhoods)

    def run(self, solution: SCFPDPSolution, time_limit: float | None = None) -> SCFPDPSolution:
        """
        Apply VND until no improvement is found across all neighborhoods.
        """
        return self.ls.run(solution, time_limit=time_limit)


# Convenience functions
def first_improvement(solution: SCFPDPSolution) -> SCFPDPSolution:
    vnd = VND(step_strategy=StepStrategyType.FIRST)
    return vnd.run(solution)


def best_improvement(solution: SCFPDPSolution) -> SCFPDPSolution:
    vnd = VND(step_strategy=StepStrategyType.BEST)
    return vnd.run(solution)


def vnd_local_search(solution: SCFPDPSolution) -> SCFPDPSolution:
    # Default VND with first-improvement
    return first_improvement(solution)