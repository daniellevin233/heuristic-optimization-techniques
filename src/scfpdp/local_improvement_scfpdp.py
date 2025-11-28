from enum import Enum

from src.scfpdp.solution import SCFPDPSolution
from scfpdp.neighbourhoods_scfpdp import (
    InsertNeighborhood,
    SwapNeighborhood,
    RelocateNeighborhood
)
from src.scfpdp.step_strategies import (
    FirstImprovement,
    BestImprovement
)


class StepStrategy(Enum):
    FIRST = "FIRST"
    BEST = "BEST"


class VNDImprover:
    """
    Variable Neighborhood Descent (VND) for SCFPDP.
    Uses a fixed deterministic order of neighborhoods:
    INSERT -> SWAP -> RELOCATE
    """

    def __init__(self, step_strategy: StepStrategy):
        if step_strategy == StepStrategy.FIRST:
            self.step_strategy = FirstImprovement()
        elif step_strategy == StepStrategy.BEST:
            self.step_strategy = BestImprovement()
        else:
            raise ValueError(f"Unknown step strategy: {step_strategy}")

        # fixed order for VND
        self.neighborhoods = [
            InsertNeighborhood(),
            SwapNeighborhood(),
            RelocateNeighborhood()
        ]

    def improve(self, solution: SCFPDPSolution) -> SCFPDPSolution:
        """
        Apply VND until no improvement in a full cycle.
        """
        improved = True

        best_solution = solution
        while improved:
            improved = False
            for nh in self.neighborhoods:
                candidate, was_improved = self.step_strategy.improve(best_solution, nh)
                if was_improved:
                    best_solution = candidate
                    improved = True
                    break  # restart neighborhoods

        return best_solution


def first_improvement(solution: SCFPDPSolution) -> SCFPDPSolution:
    """LS: FIRST IMPROVEMENT."""
    improver = VNDImprover(step_strategy=StepStrategy.FIRST)
    return improver.improve(solution)


def best_improvement(solution: SCFPDPSolution) -> SCFPDPSolution:
    """LS: BEST IMPROVEMENT."""
    improver = VNDImprover(step_strategy=StepStrategy.BEST)
    return improver.improve(solution)


def vnd_local_search(solution: SCFPDPSolution) -> SCFPDPSolution:
    """VND (fixed order)."""

    return first_improvement(solution)  # could allow best too