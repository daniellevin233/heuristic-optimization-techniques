from src.scfpdp.solution import SCFPDPSolution
from src.scfpdp.neighbourhoods_scfpdp import (
    InsertNeighborhood,
    SwapNeighborhood,
    RelocateNeighborhood
)
from src.scfpdp.step_strategies import (
    FirstImprovement,
    BestImprovement
)


class VNDImprover:
    """
    Variable Neighborhood Descent (VND) for SCFPDP.
    Uses a fixed deterministic order of neighborhoods:
    INSERT -> SWAP -> RELOCATE
    """

    def __init__(self, step_strategy="first"):
        if step_strategy.lower() == "first":
            self.step_strategy = FirstImprovement()
        elif step_strategy.lower() == "best":
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

        while improved:
            improved = False
            for nh in self.neighborhoods:
                candidate, was_improved = self.step_strategy.improve(solution, [nh])
                if was_improved:
                    solution = candidate
                    improved = True
                    break  # restart neighborhoods

        return solution


def first_improvement(solution: SCFPDPSolution) -> SCFPDPSolution:
    """LS: FIRST IMPROVEMENT."""
    improver = VNDImprover(step_strategy="first")
    return improver.improve(solution)


def best_improvement(solution: SCFPDPSolution) -> SCFPDPSolution:
    """LS: BEST IMPROVEMENT."""
    improver = VNDImprover(step_strategy="best")
    return improver.improve(solution)


def vnd_local_search(solution: SCFPDPSolution) -> SCFPDPSolution:
    """VND (fixed order)."""
    improver = VNDImprover(step_strategy="first")  # could allow best too
    return improver.improve(solution)
