from enum import Enum
from typing import Iterable

from abc import ABC, abstractmethod
from src.algorithms.local_search import LocalSearch
from src.neighborhoods import InsertNeighborhood, SwapNeighborhood, RelocateNeighborhood, Neighborhood
from src.solution import SCFPDPSolution


class StepStrategyType(Enum):
    FIRST = "FIRST"
    BEST = "BEST"


class StepStrategy(ABC):
    """Interface for defining how a local search step accepts a neighbor."""

    @abstractmethod
    def improve(self, solution: SCFPDPSolution, neighborhood: Neighborhood) -> tuple[SCFPDPSolution, bool]:
        """
        Applies one step of improvement.

        :param solution: current solution (mutated only on accepted moves)
        :param neighborhood: neighborhood object with generate_neighbors() method
        :return: (improved_solution, True/False)
        """
        pass


class FirstImprovement(StepStrategy):
    """Return the first improving neighbor found in the given neighborhood."""

    def improve(self, solution: SCFPDPSolution, neighborhood: Neighborhood) -> tuple[SCFPDPSolution, bool]:
        current_obj = solution.calc_objective()

        # All neighbors yield SCFPDPSolution objects only
        for neighbor in neighborhood.generate_neighbors(solution):
            if neighbor.calc_objective() < current_obj:
                return neighbor, True

        return solution, False


class BestImprovement(StepStrategy):
    """Evaluate all neighbors and return the best improving solution."""

    def improve(self, solution: SCFPDPSolution, neighborhood: Neighborhood) -> tuple[SCFPDPSolution, bool]:
        best_sol = solution
        best_obj = solution.calc_objective()
        improved = False

        for neighbor in neighborhood.generate_neighbors(solution):
            neighbor_obj = neighbor.calc_objective()
            if neighbor_obj < best_obj:
                best_obj = neighbor_obj
                best_sol = neighbor
                improved = True

        return best_sol, improved

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
