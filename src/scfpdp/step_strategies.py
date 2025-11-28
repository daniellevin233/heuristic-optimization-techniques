from abc import ABC, abstractmethod

from pymhlib.solution import Solution

from src.scfpdp.neighbourhoods import Neighborhood
from src.scfpdp.solution import SCFPDPSolution


class StepStrategy(ABC):
    """Interface for defining how a local search step accepts a neighbor."""

    @abstractmethod
    def improve(self, solution: Solution, neighborhood: Neighborhood) -> bool:
        """
        Applies one step of improvement.

        :param solution: current solution (mutated only on accepted moves)
        :param neighborhood: neighborhood object with next_move(...) methods
        :return: True if the solution was improved; False otherwise
        """
        pass


class FirstImprovement(StepStrategy):
    def improve(self, solution: SCFPDPSolution, neighborhood: Neighborhood) -> tuple[SCFPDPSolution, bool]:
        """Returns first improving solution found across any neighborhood."""
        current_obj = solution.calc_objective()
        for neighbor in neighborhood.generate_neighbors(solution):
            if neighbor.calc_objective() < current_obj:
                return neighbor, True
        return solution, False


class BestImprovement(StepStrategy):
    def improve(self, solution: SCFPDPSolution, neighborhood: Neighborhood) -> tuple[SCFPDPSolution, bool]:
        """Evaluates all improving candidates and returns the best one."""
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
