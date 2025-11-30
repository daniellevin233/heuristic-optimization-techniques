from abc import ABC, abstractmethod
from src.scfpdp.neighborhoods_scfpdp import Neighborhood
from src.scfpdp.solution import SCFPDPSolution


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