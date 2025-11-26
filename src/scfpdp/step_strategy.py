from abc import ABC, abstractmethod
from pymhlib.solution import Solution


class StepStrategy(ABC):
    """Interface for defining how a local search step accepts a neighbor."""

    @abstractmethod
    def improve(self, solution: Solution, neighborhoods: list) -> bool:
        """
        Applies one step of improvement.

        :param solution: current solution (mutated only on accepted moves)
        :param neighborhoods: list of neighborhood objects with generate(...) methods
        :return: True if the solution was improved; False otherwise
        """
        pass
