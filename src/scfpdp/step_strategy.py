from abc import ABC, abstractmethod
from typing import Iterable
from pymhlib.solution import Solution


class StepStrategy(ABC):
    """Defines the policy to choose which neighbour becomes the next solution."""

    @abstractmethod
    def choose(self, current: Solution, neighbours: Iterable[Solution]) -> Solution | None:
        """
        Select a neighbour solution, or None if no improving neighbour exists.
        Do NOT modify the solution directly.
        """
        pass
