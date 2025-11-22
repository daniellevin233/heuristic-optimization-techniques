from abc import ABC, abstractmethod
from typing import Iterable
from pymhlib.solution import Solution


class Neighbourhood(ABC):
    """Abstract neighbourhood interface for generating nearby candidate solutions."""

    @abstractmethod
    def generate(self, solution: Solution) -> Iterable[Solution]:
        """
        Yield neighbouring solutions for a given VRP solution.
        Should NOT evaluate improvements or pick best moves.
        """
        pass
