from typing import Iterable
from pymhlib.solution import Solution
from .step_strategy import StepStrategy


class FirstImprovement(StepStrategy):
    """Select the first neighbour that strictly improves the current solution."""

    def choose(self, current: Solution, neighbours: Iterable[Solution]) -> Solution | None:
        # TODO: stop at first strictly better neighbour
        ...


class BestImprovement(StepStrategy):
    """Select the globally best improving neighbour among all generated moves."""

    def choose(self, current: Solution, neighbours: Iterable[Solution]) -> Solution | None:
        # TODO: scan all, return best improvement
        ...
