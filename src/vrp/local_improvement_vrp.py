from pymhlib.scheduler import Method
from pymhlib.solution import Solution
from .neighbourhood import Neighbourhood
from .step_strategy import StepStrategy


class VRPLocalImprovement(Method):
    """
    Adapter that allows VRP neighbourhood + step strategy to run as a pymhlib Method.
    Used for GVNS local-improvement (meths_li).
    """

    def __init__(self, neighbourhood: Neighbourhood, strategy: StepStrategy):
        super().__init__()
        self.neighbourhood = neighbourhood
        self.strategy = strategy

    def apply(self, sol: Solution):
        """
        Called by pymhlib GVNS.run() through Scheduler.perform_method().
        """
        neighbours = self.neighbourhood.generate(sol)
        chosen = self.strategy.choose(sol, neighbours)

        result = self.Result()

        if chosen and chosen.is_better(sol):
            sol.copy_from(chosen)
            result.changed = True  # pymhlib uses 'changed' to reset VND loops

        return result
