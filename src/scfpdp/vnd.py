from src.scfpdp.neighborhoods_scfpdp import InsertNeighborhood, SwapNeighborhood, RelocateNeighborhood
from src.scfpdp.step_strategies import StepFirstImprovement

from src.scfpdp.step_strategy import StepStrategy
from src.scfpdp.neighborhood_order import fixed_neighborhoods


class VND:
    """
    Variable Neighborhood Descent for SCFPDP.

    Runs a sequence of neighborhoods in fixed order.
    When improvement is found in a neighborhood, it restarts from the first one.
    """

    def __init__(self, neighborhoods=None, step_function=None):
        # Default neighborhoods order
        self.neighborhoods = neighborhoods or [
            InsertNeighborhood(),
            RelocateNeighborhood(),
            SwapNeighborhood()
        ]

        # Default step function = first improvement
        self.step_function = step_function or StepFirstImprovement()

    def improve(self, solution):
        """
        Variable Neighborhood Descent:
            1) Start at k = 0 (first neighborhood).
            2) Try to improve using that neighborhood.
            3) If improvement found → restart from the 1st neighborhood.
            4) If not → move to next neighborhood.
            5) Stop when all neighborhoods give no improvement.
        """
        k = 0
        while k < len(self.neighborhoods):
            improved = self.step_function.apply(solution, self.neighborhoods[k])

            if improved: # Restart from first neighborhood (intensification)
                k = 0
            else: # Move to next neighborhood (diversification)
                k += 1

        return solution  # Final locally optimal solution
    

class VariableNeighborhoodDescent(StepStrategy):
    name = "VND"

    def improve(self, solution, neighborhoods=None):
        neighborhoods = fixed_neighborhoods()  # ignore any external list
        current = solution
        improved = True

        while improved:
            improved = False
            for nh in neighborhoods:
                move = nh.next_move(current)
                if move is not None:
                    _, current = move
                    improved = True
                    break  # restart at INSERT

        return current, True

