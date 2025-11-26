from src.scfpdp.step_strategy import StepStrategy
from src.scfpdp.solution import SCFPDPSolution

class FirstImprovement(StepStrategy):
    name = "FIRST_IMPROVEMENT"

    def improve(self, solution: SCFPDPSolution, neighborhoods: list):
        """Returns first improving solution found across any neighborhood."""
        for nh in neighborhoods:
            move = nh.next_move(solution)
            if move is not None:
                _, new_sol = move
                return new_sol, True  # improvement found immediately
        return solution, False  # no improvement found


class BestImprovement(StepStrategy):
    name = "BEST_IMPROVEMENT"

    def improve(self, solution: SCFPDPSolution, neighborhoods: list):
        """Evaluates all improving candidates and returns the best one."""
        best_sol = solution
        best_obj = solution.calc_objective()
        improved = False

        # Explore all moves in all neighborhoods
        for nh in neighborhoods:
            while True:
                move = nh.next_move(solution)
                if move is None:
                    break
                _, candidate = move

                # candidate is already feasible by design
                cand_val = candidate.calc_objective()

                if cand_val < best_obj:
                    best_obj = cand_val
                    best_sol = candidate
                    improved = True

                # continue scanning that neighborhood fully
                # DO NOT early return like FirstImprovement

        return best_sol, improved
    

class StepBestImprovement:
    """
    Best-improvement step: scans full neighborhood, applies the globally best move.

    Complete scan required. Uses same move interface:
        apply(solution, move)
        delta(solution, move)
        and solution.evaluate()
    """

    def apply(self, solution, neighborhood) -> bool:
        best_delta = float("inf")
        best_move = None

        # iterate over move candidates
        for move in neighborhood.generate(solution):
            delta = neighborhood.evaluate_move(solution, move)
            if delta < best_delta:
                best_delta = delta
                best_move = move

        # No improving move found
        if best_move is None or best_delta >= 0:
            return False

        # Apply best move
        neighborhood.apply_move(solution, best_move)
        solution.invalidate()  # recalc objective lazily
        return True

