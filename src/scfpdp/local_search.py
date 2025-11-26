import time
from pymhlib.solution import Solution

class LocalSearch:
    def __init__(self, solution, neighborhoods):
        """
        solution      -> SCFPDPSolution object (will be modified)
        neighborhoods -> list of neighborhood functions
                         each fn: move = neighborhood(solution)
                         where move = dict or tuple that contains (delta, move_data)
        """
        self.solution = solution
        self.neighborhoods = neighborhoods

    def run(self, mode="first", time_limit=900):
        """
        mode        -> "first" or "best"
        time_limit  -> seconds
        """
        start = time.time()
        improved = True

        while improved and (time.time() - start) < time_limit:
            improved = self._search_iteration(mode)

        return self.solution

    def _search_iteration(self, mode):
        """
        Perform one iteration over all neighborhoods.
        Returns True if improvement found, else False.
        """
        best_move = None
        best_delta = 0  # improvement should be negative in minimization

        for neighborhood in self.neighborhoods:
            move = neighborhood(self.solution)

            if move is None:
                continue  # no move found in this neighborhood

            delta = move.get("delta", 0)

            # FIRST IMPROVEMENT?
            if mode == "first" and delta < 0:
                self.solution.apply_move(move)
                return True

            # BEST IMPROVEMENT?
            if mode == "best" and delta < best_delta:
                best_delta = delta
                best_move = move

        # if best move was found, apply
        if mode == "best" and best_move is not None:
            self.solution.apply_move(best_move)
            return True

        return False
