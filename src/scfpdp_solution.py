import random

from pymhlib.demos.tsp import TSPInstance
from pymhlib.permutation_solution import PermutationSolution
from pymhlib.solution import Solution, TObj

from src.scfpdp_instance import SCFPDPInstance


class TSPSolution(PermutationSolution):
    """Solution to a TSP instance.

    Attributes
        - inst: associated TSPInstance
        - x: order in which cities are visited, i.e., a permutation of 0,...,n-1
    """

    to_maximize = False

    def __init__(self, inst: TSPInstance):
        super().__init__(inst.n, inst=inst)
        self.obj_val_valid = False

    def copy(self):
        sol = TSPSolution(self.inst)
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        distance = 0
        for i in range(self.inst.n - 1):
            distance += self.inst.distances[self.x[i]][self.x[i + 1]]
        distance += self.inst.distances[self.x[-1]][self.x[0]]
        return distance

    def check(self):
        """Check if valid solution.

        :raises ValueError: if problem detected.
        """
        if len(self.x) != self.inst.n:
            raise ValueError("Invalid length of solution")
        super().check()

    def construct(self, par, _result):
        """Scheduler method that constructs a new solution.

        Here we just call initialize.
        """
        self.initialize(par)

    def shaking(self, par, result):
        """Scheduler method that performs shaking by 'par'-times swapping a pair of randomly chosen cities."""
        for _ in range(par):
            a = random.randint(0, self.inst.n - 1)
            b = random.randint(0, self.inst.n - 1)
            self.x[a], self.x[b] = self.x[b], self.x[a]
        self.invalidate()
        result.changed = True

    def local_improve(self, _par, _result):
        """2-opt local search."""
        self.two_opt_neighborhood_search(True)

    def two_opt_move_delta_eval(self, p1: int, p2: int) -> int:
        """ This method performs the delta evaluation for inverting self.x from position p1 to position p2.

        The function returns the difference in the objective function if the move would be performed,
        the solution, however, is not changed.
        """
        assert p1 < p2
        n = len(self.x)
        if p1 == 0 and p2 == n - 1:
            # reversing the whole solution has no effect
            return 0
        prev = (p1 - 1) % n
        nxt = (p2 + 1) % n
        x_p1 = self.x[p1]
        x_p2 = self.x[p2]
        x_prev = self.x[prev]
        x_next = self.x[nxt]
        d = self.inst.distances
        delta = d[x_prev][x_p2] + d[x_p1][x_next] - d[x_prev][x_p1] - d[x_p2][x_next]
        return delta

    def random_move_delta_eval(self) -> Tuple[Any, TObj]:
        """Choose a random move and perform delta evaluation for it, return (move, delta_obj)."""
        return self.random_two_opt_move_delta_eval()

    def apply_neighborhood_move(self, move):
        """This method applies a given neighborhood move accepted by SA,
            without updating the obj_val or invalidating, since obj_val is updated incrementally by the SA scheduler."""
        self.apply_two_opt_move(*move)

    def crossover(self, other: 'TSPSolution') -> 'TSPSolution':
        """Perform edge recombination."""
        return self.edge_recombination(other)


class SCFPDPSolution(Solution):

    to_maximize = False

    def __init__(self, inst: SCFPDPInstance):
        super().__init__()
        self.vehicles_to_routes: dict[int, list[int]] = {vehicle_idx: [] for vehicle_idx in inst.n_K}
        self.inst = inst

    def __repr__(self):
        super().__repr__()

    def copy_from(self, other: 'SCFPDPSolution'):
        super().copy_from(other)

    def copy(self):
        sol = SCFPDPSolution(self.inst)
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        distance = 0
        # for vehicle_idx, route in self.vehicles_to_routes.items():
        #     for i in range(self.inst.n - 1):
        #         distance += self.inst.distance_matrix[self.x[i]][self.x[i + 1]]
        #     distance += self.inst.distances[self.x[-1]][self.x[0]]
        return distance

    def initialize(self, k):
        super().initialize(k)

    def check(self):
        pass


if __name__ == '__main__':
    print(SCFPDPSolution(inst=SCFPDPInstance('10/test_instance_small.txt')))