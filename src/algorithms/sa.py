from typing import Any

from pymhlib.sa import SA
from pymhlib.scheduler import Method, Result
from pymhlib.solution import Solution

from src.algorithms.beam_search import SCFPDPBeamSearch
from scfpdp.construction_heuristics import RandomizedConstructionHeuristic
from src.scfpdp.solution import SCFPDPSolution
from src.scfpdp.instance import SCFPDPInstance
from scfpdp.neighbourhoods_scfpdp import SwapNeighborhood, Neighborhood, InsertNeighborhood, RelocateNeighborhood


class SCFPDPSA:
    def __init__(self, instance: SCFPDPInstance, neighborhood: Neighborhood, own_settings: dict = None):
        self.solution = SCFPDPSolution(instance)
        self.neighborhood = neighborhood
        self.own_settings = own_settings or {}

    def solve(self) -> Solution:

        def greedy_constructor(solution: SCFPDPSolution, par: Any, result: Result) -> None:
            solution.invalidate()
            RandomizedConstructionHeuristic(solution).construct()
            result.terminate = True

        def random_constructor(solution: SCFPDPSolution, par: Any, result: Result) -> None:
            solution.invalidate()
            RandomizedConstructionHeuristic(solution).construct()
            result.terminate = True

        def beam_constructor(solution: SCFPDPSolution, par: Any, result: Result) -> None:
            solution.invalidate()
            beam_search = SCFPDPBeamSearch(solution.inst, 10, 4)
            beam_solution = beam_search.solve()
            solution.copy_from(beam_solution[0])
            result.terminate = True

        sa = SA(
            sol=self.solution,
            meths_ch=[Method("beam_construct", beam_constructor, 0),
                      Method("randomized_construct", random_constructor, 0),
                      Method("greedy_construct", greedy_constructor, 0)],
            random_move_delta_eval=lambda s: s.random_move_delta_eval(self.neighborhood),
            apply_neighborhood_move=lambda s, m: s.apply_neighborhood_move(m),
            iter_cb=None,
            own_settings=self.own_settings,
            consider_initial_sol=False
        )
        sa.run()
        return sa.incumbent


def main():
    test_instance = SCFPDPInstance('10/test_instance_small.txt')
    competition_instance = SCFPDPInstance('100/competition/instance61_nreq100_nveh2_gamma91.txt')

    instance = competition_instance

    settings = {
        'mh_titer': 10000,  # maximum number of iterations
        'mh_sa_T_init': 100.0,  # SA initial temperature
        'mh_sa_alpha': 0.95,  # SA alpha for geometric cooling
        'mh_sa_equi_iter': 1000,  # SA iterations until equilibrium
        'mh_checkit': True,  # call check() for each solution after each method application
        'mh_tciter': -1,  # maximum number of iterations without improvement (<0: turned off)
        'mh_ttime': -1,  # time limit in seconds (<0: turned off)
        'mh_tctime': -1,  # maximum time in seconds without improvement (<0: turned off)
        'mh_tobj': -1,  # objective value at which to terminate when reached (<0: turned off)
        'mh_lnewinc': True,  # write iteration log if new incumbent solution found
        'mh_lfreq': 0,  # frequency of writing iteration logs (0: none, >0: number of iterations, -1: iteration 1,2,5,10,20,...)
        'mh_workers': 4,  # number of worker processes when using multiprocessing
    }

    sa_solver = SCFPDPSA(instance, RelocateNeighborhood(), settings)
    best_solution = sa_solver.solve()
    print(best_solution)
    print(f"Final objective: {best_solution.obj():.2f}")


if __name__ == "__main__":
    main()