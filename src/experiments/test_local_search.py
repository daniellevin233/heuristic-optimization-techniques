from src.algorithms.construction_heuristics import GreedyConstructionHeuristic
from src.scfpdp.instance import SCFPDPInstance
from src.algorithms.construction_heuristics import GreedyConstructionHeuristic
from src.scfpdp.local_search import LocalSearch
from src.scfpdp.neighborhoods_scfpdp import (InsertNeighborhood,
                                             RelocateNeighborhood,
                                             SwapNeighborhood)
from src.scfpdp.solution import SCFPDPSolution
from src.scfpdp.step_strategies import BestImprovement, FirstImprovement


def test_neighborhood(solution: SCFPDPSolution, neighborhood_class):
    """
    Run local search using a single neighborhood on the provided initial solution.
    """
    nh = neighborhood_class()
    step_strategy = FirstImprovement()
    # step_strategy = BestImprovement()
    ls = LocalSearch(step_strategy, [nh])

    print(f"\nTesting neighborhood: {neighborhood_class.__name__}")
    print(f"Initial objective: {solution.calc_objective():.2f}")

    final_solution = ls.run(solution)
    print(f"Final objective: {final_solution.calc_objective():.2f}")
    return final_solution


if __name__ == "__main__":
    test_instance = SCFPDPInstance('10/test_instance_small.txt')
    competition_instance = SCFPDPInstance(
        '100/competition/instance61_nreq100_nveh2_gamma91.txt')

    # Load instance
    instance = test_instance
    
    # Construct initial solution
    initial_solution = SCFPDPSolution(instance)
    GreedyConstructionHeuristic(initial_solution).construct()
    print(
        f"Initial solution objective: {initial_solution.calc_objective():.2f}")

    # Test each neighborhood
    for neighborhood in [InsertNeighborhood, SwapNeighborhood, RelocateNeighborhood]:
        # Pass a copy of the solution to avoid modifying the same object
        test_neighborhood(initial_solution.copy(), neighborhood)
