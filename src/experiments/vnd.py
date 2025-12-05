from src.algorithms.construction_heuristics import RandomizedConstructionHeuristic
from src.instance import SCFPDPInstance
from src.algorithms.local_search import VND, FirstImprovement
from src.neighborhoods import (InsertNeighborhood,
                                      RelocateNeighborhood,
                                      SwapNeighborhood)
from src.solution import SCFPDPSolution


if __name__ == "__main__":
    instance_file = "10/test_instance_small.txt"
    instance = SCFPDPInstance(instance_file)

    # --- Generate a random initial solution ---
    initial_solution = SCFPDPSolution(instance)
    RandomizedConstructionHeuristic(initial_solution).construct()
    print("Random initial solution objective:",
          initial_solution.calc_objective())

    # --- Run VND with your neighborhoods ---
    neighborhoods = [InsertNeighborhood(), SwapNeighborhood(),
                     RelocateNeighborhood()]
    vnd = VND(neighborhoods, step_strategy=FirstImprovement())

    improved_solution = vnd.run(initial_solution)
    print("Objective after VND:", improved_solution.calc_objective())
