from src.scfpdp.instance import SCFPDPInstance
from src.scfpdp.solution import SCFPDPSolution, Route
from src.scfpdp.neighborhoods_scfpdp import InsertNeighborhood, SwapNeighborhood, RelocateNeighborhood
from src.scfpdp.local_search import VND, FirstImprovement
import random


def random_initial_solution(instance: SCFPDPInstance) -> SCFPDPSolution:
    sol = SCFPDPSolution(instance)
    all_requests = list(range(instance.n))
    random.shuffle(all_requests)

    for req in all_requests:
        # Try to insert pickup and dropoff randomly into some route
        route_idx = random.randint(0, instance.n_K - 1)
        route = sol.routes[route_idx]
        pickup_pos = random.randint(0, len(route.route))
        dropoff_pos = random.randint(pickup_pos + 1, len(route.route) + 1)
        try:
            route.serve_request(req, pickup_pos, dropoff_pos - pickup_pos)
        except ValueError:
            continue  # skip if infeasible
        route.recompute_route_distance()
    return sol


if __name__ == "__main__":
    instance_file = "10/test_instance_small.txt"
    instance = SCFPDPInstance(instance_file)

    # --- Generate a random initial solution ---
    initial_solution = random_initial_solution(instance)
    print("Random initial solution objective:", initial_solution.calc_objective())

    # --- Run VND with your neighborhoods ---
    neighborhoods = [InsertNeighborhood(), SwapNeighborhood(), RelocateNeighborhood()]
    vnd = VND(neighborhoods, step_strategy=FirstImprovement())

    improved_solution = vnd.run(initial_solution)
    print("Objective after VND:", improved_solution.calc_objective())



# from src.scfpdp.instance import SCFPDPInstance
# from src.scfpdp.solution import SCFPDPSolution
# from src.scfpdp.neighborhoods_scfpdp import InsertNeighborhood, SwapNeighborhood, RelocateNeighborhood
# from src.scfpdp.local_search import LocalSearch
# from src.scfpdp.step_strategies import FirstImprovement
# from src.scfpdp.vnd_improvement_scfpdp import VND, StepStrategyType

# def test_vnd_neighborhoods(instance: SCFPDPInstance):
#     # Construct initial solution (greedy deterministic)
#     initial_solution = SCFPDPSolution(inst=instance)
#     from src.scfpdp.construction_heuristics import GreedyConstructionHeuristic
#     GreedyConstructionHeuristic(initial_solution).construct()
    
#     print(f"Initial solution objective: {initial_solution.calc_objective():.2f}\n")

#     neighborhoods = [InsertNeighborhood(), SwapNeighborhood(), RelocateNeighborhood()]

#     for nh in neighborhoods:
#         print(f"Testing neighborhood: {nh.__class__.__name__}")
#         # single neighborhood in LocalSearch for single-neighborhood VND run
#         ls = LocalSearch(step_strategy=FirstImprovement(), neighborhoods=[nh])
#         current_solution = initial_solution.copy()
#         print(f"Initial objective: {current_solution.calc_objective():.2f}")
#         final_solution = ls.run(current_solution)
#         print(f"Final objective: {final_solution.calc_objective():.2f}\n")


# if __name__ == "__main__":
#     # Instances
#     test_instance = SCFPDPInstance('10/test_instance_small.txt')
#     competition_instance = SCFPDPInstance('100/competition/instance61_nreq100_nveh2_gamma91.txt')

#     instance = competition_instance

#     test_vnd_neighborhoods(instance)




