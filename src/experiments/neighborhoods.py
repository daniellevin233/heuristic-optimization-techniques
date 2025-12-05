import random
from src.instance import SCFPDPInstance
from src.solution import SCFPDPSolution
from src.neighborhoods import InsertNeighborhood, SwapNeighborhood, RelocateNeighborhood
from src.algorithms.local_search import LocalSearch, FirstImprovement


def random_initial_solution(instance: SCFPDPInstance) -> SCFPDPSolution:
    """Generate a feasible but random solution."""
    sol = SCFPDPSolution(instance)
    all_requests = list(range(instance.n))
    random.shuffle(all_requests)

    for req in all_requests:
        route_idx = random.randint(0, instance.n_K - 1)
        route = sol.routes[route_idx]
        pickup_pos = random.randint(0, len(route.route))
        dropoff_pos = random.randint(pickup_pos + 1, len(route.route) + 1)
        try:
            route.serve_request(req, pickup_pos, dropoff_pos - pickup_pos)
        except ValueError:
            continue
        route.recompute_route_distance()
    return sol


def test_neighborhood(solution: SCFPDPSolution, neighborhood_class):
    nh = neighborhood_class()
    print(f"\nTesting neighborhood: {nh.__class__.__name__}")
    print("Initial objective:", solution.calc_objective())

    # Run first-improvement local search with a single neighborhood
    ls = LocalSearch(step_strategy=FirstImprovement(), neighborhoods=[nh])
    final_solution = ls.run(solution)

    print("Final objective:", final_solution.calc_objective())
    return final_solution


if __name__ == "__main__":
    # test_instance = SCFPDPInstance('10/test_instance_small.txt')
    test_instance = SCFPDPInstance('50/test/instance31_nreq50_nveh2_gamma50.txt')
    competition_instance = SCFPDPInstance('100/competition/instance61_nreq100_nveh2_gamma91.txt')
    
    # Load instance
    instance = test_instance

    # Generate random initial solution
    initial_solution = random_initial_solution(instance)
    print("Random initial solution objective:", initial_solution.calc_objective())

    # Test each neighborhood independently
    for neighborhood_class in [InsertNeighborhood, SwapNeighborhood, RelocateNeighborhood]:
        test_neighborhood(initial_solution.copy(), neighborhood_class)