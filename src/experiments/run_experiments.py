from src.algorithms.vnd import FirstImprovement, BestImprovement
from src.instance import SCFPDPInstance
from src.solution import SCFPDPSolution
from src.algorithms.construction_heuristics import GreedyConstructionHeuristic, RandomizedConstructionHeuristic
from src.algorithms.local_search import VND, LocalSearch
from src.neighborhoods import InsertNeighborhood, SwapNeighborhood, RelocateNeighborhood
from src.algorithms.grasp import GraspSCFPDP


def run_local_search(instance_file: str):
    instance = SCFPDPInstance(instance_file)
    print(f"Loaded instance: {instance_file} (n={instance.n}, vehicles={instance.n_K})")

    # Initial solution using greedy construction
    initial_solution = SCFPDPSolution(instance)
    GreedyConstructionHeuristic(initial_solution).construct()
    print("Initial solution objective:", initial_solution.calc_objective())

    # Neighborhoods
    neighborhoods = [InsertNeighborhood(), SwapNeighborhood(), RelocateNeighborhood()]

    # Step functions
    step_funcs = [FirstImprovement(), BestImprovement()]

    for step_func in step_funcs:
        print(f"\nRunning Local Search with {step_func.__class__.__name__}")
        ls = LocalSearch(step_strategy=step_func, neighborhoods=neighborhoods)
        solution = ls.run(initial_solution.copy())
        print(f"Final objective: {solution.calc_objective()}\n")


def run_grasp(instance_file: str, alpha=0.3, max_iter=50):
    instance = SCFPDPInstance(instance_file)

    def construct_fn(a=None):
        sol = SCFPDPSolution(instance)
        RandomizedConstructionHeuristic(sol).construct()
        return sol

    neighborhoods = [InsertNeighborhood(), SwapNeighborhood(), RelocateNeighborhood()]
    vnd = VND(neighborhoods, step_strategy=FirstImprovement())

    def local_search_fn(sol):
        return vnd.run(sol)

    grasp = GraspSCFPDP(
        construct=construct_fn,
        local_search=local_search_fn,
        alpha=alpha,
        max_iter=max_iter
    )

    best = grasp.run(verbose=True)
    print("\nGRASP result")
    print("Best objective:", best.calc_objective())
    print(best)


if __name__ == "__main__":
    test_instance_file = '10/test_instance_small.txt'
    competition_instance_file = '100/competition/instance61_nreq100_nveh2_gamma91.txt'
    instance_file = competition_instance_file

    print(f"\n=== Running experiment on instance: {instance_file} ===\n")
    run_local_search(instance_file)
    
    # to run GRASP as well
    # run_grasp(instance_file)