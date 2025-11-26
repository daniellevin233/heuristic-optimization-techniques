from src.scfpdp.instance import SCFPDPInstance
from src.scfpdp.solution import SCFPDPSolution
from src.scfpdp.step_strategies import StepBestImprovement
from src.scfpdp.vnd import VND
from src.scfpdp.local_improvement_scfpdp import VNDImprover
from src.scfpdp.grasp import GraspSCFPDP

if __name__ == "__main__":
    instance = SCFPDPInstance("../instances/10/test_instance_small.txt")
    sol = SCFPDPSolution(instance)

    # Construct using previously implemented heuristic
    sol.initialize(0)

    print("Initial solution:")
    print(sol)

    # ## for VND testing
    vnd = VND()
    vnd.improve(sol)

    print("\nAfter VND:")
    print(sol)


    # step_best = StepBestImprovement()
    # vnd = VND(step_function=step_best)
    # vnd.improve(sol)

    ## for VNDImprover testing
    sol.initialize(k=0)  # deterministic greedy solution

    print("Before VND:")
    print(sol)

    vnd = VNDImprover(step_strategy="best")
    sol = vnd.improve(sol)

    print("\nAfter VND:")
    print(sol)

    ## for GRASP testing
    solver = GraspSCFPDP(max_iter=100, local_search="FIRST")
    # solver = GraspSCFPDP(max_iter=50, max_time=60, local_search="VND")
    best = solver.run(instance)