from src.scfpdp.instance import SCFPDPInstance
from src.scfpdp.solution import SCFPDPSolution
from src.scfpdp.local_improvement_scfpdp import VNDImprover, vnd_local_search, first_improvement, best_improvement
from src.scfpdp.grasp import GraspSCFPDP

if __name__ == "__main__":
    test_instance = SCFPDPInstance('10/test_instance_small.txt')
    competition_instance = SCFPDPInstance('100/competition/instance61_nreq100_nveh2_gamma91.txt')

    instance = competition_instance

    sol = SCFPDPSolution(instance)
    sol.initialize(0)

    print("Initial solution:")
    print(sol)

    print("\nAfter first improvement VND:")
    sol = first_improvement(sol)
    print(sol)

    sol = SCFPDPSolution(instance)
    sol.initialize(0)
    sol = best_improvement(sol)
    print("\nAfter best improvement VND:")
    print(sol)

    ## for GRASP testing
    solver = GraspSCFPDP(max_iter=100, local_search="FIRST")
    # solver = GraspSCFPDP(max_iter=50, max_time=60, local_search="VND")
    best = solver.run(instance)