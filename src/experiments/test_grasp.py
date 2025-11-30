# src/experiments/test_grasp.py
from pathlib import Path

from src.scfpdp.instance import SCFPDPInstance
from src.scfpdp.solution import SCFPDPSolution
from src.algorithms.construction_heuristics import RandomizedConstructionHeuristic
from src.scfpdp.neighborhoods_scfpdp import (
    InsertNeighborhood,
    SwapNeighborhood,
    RelocateNeighborhood,
)
from src.scfpdp.local_search import VND
from src.scfpdp.step_strategies import FirstImprovement
from src.scfpdp.grasp import GraspSCFPDP


# -----------------------------
# GRASP TEST
# -----------------------------
def test_grasp(instance: SCFPDPInstance, alpha: float = 0.3, max_iter: int = 50):

    print(f"Instance loaded: n={instance.n}, vehicles={instance.n_K}")

    # -------------------------
    # Construction wrapper
    # -------------------------
    def construct(alpha_val: float) -> SCFPDPSolution:
        # Create empty solution
        sol = SCFPDPSolution(instance)

        # map alpha → top_k
        top_k = max(1, int(round(alpha_val * instance.n)))
        top_k = min(instance.n, top_k)

        ctor = RandomizedConstructionHeuristic(
            sol,
            top_random_pickups_to_consider=top_k
        )
        ctor.construct()     # uses no alpha parameter

        return sol

    # -------------------------
    # VND local search
    # -------------------------
    neighborhoods = [
        InsertNeighborhood(),
        SwapNeighborhood(),
        RelocateNeighborhood(),
    ]

    vnd = VND(
        neighborhoods=neighborhoods,
        step_strategy=FirstImprovement()
    )

    def local_search(sol: SCFPDPSolution) -> SCFPDPSolution:
        return vnd.run(sol)

    # -------------------------
    # GRASP engine
    # -------------------------
    grasp = GraspSCFPDP(
        construct=construct,
        local_search=local_search,
        alpha=alpha,
        max_iter=max_iter,
    )

    print("\nRunning GRASP...")
    best = grasp.run(verbose=True)

    print("\n--- GRASP RESULT ---")
    print("Best objective:", best.calc_objective())
    print(best)

    return best


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    instance_file = "50/competition/instance61_nreq50_nveh2_gamma44.txt"

    # Load the actual instance
    instance = SCFPDPInstance(instance_file)

    test_grasp(instance, alpha=0.30, max_iter=8)