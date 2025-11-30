"""
SCFPDP Algorithm Comparison Experiments
---------------------------------------
Evaluates:
(a) Deterministic vs. Randomized construction, GRASP
(b) Local search on multiple neighborhoods × 2 step strategies
(c) VND
For each instance size, runs each algorithm multiple times and records:
- objective
- fairness
- travel duration
- runtime
- objective over iterations (if available)

This is structured similarly to beam search experiments.
"""

import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.scfpdp.instance import SCFPDPInstance
from src.scfpdp.solution import SCFPDPSolution
from src.algorithms.construction_heuristics import (
    GreedyConstructionHeuristic,
    RandomizedConstructionHeuristic,
)
from src.scfpdp.neighborhoods_scfpdp import (
    InsertNeighborhood,
    SwapNeighborhood,
    RelocateNeighborhood,
)
from src.scfpdp.local_search import VND
from src.scfpdp.step_strategies import FirstImprovement, BestImprovement
from src.scfpdp.grasp import GraspSCFPDP

from src.utils import find_project_root


# ---------------------------------------------------------------------------
# Helper: Fairness & Travel-Time extraction (change if your solution stores differently)
# ---------------------------------------------------------------------------
def extract_metrics(sol: SCFPDPSolution) -> dict:
    return {
        "objective": sol.calc_objective(),
        "fairness": sol.compute_fairness() if hasattr(sol, "compute_fairness") else None,
        "travel_time": sol.compute_total_travel_time() if hasattr(sol, "compute_total_travel_time") else None,
    }


# ---------------------------------------------------------------------------
# Run a single algorithm once
# ---------------------------------------------------------------------------
def timed_run(func):
    t0 = time.perf_counter()
    sol = func()
    t1 = time.perf_counter()
    return sol, t1 - t0


# ---------------------------------------------------------------------------
# Local Search Variants
# ---------------------------------------------------------------------------
NEIGHBORHOOD_SETS = {
    "INSERT": [InsertNeighborhood()],
    "SWAP": [SwapNeighborhood()],
    "RELOCATE": [RelocateNeighborhood()],
    "INSERT+SWAP": [InsertNeighborhood(), SwapNeighborhood()],
    "INSERT+RELOCATE": [InsertNeighborhood(), RelocateNeighborhood()],
    "3-NEIGHBORHOOD": [InsertNeighborhood(), SwapNeighborhood(), RelocateNeighborhood()],
}

STEP_STRATS = {
    "FIRST": FirstImprovement(),
    "BEST": BestImprovement(),
}


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------
def evaluate_instance(instance_path: Path, runs: int = 5):
    inst = SCFPDPInstance(str(instance_path))
    results = []

    print(f"\nProcessing: {instance_path.name} (n={inst.n})")

    # ----------------------------
    # Deterministic construction
    # ----------------------------
    for _ in range(runs):
        sol0 = SCFPDPSolution(inst)
        g = GreedyConstructionHeuristic(sol0)
        sol_greedy, rt = timed_run(lambda: (g.construct() or sol0))
        results.append(("DETERMINISTIC", extract_metrics(sol_greedy), rt))

    # ----------------------------
    # Randomized construction
    # ----------------------------
    for _ in range(runs):
        sol0 = SCFPDPSolution(inst)
        rnd = RandomizedConstructionHeuristic(sol0, top_random_pickups_to_consider=10)
        sol_rnd, rt = timed_run(lambda: (rnd.construct() or sol0))
        results.append(("RANDOMIZED", extract_metrics(sol_rnd), rt))

    # ----------------------------
    # GRASP  (updated API)
    # ----------------------------
    # construction function
    def grasp_construct(alpha_val: float):
        sol = SCFPDPSolution(inst)
        # map alpha [0,1] to top_k in [1, n]
        top_k = max(1, int(round(alpha_val * inst.n)))
        top_k = min(inst.n, top_k)
        ctor = RandomizedConstructionHeuristic(sol, top_random_pickups_to_consider=top_k)
        ctor.construct()
        return sol

    # local search for GRASP (full 3‑neighborhood VND, FIRST improvement)
    grasp_vnd = VND(
        [InsertNeighborhood(), SwapNeighborhood(), RelocateNeighborhood()],
        FirstImprovement(),
    )

    def grasp_local_search(sol: SCFPDPSolution):
        return grasp_vnd.run(sol)

    for _ in range(runs):
        solver = GraspSCFPDP(
            construct=grasp_construct,
            local_search=grasp_local_search,
            alpha=0.3,
            max_iter=50,
        )
        sol_g, rt = timed_run(lambda: solver.run(verbose=False))
        results.append(("GRASP", extract_metrics(sol_g), rt))

    # ----------------------------
    # Local Search Variants
    # ----------------------------
    det_sol = SCFPDPSolution(inst)
    GreedyConstructionHeuristic(det_sol).construct()

    for n_name, nhs in NEIGHBORHOOD_SETS.items():
        for s_name, step in STEP_STRATS.items():
            ls_name = f"LS-{n_name}-{s_name}"

            for _ in range(runs):
                sol_ls = det_sol.copy()

                vnd = VND(nhs, step)
                sol_final, rt = timed_run(lambda: vnd.run(sol_ls))

                results.append((ls_name, extract_metrics(sol_final), rt))

    # ----------------------------
    # VND from deterministic
    # ----------------------------
    for _ in range(runs):
        sol_vnd = det_sol.copy()
        full_vnd = VND(
            [InsertNeighborhood(), SwapNeighborhood(), RelocateNeighborhood()],
            FirstImprovement(),
        )
        sol_final, rt = timed_run(lambda: full_vnd.run(sol_vnd))
        results.append(("VND", extract_metrics(sol_final), rt))

    return results


# ---------------------------------------------------------------------------
# Multi-instance runner
# ---------------------------------------------------------------------------
def evaluate_all(instance_sizes: list[str], runs=5):
    project_root = find_project_root()
    inst_root = project_root / "instances"

    all_results = []

    for size in instance_sizes:
        inst_dir = inst_root / size / "competition"
        files = sorted(list(inst_dir.glob("*.txt")))
        if not files:
            continue

        print(f"\n### Instance size: {size} ###\n")

        for f in tqdm(files, desc=f"Size {size}"):
            res = evaluate_instance(f, runs=runs)
            for algo, metrics, rt in res:
                all_results.append([size, f.name, algo, rt, metrics])

    return all_results


if __name__ == "__main__":
    sizes = ["50", "100", "200"]
    results = evaluate_all(sizes, runs=5)

    out = Path("experiment_results.npy")
    np.save(out, results)
    print(f"\nSaved results to {out}")