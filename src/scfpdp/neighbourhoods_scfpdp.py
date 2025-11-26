from src.scfpdp.solution import SCFPDPSolution, Route

import random
from typing import Optional, Tuple
from src.scfpdp.moves import InsertMove, SwapMove, RelocateMove

class Neighborhood:
    """Base class for all neighborhoods (lazy move generation)."""

    def __init__(self, name: str):
        self.name = name

    def next_move(self, solution: SCFPDPSolution):
        """Must be implemented by subclasses."""
        raise NotImplementedError


class InsertNeighborhood(Neighborhood):
    """Insert any pickup or dropoff at a different valid position in SAME route."""

    def __init__(self):
        super().__init__("INSERT")

    def next_move(self, solution: SCFPDPSolution):
        best_obj = solution.calc_objective()

        for r_idx, route in enumerate(solution.routes):
            for pos_from in range(len(route.route)):
                node = route.route[pos_from]

                # Try insertion in ALL possible indexes in the same route
                for pos_to in range(len(route.route) + 1):
                    if pos_to == pos_from:
                        continue

                    # Make tentative copy
                    new_sol = solution.copy()
                    new_route = new_sol.routes[r_idx]

                    # Remove and insert
                    moved = new_route.route.pop(pos_from)
                    new_route.route.insert(pos_to, moved)

                    # Feasibility: pickup before dropoff
                    if not _valid_request_order(new_route, moved, new_sol.inst.n):
                        continue

                    # Feasibility: vehicle capacity
                    if not _valid_capacity(new_route, new_sol.inst):
                        continue

                    # Compute new objective
                    new_obj = new_sol.calc_objective()

                    # Improving? If yes => return it lazily
                    if new_obj < best_obj:
                        return (("INSERT", r_idx, pos_from, pos_to), new_sol)

        # Nothing found
        return None


class SwapNeighborhood(Neighborhood):
    """Swap any two requests (pickup or dropoff) inside the SAME route."""

    def __init__(self):
        super().__init__("SWAP")

    def next_move(self, solution: SCFPDPSolution):
        best_obj = solution.calc_objective()

        for r_idx, route in enumerate(solution.routes):
            L = len(route.route)
            for i in range(L):
                for j in range(i + 1, L):
                    new_sol = solution.copy()
                    new_route = new_sol.routes[r_idx]

                    # Swap nodes
                    new_route.route[i], new_route.route[j] = new_route.route[j], new_route.route[i]

                    # Validate
                    if not _valid_full_route(new_route, new_sol.inst):
                        continue

                    new_obj = new_sol.calc_objective()
                    if new_obj < best_obj:
                        return (("SWAP", r_idx, i, j), new_sol)

        return None


class RelocateNeighborhood(Neighborhood):
    """Move pickup or dropoff into ANOTHER route or different position in SAME route."""

    def __init__(self):
        super().__init__("RELOCATE")

    def next_move(self, solution: SCFPDPSolution):
        best_obj = solution.calc_objective()

        for r_from, route in enumerate(solution.routes):
            for pos_from in range(len(route.route)):
                node = route.route[pos_from]

                for r_to, target in enumerate(solution.routes):
                    for pos_to in range(len(target.route) + 1):

                        # Copy
                        new_sol = solution.copy()
                        new_from = new_sol.routes[r_from]
                        new_to = new_sol.routes[r_to]

                        moved = new_from.route.pop(pos_from)
                        new_to.route.insert(pos_to, moved)

                        # Check feasibility for both routes
                        if not _valid_full_route(new_from, new_sol.inst):
                            continue
                        if not _valid_full_route(new_to, new_sol.inst):
                            continue

                        new_obj = new_sol.calc_objective()
                        if new_obj < best_obj:
                            return (("RELOCATE", r_from, pos_from, r_to, pos_to), new_sol)

        return None


# ----------------- CONSTRAINTS VALIDAITON -----------------

def _valid_request_order(route: Route, moved_node, n):
    """Returns True if pickup comes before dropoff for that request."""
    if moved_node < n:
        # pickup
        try:
            return route.route.index(moved_node) < route.route.index(moved_node + n)
        except ValueError:
            return False
    else:
        # dropoff
        try:
            return route.route.index(moved_node - n) < route.route.index(moved_node)
        except ValueError:
            return False

def _valid_capacity(route: Route, inst):
    try:
        route.recompute_route_distance()
        route.check()
        return True
    except ValueError:
        return False

def _valid_full_route(route: Route, inst):
    try:
        route.recompute_route_distance()
        route.check()
        return True
    except ValueError:
        return False


# ------------------ RANDOMIZED NEIGHBORHOODS ------------------
MAX_NEIGHBOR_SAMPLES = 50  # tune if needed


# ------------------ RANDOM INSERT ------------------
class RandomInsertNeighborhood:
    name = "INSERT"

    def next_move(self, sol: SCFPDPSolution) -> Optional[Tuple[object, SCFPDPSolution]]:
        inst = sol.inst
        for _ in range(MAX_NEIGHBOR_SAMPLES):
            # pick random request to insert, random src/dst route
            req = random.randint(0, inst.n - 1)
            rf = random.randint(0, inst.n_K - 1)
            rt = random.randint(0, inst.n_K - 1)
            pickup_at = random.randint(0, len(sol.routes[rt].route))
            drop_distance = random.randint(1, max(1, len(sol.routes[rt].route) - pickup_at + 1))

            move = InsertMove(rf, rt, req, pickup_at, drop_distance)
            candidate = move.apply(sol)

            if candidate.calc_objective() < sol.calc_objective():
                return move, candidate
        return None


# ------------------ RANDOM SWAP ------------------
class RandomSwapNeighborhood:
    name = "SWAP"

    def next_move(self, sol: SCFPDPSolution):
        inst = sol.inst
        for _ in range(MAX_NEIGHBOR_SAMPLES):
            ra = random.randint(0, inst.n_K - 1)
            rb = random.randint(0, inst.n_K - 1)

            # need valid requests in the route to swap
            if not sol.routes[ra].served_requests or not sol.routes[rb].served_requests:
                continue

            qa = random.choice(sol.routes[ra].served_requests)
            qb = random.choice(sol.routes[rb].served_requests)

            move = SwapMove(ra, rb, qa, qb)
            candidate = move.apply(sol)

            if candidate.calc_objective() < sol.calc_objective():
                return move, candidate
        return None


# ------------------ RANDOM RELOCATE ------------------
class RandomRelocateNeighborhood:
    name = "RELOCATE"

    def next_move(self, sol: SCFPDPSolution):
        inst = sol.inst
        for _ in range(MAX_NEIGHBOR_SAMPLES):
            rf = random.randint(0, inst.n_K - 1)
            rt = random.randint(0, inst.n_K - 1)

            if not sol.routes[rf].served_requests:
                continue

            req = random.choice(sol.routes[rf].served_requests)
            pickup_at = random.randint(0, len(sol.routes[rt].route))
            drop_distance = random.randint(1, max(1, len(sol.routes[rt].route) - pickup_at + 1))

            move = RelocateMove(rf, rt, req, pickup_at, drop_distance)
            candidate = move.apply(sol)

            if candidate.calc_objective() < sol.calc_objective():
                return move, candidate
        return None