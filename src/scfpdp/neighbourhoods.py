from src.scfpdp.solution import SCFPDPSolution, Route


class Neighborhood:
    """Base class for all neighborhoods (lazy move generation)."""

    def __init__(self, name: str):
        self.name = name

    def generate_neighbors(self, solution: SCFPDPSolution):
        """Must be implemented by subclasses. Yields all valid neighbors."""
        raise NotImplementedError


class InsertNeighborhood(Neighborhood):
    """Insert any pickup or dropoff at a different valid position in SAME route."""

    def __init__(self):
        super().__init__("INSERT")

    def generate_neighbors(self, solution: SCFPDPSolution):
        for r_idx, route in enumerate(solution.routes):
            for pos_from in range(len(route)):
                for pos_to in range(len(route) + 1):
                    if pos_to == pos_from:
                        continue

                    new_sol = solution.copy()
                    new_route: Route = new_sol.routes[r_idx]

                    try:
                        new_route.move_from_to(pos_from, pos_to)
                    except ValueError:
                        continue

                    yield (("INSERT", r_idx, pos_from, pos_to), new_sol)


class SwapNeighborhood(Neighborhood):
    """Swap any two requests (pickup or dropoff) inside the SAME route."""

    def __init__(self):
        super().__init__("SWAP")

    def generate_neighbors(self, solution: SCFPDPSolution):
        for r_idx, route in enumerate(solution.routes):
            L = len(route)
            for i in range(L):
                for j in range(i + 1, L):
                    new_sol = solution.copy()
                    new_route = new_sol.routes[r_idx]

                    try:
                        new_route.swap_locations(i, j)
                    except ValueError:
                        continue

                    yield (("SWAP", r_idx, i, j), new_sol)


class RelocateNeighborhood(Neighborhood):
    """Move pickup or dropoff into ANOTHER route or different position in SAME route."""

    def __init__(self):
        super().__init__("RELOCATE")

    def generate_neighbors(self, solution: SCFPDPSolution):
        for r_from, route in enumerate(solution.routes):
            for pos_from in range(len(route.route)):
                for r_to, target in enumerate(solution.routes):
                    for pos_to in range(len(target.route) + 1):

                        new_sol = solution.copy()
                        new_from = new_sol.routes[r_from]
                        new_to = new_sol.routes[r_to]

                        try:
                            new_from.relocate_location_to(pos_from, new_to, pos_to)
                        except ValueError:
                            continue

                        yield (("RELOCATE", r_from, pos_from, r_to, pos_to), new_sol)
