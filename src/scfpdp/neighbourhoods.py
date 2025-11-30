import random
from abc import abstractmethod, ABC
from typing import Generator, Any
from dataclasses import dataclass

from src.scfpdp.solution import SCFPDPSolution, Route


class Neighborhood(ABC):
    """Base class for all neighborhoods (lazy move generation)."""

    @staticmethod
    @abstractmethod
    def generate_random_neighbor(solution: SCFPDPSolution) -> tuple[Any, SCFPDPSolution] | tuple[Any, None]:
        """Generate random neighbor and return (move_metadata, neighbor_solution)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def generate_neighbors(solution: SCFPDPSolution) -> Generator[SCFPDPSolution]:
        """Must be implemented by subclasses. Yields all valid neighbors."""
        raise NotImplementedError

    @staticmethod
    def pick_random_non_empty_route(solution: SCFPDPSolution) -> tuple[int, Route] | None:
        non_empty_routes = [(i, r) for i, r in enumerate(solution.routes) if len(r) > 0]
        if not non_empty_routes:
            return None
        return random.choice(non_empty_routes)


class InsertNeighborhood(Neighborhood):
    """Insert any pickup or dropoff at a different valid position in SAME route."""

    @dataclass
    class Move:
        route_idx: int
        pos_from: int
        pos_to: int

        def move(self, solution: SCFPDPSolution) -> None:
            route = solution.routes[self.route_idx]
            route.move_from_to(self.pos_from, self.pos_to)

    @staticmethod
    def generate_random_neighbor(solution: SCFPDPSolution) -> tuple[Any, SCFPDPSolution] | tuple[Any, None]:
        result = Neighborhood.pick_random_non_empty_route(solution)
        if result is None:
            return None, None
        route_idx, route = result

        pos_from = random.randint(0, len(route) - 1)
        pos_to = random.randint(0, len(route))

        if pos_from == pos_to:
            return None, None

        neighbor = solution.copy()
        move = InsertNeighborhood.Move(route_idx, pos_from, pos_to)
        try:
            move.move(neighbor)
            return move, neighbor
        except ValueError:
            return None, None

    @staticmethod
    def generate_neighbors(solution: SCFPDPSolution) -> Generator[SCFPDPSolution]:
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

                    yield new_sol


class SwapNeighborhood(Neighborhood):
    """Swap any two requests (pickup or dropoff) inside the SAME route."""

    @dataclass
    class Move:
        route_idx: int
        pos_i: int
        pos_j: int

        def move(self, solution: SCFPDPSolution) -> None:
            route = solution.routes[self.route_idx]
            route.swap_locations(self.pos_i, self.pos_j)

    @staticmethod
    def generate_random_neighbor(solution: SCFPDPSolution) -> tuple[Any, SCFPDPSolution] | tuple[Any, None]:
        result = Neighborhood.pick_random_non_empty_route(solution)
        if result is None:
            return None, None
        route_idx, route = result

        if len(route) < 2:
            return None, None

        i, j = random.sample(range(len(route)), 2)
        if i > j:
            i, j = j, i

        neighbor = solution.copy()
        move = SwapNeighborhood.Move(route_idx, i, j)
        try:
            move.move(neighbor)
            return move, neighbor
        except ValueError:
            return None, None

    @staticmethod
    def generate_neighbors(solution: SCFPDPSolution) -> Generator[SCFPDPSolution]:
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

                    yield new_sol


class RelocateNeighborhood(Neighborhood):
    """Relocate a full request (pickup + dropoff) to another route or new position."""

    @dataclass
    class Move:
        route_from_idx: int
        pos_from: int
        route_to_idx: int
        pos_to: int

        def move(self, solution: SCFPDPSolution) -> None:
            route_from = solution.routes[self.route_from_idx]
            request_id = route_from.route[self.pos_from]
            route_to = solution.routes[self.route_to_idx]
            route_from.relocate_request_to(request_id, route_to, self.pos_to, 1)

    @staticmethod
    def generate_random_neighbor(solution: SCFPDPSolution) -> tuple[Any, SCFPDPSolution] | tuple[Any, None]:
        result = Neighborhood.pick_random_non_empty_route(solution)
        if result is None:
            return None, None
        route_from_idx, route_from = result
        route_from = solution.routes[route_from_idx]
        pos_from = random.randint(0, len(route_from) - 1)

        route_to_idx = random.randint(0, len(solution.routes) - 1)
        route_to = solution.routes[route_to_idx]
        pos_to = random.randint(0, len(route_to))

        move = RelocateNeighborhood.Move(route_from_idx, pos_from, route_to_idx, pos_to)
        neighbor = solution.copy()
        try:
            move.move(neighbor)
            return move, neighbor
        except ValueError:
            return None, None

    @staticmethod
    def generate_neighbors(solution: SCFPDPSolution) -> Generator[SCFPDPSolution]:
        n = solution.inst.n

        for source_route_idx, source_route in enumerate(solution.routes):
            for pos_from in range(len(source_route)):

                request_id = source_route.route[pos_from] % n

                for target_route_idx, target_route in enumerate(solution.routes):
                    for pos_to in range(len(target_route) + 1):

                        new_sol = solution.copy()
                        new_from = new_sol.routes[source_route_idx]
                        new_to = new_sol.routes[target_route_idx]

                        try:
                            new_from.relocate_request_to(request_id, new_to, pos_to, 1)
                        except ValueError:
                            continue

                        yield new_sol
