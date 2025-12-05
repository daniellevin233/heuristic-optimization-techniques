import random
from abc import ABC, abstractmethod
from typing import Generator, Any
from dataclasses import dataclass

from src.solution import SCFPDPSolution, Route


class Neighborhood(ABC):
    """Base class for all neighborhoods."""

    @staticmethod
    @abstractmethod
    def generate_random_neighbor(solution: SCFPDPSolution) -> tuple[Any, SCFPDPSolution] | tuple[Any, None]:
        """Generate a single random neighbor: (move_metadata, neighbor_solution)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def generate_neighbors(solution: SCFPDPSolution) -> Generator[SCFPDPSolution, None, None]:
        """Yield all feasible neighbors (SCFPDPSolution objects only)."""
        raise NotImplementedError

    @staticmethod
    def pick_random_non_empty_route(solution: SCFPDPSolution) -> tuple[int, Route] | None:
        non_empty_routes = [(i, r) for i, r in enumerate(solution.routes) if len(r) > 0]
        if not non_empty_routes:
            return None
        return random.choice(non_empty_routes)


class InsertNeighborhood(Neighborhood):
    """Insert any pickup or dropoff at a different position in the same route."""

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
        res = Neighborhood.pick_random_non_empty_route(solution)
        if res is None:
            return None, None
        route_idx, route = res
        if len(route) < 2:
            return None, None
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
    def generate_neighbors(solution: SCFPDPSolution) -> Generator[SCFPDPSolution, None, None]:
        for r_idx, route in enumerate(solution.routes):
            for pos_from in range(len(route)):
                for pos_to in range(len(route) + 1):
                    if pos_from == pos_to:
                        continue
                    new_sol = solution.copy()
                    try:
                        new_sol.routes[r_idx].move_from_to(pos_from, pos_to)
                        yield new_sol
                    except ValueError:
                        continue


class SwapNeighborhood(Neighborhood):
    """Swap two locations in the same route."""

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
        res = Neighborhood.pick_random_non_empty_route(solution)
        if res is None:
            return None, None
        route_idx, route = res
        if len(route) < 2:
            return None, None
        i, j = sorted(random.sample(range(len(route)), 2))
        neighbor = solution.copy()
        move = SwapNeighborhood.Move(route_idx, i, j)
        try:
            move.move(neighbor)
            return move, neighbor
        except ValueError:
            return None, None

    @staticmethod
    def generate_neighbors(solution: SCFPDPSolution) -> Generator[SCFPDPSolution, None, None]:
        for r_idx, route in enumerate(solution.routes):
            L = len(route)
            for i in range(L):
                for j in range(i + 1, L):
                    new_sol = solution.copy()
                    try:
                        new_sol.routes[r_idx].swap_locations(i, j)
                        yield new_sol
                    except ValueError:
                        continue


class RelocateNeighborhood(Neighborhood):
    """Relocate a full request (pickup + dropoff) to another route or different position."""

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
        res = Neighborhood.pick_random_non_empty_route(solution)
        if res is None:
            return None, None
        route_from_idx, route_from = res
        if len(route_from) == 0:
            return None, None
        pos_from = random.randint(0, len(route_from) - 1)
        route_to_idx = random.randint(0, len(solution.routes) - 1)
        pos_to = random.randint(0, len(solution.routes[route_to_idx]))
        neighbor = solution.copy()
        move = RelocateNeighborhood.Move(route_from_idx, pos_from, route_to_idx, pos_to)
        try:
            move.move(neighbor)
            return move, neighbor
        except ValueError:
            return None, None

    @staticmethod
    def generate_neighbors(solution: SCFPDPSolution) -> Generator[SCFPDPSolution, None, None]:
        n = solution.inst.n
        for src_idx, src_route in enumerate(solution.routes):
            for pos_from in range(len(src_route)):
                request_id = src_route.route[pos_from] % n
                for tgt_idx, tgt_route in enumerate(solution.routes):
                    for pos_to in range(len(tgt_route) + 1):
                        new_sol = solution.copy()
                        try:
                            new_sol.routes[src_idx].relocate_request_to(request_id, new_sol.routes[tgt_idx], pos_to, 1)
                            yield new_sol
                        except ValueError:
                            continue