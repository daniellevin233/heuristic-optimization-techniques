from abc import ABC, abstractmethod
import random

import numpy as np

from src.scfpdp.instance import SCFPDPInstance
from src.scfpdp.solution import SCFPDPSolution, Route


class ConstructionHeuristic(ABC):

    def __init__(self, initial_solution: SCFPDPSolution) -> None:
        self.instance: SCFPDPInstance = initial_solution.inst
        self.solution: SCFPDPSolution = initial_solution

    @abstractmethod
    def _select_next_request(self, route: Route, excluded_requests: set[int], insertion_position: int) -> int | None:
        raise NotImplementedError()

    @staticmethod
    def select_insertion_position(route: Route) -> int:
        """Insert in the end, just before the end depot"""
        return len(route)

    def _find_closest_pickup_location(self, from_location_idx: int, excluded_requests: set[int]) -> int | None:
        """
        Find the closest pickup location to from_location_idx out of the remaining ones
        """
        if len(excluded_requests) >= self.instance.n:
            return None

        # 1:n+1 to skip over depot
        relevant_pickup_locations = self.instance.distance_matrix[from_location_idx, 1:self.instance.n + 1].copy()
        # mark excluded pickup candidates as inf
        relevant_pickup_locations[list(excluded_requests)] = np.inf

        if np.all(relevant_pickup_locations == np.inf):
            return None

        # randomized tie-breaking can be applied here, argmin will return the index of the first minimum value
        closest_request = np.argmin(relevant_pickup_locations)
        return int(closest_request)

    @staticmethod
    def select_dropoff_distance(route: Route) -> int:
        """By default drop off directly after pickup"""
        return 1

    def construct(self) -> None:
        served_requests: set[int] = set()
        full_routes: list[Route] = []

        while len(served_requests) < self.instance.gamma and len(full_routes) < len(self.solution.routes):
            for current_route in self.solution.routes:  # go over every route
                if current_route in full_routes:
                    continue

                next_insert_at = self.select_insertion_position(current_route)
                next_request = self._select_next_request(current_route, served_requests, next_insert_at)
                # If there is no feasible next request for this route, move on to the next route.
                if next_request is None:
                    full_routes.append(current_route)
                    continue

                dropoff_distance = self.select_dropoff_distance(current_route)

                # In this greedy construction - append pickup and dropoff together as the last two stops before depot
                current_route.serve_request(next_request, next_insert_at, dropoff_distance)
                served_requests.add(next_request)

                if len(served_requests) >= self.instance.gamma:
                    break

        # validate to make sure the requirements were fulfilled
        try:
            self.solution.check()
        except ValueError:
            raise ValueError(f"Couldn't construct a satisfying solution. Best solution found: \n\n {self.solution}")


class GreedyConstructionHeuristic(ConstructionHeuristic):
    def _find_closest_fitting_pickup_location(self, route: Route, excluded_requests: set[int], from_location_idx: int) -> int | None:
        """
        Find the closest fitting request from from_location_idx out of the remaining ones
        """
        closest_request = self._find_closest_pickup_location(from_location_idx, excluded_requests)
        non_fitting_requests = set()

        while closest_request is not None:
            if route.can_take_request(closest_request, from_location_idx):
                return closest_request
            non_fitting_requests.add(closest_request)
            closest_request = self._find_closest_pickup_location(
                from_location_idx, excluded_requests | non_fitting_requests
            )

        return None

    def _select_next_request(self, route: Route, excluded_requests: set[int], insertion_position: int) -> int | None:
        """Greedily select the closest to the insertion position fitting pickup"""
        return self._find_closest_fitting_pickup_location(route, excluded_requests, insertion_position)


class RandomizedConstructionHeuristic(GreedyConstructionHeuristic):
    def __init__(self, initial_solution: SCFPDPSolution, top_random_pickups_to_consider: int = 10):
        super().__init__(initial_solution)
        self.top_random_pickups_to_consider = top_random_pickups_to_consider

    def get_rcl_of_next_requests(self, route: Route, excluded_requests: set[int], insertion_location_idx: int) -> list[int] | None:
        """Select request using RCL of length top_random_pickups_to_consider to insert at the insertion location."""
        capacity_at_insertion = route.get_capacity_at_position(insertion_location_idx)
        remaining_capacity = self.instance.C - capacity_at_insertion

        distances_to_pickups = self.instance.distance_matrix[insertion_location_idx, 1:self.instance.n + 1]

        candidates = []
        for request_id in range(self.instance.n):
            # request hasn't been served yet and it can be taken without exceeding capacity
            if request_id not in excluded_requests and self.instance.demands[request_id] <= remaining_capacity:
                candidates.append((distances_to_pickups[request_id], request_id))

        if not candidates:
            return None

        candidates.sort()
        rcl = candidates[:min(self.top_random_pickups_to_consider, len(candidates))]
        return [c[1] for c in rcl]

    def _select_next_request(self, route: Route, excluded_requests: set[int], insertion_location_idx: int) -> int | None:
        """Select request using RCL of length 10 to insert at the insertion location."""
        rcl = self.get_rcl_of_next_requests(route, excluded_requests, insertion_location_idx)
        return random.choice(rcl)


if __name__ == '__main__':
    test_instance = SCFPDPInstance('10/test_instance_small.txt')
    competition_instance = SCFPDPInstance('1000/competition/instance61_nreq1000_nveh20_gamma879.txt')

    instance = competition_instance

    _initial_solution = SCFPDPSolution(inst=instance)
    GreedyConstructionHeuristic(_initial_solution).construct()
    print("Greedy solution: ")
    print(_initial_solution)

    _initial_solution_1 = SCFPDPSolution(inst=instance)
    RandomizedConstructionHeuristic(_initial_solution_1).construct()
    print("\n\nRandomized solution: ")
    print(_initial_solution_1)

    best_solution = _initial_solution if _initial_solution.calc_objective() < _initial_solution_1.calc_objective() else _initial_solution_1
    best_algo = "greedy_construction" if _initial_solution.calc_objective() < _initial_solution_1.calc_objective() else "randomized_construction"

    print(f"Best solution ({best_algo}): ")
    print(best_solution)

    best_solution.write_to_file(algorithm=best_algo)
