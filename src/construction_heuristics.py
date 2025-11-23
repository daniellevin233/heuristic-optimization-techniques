from abc import ABC

import numpy as np

from src.scfpdp.instance import SCFPDPInstance
from src.scfpdp.solution import SCFPDPSolution, Route


class ConstructionHeuristic(ABC):
    def construct(self, solution: SCFPDPSolution) -> None:
        raise NotImplementedError('Construction heuristic is not implemented.')


class GreedyConstructionHeuristic(ConstructionHeuristic):
    @staticmethod
    def _find_closest_pickup_location(instance: SCFPDPInstance, last_dropoff_idx: int, excluded_requests: set[int]) -> int:
        """
        last_dropoff_idx: index of the last dropoff or depot before the next pickup
        """
        if len(excluded_requests) == instance.n - 1:
            raise ValueError("No more requests to serve")
        relevant_pickup_locations = instance.distance_matrix[last_dropoff_idx, 1:instance.n + 1]
        # mark excluded pickup candidates as inf
        relevant_pickup_locations[list(excluded_requests)] = np.inf
        # randomized tie-breaking can be applied here, argmin will return the index of the first minimum value
        closest_requests = np.argmin(relevant_pickup_locations)

        return int(closest_requests)

    def construct(self, solution: SCFPDPSolution) -> None:
        served_requests: set[int] = set()
        completed_routes: list[Route] = []
        while len(served_requests) < solution.inst.gamma:
            for current_route in solution.routes:  # go over every route
                if current_route in completed_routes:
                    continue

                next_request = self._find_closest_pickup_location(solution.inst, current_route.route[-2], served_requests)

                # if capacity is exceeded, move on to the next route
                if solution.inst.demands[next_request] + current_route.get_current_capacity() > solution.inst.C:
                    completed_routes.append(current_route)
                    continue

                # in this greedy construction - append pickup and dropoff together as the last two stops before depot
                current_route.serve_request(
                    next_request,
                    len(current_route.route) - 1, # right before end depot
                    1,
                )
                served_requests.add(next_request)
                if len(served_requests) >= solution.inst.gamma:
                    break
        # validate to make sure the requirements were fulfilled
        solution.check()


class RandomizedConstructionHeuristic(GreedyConstructionHeuristic):
    pass

if __name__ == '__main__':
    initial_solution = SCFPDPSolution(inst=SCFPDPInstance('10/test_instance_small.txt'))
    initial_solution.initialize(0)
    print(initial_solution)