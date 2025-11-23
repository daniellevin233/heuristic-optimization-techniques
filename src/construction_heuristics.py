from abc import ABC

import numpy as np

from src.scfpdp.instance import SCFPDPInstance
from src.scfpdp.solution import SCFPDPSolution, Route


class ConstructionHeuristic(ABC):
    def construct(self, solution: SCFPDPSolution) -> None:
        raise NotImplementedError('Construction heuristic is not implemented.')


class GreedyConstructionHeuristic(ConstructionHeuristic):
    @staticmethod
    def _find_closest_pickup_location(instance: SCFPDPInstance, last_dropoff_idx: int, excluded_requests: set[int]) -> int | None:
        """
        last_dropoff_idx: index of the last dropoff or depot before the next pickup
        """
        if len(excluded_requests) == instance.n - 1:
            return None

        # 1:n+1 to skip over depot
        relevant_pickup_locations = instance.distance_matrix[last_dropoff_idx, 1:instance.n + 1]
        # mark excluded pickup candidates as inf
        relevant_pickup_locations[list(excluded_requests)] = np.inf
        # randomized tie-breaking can be applied here, argmin will return the index of the first minimum value
        closest_request = np.argmin(relevant_pickup_locations)

        return int(closest_request)

    @staticmethod
    def _find_closest_fitting_request(instance: SCFPDPInstance, route: Route, excluded_requests: set[int]) -> int | None:
        closest_request = GreedyConstructionHeuristic._find_closest_pickup_location(instance, route.route[-2], excluded_requests)
        non_fitting_requests = set()
        while closest_request != np.inf and closest_request is not None:
            if route.can_take_request(closest_request):
                return closest_request
            non_fitting_requests.add(closest_request)
            closest_request = GreedyConstructionHeuristic._find_closest_pickup_location(instance, route.route[-2], excluded_requests | non_fitting_requests)
        return None

    def construct(self, solution: SCFPDPSolution) -> None:
        served_requests: set[int] = set()
        full_routes: list[Route] = []
        while len(served_requests) < solution.inst.gamma and len(full_routes) < len(solution.routes):
            for current_route in solution.routes:  # go over every route
                if current_route in full_routes:
                    continue

                next_request = self._find_closest_fitting_request(solution.inst, current_route, served_requests)

                # If there is no feasible next request for this route, move on to the next route.
                if next_request is None:
                    full_routes.append(current_route)
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
        try:
            solution.check()
        except ValueError:
            raise ValueError(f"Couldn't construct a satisfying solution. Best solution found: \n\n {solution}")


class RandomizedConstructionHeuristic(GreedyConstructionHeuristic):
    pass

if __name__ == '__main__':
    # initial_solution = SCFPDPSolution(inst=SCFPDPInstance('10/test_instance_small.txt'))
    initial_solution = SCFPDPSolution(inst=SCFPDPInstance('50/train/instance1_nreq50_nveh2_gamma50.txt'))
    initial_solution.initialize(0)
    print(initial_solution)