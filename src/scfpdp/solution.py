import copy
from pathlib import Path
from typing import Any

from pymhlib.solution import Solution

from src.scfpdp.instance import SCFPDPInstance


class Route:
    def __init__(self, instance: SCFPDPInstance):
        self.instance = instance
        self.route: list[int] = []
        self.distance: int = 0
        self.served_requests: list[int] = []

    def __repr__(self):
        return f"Served requests: {{{', '.join(str(i) for i in self.served_requests)}}} along the route: {["depot"] + self.route + ["depot"]} (distance={self.distance:.2f}; capacity={self.get_carried_capacity()}/{self.instance.C})"

    def __len__(self) -> int:
        return len(self.route)

    def recompute_route_distance(self) -> None:
        route_distance = 0
        route_with_depots = [0] + self.route + [0]
        for i in route_with_depots[:-1]:
            route_distance += self.instance.distance_matrix[i][i + 1]
        self.distance = route_distance

    def insert_location(self, location_idx: int, at: int) -> None:
        if location_idx in self.route:
            raise ValueError(f"Request {location_idx} is already in the route")
        if location_idx in self.served_requests:
            raise ValueError(f"Request {location_idx} is already served")

        self.route.insert(at, location_idx)
        # serve the request if it's a pickup index
        if location_idx < self.instance.n:
            self.served_requests.append(location_idx)

    def serve_request(self, request_id: int, pickup_at: int, dropoff_distance_from_pickup: int) -> None:
        if pickup_at > len(self.route):
            raise ValueError(f"Pickup insertion index is out of range: {pickup_at}; route length: {len(self.route)}")
        if dropoff_distance_from_pickup <= 0:
            raise ValueError("Dropoff must be at least one position to the right from the pickup")
        dropoff_at = pickup_at + dropoff_distance_from_pickup
        if dropoff_at > len(self.route) + 1:
            raise ValueError(f"Dropoff insertion index is out of range: {dropoff_at}; route length: {len(self.route)}")
        self.insert_location(request_id, pickup_at)
        self.insert_location(request_id + self.instance.n, dropoff_at)
        self.check()

        # todo delta evaluation will integrate here to replace this inefficient recalculation at every insert
        self.recompute_route_distance()

    def can_take_request(self, request_id: int, at_position: int) -> bool:
        return self.get_capacity_at_position(at_position) + self.instance.demands[request_id] <= self.instance.C

    def get_carried_capacity(self) -> int:
        return self.get_capacity_at_position(len(self.route))

    def get_capacity_at_position(self, position: int) -> int:
        """
        Calculate vehicle capacity at a specific position in the route given what it has picked up so far
            but has not dropped off yet
        """
        capacity = 0
        for location_idx in self.route[:position]:
            if location_idx < self.instance.n:  # pickup
                capacity += self.instance.demands[location_idx]
            else:  # dropoff
                capacity -= self.instance.demands[location_idx - self.instance.n]
        return capacity

    def _check_capacity_constraint_at_position(self, position: int) -> int:
        capacity = 0
        for location_idx in self.route[:position]:
            if location_idx < self.instance.n:  # pickup
                capacity += self.instance.demands[location_idx]
                if capacity > self.instance.C:
                    raise ValueError(f"Capacity constraint violation at {location_idx} is violated")
            else:  # dropoff
                capacity -= self.instance.demands[location_idx - self.instance.n]
        return capacity

    def check_capacity_constraint(self) -> None:
        total_capacity = self._check_capacity_constraint_at_position(len(self.route))
        if total_capacity != 0:
            raise ValueError(f"Capacity constraint is violated with total capacity {total_capacity} instead of 0")

    def n_served_requests(self) -> int:
        return len(self.served_requests)

    def check(self) -> None:
        if len(self.route) % 2 !=  0:
            raise ValueError(f"Route of odd length: {len(self.route)}; the length must be even")

        for request_id in self.served_requests:
            pickup_idx = request_id
            dropoff_idx = request_id + self.instance.n

            if pickup_idx not in self.route:
                raise ValueError(f"Request {request_id} is marked as served but pickup location {pickup_idx} not in route")

            if dropoff_idx not in self.route:
                raise ValueError(f"Request {request_id} is marked as served but dropoff location {dropoff_idx} not in route")

            pickup_pos = self.route.index(pickup_idx)
            dropoff_pos = self.route.index(dropoff_idx)

            if pickup_pos >= dropoff_pos:
                raise ValueError(f"Request {request_id}: dropoff at position {dropoff_pos} must come after pickup at position {pickup_pos}")

        self.check_capacity_constraint()

    def swap_locations(self, location_a, location_b) -> None:
        self.route[location_a], self.route[location_b] = self.route[location_b], self.route[location_a]
        self.check()
        self.recompute_route_distance()

    def move_from_to(self, move_from: int, move_to: int) -> None:
        moved_value = self.route.pop(move_from)
        self.route.insert(move_to, moved_value)
        self.check()
        self.recompute_route_distance()

    def relocate_request_to(self, request_id: int, route_to: 'Route', pos_to: int, dropoff_distance_from_pickup: int) -> None:
        self.remove_request(request_id)
        route_to.serve_request(request_id, pos_to, dropoff_distance_from_pickup)

    def remove_request(self, request_id: int) -> None:
        """Remove both pickup and dropoff locations for a request from this route."""
        pickup_idx = request_id
        dropoff_idx = request_id + self.instance.n

        pickup_pos = self.route.index(pickup_idx)
        dropoff_pos = self.route.index(dropoff_idx)

        for idx in sorted([pickup_pos, dropoff_pos], reverse=True):
            self.route.pop(idx)

        if request_id in self.served_requests:
            self.served_requests.remove(request_id)

        self.check()
        self.recompute_route_distance()



class SCFPDPSolution(Solution):

    to_maximize = False

    def __init__(self, inst: SCFPDPInstance):
        super().__init__(inst)
        self.routes: list[Route] = [Route(inst) for _ in range(inst.n_K)]
        self.inst = inst  # this is overridden simply to help compiler with type hinting

    def __repr__(self):
        is_valid = True
        error = 'N/A'
        try:
            self.check()
        except ValueError as e:
            is_valid = False
            error = e.args[0]
        objective_message = f'Objective: {self.calc_objective():.2f}'
        lines = [f"SCFPDPSolution({objective_message if is_valid else f'Invalid solution: "{error}";{objective_message}'})"]
        for vehicle_i, route in enumerate(self.routes):
            lines.append(f"  Vehicle {vehicle_i}: {route}")
        total_requests = sum(route.n_served_requests() for route in self.routes)
        lines.append(f"  Total requests served: {total_requests}/{self.inst.n} (min required: {self.inst.gamma})")
        return '\n'.join(lines)

    def copy_from(self, other: 'SCFPDPSolution'):
        self.routes = copy.deepcopy(other.routes)

    def copy(self):
        sol = SCFPDPSolution(self.inst)
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        total_distance, sum_of_squares = 0, 0
        for route in self.routes:
            total_distance += route.distance
            sum_of_squares += route.distance**2
        if sum_of_squares == 0:
            return 0
        jain = total_distance / (self.inst.n_K * sum_of_squares)
        return total_distance + self.inst.rho * (1 - jain)

    def initialize(self, k):
        from src.scfpdp.construction_heuristics import GreedyConstructionHeuristic
        self.invalidate()
        GreedyConstructionHeuristic(self).construct()

    def check(self):
        super().check()

        # number of routes is not greater than number of vehicles (not sure this check is required)
        if len(self.routes) > self.inst.n_K:
            raise ValueError(f"Expected up to {self.inst.n_K} routes but got {len(self.routes)}")

        all_served_requests = set()
        for route_i, route in enumerate(self.routes):
            try:
                route.check()
            except ValueError as e:
                raise ValueError(f"Route {route_i} is invalid: {e}") from e

            for request_id in route.served_requests:
                if request_id in all_served_requests:
                    raise ValueError(f"Request {request_id} appears in multiple routes")
                all_served_requests.add(request_id)

        n_total_served_requests = len(all_served_requests)
        if n_total_served_requests < self.inst.gamma:
            raise ValueError(f"Not enough requests served: {n_total_served_requests} < {self.inst.gamma} (gamma)")
        return None

    def get_all_served_requests(self) -> set[int]:
        return set().union(*[r.served_requests for r in self.routes])

    def is_complete(self) -> bool:
        # self.check()
        return len(self.get_all_served_requests()) == self.inst.gamma

    def random_move_delta_eval(self, neighborhood: "Neighborhood") -> tuple[Any, float]:
        move, test_solution = neighborhood.generate_random_neighbor(self)
        if test_solution is None:  # no valid solution - let SA iterate further by skipping this infeasible
            return None, float('inf')
        current_obj = self.calc_objective()
        new_obj = test_solution.calc_objective()
        delta = new_obj - current_obj
        return move, delta

    def apply_neighborhood_move(self, move):
        move.move(self)

    def write_to_file(self, algorithm: str) -> None:
        instance_file = Path(self.inst.file_name)
        instance_name = instance_file.stem

        current = Path.cwd()
        project_root = current
        while project_root != project_root.parent:
            if (project_root / "instances").exists():
                break
            project_root = project_root.parent

        solutions_dir = project_root / "solutions"
        solutions_dir.mkdir(parents=True, exist_ok=True)

        output_file = solutions_dir / f"{instance_name}_{algorithm}_{self.calc_objective():.2f}.txt"
        
        with open(output_file, 'w') as f:
            f.write(instance_name + "\n")
            for route in self.routes:
                route_str = ' '.join(str(loc + 1) for loc in route.route)
                f.write(f"{route_str}\n")


if __name__ == '__main__':
    print(SCFPDPSolution(inst=SCFPDPInstance('10/test_instance_small.txt')))