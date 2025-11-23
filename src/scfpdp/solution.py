import copy

from pymhlib.solution import Solution

from src.scfpdp.instance import SCFPDPInstance

class Route:
    def __init__(self, instance: SCFPDPInstance):
        self.instance = instance
        self.route: list[int] = [0, 0]  # start and end every route at depot
        self.distance: int = 0
        self.served_requests: list[int] = []

    def __repr__(self):
        return f"Served requests: {{{', '.join(str(i) for i in self.served_requests)}}} along the route: {self.route} (distance={self.distance:.2f}; capacity={self.get_current_capacity()}/{self.instance.C})"

    def recompute_route_distance(self) -> None:
        route_distance = 0
        for i in self.route[:-1]:
            route_distance += self.instance.distance_matrix[i][i + 1]
        self.distance = route_distance

    def insert_location(self, location_idx: int, at: int) -> None:
        assert location_idx not in self.route[1:-1], f"Cannot insert location {location_idx} that has already been added to the route: {self.route}"
        self.route.insert(at, location_idx)
        # serve the request either by its pickup or dropoff location if not there already
        if location_idx % self.instance.n not in self.served_requests:
            self.served_requests.append(location_idx % self.instance.n)
        # todo delta evaluation will integrate here to replace this inefficient recalculation at every insert
        self.recompute_route_distance()

    def can_take_request(self, request_id: int) -> bool:
        return self.get_current_capacity() + self.instance.demands[request_id] <= self.instance.C

    def serve_request(self, pickup_idx: int, pickup_at: int, dropoff_distance_from_pickup: int) -> None:
        assert pickup_at <= len(self.route), f"Pickup insertion index is out of range: {pickup_at}; route length: {len(self.route)}"
        assert dropoff_distance_from_pickup > 0, f"Dropoff must be at least one position to the right from the pickup"
        dropoff_at = pickup_at + dropoff_distance_from_pickup
        assert dropoff_at <= len(self.route) + 1, f"Dropoff insertion index is out of range: {dropoff_at}; route length: {len(self.route)}"
        self.insert_location(pickup_idx, pickup_at)
        self.insert_location(pickup_idx + self.instance.n, dropoff_at)

    def get_current_capacity(self) -> int:
        return sum(self.instance.demands[i] for i in self.served_requests)

    def n_served_requests(self) -> int:
        return len(self.served_requests)

    def check(self) -> None:
        if len(self.route) < 2:
            raise ValueError(f"Route too short: {len(self.route)} < 2; it must contain depot as start and end at the very least")

        if self.route[0] != 0:
            raise ValueError(f"Route does not start at depot: first location is {self.route[0]}")

        if self.route[-1] != 0:
            raise ValueError(f"Route does not end at depot: last location is {self.route[-1]} ({self.route})")

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

        current_capacity = self.get_current_capacity()
        if current_capacity > self.instance.C:
            raise ValueError(f"Route capacity exceeded: {current_capacity} > {self.instance.C}")



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
        lines = [f"SCFPDPSolution({f'Objective: {self.calc_objective():.2f}' if is_valid else f'Invalid solution: "{error}"'})"]
        for vehicle_i, route in enumerate(self.routes):
            lines.append(f"  Vehicle {vehicle_i}: {route}")
        total_requests = sum(route.n_served_requests() for route in self.routes)
        lines.append(f"  Total requests served: {total_requests}/{self.inst.n} (min required: {self.inst.gamma})")
        return '\n'.join(lines)

    def copy_from(self, other: 'SCFPDPSolution'):
        super().copy_from(other)
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
        jain = total_distance / (self.inst.n * sum_of_squares)
        return total_distance + self.inst.rho * (1 - jain)

    def initialize(self, k):
        from src.construction_heuristics import GreedyConstructionHeuristic
        GreedyConstructionHeuristic().construct(self)
        self.invalidate()

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


if __name__ == '__main__':
    print(SCFPDPSolution(inst=SCFPDPInstance('10/test_instance_small.txt')))