import numpy as np
import math


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def distance_from(self, other: "Point") -> int:
        """ Euclidean distance between two points. """
        return math.ceil(math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2))


class SCFPDPInstance:
    """An instance of the Selective Capacitated Fair Pickup and Delivery Problem.

    Attributes:
        n: number of customer requests
        n_K: number of vehicles
        C: vehicle capacity
        gamma: minimum number of requests to fulfill
        rho: fairness weight parameter
        demands: array of demand values for each request [c1, c2, ..., cn]
        depot_location: (x, y) coordinates of the depot
        pickup_locations: array of (x, y) coordinates for each pickup location
        dropoff_locations: array of (x, y) coordinates for each dropoff location
        distance_matrix: precomputed distance matrix for all locations
    """

    def __init__(self, file_name: str):
        """
        Args:
            file_name: path to the instance file
        """
        self.n = None
        self.n_K = None
        self.C = None
        self.gamma = None
        self.rho = None

        self.demands: np.ndarray | None = None
        self.depot_location: Point | None = None
        self.pickup_locations: list[Point] | None = None
        self.dropoff_locations: list[Point] | None = None
        self.distance_matrix: np.ndarray[int] | None = None

        self._parse_file(file_name)
        self._compute_distances()

    def _parse_file(self, file_name: str):
        """
        File format:
            n n_K C gamma rho
            # demands
            c1 c2 ... cn
            # request locations
            x_depot y_depot
            x_pickup1 y_pickup1 x_pickup2 y_pickup2 ... x_pickupn y_pickupn
            x_dropoff1 y_dropoff1 x_dropoff2 y_dropoff2 ... x_dropoffn y_dropoffn
        """
        with open(f"../instances/{file_name}", 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        params = lines[0].split()
        self.n = int(params[0])
        self.n_K = int(params[1])
        self.C = int(params[2])
        self.gamma = int(params[3])
        self.rho = float(params[4])

        demands_idx = lines.index('# demands') + 1
        self.demands = np.array([int(x) for x in lines[demands_idx].split()])

        depot_idx = lines.index('# request locations') + 1
        depot_coords = lines[depot_idx].split()
        self.depot_location = Point(int(depot_coords[0]), int(depot_coords[1]))

        pickup_start_idx = depot_idx + 1
        self.pickup_locations = np.array(
            [
                Point(int(lines[pickup_start_idx + i].split()[0]),
                      int(lines[pickup_start_idx + i].split()[1]))
                for i in range(self.n)
            ]
        )

        dropoff_start_idx = pickup_start_idx + self.n
        self.dropoff_locations = np.array(
            [
                Point(int(lines[dropoff_start_idx + i].split()[0]),
                      int(lines[dropoff_start_idx + i].split()[1]))
                for i in range(self.n)
            ]
        )

    def _compute_distances(self):
        """Compute the distance matrix for all locations.

        Distance is Euclidean distance rounded up to next integer.

        Matrix layout:
            Index 0: depot
            Index 1 to n: pickup locations for requests 1 to n
            Index n+1 to 2n: drop-off locations for requests 1 to n

        Total size: (2n + 1) x (2n + 1)
        """
        all_locations = [self.depot_location] + list(self.pickup_locations) + list(self.dropoff_locations)
        size = len(all_locations)  # = 2n + 1
        self.distance_matrix = np.zeros((size, size))

        for i in range(size):
            for j in range(i + 1, size):
                dist = all_locations[i].distance_from(all_locations[j])
                self.distance_matrix[i][j] = dist
                self.distance_matrix[j][i] = dist

        assert np.array_equal(self.distance_matrix, self.distance_matrix.T)

    def compute_route_distance(self, route: list[int]) -> float:
        from_depot = self.depot_location.distance_from(self.pickup_locations[route[0]])
        to_depot = self.dropoff_locations[route[-1]].distance_from(self.depot_location)
        route_distance = 0
        for i in route[1:-1]:
            route_distance += self.pickup_locations[i].distance_from(self.dropoff_locations[i])
        return from_depot + route_distance + to_depot


    def __repr__(self):
        return (f"SCFPDPInstance(n_requests={self.n}, n_vehicles={self.n_K}, capacity={self.C}, "
                f"min_n_of_requests_to_serve={self.gamma}, fairness_weight={self.rho})")

if __name__ == '__main__':
    instance = SCFPDPInstance('10/test_instance_small.txt')
    print(instance)
    print(instance.distance_matrix)
    print(instance.compute_route_distance(list(range(20))))
