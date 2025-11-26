from copy import deepcopy


class InsertMove:
    """
    Represents inserting request_id's pickup and dropoff into a specific vehicle route.
    """

    def __init__(self, vehicle_index: int, request_id: int, pickup_index: int, dropoff_index: int):
        self.vehicle_index = vehicle_index
        self.request_id = request_id
        self.pickup_index = pickup_index
        self.dropoff_index = dropoff_index

    def __repr__(self):
        return f"InsertMove(V={self.vehicle_index}, Req={self.request_id}, P={self.pickup_index}, D={self.dropoff_index})"

    # -------- evaluate impact using a TEMP copy --------
    def delta(self, solution):
        temp = solution.copy()
        self.apply(temp)
        new_obj = temp.calc_objective()
        return new_obj - solution.calc_objective()

    # -------- mutate the REAL solution (only when accepted) --------
    def apply(self, solution):
        route = solution.routes[self.vehicle_index]
        # use existing logic from Route class
        # dropoff_index shifts after pickup inserted!
        route.serve_request(self.request_id, self.pickup_index, self.dropoff_index - self.pickup_index)
        solution.invalidate()
