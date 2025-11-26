# src/scfpdp/neighbourhoods.py

from src.scfpdp.neighbourhood import Neighborhood
from src.scfpdp.solution import SCFPDPSolution
from src.scfpdp.instance import SCFPDPInstance
from src.scfpdp.moves import InsertMove


class InsertRequestNeighborhood(Neighborhood):
    """
    Tries to insert a non-served request into all vehicles' routes,
    exploring all feasible pickup & dropoff insertion positions.
    """

    def generate(self, solution: SCFPDPSolution):
        inst: SCFPDPInstance = solution.inst

        served = set()
        for r in solution.routes:
            served.update(r.served_requests)

        unserved = [i for i in range(inst.n) if i not in served]

        # Try to insert each unserved request into every route
        for request_id in unserved:
            demand = inst.demands[request_id]

            for route_idx, route in enumerate(solution.routes):
                # capacity check here avoids wasted moves generation
                if not route.can_take_request(request_id):
                    continue

                pickup_loc = request_id
                dropoff_loc = request_id + inst.n

                # valid pickup positions: between depots and before last depot
                for pickup_at in range(1, len(route.route)):  
                    # ensure location not duplicated
                    if pickup_loc in route.route:
                        continue

                    # dropoff must be to the right of pickup
                    for drop_offset in range(1, len(route.route) - pickup_at + 1):
                        drop_at = pickup_at + drop_offset
                        if drop_at > len(route.route):
                            continue

                        # ensure no duplicate dropoff
                        if dropoff_loc in route.route:
                            continue

                        # yield feasible move
                        yield InsertMove(
                            vehicle_index=route_idx,
                            request_id=request_id,
                            pickup_index=pickup_at,
                            dropoff_index=drop_at
                        )
