from __future__ import annotations
from typing import List, Tuple, Optional, Generator

from src.scfpdp.solution import SCFPDPSolution, Route


# ---------- INSERT MOVE ----------
class InsertMove:
    def __init__(self, route_from: int, route_to: int, request: int,
                 pickup_at: int, dropoff_offset: int):
        self.rf = route_from
        self.rt = route_to
        self.req = request
        self.pickup_at = pickup_at
        self.dropoff_offset = dropoff_offset

    def apply(self, sol: SCFPDPSolution) -> SCFPDPSolution:
        new_sol = sol.copy()
        route_from = new_sol.routes[self.rf]
        route_to = new_sol.routes[self.rt]

        # remove request pair from old route if inter-route insert
        if self.rf != self.rt:
            # remove pickup and dropoff
            route_from.route.remove(self.req)
            route_from.route.remove(self.req + sol.inst.n)
            route_from.served_requests.remove(self.req)
            route_from.recompute_route_distance()

        # insert into target route
        route_to.serve_request(self.req, self.pickup_at, self.dropoff_offset)
        new_sol.invalidate()
        return new_sol


# ---------- SWAP MOVE ----------
class SwapMove:
    def __init__(self, route_a: int, route_b: int, req_a: int, req_b: int):
        self.ra = route_a
        self.rb = route_b
        self.qa = req_a
        self.qb = req_b

    def apply(self, sol: SCFPDPSolution) -> SCFPDPSolution:
        new_sol = sol.copy()
        n = sol.inst.n

        rA = new_sol.routes[self.ra]
        rB = new_sol.routes[self.rb]

        # remove both old requests
        for r, q in [(rA, self.qa), (rB, self.qb)]:
            r.route.remove(q)
            r.route.remove(q + n)
            r.served_requests.remove(q)
            r.recompute_route_distance()

        # swap / insert cross
        rA.serve_request(self.qb, pickup_at=1, dropoff_distance_from_pickup=1)
        rB.serve_request(self.qa, pickup_at=1, dropoff_distance_from_pickup=1)

        new_sol.invalidate()
        return new_sol


# ---------- RELOCATE MOVE ----------
class RelocateMove:
    def __init__(self, route_from: int, route_to: int, request: int,
                 pickup_at: int, dropoff_offset: int):
        self.rf = route_from
        self.rt = route_to
        self.req = request
        self.pickup_at = pickup_at
        self.dropoff_offset = dropoff_offset

    def apply(self, sol: SCFPDPSolution) -> SCFPDPSolution:
        new_sol = sol.copy()
        route_from = new_sol.routes[self.rf]
        route_to = new_sol.routes[self.rt]

        # remove from original
        route_from.route.remove(self.req)
        route_from.route.remove(self.req + sol.inst.n)
        route_from.served_requests.remove(self.req)
        route_from.recompute_route_distance()

        # insert in new
        route_to.serve_request(self.req, self.pickup_at, self.dropoff_offset)
        new_sol.invalidate()
        return new_sol
