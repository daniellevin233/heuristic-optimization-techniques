from typing import Iterable
from pymhlib.solution import Solution
from .neighbourhood import Neighbourhood


class SwapCustomers(Neighbourhood):
    """Neighbourhood: Swap two customers between routes."""

    def generate(self, solution: Solution) -> Iterable[Solution]:
        # TODO: implement swap logic
        ...


class RelocateCustomer(Neighbourhood):
    """Neighbourhood: Move a customer to a different position or route."""

    def generate(self, solution: Solution) -> Iterable[Solution]:
        # TODO: implement relocate logic
        ...


class TwoOptRoute(Neighbourhood):
    """Neighbourhood: Apply 2-opt on a single route only (local route improvement)."""

    def generate(self, solution: Solution) -> Iterable[Solution]:
        # TODO: implement 2-OPT moves
        ...
