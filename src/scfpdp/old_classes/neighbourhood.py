from __future__ import annotations
from typing import Iterable, Protocol
from src.scfpdp.solution import SCFPDPSolution


class Move(Protocol):
    """Represents a mutable neighborhood move."""

    def is_feasible(self, solution: SCFPDPSolution) -> bool:
        """Check if applying this move keeps feasibility constraints."""
        ...

    def delta(self, solution: SCFPDPSolution) -> float:
        """
        Compute objective delta if the move is applied.
        NOTE: solution must NOT be modified here.
        """
        ...

    def apply(self, solution: SCFPDPSolution) -> None:
        """Mutates the solution in-place following delta decisions."""
        ...


class INeighborhood(Protocol):
    """Provides neighbor moves for local search exploration."""

    def generate(self, solution: SCFPDPSolution) -> Iterable[Move]:
        """
        Lazily produce valid candidate moves on demand.
        Moves MUST NOT be applied here.
        """
        ...
