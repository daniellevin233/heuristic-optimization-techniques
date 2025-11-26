from src.scfpdp.neighborhoods_scfpdp import (
    InsertNeighborhood,
    SwapNeighborhood,
    RelocateNeighborhood
)

def fixed_neighborhoods():
    """Static order: INSERT → SWAP → RELOCATE."""
    return [
        InsertNeighborhood(),
        SwapNeighborhood(),
        RelocateNeighborhood()
    ]

