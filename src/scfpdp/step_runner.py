# src/scfpdp/step_runner.py
from src.scfpdp.neighborhoods.insert import InsertNeighborhood
from src.scfpdp.neighborhoods.swap import SwapNeighborhood
from src.scfpdp.neighborhoods.relocate import RelocateNeighborhood

def fixed_neighborhood_order():
    """Fixed neighborhood priority: INSERT → SWAP → RELOCATE"""
    return [
        InsertNeighborhood(),
        SwapNeighborhood(),
        RelocateNeighborhood()
    ]
