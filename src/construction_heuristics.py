from src.scfpdp_instance import SCFPDPInstance
from src.scfpdp_solution import SCFPDPSolution


class GreedyConstructionHeuristic:
    def __init__(self, instance: SCFPDPInstance):
        self.instance = instance

    def construct(self) -> SCFPDPSolution:
        return SCFPDPSolution(self.instance)

class RandomizedConstructionHeuristic(GreedyConstructionHeuristic):
    pass

if __name__ == '__main__':
    SCFPDPSolution(inst=SCFPDPInstance('10/test_instance_small.txt'))