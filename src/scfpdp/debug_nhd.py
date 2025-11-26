from src.scfpdp.instance import SCFPDPInstance
from src.scfpdp.solution import SCFPDPSolution
from src.scfpdp.neighbourhoods import InsertRequestNeighborhood

inst = SCFPDPInstance("../instances/10/test_instance_small.txt")
sol = SCFPDPSolution(inst)
sol.initialize(0)

ins = InsertRequestNeighborhood()
moves = list(ins.generate(sol))[:10]
print(moves)