from pymhlib.gvns import GVNS
from pymhlib.solution import Solution
from .local_improvement_vrp import VRPLocalImprovement


class VRPSearchBuilder:
    """Builder pattern for configuring a GVNS for VRP optimization."""

    def __init__(self, initial: Solution):
        self.initial = initial
        self.construct_methods = []
        self.local_improvements: list[VRPLocalImprovement] = []
        self.shaking_methods = []

    def with_construction(self, m):
        self.construct_methods.append(m)
        return self

    def with_local_improvement(self, li: VRPLocalImprovement):
        self.local_improvements.append(li)
        return self

    def with_shaking(self, m):
        self.shaking_methods.append(m)
        return self

    def build(self) -> GVNS:
        """Return a complete GVNS configured for VRP."""
        return GVNS(
            sol=self.initial,
            meths_ch=self.construct_methods,
            meths_li=self.local_improvements,
            meths_sh=self.shaking_methods,
            own_settings=None,
            consider_initial_sol=True
        )
