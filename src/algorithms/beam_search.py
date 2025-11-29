import random
import time

from src.scfpdp.construction_heuristics import RandomizedConstructionHeuristic
from src.scfpdp.instance import SCFPDPInstance
from src.scfpdp.solution import SCFPDPSolution


class SCFPDPBeamSearch:
    def __init__(self, instance: SCFPDPInstance, beam_width: int, branching_factor: int = 2):
        self.instance: SCFPDPInstance = instance
        self.solution = SCFPDPSolution(instance)
        self.branching_factor = branching_factor
        self.beam_width = beam_width

    def pick_next_route_to_branch_from(self, partial_solution: SCFPDPSolution) -> int:
        # pick the route for successors generation with minimal current distance
        # return np.argmin([r.distance for r in partial_solution.routes])
        # pick a route for successors generation randomly
        return random.choice(range(len(partial_solution.routes)))

    def generate_successors(self, partial_solution: SCFPDPSolution) -> list[SCFPDPSolution]:
        route_for_search_idx = self.pick_next_route_to_branch_from(partial_solution)
        route_for_search = partial_solution.routes[route_for_search_idx]

        # get the next requests to serve to construct the candidate list for the beam search
        next_requests = RandomizedConstructionHeuristic(partial_solution, self.branching_factor).get_rcl_of_next_requests(
            route_for_search,
            partial_solution.get_all_served_requests(),
            len(route_for_search)
        )

        # construct the successors solutions instances
        successors = []
        for next_request in next_requests:
            new_partial_solution = partial_solution.copy()
            new_partial_solution.routes[route_for_search_idx].serve_request(
                next_request,
                RandomizedConstructionHeuristic.select_insertion_position(route_for_search),
                RandomizedConstructionHeuristic.select_dropoff_distance(route_for_search)
            )
            successors.append(new_partial_solution)
        return successors

    def select_beam(self, candidates: list[SCFPDPSolution]) -> list[SCFPDPSolution]:
        """
        Select the best beta partial solutions to keep in the beam.
        """
        candidate_to_objective_value = {c: c.calc_objective() for c in candidates}
        sorted_candidates = sorted(candidate_to_objective_value.items(), key=lambda item: item[1])
        return [c for c, _ in sorted_candidates[:self.beam_width]]

    def solve(self) -> tuple[SCFPDPSolution, float]:
        """
        Execute beam search to find a solution.
        """
        start = time.time()

        current_beam_candidates: list[SCFPDPSolution] = [SCFPDPSolution(self.instance)]
        best_complete_solution = None

        while current_beam_candidates:
            next_step_candidates: list[SCFPDPSolution] = []

            for partial in current_beam_candidates:
                if partial.is_complete():
                    if best_complete_solution is None or \
                            partial.calc_objective() < best_complete_solution.calc_objective():
                        best_complete_solution = partial
                else:
                    extensions = self.generate_successors(partial)
                    next_step_candidates.extend(extensions)

            if next_step_candidates:
                current_beam_candidates = self.select_beam(next_step_candidates)
            else:
                break

        end = time.time() - start

        if best_complete_solution is None:
            raise Exception("No complete solution found!")

        return best_complete_solution, end

    def print_result(self, result: tuple[SCFPDPSolution, float]):
        print(f"Generated solution:\n {result} \n\n Took {result[1]:.6f} seconds")


def run_experiment(instance: SCFPDPInstance, beam_widths: list[int]) -> tuple[list[int], list[float]]:
    """
    Run beam search with different beam widths and collect results.

    Args:
        instance: The problem instance to solve
        beam_widths: list of beam width values to test

    Returns:
        tuple of (solution_qualities, runtimes)
    """
    solution_qualities, runtimes = [], []
    for beam_width in beam_widths:
        beam_search = SCFPDPBeamSearch(instance, beam_width)
        solution, runtime = beam_search.solve()
        solution_qualities.append(solution.calc_objective())
        runtimes.append(runtime)
    return solution_qualities, runtimes


def main():
    test_instance = SCFPDPInstance('10/test_instance_small.txt')
    competition_instance = SCFPDPInstance('100/competition/instance61_nreq100_nveh2_gamma91.txt')

    instance = competition_instance

    beta = 10
    branching_factor = 4
    beam_search = SCFPDPBeamSearch(instance, beta, branching_factor)
    result = beam_search.solve()
    beam_search.print_result(result)
    # result[0].write_to_file(f"beam_search_{beta}_{branching_factor}")


if __name__ == "__main__":
    main()
