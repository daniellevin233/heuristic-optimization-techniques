"""
Bipartitioning Problem - Beam Search Implementation
Exercise 4: Heuristic Optimization Techniques WS 2025/26

Author: Daniel Levin (12433760)
"""

import random
import matplotlib.pyplot as plt
import time
from datetime import datetime


class BipartitioningInstance:
    """Represents a bipartitioning problem instance."""

    def __init__(self, numbers: list[int]):
        """
        Initialize bipartitioning instance.

        Args:
            numbers: list of positive integers to partition
        """
        self.numbers = numbers
        self.n = len(numbers)
        self.total_sum = sum(numbers)

    def evaluate_partial_partition(self, set_E: set[int], set_F: set[int]) -> int:
        e_values = [self.numbers[i] for i in set_E]
        f_values = [self.numbers[i] for i in set_F]
        return abs(sum(e_values) - sum(f_values))

    def evaluate_partition(self, set_E: set[int]) -> int:
        """
        Evaluate the imbalance of a partition.

        Args:
            set_E: set of indices that belong to partition E

        Returns:
            The absolute difference between partition sums
        """
        e_values = [self.numbers[i] for i in set_E]
        f_values = [self.numbers[i] for i in range(self.n) if i not in set_E]
        return abs(sum(e_values) - sum(f_values))

    def is_complete_solution(self, set_E: set[int], set_F: set[int]) -> bool:
        """
        Check if we have made decisions for all elements.

        Args:
            set_E: Current partial solution

        Returns:
            True if solution is complete
        """
        return len(set_E) + len(set_F) == self.n


class PartialSolution:
    """Represents a partial solution in the beam search."""

    def __init__(self, bipartitioning_instance: BipartitioningInstance, assigned_to_E: set[int], assigned_to_F: set[int],
                 unassigned: set[int]):
        """
        Initialize partial solution.

        Args:
            bipartitioning_instance: The bipartitioning instance
            assigned_to_E: Indices assigned to set E
            assigned_to_F: Indices assigned to set F
            unassigned: Indices not yet assigned
        """
        self.bipartitioning_instance = bipartitioning_instance
        self.assigned_to_E = assigned_to_E
        self.assigned_to_F = assigned_to_F
        self.unassigned = unassigned

    def __str__(self):
        return f"E:{str(self.assigned_to_E)}; F:{str(self.assigned_to_F)}; unassigned:{str(self.unassigned)}; objective:{str(self.evaluate_objective_value())}"

    def is_complete(self) -> bool:
        """Check if this is a complete solution."""
        return not self.unassigned

    def evaluate_g(self) -> int:
        return self.bipartitioning_instance.evaluate_partial_partition(self.assigned_to_E, self.assigned_to_F)

    def evaluate_h(self) -> int:
        # unassigned_nums = [self.bipartitioning_instance.numbers[i] for i in self.unassigned]
        # sorted_nums = sorted(unassigned_nums, reverse=True)
        # return reduce(lambda x, y: abs(x - y), sorted_nums, 0)
        return 0
    
    def evaluate_objective_value(self) -> int:
        return self.evaluate_g() + self.evaluate_h()
    
    def evaluate_complete_solution(self) -> int:
        if not self.is_complete():
            raise Exception('Solution is not complete, can\'t evaluate complete solution')
        return self.bipartitioning_instance.evaluate_partition(self.assigned_to_E)


class BeamSearch:
    """Beam Search solver for bipartitioning problem."""

    def __init__(self, instance: BipartitioningInstance, beam_width: int):
        """
        Initialize beam search.

        Args:
            instance: The bipartitioning problem instance
            beam_width: Maximum number of partial solutions to keep (�)
        """
        self.instance = instance
        self.beam_width = beam_width

    def generate_successors(self, partial: PartialSolution) -> list[PartialSolution]:
        """
        Generate all successor partial solutions from current partial solution.

        This is the branching step: for the next unassigned element,
        create two branches (assign to E or assign to F).

        Args:
            partial: Current partial solution

        Returns:
            list of successor partial solutions
        """
        # unassigned_index = next(iter(partial.unassigned))
        unassigned_index = random.choice(tuple(partial.unassigned))
        # print(f"Generating successors for {partial}")
        partial_1 = PartialSolution(self.instance, partial.assigned_to_E | {unassigned_index}, partial.assigned_to_F, partial.unassigned - {unassigned_index})
        # print(f"First successor: {partial_1}")
        partial_2 = PartialSolution(self.instance, partial.assigned_to_E, partial.assigned_to_F | {unassigned_index}, partial.unassigned - {unassigned_index})
        # print(f"First successor: {partial_2}")
        # print()
        return [partial_1, partial_2]

    def select_beam(self, candidates: list[PartialSolution]) -> list[PartialSolution]:
        """
        Select the best � partial solutions to keep in the beam.

        Args:
            candidates: All candidate partial solutions

        Returns:
            At most beam_width best partial solutions
        """
        candidate_to_objective_value = {c: c.evaluate_objective_value() for c in candidates}
        sorted_candidates = sorted(candidate_to_objective_value.items(), key=lambda item: item[1])
        return [c for c, _ in sorted_candidates[:self.beam_width]]

    def solve(self) -> tuple[set[int], int, float]:
        """
        Execute beam search to find a solution.

        Returns:
            tuple of (best_partition_E, imbalance, runtime_seconds)
        """
        start = time.time()

        current_beam_candidates = [PartialSolution(self.instance, set(), set(),
                                        set(range(self.instance.n)))]
        best_complete_solution = None

        while current_beam_candidates:
            next_step_candidates = []

            for partial in current_beam_candidates:
                if partial.is_complete():
                    if best_complete_solution is None or \
                            partial.evaluate_complete_solution() < best_complete_solution.evaluate_complete_solution():
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

        return best_complete_solution.assigned_to_E, best_complete_solution.evaluate_complete_solution(), end

    def print_result(self, result: tuple[set[int], int, float]):
        E = {self.instance.numbers[i] for i in result[0]}
        F = {self.instance.numbers[i] for i in range(self.instance.n) if i not in result[0]}
        print(f"E:{E} F:{F} Objective: {result[1]} Took {result[2]:.6f} seconds")


def generate_random_instance(n: int, min_value: int = 1, max_value: int = 10000) -> BipartitioningInstance:
    """
    Generate a random bipartitioning instance.

    Args:
        n: Number of elements
        min_value: Minimum value for random numbers
        max_value: Maximum value for random numbers

    Returns:
        Random bipartitioning instance
    """
    numbers = [random.randint(min_value, max_value) for _ in range(n)]
    return BipartitioningInstance(numbers)


def run_experiment(instance: BipartitioningInstance, beam_widths: list[int]) -> tuple[list[int], list[float]]:
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
        beam_search = BeamSearch(instance, beam_width)
        solution = beam_search.solve()
        solution_qualities.append(solution[1])
        runtimes.append(solution[2])
    return solution_qualities, runtimes


def plot_results(beam_widths: list[int], objective_value: list[int], runtimes: list[float], instance_size: int, save_plot: bool = False):
    """
    Plot solution quality and runtime vs beam width.

    Args:
        beam_widths: list of beam width values tested
        objective_value: list of solution qualities (imbalances)
        runtimes: list of runtimes in seconds
        instance_size: number of elements in the instance (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Solution Quality (Imbalance) vs Beam Width
    ax1.plot(beam_widths, objective_value, marker='o', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Beam Width (β)', fontsize=12)
    ax1.set_ylabel('Imbalance', fontsize=12)
    ax1.set_title(f'Solution Quality vs Beam Width (n={instance_size})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(beam_widths)

    # Plot 2: Runtime vs Beam Width
    ax2.plot(beam_widths, runtimes, marker='s', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Beam Width (β)', fontsize=12)
    ax2.set_ylabel('Runtime (seconds)', fontsize=12)
    ax2.set_title(f'Runtime vs Beam Width (n={instance_size})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(beam_widths)

    plt.tight_layout()

    if save_plot:
        # Generate filename with timestamp and parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_str = f"_n{instance_size}"
        beta_range = f"_b{min(beam_widths)}-{max(beam_widths)}"
        filename = f"bipartitioning_plots/beam_search_results{n_str}{beta_range}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        print(f"Plot saved as '{filename}'")

    plt.show()


def main(save_plot: bool):
    """Main execution: generate instance, run experiments, plot results."""

    # Configuration
    random.seed(40)  # For reproducibility
    instance_size = 1000  # Adjust to see significant differences
    # beam_widths = [1, 5, 10, 20, 30, 40, 50, 75, 100]  # Experiment with different � values
    beam_widths = [1] + list(range(0, 110, 10))[1:]

    print("=" * 60)
    print("Bipartitioning Problem - Beam Search")
    print("Exercise 4: Heuristic Optimization Techniques")
    print("=" * 60)
    print()

    # Generate random instance
    print(f"Generating random instance with n={instance_size}...")
    instance = generate_random_instance(instance_size)
    print(f"Numbers: {instance.numbers}")
    print(f"Total sum: {instance.total_sum}")
    print(f"Target sum per partition: {instance.total_sum / 2}")
    print()

    # Run experiments
    print("Running experiments with different beam widths...")
    qualities, runtimes = run_experiment(instance, beam_widths)

    # Display results
    print()
    print("Results:")
    print("-" * 60)
    print(f"{'Beam Width (�)':<20} {'Imbalance':<15} {'Runtime (s)':<15}")
    print("-" * 60)
    for bw, q, t in zip(beam_widths, qualities, runtimes):
        print(f"{bw:<20} {q:<15} {t:<15.6f}")
    print("-" * 60)
    print()

    # Plot results
    print("Generating plots...")
    plot_results(beam_widths, qualities, runtimes, instance_size, save_plot=save_plot)
    print("Done!")

def tmp():
    # print(plot_results([1,2,5,10], [1000, 995, 900, 850], [0.001, 0.01, 0.1, 1.0], 4))
    instance = BipartitioningInstance([1,3,6,10])
    # instance = BipartitioningInstance([10,6,3,1])
    beam_search = BeamSearch(instance, 1)
    result = beam_search.solve()
    beam_search.print_result(result)


if __name__ == "__main__":
    main(save_plot=True)
    # tmp()
