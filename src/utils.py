from datetime import datetime

from matplotlib import pyplot as plt


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
