from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from matplotlib import pyplot as plt


@dataclass
class PlotData:
    """Configuration for a single plot."""
    x_values: list[float]
    y_values: list[float]
    x_label: str
    y_label: str
    title: str
    color: str = 'blue'
    alpha: float = 0.7
    plot_type: str = 'bar'  # 'bar' or 'line'
    marker: str = 'o'
    linewidth: float = 2
    markersize: float = 8
    bar_width: float = 5
    bar_width_relative: float | None = None  # Relative bar width (0.0-1.0), overrides bar_width if set
    x_ticks: list[int] | None = None  # Optional custom x-tick values
    x_tick_labels: list[str] | None = None  # Optional custom x-tick labels
    y_scale: str = 'linear'  # 'linear' or 'log'


@dataclass
class ExperimentPlotConfig:
    """Configuration for experiment plots (typically 2 subplots side by side)."""
    algorithm_name: str
    plot_suptitle: str
    plot1: PlotData  # typically objective values
    plot2: PlotData  # typically runtimes
    save_plot: bool = False
    figsize: tuple[float, float] = (14, 5)


def find_project_root() -> Path:
    """
    Find the project root by walking up directories until finding one with 'instances' folder.

    Returns:
        Path to the project root directory
    """
    current = Path.cwd()
    while current != current.parent:
        if (current / "instances").exists():
            return current
        current = current.parent
    return Path.cwd()


def plot_experiment_results(config: ExperimentPlotConfig):
    """
    Unified plotting function for experiment results (2 subplots side by side).

    Args:
        config: ExperimentPlotConfig containing all plot parameters
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.figsize)
    fig.suptitle(config.algorithm_name, fontsize=14, fontweight='bold')

    # Calculate bar width for plot 1 if relative width is specified
    bar_width1 = config.plot1.bar_width
    if config.plot1.bar_width_relative is not None and len(config.plot1.x_values) > 1:
        x_range = max(config.plot1.x_values) - min(config.plot1.x_values)
        bar_width1 = x_range * config.plot1.bar_width_relative / len(config.plot1.x_values)

    # Plot 1
    if config.plot1.plot_type == 'bar':
        ax1.bar(config.plot1.x_values, config.plot1.y_values,
                color=config.plot1.color, alpha=config.plot1.alpha,
                label=config.algorithm_name, width=bar_width1)
    else:  # line
        ax1.plot(config.plot1.x_values, config.plot1.y_values,
                marker=config.plot1.marker, linewidth=config.plot1.linewidth,
                markersize=config.plot1.markersize, color=config.plot1.color,
                label=config.algorithm_name)

    ax1.set_xlabel(config.plot1.x_label, fontsize=12)
    ax1.set_ylabel(config.plot1.y_label, fontsize=12)
    ax1.set_title(config.plot1.title, fontsize=14, fontweight='bold')
    ax1.set_yscale(config.plot1.y_scale)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()

    # Set custom x-ticks for plot 1 if provided
    if config.plot1.x_ticks is not None:
        ax1.set_xticks(config.plot1.x_ticks)
    if config.plot1.x_tick_labels is not None:
        ax1.set_xticklabels(config.plot1.x_tick_labels, rotation=45, ha='right')

    # Calculate bar width for plot 2 if relative width is specified
    bar_width2 = config.plot2.bar_width
    if config.plot2.bar_width_relative is not None and len(config.plot2.x_values) > 1:
        x_range = max(config.plot2.x_values) - min(config.plot2.x_values)
        bar_width2 = x_range * config.plot2.bar_width_relative / len(config.plot2.x_values)

    # Plot 2
    if config.plot2.plot_type == 'bar':
        ax2.bar(config.plot2.x_values, config.plot2.y_values,
                color=config.plot2.color, alpha=config.plot2.alpha,
                label=config.algorithm_name, width=bar_width2)
    else:  # line
        ax2.plot(config.plot2.x_values, config.plot2.y_values,
                marker=config.plot2.marker, linewidth=config.plot2.linewidth,
                markersize=config.plot2.markersize, color=config.plot2.color,
                label=config.algorithm_name)

    ax2.set_xlabel(config.plot2.x_label, fontsize=12)
    ax2.set_ylabel(config.plot2.y_label, fontsize=12)
    ax2.set_title(config.plot2.title, fontsize=14, fontweight='bold')
    ax2.set_yscale(config.plot2.y_scale)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    # Set custom x-ticks for plot 2 if provided
    if config.plot2.x_ticks is not None:
        ax2.set_xticks(config.plot2.x_ticks)
    if config.plot2.x_tick_labels is not None:
        ax2.set_xticklabels(config.plot2.x_tick_labels, rotation=45, ha='right')

    plt.tight_layout()

    if config.save_plot:
        project_root = find_project_root()
        plots_dir = project_root / "plots"
        plots_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = plots_dir / f"{config.algorithm_name.lower().replace(' ', '_')}_results_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")

    plt.show()
