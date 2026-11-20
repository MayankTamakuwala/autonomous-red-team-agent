"""Overlay the training curves of several runs on one chart.

Generic on purpose: each run gets a label, and the chart doesn't care what the labels mean
(seeds, DQN vs Double DQN, sparse vs shaped reward). Runs that share a label are one group:
their curves are averaged and drawn with a band of one standard deviation across the runs.

Like charts.py, the x-axis is training steps, so runs of different lengths line up.
"""

from dataclasses import dataclass
from itertools import cycle

import numpy as np
from matplotlib.figure import Figure

from src.viz.charts import BACKGROUND, GRID, MUTED, PANEL, STEPS_LABEL, TEXT, style_axes
from src.viz.training_log import moving_average

GROUP_COLORS = ["#ffb547", "#4cc9f0", "#b388ff", "#06d6a0", "#ff5c7a", "#f4a3d7", "#9be564"]
GRID_POINTS = 400  # points on the shared step axis that a group's runs are averaged on


@dataclass(frozen=True)
class RunGroup:
    """Runs that share a label."""

    label: str
    logs: tuple

    @property
    def legend(self):
        """The label, plus how many runs were averaged when there is more than one."""
        return self.label if len(self.logs) == 1 else f"{self.label} (mean of {len(self.logs)})"


def group_runs(labeled_logs):
    """Group (label, TrainingLog) pairs by label, in the order labels first appear."""
    groups = {}
    for label, log in labeled_logs:
        groups.setdefault(label, []).append(log)
    return [RunGroup(label, tuple(logs)) for label, logs in groups.items()]


def group_curve(logs, column, window):
    """A group's moving-average curve for one episodes.csv column, against training steps.

    Returns (steps, mean, std). A single run is returned as is (std of zero). Several runs end
    episodes at different steps, so each run's curve is interpolated onto a shared grid that
    covers only the steps every run reached, then averaged.
    """
    curves = [(log.total_steps, moving_average(getattr(log, column), window)) for log in logs]
    if len(curves) == 1:
        steps, curve = curves[0]
        return steps, curve, np.zeros_like(curve)
    start = max(steps[0] for steps, _ in curves)
    end = min(steps[-1] for steps, _ in curves)
    grid = np.linspace(start, end, GRID_POINTS)
    stacked = np.array([np.interp(grid, steps, curve) for steps, curve in curves])
    return grid, stacked.mean(axis=0), stacked.std(axis=0)


def comparison_figure(labeled_logs, size=(9.0, 6.0), dpi=100, window=20):
    """Figure overlaying reward (top) and steps (bottom) per episode for each group of runs.

    Args:
        labeled_logs: (label, TrainingLog) pairs. Pairs with the same label are averaged.
        window: episodes in each run's moving average.
    """
    groups = group_runs(labeled_logs)
    fig = Figure(figsize=size, dpi=dpi, facecolor=BACKGROUND)
    reward_ax, steps_ax = fig.subplots(2, 1, sharex=True)
    style_axes(reward_ax, "Total reward per episode")
    style_axes(steps_ax, "Steps per episode")

    for color, group in zip(cycle(GROUP_COLORS), groups):
        for ax, column in ((reward_ax, "reward"), (steps_ax, "steps")):
            steps, mean, std = group_curve(group.logs, column, window)
            ax.plot(steps, mean, color=color, linewidth=2.0, label=group.legend)
            if len(group.logs) > 1:
                ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18, lw=0)

    steps_ax.set_xlabel(STEPS_LABEL, color=MUTED)
    for ax in (reward_ax, steps_ax):
        ax.legend(loc="best", fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    band = ", band = ±1 std across runs" if any(len(g.logs) > 1 for g in groups) else ""
    fig.suptitle(f"{window}-episode moving average{band}", color=TEXT, fontsize=11)
    fig.tight_layout()
    return fig


def save_comparison_figure(labeled_logs, path, window=20):
    """Save the comparison chart as an image file (format from the extension)."""
    fig = comparison_figure(labeled_logs, window=window)
    fig.savefig(path, facecolor=fig.get_facecolor())
