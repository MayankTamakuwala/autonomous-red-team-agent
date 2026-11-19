"""Training charts drawn with matplotlib: total reward and steps per episode.

The GUI draws the figure off-screen and shows the pixels; the same figure can be saved as a
PNG for reports. pyplot is never imported, so matplotlib opens no windows of its own.
"""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.viz.training_log import moving_average

# Matches the GUI's dark theme (see theme.py)
BACKGROUND = "#0b1020"
PANEL = "#111833"
TEXT = "#c9d3ee"
MUTED = "#6c7799"
GRID = "#1f2947"
REWARD = "#ffb547"
STEPS = "#4cc9f0"
MISSED = "#ff5c7a"


def training_figure(log, size=(9.0, 6.0), dpi=100, window=20):
    """Figure with reward per episode on top and steps per episode below.

    Raw values are faint; the moving average over `window` episodes is drawn bold. Episodes
    that did not reach the goal are marked along the bottom of the steps chart.
    """
    fig = Figure(figsize=size, dpi=dpi, facecolor=BACKGROUND)
    reward_ax, steps_ax = fig.subplots(2, 1, sharex=True)
    episodes = log.episode

    _plot_series(reward_ax, episodes, log.reward, REWARD, window, "Total reward per episode")
    _plot_series(steps_ax, episodes, log.steps, STEPS, window, "Steps per episode")
    missed = ~log.goal_reached
    if missed.any():
        steps_ax.scatter(
            episodes[missed],
            np.full(missed.sum(), steps_ax.get_ylim()[0]),
            marker="|",
            color=MISSED,
            s=60,
            label="goal not reached",
            zorder=3,
        )
    steps_ax.set_xlabel("Episode", color=MUTED)
    for ax in (reward_ax, steps_ax):
        ax.legend(loc="best", fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle(
        f"{log.run_name}  ·  goal reached in {log.goal_reached.sum()} of {len(log)} episodes",
        color=TEXT,
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def figure_pixels(fig):
    """Draw a figure off-screen. Returns (RGBA bytes, (width, height))."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    return bytes(canvas.buffer_rgba()), (width, height)


def save_training_figure(log, path, window=20):
    """Save the training charts for a run as an image file (format from the extension)."""
    fig = training_figure(log, window=window)
    fig.savefig(path, facecolor=fig.get_facecolor())


def _plot_series(ax, episodes, values, color, window, title):
    ax.set_facecolor(PANEL)
    ax.plot(episodes, values, color=color, alpha=0.25, linewidth=0.8, label="per episode")
    ax.plot(
        episodes,
        moving_average(values, window),
        color=color,
        linewidth=2.2,
        label=f"{window}-episode average",
    )
    ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    ax.grid(color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
