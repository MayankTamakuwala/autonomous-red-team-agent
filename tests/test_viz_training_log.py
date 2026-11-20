"""Tests for reading training logs and drawing the charts (training_log.py, charts.py)."""

import csv

import numpy as np

from src.viz.charts import figure_pixels, save_training_figure, training_figure
from src.viz.training_log import moving_average, read_log

COLUMNS = ["episode", "total_steps", "steps", "reward", "goal_reached", "epsilon", "mean_loss"]


def write_log(path, n=30):
    """A CSV shaped like the one scripts/train.py writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        total = 0
        for episode in range(n):
            steps = max(6, 100 - 3 * episode)
            total += steps
            writer.writerow([episode, total, steps, 200 - steps, episode % 7 != 3, 0.5, ""])
    return path


def test_read_log(tmp_path):
    """Columns come back as arrays, and the run name is the folder's name."""
    log = read_log(write_log(tmp_path / "tiny_seed0" / "episodes.csv"))
    assert len(log) == 30 and log.run_name == "tiny_seed0"
    assert log.steps[0] == 100 and log.reward[0] == 100.0
    assert log.total_steps[-1] == log.steps.sum()
    assert log.goal_reached.sum() == 26


def test_moving_average():
    """Each point averages up to `window` values ending at it."""
    np.testing.assert_allclose(moving_average([2, 4, 6, 8], 2), [2, 3, 5, 7])
    np.testing.assert_allclose(moving_average([1, 2, 3], 10), [1, 1.5, 2])
    assert len(moving_average([], 5)) == 0


def test_charts_render(tmp_path):
    """The figure draws to pixels of the requested size and saves as an image."""
    log = read_log(write_log(tmp_path / "run" / "episodes.csv"))
    pixels, (width, height) = figure_pixels(training_figure(log, size=(6, 4), dpi=50))
    assert (width, height) == (300, 200) and len(pixels) == width * height * 4
    out = tmp_path / "charts.png"
    save_training_figure(log, out)
    assert out.stat().st_size > 0


def test_charts_plot_against_training_steps(tmp_path):
    """Curves use the total_steps column for x, so short solved episodes don't stretch the axis."""
    log = read_log(write_log(tmp_path / "run" / "episodes.csv"))
    fig = training_figure(log, size=(6, 4), dpi=50)
    for ax in fig.get_axes():
        for line in ax.get_lines():
            np.testing.assert_array_equal(line.get_xdata(), log.total_steps)
    assert fig.get_axes()[-1].get_xlabel() == "Training steps"
