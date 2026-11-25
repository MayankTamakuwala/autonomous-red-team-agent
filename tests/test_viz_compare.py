"""Tests for overlaying several runs' training curves (src/viz/compare.py)."""

import numpy as np

from scripts.compare_runs import config_runs, parse_run
from src.viz.compare import comparison_figure, group_curve, group_runs, save_comparison_figure
from src.viz.training_log import read_log
from tests.test_viz_training_log import write_log


def make_logs(tmp_path, names, n=30):
    """One fake training log per name, each in its own run folder."""
    return [read_log(write_log(tmp_path / name / "episodes.csv", n=n)) for name in names]


def test_group_runs_keeps_label_order(tmp_path):
    """Runs with the same label form one group; groups keep the order labels first appear in."""
    a0, a1, b0 = make_logs(tmp_path, ["a0", "a1", "b0"])
    groups = group_runs([("shaped", a0), ("sparse", b0), ("shaped", a1)])
    assert [g.label for g in groups] == ["shaped", "sparse"]
    assert groups[0].logs == (a0, a1) and groups[0].legend == "shaped (mean of 2)"
    assert groups[1].legend == "sparse"


def test_single_run_curve_is_its_moving_average(tmp_path):
    """One run is plotted at its own episode ends, with no spread."""
    (log,) = make_logs(tmp_path, ["only"])
    steps, mean, std = group_curve((log,), "reward", window=5)
    np.testing.assert_array_equal(steps, log.total_steps)
    assert len(mean) == len(log) and not std.any()


def test_identical_runs_average_with_zero_spread(tmp_path):
    """Averaging copies of a run gives the same curve back and a zero-width band."""
    logs = make_logs(tmp_path, ["s0", "s1", "s2"])
    steps, mean, std = group_curve(tuple(logs), "steps", window=5)
    assert steps[0] == logs[0].total_steps[0] and steps[-1] == logs[0].total_steps[-1]
    np.testing.assert_allclose(std, 0, atol=1e-9)
    assert mean[0] == logs[0].steps[0]


def test_group_curve_covers_only_steps_every_run_reached(tmp_path):
    """A shorter run limits the shared axis to the steps both runs reached."""
    short = read_log(write_log(tmp_path / "short" / "episodes.csv", n=10))
    long_ = read_log(write_log(tmp_path / "long" / "episodes.csv", n=30))
    steps, _, _ = group_curve((short, long_), "reward", window=5)
    assert steps[-1] == short.total_steps[-1]


def test_comparison_figure_draws_one_line_per_group(tmp_path):
    """Each group adds one line to both charts, and the figure saves as an image."""
    a0, a1, b0 = make_logs(tmp_path, ["a0", "a1", "b0"])
    labeled = [("DQN", a0), ("DQN", a1), ("Double DQN", b0)]
    fig = comparison_figure(labeled, size=(6, 4), dpi=50)
    assert all(len(ax.get_lines()) == 2 for ax in fig.get_axes())
    out = tmp_path / "comparison.png"
    save_comparison_figure(labeled, out)
    assert out.stat().st_size > 0


def test_parse_run():
    """RUN=LABEL splits into both parts; a bare RUN has no label."""
    assert parse_run("tiny_seed0=sparse") == ("tiny_seed0", "sparse")
    assert parse_run("tiny_seed0") == ("tiny_seed0", None)


def test_config_runs_groups_by_algorithm(tmp_path):
    """Runs of one config are found and grouped by the part between config and seed; runs of
    configs that only share a prefix (tiny-small) and old names without a group are skipped."""
    for name in [
        "tiny_ddqn_seed10",
        "tiny_ddqn_seed2",
        "tiny_dqn_seed0",
        "tiny-small_ddqn_seed0",
        "tiny_seed0",
    ]:
        write_log(tmp_path / name / "episodes.csv", n=3)
    (tmp_path / "tiny_ddqn_seed5").mkdir()  # no episodes.csv yet: skipped
    found = config_runs("configs/tiny.yml", tmp_path)
    assert [(group, folder.name) for group, folder in found] == [
        ("ddqn", "tiny_ddqn_seed2"),
        ("ddqn", "tiny_ddqn_seed10"),
        ("dqn", "tiny_dqn_seed0"),
    ]
    assert not config_runs("small-linear", tmp_path)
