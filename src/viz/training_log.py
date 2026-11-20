"""Reads the per-episode CSV that scripts/train.py writes (results/<run_name>/episodes.csv)."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrainingLog:
    """One training run's per-episode columns, as NumPy arrays."""

    path: Path
    episode: np.ndarray
    total_steps: np.ndarray  # environment steps taken in training when the episode ended
    steps: np.ndarray
    reward: np.ndarray
    goal_reached: np.ndarray

    @property
    def run_name(self):
        """The run's folder name, e.g. "tiny_seed0"."""
        return self.path.parent.name

    def __len__(self):
        return len(self.episode)


def read_log(path):
    """Read an episodes.csv file. Columns the charts don't use are ignored."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return TrainingLog(
        path=path,
        episode=np.array([int(r["episode"]) for r in rows], dtype=int),
        total_steps=np.array([int(r["total_steps"]) for r in rows], dtype=int),
        steps=np.array([int(r["steps"]) for r in rows], dtype=int),
        reward=np.array([float(r["reward"]) for r in rows], dtype=float),
        goal_reached=np.array([r["goal_reached"] == "True" for r in rows], dtype=bool),
    )


def moving_average(values, window):
    """Mean of each value and up to window - 1 values before it.

    The first few points average over fewer values, so the curve starts at the first episode.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    sums = np.cumsum(np.insert(values, 0, 0.0))
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - window)
    return (sums[ends] - sums[starts]) / (ends - starts)
