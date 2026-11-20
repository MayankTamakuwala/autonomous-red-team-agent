"""Tests for scripts/evaluate.py: playing a greedy episode and reading training speed."""

import csv

from scripts.evaluate import learned_at_step, play_episode
from src.agent.dqn_agent import DQNAgent


def write_log(path, rewards):
    """A fake episodes.csv where every episode is 10 steps long."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_steps", "reward"])
        for i, reward in enumerate(rewards):
            writer.writerow([i, (i + 1) * 10, reward])


def test_learned_at_step(tmp_path):
    """Returns the training step where the windowed mean first reaches the threshold."""
    log = tmp_path / "episodes.csv"
    write_log(log, [0, 0, 200, 200, 0])
    # Window means: [0, 0] = 0, [0, 200] = 100, [200, 200] = 200 -> first reached at episode 3
    assert learned_at_step(log, threshold=150, window=2) == 40
    assert learned_at_step(log, threshold=250, window=2) is None


def test_play_episode(tiny_env):
    """An untrained agent's episode stops at max_steps and gives sensible metrics."""
    agent = DQNAgent(tiny_env.observation_space.shape[0], tiny_env.action_space.n)
    reward, steps, goal, redundant_ratio = play_episode(tiny_env, agent, 1, max_steps=20)

    assert isinstance(reward, float)
    assert 1 <= steps <= 20
    assert isinstance(goal, bool)
    assert 0.0 <= redundant_ratio <= 1.0
