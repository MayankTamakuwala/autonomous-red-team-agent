"""Tests for ReplayBuffer (src/agent/replay_buffer.py)."""

import numpy as np
import torch

from src.agent.replay_buffer import ReplayBuffer

OBS_SIZE = 4


def fill(buffer, n_steps):
    """Add n_steps fake steps. Step i has every observation value equal to i."""
    for i in range(n_steps):
        obs = np.full(OBS_SIZE, i, dtype=np.float32)
        buffer.add(obs, i % 3, float(i), obs + 1, i % 2 == 0)


def test_len_counts_stored_steps():
    """len() is the number of steps stored."""
    buffer = ReplayBuffer(capacity=10)
    assert len(buffer) == 0
    fill(buffer, 4)
    assert len(buffer) == 4


def test_sample_shapes_and_types():
    """sample() returns 5 tensors with batch_size rows and the expected dtypes."""
    buffer = ReplayBuffer(capacity=10)
    fill(buffer, 10)
    obs, actions, rewards, next_obs, dones = buffer.sample(batch_size=6)

    assert obs.shape == next_obs.shape == (6, OBS_SIZE)
    assert actions.shape == rewards.shape == dones.shape == (6,)
    assert obs.dtype == next_obs.dtype == rewards.dtype == dones.dtype == torch.float32
    assert actions.dtype == torch.int64
    assert set(dones.tolist()) <= {0.0, 1.0}


def test_oldest_steps_are_dropped_when_full():
    """When full, adding a step removes the oldest one."""
    buffer = ReplayBuffer(capacity=3)
    fill(buffer, 5)
    assert len(buffer) == 3

    obs, *_ = buffer.sample(batch_size=3)
    assert set(obs[:, 0].tolist()) == {2.0, 3.0, 4.0}
