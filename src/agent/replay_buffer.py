"""Stores the agent's past steps so it can learn from random batches of them.

Consecutive steps in an episode are very similar to each other. Learning from them in order
makes the network forget older lessons, so instead every step is saved here and the agent
learns from a random mix of saved steps. Each step can also be reused many times.
"""

from collections import deque
import random
import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size store of (obs, action, reward, next_obs, done) steps.

    When the buffer is full, adding a new step drops the oldest one.
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        """Save one step."""
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        """Pick batch_size random steps (no repeats).

        Returns:
            Five tensors, each with batch_size rows:
            obs and next_obs (float32, one row per observation), actions (int64),
            rewards (float32), and dones (float32: 1.0 if the episode ended, else 0.0).
        """
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(obs), dtype=torch.float32),
            torch.as_tensor(np.array(actions), dtype=torch.int64),
            torch.as_tensor(np.array(rewards), dtype=torch.float32),
            torch.as_tensor(np.array(next_obs), dtype=torch.float32),
            torch.as_tensor(np.array(dones), dtype=torch.float32),
        )

    def __len__(self):
        """Number of steps stored (training waits until there are enough to sample)."""
        return len(self.buffer)
