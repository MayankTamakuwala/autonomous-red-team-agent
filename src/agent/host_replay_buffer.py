"""Replay buffer for networks with different numbers of hosts.

The buffer remembers the experiences the agent collects while interacting
with the environment.

Because different networks can have different numbers of hosts, we need to
make their observations the same size before putting them into a training batch.

We do this by adding empty hosts to smaller networks. Two masks keep track
of which hosts and actions are actually real.
"""

from collections import deque
import random

import numpy as np
import torch

from src.env.common_view import ACTION_TYPES


def pad_hosts(obs, n_hosts):
    """Add empty hosts until the observation has exactly n_hosts hosts.

    We always keep the real hosts at the beginning of the observation.
    The final row, which contains the previous action's result, stays at the end.

    This allows observations from networks of different sizes to have the same
    shape, so they can be combined into a single batch.
    """

    # Create an empty observation with room for all hosts and the result row.
    padded = np.zeros((n_hosts + 1, obs.shape[1]), dtype=np.float32)

    # Copy the real hosts into the beginning.
    # Any remaining rows stay empty and represent padding.
    n_real_hosts = len(obs) - 1
    padded[:n_real_hosts] = obs[:-1]

    # Keep the previous action's result in the final row.
    padded[-1] = obs[-1]

    return padded


class HostReplayBuffer:
    """Store experiences collected from networks of different sizes.

    Each experience contains:
        - The observation before the action.
        - The action the agent selected.
        - The reward received.
        - The observation after the action.
        - Whether the episode ended.
        - A mask describing which actions were available.

    The buffer has a fixed capacity. When it fills up, adding a new experience
    automatically removes the oldest one.
    """

    def __init__(self, capacity):
        # Keep experiences in the order they were collected.
        # Once we reach capacity, the oldest experience is removed automatically.
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done, action_mask):
        """Remember one experience from the environment.

        We also save the action mask because different networks can have
        different available actions. We need the original mask later when
        training from this experience.
        """

        experience = (obs, action, reward, next_obs, done, action_mask)

        self.buffer.append(experience)

    def sample(self, batch_size):
        """Randomly select experiences and prepare them for training.

        Networks in the same batch can have different numbers of hosts.
        We pad their observations to match the largest network in the batch.

        Returns:
            obs:         Observations before the actions.
            actions:     Actions selected by the agent.
            rewards:     Rewards received.
            next_obs:    Observations after the actions.
            dones:       Whether each episode ended.
            host_mask:   Which hosts are real and which are padding.
            action_mask: Which actions are available in each network.
        """

        # Randomly pick experiences from memory.
        # Each experience contains an observation, action, reward, next observation,
        # done flag, and action mask.
        batch = random.sample(self.buffer, batch_size)

        # Separate the experiences into their individual components.
        # This lets us prepare each type of data independently.
        (observations, actions, rewards, next_observations, dones, action_masks) = zip(*batch)

        # Find the largest network in this batch.
        # The extra row in each observation contains the previous action's result,
        # so we subtract one to get the actual number of hosts.
        most_hosts = max(len(observation) - 1 for observation in observations)

        # Every observation in the batch will have most_hosts positions.
        #
        # Initially, all positions are False. We will mark the real hosts
        # and available actions as we go through each experience.
        host_mask = np.zeros((batch_size, most_hosts), dtype=bool)

        action_mask = np.zeros((batch_size, most_hosts * len(ACTION_TYPES)), dtype=bool)

        # Record which hosts and actions actually exist in each network.
        #
        # Smaller networks have fewer real hosts and actions, so their
        # remaining positions stay False and represent padding.
        for i, (observation, available_actions) in enumerate(zip(observations, action_masks)):

            # Mark all the real hosts.
            n_real_hosts = len(observation) - 1
            host_mask[i, :n_real_hosts] = True

            # Copy the original action mask.
            # Any additional positions remain False because they belong to padding.
            action_mask[i, : len(available_actions)] = available_actions

        # All observations must have the same shape before we can stack them.
        # Add empty hosts to smaller networks.
        padded_observations = [pad_hosts(observation, most_hosts) for observation in observations]

        padded_next_observations = [
            pad_hosts(observation, most_hosts) for observation in next_observations
        ]

        # Stack the padded observations into batches and convert everything
        # into PyTorch tensors with the appropriate data types.
        obs_tensor = torch.as_tensor(np.array(padded_observations), dtype=torch.float32)

        actions_tensor = torch.as_tensor(np.array(actions), dtype=torch.int64)

        rewards_tensor = torch.as_tensor(np.array(rewards), dtype=torch.float32)

        next_obs_tensor = torch.as_tensor(np.array(padded_next_observations), dtype=torch.float32)

        dones_tensor = torch.as_tensor(np.array(dones), dtype=torch.float32)

        host_mask_tensor = torch.as_tensor(host_mask)

        action_mask_tensor = torch.as_tensor(action_mask)

        # Return everything needed to calculate the Q-learning update.
        return (
            obs_tensor,
            actions_tensor,
            rewards_tensor,
            next_obs_tensor,
            dones_tensor,
            host_mask_tensor,
            action_mask_tensor,
        )

    def __len__(self):
        """Return how many experiences are currently stored in memory."""
        return len(self.buffer)
