"""The DQN agent: holds the Q-network and chooses actions from observations."""

import random

import torch

from src.agent.q_network import QNetwork


class DQNAgent:
    """Agent that picks actions using a Q-network.

    Args:
        obs_size: length of the observation vector (env.observation_space.shape[0]).
        n_actions: number of actions (env.action_space.n).
        hidden_size: number of units in each hidden layer of the Q-network.
        device: where the network runs, e.g. "cpu", "cuda", or "mps".
    """

    def __init__(self, obs_size, n_actions, hidden_size=64, device="cpu"):
        self.n_actions = n_actions
        self.device = device
        self.q_net = QNetwork(obs_size, n_actions, hidden_size).to(device)

    def act(self, obs, epsilon):
        """Choose an action with epsilon-greedy.

        With probability epsilon, a random action (explore). Otherwise, the action with the
        highest score from the Q-network (exploit). Use epsilon=0 to always take the best action,
        e.g. for evaluation.

        Returns:
            The action as a Python int, which env.step() accepts.
        """
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            scores = self.q_net(obs_tensor)
        return int(scores.argmax())
