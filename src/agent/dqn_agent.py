"""The DQN agent: holds the Q-network and target network, chooses actions, and learns."""

import random

import torch
from torch import nn

from src.agent.q_network import QNetwork


class DQNAgent:
    """Agent that picks actions using a Q-network.

    Args:
        obs_size: length of the observation vector (env.observation_space.shape[0]).
        n_actions: number of actions (env.action_space.n).
        hidden_size: number of units in each hidden layer of the Q-network.
        device: where the network runs, e.g. "cpu", "cuda", or "mps".
    """

    def __init__(self, obs_size, n_actions, hidden_size=64, device="cpu", lr=0.001, gamma=0.99):
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma

        # Network learns and picks actions
        self.q_net = QNetwork(obs_size, n_actions, hidden_size).to(device)

        # Make target network (frozen copy of the Qnetwork) to compute targets
        self.target_net = QNetwork(obs_size, n_actions, hidden_size).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Adam Optimizer (applies update rule to the weights). Updates on Qnetwork weights
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

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

    def learn(self, batch):
        """Do one DQN update from a batch sampled from the replay buffer.

        Args:
            batch: the 5 tensors returned by ReplayBuffer.sample().

        Returns:
            The loss as a Python float (lower = predictions closer to targets).
        """
        obs, actions, rewards, next_obs, dones = (t.to(self.device) for t in batch)

        # This calls QNetworks foward method
        q_values = self.q_net(obs)

        predicted = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_value = self.target_net(next_obs).max(dim=1).values
            target = rewards + self.gamma * next_value * (1 - dones)

        # Huber loss so large errors don't cause huge updates.
        loss = nn.functional.smooth_l1_loss(predicted, target)

        # Clear old gradients, compute new ones, let Adam step.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        """Copy the Q-network's current weights into the target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        """Save the Q-network's weights to a file."""
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        """Load Q-network weights from a file, and copy them into the target network too."""
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.sync_target()
