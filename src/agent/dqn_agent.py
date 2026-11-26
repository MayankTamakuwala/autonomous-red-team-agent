"""The DQN agent: holds the Q-network and target network, chooses actions, and learns."""

import random

import numpy as np
import torch
from torch import nn

from src.agent.host_q_network import HostQNetwork
from src.agent.q_network import QNetwork


def mask_out(scores, action_mask):
    """Give actions that can't be picked a score of -inf, so max and argmax skip them.

    action_mask: True for actions that can be picked. None means every action can.
    """
    if action_mask is None:
        return scores
    return scores.masked_fill(~action_mask, float("-inf"))


class DQNAgent:
    """Agent that picks actions using a Q-network.

    Args:
        obs_size: length of the observation vector (env.observation_space.shape[0]).
        n_actions: number of actions (env.action_space.n).
        hidden_size: number of units in each hidden layer of the Q-network.
        device: where the network runs, e.g. "cpu", "cuda", or "mps".
        double: if True, use Double DQN targets: the Q-network picks the next action
        and the target network scores it. If False, plain DQN.
        host_wise: if True, use HostQNetwork, which works on any CommonView network, so
        obs_size and n_actions are not needed. If False, QNetwork for one network.
    """

    def __init__(
        self,
        obs_size=None,
        n_actions=None,
        hidden_size=64,
        device="cpu",
        lr=0.001,
        gamma=0.99,
        double=False,
        host_wise=False,
    ):
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.double = double

        # Network learns and picks actions, and the target network (a frozen copy of it)
        # computes targets
        if host_wise:
            self.q_net = HostQNetwork(hidden_size).to(device)
            self.target_net = HostQNetwork(hidden_size).to(device)
        else:
            self.q_net = QNetwork(obs_size, n_actions, hidden_size).to(device)
            self.target_net = QNetwork(obs_size, n_actions, hidden_size).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Adam Optimizer (applies update rule to the weights). Updates on Qnetwork weights
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def act(self, obs, epsilon, action_mask=None):
        """Choose an action with epsilon-greedy.

        With probability epsilon, a random action (explore). Otherwise, the action with the
        highest score from the Q-network (exploit). Use epsilon=0 to always take the best action,
        e.g. for evaluation.

        action_mask: True for actions this network has (CommonView's action_mask). Only those
        are picked. None means every action can be picked.

        Returns:
            The action as a Python int, which env.step() accepts.
        """
        if random.random() < epsilon:
            if action_mask is None:
                return random.randrange(self.n_actions)
            return random.choice(np.flatnonzero(action_mask).tolist())
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            scores = self.q_net(obs_tensor)
            if action_mask is not None:
                scores = mask_out(scores, torch.as_tensor(action_mask, device=self.device))
        return int(scores.argmax())

    @staticmethod
    def scores(net, obs, host_mask):
        """Run one of the Q-networks. The host-wise one also needs to know which hosts are real.

        host_mask is None for batches from ReplayBuffer, which only the per-network one learns from.
        """
        return net(obs) if host_mask is None else net(obs, host_mask)

    def learn(self, batch):
        """Do one DQN update from a batch sampled from the replay buffer.

        Args:
            batch: the 5 tensors returned by ReplayBuffer.sample(), or the 7 returned by
            HostReplayBuffer.sample() (which adds host_mask and action_mask).

        Returns:
            The loss as a Python float (lower = predictions closer to targets).
        """
        obs, actions, rewards, next_obs, dones, *masks = (t.to(self.device) for t in batch)
        host_mask, action_mask = masks if masks else (None, None)

        q_values = self.scores(self.q_net, obs, host_mask)

        predicted = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Only actions the network has can be the next action
            next_scores = mask_out(self.scores(self.target_net, next_obs, host_mask), action_mask)
            if self.double:
                # Q-network picks the best next action, target network scores it
                q_next = mask_out(self.scores(self.q_net, next_obs, host_mask), action_mask)
                best_next = q_next.argmax(dim=1, keepdim=True)
                next_value = next_scores.gather(1, best_next).squeeze(1)
            else:
                next_value = next_scores.max(dim=1).values
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
