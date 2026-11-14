"""Tests for QNetwork (src/agent/q_network.py) and DQNAgent (src/agent/dqn_agent.py)."""

import torch

from src.agent.dqn_agent import DQNAgent
from src.agent.q_network import QNetwork


def test_q_network_gives_one_score_per_action():
    """One observation gives n_actions scores; a batch gives one row of scores per observation."""
    net = QNetwork(obs_size=56, n_actions=18)
    assert net(torch.zeros(56)).shape == (18,)
    assert net(torch.zeros(5, 56)).shape == (5, 18)


def test_random_actions_are_valid_ints():
    """With epsilon=1, act() returns varied random actions, all Python ints in range."""
    agent = DQNAgent(obs_size=56, n_actions=18)
    obs = torch.zeros(56).numpy()
    actions = [agent.act(obs, epsilon=1.0) for _ in range(200)]

    assert all(isinstance(a, int) and 0 <= a < 18 for a in actions)
    assert len(set(actions)) > 10


def test_greedy_action_is_the_highest_score():
    """With epsilon=0, act() always returns the action the network scores highest."""
    agent = DQNAgent(obs_size=56, n_actions=18)
    obs = torch.rand(56).numpy()
    best = int(agent.q_net(torch.as_tensor(obs)).argmax())

    assert all(agent.act(obs, epsilon=0.0) == best for _ in range(20))


def test_env_accepts_agent_actions(tiny_env):
    """Actions chosen by the agent can be passed straight to env.step()."""
    obs, _ = tiny_env.reset(seed=0)
    agent = DQNAgent(tiny_env.observation_space.shape[0], tiny_env.action_space.n)
    for epsilon in (1.0, 0.0):
        obs, *_ = tiny_env.step(agent.act(obs, epsilon))
