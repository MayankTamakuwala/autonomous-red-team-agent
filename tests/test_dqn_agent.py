"""Tests for QNetwork (src/agent/q_network.py) and DQNAgent (src/agent/dqn_agent.py)."""

import numpy as np
import torch

from src.agent.dqn_agent import DQNAgent
from src.agent.q_network import QNetwork
from src.env.common_view import N_FEATURES, CommonView
from src.env.env_setup import SCENARIO_DIR, make_env


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


def test_host_wise_agent_only_picks_actions_the_network_has():
    """On a CommonView network, act() never picks an action missing from action_mask."""
    env = CommonView(make_env(SCENARIO_DIR / "tiny.yml", seed=0))
    obs, _ = env.reset(seed=0)
    agent = DQNAgent(host_wise=True)

    random_actions = [agent.act(obs, 1.0, env.action_mask) for _ in range(200)]
    assert all(env.action_mask[a] for a in random_actions)
    assert agent.act(obs, 0.0, env.action_mask) in np.flatnonzero(env.action_mask)


def test_greedy_action_skips_missing_actions():
    """With epsilon=0, act() picks the highest score among the actions the network has."""
    torch.manual_seed(0)
    agent = DQNAgent(host_wise=True)
    obs = torch.rand(3 + 1, N_FEATURES)
    scores = agent.q_net(obs).detach()

    action_mask = np.ones(len(scores), dtype=bool)
    action_mask[int(scores.argmax())] = False  # the best action is missing from this network
    second_best = int(scores.argsort(descending=True)[1])
    assert agent.act(obs.numpy(), 0.0, action_mask) == second_best
