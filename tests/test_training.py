"""Tests for DQNAgent learning (src/agent/dqn_agent.py) and the epsilon schedule."""

import numpy as np
import torch

from src.agent.dqn_agent import DQNAgent
from src.agent.replay_buffer import ReplayBuffer
from src.agent.schedules import linear_epsilon

OBS_SIZE = 4
N_ACTIONS = 3


def make_batch(n_steps=16):
    """A batch of fake experiences with random observations."""
    buffer = ReplayBuffer(capacity=n_steps)
    rng = np.random.default_rng(0)
    for i in range(n_steps):
        obs = rng.random(OBS_SIZE).astype(np.float32)
        buffer.add(obs, i % N_ACTIONS, float(i % 2), obs + 1, i % 4 == 0)
    return buffer.sample(n_steps)


def weights(net):
    """A detached copy of every weight tensor in a network."""
    return [p.detach().clone() for p in net.parameters()]


def same_weights(before, after):
    """True if every weight tensor is unchanged."""
    return all(torch.equal(a, b) for a, b in zip(before, after))


def test_learn_updates_q_net_but_not_target_double():
    """learn() returns a float loss, changes the Q-network, and leaves the target network alone."""
    torch.manual_seed(0)
    agent = DQNAgent(OBS_SIZE, N_ACTIONS, double=True)
    q_before, target_before = weights(agent.q_net), weights(agent.target_net)

    loss = agent.learn(make_batch())

    assert isinstance(loss, float)
    assert not same_weights(q_before, weights(agent.q_net))
    assert same_weights(target_before, weights(agent.target_net))


def test_learn_updates_q_net_but_not_target():
    """learn() returns a float loss, changes the Q-network, and leaves the target network alone."""
    torch.manual_seed(0)
    agent = DQNAgent(OBS_SIZE, N_ACTIONS)
    q_before, target_before = weights(agent.q_net), weights(agent.target_net)

    loss = agent.learn(make_batch())

    assert isinstance(loss, float)
    assert not same_weights(q_before, weights(agent.q_net))
    assert same_weights(target_before, weights(agent.target_net))


def test_sync_target_copies_q_net():
    """After learning, sync_target() makes the target network match the Q-network."""
    torch.manual_seed(0)
    agent = DQNAgent(OBS_SIZE, N_ACTIONS)
    agent.learn(make_batch())
    agent.sync_target()
    assert same_weights(weights(agent.q_net), weights(agent.target_net))


def test_save_and_load_round_trip(tmp_path):
    """A saved Q-network loads back with identical weights, into both networks."""
    torch.manual_seed(0)
    agent = DQNAgent(OBS_SIZE, N_ACTIONS)
    agent.learn(make_batch())
    path = tmp_path / "agent.pt"
    agent.save(path)

    fresh = DQNAgent(OBS_SIZE, N_ACTIONS)
    fresh.load(path)
    assert same_weights(weights(agent.q_net), weights(fresh.q_net))
    assert same_weights(weights(agent.q_net), weights(fresh.target_net))


def test_linear_epsilon():
    """Epsilon starts at start, is halfway at half the decay, and stays at end afterward."""
    assert linear_epsilon(0, 1.0, 0.0, 100) == 1.0
    assert abs(linear_epsilon(50, 1.0, 0.0, 100) - 0.5) < 1e-9
    assert linear_epsilon(100, 1.0, 0.0, 100) == 0.0
    assert linear_epsilon(500, 1.0, 0.0, 100) == 0.0
