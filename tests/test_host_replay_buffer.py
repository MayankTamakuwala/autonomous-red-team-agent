"""Tests for HostReplayBuffer (src/agent/host_replay_buffer.py)."""

import numpy as np
import torch

from src.agent.host_q_network import HostQNetwork
from src.agent.host_replay_buffer import HostReplayBuffer
from src.env.common_view import ACTION_TYPES, N_FEATURES, CommonView
from src.env.env_setup import SCENARIO_DIR, make_env


def fake_step(n_hosts, value):
    """A step from a network with n_hosts hosts. Every host value is `value`, the result is 9."""
    obs = np.full((n_hosts + 1, N_FEATURES), value, dtype=np.float32)
    obs[-1] = 9
    action_mask = np.ones(n_hosts * len(ACTION_TYPES), dtype=bool)
    action_mask[1] = False  # one action this network doesn't have
    return obs, 0, 1.0, obs + 1, False, action_mask


def test_mixes_networks_of_different_sizes():
    """A batch with a 3-host and a 16-host step is padded to 16 hosts."""
    buffer = HostReplayBuffer(capacity=10)
    buffer.add(*fake_step(3, value=1))
    buffer.add(*fake_step(16, value=2))
    obs, actions, rewards, next_obs, dones, host_mask, action_mask = buffer.sample(batch_size=2)

    assert obs.shape == next_obs.shape == (2, 16 + 1, N_FEATURES)
    assert actions.shape == rewards.shape == dones.shape == (2,)
    assert host_mask.shape == (2, 16)
    assert action_mask.shape == (2, 16 * len(ACTION_TYPES))
    assert sorted(host_mask.sum(dim=1).tolist()) == [3, 16]


def test_padding_keeps_hosts_and_result_in_place():
    """The 3-host step keeps its hosts at the front, its result last, and adds empty hosts."""
    buffer = HostReplayBuffer(capacity=10)
    buffer.add(*fake_step(3, value=1))
    buffer.add(*fake_step(16, value=2))
    obs, *_, host_mask, action_mask = buffer.sample(batch_size=2)

    small = int(host_mask.sum(dim=1).argmin())  # which row is the 3-host step
    assert (obs[small, :3] == 1).all()  # real hosts
    assert (obs[small, 3:-1] == 0).all()  # added hosts are empty
    assert (obs[small, -1] == 9).all()  # result row stays last
    assert host_mask[small].tolist() == [True] * 3 + [False] * 13

    n_real = 3 * len(ACTION_TYPES)
    assert action_mask[small, :n_real].sum() == n_real - 1  # all but the missing action
    assert not action_mask[small, n_real:].any()  # no actions on added hosts


def test_batch_goes_into_host_q_network():
    """Steps from tiny and medium-multi-site make a batch HostQNetwork can score."""
    buffer = HostReplayBuffer(capacity=10)
    for name in ["tiny", "medium-multi-site"]:
        env = CommonView(make_env(SCENARIO_DIR / f"{name}.yml", seed=0))
        obs, _ = env.reset(seed=0)
        action = int(np.flatnonzero(env.action_mask)[0])
        next_obs, reward, done, _, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, done, env.action_mask)

    obs, *_, host_mask, action_mask = buffer.sample(batch_size=2)
    scores = HostQNetwork()(obs, host_mask)
    assert scores.shape == action_mask.shape == (2, 16 * len(ACTION_TYPES))
    assert scores.dtype == torch.float32
