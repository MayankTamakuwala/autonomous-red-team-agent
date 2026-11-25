"""Tests for HostQNetwork (src/agent/host_q_network.py)."""

import torch

from src.agent.host_q_network import HostQNetwork
from src.env.common_view import ACTION_TYPES, N_FEATURES, CommonView
from src.env.env_setup import SCENARIO_DIR, make_env


def test_scores_for_any_number_of_hosts():
    """The same network gives 12 scores per host for 3 hosts and for 16 hosts."""
    torch.manual_seed(0)
    net = HostQNetwork()
    assert net(torch.rand(3 + 1, N_FEATURES)).shape == (3 * len(ACTION_TYPES),)
    assert net(torch.rand(2, 16 + 1, N_FEATURES)).shape == (2, 16 * len(ACTION_TYPES))


def test_padding_does_not_change_scores():
    """A 3-host network padded to 16 hosts gets the same scores for its real hosts."""
    torch.manual_seed(0)
    net = HostQNetwork()
    small = torch.rand(3 + 1, N_FEATURES)

    padded = torch.zeros(16 + 1, N_FEATURES)
    padded[:3] = small[:-1]  # the 3 real hosts
    padded[-1] = small[-1]  # the last action's result stays in the last row
    host_mask = torch.zeros(16, dtype=torch.bool)
    host_mask[:3] = True

    alone = net(small)
    in_batch = net(padded.unsqueeze(0), host_mask.unsqueeze(0))[0]
    assert torch.allclose(alone, in_batch[: 3 * len(ACTION_TYPES)], atol=1e-6)


def test_reads_common_view_observations():
    """It scores a real CommonView observation, one score per CommonView action."""
    env = CommonView(make_env(SCENARIO_DIR / "medium-multi-site.yml", seed=0))
    obs, _ = env.reset(seed=0)
    scores = HostQNetwork()(torch.as_tensor(obs))
    assert scores.shape == (env.action_space.n,)
