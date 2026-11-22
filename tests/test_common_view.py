"""Tests for CommonView (src/env/common_view.py)."""

import pytest

from src.env.common_view import ACTION_TYPES, N_FEATURES, SERVICES, CommonView
from src.env.env_setup import SCENARIO_DIR, make_env


def common_env(name):
    """A PenGym scenario wrapped in MemoryWrapper and CommonView."""
    return CommonView(make_env(SCENARIO_DIR / f"{name}.yml", seed=0))


def action(env, host, kind):
    """The CommonView action for an action type on a host address."""
    return env.hosts.index(host) * len(ACTION_TYPES) + ACTION_TYPES.index(kind)


@pytest.mark.parametrize("name, n_hosts", [("tiny", 3), ("medium-multi-site", 16)])
def test_same_columns_for_every_network(name, n_hosts):
    """Every network gives one row per host plus the result row, with the same 15 columns."""
    obs, _ = common_env(name).reset(seed=0)
    assert obs.shape == (n_hosts + 1, N_FEATURES)


@pytest.mark.parametrize("name", ["tiny", "tiny-hard", "medium-multi-site"])
def test_action_mapping(name):
    """Each allowed (host, type) maps to the env action with that host and name, once each."""
    env = common_env(name)
    env_actions = env.unwrapped.action_space.actions
    allowed = [a for a in range(env.action_space.n) if env.action_mask[a]]
    assert len(allowed) == len(env_actions)

    for a in allowed:
        host, kind = divmod(a, len(ACTION_TYPES))
        env_action = env_actions[env.action_table.flat[a]]
        assert env_action.target == env.hosts[host]
        assert env_action.name == ACTION_TYPES[kind]


def test_copies_the_right_values():
    """After e_ssh on tiny's first host, its row shows user access, linux, and ssh only."""
    env = common_env("tiny")
    env.reset(seed=0)
    obs, *_ = env.step(action(env, (1, 0), "e_ssh"))

    row = obs[0]
    compromised, access = row[0], row[5]
    linux, services = row[6], row[7 : 7 + len(SERVICES)]
    assert compromised == 1 and access == 1
    assert linux == 1
    assert services.tolist() == [1, 0, 0, 0, 0]  # ssh, and no ftp/http/samba/smtp on tiny
    assert obs[-1, 0] == 1  # last action succeeded


def test_action_the_network_does_not_have():
    """Choosing an action type the network doesn't have is an error, not a silent no-op."""
    env = common_env("tiny")
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action(env, (1, 0), "e_samba"))
