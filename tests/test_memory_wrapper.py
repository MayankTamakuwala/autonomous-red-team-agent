"""Tests for MemoryWrapper (src/env/memory_wrapper.py), using NASim's tiny scenario.

On tiny, action 0 is a service scan and action 1 is an OS scan, both on host (1, 0),
which is reachable from the start.
"""

import numpy as np

SERVICE_SCAN = 0
OS_SCAN = 1


def test_observation_is_flat_and_matches_space(tiny_env):
    """Observations are 1D and match the declared observation space."""
    obs, _ = tiny_env.reset(seed=0)
    assert obs.shape == tiny_env.observation_space.shape == (56,)
    obs, *_ = tiny_env.step(SERVICE_SCAN)
    assert obs.shape == (56,)


def test_services_are_remembered_after_os_scan(tiny_env):
    """What a service scan revealed is still there after a later OS scan."""
    tiny_env.reset(seed=0)
    before_scan = tiny_env.knowledge[:-1].copy()

    tiny_env.step(SERVICE_SCAN)
    after_service_scan = tiny_env.knowledge[:-1].copy()
    assert (after_service_scan != before_scan).any(), "service scan should reveal something"

    tiny_env.step(OS_SCAN)
    revealed = after_service_scan != 0
    assert np.array_equal(tiny_env.knowledge[:-1][revealed], after_service_scan[revealed])


def test_reset_clears_memory(tiny_env):
    """A new episode starts with no knowledge from the previous one."""
    first_obs, _ = tiny_env.reset(seed=0)
    tiny_env.step(SERVICE_SCAN)
    tiny_env.step(OS_SCAN)
    new_obs, _ = tiny_env.reset(seed=0)
    assert np.array_equal(first_obs, new_obs)
