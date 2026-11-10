"""Shared test fixtures."""

import pytest

from src.env.env_setup import make_env


@pytest.fixture
def tiny_env():
    """NASim's tiny scenario (18 actions, observation of length 56), with memory."""
    return make_env("tiny", seed=0)
