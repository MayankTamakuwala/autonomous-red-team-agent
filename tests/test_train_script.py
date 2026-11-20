"""Tests for how scripts/train.py reads a config and names the run."""

from scripts.train import load_config


def test_run_name(tmp_path):
    """The run is named after the config file (or --name), then the algorithm, then the seed."""
    config = tmp_path / "tiny.yml"
    config.write_text("seed: 0\n", encoding="utf-8")

    assert load_config(config)["run_name"] == "tiny_dqn_seed0"
    assert load_config(config, seed_override=3)["run_name"] == "tiny_dqn_seed3"
    assert load_config(config, run_name_override="exp")["run_name"] == "exp_dqn_seed0"
    assert load_config(config, 3, "exp")["run_name"] == "exp_dqn_seed3"


def test_run_name_double_dqn(tmp_path):
    """A config with double_dqn: true gets "ddqn" in its run name."""
    config = tmp_path / "tiny.yml"
    config.write_text("seed: 0\ndouble_dqn: true\n", encoding="utf-8")

    assert load_config(config, seed_override=2)["run_name"] == "tiny_ddqn_seed2"
