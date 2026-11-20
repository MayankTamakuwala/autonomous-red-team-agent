"""Tests for how scripts/train.py reads a config and names the run."""

from scripts.train import load_config


def test_run_name(tmp_path):
    """The run is named after the config file (or --name), always with the seed added."""
    config = tmp_path / "tiny.yml"
    config.write_text("seed: 0\n", encoding="utf-8")

    assert load_config(config)["run_name"] == "tiny_seed0"
    assert load_config(config, seed_override=3)["run_name"] == "tiny_seed3"
    assert load_config(config, run_name_override="exp")["run_name"] == "exp_seed0"
    assert load_config(config, 3, "exp")["run_name"] == "exp_seed3"
