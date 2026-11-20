"""Train a DQN agent on one scenario.

Usage:
    python scripts/train.py --config configs/tiny.yml
    python scripts/train.py --config configs/tiny.yml --seed 1
    python scripts/train.py --config configs/tiny.yml --seed 1 --name tiny-test   # tiny-test_seed1

Writes:
    results/<run_name>/episodes.csv   one row per episode
    results/<run_name>/config.yml     the exact settings used for this run
    checkpoints/<run_name>.pt         the trained Q-network weights
"""

import argparse
import contextlib
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
# Lets `python scripts/train.py` import from src/ (the repo root is not on the path otherwise)
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from src.agent.dqn_agent import DQNAgent
from src.agent.replay_buffer import ReplayBuffer
from src.agent.schedules import linear_epsilon
from src.env.env_setup import make_env

# pylint: enable=wrong-import-position

# Used for any setting a config file leaves out
DEFAULTS = {
    "scenario": "tiny",
    "seed": 0,
    "total_steps": 20_000,
    "max_episode_steps": 1000,
    "hidden_size": 64,
    "lr": 1e-3,
    "gamma": 0.99,
    "batch_size": 32,
    "buffer_capacity": 50_000,
    "learning_starts": 1000,
    "train_every": 1,
    "target_update": 1000,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 10_000,
    "log_every": 20,
    "double_dqn": False,
}


def load_config(path, seed_override=None, run_name_override=None):
    """Read a YAML config, fill in missing settings from DEFAULTS, and name the run."""
    with open(path, encoding="utf-8") as f:
        cfg = {**DEFAULTS, **(yaml.safe_load(f) or {})}
    if seed_override is not None:
        cfg["seed"] = seed_override
    name = run_name_override or Path(path).stem
    cfg["run_name"] = f"{name}_seed{cfg['seed']}"
    return cfg


def set_seeds(seed):
    """Seed Python (epsilon-greedy), NumPy (NASim), and PyTorch (network weights)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# pylint: disable-next=too-many-arguments
def run_episodes(cfg, env, agent, buffer, writer, devnull, checkpoint):
    """Main loop: act, store, learn, and write one CSV row per finished episode.

    Saves the Q-network to `checkpoint` whenever the mean reward over the last
    log_every episodes reaches a new best, so the file always holds the best version.
    """
    best_mean = float("-inf")
    episode = 0
    obs, _ = env.reset(seed=cfg["seed"])
    ep_reward, ep_steps, ep_losses = 0.0, 0, []
    recent_rewards = []

    for step in range(1, cfg["total_steps"] + 1):
        # 1. Act: explore or exploit depending on how far into training we are
        epsilon = linear_epsilon(step, cfg["eps_start"], cfg["eps_end"], cfg["eps_decay_steps"])
        action = agent.act(obs, epsilon)
        with contextlib.redirect_stdout(devnull):  # PenGym prints a line per action
            next_obs, reward, done, truncated, _ = env.step(action)

        # 2. Store: only `done` (goal reached) counts as an ending. Running out of steps is not,
        #    because that state still had future value; the episode was just cut off.
        buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_reward += reward
        ep_steps += 1

        # 3. Learn: once the buffer has enough experiences, update from a random batch
        if len(buffer) >= cfg["learning_starts"] and step % cfg["train_every"] == 0:
            ep_losses.append(agent.learn(buffer.sample(cfg["batch_size"])))

        # 4. Refresh the target network every target_update steps
        if step % cfg["target_update"] == 0:
            agent.sync_target()

        # 5. Episode over: log it and start a new one
        if done or truncated or ep_steps >= cfg["max_episode_steps"]:
            mean_loss = round(float(np.mean(ep_losses)), 4) if ep_losses else ""
            writer.writerow(
                [episode, step, ep_steps, ep_reward, bool(done), round(epsilon, 4), mean_loss]
            )
            recent_rewards.append(ep_reward)
            episode += 1

            if episode % cfg["log_every"] == 0:
                mean_reward = float(np.mean(recent_rewards))
                print(
                    f"Episode {episode:>5} | step {step:>8} | epsilon {epsilon:.2f} | "
                    f"mean reward (last {len(recent_rewards)}) {mean_reward:>8.1f}"
                )
                # 6. Keep the best version of the network seen so far
                if mean_reward > best_mean:
                    best_mean = mean_reward
                    agent.save(checkpoint)
                    print(f"  New best ({best_mean:.1f}), saved checkpoint")
                recent_rewards = []

            obs, _ = env.reset(seed=cfg["seed"] + episode)
            ep_reward, ep_steps, ep_losses = 0.0, 0, []


def train(cfg):
    """Build everything from the config, train, and save results and the best network."""
    set_seeds(cfg["seed"])
    env = make_env(cfg["scenario"], seed=cfg["seed"])
    agent = DQNAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        hidden_size=cfg["hidden_size"],
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        double=cfg["double_dqn"],
    )
    buffer = ReplayBuffer(cfg["buffer_capacity"])

    run_dir = REPO_ROOT / "results" / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    checkpoint = REPO_ROOT / "checkpoints" / f"{cfg['run_name']}.pt"
    checkpoint.parent.mkdir(exist_ok=True)

    print(f"Training {cfg['run_name']} on {cfg['scenario']} for {cfg['total_steps']} steps")
    with (
        open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8") as csv_file,
        open(os.devnull, "w", encoding="utf-8") as devnull,
    ):
        writer = csv.writer(csv_file)
        writer.writerow(
            ["episode", "total_steps", "steps", "reward", "goal_reached", "epsilon", "mean_loss"]
        )
        run_episodes(cfg, env, agent, buffer, writer, devnull, checkpoint)

    print(f"Saved results to {run_dir} and best Q-network to {checkpoint}")


def main():
    """Parse arguments and train."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    parser.add_argument("--seed", type=int, default=None, help="Override the config's seed")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name (default: the config's filename). The seed is added to the end.",
    )
    args = parser.parse_args()
    train(load_config(args.config, args.seed, args.name))


if __name__ == "__main__":
    main()
