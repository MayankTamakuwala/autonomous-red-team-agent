"""Measure trained agents with exploration turned off, for every seed of one config.

Usage:
    python scripts/evaluate.py --config configs/tiny.yml
    python scripts/evaluate.py --config configs/small-linear.yml --seeds 0,1,2 --episodes 20
    python scripts/evaluate.py --config configs/tiny.yml --name tiny-test   # tiny-test_seed<N>

Reads, for each seed N (what scripts/train.py wrote):
    checkpoints/<name>_seed<N>.pt          the trained Q-network
    results/<name>_seed<N>/config.yml      the settings it was trained with
    results/<name>_seed<N>/episodes.csv    its training log, for how fast it learned

Writes:
    results/eval/<name>.csv                one row per seed
"""

import argparse
import contextlib
import csv
import io
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
# Lets `python scripts/evaluate.py` import from src/ (the repo root is not on the path otherwise)
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from src.agent.dqn_agent import DQNAgent
from src.env.env_setup import make_env

# pylint: enable=wrong-import-position

# Training episodes use reset seeds seed, seed+1, ... (tens of thousands at most),
# so evaluation episodes start far above that and never repeat a training episode.
EVAL_SEED_START = 1_000_000


def play_episode(env, agent, episode_seed, max_steps):
    """Play one episode, always taking the agent's best action (epsilon=0).

    Returns:
        (total reward, steps, goal reached, redundant-action ratio). An action is redundant
        when the agent's knowledge of the hosts (the MemoryWrapper grid without its last
        row) is exactly the same after it as before it.
    """
    obs, _ = env.reset(seed=episode_seed)
    total_reward, steps, redundant = 0.0, 0, 0
    done = truncated = False
    with contextlib.redirect_stdout(io.StringIO()):  # PenGym prints a line per action
        while not (done or truncated) and steps < max_steps:
            known_before = env.knowledge[:-1].copy()
            obs, reward, done, truncated, _ = env.step(agent.act(obs, epsilon=0.0))
            total_reward += reward
            steps += 1
            if np.array_equal(env.knowledge[:-1], known_before):
                redundant += 1
    return total_reward, steps, bool(done), redundant / steps


def learned_at_step(episodes_csv, threshold, window=20):
    """Training step at which the mean reward of the last `window` episodes first reached
    `threshold`, or None if it never did."""
    with open(episodes_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rewards = [float(row["reward"]) for row in rows]
    for end in range(window, len(rows) + 1):
        if np.mean(rewards[end - window : end]) >= threshold:
            return int(rows[end - 1]["total_steps"])
    return None


def evaluate_run(run_name, n_episodes, threshold):
    """Load one trained run, play n_episodes with it, and return its average metrics."""
    run_dir = REPO_ROOT / "results" / run_name
    with open(run_dir / "config.yml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = make_env(cfg["scenario"], seed=EVAL_SEED_START)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg["hidden_size"])
    agent.load(REPO_ROOT / "checkpoints" / f"{run_name}.pt")

    episodes = [
        play_episode(env, agent, EVAL_SEED_START + i, cfg["max_episode_steps"])
        for i in range(n_episodes)
    ]
    rewards, steps, goals, redundant = zip(*episodes)
    return {
        "run": run_name,
        "seed": cfg["seed"],
        "double_dqn": cfg.get("double_dqn", False),  # runs from before #21 were plain DQN
        "episodes": n_episodes,
        "reward": float(np.mean(rewards)),
        "steps": float(np.mean(steps)),
        "goal_rate": float(np.mean(goals)),
        "redundant_ratio": float(np.mean(redundant)),
        "learned_at_step": learned_at_step(run_dir / "episodes.csv", threshold),
    }


def mean_std(values, fmt):
    """'mean ± std' of the values, formatted with fmt; None values are left out and counted."""
    found = [v for v in values if v is not None]
    if not found:
        return "never"
    text = f"{fmt.format(np.mean(found))} ± {fmt.format(np.std(found))}"
    missing = len(values) - len(found)
    return f"{text} ({missing} never)" if missing else text


def print_table(rows):
    """Print one line per seed, then mean ± std across seeds."""
    print(f"{'run':<30}{'reward':>9}{'steps':>8}{'goal':>7}{'redundant':>11}{'learned at':>13}")
    for row in rows:
        learned = row["learned_at_step"]
        learned = "never" if learned is None else f"{learned:,}"
        print(
            f"{row['run']:<30}{row['reward']:>9.1f}{row['steps']:>8.1f}{row['goal_rate']:>7.0%}"
            f"{row['redundant_ratio']:>11.0%}{learned:>13}"
        )

    print(f"\nAcross {len(rows)} seeds (mean ± std):")
    print(f"  reward           {mean_std([r['reward'] for r in rows], '{:.1f}')}")
    print(f"  steps            {mean_std([r['steps'] for r in rows], '{:.1f}')}")
    print(f"  goal rate        {mean_std([r['goal_rate'] for r in rows], '{:.0%}')}")
    print(f"  redundant ratio  {mean_std([r['redundant_ratio'] for r in rows], '{:.0%}')}")
    print(f"  learned at step  {mean_std([r['learned_at_step'] for r in rows], '{:,.0f}')}")


def main():
    """Parse arguments, evaluate every seed, print the table, and save it."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="The config the runs were trained with")
    parser.add_argument(
        "--name", default=None, help="Run name used in training (default: the config's filename)"
    )
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Training seeds, comma-separated")
    parser.add_argument("--episodes", type=int, default=10, help="Evaluation episodes per seed")
    parser.add_argument(
        "--threshold",
        type=float,
        default=170,
        help="20-episode mean training reward that counts as learned (for 'learned at step')",
    )
    args = parser.parse_args()

    name = args.name or Path(args.config).stem
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = [evaluate_run(f"{name}_seed{seed}", args.episodes, args.threshold) for seed in seeds]
    print_table(rows)

    out = REPO_ROOT / "results" / "eval" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
