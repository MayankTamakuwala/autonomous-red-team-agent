"""Smoke test: load a PenGym scenario in simulation mode and run a random agent.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --scenario tiny --episodes 20
    python scripts/smoke_test.py --scenario external/PenGym/database/scenarios/medium-multi-site.yml
"""

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

# Lets `python scripts/smoke_test.py` import from src/ (the repo root is not on the path otherwise)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env_setup import SCENARIO_DIR, make_env  # pylint: disable=wrong-import-position

DEFAULT_SCENARIO = SCENARIO_DIR / "medium-multi-site.yml"


def run_random_episode(env, max_steps, seed, verbose):
    """Run one episode with random actions. Returns (steps, total_reward, goal_reached)."""
    env.reset(seed=seed)
    total_reward = 0.0
    out = sys.stdout if verbose else io.StringIO()

    for step in range(1, max_steps + 1):
        action = int(env.action_space.sample())
        with contextlib.redirect_stdout(out):
            _, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if done or truncated:
            return step, total_reward, done

    return max_steps, total_reward, False


def main():
    """Parse arguments, run random episodes, and print a summary."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scenario",
        default=str(DEFAULT_SCENARIO),
        help="Path to a PenGym/NASim .yml scenario, or a NASim benchmark name",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="Show PenGym's per-action output")
    args = parser.parse_args()

    env = make_env(args.scenario, seed=args.seed)
    print(f"Scenario:     {args.scenario}")
    print(f"Actions:      {env.action_space.n}")
    print(f"Observation:  {env.observation_space.shape}")
    print()

    results = []
    for ep in range(args.episodes):
        steps, reward, success = run_random_episode(
            env, args.max_steps, args.seed + ep, args.verbose
        )
        results.append((steps, reward, success))
        print(
            f"Episode {ep + 1:>3}: steps={steps:>5}  reward={reward:>9.1f}  goal_reached={success}"
        )

    steps, rewards, successes = zip(*results)
    print()
    print(
        f"Success rate: {np.mean(successes):.0%}  |  "
        f"mean steps: {np.mean(steps):.1f}  |  mean reward: {np.mean(rewards):.1f}"
    )


if __name__ == "__main__":
    main()
