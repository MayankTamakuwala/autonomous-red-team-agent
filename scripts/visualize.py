"""Open the demo GUI: replay an attack on the network map, play it, and see training charts.

Usage:
    python scripts/visualize.py --run tiny_seed0           # trained agent + its charts
    python scripts/visualize.py --run tiny_seed2           # another seed of the same config
    python scripts/visualize.py --run tiny_seed0 --manual  # you pick the actions
    python scripts/visualize.py                            # the most recently trained run
    python scripts/visualize.py --scenario external/PenGym/database/scenarios/small-honeypot.yml
    python scripts/visualize.py --run tiny_seed0 --save-charts tiny.png

A run is what scripts/train.py writes for one config and seed, named <config>_seed<N>:
results/<run>/config.yml (its settings, including the scenario), results/<run>/episodes.csv,
and checkpoints/<run>.pt. Without a checkpoint, the replay uses a random agent.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
# Lets `python scripts/visualize.py` import from src/ (the repo root is not on the path otherwise)
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# pylint: disable=wrong-import-position
from src.viz.app import App
from src.viz.charts import save_training_figure
from src.viz.session import AgentPolicy, RandomPolicy, Session, load_agent
from src.viz.training_log import read_log

# pylint: enable=wrong-import-position

RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


def parse_args():
    """Read the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run", help="Run name, e.g. tiny_seed0 (default: most recent run)")
    source.add_argument("--scenario", help="Scenario .yml path or NASim name, without a run")
    parser.add_argument("--model", help="Checkpoint to load instead of the run's own")
    parser.add_argument("--log", help="episodes.csv to chart instead of the run's own")
    parser.add_argument("--random", action="store_true", help="Replay a random agent")
    parser.add_argument("--manual", action="store_true", help="Start in manual mode")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Trained agent's exploration")
    parser.add_argument("--episode-seed", type=int, default=0, help="Seed of the replayed episode")
    parser.add_argument(
        "--save-charts", metavar="PATH", help="Save the charts as an image and exit"
    )
    return parser.parse_args()


def repo_path(path):
    """A path as given if it exists, otherwise relative to the repo root."""
    path = Path(path)
    return path if path.is_absolute() or path.exists() else REPO_ROOT / path


def available_runs():
    """Names of the runs in results/ that have a saved config, most recent first."""
    configs = sorted(
        RESULTS_DIR.glob("*/config.yml"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return [path.parent.name for path in configs]


def load_run_config(run):
    """The settings scripts/train.py saved for a run. Exits with the known runs if it's missing."""
    path = RESULTS_DIR / run / "config.yml"
    if not path.is_file():
        runs = ", ".join(available_runs()) or "none yet (train one with scripts/train.py)"
        sys.exit(f"No run named {run!r} (looked for {path}). Available runs: {runs}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_run(args):
    """Scenario, step cap, checkpoint, log, and a title, from --run or --scenario."""
    if args.scenario:
        scenario = repo_path(args.scenario)
        scenario = scenario if scenario.is_file() else args.scenario
        title = Path(str(scenario)).stem
        return scenario, None, args.model, args.log, title
    run = args.run
    if run is None:
        runs = available_runs()
        if not runs:
            sys.exit("No runs in results/ yet: train one with scripts/train.py, or use --scenario")
        run = runs[0]
    cfg = load_run_config(run)
    scenario = repo_path(cfg["scenario"])
    scenario = scenario if scenario.is_file() else cfg["scenario"]
    model = args.model or CHECKPOINT_DIR / f"{run}.pt"
    log = args.log or RESULTS_DIR / run / "episodes.csv"
    title = f"{Path(str(scenario)).stem}  ·  {run}"
    return scenario, cfg.get("max_episode_steps"), model, log, title


def build_policies(args, session, model):
    """The agents to offer (trained first, if its checkpoint loads) and a startup notice."""
    random_agent = RandomPolicy(session.n_actions, seed=args.episode_seed)
    if args.random or model is None:
        return [random_agent], ""
    model = Path(model)
    if not model.is_file():
        if args.model:
            sys.exit(f"No checkpoint at {model}")
        return [random_agent], f"No checkpoint at {model.name}: showing a random agent"
    try:
        agent = load_agent(model, session.env)
    except ValueError as error:
        if args.model:
            sys.exit(str(error))
        return [random_agent], f"{model.name} doesn't fit this scenario: showing a random agent"
    session.advisor = agent
    session.restart()
    trained = AgentPolicy(agent, args.epsilon, label=f"trained agent ({model.name})")
    return [trained, random_agent], ""


def main():
    """Parse arguments, load the run, and open the GUI (or save the charts)."""
    args = parse_args()
    scenario, max_steps, model, log_path, title = resolve_run(args)
    log = read_log(log_path) if log_path and Path(log_path).is_file() else None

    if args.save_charts:
        if log is None:
            sys.exit(f"No training log at {log_path}")
        save_training_figure(log, args.save_charts)
        print(f"Saved charts for {log.run_name} to {args.save_charts}")
        return

    session = Session(scenario, seed=args.episode_seed, max_steps=max_steps)
    policies, notice = build_policies(args, session, model)
    App(session, policies, log=log, title=title, manual=args.manual, notice=notice).run()


if __name__ == "__main__":
    main()
