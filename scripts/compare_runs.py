"""Overlay the training curves of several runs on one chart and save it as an image.

Usage:
    python scripts/compare_runs.py --config tiny          # every tiny run, grouped by algorithm
    python scripts/compare_runs.py tiny_ddqn_seed0 tiny_ddqn_seed1 tiny_ddqn_seed2
    python scripts/compare_runs.py tiny_dqn_seed{0..4}=DQN tiny_ddqn_seed{0..4}="Double DQN"

--config NAME finds every run in results/ named <NAME>_<group>_seed<N>, the way
scripts/train.py names runs, and averages each group's seeds into one curve. The group is
whatever sits between the config name and the seed (dqn, ddqn, ...), so "tiny" matches
tiny_ddqn_seed0 but not tiny-small_ddqn_seed0. NAME can also be a config path like
configs/tiny.yml.

Each RUN argument is RUN or RUN=LABEL. RUN is a folder name in results/, or a path to a run
folder or to its episodes.csv. Without a label, the run's name is used. Runs that share a label
are averaged into one curve, with a band of one standard deviation across them.
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
# Lets `python scripts/compare_runs.py` import from src/ (the repo root is not on the path)
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from src.viz.compare import save_comparison_figure
from src.viz.training_log import read_log

# pylint: enable=wrong-import-position


def parse_run(arg):
    """Split "RUN=LABEL" into (RUN, LABEL). Without "=LABEL", the label is None."""
    run, _, label = arg.partition("=")
    return run, label or None


def episodes_csv(run):
    """The episodes.csv for a run name, run folder, or CSV path. Exits if it doesn't exist."""
    path = Path(run)
    if not path.exists():
        path = RESULTS_DIR / run
    if path.is_dir():
        path = path / "episodes.csv"
    if not path.is_file():
        runs = sorted(p.parent.name for p in RESULTS_DIR.glob("*/episodes.csv"))
        sys.exit(f"No episodes.csv for '{run}'. Runs in results/: {', '.join(runs) or 'none'}")
    return path


def config_runs(config, results_dir=RESULTS_DIR):
    """Every run of one config in results_dir, as (group, run folder) pairs.

    Matches folders named <config>_<group>_seed<N> that hold an episodes.csv, sorted by group
    and then seed number.
    """
    name = Path(config).stem
    pattern = re.compile(rf"{re.escape(name)}_(?P<group>.+)_seed(?P<seed>\d+)")
    found = []
    for path in results_dir.glob("*/episodes.csv"):
        match = pattern.fullmatch(path.parent.name)
        if match:
            found.append((match["group"], int(match["seed"]), path.parent))
    return [(group, folder) for group, _, folder in sorted(found)]


def main():
    """Parse arguments, read each run's log, and save the comparison chart."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="*", metavar="RUN[=LABEL]", help="Runs to overlay")
    parser.add_argument(
        "--config", help="Compare every run of this config, grouped by algorithm (e.g. tiny)"
    )
    parser.add_argument(
        "--out",
        help="Image file to write (default: results/comparison_<config>.png with --config, "
        "otherwise results/comparison.png)",
    )
    parser.add_argument("--window", type=int, default=20, help="Episodes per moving average")
    args = parser.parse_args()
    if not args.runs and not args.config:
        parser.error("name some runs, or use --config")

    labeled_runs = [parse_run(arg) for arg in args.runs]
    if args.config:
        found = config_runs(args.config)
        if not found:
            sys.exit(f"No runs named {Path(args.config).stem}_<algorithm>_seed<N> in results/")
        labeled_runs += [(str(folder), group) for group, folder in found]

    labeled_logs = []
    for run, label in labeled_runs:
        log = read_log(episodes_csv(run))
        labeled_logs.append((label or log.run_name, log))

    default_name = f"comparison_{Path(args.config).stem}" if args.config else "comparison"
    out = Path(args.out or RESULTS_DIR / f"{default_name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_comparison_figure(labeled_logs, out, window=args.window)
    groups = sorted({label for label, _ in labeled_logs})
    print(f"Saved a comparison of {len(labeled_logs)} runs ({', '.join(groups)}) to {out}")


if __name__ == "__main__":
    main()
