"""Overlay the training curves of several runs on one chart and save it as an image.

Usage:
    python scripts/compare_runs.py tiny_seed0 tiny_seed1 tiny_seed2
    python scripts/compare_runs.py tiny_dqn_seed0=DQN tiny_ddqn_seed0="Double DQN"
    python scripts/compare_runs.py tiny_sparse_seed{0..4}=sparse tiny_shaped_seed{0..4}=shaped

Each argument is RUN or RUN=LABEL. RUN is a folder name in results/, or a path to a run folder
or to its episodes.csv. Without a label, the run's name is used. Runs that share a label are
averaged into one curve, with a band of one standard deviation across them.
"""

import argparse
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


def main():
    """Parse arguments, read each run's log, and save the comparison chart."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="+", metavar="RUN[=LABEL]", help="Runs to overlay")
    parser.add_argument(
        "--out", default=str(RESULTS_DIR / "comparison.png"), help="Image file to write"
    )
    parser.add_argument("--window", type=int, default=20, help="Episodes per moving average")
    args = parser.parse_args()

    labeled_logs = []
    for run, label in map(parse_run, args.runs):
        log = read_log(episodes_csv(run))
        labeled_logs.append((label or log.run_name, log))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_comparison_figure(labeled_logs, args.out, window=args.window)
    print(f"Saved a comparison of {len(labeled_logs)} runs to {args.out}")


if __name__ == "__main__":
    main()
