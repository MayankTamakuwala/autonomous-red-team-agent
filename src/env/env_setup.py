"""One place to create PenGym environments in simulation mode.

Usage:
    from src.env.env_setup import make_env, SCENARIO_DIR

    env = make_env("tiny", seed=0)                                  # NASim built-in benchmark
    env = make_env(SCENARIO_DIR / "medium-multi-site.yml", seed=0)  # PenGym scenario file

Nothing runs on import. Everything (path setup, simulation flags, env creation) happens
inside make_env().
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

from src.env.memory_wrapper import MemoryWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]
PENGYM_DIR = REPO_ROOT / "external" / "PenGym"
SCENARIO_DIR = PENGYM_DIR / "database" / "scenarios"


def make_env(scenario, seed=None, memory=True):
    """Create a PenGym env in NASim simulation mode.

    Args:
        scenario: a NASim benchmark name (e.g. "tiny") or a path to a .yml scenario file.
            Note: the same name can exist in both places with different contents. "tiny" loads
            NASim's original; SCENARIO_DIR / "tiny.yml" would load PenGym's edited version.
        seed: if given, seeds NumPy's global Random Number Generator (NASim uses it to roll exploit
            success) and the action space. Still pass a seed to env.reset() for a fully reproducible
            episode.
        memory: if True, wrap the env in MemoryWrapper so observations accumulate everything
            seen so far in the episode. If False, return the raw env, whose observations only
            show what the last action revealed.

    Returns:
        A Gymnasium env. Observations are 1D vectors in both cases.
    """
    if str(PENGYM_DIR) not in sys.path:
        sys.path.insert(0, str(PENGYM_DIR))

    import pengym  # pylint: disable=import-outside-toplevel
    import pengym.utilities as pengym_utils  # pylint: disable=import-outside-toplevel

    pengym_utils.ENABLE_PENGYM = False  # no real cyber range
    pengym_utils.ENABLE_NASIM = True  # simulation only

    # MemoryWrapper needs the observation as a 2D grid (one row per host)
    flat_obs = not memory

    # PenGym prints a line when it creates an env; keep it quiet
    with contextlib.redirect_stdout(io.StringIO()):
        if Path(scenario).is_file():
            env = pengym.load(str(scenario), flat_obs=flat_obs)
        else:
            env = pengym.create_environment(str(scenario), flat_obs=flat_obs)

    if seed is not None:
        np.random.seed(seed)
        env.action_space.seed(seed)

    return MemoryWrapper(env) if memory else env
