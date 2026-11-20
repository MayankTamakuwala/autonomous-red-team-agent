# Autonomous Red-Team Agent for Penetration Testing in Simulated Networks

CS 5100: Foundations of AI, Fall 2026, Northeastern University

**Team:** Mayank Devangkumar Tamakuwala, Arun Daniel Kennedy, Hakim Badmus

## Overview

Penetration testing finds weaknesses in a network before real attackers do, but it is slow, expensive,
and limited by the number of skilled testers available. This project builds a reinforcement learning
agent that learns the core exploration part of that job inside a simulated, partially observable network:
finding an efficient path to the most valuable targets without generating unnecessary noise.

The central question is reward design: can reward functions teach an agent to behave like a careful,
patient tester instead of a noisy, brute-force one? We will compare at least two formulations:

- **Sparse reward:** rewards only reaching the final target.
- **Shaped reward:** additionally penalizes redundant or noisy actions.

Agents will be evaluated on total actions per episode, proportion of redundant scans, and steps to goal
across different network configurations.

The environment is [PenGym](https://github.com/cyb3rlab/PenGym) in simulation mode (built on
[NASim](https://github.com/Jjschwartz/NetworkAttackSimulator)). The policy and learning algorithm are
implemented from scratch, without pre-built RL libraries.

## Environment Setup

Everything runs in PenGym's **simulation mode**, which is pure Python. You do **not** need Metasploit, nmap,
KVM, or CyRIS (those are only for PenGym's real cyber-range mode). Tested with Python 3.12 on macOS.

### 1. Clone the repository with its submodule

PenGym is included as a git submodule in `external/PenGym`, so clone with `--recurse-submodules`:

```bash
git clone --recurse-submodules https://github.com/MayankTamakuwala/autonomous-red-team-agent.git
cd autonomous-red-team-agent
```

If you already cloned without it (`external/PenGym` is empty), run:

```bash
git submodule update --init
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`pymetasploit3`, `python-nmap`, and `psutil` are in `requirements.txt` only because PenGym imports them at
startup. They are client libraries; nothing connects to Metasploit or nmap in simulation mode.

### 3. Verify the setup

Run a random agent on PenGym's `medium-multi-site` scenario:

```bash
python scripts/smoke_test.py
```

It should print the action and observation sizes, then one line per episode. Other options:

```bash
python scripts/smoke_test.py --scenario tiny --episodes 20   # NASim built-in benchmark
python scripts/smoke_test.py --verbose                       # show every action PenGym takes
```

### 4. (Optional) Run PenGym's own deterministic demo

PenGym's `run.py` needs a config file whose `pengym_source` points at your local clone. Create a local copy
(ignored by git, since the path is machine-specific):

```bash
mkdir -p configs
sed "s|^pengym_source:.*|pengym_source: \"${PWD#/}/external/PenGym\"|" \
    external/PenGym/pengym/CONFIG.yml > configs/pengym_local.yml
```

PenGym builds this path by prepending `/`, so this demo assumes macOS or Linux. On Windows, use WSL for this
step (the smoke test above does not need a config file).

Then run it. The `-n -d` flags must come **before** the config path, otherwise they are ignored and PenGym
tries to connect to a Metasploit server:

```bash
cd external/PenGym
python run.py -n -d ../../configs/pengym_local.yml
```

A successful run ends with `* NORMAL execution: 16 steps`.

## Code Style and Tests

All Python code is formatted with [black](https://black.readthedocs.io/) and linted with
[pylint](https://pylint.readthedocs.io/). Tests live in `tests/` and run with
[pytest](https://docs.pytest.org/). Settings live in `pyproject.toml`. A GitHub Action runs all three
on every push and pull request, and fails if any check does not pass.

Install the tools once:

```bash
pip install -r requirements-dev.txt
```

Before committing, run:

```bash
black .
pylint $(git ls-files '*.py')
pytest
```

`black .` rewrites files in place to the shared style. `pylint` reports issues you need to fix by hand.
`pytest` runs the tests (a few seconds, on NASim's tiny scenario).

## Demo GUI

`scripts/visualize.py` opens a window with two views; Tab switches between them.

- **Attack replay:** the network as a radial map. The attacker sits in the center and each ring is one
  more hop from the internet; a subnet is an arc of hosts on its ring. An episode plays step by step:
  each action travels from the host it was launched from, and hosts change as the agent discovers them
  (cyan), gets a user shell (`$`, amber), and gets root (`#`, red). Gold rings mark sensitive hosts and
  dashed magenta rings mark honeypots. The side panel shows the action, its result and reward, the
  running total, redundant and failed action counts, and the trained agent's top-scored next actions.
- **Training charts:** total reward and steps per episode from a run's `episodes.csv`, with a
  20-episode moving average, plotted against training steps so curves from different runs line up.

Train a config, then open the GUI for the run it produced. Runs are named `<config>_seed<N>`:

```bash
python scripts/train.py --config configs/tiny.yml
python scripts/visualize.py --run tiny_seed0
```

This reads the run's settings from `results/tiny_seed0/config.yml`, replays
`checkpoints/tiny_seed0.pt` (or a random agent if there is no checkpoint), and charts
`results/tiny_seed0/episodes.csv`. Other options:

```bash
python scripts/visualize.py --run tiny_seed2                  # another seed of the same config
python scripts/visualize.py                                   # the most recently trained run
python scripts/visualize.py --run tiny_seed0 --manual         # you choose the actions
python scripts/visualize.py --scenario external/PenGym/database/scenarios/small-honeypot.yml
python scripts/visualize.py --run tiny_seed0 --save-charts tiny.png   # charts as an image
```

**Manual mode** (M, or `--manual`): click a host and pick an action from its menu. Backspace undoes
the last action, and T lets the agent take over from the current state. Pressing M during a replay
takes control from the step on screen.

**Keys:** Space play/pause · ←/→ step · Home/End jump · +/− speed · A switch between the trained and
random agent · N new episode · G show the hidden ground truth on hover · H list all keys.

## Reproducing Results

_To be added._

## Code Organization

_To be added._

## References

1. cyb3rlab. PenGym: Pentesting training framework for reinforcement learning agents. https://github.com/cyb3rlab/PenGym
2. Nguyen, H. P. T., Hasegawa, K., Fukushima, K., & Beuran, R. (2025). PenGym: Realistic training environment for reinforcement learning pentesting agents. *Computers & Security*, 148, 104140. https://doi.org/10.1016/j.cose.2024.104140
3. Phuong, H., Nguyen, T., Chen, Z., Hasegawa, K., Fukushima, K., & Beuran, R. PenGym: Pentesting training framework for reinforcement learning agents. https://doi.org/10.5220/0012367300003648
4. Schwartz, J. (2022). Jjschwartz/NetworkAttackSimulator. GitHub. https://github.com/Jjschwartz/NetworkAttackSimulator
