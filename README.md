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

## Reproducing Results

_To be added._

## Code Organization

_To be added._

## References

1. cyb3rlab. PenGym: Pentesting training framework for reinforcement learning agents. https://github.com/cyb3rlab/PenGym
2. Nguyen, H. P. T., Hasegawa, K., Fukushima, K., & Beuran, R. (2025). PenGym: Realistic training environment for reinforcement learning pentesting agents. *Computers & Security*, 148, 104140. https://doi.org/10.1016/j.cose.2024.104140
3. Phuong, H., Nguyen, T., Chen, Z., Hasegawa, K., Fukushima, K., & Beuran, R. PenGym: Pentesting training framework for reinforcement learning agents. https://doi.org/10.5220/0012367300003648
4. Schwartz, J. (2022). Jjschwartz/NetworkAttackSimulator. GitHub. https://github.com/Jjschwartz/NetworkAttackSimulator
