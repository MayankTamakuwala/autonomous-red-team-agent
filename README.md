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

_To be added._

## Reproducing Results

_To be added._

## Code Organization

_To be added._

## References

1. cyb3rlab. PenGym: Pentesting training framework for reinforcement learning agents. https://github.com/cyb3rlab/PenGym
2. Nguyen, H. P. T., Hasegawa, K., Fukushima, K., & Beuran, R. (2025). PenGym: Realistic training environment for reinforcement learning pentesting agents. *Computers & Security*, 148, 104140. https://doi.org/10.1016/j.cose.2024.104140
3. Phuong, H., Nguyen, T., Chen, Z., Hasegawa, K., Fukushima, K., & Beuran, R. PenGym: Pentesting training framework for reinforcement learning agents. https://doi.org/10.5220/0012367300003648
4. Schwartz, J. (2022). Jjschwartz/NetworkAttackSimulator. GitHub. https://github.com/Jjschwartz/NetworkAttackSimulator
