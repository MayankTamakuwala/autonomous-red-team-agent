# Benchmark Scenarios

This document lists the network scenarios available in PenGym and NASim, and the subset we use to train and evaluate our agent.

## Purpose

Benchmark scenarios are fixed networks that every experiment runs on, so results are comparable. On each scenario, our learned agents (sparse and shaped reward) are compared against:

- **Random agent:** the floor. A trained agent should clearly beat it.
- **Hand-coded / optimal path:** the ceiling. Shows what near-perfect play looks like.

The main comparison is **sparse vs shaped reward** on the same scenarios.

## Available scenarios

PenGym's versions differ from NASim's: Windows is disabled (1 OS instead of 2), some processes are renamed for Linux (`daclsvc` to `proftpd`, `schtask` to `cron`), and exploit probabilities are 0.999999 instead of values like 0.9. Because of the OS change, each PenGym observation row is one column shorter than NASim's.

Subnet counts below exclude the internet.

| Scenario | Subnets | Hosts | Actions | Obs (rows x cols) | Obs size | Step limit | Sensitive hosts |
|---|---|---|---|---|---|---|---|
| tiny | 3 | 3 | 18 | 4 x 14 | 56 | 1000 | (2,0), (3,0) |
| tiny-hard | 3 | 3 | 27 | 4 x 17 | 68 | 1000 | (2,0), (3,0) |
| tiny-small | 4 | 5 | 45 | 6 x 19 | 114 | 1000 | (2,0), (4,0) |
| small-honeypot | 4 | 8 | 72 | 9 x 22 | 198 | 1000 | (2,0), (4,0) |
| small-linear | 6 | 8 | 72 | 9 x 21 | 189 | 1000 | (3,0), (4,0) |
| medium | 5 | 16 | 192 | 17 x 26 | 442 | 2000 | (2,0), (5,0) |
| medium-single-site | 1 | 16 | 192 | 17 x 33 | 561 | 2000 | (1,3), (1,8) |
| medium-multi-site | 6 | 16 | 192 | 17 x 28 | 476 | 2000 | (2,1), (3,4) |

NASim also includes `small` (static, not in PenGym) and generated scenarios (`tiny-gen`, `small-gen`, `medium-gen`, `large-gen`, `huge-gen`, `pocp-1-gen`, `pocp-2-gen`). The larger generated ones (322 to 3515 actions) are out of scope.

## Chosen scenarios

| Scenario | Subnets | Hosts | Actions | Obs size | Step limit | Sensitive hosts |
|---|---|---|---|---|---|---|
| tiny | 3 | 3 | 18 | 56 | 1000 | (2,0), (3,0) |
| tiny-small | 4 | 5 | 45 | 114 | 1000 | (2,0), (4,0) |
| small-honeypot | 4 | 8 | 72 | 198 | 1000 | (2,0), (4,0) |
| medium-single-site | 1 | 16 | 192 | 561 | 2000 | (1,3), (1,8) |
| medium-multi-site | 6 | 16 | 192 | 476 | 2000 | (2,1), (3,4) |

### Why each was picked

- **tiny:** smallest scenario, used to check that the agent and training loop work at all.
- **tiny-small:** adds an internal subnet that requires pivoting, while staying fast to train.
- **small-honeypot:** contains a honeypot host (3,2) worth -100, which tests whether the agent learns careful behavior rather than compromising everything it can reach.
- **medium-single-site:** same action space as medium-multi-site, but a single flat subnet with a 4-action optimal path. Separates the effect of action space size from network complexity: if the agent succeeds here but fails on medium-multi-site, the difficulty is multi-step pivoting rather than size. This can be scraped if time is limited.
- **medium-multi-site:** PenGym's main scenario, used in its real cyber range demo, with a known 16-step hand-coded solution for baseline comparison. Used for final results.

**Optional:** small-linear (72 actions) can be added alongside small-honeypot as an 8-host scenario without a trap, to help interpret honeypot results.

## FYI

- Always load PenGym's YAML by file path. `--scenario tiny` loads NASim's built-in tiny, not PenGym's `tiny.yml`.
