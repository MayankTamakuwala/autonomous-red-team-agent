# NASim / PenGym Notes

## Section 1: Networking basics (as NASim models them)

**Host:** a single computer, addressed as `(subnet, host)`. Host numbers start at 0.
- Example: `(3, 4)` in medium-multi-site is the fifth host in subnet 3 (main site user subnet).

**Subnet:** a group of hosts on the same local network. Hosts in the same subnet can reach each other; firewall rules only apply between subnets.
- Example: `subnets: [2, 2, 6, 2, 2, 2]` in medium-multi-site means subnet 1 has 2 hosts, subnet 2 has 2, subnet 3 has 6, and so on (16 hosts total).

**Internet (subnet 0):** where the agent starts. It contains no hosts, so it appears in `topology` but not in `subnets`.
- Example: medium-multi-site has 6 entries in `subnets` but a 7×7 topology, because the internet is the extra row and column.

**Topology:** which subnets are directly connected. A square matrix of size (number of subnets + 1). Row = from, column = to, 1 = connected. The diagonal is always 1.
- Example: in medium-multi-site, the internet row is `[1, 1, 0, 0, 1, 1, 1]`, so the internet connects to subnets 1, 4, 5, 6 but not 2 or 3.

**Subnet firewall:** for each connected pair, an allow list of services that can pass, per direction. `[]` means connected but fully blocked.
- Example: `(0, 6): [samba]` means only samba traffic can go from the internet to subnet 6. `(0, 1): []` means the internet connects to subnet 1 but nothing gets through.

**Host firewall:** optional, set per host in `host_configurations`. A deny list of services blocked from specific source hosts, applied on top of the subnet firewall.
- Example: in tiny, host `(2, 0)` has `firewall: (1, 0): [ssh]`, meaning it refuses ssh coming from `(1, 0)`.

**Service:** a program that accepts connections from other computers over the network (ssh, ftp, http, samba, smtp). Anything that can get traffic through the firewall can reach it, which is why exploits target services.
- Example: `(6, 1)` in medium-multi-site has `services: [ssh, samba]`, so it can be attacked through either one if the firewall allows it.

**OS:** the host's operating system. Some exploits require a specific OS.
- Example: `e_ssh` has `os: linux`, so it only works on Linux hosts. `e_http` has `os: None`, so it works on any OS.

**Process:** a program running on a host that does not accept connections from other computers. It can't be reached from outside, only from inside the host. Privilege escalation targets processes.
- Example: `(3, 4)` in medium-multi-site has `processes: [tomcat]`, so once the agent is inside as user, `pe_tomcat` can upgrade it to root.

**Access levels:** none (0), user (1), root (2). Each exploit or privesc defines which level it gives. Sensitive hosts give their value when the agent reaches root.
- Example: `e_ssh` gives `access: user`, while `e_samba` gives `access: root` directly. So exploiting `(3, 4)` via ssh still requires a privesc to collect its 100.

**Discovered vs reachable:** discovered = the agent knows the host exists. Reachable = the agent can currently send traffic to it.
- Example: in tiny, a subnet scan from `(1, 0)` discovers `(2, 0)`, but `(2, 0)` is not reachable from `(1, 0)` because `(1, 2): []` blocks it.

**Pivoting:** using a compromised host as the source for further actions. Hosts not reachable from the internet can only be attacked through a chain of compromised hosts.
- Example: in medium-multi-site, the internet can't reach subnet 2, but after compromising `(6, 1)`, the agent can exploit `(2, 1)` from it via smtp because `(6, 2): [smtp]`.

### Conditions for an exploit to succeed

1. The target runs the exploit's service
2. The firewall allows that service from the agent's source (internet or a compromised host)
3. The target's OS matches the exploit's OS (if one is specified)

Failed actions still cost points.
- Example: exploiting `(1, 0)` in medium-multi-site with `e_ssh` from the internet fails even though it runs ssh, because `(0, 1): []` blocks all traffic.

---

## Section 2: Scenario `.yml` format

A scenario file defines the whole network: its layout, the rules of what exists, every host's setup, and the goal. NASim reads it and builds the environment. Fields fall into three groups.

### Group 1: Layout

**`subnets`:** number of hosts in each subnet, starting at subnet 1 (the internet is not listed).
- Example: `subnets: [2, 2, 6, 2, 2, 2]` → 6 subnets, 16 hosts.

**`topology`:** connection matrix, size (subnets + 1) × (subnets + 1), including the internet as row/column 0. 1 = connected.
- Example: internet row `[1, 1, 0, 0, 1, 1, 1]` → internet connects to subnets 1, 4, 5, 6.

### Group 2: What exists in this world

**`os`, `services`, `processes`:** the full list of each type in the scenario. List order matters: it sets the column order in the observation.
- Example: `services: [ssh, ftp, http, samba, smtp]` → service columns appear in that order in every host row.

**`exploits`:** each exploit defines:
- `service`: service the target must run
- `os`: OS the target must have (`None` = any)
- `prob`: success probability if conditions are met
- `cost`: points subtracted when used
- `access`: access gained on success (user or root)

Example:

```yaml
e_samba:
  service: samba
  os: linux
  prob: 0.999999
  cost: 2
  access: root
```

→ works on Linux hosts running samba, costs 2, gives root directly.

**`privilege_escalation`:** same fields, but `process` instead of `service`. Requires the agent to already have user access on the target.
- Example: `pe_tomcat` needs tomcat running and Linux, costs 1, gives root.

**Scan costs:** `service_scan_cost`, `os_scan_cost`, `subnet_scan_cost`, `process_scan_cost`.
- Example: all are 1 in every provided scenario.

### Group 3: The level

**`sensitive_hosts`:** target hosts and their values. All other hosts are worth 0 unless given a value in `host_configurations`.
- Example: `(2, 1): 100` and `(3, 4): 100` in medium-multi-site → max reward of 200 before costs.

**`host_configurations`:** every host's OS, services, and processes. This is the hidden ground truth the agent discovers by scanning. Optional extras:
- `value`: overrides the host's value
- `firewall`: host-level deny list by source host

Example:

```yaml
(3, 4):
  os: linux
  services: [ssh]
  processes: [tomcat]
```

**`firewall`:** allow list of services per direction, `(from, to): [services]`, for each connected pair in topology. `[]` = fully blocked.
- Example: `(0, 6): [samba]` → only samba passes from the internet to subnet 6.

**`step_limit`:** max actions per episode before it ends unsuccessfully.
- Example: 2000 in medium-multi-site, 1000 in tiny.

### Special case: honeypots

A honeypot is a decoy host that penalizes the agent for compromising it, set with a negative value.

Example (small-honeypot):

```yaml
(3, 2):
  services: [ftp, http]
  processes: [proftpd]
  value: -100
```

### Gotchas

- **Header comments are not always accurate.** They're left over from NASim's originals. Medium-multi-site's header says "2 OS," but Windows is commented out. Trust the actual fields.
- **Inline comments show original values.** `prob: 0.999999 #0.9` means exploits were originally riskier; in these files they effectively always succeed.
- **Some processes were renamed for Linux.** `proftpd #daclsvc` and `cron #schtask` replaced original Windows processes.
- **The "Optimal path" header is not read by the code.** It's the author's best-case solution assuming full knowledge of the network. Useful as a perfect-score reference, but a real agent must also spend actions scanning.
  - Best score would be like 187 ish on medium-multi-site rather than 195.
- **OS count affects observation size, not action count.** Adding an OS adds one column per host row; the number of actions stays the same.
- To actually test exploration, we'd want to train on varied networks and test on ones it hasn't seen. NASim can generate random scenarios for this.

---

## Section 3: Action space

Each step, the agent picks one action aimed at one host. There are 6 action types, split into remote actions (sent over the network from the internet or a compromised host) and on-host actions (performed on a host the agent already has access to).

### Remote actions

Require a path to the target: it must be reachable from the internet or from a compromised host, through firewall rules that allow the traffic.

**Service scan:** reveals all services running on the target.
- Example: service scan on `(6, 1)` from the internet reveals ssh and samba.

**OS scan:** reveals the target's OS.
- Example: OS scan on `(6, 1)` reveals linux.

**Exploit:** attempts to gain access through one specific service. Succeeds only if the target runs that service, the firewall allows it from the agent's source, and the OS matches (if required). Gives the access level defined in the YAML.
- Example: `e_samba` on `(6, 1)` from the internet succeeds because `(0, 6): [samba]` allows it and the host runs samba. Gives root.

### On-host actions

Require the agent to already have access on the target host itself. Firewalls do not apply.

**Subnet scan:** discovers all hosts in every subnet connected (per topology) to the target host's subnet, including its own.
- Example: subnet scan on `(6, 1)` discovers `(6, 0)`, `(2, 0)`, and `(2, 1)`, since subnet 6 connects to subnet 2. A subnet scan on `(2, 1)` discovers all 16 hosts, since subnet 2 connects to every subnet.

**Process scan:** reveals all processes running on the target.
- Example: process scan on `(3, 4)` reveals tomcat.

**Privilege escalation:** upgrades user access to root using a process running on the host. Requires user access on the target and the matching process and OS.
- Example: `pe_tomcat` on `(3, 4)` after gaining user via `e_ssh` gives root.

**Discovered ≠ reachable:** a subnet scan can reveal hosts the agent still can't attack.
- Example: a subnet scan on `(2, 1)` discovers all user subnet hosts, but they can only be exploited via http, since `(2, 3): [http]`.

### Action properties

Every action has:
- `target`: the host address it's aimed at
- `cost`: from the YAML (exploit/privesc cost, or the `*_scan_cost` fields)
- `prob`: chance of success when conditions are met (scans always succeed)
- `req_access`: access needed on the target (on-host) or on the pivot host (remote)

### Failed actions

Failed actions still cost points. The observation's auxiliary row reports why:
- **Connection error:** target unreachable (no path or blocked by firewall)
- **Permission error:** insufficient access
- **Undefined error:** random failure based on `prob`

Example: `pe_tomcat` on `(6, 0)` from the internet fails with a permission error and costs 1, because the agent has no access on `(6, 0)`, even though it runs tomcat and Linux.

### Flat action space (how actions are represented in code)

NASim's flat action space represents each action as a single integer. Every host gets one copy of every action, grouped by host.
- Example: in medium-multi-site, each host has 12 actions, so indices 0 to 11 belong to the first host, 12 to 23 to the second, and so on, for 192 total.

> **TODO:** Need to verify this.

---

## Section 4: How the number of actions grows

The total number of actions is:

```
|A| = hosts × (exploits + privescs + 4)
```

Every host gets one copy of every exploit, every privilege escalation, and the 4 scans (service, OS, subnet, process), whether or not that action could ever succeed on that host.

Example: medium-multi-site has 16 hosts, 5 exploits, and 3 privescs → 16 × (5 + 3 + 4) = 192.

Across the provided scenarios:

| Scenario | Hosts | Exploits | Privescs | Per host | Total |
|---|---|---|---|---|---|
| Tiny | 3 | 1 | 1 | 6 | 18 |
| Tiny (harder) | 3 | 3 | 2 | 9 | 27 |
| Tiny-small | 5 | 3 | 2 | 9 | 45 |
| Small honeypot | 8 | 3 | 2 | 9 | 72 |
| Small linear | 8 | 3 | 2 | 9 | 72 |
| Medium (one public subnet) | 16 | 5 | 3 | 12 | 192 |
| Medium single-site | 16 | 5 | 3 | 12 | 192 |
| Medium multi-site | 16 | 5 | 3 | 12 | 192 |

**What affects the count:** number of hosts, exploits, and privescs.

**What doesn't:** subnets, topology, firewall rules, number of OSes, or each host's services and processes.
- Example: all three medium scenarios have different network layouts but the same 192 actions, because they share the same host count and exploit/privesc lists.

**Growth:**
- Adding 1 host adds (exploits + privescs + 4) actions.
- Adding 1 exploit or privesc adds 1 action per host.
- Example: medium-multi-site with 4 extra hosts and 1 extra exploit → 20 × (6 + 3 + 4) = 260.

---

## Section 6: Default reward

Each step's reward is the value gained minus the action's cost:

```
reward = value gained − action cost
```

- **Cost:** every action subtracts its cost, whether it succeeds or fails. Exploit and privesc costs come from the YAML; scans use the `*_scan_cost` fields.
- **Value:** gained when the agent reaches root on a host. Sensitive hosts have their listed value. All other hosts are 0 unless given a value in `host_configurations`.
- **Discovery value:** a reward for discovering a host. It's 0 in all provided scenarios.

The episode return is the sum of all step rewards.

**Episode ends when:**
- All sensitive hosts are rooted → success (`done = True`)
- `step_limit` is reached → failure (`truncated = True`)

### Example: tiny, clean path with scans

| # | Action | Cost | Value | Reward |
|---|---|---|---|---|
| 1 | Service scan (1, 0) | 1 | 0 | −1 |
| 2 | e_ssh (1, 0) | 1 | 0 | −1 |
| 3 | Subnet scan (1, 0) | 1 | 0 | −1 |
| 4 | Service scan (3, 0) | 1 | 0 | −1 |
| 5 | e_ssh (3, 0) | 1 | 0 | −1 |
| 6 | Process scan (3, 0) | 1 | 0 | −1 |
| 7 | pe_tomcat (3, 0) | 1 | 100 | +99 |
| 8 | Service scan (2, 0) | 1 | 0 | −1 |
| 9 | e_ssh (2, 0) | 1 | 0 | −1 |
| 10 | Process scan (2, 0) | 1 | 0 | −1 |
| 11 | pe_tomcat (2, 0) | 1 | 100 | +99 |

Total: 200 − 11 = **189**. The file's optimal path scores 195 because it skips scans by assuming full knowledge.

### Example: medium-multi-site optimal path

| Action | Cost | Value |
|---|---|---|
| e_samba (6, 1) | 2 | 0 |
| Subnet scan (6, 1) | 1 | 0 |
| e_smtp (2, 1) | 3 | 100 (gives root directly) |
| Subnet scan (2, 1) | 1 | 0 |
| e_http (3, 1) | 2 | 0 |
| e_ssh (3, 4) | 3 | 0 (user only) |
| pe_tomcat (3, 4) | 1 | 100 |

Total: 200 − 13 = **187**.

### Other examples

- **Honeypot:** rooting `(3, 2)` in small-honeypot gives −100.
- **Failed action:** `e_ssh (2, 0)` in tiny while only owning `(1, 0)` fails with a connection error and still gives −1.
- **Random baseline:** tiny averaged 103.4 and medium-multi-site averaged −1116.2, since wasted actions add up.

### Notes

- `smoke_test.py` caps episodes at 1000 steps, but medium-multi-site's YAML sets `step_limit: 2000`. Check which limit training uses.
- The default reward treats a useful scan and a redundant scan the same (both −1). Separating them is the job of the shaped reward.

---

## Section 7: Gotchas and things to verify

### Loading the environment

- Simulation mode is set by two flags before loading: `pengym_utils.ENABLE_PENGYM = False` and `pengym_utils.ENABLE_NASIM = True`. Without them, PenGym tries to connect to real machines.
- Two ways to load: `pengym.load("path/to/file.yml")` for a YAML file, or `pengym.create_environment("tiny")` for a NASim built-in benchmark by name.
- **Same name, different file:** a benchmark name loads NASim's original version, while a PenGym YAML path loads PenGym's edited version (Linux-only, different processes, exploit prob 0.999999 vs 0.9). Always note which one an experiment used.
- PenGym prints a line per action. `smoke_test.py` hides it with `redirect_stdout`; training loops should do the same or logs will flood.

### Stepping

- Actions must be plain Python ints. `env.action_space.sample()` returns a NumPy int, which NASim rejects. Use `int(...)`.
- `env.step()` returns 5 values: `obs, reward, done, truncated, info`. `done` = goal reached, `truncated` = step limit hit.
- **Step limits:** `smoke_test.py` caps episodes at 1000 (`--max-steps`), but scenario YAMLs set their own `step_limit` (2000 for medium-multi-site). Pick one deliberately for training and evaluation.

### Reproducibility (IMPORTANT)

NASim rolls exploit success with NumPy's global RNG, so seed all three:

```python
np.random.seed(seed)
env.action_space.seed(seed)
env.reset(seed=seed)
```

**Be sure to write down what your seeds are on any given run**, so we can reproduce the outcome. Also helps with shaped vs sparse reward.

### Environment behavior

- **Actions only name a target, not a source.** The environment checks whether any compromised host (or the internet) can reach the target. The same action can fail one step and succeed later because the network state changed.
- **Scans likely ignore firewalls; exploits don't.** Scans need only a topology connection. Exploits also need the firewall to allow their service. A successful scan does not mean the exploit will work.
- **Firewall rules are not observable.** The agent only sees connection errors and the reachable flag.
- **0 in the observation is ambiguous:** it can mean "not present" or "not observed yet."
- **Exploits don't require scanning first.** On a fixed scenario, a trained agent can memorize the network and skip scans entirely.
- **Most actions are invalid at any step.** `env.get_action_mask()` can filter them, but masking redundant actions removes what the shaped reward is meant to teach. Decide as a team.
- **YAML header comments can be wrong** (e.g., "2 OS" when Windows is disabled). Trust the actual fields.

### To verify before building the agent

- [ ] **Does the observation keep past info?** Scan a host, take an unrelated action, check `env.last_obs.get_readable()`. If the scan info is gone, the agent needs its own memory.
- [ ] **Do scans ignore firewalls?** In tiny: `e_ssh (1, 0)` → subnet scan `(1, 0)` → service scan `(2, 0)`. Success = scans ignore firewalls.
- [ ] **Flat action order within a host:**
  ```python
  for i in range(12):
      print(i, env.action_space.get_action(i))
  ```
- [ ] **Do generated benchmarks work through PenGym?**
  ```bash
  python scripts/smoke_test.py --scenario tiny-gen
  ```

### Evaluation

Training and testing on the same scenario lets the agent memorize a path instead of learning to explore. To test real exploration, evaluate on networks it hasn't trained on.
