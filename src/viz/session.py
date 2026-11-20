"""Runs an episode step by step and records what happened, for the GUI to play back.

A Session keeps one Frame per step: what every host looked like afterwards, the action, its
result and reward, and where a remote action was launched from. The agent replay records a
whole episode up front; manual mode records one frame per click. Both produce the same frames,
so playback, stepping back, and the timeline work the same way in either mode.

Changing the past (undo, or taking control in the middle of a replay) resets the env with the
same seed and replays the recorded actions. NASim draws its randomness from NumPy's global
generator, so the same seed and the same actions always give the same episode.

Usage:
    session = Session(SCENARIO_DIR / "tiny.yml", seed=0)
    session.run(RandomPolicy(session.n_actions, seed=0))
    for frame in session.frames: ...
"""

import contextlib
import io
import random
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch

from src.agent.dqn_agent import DQNAgent
from src.env.env_setup import make_env

INTERNET = "internet"  # source of remote actions sent straight from the attacker
ROOT_ACCESS = 2  # NASim's AccessLevel.ROOT


class HostStatus(IntEnum):
    """What the attacker has achieved on a host, from nothing to full control."""

    UNKNOWN = 0  # the attacker doesn't know the host exists
    DISCOVERED = 1  # known to exist, but not compromised
    USER = 2  # compromised with user access
    ROOT = 3  # compromised with root access


@dataclass(frozen=True)
class ActionInfo:  # pylint: disable=too-many-instance-attributes
    """One entry of the flat action space, in plain Python types."""

    index: int
    kind: str  # service_scan, os_scan, subnet_scan, process_scan, exploit, or privesc
    name: str  # NASim's name, e.g. "e_ssh" or "subnet_scan"
    target: tuple
    cost: float
    remote: bool  # sent over the network; otherwise it runs on the target host itself
    service: str = ""  # exploits only
    process: str = ""  # privilege escalations only

    @property
    def label(self):
        """Short text such as "e_ssh (2,0)"."""
        return f"{self.name} {address_label(self.target)}"


@dataclass(frozen=True)
class Frame:  # pylint: disable=too-many-instance-attributes
    """The network after one step, and what that step did. Frame 0 is the start of the episode."""

    step: int
    hosts: dict  # address -> HostStatus
    action: ActionInfo = None  # None for frame 0
    success: bool = True
    failure: str = ""  # why the action failed, in words
    reward: float = 0.0
    total: float = 0.0
    redundant: bool = False  # the agent's knowledge did not change (same definition as #16)
    redundant_count: int = 0  # redundant actions so far
    failed_count: int = 0  # failed actions so far
    source: object = None  # INTERNET or the host address a remote action came from
    revealed: tuple = ()  # hosts this action discovered
    gained: HostStatus = None  # new access on the target, if the action gave any
    learned: dict = None  # facts this action revealed about its target, e.g. {"services": ...}
    done: bool = False  # every sensitive host has root access
    truncated: bool = False  # the step limit was reached
    hints: tuple = ()  # (ActionInfo, Q-value) pairs: the trained agent's top picks from here

    @property
    def over(self):
        """True if the episode ended at this frame."""
        return self.done or self.truncated


class Session:  # pylint: disable=too-many-instance-attributes
    """One episode in a scenario, with a Frame for every step taken so far."""

    def __init__(self, scenario, seed=0, max_steps=None, advisor=None):
        """
        Args:
            scenario: a NASim benchmark name or a path to a scenario .yml (see make_env).
            seed: seeds the env, so the same actions always give the same episode.
            max_steps: ends the episode early, like max_episode_steps in a training config.
            advisor: a DQNAgent whose top-scored actions are stored in every frame, or None.
        """
        self.scenario = scenario
        self.seed = seed
        self.env = make_env(scenario, seed=seed)
        self.sim = self.env.unwrapped
        self.network = self.sim.network
        self.addresses = list(self.network.address_space)
        self.n_actions = self.env.action_space.n
        self.actions = [
            describe_action(i, self.sim.action_space.get_action(i)) for i in range(self.n_actions)
        ]
        limit = self.sim.scenario.step_limit
        self.max_steps = min(max_steps, limit) if max_steps else limit
        self.advisor = advisor
        self.frames = []
        self.obs = None
        self._footholds = []  # compromised hosts, oldest first
        self.restart()

    @property
    def head(self):
        """The latest frame."""
        return self.frames[-1]

    @property
    def finished(self):
        """True once the episode has ended."""
        return self.head.over

    def actions_on(self, address):
        """Every action that targets the given host, in action-space order."""
        return [a for a in self.actions if a.target == address]

    def restart(self, seed=None):
        """Start the episode over (optionally with a new seed), keeping only frame 0."""
        if seed is not None:
            self.seed = seed
        np.random.seed(self.seed)
        self.env.action_space.seed(self.seed)
        self.obs, _ = self.env.reset(seed=self.seed)
        self._footholds = []
        self.frames = [Frame(step=0, hosts=self._statuses(), hints=self._hints())]

    def step(self, index):
        """Take one action, record its frame, and return it."""
        if self.finished:
            raise RuntimeError("The episode is over: restart or rewind before acting")
        action, before = self.actions[index], self.head
        knowledge = self.env.knowledge[:-1].copy()
        source = self._source(action)

        with contextlib.redirect_stdout(io.StringIO()):  # PenGym prints a line per action
            self.obs, reward, done, truncated, info = self.env.step(index)

        hosts = self._statuses()
        unknown = HostStatus.UNKNOWN
        revealed = tuple(
            a for a in self.addresses if before.hosts[a] == unknown and hosts[a] != unknown
        )
        old, new = before.hosts[action.target], hosts[action.target]
        gained = new if new > old and new >= HostStatus.USER else None
        if gained and action.target not in self._footholds:
            self._footholds.append(action.target)

        success = bool(info["success"])
        redundant = bool(np.array_equal(knowledge, self.env.knowledge[:-1]))
        frame = Frame(
            step=before.step + 1,
            hosts=hosts,
            action=action,
            success=success,
            failure="" if success else self._failure_reason(action, info, before),
            reward=float(reward),
            total=before.total + float(reward),
            redundant=redundant,
            redundant_count=before.redundant_count + redundant,
            failed_count=before.failed_count + (not success),
            source=source,
            revealed=revealed,
            gained=gained,
            learned=_learned(info) if success else None,
            done=bool(done),
            truncated=bool(truncated) or (not done and before.step + 1 >= self.max_steps),
            hints=self._hints(),
        )
        self.frames.append(frame)
        return frame

    def run(self, policy):
        """Let a policy act until the episode ends. Returns the frames it added."""
        start = len(self.frames)
        while not self.finished:
            self.step(policy(self.obs))
        return self.frames[start:]

    def rewind(self, step):
        """Go back to the given step, forgetting every later frame."""
        actions = [frame.action.index for frame in self.frames[1 : step + 1]]
        self.restart()
        for index in actions:
            self.step(index)

    def host_truth(self, address):
        """The hidden ground truth about a host: what the attacker could find by scanning."""
        host = self.network.hosts[address]
        return {
            "os": _present(host.os),
            "services": _present(host.services),
            "processes": _present(host.processes),
            "value": float(host.value),
        }

    def link_rules(self, a, b):
        """Services the subnet firewall lets through, in each direction (None if unlinked)."""
        firewall = self.network.firewall
        if (a, b) not in firewall and (b, a) not in firewall:
            return None
        return list(firewall.get((a, b), [])), list(firewall.get((b, a), []))

    def _statuses(self):
        state = self.sim.current_state
        statuses = {}
        for address in self.addresses:
            host = state.get_host(address)
            if host.access >= ROOT_ACCESS:
                statuses[address] = HostStatus.ROOT
            elif host.compromised:
                statuses[address] = HostStatus.USER
            elif host.discovered:
                statuses[address] = HostStatus.DISCOVERED
            else:
                statuses[address] = HostStatus.UNKNOWN
        return statuses

    def _source(self, action):
        """Where a remote action is launched from, using PenGym's own firewall checks.

        The internet if it can reach the target; otherwise the most recently compromised host
        that can. If nothing can (the action will fail), the closest foothold or the internet.
        """
        if not action.remote:
            return None
        network, state = self.network, self.sim.current_state
        subnet = action.target[0]
        if network.subnet_public(subnet) and (
            action.kind != "exploit" or action.service in network.firewall.get((0, subnet), [])
        ):
            return INTERNET
        footholds = [
            a for a in reversed(self._footholds) if a != action.target and state.host_compromised(a)
        ]
        for source in footholds:
            if action.kind != "exploit":
                if network.subnets_connected(source[0], subnet):
                    return source
            elif network.subnet_traffic_permitted(
                source[0], subnet, action.service
            ) and network.host_traffic_permitted(source, action.target, action.service):
                return source
        nearby = [a for a in footholds if network.subnets_connected(a[0], subnet)]
        return nearby[0] if nearby else INTERNET

    def _failure_reason(self, action, info, before):
        """Why an action failed, in words, from NASim's error flags."""
        if info.get("connection_error"):
            return self._connection_failure(action, before)
        if info.get("permission_error"):
            return "no foothold can reach the target"
        if info.get("undefined_error"):
            return "failed by chance"
        if action.kind in ("exploit", "privesc"):
            return f"target not vulnerable to {action.name}"
        return "action failed"

    def _connection_failure(self, action, before):
        """NASim reports several different problems as a connection error; tell them apart."""
        if before.hosts[action.target] == HostStatus.UNKNOWN:
            return "target not discovered yet"
        if action.kind == "privesc":
            return "needs a shell on the target first"
        if action.kind == "exploit" and self.sim.current_state.host_reachable(action.target):
            return f"firewall blocks {action.service}"
        return "target not reachable yet"

    def _hints(self, k=3):
        if self.advisor is None:
            return ()
        with torch.no_grad():
            obs = torch.as_tensor(self.obs, dtype=torch.float32, device=self.advisor.device)
            scores = self.advisor.q_net(obs)
        top = torch.topk(scores, min(k, self.n_actions))
        return tuple((self.actions[int(i)], float(v)) for v, i in zip(top.values, top.indices))


class RandomPolicy:
    """Picks actions uniformly at random, from its own seeded generator."""

    label = "random agent"

    def __init__(self, n_actions, seed=0):
        self.n_actions = n_actions
        self.rng = random.Random(seed)

    def __call__(self, obs):
        return self.rng.randrange(self.n_actions)


class AgentPolicy:
    """Acts with a trained DQNAgent, epsilon-greedy (epsilon=0 always takes the best action)."""

    def __init__(self, agent, epsilon=0.0, label="trained agent"):
        self.agent = agent
        self.epsilon = epsilon
        self.label = label

    def __call__(self, obs):
        return self.agent.act(obs, self.epsilon)


def load_agent(path, env, device="cpu"):
    """Build a DQNAgent that fits a saved checkpoint and load it with DQNAgent.load.

    Checkpoints only hold weights, so the network's sizes are read from the weight shapes.

    Raises:
        ValueError: if the checkpoint was trained on a scenario with different sizes.
    """
    state = torch.load(path, map_location=device)
    matrices = [w for name, w in state.items() if name.endswith("weight") and w.dim() == 2]
    hidden_size, obs_size = matrices[0].shape
    n_actions = matrices[-1].shape[0]
    env_obs, env_actions = env.observation_space.shape[0], env.action_space.n
    if (obs_size, n_actions) != (env_obs, env_actions):
        raise ValueError(
            f"{path} does not fit this scenario: it was trained with observations of length "
            f"{obs_size} and {n_actions} actions, but this scenario has {env_obs} and "
            f"{env_actions}. Check that the config's scenario is the one it was trained on."
        )
    agent = DQNAgent(obs_size, n_actions, hidden_size=hidden_size, device=device)
    agent.load(path)
    agent.q_net.eval()
    return agent


def describe_action(index, action):
    """Turn a NASim action object into an ActionInfo."""
    if action.is_exploit():
        kind = "exploit"
    elif action.is_privilege_escalation():
        kind = "privesc"
    elif action.is_service_scan():
        kind = "service_scan"
    elif action.is_os_scan():
        kind = "os_scan"
    elif action.is_subnet_scan():
        kind = "subnet_scan"
    else:
        kind = "process_scan"
    return ActionInfo(
        index=index,
        kind=kind,
        name=action.name,
        target=tuple(int(x) for x in action.target),
        cost=float(action.cost),
        remote=bool(action.is_remote()),
        service=getattr(action, "service", None) or "",
        process=getattr(action, "process", None) or "",
    )


def address_label(address):
    """Compact host address text, e.g. "(2,0)"."""
    return f"({address[0]},{address[1]})"


def known_facts(frames, address):
    """What the attacker has learned about a host so far, by merging the frames' findings."""
    facts = {"os": set(), "services": set(), "processes": set()}
    for frame in frames:
        if frame.learned and frame.action.target == address:
            for kind, names in frame.learned.items():
                facts[kind].update(names)
    return {kind: sorted(names) for kind, names in facts.items()}


def _learned(info):
    """Facts a successful action revealed about its target, from NASim's result info."""
    learned = {kind: _present(info.get(kind) or {}) for kind in ("os", "services", "processes")}
    return {kind: names for kind, names in learned.items() if names}


def _present(flags):
    """Names whose flag is set, from a NASim dict such as {"ssh": True, "ftp": False}."""
    return [name for name, on in flags.items() if on]
