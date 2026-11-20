"""Tests for episode recording, rewinding, and model loading (src/viz/session.py)."""

import pytest
import torch

from src.agent.dqn_agent import DQNAgent
from src.env.env_setup import SCENARIO_DIR
from src.viz.session import INTERNET, HostStatus, RandomPolicy, Session, known_facts, load_agent

TINY = SCENARIO_DIR / "tiny.yml"

# The 6-step solution of PenGym's tiny: (2, 0) refuses ssh from (1, 0), so it is reached via (3, 0)
SOLUTION = [
    ("e_ssh", (1, 0)),
    ("subnet_scan", (1, 0)),
    ("e_ssh", (3, 0)),
    ("e_ssh", (2, 0)),
    ("pe_tomcat", (2, 0)),
    ("pe_tomcat", (3, 0)),
]


def action_index(session, name, target):
    """Flat index of the action with this name and target."""
    return next(a.index for a in session.actions if a.name == name and a.target == target)


def play_solution(session):
    """Take the solution's actions and return their frames."""
    return [session.step(action_index(session, *move)) for move in SOLUTION]


def test_start_frame_shows_only_internet_facing_hosts():
    """At reset only (1, 0), in the subnet that touches the internet, is discovered."""
    frame = Session(TINY).head
    assert frame.step == 0 and frame.action is None
    assert frame.hosts == {
        (1, 0): HostStatus.DISCOVERED,
        (2, 0): HostStatus.UNKNOWN,
        (3, 0): HostStatus.UNKNOWN,
    }


def test_solution_frames():
    """The solution's frames record rewards, access gained, discoveries, and pivots."""
    session = Session(TINY)
    frames = play_solution(session)

    assert [f.total for f in frames] == [-1, -2, -3, -4, 95, 194]
    assert frames[0].source == INTERNET and frames[0].gained == HostStatus.USER
    assert set(frames[1].revealed) == {(2, 0), (3, 0)}
    assert frames[2].source == (1, 0)
    assert frames[3].source == (3, 0)  # the pivot around (2, 0)'s firewall
    assert frames[4].gained == HostStatus.ROOT and frames[4].source is None
    assert frames[-1].done and session.finished
    assert all(f.success for f in frames)
    assert frames[-1].redundant_count == 0 and frames[-1].failed_count == 0
    with pytest.raises(RuntimeError):
        session.step(0)


def test_failures_are_explained():
    """Failed actions say why: unknown target, missing foothold, or a firewall."""
    session = Session(TINY)
    unknown = session.step(action_index(session, "e_ssh", (2, 0)))
    assert not unknown.success and unknown.failure == "target not discovered yet"
    no_shell = session.step(action_index(session, "pe_tomcat", (1, 0)))
    assert no_shell.failure == "needs a shell on the target first"
    assert no_shell.redundant and no_shell.failed_count == 2


def test_rewind_reproduces_the_episode():
    """Going back and replaying the same actions gives exactly the same frames."""
    session = Session(SCENARIO_DIR / "small-honeypot.yml", seed=3, max_steps=300)
    session.run(RandomPolicy(session.n_actions, seed=3))
    original = [(f.action.index, f.success, f.reward, f.hosts) for f in session.frames[1:]]

    middle = len(original) // 2
    session.rewind(middle)
    assert len(session.frames) == middle + 1
    for index, *_ in original[middle:]:
        session.step(index)
    replayed = [(f.action.index, f.success, f.reward, f.hosts) for f in session.frames[1:]]
    assert replayed == original


def test_max_steps_ends_the_episode():
    """An episode that hits max_steps is truncated."""
    session = Session(TINY, max_steps=5)
    session.run(lambda obs: action_index(session, "os_scan", (1, 0)))
    assert len(session.frames) == 6 and session.head.truncated and not session.head.done


def test_known_facts_merge_scan_results():
    """Facts learned about a host accumulate over the frames."""
    session = Session(TINY)
    session.step(action_index(session, "service_scan", (1, 0)))
    session.step(action_index(session, "os_scan", (1, 0)))
    facts = known_facts(session.frames[1:], (1, 0))
    assert facts == {"os": ["linux"], "services": ["ssh"], "processes": []}


def test_load_agent_reads_sizes_from_the_checkpoint(tmp_path):
    """load_agent rebuilds the network from the saved weights, and hints use its Q-values."""
    session = Session(TINY)
    obs_size, n_actions = session.env.observation_space.shape[0], session.n_actions
    torch.manual_seed(0)
    saved = DQNAgent(obs_size, n_actions, hidden_size=32)
    path = tmp_path / "agent.pt"
    saved.save(path)

    agent = load_agent(path, session.env)
    assert agent.q_net.layers[0].out_features == 32
    session.advisor = agent
    session.restart()
    assert len(session.head.hints) == 3


def test_load_agent_rejects_another_scenarios_checkpoint(tmp_path):
    """A checkpoint with different observation or action sizes is refused clearly."""
    session = Session(TINY)
    path = tmp_path / "other.pt"
    DQNAgent(10, 4).save(path)
    with pytest.raises(ValueError, match="does not fit this scenario"):
        load_agent(path, session.env)
