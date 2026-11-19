"""Smoke tests for the GUI (src/viz/app.py), run without a screen.

SDL's dummy video driver gives pygame an invisible window, so the real drawing code runs in CI.
"""

import os

import pygame
import pytest

from src.env.env_setup import SCENARIO_DIR
from src.viz.app import App
from src.viz.session import RandomPolicy, Session

FRAME = 1 / 60


@pytest.fixture(name="app")
def app_fixture(monkeypatch):
    """The GUI on PenGym's tiny scenario with a random agent, in an invisible window."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    session = Session(SCENARIO_DIR / "tiny.yml", seed=0)
    gui = App(session, [RandomPolicy(session.n_actions, seed=0)], title="tiny")
    yield gui
    pygame.quit()


def ticks(gui, seconds):
    """Run the GUI for a while, as fast as possible."""
    for _ in range(round(seconds / FRAME)):
        gui.tick(FRAME)


def test_replay_plays_through_the_episode(app):
    """The agent's episode is recorded at startup and playback advances through it."""
    assert app.session.finished and app.playback.playing
    app.command("faster")
    app.command("faster")
    ticks(app, 3)
    assert app.playback.cursor > 1
    app.command("end")
    assert app.playback.cursor == len(app.session.frames) - 1


def test_manual_mode_acts_undoes_and_hands_back(app):
    """Taking control drops the rest of the replay; actions, undo, and takeover all work."""
    app.seek(0)
    app.command("mode")
    assert app.manual and len(app.session.frames) == 1

    host = (1, 0)
    position = [round(v) for v in app.map.host_position(host)]
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=position, button=1))
    ticks(app, FRAME)
    assert app.ui.menu is not None and app.ui.menu.address == host

    exploit = next(a for a in app.ui.menu.actions if a.kind == "exploit")
    app.act(exploit.index)
    ticks(app, 1.5)
    assert app.session.head.action == exploit and app.session.head.success

    app.command("undo")
    assert len(app.session.frames) == 1 and app.playback.cursor == 0

    app.command("takeover")
    assert not app.manual and app.session.finished


def test_every_view_draws(app):
    """Charts (without a log), help, and ground truth draw without errors."""
    app.ui.hover = (1, 0)
    app.command("truth")
    ticks(app, 0.2)
    app.command("view")
    ticks(app, 0.1)
    app.command("help")
    ticks(app, 0.1)
    assert os.environ["SDL_VIDEODRIVER"] == "dummy"
