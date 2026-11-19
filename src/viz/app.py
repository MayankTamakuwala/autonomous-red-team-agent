"""The demo GUI: attack replay on the radial network map, manual play, and training charts.

Start it with scripts/visualize.py. Two views, switched with Tab:
  - Attack replay: an episode played step by step on the network map. In manual mode you pick
    the actions yourself by clicking hosts, and can hand control back to the agent at any time.
  - Training charts: total reward and steps per episode from a training run's CSV.
"""

from dataclasses import dataclass, field

import pygame

from src.viz import hud, theme
from src.viz.charts import figure_pixels, training_figure
from src.viz.controls import BarState, ControlBar
from src.viz.layout import radial_layout
from src.viz.map_view import MapView, Scene

TOP_BAR = 44
BOTTOM_BAR = 72
PANEL_WIDTH = 390
STEP_SECONDS = 1.1  # one step's animation at 1x speed
PAUSE_SECONDS = 0.35  # pause between steps at 1x speed
SPEEDS = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
TOAST_SECONDS = 4.0

KEY_COMMANDS = {
    pygame.K_SPACE: "play",
    pygame.K_RIGHT: "forward",
    pygame.K_LEFT: "back",
    pygame.K_HOME: "start",
    pygame.K_END: "end",
    pygame.K_EQUALS: "faster",
    pygame.K_PLUS: "faster",
    pygame.K_KP_PLUS: "faster",
    pygame.K_MINUS: "slower",
    pygame.K_KP_MINUS: "slower",
    pygame.K_m: "mode",
    pygame.K_t: "takeover",
    pygame.K_BACKSPACE: "undo",
    pygame.K_a: "agent",
    pygame.K_n: "new",
    pygame.K_r: "replay",
    pygame.K_g: "truth",
    pygame.K_TAB: "view",
    pygame.K_f: "fullscreen",
    pygame.K_h: "help",
    pygame.K_SLASH: "help",
}
NUMBER_KEYS = [pygame.K_1 + i for i in range(9)] + [pygame.K_0]


@dataclass
class Playback:
    """Which frame is on screen, and how far its action's animation has played."""

    cursor: int = 0
    progress: float = 1.0  # 0 = the action just started, 1 = finished
    playing: bool = False
    speed_index: int = 2
    waited: float = 0.0  # time spent paused after the current step

    @property
    def speed(self):
        """Playback speed multiplier."""
        return SPEEDS[self.speed_index]


@dataclass
class Ui:  # pylint: disable=too-many-instance-attributes
    """Interface state that is not about playback."""

    view: str = "replay"  # "replay" or "charts"
    hover: tuple = None  # host under the mouse
    menu: hud.ActionMenu = None  # open action menu (manual mode)
    show_help: bool = False
    show_truth: bool = False
    dragging: bool = False  # dragging the timeline's playhead
    toast: str = ""
    toast_until: float = 0.0
    tabs: dict = field(default_factory=dict)  # view name -> tab rect


@dataclass
class Rects:
    """Screen regions, recomputed when the window changes size."""

    top: pygame.Rect
    map: pygame.Rect
    panel: pygame.Rect
    controls: pygame.Rect
    charts: pygame.Rect


class App:  # pylint: disable=too-many-instance-attributes
    """Window, input handling, and drawing for the whole GUI."""

    def __init__(self, session, policies, log=None, title="", manual=False, notice=""):
        """
        Args:
            session: the Session to replay and play in.
            policies: agents that A cycles through; the first one plays at startup.
            log: TrainingLog for the charts view, or None.
            title: scenario/run description for the window title and top bar.
            manual: start in manual mode instead of replaying the agent.
            notice: message shown briefly at startup (e.g. "no checkpoint found").
        """
        pygame.display.init()
        pygame.font.init()
        theme.reset_caches()
        pygame.display.set_caption(f"Red-team agent  ·  {title}")
        self.screen = pygame.display.set_mode((1280, 780), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.session = session
        self.policies = policies
        self.policy = policies[0]
        self.log = log
        self.title = title
        self.map = MapView(radial_layout(session.network), session)
        self.controls = ControlBar()
        self.playback = Playback()
        self.ui = Ui()
        self.manual = manual
        self.time = 0.0
        self.running = True
        self._chart = None  # (size, surface) of the rendered training charts
        self.rects = None
        self._resize()
        if not manual:
            self._record()
        if notice:
            self.toast(notice, seconds=10)

    # ----- main loop -----

    def run(self):
        """Run until the window is closed."""
        while self.running:
            self.tick(self.clock.tick(60) / 1000)
        pygame.quit()

    def tick(self, dt):
        """Handle input, advance playback by dt seconds, and draw one frame."""
        for event in pygame.event.get():
            self.handle(event)
        self.update(dt)
        self.draw()
        pygame.display.flip()

    def update(self, dt):
        """Advance the animation and, while playing, move on to the next step."""
        self.time += dt
        playback, last = self.playback, len(self.session.frames) - 1
        playback.cursor = min(playback.cursor, last)
        if playback.progress < 1.0:
            playback.progress = min(1.0, playback.progress + dt * playback.speed / STEP_SECONDS)
        elif playback.playing:
            if playback.cursor >= last:
                playback.playing = False
                return
            playback.waited += dt * playback.speed
            if playback.waited >= PAUSE_SECONDS:
                playback.waited = 0.0
                playback.cursor += 1
                playback.progress = 0.0

    # ----- input -----

    def handle(self, event):
        """React to one pygame event."""
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type in (pygame.VIDEORESIZE, pygame.WINDOWSIZECHANGED):
            self._resize()
        elif event.type == pygame.KEYDOWN:
            self._on_key(event.key)
        elif event.type == pygame.MOUSEMOTION:
            self._on_motion(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._on_click(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.ui.dragging = False

    def _on_key(self, key):
        if key == pygame.K_ESCAPE:
            if self.ui.menu or self.ui.show_help:
                self.ui.menu, self.ui.show_help = None, False
            else:
                self.running = False
        elif self.ui.menu and key in NUMBER_KEYS:
            row = NUMBER_KEYS.index(key)
            if row < len(self.ui.menu.actions):
                self.act(self.ui.menu.actions[row].index)
        elif key in KEY_COMMANDS:
            self.command(KEY_COMMANDS[key])

    def _on_motion(self, pos):
        if self.ui.dragging:
            self.seek(self.controls.step_at(pos, len(self.session.frames), force=True))
        in_map = self.ui.view == "replay" and self.rects.map.collidepoint(pos)
        self.ui.hover = self.map.host_at(pos) if in_map else None

    def _on_click(self, pos):
        if self.ui.show_help:
            self.ui.show_help = False
            return
        for view, rect in self.ui.tabs.items():
            if rect.collidepoint(pos):
                self.ui.view = view
                return
        if self.ui.view != "replay":
            return
        if self.ui.menu:
            action = self.ui.menu.action_at(pos)
            inside = self.ui.menu.rect.collidepoint(pos)
            self.ui.menu = None
            if action:
                self.act(action.index)
            if inside:
                return
        self._click_replay(pos)

    def _click_replay(self, pos):
        name = self.controls.button_at(pos)
        if name:
            self.command(name)
            return
        step = self.controls.step_at(pos, len(self.session.frames))
        if step is not None:
            self.seek(step)
            self.ui.dragging = True
            return
        host = self.map.host_at(pos) if self.rects.map.collidepoint(pos) else None
        if host is None:
            return
        if self.manual:
            self.ui.menu = hud.ActionMenu(
                host, self.session.actions_on(host), self.map.host_position(host), self.rects.map
            )
        else:
            self.toast("Press M (manual mode) to choose actions yourself")

    def command(self, name):
        """Run a named command, from a key or a button."""
        commands = {
            "play": self._toggle_play,
            "forward": self._step_forward,
            "back": self._step_back,
            "start": lambda: self.seek(0),
            "end": lambda: self.seek(len(self.session.frames) - 1),
            "faster": lambda: self._change_speed(1),
            "slower": lambda: self._change_speed(-1),
            "speed": lambda: self._change_speed(1),
            "mode": self._toggle_mode,
            "takeover": self._agent_takeover,
            "undo": self._undo,
            "agent": self._switch_agent,
            "new": self._new_episode,
            "replay": self._replay,
            "truth": self._toggle_truth,
            "view": self._toggle_view,
            "fullscreen": self._toggle_fullscreen,
            "help": self._toggle_help,
        }
        commands[name]()

    # ----- playback commands -----

    def _toggle_play(self):
        """Play or pause. At the end of a replay, play starts over from the beginning."""
        playback = self.playback
        if playback.playing:
            playback.playing = False
            return
        if playback.cursor >= len(self.session.frames) - 1:
            if self.manual:
                self.toast("Manual mode: click a host to act, or T to let the agent continue")
                return
            playback.cursor = 0
        playback.playing = True

    def _step_forward(self):
        """Show the next step, animated."""
        self.playback.playing = False
        if self.playback.cursor < len(self.session.frames) - 1:
            self.playback.cursor += 1
            self.playback.progress = 0.0

    def _step_back(self):
        """Show the previous step."""
        self.playback.playing = False
        if self.playback.cursor > 0:
            self.playback.cursor -= 1
            self.playback.progress = 1.0

    def seek(self, index):
        """Jump to a step without animating it."""
        last = len(self.session.frames) - 1
        self.playback.playing = False
        self.playback.cursor = max(0, min(index, last))
        self.playback.progress = 1.0

    def _change_speed(self, delta):
        """Faster (delta > 0) or slower playback."""
        index = self.playback.speed_index + delta
        self.playback.speed_index = max(0, min(index, len(SPEEDS) - 1))

    def _replay(self):
        """Play the episode again from the start."""
        self.playback.cursor, self.playback.progress, self.playback.playing = 0, 1.0, True

    # ----- agent and manual mode -----

    def _toggle_mode(self):
        """Manual mode on (from the step on screen) or off (the agent takes over)."""
        if self.manual:
            self._agent_takeover()
        else:
            self._take_control()

    def _take_control(self):
        """Switch to manual mode at the step on screen; later steps of the replay are dropped."""
        self._continue_from_cursor()
        self.manual = True
        self.playback.playing = False
        self.toast("Manual mode: click a host and pick an action. T hands back to the agent.")

    def _agent_takeover(self):
        """Let the current agent play the rest of the episode from the step on screen."""
        self._continue_from_cursor()
        self.manual, self.ui.menu = False, None
        if not self.session.finished:
            self.session.run(self.policy)
            self.toast(f"The {self.policy.label} takes over from step {self.playback.cursor}")
        self.playback.playing = True

    def act(self, index):
        """Take an action picked in manual mode, and animate it."""
        self.ui.menu = None
        self._continue_from_cursor()
        if self.session.finished:
            self.toast("The episode is over: Backspace to undo, N for a new episode")
            return
        self.session.step(index)
        self.playback.cursor = len(self.session.frames) - 1
        self.playback.progress, self.playback.playing = 0.0, False

    def _undo(self):
        """Take back the last manual action."""
        if not self.manual:
            self.toast("Undo works in manual mode (M)")
            return
        step = self.playback.cursor - 1
        if step < 0:
            return
        self.session.rewind(step)
        self.ui.menu = None
        self.seek(step)

    def _switch_agent(self):
        """Cycle through the available agents and replay the episode with the next one."""
        if len(self.policies) < 2:
            self.toast("Only the random agent is available: no trained checkpoint was loaded")
            return
        position = self.policies.index(self.policy)
        self.policy = self.policies[(position + 1) % len(self.policies)]
        if self.manual:
            self.toast(f"T now hands control to the {self.policy.label}")
        else:
            self._record()
            self.toast(f"Replaying with the {self.policy.label}")

    def _new_episode(self):
        """Start a new episode with the next seed."""
        self.session.restart(self.session.seed + 1)
        self.ui.menu = None
        if self.manual:
            self.seek(0)
        else:
            self._record()

    def _record(self):
        """Record the whole episode with the current agent and play it from the start."""
        self.session.restart()
        self.session.run(self.policy)
        self.playback = Playback(speed_index=self.playback.speed_index, playing=True)

    def _continue_from_cursor(self):
        """Forget the steps after the one on screen, so the episode continues from it."""
        if self.playback.cursor < len(self.session.frames) - 1:
            self.session.rewind(self.playback.cursor)
        self.playback.progress = 1.0

    # ----- view commands -----

    def _toggle_truth(self):
        """Show or hide the hidden ground truth in host tooltips."""
        self.ui.show_truth = not self.ui.show_truth
        self.toast("Ground truth " + ("on: hover a host" if self.ui.show_truth else "off"))

    def _toggle_view(self):
        """Switch between the replay and the training charts."""
        self.ui.view = "charts" if self.ui.view == "replay" else "replay"
        self.ui.menu = None

    def _toggle_help(self):
        """Show or hide the key bindings."""
        self.ui.show_help = not self.ui.show_help

    def _toggle_fullscreen(self):
        """Switch between a window and fullscreen."""
        pygame.display.toggle_fullscreen()
        self._resize()

    def toast(self, text, seconds=TOAST_SECONDS):
        """Show a short message at the top of the screen."""
        self.ui.toast, self.ui.toast_until = text, self.time + seconds

    # ----- drawing -----

    def _resize(self):
        width, height = self.screen.get_size()
        middle = height - TOP_BAR - BOTTOM_BAR
        self.rects = Rects(
            top=pygame.Rect(0, 0, width, TOP_BAR),
            map=pygame.Rect(0, TOP_BAR, width - PANEL_WIDTH, middle),
            panel=pygame.Rect(width - PANEL_WIDTH, TOP_BAR, PANEL_WIDTH, middle),
            controls=pygame.Rect(0, height - BOTTOM_BAR, width, BOTTOM_BAR),
            charts=pygame.Rect(24, TOP_BAR + 16, width - 48, height - TOP_BAR - 32),
        )
        self.map.fit(self.rects.map, padding=20)
        self.ui.menu = None

    def draw(self):
        """Draw the current view."""
        self.screen.fill(theme.BACKGROUND)
        if self.ui.view == "replay":
            self._draw_replay()
        else:
            self._draw_charts()
        self._draw_top_bar()
        if self.ui.show_help:
            hud.draw_help(self.screen, self.screen.get_rect())
        if self.time < self.ui.toast_until:
            self._draw_toast()

    def _draw_replay(self):
        frames, playback = self.session.frames, self.playback
        mouse = pygame.mouse.get_pos()
        frame = frames[playback.cursor]
        before = frames[playback.cursor - 1] if playback.cursor > 0 else frame
        selected = self.ui.menu.address if self.ui.menu else None
        progress = playback.progress if playback.cursor > 0 else 1.0
        scene = Scene(before, frame, progress, self.time, self.ui.hover, selected)
        self.map.draw(self.screen, scene)
        self._draw_mode_badge()

        sensitive = self.map.sensitive
        hud.draw_panel(
            self.screen,
            self.rects.panel,
            hud.PanelState(
                frames, playback.cursor, self.title, self.policy.label, self.manual, sensitive
            ),
        )
        state = BarState(
            frames=frames,
            cursor=playback.cursor,
            playing=playback.playing,
            speed=playback.speed,
            manual=self.manual,
            agent_label=self.policy.label,
            can_switch_agent=len(self.policies) > 1,
            sensitive=sensitive,
        )
        self.controls.draw(self.screen, self.rects.controls, state, mouse)
        if self.ui.menu:
            self.ui.menu.draw(self.screen, mouse, frame.hosts[self.ui.menu.address])
        elif self.ui.hover:
            hud.draw_tooltip(
                self.screen,
                self.map.host_position(self.ui.hover),
                self.ui.hover,
                self.session,
                frames[: playback.cursor + 1],
                self.ui.show_truth,
                self.rects.map,
            )

    def _draw_mode_badge(self):
        x, y = self.rects.map.x + 16, self.rects.map.y + 14
        if self.manual:
            text, color = "MANUAL  ·  click a host to act  ·  T: agent takes over", theme.ATTACKER
        else:
            text, color = "M: take control  ·  H: help", theme.MUTED
        theme.draw_text(self.screen, text, 12, color, (x, y), bold=self.manual)

    def _draw_top_bar(self):
        rect = self.rects.top
        pygame.draw.rect(self.screen, theme.PANEL, rect)
        pygame.draw.line(self.screen, theme.PANEL_EDGE, rect.bottomleft, rect.bottomright)
        x = rect.x + 12
        for view, label in (("replay", "ATTACK REPLAY"), ("charts", "TRAINING CHARTS")):
            width = theme.text_width(label, 13, True) + 28
            tab = pygame.Rect(x, rect.y + 7, width, rect.height - 14)
            active = self.ui.view == view
            if active:
                pygame.draw.rect(self.screen, theme.PANEL_EDGE, tab, border_radius=6)
            color = theme.TEXT if active else theme.MUTED
            theme.draw_text(self.screen, label, 13, color, tab.center, "center", bold=True)
            self.ui.tabs[view] = tab
            x += width + 6
        info = f"{self.title}   ·   Tab: switch view   ·   H: help"
        theme.draw_text(
            self.screen, info, 12, theme.MUTED, (rect.right - 16, rect.centery), "midright"
        )

    def _draw_charts(self):
        rect = self.rects.charts
        if self.log is None:
            lines = [
                "No training log for this run.",
                "Train one with:  python scripts/train.py --config configs/<name>.yml",
                "then open this view again (it reads results/<run_name>/episodes.csv).",
            ]
            for i, line in enumerate(lines):
                color = theme.TEXT if i == 0 else theme.MUTED
                theme.draw_text(
                    self.screen, line, 15, color, (rect.centerx, rect.centery + i * 28), "center"
                )
            return
        if self._chart is None or self._chart[0] != rect.size:
            figure = training_figure(self.log, size=(rect.width / 100, rect.height / 100))
            pixels, size = figure_pixels(figure)
            image = pygame.image.frombuffer(pixels, size, "RGBA").convert()
            self._chart = (rect.size, image)
        self.screen.blit(self._chart[1], rect)

    def _draw_toast(self):
        image = theme.font(13, True).render(self.ui.toast, True, theme.TEXT)
        box = image.get_rect().inflate(28, 16)
        box.midtop = (self.rects.map.centerx, self.rects.map.y + 40)
        theme.panel(self.screen, box, color=theme.PANEL, edge=theme.ATTACKER)
        self.screen.blit(image, image.get_rect(center=box.center))
