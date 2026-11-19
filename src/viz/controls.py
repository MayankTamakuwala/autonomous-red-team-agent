"""The bottom bar: playback buttons, speed, mode switches, and the episode timeline."""

from dataclasses import dataclass

import pygame

from src.viz import theme
from src.viz.session import HostStatus

BUTTON = 34  # size of the square icon buttons


@dataclass
class Button:
    """A clickable control. `name` is what the app does when it is clicked."""

    name: str
    rect: pygame.Rect
    label: str = ""  # text, if the button has no icon
    icon: str = ""  # start, back, play, pause, forward, or end
    active: bool = False


@dataclass
class BarState:  # pylint: disable=too-many-instance-attributes
    """What the bottom bar needs to know to draw itself."""

    frames: list
    cursor: int
    playing: bool
    speed: float
    manual: bool
    agent_label: str
    can_switch_agent: bool
    sensitive: dict


class ControlBar:
    """Draws the bottom bar and reports which control is under the mouse."""

    def __init__(self):
        self.buttons = []
        self.track = pygame.Rect(0, 0, 1, 1)

    def draw(self, surface, rect, state, mouse):
        """Draw the bar in rect for the given state."""
        pygame.draw.rect(surface, theme.PANEL, rect)
        pygame.draw.line(surface, theme.PANEL_EDGE, rect.topleft, rect.topright)
        self.buttons = self._build(rect, state)
        for button in self.buttons:
            self._draw_button(surface, button, button.rect.collidepoint(mouse))
        self._draw_track(surface, state, mouse)

    def button_at(self, pos):
        """Name of the button at pos, or None."""
        return next((b.name for b in self.buttons if b.rect.collidepoint(pos)), None)

    def step_at(self, pos, n_frames, force=False):
        """The frame index under pos on the timeline, or None (force: clamp to the track)."""
        if not force and not self.track.inflate(0, 16).collidepoint(pos):
            return None
        share = (pos[0] - self.track.x) / max(1, self.track.width)
        return round(min(1.0, max(0.0, share)) * max(0, n_frames - 1))

    def _build(self, rect, state):
        y = rect.centery - BUTTON // 2
        x = rect.x + 16
        buttons = []
        play_icon = "pause" if state.playing else "play"
        for name, icon in [
            ("start", "start"),
            ("back", "back"),
            ("play", play_icon),
            ("forward", "forward"),
            ("end", "end"),
        ]:
            buttons.append(Button(name, pygame.Rect(x, y, BUTTON, BUTTON), icon=icon))
            x += BUTTON + 6
        x += 10
        buttons.append(Button("slower", pygame.Rect(x, y, 26, BUTTON), label="-"))
        buttons.append(
            Button("speed", pygame.Rect(x + 28, y, 54, BUTTON), label=f"{state.speed:g}x")
        )
        buttons.append(Button("faster", pygame.Rect(x + 84, y, 26, BUTTON), label="+"))
        x += 126
        mode = "MANUAL" if state.manual else "AGENT"
        width = theme.text_width(f"MODE: {mode}", 12) + 22
        buttons.append(
            Button("mode", pygame.Rect(x, y, width, BUTTON), f"MODE: {mode}", active=state.manual)
        )
        x += width + 8
        if state.can_switch_agent:
            label = state.agent_label.upper()
            width = theme.text_width(label, 12) + 22
            buttons.append(Button("agent", pygame.Rect(x, y, width, BUTTON), label))
            x += width + 8
        self.track = pygame.Rect(x + 14, rect.centery - 14, rect.right - x - 34, 28)
        return buttons

    @staticmethod
    def _draw_button(surface, button, hovered):
        fill = theme.mix(theme.PANEL, theme.PANEL_EDGE, 1.6 if hovered else 1.0)
        edge = theme.ATTACKER if button.active else theme.PANEL_EDGE
        pygame.draw.rect(surface, fill, button.rect, border_radius=6)
        pygame.draw.rect(surface, edge, button.rect, width=1, border_radius=6)
        color = theme.TEXT if hovered or button.active else theme.mix(theme.TEXT, theme.MUTED, 0.4)
        if button.icon:
            _draw_icon(surface, button.icon, button.rect.center, color)
        else:
            theme.draw_text(surface, button.label, 12, color, button.rect.center, "center", True)

    def _draw_track(self, surface, state, mouse):
        track, frames = self.track, state.frames
        pygame.draw.rect(surface, theme.BACKGROUND, track.inflate(8, 10), border_radius=6)
        last = len(frames) - 1
        if last < 1:
            theme.draw_text(surface, "no steps yet", 11, theme.MUTED, track.center, "center")
            return

        def x_of(i):
            return track.x + track.width * i / last

        totals = [f.total for f in frames]
        low, high = min(totals), max(totals)
        span = high - low or 1.0
        points = [
            (x_of(i), track.bottom - (t - low) / span * track.height) for i, t in enumerate(totals)
        ]
        if low < 0 < high:
            zero = track.bottom - (0 - low) / span * track.height
            pygame.draw.line(surface, theme.FAINT, (track.x, zero), (track.right, zero))
        theme.polyline(surface, theme.mix(theme.BACKGROUND, theme.GOLD, 0.35), points, 1)
        played = points[: state.cursor + 1]
        theme.polyline(surface, theme.GOLD, played, 2)

        for i, frame in enumerate(frames):
            if frame.gained == HostStatus.ROOT and frame.action.target in state.sensitive:
                x = x_of(i)
                pygame.draw.line(surface, theme.GOLD, (x, track.y - 4), (x, track.y + 4), 2)

        x = x_of(state.cursor)
        pygame.draw.line(surface, theme.TEXT, (x, track.y - 6), (x, track.bottom + 6), 2)
        theme.filled_circle(surface, theme.TEXT, (x, track.y - 6), 4)
        end = "goal reached" if frames[-1].done else "step limit" if frames[-1].truncated else ""
        caption = f"step {state.cursor} of {last}" + (f" · {end}" if end else "")
        theme.draw_text(
            surface, caption, 11, theme.MUTED, (track.right, track.bottom + 7), "topright"
        )
        hovered = self.step_at(mouse, len(frames))
        if hovered is not None and track.inflate(0, 16).collidepoint(mouse):
            theme.draw_text(
                surface, f"step {hovered}", 11, theme.TEXT, (mouse[0], track.y - 8), "midbottom"
            )


def _draw_icon(surface, icon, center, color):
    x, y = center
    if icon in ("play", "forward"):
        size = 8 if icon == "play" else 6
        pygame.draw.polygon(
            surface, color, [(x - size + 2, y - size), (x - size + 2, y + size), (x + size, y)]
        )
    elif icon == "back":
        pygame.draw.polygon(surface, color, [(x + 4, y - 6), (x + 4, y + 6), (x - 8, y)])
    elif icon == "pause":
        pygame.draw.rect(surface, color, (x - 6, y - 7, 4, 14))
        pygame.draw.rect(surface, color, (x + 2, y - 7, 4, 14))
    elif icon == "start":
        pygame.draw.rect(surface, color, (x - 7, y - 6, 3, 12))
        pygame.draw.polygon(surface, color, [(x + 6, y - 6), (x + 6, y + 6), (x - 3, y)])
    elif icon == "end":
        pygame.draw.polygon(surface, color, [(x - 6, y - 6), (x - 6, y + 6), (x + 3, y)])
        pygame.draw.rect(surface, color, (x + 4, y - 6, 3, 12))
