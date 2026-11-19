"""Text-heavy parts of the replay screen: side panel, host tooltip, action menu, and help."""

from dataclasses import dataclass

import pygame

from src.viz import theme
from src.viz.session import INTERNET, HostStatus, address_label, known_facts

ROW = 20  # height of a log or menu row


@dataclass
class PanelState:
    """What the side panel shows."""

    frames: list
    cursor: int
    scenario: str
    agent_label: str
    manual: bool
    sensitive: dict


def draw_panel(surface, rect, state):
    """The side panel: step, total, current action and result, agent's picks, log, legend."""
    pygame.draw.rect(surface, theme.PANEL, rect)
    pygame.draw.line(surface, theme.PANEL_EDGE, rect.topleft, rect.bottomleft)
    x, width = rect.x + 20, rect.width - 40
    frame = state.frames[state.cursor]
    last = state.frames[-1]

    mode = "MANUAL  ·  you are the attacker" if state.manual else f"REPLAY  ·  {state.agent_label}"
    _chip(surface, mode, (x, rect.y + 16), theme.ATTACKER if state.manual else theme.MUTED)
    y = rect.y + 50

    theme.draw_text(surface, "STEP", 11, theme.MUTED, (x, y))
    theme.draw_text(surface, "TOTAL REWARD", 11, theme.MUTED, (x + width // 2, y))
    steps = f"{state.cursor} / {last.step}" if last.over else str(state.cursor)
    theme.draw_text(surface, steps, 26, theme.TEXT, (x, y + 15), bold=True)
    total_color = theme.GOLD if frame.total > 0 else theme.TEXT
    theme.draw_text(
        surface, f"{frame.total:+g}", 26, total_color, (x + width // 2, y + 15), bold=True
    )
    y += 58

    y = _draw_current_action(surface, x, y, width, frame, state)
    counts = f"redundant {frame.redundant_count}    failed {frame.failed_count}"
    theme.draw_text(surface, counts, 12, theme.MUTED, (x, y))
    y += 26

    if frame.hints:
        theme.draw_text(surface, "TRAINED AGENT WOULD PICK", 11, theme.MUTED, (x, y))
        y += 17
        for action, score in frame.hints:
            color = theme.KIND_COLORS[action.kind]
            theme.draw_text(surface, action.label, 12, color, (x + 8, y))
            theme.draw_text(surface, f"Q {score:6.1f}", 12, theme.MUTED, (x + width, y), "topright")
            y += 18
        y += 10

    legend_top = rect.bottom - 92
    _draw_log(surface, pygame.Rect(x, y, width, legend_top - y - 10), state)
    _draw_legend(surface, x, legend_top, width)


def _draw_current_action(surface, x, y, width, frame, state):
    box = pygame.Rect(x - 8, y, width + 16, 92)
    theme.panel(surface, box, color=theme.mix(theme.PANEL, theme.BACKGROUND, 0.5))
    if frame.action is None:
        known = sum(s != HostStatus.UNKNOWN for s in frame.hosts.values())
        theme.draw_text(surface, "Start of the episode", 14, theme.TEXT, (x + 4, y + 12), bold=True)
        hint = "click a host to act" if state.manual else "press Space to play"
        theme.draw_text(
            surface,
            f"{known} of {len(frame.hosts)} hosts visible",
            12,
            theme.MUTED,
            (x + 4, y + 36),
        )
        theme.draw_text(surface, hint, 12, theme.MUTED, (x + 4, y + 56))
        return y + box.height + 12

    action = frame.action
    kind_color = theme.KIND_COLORS[action.kind]
    _chip(surface, theme.KIND_NAMES[action.kind].upper(), (x + 4, y + 9), kind_color)
    theme.draw_text(surface, action.label, 15, theme.TEXT, (x + 4, y + 33), bold=True)
    if frame.source == INTERNET:
        origin = "sent from the internet"
    elif frame.source:
        origin = f"launched from {address_label(frame.source)}"
    else:
        origin = "run on the host itself"
    theme.draw_text(surface, origin, 12, theme.MUTED, (x + width // 2 + 8, y + 12))

    if frame.success:
        result = "success"
        if frame.gained:
            result += f" · {theme.STATUS_NAMES[frame.gained]}"
        if frame.revealed:
            result += f" · found {len(frame.revealed)} host" + (
                "s" if len(frame.revealed) > 1 else ""
            )
        color = theme.SUCCESS
    else:
        result, color = f"failed: {frame.failure}", theme.FAILURE
    theme.draw_text(surface, result, 12, color, (x + 4, y + 57))
    tag = "  (redundant)" if frame.redundant else ""
    theme.draw_text(surface, f"reward {frame.reward:+g}{tag}", 12, theme.MUTED, (x + 4, y + 73))
    if frame.over:
        end, end_color = (
            ("GOAL REACHED", theme.GOLD) if frame.done else ("STEP LIMIT", theme.FAILURE)
        )
        theme.draw_text(surface, end, 12, end_color, (x + width + 4, y + 73), "topright", bold=True)
    return y + box.height + 12


def _draw_log(surface, rect, state):
    if rect.height < ROW * 2:
        return
    theme.draw_text(surface, "LOG", 11, theme.MUTED, rect.topleft)
    rows = max(1, (rect.height - 18) // ROW)
    first = max(1, state.cursor - rows + 1)
    y = rect.y + 18
    for index in range(first, state.cursor + 1):
        frame = state.frames[index]
        current = index == state.cursor
        if current:
            pygame.draw.rect(
                surface, theme.PANEL_EDGE, (rect.x - 6, y - 2, rect.width + 12, ROW), 0, 4
            )
        dot = theme.SUCCESS if frame.success else theme.FAILURE
        if frame.redundant and frame.success:
            dot = theme.MUTED
        theme.filled_circle(surface, dot, (rect.x + 3, y + 8), 3)
        color = theme.TEXT if current else theme.mix(theme.TEXT, theme.MUTED, 0.5)
        theme.draw_text(surface, f"{index:>4} {frame.action.label}", 12, color, (rect.x + 12, y))
        reward_color = theme.GOLD if frame.reward > 0 else theme.MUTED
        theme.draw_text(
            surface, f"{frame.reward:+g}", 12, reward_color, (rect.right, y), "topright"
        )
        y += ROW


def _draw_legend(surface, x, y, width):
    pygame.draw.line(surface, theme.PANEL_EDGE, (x, y - 8), (x + width, y - 8))
    items = [
        (HostStatus.UNKNOWN, "unknown"),
        (HostStatus.DISCOVERED, "discovered"),
        (HostStatus.USER, "user shell"),
        (HostStatus.ROOT, "root"),
    ]
    column = width // 2
    for i, (status, name) in enumerate(items):
        cx, cy = x + 8 + (i % 2) * column, y + 10 + (i // 2) * 24
        draw_host_icon(surface, status, (cx, cy), 8)
        theme.draw_text(surface, name, 12, theme.TEXT, (cx + 16, cy), "midleft")
    cy = y + 58
    theme.ring(surface, theme.GOLD, (x + 8, cy), 8, 2)
    theme.draw_text(surface, "target (+value)", 12, theme.GOLD, (x + 24, cy), "midleft")
    theme.dashed_ring(surface, theme.HONEYPOT, (x + column + 8, cy), 8, dashes=8, width=2)
    theme.draw_text(surface, "honeypot", 12, theme.HONEYPOT, (x + column + 24, cy), "midleft")


def draw_host_icon(surface, status, center, radius):
    """A small host circle in the given state, as drawn on the map."""
    color = theme.STATUS_COLORS[status]
    if status == HostStatus.UNKNOWN:
        theme.dashed_ring(surface, color, center, radius, dashes=8, width=2)
    elif status == HostStatus.DISCOVERED:
        theme.ring(surface, color, center, radius, 2)
    else:
        theme.filled_circle(surface, color, center, radius)
    glyph = theme.STATUS_GLYPHS[status]
    if glyph:
        glyph_color = theme.MUTED if status == HostStatus.UNKNOWN else theme.BACKGROUND
        theme.draw_text(surface, glyph, 11, glyph_color, (center[0], center[1] + 1), "center", True)


def draw_tooltip(surface, host, address, session, frames, show_truth, bounds):
    """Details about a host, beside it: state, value, what's known, and the ground truth.

    host is the host's pixel position; the box stays inside bounds (the map area).
    """
    status = frames[-1].hosts[address]
    lines = [(f"host {address_label(address)}  ·  {theme.STATUS_NAMES[status]}", theme.TEXT)]
    truth = session.host_truth(address)
    if address in session.network.sensitive_hosts:
        lines.append((f"sensitive target worth {truth['value']:+g} once rooted", theme.GOLD))
    elif truth["value"] < 0:
        lines.append((f"honeypot: rooting it costs {truth['value']:+g}", theme.HONEYPOT))
    known = known_facts(frames[1:], address)
    for kind in ("os", "services", "processes"):
        names = ", ".join(known[kind]) or "?"
        lines.append((f"known {kind}: {names}", theme.MUTED))
    if show_truth:
        for kind in ("os", "services", "processes"):
            lines.append((f"actual {kind}: {', '.join(truth[kind]) or '-'}", theme.ATTACKER))
    else:
        lines.append(("G shows the hidden ground truth", theme.FAINT))

    width = max(theme.text_width(text, 12) for text, _ in lines) + 24
    rect = pygame.Rect(host[0] + 28, host[1] - 20, width, len(lines) * 18 + 14)
    if rect.right > bounds.right:  # no room on the right: put it on the left of the host
        rect.right = host[0] - 28
    rect.clamp_ip(bounds)
    theme.panel(surface, rect, color=theme.BACKGROUND, edge=theme.PANEL_EDGE)
    for i, (text, color) in enumerate(lines):
        theme.draw_text(surface, text, 12, color, (rect.x + 12, rect.y + 8 + i * 18))


class ActionMenu:
    """The list of actions for one host, shown when a host is clicked in manual mode."""

    def __init__(self, address, actions, anchor, bounds):
        self.address = address
        self.actions = actions
        width = max(theme.text_width(self._row_text(i, a), 12) for i, a in enumerate(actions)) + 70
        self.rect = pygame.Rect(anchor[0] + 24, anchor[1] - 30, width, 34 + len(actions) * ROW + 8)
        self.rect.clamp_ip(bounds)

    def action_at(self, pos):
        """The action in the row under pos, or None."""
        row = (pos[1] - self.rect.y - 34) // ROW
        if self.rect.collidepoint(pos) and 0 <= row < len(self.actions):
            return self.actions[row]
        return None

    def draw(self, surface, mouse, status):
        """Draw the menu; the row under the mouse is highlighted."""
        theme.panel(surface, self.rect, color=theme.BACKGROUND, edge=theme.ATTACKER)
        title = f"{address_label(self.address)}  ·  {theme.STATUS_NAMES[status]}"
        theme.draw_text(
            surface, title, 13, theme.TEXT, (self.rect.x + 12, self.rect.y + 10), bold=True
        )
        hovered = self.action_at(mouse)
        for i, action in enumerate(self.actions):
            y = self.rect.y + 34 + i * ROW
            if action == hovered:
                row_rect = (self.rect.x + 4, y - 1, self.rect.width - 8, ROW)
                pygame.draw.rect(surface, theme.PANEL_EDGE, row_rect, border_radius=4)
            color = theme.KIND_COLORS[action.kind]
            theme.draw_text(
                surface, self._row_text(i, action), 12, color, (self.rect.x + 12, y + 2)
            )
            theme.draw_text(
                surface,
                f"cost {action.cost:g}",
                11,
                theme.MUTED,
                (self.rect.right - 10, y + 3),
                "topright",
            )

    @staticmethod
    def _row_text(i, action):
        key = str(i + 1) if i < 9 else "0" if i == 9 else " "
        detail = action.service or action.process
        name = (
            action.name if action.kind in ("exploit", "privesc") else theme.KIND_NAMES[action.kind]
        )
        return f"{key}  {name}" + (f" ({detail})" if detail else "")


HELP = [
    ("Space", "play / pause"),
    ("Left / Right", "step back / forward"),
    ("Home / End", "jump to start / end"),
    ("+ / -", "faster / slower"),
    ("M", "manual mode: you pick the actions from here"),
    ("click a host", "(manual) choose an action for it"),
    ("Backspace", "(manual) undo the last action"),
    ("T", "(manual) let the agent take over from here"),
    ("A", "switch between the trained and random agent"),
    ("N / R", "new episode / replay from the start"),
    ("G", "show the hidden ground truth on hover"),
    ("Tab", "switch between replay and training charts"),
    ("F", "fullscreen"),
    ("H or ?", "show / hide this help"),
    ("Esc", "close menu or help, then quit"),
]


def draw_help(surface, rect):
    """Key bindings, drawn over the middle of rect."""
    box = pygame.Rect(0, 0, 560, 64 + len(HELP) * 22)
    box.center = rect.center
    shade = theme.translucent(surface.get_size())
    shade.fill((0, 0, 0, 150))
    surface.blit(shade, (0, 0))
    theme.panel(surface, box, color=theme.BACKGROUND, edge=theme.ATTACKER)
    theme.draw_text(surface, "KEYS", 14, theme.ATTACKER, (box.x + 24, box.y + 18), bold=True)
    for i, (key, what) in enumerate(HELP):
        y = box.y + 48 + i * 22
        theme.draw_text(surface, key, 13, theme.GOLD, (box.x + 24, y), bold=True)
        theme.draw_text(surface, what, 13, theme.TEXT, (box.x + 170, y))


def _chip(surface, text, pos, color):
    rect = pygame.Rect(pos[0], pos[1], theme.text_width(text, 11, True) + 16, 20)
    pygame.draw.rect(surface, theme.mix(theme.BACKGROUND, color, 0.18), rect, border_radius=10)
    pygame.draw.rect(surface, color, rect, width=1, border_radius=10)
    theme.draw_text(surface, text, 11, color, rect.center, "center", bold=True)
    return rect
