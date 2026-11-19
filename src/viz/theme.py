"""Colors, fonts, and small drawing helpers shared by the GUI's views."""

import math
from functools import lru_cache
from pathlib import Path

import pygame
from pygame import gfxdraw

from src.viz.session import HostStatus

BACKGROUND = (11, 16, 32)
PANEL = (17, 24, 51)
PANEL_EDGE = (31, 41, 71)
TEXT = (201, 211, 238)
MUTED = (108, 119, 153)
FAINT = (45, 55, 88)
ORBIT = (36, 47, 82)
ATTACKER = (167, 139, 250)
GOLD = (255, 209, 102)
SUCCESS = (110, 231, 160)
FAILURE = (255, 92, 122)
HONEYPOT = (217, 70, 239)

STATUS_COLORS = {
    HostStatus.UNKNOWN: (64, 72, 99),
    HostStatus.DISCOVERED: (76, 201, 240),
    HostStatus.USER: (255, 181, 71),
    HostStatus.ROOT: (255, 92, 122),
}
# Shell prompts: $ for a user shell, # for root
STATUS_GLYPHS = {
    HostStatus.UNKNOWN: "?",
    HostStatus.DISCOVERED: "",
    HostStatus.USER: "$",
    HostStatus.ROOT: "#",
}
STATUS_NAMES = {
    HostStatus.UNKNOWN: "unknown",
    HostStatus.DISCOVERED: "discovered",
    HostStatus.USER: "user access",
    HostStatus.ROOT: "root access",
}
KIND_COLORS = {
    "service_scan": (76, 201, 240),
    "os_scan": (76, 201, 240),
    "process_scan": (76, 201, 240),
    "subnet_scan": (94, 234, 212),
    "exploit": (255, 181, 71),
    "privesc": (255, 92, 122),
}
KIND_NAMES = {
    "service_scan": "service scan",
    "os_scan": "OS scan",
    "process_scan": "process scan",
    "subnet_scan": "subnet scan",
    "exploit": "exploit",
    "privesc": "privilege escalation",
}
# Muted hues for the subnet bands, picked so the host status colors stay readable on top
SUBNET_COLORS = [
    (99, 132, 255),
    (167, 139, 250),
    (45, 212, 191),
    (244, 114, 182),
    (163, 230, 53),
    (251, 146, 60),
    (56, 189, 248),
    (232, 121, 249),
]

# Monospace fonts tried in order. Loading a file directly avoids pygame.font.SysFont, which can
# stall for seconds while it searches the system's fonts.
FONT_FILES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "C:/Windows/Fonts/consola.ttf",
]


def subnet_color(subnet):
    """Band color of a subnet."""
    return SUBNET_COLORS[(subnet - 1) % len(SUBNET_COLORS)]


@lru_cache(maxsize=None)
def _font_file():
    """The first monospace font file that exists, or None for pygame's built-in font."""
    return next((path for path in FONT_FILES if Path(path).is_file()), None)


@lru_cache(maxsize=None)
def font(size, bold=False):
    """A monospace font, cached by size."""
    loaded = pygame.font.Font(_font_file(), size)
    loaded.set_bold(bold)
    return loaded


def reset_caches():
    """Forget cached fonts and sprites. Needed after pygame.quit(), which invalidates fonts."""
    font.cache_clear()
    _glow_sprite.cache_clear()


def draw_text(surface, text, size, color, pos, anchor="topleft", bold=False, alpha=255):
    """Draw text with one corner or edge at pos (any pygame.Rect anchor name). Returns its rect."""
    image = font(size, bold).render(str(text), True, color)
    if alpha < 255:
        image.set_alpha(alpha)
    rect = image.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
    surface.blit(image, rect)
    return rect


def text_width(text, size, bold=False):
    """Width in pixels of text in the GUI font."""
    return font(size, bold).size(str(text))[0]


def mix(color, other, amount):
    """Blend two RGB colors: amount=0 gives color, amount=1 gives other."""
    return tuple(round(a + (b - a) * amount) for a, b in zip(color, other))


def filled_circle(surface, color, center, radius):
    """Anti-aliased filled circle."""
    x, y, r = round(center[0]), round(center[1]), max(1, round(radius))
    gfxdraw.filled_circle(surface, x, y, r, color)
    gfxdraw.aacircle(surface, x, y, r, color)


def ring(surface, color, center, radius, width=2):
    """Anti-aliased circle outline."""
    x, y, r = round(center[0]), round(center[1]), max(1, round(radius))
    if width > 1:
        pygame.draw.circle(surface, color, (x, y), r, width)
    gfxdraw.aacircle(surface, x, y, r, color)


def dashed_ring(surface, color, center, radius, dashes=10, width=2):
    """Circle outline made of dashes."""
    rect = pygame.Rect(0, 0, 2 * radius, 2 * radius)
    rect.center = (round(center[0]), round(center[1]))
    step = 2 * math.pi / dashes
    for i in range(dashes):
        pygame.draw.arc(surface, color, rect, i * step, i * step + step * 0.55, width)


def polyline(surface, color, points, width=1):
    """Line through several points, smoothed with an anti-aliased pass."""
    if len(points) < 2:
        return
    if width > 1:
        pygame.draw.lines(surface, color, False, points, width)
    pygame.draw.aalines(surface, color, False, points)


def translucent(size):
    """A transparent surface to draw see-through shapes on before blitting."""
    return pygame.Surface(size, pygame.SRCALPHA)


@lru_cache(maxsize=256)
def _glow_sprite(color, radius):
    size = radius * 2
    sprite = pygame.Surface((size, size))
    sprite.fill((0, 0, 0))
    for r in range(radius, 0, -1):
        strength = (1 - r / radius) ** 2
        pygame.draw.circle(sprite, tuple(round(c * strength) for c in color), (radius, radius), r)
    return sprite


def glow(surface, color, center, radius, strength=1.0):
    """Soft additive light around a point, like a neon glow."""
    radius = max(2, round(radius))
    if strength <= 0:
        return
    sprite = _glow_sprite(tuple(round(c * min(strength, 1.0)) for c in color), radius)
    surface.blit(
        sprite,
        (round(center[0]) - radius, round(center[1]) - radius),
        special_flags=pygame.BLEND_RGB_ADD,
    )


def diamond(surface, color, center, size):
    """Small filled diamond, used to mark sensitive hosts."""
    x, y = center
    points = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
    gfxdraw.filled_polygon(surface, points, color)
    gfxdraw.aapolygon(surface, points, color)


def cross(surface, color, center, size, width=3):
    """An X mark, used for failed actions."""
    x, y = center
    pygame.draw.line(surface, color, (x - size, y - size), (x + size, y + size), width)
    pygame.draw.line(surface, color, (x - size, y + size), (x + size, y - size), width)


def panel(surface, rect, color=PANEL, edge=PANEL_EDGE, radius=8):
    """Rounded panel background with a thin border."""
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    pygame.draw.rect(surface, edge, rect, width=1, border_radius=radius)
