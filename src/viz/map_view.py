"""Draws the network map: orbits, subnet bands, links, hosts, and the action being played.

Drawing is a pure function of a Scene (which frames to show and how far into the step's
animation we are), so playing, pausing, scrubbing, and manual play all draw the same way.
"""

import math
from dataclasses import dataclass

import pygame

from src.viz import layout as lay
from src.viz import theme
from src.viz.session import INTERNET, HostStatus, address_label

IMPACT = 0.45  # share of a step's animation spent before the action lands on its target
MAX_SCALE = 85.0  # pixels per layout unit, so small networks don't get comically large
LABEL_MARGIN = 1.6  # layout units kept around the outer ring for labels


@dataclass
class Scene:
    """What to draw: a frame, the frame before it, and how far its action has played."""

    before: object  # Frame shown until the action lands
    frame: object  # Frame whose action is being played
    progress: float = 1.0  # 0 = action just started, 1 = finished
    clock: float = 0.0  # seconds since the app started, for idle animations
    hover: tuple = None  # host under the mouse
    selected: tuple = None  # host whose action menu is open


class MapView:  # pylint: disable=too-many-instance-attributes
    """Draws an episode's frames on the radial layout, and finds hosts under the mouse."""

    def __init__(self, layout, session):
        self.layout = layout
        self.sensitive = dict(session.network.sensitive_hosts)
        self.values = {a: session.host_truth(a)["value"] for a in session.addresses}
        self.rect = pygame.Rect(0, 0, 1, 1)
        self.scale = 1.0
        self.origin = (0.0, 0.0)
        self.node_radius = 12
        self._background = None
        self._links = {}

    # ----- geometry -----

    def fit(self, rect, padding=0):
        """Scale the map to fill rect minus padding, and pre-draw the parts that never change."""
        self.rect = pygame.Rect(rect)
        inner = self.rect.inflate(-2 * padding, -2 * padding)
        xs, ys = self._content_bounds()
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        self.scale = min(inner.width / width, inner.height / height, MAX_SCALE)
        self.origin = (
            inner.centerx - (min(xs) + width / 2) * self.scale,
            inner.centery - (min(ys) + height / 2) * self.scale,
        )
        self.node_radius = round(min(20, max(9, 0.3 * self.scale)))
        self._links = {
            (a, b): [self.to_screen(p) for p in self.layout.link_path(a, b)]
            for a, b in self.layout.links
        }
        self._background = self._draw_background()

    def to_screen(self, point):
        """Layout units to pixels."""
        return (self.origin[0] + point[0] * self.scale, self.origin[1] + point[1] * self.scale)

    def host_position(self, address):
        """Pixel position of a host."""
        return self.to_screen(self.layout.hosts[address])

    def host_at(self, pos):
        """The host under a pixel position, or None."""
        reach = (self.node_radius + 4) ** 2
        for address in self.layout.hosts:
            x, y = self.host_position(address)
            if (x - pos[0]) ** 2 + (y - pos[1]) ** 2 <= reach:
                return address
        return None

    def _content_bounds(self):
        """x and y ranges (layout units) of everything drawn, with room for labels."""
        points = [(-1.0, -1.0), (1.0, 1.4)]  # the attacker and its label
        for arc in self.layout.subnets.values():
            for angle in (*arc.host_angles, *arc.span):
                for radius in (arc.radius - 0.6, arc.radius + LABEL_MARGIN):
                    points.append((radius * math.cos(angle), radius * math.sin(angle)))
        for a, b in self.layout.links:
            points.extend(self.layout.link_path(a, b))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (min(xs) - 0.6, max(xs) + 0.6), (min(ys) - 0.3, max(ys) + 0.3)

    def _label_spot(self, radius, angle, extra=0):
        """Where to anchor a label just outside a ring, and which side of it to anchor."""
        out = radius * self.scale + max(lay.BAND_HALF_WIDTH * self.scale, self.node_radius)
        out += 4 + extra
        dx, dy = math.cos(angle), math.sin(angle)
        point = (self.origin[0] + dx * out, self.origin[1] + dy * out)
        if dy < -0.7:
            return point, "midbottom"
        if dy > 0.7:
            return point, "midtop"
        return point, "midleft" if dx > 0 else "midright"

    # ----- drawing -----

    def draw(self, surface, scene):
        """Draw the map for one scene."""
        surface.blit(self._background, self.rect)
        surface.set_clip(self.rect)
        landed = scene.progress >= IMPACT
        hosts = scene.frame.hosts if landed else scene.before.hosts
        self._draw_links(surface, hosts)
        self._draw_attacker(surface, scene.clock)
        if scene.frame.action is not None:
            self._draw_route(surface, scene)
        for address, status in hosts.items():
            self._draw_host(surface, address, status, scene)
        if scene.frame.action is not None and landed and scene.progress < 1.0:
            self._draw_impact(surface, scene, (scene.progress - IMPACT) / (1 - IMPACT))
        surface.set_clip(None)

    def _draw_background(self):
        """Orbits, subnet bands, and subnet labels, drawn once per window size."""
        background = pygame.Surface(self.rect.size)
        background.fill(theme.BACKGROUND)
        center = self._local(self.to_screen((0, 0)))
        theme.glow(background, (28, 36, 74), center, self.layout.outer_radius * self.scale * 1.1)
        for radius in self.layout.ring_radii[1:]:
            theme.ring(background, theme.ORBIT, center, radius * self.scale, width=1)

        bands = theme.translucent(self.rect.size)
        for subnet, arc in self.layout.subnets.items():
            color = theme.subnet_color(subnet)
            outline = [self._local(p) for p in self._band_outline(arc)]
            pygame.draw.polygon(bands, (*color, 40), outline)
            pygame.draw.aalines(bands, (*color, 110), True, outline)
        background.blit(bands, (0, 0))

        for subnet, arc in self.layout.subnets.items():
            point, anchor = self._label_spot(arc.radius, arc.center, extra=17)
            theme.draw_text(
                background,
                f"SUBNET {subnet}",
                12,
                theme.mix(theme.subnet_color(subnet), theme.TEXT, 0.25),
                self._local(point),
                anchor=anchor,
                bold=True,
            )
        return background

    def _band_outline(self, arc):
        """Screen points around a subnet's band: along the outer edge, back along the inner."""
        start, end = arc.span
        steps = max(4, int(abs(end - start) / 0.05))
        inner, outer = arc.radius - lay.BAND_HALF_WIDTH, arc.radius + lay.BAND_HALF_WIDTH
        angles = [start + (end - start) * i / steps for i in range(steps + 1)]
        outer_edge = [(outer * math.cos(a), outer * math.sin(a)) for a in angles]
        inner_edge = [(inner * math.cos(a), inner * math.sin(a)) for a in reversed(angles)]
        return [self.to_screen(p) for p in outer_edge + inner_edge]

    def _local(self, point):
        """A screen point relative to the map's top-left corner (for the cached background)."""
        return (point[0] - self.rect.x, point[1] - self.rect.y)

    def _draw_links(self, surface, hosts):
        footholds = {lay.INTERNET} | {a[0] for a, s in hosts.items() if s >= HostStatus.USER}
        for (a, b), points in self._links.items():
            level = (a in footholds) + (b in footholds)
            if level == 2:
                color = theme.mix(theme.FAINT, theme.STATUS_COLORS[HostStatus.USER], 0.45)
                width = 2
            elif level == 1:
                color = theme.mix(theme.FAINT, theme.STATUS_COLORS[HostStatus.DISCOVERED], 0.35)
                width = 2
            else:
                color, width = theme.FAINT, 1
            theme.polyline(surface, color, points, width)

    def _draw_attacker(self, surface, clock):
        center = self.to_screen((0, 0))
        radius = lay.ATTACKER_RADIUS * self.scale * 0.8
        pulse = 0.55 + 0.25 * math.sin(clock * 2.0)
        theme.glow(surface, theme.ATTACKER, center, radius * 3.2, pulse)
        corners = [
            (center[0] + radius * math.cos(a), center[1] + radius * math.sin(a))
            for a in (math.pi / 6 + i * math.pi / 3 for i in range(6))
        ]
        pygame.draw.polygon(surface, theme.mix(theme.BACKGROUND, theme.ATTACKER, 0.35), corners)
        pygame.draw.aalines(surface, theme.ATTACKER, True, corners)
        pygame.draw.polygon(surface, theme.ATTACKER, corners, 2)
        theme.draw_text(surface, ">_", 14, theme.TEXT, center, anchor="center", bold=True)
        theme.draw_text(
            surface,
            "ATTACKER",
            11,
            theme.ATTACKER,
            (center[0], center[1] + radius + 6),
            anchor="midtop",
            bold=True,
        )

    def _draw_host(self, surface, address, status, scene):
        x, y = self.host_position(address)
        r = self.node_radius
        color = theme.STATUS_COLORS[status]
        value = self.values.get(address, 0.0)

        if status >= HostStatus.USER:
            theme.glow(surface, color, (x, y), r * 2.6, 0.55)
        if address in self.sensitive:
            gold = (
                theme.GOLD if status == HostStatus.ROOT else theme.mix(theme.GOLD, theme.FAINT, 0.3)
            )
            theme.ring(surface, gold, (x, y), r + 5, width=2)
            theme.glow(
                surface, theme.GOLD, (x, y), r * 2.2, 0.25 + 0.15 * math.sin(scene.clock * 3)
            )
        elif value < 0:
            theme.dashed_ring(surface, theme.HONEYPOT, (x, y), r + 5, dashes=8, width=2)

        if status == HostStatus.UNKNOWN:
            theme.filled_circle(surface, theme.BACKGROUND, (x, y), r)
            theme.dashed_ring(surface, color, (x, y), r, dashes=10, width=2)
        elif status == HostStatus.DISCOVERED:
            theme.filled_circle(surface, theme.mix(theme.BACKGROUND, color, 0.18), (x, y), r)
            theme.ring(surface, color, (x, y), r, width=2)
        else:
            theme.filled_circle(surface, color, (x, y), r)
        glyph = theme.STATUS_GLYPHS[status]
        if glyph:
            glyph_color = theme.MUTED if status == HostStatus.UNKNOWN else theme.BACKGROUND
            theme.draw_text(
                surface, glyph, max(11, r), glyph_color, (x, y + 1), "center", bold=True
            )

        if address in self.sensitive:
            filled = status == HostStatus.ROOT
            theme.diamond(surface, theme.GOLD if filled else theme.FAINT, (x + r, y - r), 5)
        if address in (scene.hover, scene.selected):
            pulse = 2 * math.sin(scene.clock * 6) if address == scene.selected else 0
            theme.ring(surface, theme.TEXT, (x, y), r + 9 + pulse, width=1)
        self._draw_host_label(surface, address, status, value)

    def _draw_host_label(self, surface, address, status, value):
        color = theme.MUTED if status == HostStatus.UNKNOWN else theme.TEXT
        if address in self.sensitive:
            color = theme.GOLD
        elif value < 0:
            color = theme.HONEYPOT
        point, anchor = self._label_spot(*self.layout.polar(address))
        theme.draw_text(surface, address_label(address), 11, color, point, anchor=anchor)

    def _route_points(self, frame):
        source = lay.INTERNET if frame.source == INTERNET else frame.source
        return [self.to_screen(p) for p in self.layout.host_path(source, frame.action.target)]

    def _draw_route(self, surface, scene):
        frame, action = scene.frame, scene.frame.action
        color = theme.KIND_COLORS[action.kind]
        target = self.host_position(action.target)
        if not action.remote:
            if scene.progress < IMPACT:  # on-host action: a ring closing in on the host
                shrink = 1 - scene.progress / IMPACT
                theme.ring(surface, color, target, self.node_radius * (1 + 2.5 * shrink), 2)
            elif scene.progress >= 1.0:
                theme.ring(surface, color, target, self.node_radius + 4, 1)
            return
        points = self._route_points(frame)
        if scene.progress < IMPACT:
            head = scene.progress / IMPACT * (len(points) - 1)
            done = points[: int(head) + 1]
            theme.polyline(surface, theme.mix(theme.BACKGROUND, color, 0.5), done, 2)
            for i in range(max(0, int(head) - 8), int(head) + 1):
                fade = 1 - (head - i) / 9
                theme.glow(surface, color, points[i], 4 + 8 * fade, 0.5 * fade)
            theme.filled_circle(
                surface, theme.mix(color, (255, 255, 255), 0.5), points[int(head)], 4
            )
        else:
            dim = 0.35 if scene.progress >= 1.0 else 0.7
            theme.polyline(surface, theme.mix(theme.BACKGROUND, color, dim), points, 2)
            if scene.progress >= 1.0:
                theme.ring(surface, color, target, self.node_radius + 4, 1)

    def _draw_impact(self, surface, scene, t):
        """Effects right after an action lands; t goes from 0 to 1 as they fade out."""
        frame, action = scene.frame, scene.frame.action
        x, y = self.host_position(action.target)
        r = self.node_radius
        fade = 1 - t
        if frame.success:
            color = (
                theme.STATUS_COLORS[frame.gained]
                if frame.gained
                else theme.KIND_COLORS[action.kind]
            )
            theme.glow(
                surface, color, (x, y), r * (2.5 + 2.5 * t) if frame.gained else r * 2.5, fade
            )
            if action.kind == "subnet_scan":  # radar wave out from the host
                wave = r + t * self.scale * 4.0
                overlay = theme.translucent(self.rect.size)
                center = (x - self.rect.x, y - self.rect.y)
                pygame.draw.circle(overlay, (*color, round(160 * fade)), center, wave, 2)
                surface.blit(overlay, self.rect)
                for address in frame.revealed:
                    theme.glow(surface, color, self.host_position(address), r * 3, fade)
            elif action.kind.endswith("scan"):
                theme.ring(surface, color, (x, y), r + 4 + 14 * t, 1)
        else:
            theme.cross(surface, theme.FAILURE, (x, y), r * 0.7 * (1.3 - 0.3 * t), width=3)
            theme.draw_text(
                surface,
                frame.failure,
                11,
                theme.FAILURE,
                (x, y + r + 16),
                anchor="midtop",
                alpha=round(255 * min(1.0, fade * 1.6)),
            )
        reward_color = theme.GOLD if frame.reward > 0 else theme.MUTED
        theme.draw_text(
            surface,
            f"{frame.reward:+g}",
            20 if frame.reward > 0 else 13,
            reward_color,
            (x, y - r - 8 - 34 * t),
            anchor="midbottom",
            bold=frame.reward > 0,
            alpha=round(255 * min(1.0, fade * 1.5)),
        )
