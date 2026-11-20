"""Radial "orbit" layout of a scenario's network, for the GUI.

The attacker (the internet) sits at the center. Each subnet goes on the ring whose number is
its distance in hops from the internet, and its hosts sit side by side on that ring, so a
subnet is an arc of hosts. Around each ring, subnets are placed so that linked subnets end up
close together.

Coordinates are in layout units, where neighboring hosts of a subnet are 1 unit apart; the GUI
scales units to pixels. Angles follow screen coordinates (y points down), so -pi/2 is straight up.

Usage:
    layout = radial_layout(env.unwrapped.network)
    x, y = layout.hosts[(2, 0)]
    points = layout.link_path(1, 2)
"""

import itertools
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

INTERNET = 0
TOP = -math.pi / 2

HOST_SPACING = 1.0  # arc length between neighboring hosts of a subnet
SUBNET_GAP = 1.0  # extra arc length kept free between neighboring subnets on a ring
FIRST_RING = 2.4  # smallest radius of ring 1
RING_GAP = 2.2  # smallest distance between consecutive rings
MAX_RING_FILL = 0.9  # share of a ring's circumference that subnets may use
BAND_HALF_WIDTH = 0.45  # half the thickness of a subnet's band
ATTACKER_RADIUS = 0.7  # radius of the attacker node at the center
SAME_RING_WEIGHT = 0.5  # same-ring links are drawn along the ring, so their length matters less
N_CANDIDATES = 72  # angles tried for each subnet on rings 2 and up (every 5 degrees)
MAX_JOINT = 3  # place up to this many subnets of a ring together; more are placed one by one
MAX_ORDERS = 720  # most orders of ring 1 that are tried


@dataclass(frozen=True)
class SubnetArc:
    """Where one subnet sits: its ring, the angle of its middle, and its hosts' angles."""

    subnet: int
    ring: int
    radius: float
    center: float
    host_angles: tuple  # one angle per host, in host-number order

    @property
    def span(self):
        """(start, end) angles of the subnet's band, padded by half a host on each side."""
        pad = 0.5 * HOST_SPACING / self.radius
        return self.host_angles[0] - pad, self.host_angles[-1] + pad


@dataclass(frozen=True)
class NetworkLayout:
    """Positions of everything drawn on the network map, in layout units."""

    ring_radii: tuple  # radius of each ring; ring 0 is the attacker at the center
    subnets: dict  # subnet number -> SubnetArc (the internet is not included)
    hosts: dict  # (subnet, host) -> (x, y)
    links: tuple  # (a, b) subnet pairs from the topology with a < b; subnet 0 is the internet

    @property
    def outer_radius(self):
        """Radius of the outermost ring."""
        return self.ring_radii[-1]

    def link_path(self, a, b, steps=32):
        """Points along the drawn link between subnets a and b.

        Links leave a band from its middle. Links between rings follow the rings (radius and
        angle change together), and links within a ring bow inward so they don't cross hosts.
        """
        if INTERNET in (a, b):
            arc = self.subnets[b if a == INTERNET else a]
            return _polar_path(
                ATTACKER_RADIUS, arc.center, arc.radius - BAND_HALF_WIDTH, arc.center
            )
        arc_a, arc_b = self.subnets[a], self.subnets[b]
        if arc_a.ring == arc_b.ring:
            radius = arc_a.radius - BAND_HALF_WIDTH
            return _bowed_path(radius, arc_a.center, arc_b.center, steps)
        inner, outer = sorted((arc_a, arc_b), key=lambda arc: arc.radius)
        return _polar_path(
            inner.radius + BAND_HALF_WIDTH,
            inner.center,
            outer.radius - BAND_HALF_WIDTH,
            outer.center,
            steps,
        )

    def host_path(self, source, target, steps=32):
        """Points from a source (a host address, or INTERNET) to a target host's center.

        Used to animate an attack traveling from the host it was launched from.
        """
        t_radius, t_angle = self.polar(target)
        if source == INTERNET:
            return _polar_path(ATTACKER_RADIUS, t_angle, t_radius, t_angle)
        s_radius, s_angle = self.polar(source)
        if math.isclose(s_radius, t_radius):
            return _bowed_path(s_radius, s_angle, t_angle, steps)
        return _polar_path(s_radius, s_angle, t_radius, t_angle, steps)

    def polar(self, address):
        """(radius, angle) of a host."""
        arc = self.subnets[address[0]]
        return arc.radius, arc.host_angles[address[1]]


def radial_layout(network):
    """Lay out a NASim/PenGym network (env.unwrapped.network)."""
    return layout_from_topology(network.topology, network.address_space)


def layout_from_topology(topology, addresses):
    """Lay out a network from its subnet adjacency matrix and its host addresses."""
    rings = subnet_rings(topology)
    host_ids = {}
    for subnet, host in sorted(addresses):
        host_ids.setdefault(subnet, []).append(host)

    members = {}
    for subnet in sorted(host_ids):
        members.setdefault(rings[subnet], []).append(subnet)
    radii = _ring_radii(members, host_ids)

    size = len(topology)
    links = tuple(
        (a, b)
        for a in range(size)
        for b in range(a + 1, size)
        if (topology[a][b] or topology[b][a]) and {a, b} <= set(host_ids) | {INTERNET}
    )
    footprint = {
        s: (len(ids) * HOST_SPACING + SUBNET_GAP) / radii[rings[s]] for s, ids in host_ids.items()
    }
    centers = _best_centers(members, footprint, links, rings)

    arcs, hosts = {}, {}
    for subnet, ids in host_ids.items():
        radius = radii[rings[subnet]]
        step = HOST_SPACING / radius
        offsets = [i - (len(ids) - 1) / 2 for i in range(len(ids))]
        angles = tuple(centers[subnet] + offset * step for offset in offsets)
        arcs[subnet] = SubnetArc(subnet, rings[subnet], radius, centers[subnet], angles)
        for host, angle in zip(ids, angles):
            hosts[(subnet, host)] = (radius * math.cos(angle), radius * math.sin(angle))
    return NetworkLayout(tuple(radii), arcs, hosts, links)


def subnet_rings(topology):
    """Hops from the internet to each subnet (breadth-first search over the topology).

    Subnets with no path from the internet go one ring past the deepest reachable one.
    """
    size = len(topology)
    rings = {INTERNET: 0}
    queue = deque([INTERNET])
    while queue:
        subnet = queue.popleft()
        for other in range(size):
            if other not in rings and (topology[subnet][other] or topology[other][subnet]):
                rings[other] = rings[subnet] + 1
                queue.append(other)
    deepest = max(rings.values())
    for subnet in range(size):
        rings.setdefault(subnet, deepest + 1)
    return rings


def _ring_radii(members, host_ids):
    """Radius of every ring: evenly spaced, but large enough to fit the ring's hosts."""
    radii = [0.0]
    for ring in range(1, max(members, default=0) + 1):
        arc_needed = sum(
            len(host_ids[s]) * HOST_SPACING + SUBNET_GAP for s in members.get(ring, [])
        )
        smallest = arc_needed / (2 * math.pi * MAX_RING_FILL)
        base = FIRST_RING if ring == 1 else radii[-1] + RING_GAP
        radii.append(max(base, smallest))
    return radii


def _best_centers(members, footprint, links, rings):
    """Center angle of every subnet, for the order of ring 1 that keeps links shortest.

    Ring 1 is spread evenly around the attacker, so the map stays radial. Each order of ring 1
    is tried, outer rings are placed to suit it, and the order with the shortest links wins.
    """
    first = members.get(1, [])
    best_cost, best = math.inf, {}
    for order in _ring_one_orders(first):
        centers = _spread_evenly(order, footprint)
        for ring in sorted(r for r in members if r > 1):
            centers.update(_place_ring(members[ring], centers, footprint, links, rings))
        cost = _link_cost(centers, links, rings)
        if cost < best_cost - 1e-9:
            best_cost, best = cost, centers
    return best


def _ring_one_orders(subnets):
    """Orders of ring 1 worth trying. Rotations look the same, so the first subnet stays put."""
    if len(subnets) <= 2:
        return [list(subnets)]
    rest = itertools.permutations(subnets[1:])
    return [[subnets[0], *order] for order in itertools.islice(rest, MAX_ORDERS)]


def _spread_evenly(order, footprint):
    """Centers that spread subnets evenly around a ring, the first one at the top."""
    if not order:
        return {}
    free = max(0.0, 2 * math.pi - sum(footprint[s] for s in order))
    gap = free / len(order)
    angle = TOP - footprint[order[0]] / 2
    centers = {}
    for subnet in order:
        centers[subnet] = angle + footprint[subnet] / 2
        angle += footprint[subnet] + gap
    return centers


def _place_ring(subnets, placed, footprint, links, rings):
    """Centers for one ring's subnets, close to the already placed subnets they link to."""
    candidates = TOP + np.arange(N_CANDIDATES) * (2 * math.pi / N_CANDIDATES)
    pull = {s: np.zeros(N_CANDIDATES) for s in subnets}
    for a, b in links:
        for subnet, other in ((a, b), (b, a)):
            if subnet in pull and other in placed and other != INTERNET:
                pull[subnet] += _weight(subnet, other, rings) * _angle_gap(
                    candidates, placed[other]
                )

    if len(subnets) <= MAX_JOINT:
        return _place_jointly(subnets, candidates, pull, footprint, links)
    return _place_one_by_one(subnets, candidates, pull, footprint, links)


def _place_jointly(subnets, candidates, pull, footprint, links):
    """Try every combination of candidate angles for a few subnets and keep the best."""
    n = len(subnets)
    grids = np.meshgrid(*([candidates] * n), indexing="ij")
    total = np.zeros(grids[0].shape)
    for i, subnet in enumerate(subnets):
        shape = [1] * n
        shape[i] = len(candidates)
        total = total + pull[subnet].reshape(shape)
    for i, j in itertools.combinations(range(n), 2):
        gap = _angle_gap(grids[i], grids[j])
        if _linked(subnets[i], subnets[j], links):
            total = total + SAME_RING_WEIGHT * gap
        overlap = gap < (footprint[subnets[i]] + footprint[subnets[j]]) / 2
        total = np.where(overlap, np.inf, total)
    best = np.unravel_index(np.argmin(total), total.shape)
    if not np.isfinite(total[best]):
        return _spread_evenly(subnets, footprint)
    return {subnet: float(candidates[best[i]]) for i, subnet in enumerate(subnets)}


def _place_one_by_one(subnets, candidates, pull, footprint, links):
    """Place many subnets of one ring greedily, most strongly linked first."""
    centers = {}
    for subnet in sorted(subnets, key=lambda s: -pull[s].max()):
        cost = pull[subnet].copy()
        for other, angle in centers.items():
            gap = _angle_gap(candidates, angle)
            if _linked(subnet, other, links):
                cost += SAME_RING_WEIGHT * gap
            cost[gap < (footprint[subnet] + footprint[other]) / 2] = np.inf
        if not np.isfinite(cost).any():
            return _spread_evenly(subnets, footprint)
        centers[subnet] = float(candidates[np.argmin(cost)])
    return centers


def _link_cost(centers, links, rings):
    """Total angular length of all links between subnets (links to the internet excluded)."""
    return sum(
        _weight(a, b, rings) * float(_angle_gap(centers[a], centers[b]))
        for a, b in links
        if INTERNET not in (a, b)
    )


def _weight(a, b, rings):
    return SAME_RING_WEIGHT if rings[a] == rings[b] else 1.0


def _linked(a, b, links):
    return (min(a, b), max(a, b)) in links


def _angle_gap(a, b):
    """Smallest absolute difference between two angles (works on NumPy arrays too)."""
    return np.abs((np.asarray(a) - b + np.pi) % (2 * np.pi) - np.pi)


def _turn(start, end):
    """Signed smallest rotation from one angle to another."""
    return (end - start + math.pi) % (2 * math.pi) - math.pi


def _polar_path(r0, a0, r1, a1, steps=2):
    """Points that move from (r0, a0) to (r1, a1) changing radius and angle together."""
    turn = _turn(a0, a1)
    steps = steps if abs(turn) > 1e-6 else 2
    points = []
    for t in np.linspace(0.0, 1.0, steps):
        radius, angle = r0 + (r1 - r0) * t, a0 + turn * t
        points.append((radius * math.cos(angle), radius * math.sin(angle)))
    return points


def _bowed_path(radius, a0, a1, steps):
    """Points along a ring from angle a0 to a1, bowing toward the center in the middle."""
    turn = _turn(a0, a1)
    bow = min(0.9, 0.3 * abs(turn) * radius)
    points = []
    for t in np.linspace(0.0, 1.0, steps):
        r, angle = radius - bow * math.sin(math.pi * t), a0 + turn * t
        points.append((r * math.cos(angle), r * math.sin(angle)))
    return points
