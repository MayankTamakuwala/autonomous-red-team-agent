"""Tests for the radial network layout (src/viz/layout.py)."""

import math
from itertools import combinations

import pytest

from src.env.env_setup import SCENARIO_DIR, make_env
from src.viz.layout import HOST_SPACING, SUBNET_GAP, radial_layout, subnet_rings

SCENARIOS = [
    "tiny",
    "tiny-small",
    "small-honeypot",
    "small-linear",
    "medium",
    "medium-single-site",
    "medium-multi-site",
]


def network(name):
    """The PenGym network of a scenario file."""
    return make_env(SCENARIO_DIR / f"{name}.yml").unwrapped.network


def test_rings_are_hops_from_the_internet():
    """Subnet 1 touches the internet in tiny; subnets 2 and 3 are one hop further."""
    assert subnet_rings(network("tiny").topology) == {0: 0, 1: 1, 2: 2, 3: 2}


def test_medium_multi_site_rings():
    """The four internet-facing subnets form ring 1; the two inner subnets form ring 2."""
    layout = radial_layout(network("medium-multi-site"))
    rings = {s: arc.ring for s, arc in layout.subnets.items()}
    assert rings == {1: 1, 4: 1, 5: 1, 6: 1, 2: 2, 3: 2}


@pytest.mark.parametrize("name", SCENARIOS)
def test_every_host_sits_on_its_subnets_ring(name):
    """Each host is placed, at the radius of its subnet's ring, and rings grow outward."""
    net = network(name)
    layout = radial_layout(net)
    assert set(layout.hosts) == set(net.address_space)
    for (subnet, _), (x, y) in layout.hosts.items():
        assert math.hypot(x, y) == pytest.approx(layout.subnets[subnet].radius)
    assert list(layout.ring_radii) == sorted(layout.ring_radii)


@pytest.mark.parametrize("name", SCENARIOS)
def test_hosts_do_not_overlap(name):
    """Any two hosts are at least most of a host spacing apart."""
    points = list(radial_layout(network(name)).hosts.values())
    for (x1, y1), (x2, y2) in combinations(points, 2):
        assert math.hypot(x1 - x2, y1 - y2) > 0.9 * HOST_SPACING


@pytest.mark.parametrize("name", SCENARIOS)
def test_subnets_on_a_ring_keep_apart(name):
    """Bands on the same ring never overlap: the gap between them is at least SUBNET_GAP."""
    arcs = list(radial_layout(network(name)).subnets.values())
    for a, b in combinations(arcs, 2):
        if a.ring != b.ring:
            continue
        between = abs((a.center - b.center + math.pi) % (2 * math.pi) - math.pi)
        half_widths = (
            a.host_angles[-1] - a.host_angles[0] + b.host_angles[-1] - b.host_angles[0]
        ) / 2
        assert between * a.radius >= half_widths * a.radius + 0.99 * (HOST_SPACING + SUBNET_GAP)


def test_links_follow_the_topology():
    """Every connected pair of subnets gets a link, and link paths join the right places."""
    net = network("tiny-small")
    layout = radial_layout(net)
    size = len(net.topology)
    expected = {(a, b) for a in range(size) for b in range(a + 1, size) if net.topology[a][b]}
    assert set(layout.links) == expected
    for a, b in layout.links:
        points = layout.link_path(a, b)
        assert len(points) >= 2
        for end, subnet in ((points[0], a), (points[-1], b)):
            if subnet == 0:
                continue
            arc = layout.subnets[subnet]
            assert math.hypot(*end) == pytest.approx(arc.radius, abs=0.5)
