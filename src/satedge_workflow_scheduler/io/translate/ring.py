"""Expansion of a ring constellation into the explicit planes and links it stands for."""

from __future__ import annotations

from satedge_workflow_scheduler.io.documents import (
    LinkDoc,
    LinkParamsDoc,
    PlaneDoc,
    RingTopologyDoc,
    SatelliteDoc,
    WindowDoc,
)
from satedge_workflow_scheduler.io.errors import DocumentError


def expand_ring(
    ring: RingTopologyDoc, ground: str, horizon: float, hpc_rate: float
) -> tuple[list[PlaneDoc], list[LinkDoc]]:
    """The planes and links of ``ring``, with the computing windows up to ``horizon``.

    Within a plane of three or more satellites the links form a cycle; a plane of two has one
    link; a plane of one has none. The COMs are linked pairwise and to ``ground``; those links
    are windowed, to be given by the contact plan.

    Raises:
        DocumentError: If ``com_index`` is out of range, a name in the overrides or the phases
            is not a satellite, or the HPC window is not positive or is longer than its period.
    """
    if ring.com_index >= ring.satellites_per_plane:
        raise DocumentError(
            f"com_index {ring.com_index} is out of range for "
            f"{ring.satellites_per_plane} satellites per plane"
        )
    names = {
        (p, k): _name(p, k, ring.com_index)
        for p in range(1, ring.planes + 1)
        for k in range(ring.satellites_per_plane)
    }
    order = [names[key] for key in sorted(names)]
    for node in (*ring.plmm_overrides_gb, *ring.hpc_phases_s):
        if node not in order:
            raise DocumentError(f"{node} is not a satellite of the ring")
    window = ring.hpc.duration_s - ring.hpc.startup_s
    if window <= 0 or window > ring.hpc.period_s:
        raise DocumentError(
            "hpc: the window (duration_s minus startup_s) must be positive and not longer "
            "than period_s"
        )

    total = len(order)
    planes: list[PlaneDoc] = []
    for p in range(1, ring.planes + 1):
        satellites = []
        for k in range(ring.satellites_per_plane):
            node = names[(p, k)]
            index = order.index(node)
            phase = ring.hpc_phases_s.get(node, index * ring.hpc.period_s / total)
            satellites.append(
                SatelliteDoc(
                    id=node,
                    plmm_capacity_gb=ring.plmm_overrides_gb.get(
                        node, ring.plmm_capacity_gb
                    ),
                    hpc_rate_gbps=hpc_rate,
                    hpc_windows=_hpc_windows(phase, ring, horizon),
                )
            )
        planes.append(
            PlaneDoc(id=p, com=names[(p, ring.com_index)], satellites=satellites)
        )

    links: list[LinkDoc] = []
    n = ring.satellites_per_plane
    for p in range(1, ring.planes + 1):
        for i, j in _ring_pairs(n):
            links.append(_link(names[(p, i)], names[(p, j)], ring.intra_link))
    coms = [names[(p, ring.com_index)] for p in range(1, ring.planes + 1)]
    for i, a in enumerate(coms):
        for b in coms[i + 1 :]:
            links.append(_link(a, b, ring.inter_link, windowed=True))
    for com in coms:
        links.append(_link(com, ground, ring.ground_link, windowed=True))
    return planes, links


def _name(plane: int, index: int, com_index: int) -> str:
    return f"COM{plane}" if index == com_index else f"P{plane}-S{index}"


def _ring_pairs(n: int) -> list[tuple[int, int]]:
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    return [(k, (k + 1) % n) for k in range(n)]


def _link(a: str, b: str, params: LinkParamsDoc, *, windowed: bool = False) -> LinkDoc:
    return LinkDoc(
        a=a,
        b=b,
        capacity_gbps=params.capacity_gbps,
        latency_s=params.latency_s,
        windowed=windowed,
    )


def _hpc_windows(
    phase: float, ring: RingTopologyDoc, horizon: float
) -> list[WindowDoc]:
    """The windows that start before ``horizon``, one per period from ``phase``."""
    length = ring.hpc.duration_s - ring.hpc.startup_s
    windows = []
    start = phase
    while start < horizon:
        windows.append(WindowDoc(start_s=start, end_s=start + length))
        start += ring.hpc.period_s
    return windows
