"""Documents of the topology: planes, links, and the generated ring constellation."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from satedge_workflow_scheduler.io.documents.base import Document


class WindowDoc(Document):
    """A window of time, in seconds."""

    start_s: float
    end_s: float


class SatelliteDoc(Document):
    """A satellite: its PLMM, and its computing windows."""

    id: str
    plmm_capacity_gb: float
    hpc_rate_gbps: float | None = None
    hpc_windows: list[WindowDoc] = Field(default_factory=list)


class PlaneDoc(Document):
    """An orbital plane, with the satellite that is its communication node."""

    id: int
    com: str
    satellites: list[SatelliteDoc] = Field(min_length=1)


class LinkDoc(Document):
    """A link between two nodes.

    ``windows`` makes it available only in those windows. A ``windowed`` link without windows
    takes them from the contact plan.
    """

    a: str
    b: str
    capacity_gbps: float = 1.0
    latency_s: float = 0.015
    windowed: bool = False
    windows: list[WindowDoc] | None = None


class ExplicitTopologyDoc(Document):
    """The explicit topology: every plane, satellite and link is listed."""

    kind: Literal["explicit"]
    planes: list[PlaneDoc] = Field(min_length=1)
    links: list[LinkDoc] = Field(default_factory=list)


class PeriodicHpcDoc(Document):
    """Computing windows that repeat with a period, in seconds.

    Each window lasts ``duration_s``, of which ``startup_s`` goes to starting the containers.
    """

    period_s: float = 5640.0
    duration_s: float = 300.0
    startup_s: float = 0.0


class LinkParamsDoc(Document):
    """Capacity and latency of the links of a generated topology."""

    capacity_gbps: float = 1.0
    latency_s: float = 0.015


class RingTopologyDoc(Document):
    """A constellation of ``planes`` rings of ``satellites_per_plane`` satellites.

    In each ring the satellite at ``com_index`` is the communication node (COM). The COMs are
    linked to each other and to the ground station through windowed links, whose windows come
    from the contact plan. Node names: ``COM<p>`` for the COM of plane ``p`` (from 1), and
    ``P<p>-S<k>`` for the other satellites (``k`` from 0).

    Attributes:
        planes: Number of planes.
        satellites_per_plane: Satellites in each plane.
        com_index: Position of the COM in each plane.
        plmm_capacity_gb: PLMM capacity of every satellite.
        plmm_overrides_gb: PLMM capacity of some satellites, by node name.
        hpc: Computing windows of every satellite.
        hpc_phases_s: Start of the first window, by node name; satellites not listed start
            equally spaced over the period, in plane order.
        intra_link: Links between neighbours in a ring.
        inter_link: Links between COMs.
        ground_link: Links between COMs and the ground station.
    """

    kind: Literal["ring_constellation"]
    planes: int = Field(default=2, ge=1)
    satellites_per_plane: int = Field(default=9, ge=1)
    com_index: int = Field(default=0, ge=0)
    plmm_capacity_gb: float = 2000.0
    plmm_overrides_gb: dict[str, float] = Field(default_factory=dict)
    hpc: PeriodicHpcDoc = Field(default_factory=PeriodicHpcDoc)
    hpc_phases_s: dict[str, float] = Field(default_factory=dict)
    intra_link: LinkParamsDoc = Field(default_factory=LinkParamsDoc)
    inter_link: LinkParamsDoc = Field(default_factory=LinkParamsDoc)
    ground_link: LinkParamsDoc = Field(default_factory=LinkParamsDoc)


TopologyDoc = Annotated[
    ExplicitTopologyDoc | RingTopologyDoc, Field(discriminator="kind")
]
"""The topology of an instance: explicit, or a generated ring constellation."""
