"""Tests of the shared earliest-arrival search, on the network of the MILP sample."""

from __future__ import annotations

import pytest
from sample import sample_instance

from satedge_workflow_scheduler.domain.network import Network
from satedge_workflow_scheduler.scheduling.earliest import (
    Crossing,
    earliest_arrivals,
    path_to,
)


def _static(network: Network) -> Crossing:
    def crossing(arc: int, ready: float) -> tuple[float, float]:
        link = network.arcs[arc]
        return ready, ready + link.transmission_time(1.0) + link.latency

    return crossing


def test_the_source_is_reached_at_its_ready_instant() -> None:
    network = sample_instance().network
    reached = earliest_arrivals(network, {"S1": 3.0}, _static(network))
    assert reached["S1"].arrival == 3.0
    assert path_to(reached, "S1") == ()


def test_the_earliest_arrival_is_found_through_the_intermediate_satellite() -> None:
    network = sample_instance().network
    reached = earliest_arrivals(network, {"S1": 0.0}, _static(network))
    # S1 → S2 for 1 Gb: 1 + 1 = 2 s; then S2 → G: 1 / 2 + 5 = 5.5 s, so G at 7.5 s
    assert reached["S2"].arrival == pytest.approx(2.0)
    assert reached["G"].arrival == pytest.approx(7.5)
    assert path_to(reached, "G") == (0, 2)


def test_a_node_that_does_not_relay_is_reached_but_not_expanded() -> None:
    network = sample_instance().network
    reached = earliest_arrivals(
        network, {"S1": 0.0}, _static(network), relays=lambda node: node != "S2"
    )
    assert "S2" in reached
    assert "G" not in reached


def test_a_crossing_that_is_impossible_closes_the_arc() -> None:
    network = sample_instance().network
    reached = earliest_arrivals(network, {"S1": 0.0}, lambda arc, ready: None)
    assert set(reached) == {"S1"}
