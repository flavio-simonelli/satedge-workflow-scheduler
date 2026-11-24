"""Colours, time units and labels of the figures."""

from __future__ import annotations

from collections.abc import Iterable

UNITS: dict[str, float] = {"s": 1.0, "min": 60.0, "h": 3600.0}
"""Seconds in one unit of the time axis."""

PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#D55E00",  # vermilion
    "#F0E442",  # yellow
    "#000000",  # black
)
"""Colour-blind safe palette, from Okabe and Ito; the colours repeat after eight requests."""

BAND = "#d9d9d9"

TICK_SIZE = 12
LABEL_SIZE = 13
TITLE_SIZE = 14
"""Colour of the availability windows drawn behind the bars."""

LABELS: dict[str, str] = {
    "hpc": "Calcolo HPC",
    "memory": "Occupazione PLMM",
    "capacity": "capacità",
    "peak": "picco",
    "urgent": "urgente",
    "not_urgent": "non urgente",
    "window": "finestra di calcolo",
    "visible": "visibilità",
    "transmission": "trasmissione",
    "route": "percorso dei dati",
}


def seconds_per(unit: str) -> float:
    """Seconds in one ``unit``.

    Raises:
        ValueError: If the unit is not one of ``s``, ``min``, ``h``.
    """
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {sorted(UNITS)}, got {unit!r}")
    return UNITS[unit]


def time_label(unit: str) -> str:
    """The label of the time axis, such as ``Tempo [min]``."""
    seconds_per(unit)
    return f"Tempo [{unit}]"


def request_colors(request_ids: Iterable[str]) -> dict[str, str]:
    """One colour per request, in order, from the palette."""
    return {rid: PALETTE[i % len(PALETTE)] for i, rid in enumerate(request_ids)}
