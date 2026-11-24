"""Colours, time units and labels of the figures."""

from __future__ import annotations

from collections.abc import Iterable

UNITS: dict[str, float] = {"s": 1.0, "min": 60.0, "h": 3600.0}
"""Seconds in one unit of the time axis."""

PALETTE: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
"""Categorical palette, in this order: adjacent colours stay apart for colour-blind readers.

The order is the safety mechanism. Past eight requests the colours repeat, so the figures
with many requests also name them (legend, labels, tables).
"""

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
