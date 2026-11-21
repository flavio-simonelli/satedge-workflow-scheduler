"""Activities inside time windows: the earliest finish and the latest start of the model.

An activity of length ``Δ`` must lie entirely inside one window ``[t_w, T_w]`` of a set
``W``. The model defines (section "Limiti temporali a priori")

- ``EF_W(t, Δ)``: its earliest finish if it starts no earlier than ``t``;
- ``LS_W(t, Δ)``: its latest start if it ends no later than ``t``.

A task on a satellite is an activity of length ``L̄_{i,s}`` in the computing windows. A
transmission of ``a`` on arc ``e`` is an activity of length ``D_{a,e} + λ_e`` in the
visibility windows, or in the single window ``[0, +inf)`` if the arc is always available,
which is written ``windows=None`` here.

As an extension used by the heuristic, :func:`earliest_finish` can also avoid the
intervals in which the resource is already busy. The activity then occupies the resource
for ``occupied`` seconds from its start, which may be shorter than its length: an arc is
busy only while it transmits (``D_{a,e}``), not while the last bit propagates (``λ_e``).
"""

from __future__ import annotations

from satedge_workflow_scheduler.common.intervals import (
    TIME_TOLERANCE,
    BusyTimeline,
    WindowSet,
)

_FREE = BusyTimeline()


def earliest_finish(
    windows: WindowSet | None,
    ready: float,
    length: float,
    *,
    busy: BusyTimeline = _FREE,
    occupied: float | None = None,
    eps: float = TIME_TOLERANCE,
) -> float | None:
    """``EF_W(ready, length)``: earliest finish of the activity, or ``None`` if it fits nowhere.

    Without ``busy`` this is exactly ``EF_W`` of the model. With ``busy`` it also avoids the
    intervals in which the resource is already taken: it is the version used by the
    heuristic, which plans around the resources already reserved. The start of the
    activity is the result minus ``length``.

    Args:
        windows: The windows, or ``None`` for ``[0, +inf)``.
        ready: The activity cannot start earlier.
        length: The activity, from its start, must lie inside one window for this long.
        busy: Intervals in which the resource is already taken.
        occupied: How long the activity takes the resource from its start, to be kept
            clear of ``busy``. Defaults to ``length``.
        eps: Tolerance for comparisons between instants.
    """
    occupied = length if occupied is None else occupied
    if windows is None:
        return busy.first_gap(ready, occupied, eps) + length
    for window in windows:
        if window.end < ready + length - eps:
            continue
        start = busy.first_gap(max(ready, window.start), occupied, eps)
        # a later gap in the same window would end even later: try the next window
        if start + length <= window.end + eps:
            return start + length
    return None


def latest_start(
    windows: WindowSet | None,
    due: float,
    length: float,
    *,
    eps: float = TIME_TOLERANCE,
) -> float | None:
    """``LS_W(due, length)``: latest start of the activity if it ends by ``due``, or ``None``.

    Args:
        windows: The windows, or ``None`` for ``[0, +inf)``.
        due: The activity cannot end later.
        length: The activity, from its start, must lie inside one window for this long.
        eps: Tolerance for comparisons between instants.
    """
    if windows is None:
        start = due - length
        return start if start >= -eps else None
    for window in reversed(windows.windows):
        start = min(due, window.end) - length
        if start >= window.start - eps:
            return start
    return None
