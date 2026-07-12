"""Activity classification and pass-by (spec rule 5).

A per-zone FSM driven by the posterior (via the occupied hysteresis) and
channel dominance: ``EMPTY -> PASSING -> ACTIVE / SETTLED`` (5.1). The
``occupied`` binary includes PASSING (5.3) — consumers that must not react
to walk-throughs key on ``activity`` instead; that split, not suppression
of short occupancy, is how "no flicker on walk-past" is achieved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .model import Activity, Health, ZoneConfig, ZoneState
from .plan import PassBy

if TYPE_CHECKING:
    from .engine import ConductorEngine
    from .plan import Plan


def on_occupied(zst: ZoneState, now: float) -> None:
    """Occupancy turned on: enter PASSING (5.1)."""
    zst.activity = Activity.PASSING  # 5.1
    zst.occupied_since = now
    zst.dwell_seconds = 0.0  # 5.4: dwell counts continuous occupancy
    zst.peak_probability = zst.probability  # 5.2 payload
    zst.still_dominant_since = None
    zst.move_dominant_since = None


def on_vacated(zone: ZoneConfig, zst: ZoneState, now: float, plan: Plan) -> None:
    """Occupancy turned off: back to EMPTY, maybe emitting pass_by (5.2)."""
    if zst.activity is Activity.PASSING and zst.occupied_since is not None:
        # 5.2: EMPTY reached *from* PASSING emits pass_by with the zone's
        # peak probability and traversal duration. From ACTIVE/SETTLED it
        # does not.
        plan.emit(PassBy(zone.zone_id, zst.peak_probability, now - zst.occupied_since))
    zst.activity = Activity.EMPTY  # 5.1
    zst.occupied_since = None
    zst.dwell_seconds = 0.0  # 5.4: reset on EMPTY
    zst.peak_probability = 0.0
    zst.still_dominant_since = None
    zst.move_dominant_since = None


def tick_zone(engine: ConductorEngine, zone: ZoneConfig, zst: ZoneState, now: float) -> None:
    """Advance dwell, channel dominance, and the FSM (5.1, 5.4)."""
    if zst.health is not Health.OK:
        return  # 1.3: outputs hold their last state while UNKNOWN
    if not zst.occupied or zst.occupied_since is None:
        return  # EMPTY: transitions in/out happen in the filter's hysteresis
    t = engine.config.tunables
    zst.dwell_seconds = now - zst.occupied_since  # 5.4

    # Channel dominance (5.1), weighted like the evidence model (3.2). A
    # dominance clock starts when its channel takes the lead and is cleared
    # only by the opposite channel taking over; quiet frames (posterior held
    # by decay alone) leave the clocks running.
    move_score = t.k_move * zst.z_move
    still_score = t.k_still * zst.z_still
    if still_score > move_score and still_score > 0.0:
        if zst.still_dominant_since is None:
            zst.still_dominant_since = now
        zst.move_dominant_since = None
    elif move_score > still_score and move_score > 0.0:
        if zst.move_dominant_since is None:
            zst.move_dominant_since = now
        zst.still_dominant_since = None

    still_takeover = (
        zst.still_dominant_since is not None and now - zst.still_dominant_since >= t.t_settle
    )
    move_takeover = (
        zst.move_dominant_since is not None and now - zst.move_dominant_since >= t.t_settle
    )
    match zst.activity:
        case Activity.PASSING:
            if still_takeover:
                zst.activity = Activity.SETTLED  # 5.1: still-takeover
            elif zst.dwell_seconds >= t.t_dwell:
                # 5.1: occupied past t_dwell. "Ongoing move evidence" is the
                # typical case; with no still-takeover, ACTIVE is the honest
                # default for decay-held occupancy too.
                zst.activity = Activity.ACTIVE
        case Activity.ACTIVE:
            if still_takeover:
                zst.activity = Activity.SETTLED  # 5.1: t_settle smoothing
        case Activity.SETTLED:
            if move_takeover:
                zst.activity = Activity.ACTIVE  # 5.1: t_settle smoothing
        case _:
            pass
