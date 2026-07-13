"""Room and home fusion (spec rule 6).

Fusion is monotone (6.4): a zone can only add occupancy to its room, never
veto another zone — separation is done at the gate (2.2), not here. Zones
in UNKNOWN health are excluded (6.3); a room (or the home) with every
contributor unknown publishes unknown, not off, so downstream automations
can distinguish "nobody there" from "blind".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import belief
from .model import ACTIVITY_SEVERITY, Activity, Health, ZoneState

if TYPE_CHECKING:
    from .engine import ConductorEngine


def refresh(engine: ConductorEngine, now: float, dt: float = 0.0) -> None:
    """Recompute all fused outputs from zone state.

    ``dt`` is the elapsed time for home-level decay: positive on ticks,
    zero on frames/events (which may only *raise* home presence, 6.5).
    """
    _refresh_rooms(engine)
    _refresh_home(engine, dt)


def _healthy(engine: ConductorEngine, zone_ids: list[str]) -> list[ZoneState]:
    return [
        zst
        for zone_id in zone_ids
        if (zst := engine.state.zones[zone_id]).health is Health.OK  # 6.3
    ]


def _refresh_rooms(engine: ConductorEngine) -> None:
    for room_id in engine.config.room_ids():
        room = engine.state.rooms[room_id]
        members = [z.zone_id for z in engine.config.zones_in_room(room_id)]
        healthy = _healthy(engine, members)  # 6.3: UNKNOWN zones excluded
        if not healthy:
            # 6.3: all members unknown -> publish unknown, not off.
            room.occupied = None
            room.motion = None
            room.probability = None
            room.activity = None
            room.settled = None
            continue
        # 6.1: occupied iff any healthy member zone is occupied.
        room.occupied = any(zst.occupied for zst in healthy)
        # 6.2: motion iff any healthy member zone's motion channel (4.4) is
        # on — the same undamped fast channel, fused with OR.
        room.motion = any(zst.motion for zst in healthy)
        # 6.1: noisy-OR over member posteriors, for diagnostics. Monotone by
        # construction (6.4).
        p_none = 1.0
        for zst in healthy:
            p_none *= 1.0 - zst.probability
        room.probability = 1.0 - p_none
        # 6.2: maximum-severity member state; settled iff any member SETTLED.
        room.activity = max((zst.activity for zst in healthy), key=lambda a: ACTIVITY_SEVERITY[a])
        room.settled = any(zst.activity is Activity.SETTLED for zst in healthy)


def _refresh_home(engine: ConductorEngine, dt: float) -> None:
    """Home-level presence (6.5)."""
    state = engine.state
    healthy = _healthy(engine, [z.zone_id for z in engine.config.zones])
    if not healthy:
        # 6.5: all zones unhealthy -> anyone_home publishes unknown.
        # lam_home freezes: with no data, decaying would claim knowledge.
        state.anyone_home = None
        state.home_probability = None
        return
    occupied_lams = [zst.lam for zst in healthy if zst.occupied]
    if occupied_lams:
        # 6.5: any healthy occupied zone drives home presence up immediately
        # (evidence, like 4.1). Ratchet to the strongest member posterior.
        state.lam_home = max(state.lam_home, max(occupied_lams))
    elif dt > 0.0:
        # 6.5: with all zones empty, decay toward the empty prior with
        # tau_home — deliberately much slower than zone decay, because the
        # sensors do not cover every room: all-zones-empty means "not seen
        # lately", not "gone".
        state.lam_home = belief.decay_toward(
            state.lam_home, engine.lam_prior, dt, engine.config.tunables.tau_home
        )
    state.lam_home = belief.clamp(state.lam_home, engine.lam_min, engine.lam_max)  # 4.5
    # 6.5: binary follows hysteresis thresholds like 4.3.
    if not engine._home_on and state.lam_home >= engine.lam_home_on:
        engine._home_on = True
    elif engine._home_on and state.lam_home <= engine.lam_home_off:
        engine._home_on = False
    state.anyone_home = engine._home_on
    state.home_probability = belief.sigmoid(state.lam_home)
