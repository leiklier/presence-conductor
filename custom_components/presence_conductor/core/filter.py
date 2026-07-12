"""The occupancy filter (spec rule 4).

Owns every posterior change: the per-tick log-odds update with decay toward
the prior (4.1), the fast-attack path on strong gated move evidence (4.2),
the occupied hysteresis (4.3), the low-latency motion channel (4.4), and
the clamp (4.5). There is no fixed occupancy timeout anywhere in the engine
(4.1); the decay implements the hazard of departure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import activity, belief, evidence, timers
from .events import SensorFrame
from .model import Health, ZoneConfig, ZoneState

if TYPE_CHECKING:
    from .engine import ConductorEngine
    from .plan import Plan


def set_lambda(
    engine: ConductorEngine,
    zone: ZoneConfig,
    zst: ZoneState,
    lam: float,
    now: float,
    plan: Plan,
) -> None:
    """Assign a new posterior and run everything keyed to it."""
    zst.lam = belief.clamp(lam, engine.lam_min, engine.lam_max)  # 4.5
    if not zst.occupied and zst.lam >= engine.lam_on:  # 4.3: on at theta_on
        zst.occupied = True
        activity.on_occupied(zst, now)
    elif zst.occupied and zst.lam <= engine.lam_off:  # 4.3: off at theta_off
        zst.occupied = False
        activity.on_vacated(zone, zst, now, plan)
    # Between thresholds the binary holds (4.3).
    if zst.occupied:
        # Peak probability for the pass_by payload (5.2).
        zst.peak_probability = max(zst.peak_probability, zst.probability)
    evidence.update_background_clock(engine, zst, now)  # 3.4 freeze clock


def tick_zone(
    engine: ConductorEngine,
    zone: ZoneConfig,
    zst: ZoneState,
    dt: float,
    now: float,
    plan: Plan,
) -> None:
    """Rule 4.1: ``lambda <- decay(lambda) + llr * dt`` per tick."""
    if zst.health is not Health.OK:
        return  # 1.3: outputs hold their last state while UNKNOWN
    t = engine.config.tunables
    lam = belief.decay_toward(zst.lam, engine.lam_prior, dt, t.tau_decay)  # 4.1
    lam += evidence.llr(zst, t) * dt  # 4.1 / 3.2 (per second, scaled by dt)
    set_lambda(engine, zone, zst, lam, now, plan)


def on_frame(engine: ConductorEngine, frame: SensorFrame, now: float, plan: Plan) -> None:
    """Frame-side filter work: fast attack (4.2) and motion (4.4).

    Runs after :func:`.evidence.ingest_frame` stored this frame's evidence,
    so ``z_move``/gating reflect the frame being processed.
    """
    t = engine.config.tunables
    for zone in engine.config.zones_for_sensor(frame.sensor_id):
        zst = engine.state.zones[zone.zone_id]
        # 4.2: strong gated move evidence floors the posterior immediately -
        # not waiting for the next tick. This is the lights-on path.
        if zst.move_gated and zst.z_move >= t.z_attack:
            set_lambda(engine, zone, zst, max(zst.lam, engine.lam_attack), now, plan)  # 4.2
        # 4.4: gated, undamped fast channel.
        motion_evidence = zst.move_gated and (zst.z_move >= t.z_motion or frame.has_moving_target)
        if motion_evidence:
            zst.motion = True  # 4.4
            plan.start_timer(timers.motion_off(zone.zone_id), t.motion_hold)  # 4.4 (restart)
        elif zst.motion and timers.motion_off(zone.zone_id) not in engine._pending_timers:
            # No hold is pending (it fired while the zone was UNKNOWN and
            # outputs held, 1.3): this frame re-evaluates motion honestly.
            zst.motion = False  # 4.4


def on_motion_off(engine: ConductorEngine, zone_id: str, now: float, plan: Plan) -> None:
    """The motion hold expired: ``motion_hold`` without evidence (4.4)."""
    zone = engine.config.zone_or_none(zone_id)
    if zone is None:
        return
    zst = engine.state.zones[zone.zone_id]
    if zst.health is not Health.OK:
        return  # 1.3: outputs hold; the next frame re-evaluates motion
    zst.motion = False  # 4.4
