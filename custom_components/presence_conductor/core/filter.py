"""The occupancy filter (spec rule 4).

Owns every belief change: the chronological, exact constant-input
integration of the evidence rate with decay toward the prior (4.1), the
confirmed fast-attack path on strong gated move evidence (4.2), the
occupied hysteresis (4.3), the low-latency motion channel (4.4), and the
clamp (4.5). There is no fixed occupancy timeout anywhere in the engine
(4.1); the decay implements the hazard of departure.
"""

from __future__ import annotations

from itertools import pairwise
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
    """Assign a new belief and run everything keyed to it."""
    zst.lam = belief.clamp(lam, engine.lam_min, engine.lam_max)  # 4.5
    if not zst.occupied and zst.lam >= engine.lam_on:  # 4.3: on at theta_on
        zst.occupied = True
        activity.on_occupied(zst, now)
    elif zst.occupied and zst.lam <= engine.lam_off:  # 4.3: off at theta_off
        zst.occupied = False
        activity.on_vacated(zone, zst, now, plan)
    # Between thresholds the binary holds (4.3).
    if zst.occupied:
        # Peak confidence for the pass_by payload (5.2).
        zst.peak_confidence = max(zst.peak_confidence, zst.confidence)
    evidence.update_background_clock(engine, zst, now)  # 3.4 freeze clock


def advance_zone(
    engine: ConductorEngine,
    zone: ZoneConfig,
    zst: ZoneState,
    dt: float,
    now: float,
    plan: Plan,
) -> None:
    """Rule 4.1: integrate the evidence rate that was in force over ``dt``.

    Called before any event installs new evidence, so a frame's evidence is
    never applied to time before its own arrival. The rate is piecewise
    constant with breakpoints where observation windows expire (3.8): the
    interval is split at those absolute times and each segment integrates
    exactly, so the result stays invariant to tick cadence.
    """
    if zst.health is not Health.OK:
        return  # 1.3: outputs hold their last state while UNKNOWN
    if zst.recording is not None:
        return  # 3.3: suspended while calibrating — belief pinned at prior
    t = engine.config.tunables
    sensor = engine.state.sensors[zone.sensor_id]
    start = now - dt
    cuts = {start, now}
    for obs_at in (sensor.move_obs_at, sensor.still_obs_at):
        if obs_at is None:
            continue
        for window in (t.obs_budget, t.obs_hold):  # 3.8 expiry breakpoints
            expiry = obs_at + window
            if start < expiry < now:
                cuts.add(expiry)
    lam = zst.lam
    boundaries = sorted(cuts)
    for seg_start, seg_end in pairwise(boundaries):
        # Ages at the segment midpoint: u is constant inside a segment, and
        # the midpoint never lands exactly on an expiry boundary, so the
        # result is independent of how ticks slice the interval (4.1).
        mid = (seg_start + seg_end) / 2.0
        move_age = None if sensor.move_obs_at is None else mid - sensor.move_obs_at
        still_age = None if sensor.still_obs_at is None else mid - sensor.still_obs_at
        u = evidence.evidence_rate(zst, t, move_age, still_age)  # 3.2 / 3.8
        lam = belief.advance(lam, engine.lam_prior, u, seg_end - seg_start, t.tau_decay)  # 4.1
        # 4.5 applied per segment: the clamped trajectory rides the bound
        # until u changes, so the result is cadence-invariant — clamping
        # only at the end would let coarse schedules dive below the bound
        # and recover differently than fine ones.
        lam = belief.clamp(lam, engine.lam_min, engine.lam_max)
    set_lambda(engine, zone, zst, lam, now, plan)


def on_frame(engine: ConductorEngine, frame: SensorFrame, now: float, plan: Plan) -> None:
    """Frame-side filter work: fast attack (4.2) and motion (4.4).

    Runs after :func:`.evidence.ingest_frame` stored this frame's evidence,
    so the scores/gating reflect the frame being processed — and after the
    engine advanced time to ``now`` (4.1), so the attack floor lands on an
    up-to-date belief.
    """
    t = engine.config.tunables
    sensor = engine.state.sensors[frame.sensor_id]
    for zone in engine.config.zones_for_sensor(frame.sensor_id):
        zst = engine.state.zones[zone.zone_id]
        if zst.recording is not None:
            continue  # 3.3: suspended while calibrating
        path = "move_gate" if zst.move_from_gates else "move_agg"
        if zst.attack_path is not None and zst.attack_path != path:
            # A different evidence path has a different floor, tail and
            # dependence calibration. It cannot confirm the old chain —
            # even when this path switch arrived on a non-energy frame.
            zst.attack_count = 0
            zst.attack_last = None
            zst.attack_path = None
        # 4.2: strong move evidence floors the belief immediately - not
        # waiting for the next tick - once *confirmed*: attack_confirm
        # FRESH move observations, each past the analytic tail threshold
        # (candidacy is set by evidence ingest on the raw statistic).
        # Non-fresh frames carry no new move measurement (the adapter
        # re-emits its cached frame on any entity change) and leave the
        # chain untouched; elapsed time alone proves nothing.
        if sensor.move_energy_fresh:
            if zst.attack_candidate:
                # 4.2: confirmation squares the tail only if the confirming
                # observations are ~independent under H0. A calibrated
                # dependence estimate (3.7) for the path in force scales
                # the spacing: at gap tau the AR(1) residual correlation is
                # rho^tau ~ e^-2, while 1 s-spaced exceedances at rho=0.9
                # are essentially one tail event. tau = 1 (analytic
                # fallback / measured independent) keeps the raw tunables.
                tau = cal.tau if (cal := zst.stat_cal.get(path)) is not None else 1.0
                # The UI historically permitted attack_gap_min=0. Keep the
                # core total and prevent same-burst confirmation regardless;
                # preserve the configured window width when tau moves it.
                configured_min = max(0.1, t.attack_gap_min)
                gap_min = max(configured_min, tau) if tau > 1.0 else configured_min
                gap_width = max(0.0, t.attack_gap_max - configured_min)
                gap_max = gap_min + gap_width
                last = zst.attack_last
                if last is None or now - last > gap_max:
                    zst.attack_count = 1  # 4.2: chain (re)starts
                    zst.attack_last = now
                    zst.attack_path = path
                # 1 µs absorbs float noise in the gap arithmetic: monotonic
                # timestamps are large, and a 0.3 s difference can land
                # just under 0.3 in binary floats.
                elif now - last >= gap_min - 1e-6:
                    zst.attack_count += 1
                    zst.attack_last = now
                # else: within one radar burst (per-gate entities update in
                # a flurry from a single radar frame) or inside the
                # decorrelation gap - not a distinct observation (4.2).
                if zst.attack_count >= t.attack_confirm:
                    set_lambda(engine, zone, zst, max(zst.lam, engine.lam_attack), now, plan)
            else:
                zst.attack_count = 0  # 4.2: fresh non-qualifying resets
                zst.attack_last = None
                zst.attack_path = None
        # 4.4: gated, undamped fast channel. Under gate precedence (2.6) the
        # sensor-global has_moving_target flag is not zone evidence - the
        # owned gates already say where the mover is.
        motion_evidence = zst.move_gated and (
            zst.z_move >= t.z_motion or (frame.has_moving_target and not zst.move_from_gates)
        )
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
