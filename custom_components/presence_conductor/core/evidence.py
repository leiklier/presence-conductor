"""Evidence model and calibration (spec rule 3).

Per zone and per channel (move, still) a robust noise floor ``(mu, sigma)``
turns raw energies into capped z-scores (3.1, 3.2). Baselines come from
explicit RecordBaseline windows (3.3) and drift slowly with the background
while the zone is confidently empty (3.4).
"""

from __future__ import annotations

import math
from statistics import median
from typing import TYPE_CHECKING

from . import gating
from .events import RecordBaseline, SensorFrame
from .model import BaselineRecording, ChannelStats, Health, Tunables, ZoneState
from .plan import BaselineRecorded
from .timers import baseline_end

if TYPE_CHECKING:
    from .engine import ConductorEngine
    from .plan import Plan

#: Consistency factor turning a median absolute deviation into a Gaussian
#: sigma estimate (rule 3.1).
MAD_TO_SIGMA = 1.4826
#: Consistency factor for the EMA of absolute deviations (rule 3.4).
ABS_DEV_TO_SIGMA = 1.2533


def robust_stats(samples: list[float], sigma_min: float) -> tuple[float, float]:
    """Median / MAD over a calibration window, sigma floored (rule 3.1)."""
    mu = median(samples)
    mad = median([abs(s - mu) for s in samples])
    return mu, max(sigma_min, MAD_TO_SIGMA * mad)


def z_score(energy: float, stats: ChannelStats, t: Tunables) -> float:
    """``z = max(0, (energy - mu) / sigma)``, capped at ``z_cap`` (rule 3.2)."""
    return min(t.z_cap, max(0.0, (energy - stats.mu) / stats.sigma))


def llr(zst: ZoneState, t: Tunables) -> float:
    """Per-frame log-likelihood ratio, per second (rule 3.2).

    ``k_absence`` applies only when both channels are un-gated or at
    baseline (their z-scores are zero either way — :func:`ingest_frame`
    zeroes un-gated channels).
    """
    positive = t.k_move * zst.z_move + t.k_still * zst.z_still  # 3.2
    if positive > 0.0:
        return positive
    return -t.k_absence  # 3.2


def ingest_frame(engine: ConductorEngine, frame: SensorFrame) -> tuple[float | None, float | None]:
    """Normalize (1.4), gate (2.1-2.3) and score (3.2) one frame.

    Stores the resulting evidence on every zone of the frame's sensor and
    returns the normalized ``(move_energy, still_energy)``.
    """
    t = engine.config.tunables
    move_e = gating.normalize_energy(frame.move_energy)  # 1.4
    still_e = gating.normalize_energy(frame.still_energy)  # 1.4
    gates = gating.gate_frame(engine.config, frame)  # 2.1-2.3
    for zone in engine.config.zones_for_sensor(frame.sensor_id):
        zst = engine.state.zones[zone.zone_id]
        zst.move_gated, zst.still_gated = gates[zone.zone_id]
        # Rule 3.5 (rationale): the gated z keeps crediting sub-threshold
        # energy margins as evidence even when the radar's own binary
        # verdict has dropped the target.
        zst.z_move = (
            z_score(move_e, zst.move_baseline, t) if zst.move_gated and move_e is not None else 0.0
        )
        zst.z_still = (
            z_score(still_e, zst.still_baseline, t)
            if zst.still_gated and still_e is not None
            else 0.0
        )
    return move_e, still_e


def apply_frame(engine: ConductorEngine, frame: SensorFrame, now: float) -> None:
    """Frame-side evidence work: ingest, baseline collection, adaptation."""
    move_e, still_e = ingest_frame(engine, frame)
    _collect_baseline(engine, frame.sensor_id, move_e, still_e)  # 3.3
    _adapt_background(engine, frame.sensor_id, move_e, still_e, now)  # 3.4


def update_background_clock(engine: ConductorEngine, zst: ZoneState, now: float) -> None:
    """Track how long the posterior has stayed below ``p_background`` (3.4).

    Called after every posterior change. Adaptation freezes the moment the
    posterior rises: both the eligibility clock and the EMA anchor reset.
    """
    if zst.probability < engine.config.tunables.p_background:
        if zst.below_since is None:
            zst.below_since = now
    else:  # 3.4: freeze immediately
        zst.below_since = None
        zst.last_adapt_at = None


def _adapt_background(
    engine: ConductorEngine,
    sensor_id: str,
    move_e: float | None,
    still_e: float | None,
    now: float,
) -> None:
    """Slow EMA of the noise floor while the zone is confidently empty (3.4)."""
    t = engine.config.tunables
    zones = engine.config.zones_for_sensor(sensor_id)
    # Energies are per sensor, not per zone: while ANY zone of this sensor is
    # elevated, its energies are plausibly a person, so no sibling zone may
    # learn them as background (3.4: "without learning a person as noise").
    if any(engine.state.zones[z.zone_id].probability >= t.p_background for z in zones):
        return
    for zone in zones:
        zst = engine.state.zones[zone.zone_id]
        if zst.health is not Health.OK or zst.recording is not None:
            continue  # blind or actively calibrating: hold the floor
        if zst.below_since is None or now - zst.below_since < t.t_background:
            continue  # 3.4: quiet for at least t_background first
        if zst.last_adapt_at is None:
            zst.last_adapt_at = now  # anchor the EMA clock, adapt from here
            continue
        dt = now - zst.last_adapt_at
        zst.last_adapt_at = now
        if dt <= 0.0:
            continue
        alpha = 1.0 - math.exp(-dt / t.tau_background)  # 3.4: tau_background
        for energy, stats in ((move_e, zst.move_baseline), (still_e, zst.still_baseline)):
            if energy is None:
                continue
            stats.mu += alpha * (energy - stats.mu)
            deviation = ABS_DEV_TO_SIGMA * abs(energy - stats.mu)
            stats.sigma = max(t.sigma_min, stats.sigma + alpha * (deviation - stats.sigma))  # 3.1


def _collect_baseline(
    engine: ConductorEngine, sensor_id: str, move_e: float | None, still_e: float | None
) -> None:
    """Collect empty-room frames for active RecordBaseline windows (3.3).

    Collection is un-gated: the operator asserts emptiness, and the robust
    statistics shrug off brief violations.
    """
    for zone in engine.config.zones_for_sensor(sensor_id):
        recording = engine.state.zones[zone.zone_id].recording
        if recording is None:
            continue
        if move_e is not None:
            recording.move_samples.append(move_e)
        if still_e is not None:
            recording.still_samples.append(still_e)


def on_record_baseline(
    engine: ConductorEngine, event: RecordBaseline, now: float, plan: Plan
) -> None:
    """Open a calibration window (3.3). Re-issuing restarts the window."""
    zone = engine.config.zone_or_none(event.zone_id)
    if zone is None:  # unknown zone: ignore
        return
    engine.state.zones[zone.zone_id].recording = BaselineRecording()
    duration = event.duration
    if duration is None:
        duration = engine.config.tunables.baseline_duration  # 3.3: default 120 s
    plan.start_timer(baseline_end(zone.zone_id), duration)  # 3.3


def on_baseline_end(engine: ConductorEngine, zone_id: str, now: float, plan: Plan) -> None:
    """Close a calibration window and replace ``(mu, sigma)`` (3.3)."""
    zone = engine.config.zone_or_none(zone_id)
    if zone is None:
        return
    zst = engine.state.zones[zone.zone_id]
    recording, zst.recording = zst.recording, None
    if recording is None:
        return
    t = engine.config.tunables
    # A channel that produced no samples keeps its previous floor.
    if recording.move_samples:
        mu, sigma = robust_stats(recording.move_samples, t.sigma_min)  # 3.1
        zst.move_baseline = ChannelStats(mu, sigma)  # 3.3
    if recording.still_samples:
        mu, sigma = robust_stats(recording.still_samples, t.sigma_min)  # 3.1
        zst.still_baseline = ChannelStats(mu, sigma)  # 3.3
    if not recording.move_samples and not recording.still_samples:
        return  # nothing observed: keep the old floor, persist nothing
    zst.last_adapt_at = None  # new floor: re-anchor the adaptation EMA
    # 3.3: baselines persist in the config entry (adapter saves them).
    plan.persist_calibration = True
    plan.emit(
        BaselineRecorded(
            zone_id=zone.zone_id,
            move_mu=zst.move_baseline.mu,
            move_sigma=zst.move_baseline.sigma,
            still_mu=zst.still_baseline.mu,
            still_sigma=zst.still_baseline.sigma,
            frame_count=max(len(recording.move_samples), len(recording.still_samples)),
        )
    )
