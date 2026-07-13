"""Evidence model and calibration (spec rule 3).

Per zone and per channel (move, still) a robust noise floor ``(mu, sigma)``
turns raw energies into capped z-scores (3.1, 3.2). Baselines come from
explicit RecordBaseline windows (3.3) and drift slowly with the background
while the zone is confidently empty (3.4).

When a frame carries per-gate energies, each owned gate (2.4) is scored
against its own floor (3.6) and the zone's channel evidence is the max over
its owned gates (2.5), replacing the aggregate energy + distance path for
that frame (2.6). Calibration and adaptation extend per-gate under the same
windows and freeze conditions (3.6).
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


def gate_channel_z(
    values: tuple[float | None, ...] | None,
    owned: tuple[int, ...],
    floors: dict[int, ChannelStats],
    t: Tunables,
) -> float | None:
    """Zone gate evidence for one channel (rules 2.5, 3.6).

    ``None`` means the frame's gate data does not cover this zone's channel
    — no gate tuple, no owned gates (2.4), or every owned gate unknown —
    and the aggregate path applies instead (2.6).
    """
    if values is None or not owned:
        return None
    default = ChannelStats(t.default_mu, t.default_sigma)  # 3.6: uncalibrated
    scores = [
        z_score(value, floors.get(index, default), t)  # 3.6: the gate's own floor
        for index in owned
        if index < len(values) and (value := values[index]) is not None
    ]
    if not scores:
        return None
    # 2.5: max over owned gates, not the sum — a person occupies one or two
    # gates, and summing would dilute a strong local return with the noise
    # of empty gates. The max also credits two simultaneous people in
    # different zones of one sensor, impossible in the single-distance
    # model (2.1).
    return max(scores)


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


#: Normalized frame energies: aggregates plus optional per-gate tuples.
type _Energies = tuple[
    float | None,
    float | None,
    tuple[float | None, ...] | None,
    tuple[float | None, ...] | None,
]


def ingest_frame(engine: ConductorEngine, frame: SensorFrame) -> _Energies:
    """Normalize (1.4), gate (2.1-2.6) and score (3.2) one frame.

    Stores the resulting evidence on every zone of the frame's sensor and
    returns the normalized energies.
    """
    t = engine.config.tunables
    move_e = gating.normalize_energy(frame.move_energy)  # 1.4
    still_e = gating.normalize_energy(frame.still_energy)  # 1.4
    gate_move = gating.normalize_gates(frame.gate_move)  # 1.4
    gate_still = gating.normalize_gates(frame.gate_still)  # 1.4
    gates = gating.gate_frame(engine.config, frame)  # 2.1-2.3
    for zone in engine.config.zones_for_sensor(frame.sensor_id):
        zst = engine.state.zones[zone.zone_id]
        owned = engine.owned_gates[zone.zone_id]  # 2.4
        move_gated, still_gated = gates[zone.zone_id]
        z_gate_move = gate_channel_z(gate_move, owned, zst.gate_move_baselines, t)  # 2.5
        z_gate_still = gate_channel_z(gate_still, owned, zst.gate_still_baselines, t)  # 2.5
        if z_gate_move is not None:
            # 2.6: gate evidence replaces the aggregate path for this frame;
            # the channel counts as gated iff its own gates are elevated
            # (spatial attribution — 4.2 and 7.1 key on the flag).
            zst.z_move = z_gate_move
            zst.move_gated = z_gate_move > 0.0
            zst.move_from_gates = True
        else:
            # 2.6: automatic per-frame fallback to the aggregate path.
            zst.move_gated = move_gated
            zst.move_from_gates = False
            # Rule 3.5 (rationale): the gated z keeps crediting sub-threshold
            # energy margins as evidence even when the radar's own binary
            # verdict has dropped the target.
            zst.z_move = (
                z_score(move_e, zst.move_baseline, t) if move_gated and move_e is not None else 0.0
            )
        if z_gate_still is not None:
            zst.z_still = z_gate_still  # 2.6
            zst.still_gated = z_gate_still > 0.0
            zst.still_from_gates = True
        else:
            zst.still_gated = still_gated  # 2.6 fallback
            zst.still_from_gates = False
            zst.z_still = (
                z_score(still_e, zst.still_baseline, t)
                if still_gated and still_e is not None
                else 0.0
            )
    return move_e, still_e, gate_move, gate_still


def apply_frame(engine: ConductorEngine, frame: SensorFrame, now: float) -> None:
    """Frame-side evidence work: ingest, baseline collection, adaptation."""
    energies = ingest_frame(engine, frame)
    _collect_baseline(engine, frame.sensor_id, energies)  # 3.3, 3.6
    _adapt_background(engine, frame.sensor_id, energies, now)  # 3.4, 3.6


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


def _ema_update(stats: ChannelStats, energy: float, alpha: float, t: Tunables) -> None:
    """One 3.4 EMA step of a noise floor toward an observed energy."""
    stats.mu += alpha * (energy - stats.mu)
    deviation = ABS_DEV_TO_SIGMA * abs(energy - stats.mu)
    stats.sigma = max(t.sigma_min, stats.sigma + alpha * (deviation - stats.sigma))  # 3.1


def _adapt_background(
    engine: ConductorEngine,
    sensor_id: str,
    energies: _Energies,
    now: float,
) -> None:
    """Slow EMA of the noise floors while the zone is confidently empty
    (3.4), aggregate and per-gate alike (3.6)."""
    t = engine.config.tunables
    move_e, still_e, gate_move, gate_still = energies
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
            if energy is not None:
                _ema_update(stats, energy, alpha, t)
        # 3.6: per-gate floors follow the same clock, eligibility and freeze
        # conditions; a gate without a floor starts from the defaults.
        for values, floors in (
            (gate_move, zst.gate_move_baselines),
            (gate_still, zst.gate_still_baselines),
        ):
            if values is None:
                continue
            for index in engine.owned_gates[zone.zone_id]:  # 2.4
                if index >= len(values) or (energy := values[index]) is None:
                    continue
                stats = floors.setdefault(index, ChannelStats(t.default_mu, t.default_sigma))
                _ema_update(stats, energy, alpha, t)


def _collect_baseline(engine: ConductorEngine, sensor_id: str, energies: _Energies) -> None:
    """Collect empty-room frames for active RecordBaseline windows (3.3).

    Collection is un-gated: the operator asserts emptiness, and the robust
    statistics shrug off brief violations. Per-gate samples for the zone's
    owned gates ride along (3.6).
    """
    move_e, still_e, gate_move, gate_still = energies
    for zone in engine.config.zones_for_sensor(sensor_id):
        recording = engine.state.zones[zone.zone_id].recording
        if recording is None:
            continue
        if move_e is not None:
            recording.move_samples.append(move_e)
        if still_e is not None:
            recording.still_samples.append(still_e)
        for values, samples in (
            (gate_move, recording.gate_move_samples),
            (gate_still, recording.gate_still_samples),
        ):
            if values is None:
                continue
            for index in engine.owned_gates[zone.zone_id]:  # 2.4 / 3.6
                if index < len(values) and (value := values[index]) is not None:
                    samples.setdefault(index, []).append(value)


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
    if not (
        recording.move_samples
        or recording.still_samples
        or recording.gate_move_samples
        or recording.gate_still_samples
    ):
        return  # nothing observed: keep the old floors, persist nothing
    # A channel or gate that produced no samples keeps its previous floor.
    if recording.move_samples:
        mu, sigma = robust_stats(recording.move_samples, t.sigma_min)  # 3.1
        zst.move_baseline = ChannelStats(mu, sigma)  # 3.3
    if recording.still_samples:
        mu, sigma = robust_stats(recording.still_samples, t.sigma_min)  # 3.1
        zst.still_baseline = ChannelStats(mu, sigma)  # 3.3
    # 3.6: per-gate floors are replaced alongside the aggregates.
    for samples_by_gate, floors in (
        (recording.gate_move_samples, zst.gate_move_baselines),
        (recording.gate_still_samples, zst.gate_still_baselines),
    ):
        for index in sorted(samples_by_gate):
            mu, sigma = robust_stats(samples_by_gate[index], t.sigma_min)  # 3.1
            floors[index] = ChannelStats(mu, sigma)  # 3.3 / 3.6
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
