"""Evidence model and calibration (spec rule 3).

Per zone and per channel (move, still) a robust noise floor ``(mu, sigma)``
(3.1) turns raw energies into a one-sided raw statistic ``S`` — the max
over owned gates (2.5) on the gate path, the gated aggregate deviation on
the fallback path. ``S`` is biased by construction (``E[S] > 0`` under
symmetric empty noise, growing with the owned-gate count), so it enters
the filter only as the centered score ``(S - m0) / s0`` against its own
empty-room distribution (3.2, 3.7): empirical when calibrated, analytic
Gaussian (:mod:`.stats`) otherwise. The per-second evidence rate subtracts
``k_bias`` unconditionally, making the expected rate in a calibrated empty
room strictly negative — noise must never drift a zone occupied (§3).

Baselines come from RecordBaseline windows sampled on the tick clock
(3.3) and drift slowly with the background while the zone is confidently
empty (3.4).
"""

from __future__ import annotations

import math
from itertools import pairwise
from statistics import fmean, median, pstdev
from typing import TYPE_CHECKING

from . import gating, stats
from .events import RecordBaseline, SensorFrame
from .model import (
    Activity,
    BaselineRecording,
    BaselineRow,
    ChannelStats,
    Health,
    StatBaseline,
    Tunables,
    ZoneState,
)
from .plan import BaselineRecorded
from .timers import baseline_end, motion_off

if TYPE_CHECKING:
    from .engine import ConductorEngine
    from .plan import Plan

#: Consistency factor turning a median absolute deviation into a Gaussian
#: sigma estimate (rule 3.1).
MAD_TO_SIGMA = 1.4826
#: Consistency factor for the EMA of absolute deviations (rule 3.4).
ABS_DEV_TO_SIGMA = 1.2533
#: One-sided 95% normal quantile for the scale UCB (rule 3.1).
_UCB_Z = 1.645

#: Statistic-calibration keys (rules 3.2, 3.7): channel + evidence path.
STAT_KEYS = ("move_agg", "move_gate", "still_agg", "still_gate")


def robust_stats(samples: list[float], sigma_min: float, quantum: float) -> tuple[float, float]:
    """Conservative out-of-sample noise floor over a window (rule 3.1).

    ``mu`` is the median. ``sigma`` deliberately over-covers: an
    underestimated scale inflates every future score (and voids the 4.2
    tail guarantee), so instead of the MAD point estimate it uses the
    distribution-free one-sided 95% upper confidence bound for the median
    absolute deviation — the k-th order statistic of the deviations with
    ``k = ceil(n/2 + 1.645 * sqrt(n) / 2)`` — plus half a quantization
    step (integer-reported energies put deviations on a grid; a MAD that
    quantizes one step low understates the scale by 25-40%).
    """
    mu = median(samples)
    deviations = sorted(abs(s - mu) for s in samples)
    n = len(deviations)
    # Dependence discount (3.1/3.7): the rank bound counts Bernoulli
    # trials, and autocorrelated samples carry fewer effective trials —
    # n_eff = n(1-rho)/(1+rho) with rho at its own upper bound. The
    # binomial sd then inflates from sqrt(n)/2 to n/(2*sqrt(n_eff)).
    # rho comes from the lag-1 sign agreement around the median (the rank
    # trials ARE those signs), which outliers cannot poison.
    signs = [1 if sample > mu else -1 for sample in samples if sample != mu]
    agree = sum(a == b for a, b in pairwise(signs)) / (len(signs) - 1) if len(signs) > 1 else 0.0
    rho = max(0.0, 2.0 * agree - 1.0)
    rho = min(0.999, rho + _UCB_Z * math.sqrt(max(0.0, 1.0 - rho * rho) / n))
    n_eff = max(2.0, n * (1.0 - rho) / (1.0 + rho))
    k = min(n, math.ceil(n / 2 + _UCB_Z * n / (2.0 * math.sqrt(n_eff))))  # 1-based rank
    mad_ucb = deviations[k - 1]
    return mu, max(sigma_min, MAD_TO_SIGMA * (mad_ucb + quantum / 2.0))


def onesided_z(energy: float, floor: ChannelStats) -> float:
    """Raw one-sided deviation from a noise floor (rules 2.5, 3.2)."""
    return max(0.0, (energy - floor.mu) / floor.sigma)


def gate_statistic(
    values: tuple[float | None, ...] | None,
    owned: tuple[int, ...],
    floors: dict[int, ChannelStats],
    t: Tunables,
) -> float | None:
    """Raw gate-path statistic for one channel (rules 2.5, 3.6).

    ``None`` means the frame's gate data does not cover this zone's channel
    — no gate tuple, no owned gates (2.4), or every owned gate unknown —
    and the aggregate path applies instead (2.6).
    """
    if values is None or not owned:
        return None
    default = ChannelStats(t.default_mu, t.default_sigma)  # 3.6: uncalibrated
    scores = [
        onesided_z(value, floors.get(index, default))  # 3.6: the gate's own floor
        for index in owned
        if index < len(values) and (value := values[index]) is not None
    ]
    if not scores:
        return None
    # 2.5: max over owned gates, not the sum — a person occupies one or two
    # gates, and summing would dilute a strong local return with the noise
    # of empty gates. The max is a multiple-comparison statistic; 3.7's
    # centering is what keeps it honest.
    return max(scores)


def centered(raw: float, cal: StatBaseline | None, m: int, t: Tunables) -> float:
    """Center a raw statistic against its empty-room distribution (3.2).

    ``cal`` is the empirical statistic calibration (3.7); ``None`` falls
    back to the analytic distribution of a max over ``m`` gates (or ``m=1``
    for the aggregate path). Clamped to ``[-z_neg_cap, z_cap]``, then the
    clamped score's own empty-room mean (``c0``) is subtracted — the
    asymmetric clamp alone would leave a positive residual mean (3.2).
    """
    if cal is None:
        m0, s0 = stats.onesided_max_stats(m)  # 3.7 analytic fallback
        c0 = stats.clipped_mean(m, t.z_neg_cap, t.z_cap)
        tau = 1.0  # 3.7: independence assumed when uncalibrated
    else:
        m0, s0, c0, tau = cal.mu, cal.sigma, cal.clip_mu, cal.tau
    score = (raw - m0) / max(s0, t.stat_sigma_min)
    # 3.2: correlated observations carry proportionally less independent
    # information per second (tau from 3.7).
    return (min(t.z_cap, max(-t.z_neg_cap, score)) - c0) / max(1.0, tau)


def _channel_term(z: float, k: float, age: float | None, t: Tunables) -> float:
    """One channel's contribution to the evidence rate (rules 3.2, 3.8).

    The absence margin (``k_bias``) is subtracted from every observed
    score first — E[z] = 0 on calibrated empty noise (3.2/3.7), so the
    margin makes the expected observed rate exactly ``-(k_move + k_still)
    * k_bias`` for any gate count. A positive shifted score is live for
    ``obs_budget`` after its observation — a held value is one
    measurement, not one per second; a non-positive one is live for
    ``obs_hold``. Past its window the channel is silent.
    """
    if age is None:
        return 0.0  # never observed
    shifted = z - t.k_bias  # 3.2: the absence margin
    if shifted > 0.0:
        return k * shifted if age <= t.obs_budget else 0.0
    return k * shifted if age <= t.obs_hold else 0.0


def evidence_rate(
    zst: ZoneState, t: Tunables, move_age: float | None, still_age: float | None
) -> float:
    """Per-second evidence rate (rule 3.2) at the given observation ages.

    ``k_bias`` is subtracted whenever at least one channel is
    observationally live (3.8) — with both channels silent the rate is 0
    and the belief simply relaxes toward the prior (4.1). ``u_cap`` bounds
    the upward rate: centering fixes the empty process's mean, not its
    tails, and a one-sample spike must not out-accumulate a genuine entry
    (spikes are 4.2's job). This is a calibrated anomaly score, not a
    log-likelihood ratio (8.7).
    """
    rate = _channel_term(zst.z_move, t.k_move, move_age, t) + _channel_term(
        zst.z_still, t.k_still, still_age, t
    )
    return min(t.u_cap, rate)  # 3.2


#: Normalized frame energies: aggregates plus optional per-gate tuples.
type _Energies = tuple[
    float | None,
    float | None,
    tuple[float | None, ...] | None,
    tuple[float | None, ...] | None,
]


def ingest_frame(engine: ConductorEngine, frame: SensorFrame, now: float | None) -> _Energies:
    """Normalize (1.4), gate (2.1-2.7) and score (3.2) one frame.

    Stores the resulting centered evidence on every zone of the frame's
    sensor and returns the normalized energies. ``now = None`` is the seed
    path (7.1): flag recency is unknown, so only flag-on distances gate.
    """
    t = engine.config.tunables
    sensor = engine.state.sensors.get(frame.sensor_id)
    move_e = gating.normalize_energy(frame.move_energy)  # 1.4
    still_e = gating.normalize_energy(frame.still_energy)  # 1.4
    gate_move = gating.normalize_gates(frame.gate_move)  # 1.4
    gate_still = gating.normalize_gates(frame.gate_still)  # 1.4
    gates = gating.gate_frame(engine.config, frame, sensor, now)  # 2.1-2.3, 2.7
    for zone in engine.config.zones_for_sensor(frame.sensor_id):
        zst = engine.state.zones[zone.zone_id]
        if zst.recording is not None:
            # 3.3: the estimator is suspended while calibrating — frames
            # feed the calibration rows only; the published state stays
            # the asserted "empty" (on_record_baseline pinned it).
            continue
        owned = engine.owned_gates[zone.zone_id]  # 2.4
        move_gated, still_gated = gates[zone.zone_id]
        if t.use_gate_evidence:
            raw_move = gate_statistic(gate_move, owned, zst.gate_move_baselines, t)  # 2.5
            raw_still = gate_statistic(gate_still, owned, zst.gate_still_baselines, t)  # 2.5
        else:
            # 2.6: gate evidence is experimental until real gate-path
            # timing is captured; calibration still records gate rows.
            raw_move = raw_still = None
        gate_tau, agg_tau = engine.attack_thresholds[zone.zone_id]  # 4.2
        if raw_move is not None:
            # 2.6: gate evidence replaces the aggregate path for this frame;
            # the channel counts as gated iff its own gates are elevated
            # (spatial attribution — 4.2 and 7.1 key on the flag).
            zst.move_gated = raw_move > 0.0
            zst.move_from_gates = True
            zst.z_move = centered(raw_move, zst.stat_cal.get("move_gate"), len(owned), t)
            # 4.2: candidacy is a tail event on the RAW statistic against
            # the analytic threshold for this path's gate count.
            zst.attack_candidate = raw_move >= gate_tau
        else:
            # 2.6: automatic per-frame fallback to the aggregate path.
            zst.move_gated = move_gated
            zst.move_from_gates = False
            # Rule 3.5 (rationale): while the frozen distance stays usable
            # (2.7), sub-threshold energy margins keep counting as evidence
            # even when the radar's own binary verdict has dropped the
            # target. An un-gated channel scores S = 0, which centers to a
            # negative score — absence evidence, continuously (3.2).
            raw = (
                onesided_z(move_e, zst.move_baseline) if move_gated and move_e is not None else 0.0
            )
            zst.z_move = centered(raw, zst.stat_cal.get("move_agg"), 1, t)
            zst.attack_candidate = raw >= agg_tau  # 4.2 (raw = 0 if un-gated)
        if raw_still is not None:
            zst.still_gated = raw_still > 0.0  # 2.6
            zst.still_from_gates = True
            zst.z_still = centered(raw_still, zst.stat_cal.get("still_gate"), len(owned), t)
        else:
            zst.still_gated = still_gated  # 2.6 fallback
            zst.still_from_gates = False
            raw = (
                onesided_z(still_e, zst.still_baseline)
                if still_gated and still_e is not None
                else 0.0
            )
            zst.z_still = centered(raw, zst.stat_cal.get("still_agg"), 1, t)
    return move_e, still_e, gate_move, gate_still


def apply_frame(engine: ConductorEngine, frame: SensorFrame, now: float) -> None:
    """Frame-side evidence work: freshness, ingest, sensor caches,
    adaptation."""
    sensor = engine.state.sensors[frame.sensor_id]
    # 1.1 observation clock: a counter advance is a new measurement; the
    # values alone cannot say (the adapter re-emits its cached frame on
    # any entity change, and a forced same-value re-publication IS a new
    # measurement).
    if frame.move_obs != sensor.move_obs:
        sensor.move_obs = frame.move_obs
        sensor.move_obs_at = now
    if frame.still_obs != sensor.still_obs:
        sensor.still_obs = frame.still_obs
        sensor.still_obs_at = now
    sensor.move_energy_fresh = frame.move_energy_obs != sensor.move_energy_obs  # 4.2
    sensor.move_energy_obs = frame.move_energy_obs
    energies = ingest_frame(engine, frame, now)
    move_e, still_e, gate_move, gate_still = energies
    # Caches feed the tick-aligned calibration rows (3.3); a None field
    # (never reported) keeps its previous value like the adapter's view.
    if move_e is not None:
        sensor.last_move_e = move_e
    if still_e is not None:
        sensor.last_still_e = still_e
    if gate_move is not None:
        sensor.last_gate_move = gate_move
    if gate_still is not None:
        sensor.last_gate_still = gate_still
    # 2.7: flag recency, after gating — the hold measures from when the
    # flag was last on, and a flag-on frame gates through the flag itself.
    if frame.has_moving_target:
        sensor.move_flag_at = now
    if frame.has_still_target:
        sensor.still_flag_at = now
    _adapt_background(engine, frame.sensor_id, energies, now)  # 3.4, 3.6


def update_background_clock(engine: ConductorEngine, zst: ZoneState, now: float) -> None:
    """Track how long the confidence has stayed below ``p_background`` (3.4).

    Called after every belief change. Adaptation freezes the moment the
    confidence rises: both the eligibility clock and the EMA anchor reset.
    """
    if zst.confidence < engine.config.tunables.p_background:
        if zst.below_since is None:
            zst.below_since = now
    else:  # 3.4: freeze immediately
        zst.below_since = None
        zst.last_adapt_at = None


def _ema_update(stats_: ChannelStats, energy: float, alpha: float, t: Tunables) -> None:
    """One 3.4 EMA step of a noise floor toward an observed energy.

    The deviation target carries the same half-quantum guard as 3.1: the
    EMA's sampling error is negligible at its time constant, the
    quantization bias is not.
    """
    stats_.mu += alpha * (energy - stats_.mu)
    deviation = ABS_DEV_TO_SIGMA * (abs(energy - stats_.mu) + t.energy_quantum / 2.0)
    stats_.sigma = max(t.sigma_min, stats_.sigma + alpha * (deviation - stats_.sigma))  # 3.1


def _adapt_background(
    engine: ConductorEngine,
    sensor_id: str,
    energies: _Energies,
    now: float,
) -> None:
    """Slow EMA of the noise floors while the zone is confidently empty
    (3.4), aggregate and per-gate alike (3.6). Floors only: the statistic
    calibration (3.7) refreshes on RecordBaseline."""
    t = engine.config.tunables
    move_e, still_e, gate_move, gate_still = energies
    zones = engine.config.zones_for_sensor(sensor_id)
    # Energies are per sensor, not per zone: while ANY zone of this sensor is
    # elevated, its energies are plausibly a person, so no sibling zone may
    # learn them as background (3.4: "without learning a person as noise").
    if any(engine.state.zones[z.zone_id].confidence >= t.p_background for z in zones):
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
        for energy, floor in ((move_e, zst.move_baseline), (still_e, zst.still_baseline)):
            if energy is not None:
                _ema_update(floor, energy, alpha, t)
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
                floor = floors.setdefault(index, ChannelStats(t.default_mu, t.default_sigma))
                _ema_update(floor, energy, alpha, t)


def collect_baseline_rows(engine: ConductorEngine) -> None:
    """One tick-aligned calibration row per recording zone (rule 3.3).

    Sampled from the sensor caches on the tick clock — never per entity
    change, which would weight samples by publish frequency and tear gate
    tuples across radar frames.
    """
    for zone in engine.config.zones:
        recording = engine.state.zones[zone.zone_id].recording
        if recording is None:
            continue
        sensor = engine.state.sensors[zone.sensor_id]
        if (
            sensor.last_move_e is None
            and sensor.last_still_e is None
            and sensor.last_gate_move is None
            and sensor.last_gate_still is None
        ):
            continue  # nothing ever reported: no row
        recording.rows.append(
            BaselineRow(
                move_e=sensor.last_move_e,
                still_e=sensor.last_still_e,
                gate_move=sensor.last_gate_move,
                gate_still=sensor.last_gate_still,
                # 3.1/3.7: held/deduplicated rows repeat one measurement.
                move_fresh=sensor.move_obs != recording.last_move_obs,
                still_fresh=sensor.still_obs != recording.last_still_obs,
            )
        )
        recording.last_move_obs = sensor.move_obs
        recording.last_still_obs = sensor.still_obs


def on_record_baseline(
    engine: ConductorEngine, event: RecordBaseline, now: float, plan: Plan
) -> None:
    """Open a calibration window (3.3). Re-issuing restarts the window.

    The zone's estimator is suspended for the window: the operator has
    asserted emptiness, and scoring frames against the old (possibly
    wrong) floors would let the calibration manufacture occupancy and
    ratchet home memory. The published state becomes exactly the
    assertion — empty at the prior — without a ``pass_by`` (nobody
    traversed; the estimate was declared void).
    """
    zone = engine.config.zone_or_none(event.zone_id)
    if zone is None:  # unknown zone: ignore
        return
    zst = engine.state.zones[zone.zone_id]
    zst.recording = BaselineRecording()
    zst.lam = engine.lam_prior  # 3.3: pinned at the empty prior
    zst.occupied = False
    zst.activity = Activity.EMPTY  # 3.3: no pass_by on this transition
    zst.occupied_since = None
    zst.dwell_seconds = 0.0
    zst.peak_confidence = 0.0
    zst.still_dominant_since = None
    zst.move_dominant_since = None
    zst.motion = False
    plan.cancel_timer(motion_off(zone.zone_id))  # 4.4 hold is void
    zst.z_move = zst.z_still = 0.0
    zst.move_gated = zst.still_gated = False
    zst.move_from_gates = zst.still_from_gates = False
    zst.attack_candidate = False  # 4.2: chain cleared
    zst.attack_count = 0
    zst.attack_last = None
    duration = event.duration
    if duration is None:
        duration = engine.config.tunables.baseline_duration  # 3.3: default 120 s
    plan.start_timer(baseline_end(zone.zone_id), duration)  # 3.3


def _lag1_autocorr(values: list[float]) -> float:
    """Lag-1 autocorrelation of a sequence; 0 for degenerate variance."""
    n = len(values)
    mean = fmean(values)
    var = sum((v - mean) ** 2 for v in values)
    if var <= 0.0:
        return 0.0
    cov = sum((values[i] - mean) * (values[i + 1] - mean) for i in range(n - 1))
    return max(0.0, min(0.999, cov / var))


def _distinct(values: list[float]) -> list[float]:
    """Collapse consecutive duplicates (3.1): held rows are one measurement."""
    out: list[float] = []
    for value in values:
        if not out or value != out[-1]:
            out.append(value)
    return out


def _stat_of(raws: list[float], m: int, t: Tunables) -> StatBaseline | None:
    """Empirical ``(m0, s0, c0)`` of the raw statistic over a window (3.7),
    shrunk toward safety.

    A short window estimates a scale poorly, and an underestimated ``s0``
    inflates every future score, so: at least ``stat_min_rows`` rows are
    required (else the analytic fallback stays), and ``s0`` is floored by
    the analytic reference deviation for the path's gate count — the
    window may recentre the score but never sharpen it beyond Gaussian.
    ``c0`` is the measured mean of the exact runtime transform (clamped
    score) over the window, so the final score is mean-zero on H0 (3.2).
    """
    if len(raws) < max(2, t.stat_min_rows):
        return None
    reference_s0 = stats.onesided_max_stats(m)[1]  # 3.7: safety floor
    m0 = fmean(raws)
    s0 = max(t.stat_sigma_min, reference_s0, pstdev(raws))
    c0 = fmean(min(t.z_cap, max(-t.z_neg_cap, (raw - m0) / s0)) for raw in raws)
    # 3.7: integrated autocorrelation time under an AR(1) assumption —
    # correlated empty noise must integrate at its information rate (3.2).
    # rho takes its one-sided 95% upper bound (se ~ sqrt((1-rho^2)/n)):
    # under strong dependence the window holds few independent samples, and
    # an underestimated tau under-discounts the evidence walk.
    rho = _lag1_autocorr(raws)
    rho_ucb = min(0.999, rho + 1.645 * math.sqrt(max(0.0, 1.0 - rho * rho) / len(raws)))
    tau = max(1.0, min(t.tau_int_max, (1.0 + rho_ucb) / (1.0 - rho_ucb)))
    return StatBaseline(m0, s0, c0, tau)


def on_baseline_end(engine: ConductorEngine, zone_id: str, now: float, plan: Plan) -> None:
    """Close a calibration window: replace floors (3.1, 3.6) and the
    statistic calibration (3.7)."""
    zone = engine.config.zone_or_none(zone_id)
    if zone is None:
        return
    zst = engine.state.zones[zone.zone_id]
    recording, zst.recording = zst.recording, None
    if recording is None:
        return
    t = engine.config.tunables
    rows = recording.rows
    if len(rows) < max(1, t.stat_min_rows):
        # 3.3 coverage: a scale cannot be certified from a handful of
        # points — keep the old calibration entirely, persist nothing.
        return
    owned = engine.owned_gates[zone.zone_id]  # 2.4

    def floor_of(column: list[float]) -> ChannelStats | None:
        # 3.1: only distinct observations carry information; a rank UCB
        # over held repeats overstates its confidence.
        samples = _distinct(column)
        if len(samples) >= t.stat_min_rows:
            return ChannelStats(*robust_stats(samples, t.sigma_min, t.energy_quantum))
        if (
            len(column) >= t.stat_min_rows
            and samples
            and (max(samples) - min(samples) <= t.energy_quantum)
        ):
            # 3.1: quiescent channel — the empty signal simply never moves;
            # no scale can or need be estimated beyond the floors.
            return ChannelStats(median(samples), t.sigma_min)
        return None  # too few independent changes: keep the old floor

    # 3.1/3.3: aggregate floors.
    move_samples = [row.move_e for row in rows if row.move_e is not None]
    still_samples = [row.still_e for row in rows if row.still_e is not None]
    if (floor := floor_of(move_samples)) is not None:
        zst.move_baseline = floor
    if (floor := floor_of(still_samples)) is not None:
        zst.still_baseline = floor
    # 3.6: per-gate floors are replaced alongside the aggregates.
    for column_of, floors in (
        (lambda row: row.gate_move, zst.gate_move_baselines),
        (lambda row: row.gate_still, zst.gate_still_baselines),
    ):
        for index in owned:
            column = [
                values[index]
                for row in rows
                if (values := column_of(row)) is not None
                and index < len(values)
                and values[index] is not None
            ]
            if (floor := floor_of(column)) is not None:
                floors[index] = floor  # 3.3 / 3.6
    # 3.7: statistic calibration — the post-aggregation raw statistic per
    # channel and path, scored against the floors just recorded, so the
    # centered score's empty-room mean is ~0 by construction. Full replace:
    # paths without rows fall back to the analytic values.
    # 3.7: statistics over fresh rows only — held rows repeat measurements.
    move_rows = [row for row in rows if row.move_fresh]
    still_rows = [row for row in rows if row.still_fresh]
    zst.stat_cal = {}
    for key, m, raws in (
        (
            "move_agg",
            1,
            [
                onesided_z(row.move_e, zst.move_baseline)
                for row in move_rows
                if row.move_e is not None
            ],
        ),
        (
            "still_agg",
            1,
            [
                onesided_z(row.still_e, zst.still_baseline)
                for row in still_rows
                if row.still_e is not None
            ],
        ),
        (
            "move_gate",
            len(owned),
            [
                s
                for row in move_rows
                if (s := gate_statistic(row.gate_move, owned, zst.gate_move_baselines, t))
                is not None
            ],
        ),
        (
            "still_gate",
            len(owned),
            [
                s
                for row in still_rows
                if (s := gate_statistic(row.gate_still, owned, zst.gate_still_baselines, t))
                is not None
            ],
        ),
    ):
        stat = _stat_of(raws, m, t)
        if stat is not None:
            zst.stat_cal[key] = stat
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
            frame_count=len(rows),
        )
    )
