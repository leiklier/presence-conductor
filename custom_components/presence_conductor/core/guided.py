"""Guided occupied-emission calibration state machine.

The empty-room baseline remains the conservative foundation. A full run then
records independent training and validation phases over the already
empty-standardized move/still feature pair. Raw labeled rows are held only in
memory; only a compatible profile that passes held-out validation persists.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from . import belief, emissions, evidence, stats, timers
from .events import (
    AdvanceFullCalibration,
    CancelCalibration,
    RecordBaseline,
    StartFullCalibration,
)
from .model import (
    Activity,
    ChannelStats,
    EmissionScenario,
    GuidedCalibrationSession,
    GuidedPhase,
    GuidedPhaseRecording,
    ZoneCalibrationSnapshot,
)
from .plan import BaselineRecorded, FullCalibrationProgress, FullCalibrationRecorded

if TYPE_CHECKING:
    from .engine import ConductorEngine
    from .plan import Plan


PHASES: tuple[GuidedPhase, ...] = (
    GuidedPhase.TRAIN_EMPTY,
    GuidedPhase.TRAIN_MOVING,
    GuidedPhase.TRAIN_STANDING,
    GuidedPhase.TRAIN_SEATED,
    GuidedPhase.VALIDATE_MOVING,
    GuidedPhase.VALIDATE_STANDING,
    GuidedPhase.VALIDATE_SEATED,
    GuidedPhase.VALIDATE_EMPTY,
)

DEFAULT_PHASE_DURATION = 60.0
MIN_PHASE_SAMPLES = 15
MIN_SENSITIVITY = 0.70
MIN_SPECIFICITY = 0.80
MIN_SCENARIO_RECALL = 0.50


def sensor_is_suspended(engine: ConductorEngine, sensor_id: str) -> bool:
    """Whether an intentional full calibration suppresses this sensor."""

    return any(
        engine.state.zones[zone.zone_id].guided_calibration is not None
        for zone in engine.config.zones_for_sensor(sensor_id)
    )


def start_full(
    engine: ConductorEngine, event: StartFullCalibration, now: float, plan: Plan
) -> None:
    """Start with the required transactional empty-room baseline."""

    zone = engine.config.zone_or_none(event.zone_id)
    if zone is None:
        return
    for sibling in engine.config.zones_for_sensor(zone.sensor_id):
        state = engine.state.zones[sibling.zone_id]
        if state.recording is not None or state.guided_calibration is not None:
            plan.emit_control(
                FullCalibrationRecorded(
                    zone.zone_id,
                    False,
                    reason="another calibration is active on this sensor",
                )
            )
            return

    zst = engine.state.zones[zone.zone_id]
    zst.guided_calibration = GuidedCalibrationSession(previous=_snapshot(zst))
    zst.last_calibration_result = None
    zst.last_calibration_error = None
    _pin_sensor_empty(engine, zone.sensor_id, plan)
    evidence.on_record_baseline(
        engine,
        # Preserve the existing baseline transaction and coverage rules.
        RecordBaseline(zone.zone_id, event.baseline_duration),
        now,
        plan,
    )
    plan.emit_control(
        FullCalibrationProgress(zone.zone_id, "recording", "empty_baseline", 0, len(PHASES))
    )


def on_baseline_closed(engine: ConductorEngine, zone_id: str, plan: Plan) -> None:
    """Advance a full session only when its empty baseline committed."""

    zst = engine.state.zones.get(zone_id)
    if zst is None or zst.guided_calibration is None:
        return
    outcome = next(
        (
            event
            for event in reversed(plan.events)
            if isinstance(event, BaselineRecorded) and event.zone_id == zone_id
        ),
        None,
    )
    if outcome is None or not outcome.success:
        _fail(engine, zone_id, plan, "empty-room baseline was rejected")
        return
    session = zst.guided_calibration
    session.path = (
        "gate"
        if engine.config.tunables.use_gate_evidence and zst.gate_move_ready and zst.gate_still_ready
        else "aggregate"
    )
    session.status = "waiting"
    plan.emit_control(
        FullCalibrationProgress(
            zone_id,
            "waiting",
            PHASES[0].value,
            1,
            len(PHASES),
        )
    )


def advance(engine: ConductorEngine, event: AdvanceFullCalibration, now: float, plan: Plan) -> None:
    """Start the next operator-confirmed labeled phase."""

    zone = engine.config.zone_or_none(event.zone_id)
    if zone is None:
        return
    zst = engine.state.zones[zone.zone_id]
    session = zst.guided_calibration
    if session is None or session.status != "waiting" or session.next_phase >= len(PHASES):
        return
    sensor = engine.state.sensors[zone.sensor_id]
    phase = PHASES[session.next_phase]
    session.recording = GuidedPhaseRecording(
        phase=phase,
        started_at=now,
        last_frame_obs=sensor.frame_obs,
    )
    session.status = "recording"
    duration = DEFAULT_PHASE_DURATION if event.duration is None else max(5.0, event.duration)
    plan.start_timer(timers.guided_phase_end(zone.zone_id), duration)
    plan.emit_control(
        FullCalibrationProgress(
            zone.zone_id,
            "recording",
            phase.value,
            session.next_phase + 1,
            len(PHASES),
        )
    )


def collect_rows(engine: ConductorEngine, now: float) -> None:
    """Collect at most one coherent feature row per tick and observation epoch."""

    for zone in engine.config.zones:
        zst = engine.state.zones[zone.zone_id]
        session = zst.guided_calibration
        recording = session.recording if session is not None else None
        if recording is None:
            continue
        sensor = engine.state.sensors[zone.sensor_id]
        if sensor.frame_obs == recording.last_frame_obs:
            continue
        recording.last_frame_obs = sensor.frame_obs
        t = engine.config.tunables
        if (
            sensor.move_obs_at is None
            or sensor.still_obs_at is None
            or now - sensor.move_obs_at > t.obs_budget
            or now - sensor.still_obs_at > t.obs_budget
        ):
            continue  # the learned two-channel runtime branch would be silent
        if zst.move_from_gates != zst.still_from_gates:
            continue  # mixed transforms are not one two-feature model
        path = "gate" if zst.move_from_gates else "aggregate"
        if recording.path is None:
            recording.path = path
        if recording.path != path or session.path != path:
            continue  # per-frame fallback remains on the legacy estimator
        previous = recording.last_observed_at or recording.started_at
        recording.maximum_observation_gap = max(recording.maximum_observation_gap, now - previous)
        recording.last_observed_at = now
        recording.rows.append((zst.z_move, zst.z_still))
        # The joint branch begins when the newer channel arrives and expires
        # when the older reaches obs_budget. Replay that exact overlap.
        recording.observed_times.append(max(sensor.move_obs_at, sensor.still_obs_at))
        recording.evidence_budgets.append(
            max(0.0, t.obs_budget - abs(sensor.move_obs_at - sensor.still_obs_at))
        )


def end_phase(engine: ConductorEngine, zone_id: str, now: float, plan: Plan) -> None:
    """Validate one phase, then fit and validate after the final phase."""

    zone = engine.config.zone_or_none(zone_id)
    if zone is None:
        return
    zst = engine.state.zones[zone_id]
    session = zst.guided_calibration
    recording = session.recording if session is not None else None
    if session is None or recording is None:
        return
    sensor = engine.state.sensors[zone.sensor_id]
    session.recording = None
    if not sensor.available:
        _fail(engine, zone_id, plan, "sensor unavailable when the phase closed")
        return
    if (
        recording.last_observed_at is None
        or now - recording.last_observed_at > engine.config.tunables.stale_after
        or recording.maximum_observation_gap > engine.config.tunables.stale_after
    ):
        _fail(engine, zone_id, plan, "accepted feature observations were stale or interrupted")
        return
    if len(recording.rows) < MIN_PHASE_SAMPLES:
        _fail(
            engine,
            zone_id,
            plan,
            f"{len(recording.rows)} fresh samples; need at least {MIN_PHASE_SAMPLES}",
        )
        return

    target = (
        session.validation if recording.phase.value.startswith("validate_") else session.training
    )
    target[recording.phase] = recording.rows
    if target is session.validation:
        session.validation_times[recording.phase] = recording.observed_times
        session.validation_budgets[recording.phase] = recording.evidence_budgets
    session.next_phase += 1
    if session.next_phase < len(PHASES):
        session.status = "waiting"
        plan.emit_control(
            FullCalibrationProgress(
                zone_id,
                "waiting",
                PHASES[session.next_phase].value,
                session.next_phase + 1,
                len(PHASES),
                samples=len(recording.rows),
            )
        )
        return
    _finish(engine, zone_id, plan)


def cancel(engine: ConductorEngine, event: CancelCalibration, plan: Plan) -> None:
    """Cancel any calibration for the zone; committed data remains untouched."""

    zone = engine.config.zone_or_none(event.zone_id)
    if zone is None:
        return
    zst = engine.state.zones[zone.zone_id]
    active = zst.recording is not None or zst.guided_calibration is not None
    if not active:
        return
    _restore_previous(zst)
    zst.recording = None
    zst.guided_calibration = None
    zst.last_calibration_result = "cancelled"
    zst.last_calibration_error = "cancelled by operator"
    plan.cancel_timer(timers.baseline_end(zone.zone_id))
    plan.cancel_timer(timers.guided_phase_end(zone.zone_id))
    _release_sensor(engine, zone.sensor_id, plan)
    plan.emit_control(FullCalibrationRecorded(zone.zone_id, False, reason="cancelled by operator"))


def _finish(engine: ConductorEngine, zone_id: str, plan: Plan) -> None:
    zst = engine.state.zones[zone_id]
    session = zst.guided_calibration
    assert session is not None and session.path is not None
    owned = engine.owned_gates[zone_id]
    zone = engine.config.zone(zone_id)
    fingerprint = profile_fingerprint(engine.config, zone, owned, zst, session.path)
    try:
        profile = emissions.fit_occupied_profile(
            session.training[GuidedPhase.TRAIN_EMPTY],
            session.training[GuidedPhase.TRAIN_MOVING],
            session.training[GuidedPhase.TRAIN_STANDING]
            + session.training[GuidedPhase.TRAIN_SEATED],
            path=session.path,
            fingerprint=fingerprint,
        )
        metrics = emissions.validate_profile(
            profile,
            (
                EmissionScenario(
                    "empty", False, tuple(session.validation[GuidedPhase.VALIDATE_EMPTY])
                ),
                EmissionScenario(
                    "moving", True, tuple(session.validation[GuidedPhase.VALIDATE_MOVING])
                ),
                EmissionScenario(
                    "standing", True, tuple(session.validation[GuidedPhase.VALIDATE_STANDING])
                ),
                EmissionScenario(
                    "seated", True, tuple(session.validation[GuidedPhase.VALIDATE_SEATED])
                ),
            ),
        )
    except (KeyError, ValueError) as err:
        _fail(engine, zone_id, plan, f"profile fitting failed: {err}")
        return

    confusion = metrics.confusion
    sensitivity = confusion.sensitivity or 0.0
    specificity = confusion.specificity or 0.0
    weak = [
        metric.name
        for metric in metrics.scenarios
        if metric.expected_occupied and (metric.confusion.sensitivity or 0.0) < MIN_SCENARIO_RECALL
    ]
    empty_rates = [
        emissions.learned_evidence_rate(profile, row)
        for row in session.validation[GuidedPhase.VALIDATE_EMPTY]
    ]
    occupied_means = {
        phase.value.removeprefix("validate_"): sum(
            emissions.learned_evidence_rate(profile, row) for row in rows
        )
        / len(rows)
        for phase, rows in session.validation.items()
        if phase is not GuidedPhase.VALIDATE_EMPTY
    }
    empty_mean = sum(empty_rates) / len(empty_rates)
    metrics = replace(
        metrics,
        empty_mean_rate=empty_mean,
        occupied_mean_rates=occupied_means,
    )
    temporal_failures = []
    for phase, rows in session.validation.items():
        maximum = _replay_validation(
            engine,
            profile,
            rows,
            session.validation_times[phase],
            session.validation_budgets[phase],
        )
        expected_occupied = phase is not GuidedPhase.VALIDATE_EMPTY
        if expected_occupied and maximum < engine.lam_on:
            temporal_failures.append(f"{phase.value.removeprefix('validate_')} did not latch")
        if not expected_occupied and maximum >= engine.lam_on:
            temporal_failures.append("empty validation transiently latched")
    if (
        sensitivity < MIN_SENSITIVITY
        or specificity < MIN_SPECIFICITY
        or weak
        or empty_mean >= 0.0
        or any(mean <= 0.0 for mean in occupied_means.values())
        or temporal_failures
    ):
        reason = (
            f"held-out validation rejected: sensitivity {sensitivity:.1%}, "
            f"specificity {specificity:.1%}"
        )
        if weak:
            reason += f"; weak scenarios: {', '.join(weak)}"
        if empty_mean >= 0.0:
            reason += f"; mean empty drive {empty_mean:.3f} is not negative"
        weak_drive = [name for name, mean in occupied_means.items() if mean <= 0.0]
        if weak_drive:
            reason += f"; non-positive occupied drive: {', '.join(weak_drive)}"
        if temporal_failures:
            reason += f"; temporal replay: {', '.join(temporal_failures)}"
        zst.last_validation = metrics
        _fail(engine, zone_id, plan, reason, metrics=metrics)
        return

    profile = replace(profile, validation=metrics)
    zst.occupied_profile = profile
    zst.last_validation = metrics
    zst.last_calibration_result = "full_ready"
    zst.last_calibration_error = None
    _release_sensor(engine, zone.sensor_id, plan)
    zst.guided_calibration = None
    plan.persist_calibration = True
    plan.persist_calibration_zones.add(zone_id)
    plan.emit_control(FullCalibrationRecorded(zone_id, True, metrics=metrics))


def _replay_validation(
    engine: ConductorEngine,
    profile,
    rows: list[tuple[float, float]],
    observed_times: list[float],
    evidence_budgets: list[float],
) -> float:
    """Replay an ordered held-out capture through the actual belief ODE."""

    t = engine.config.tunables
    lam = engine.lam_prior
    maximum = lam
    for index, row in enumerate(rows):
        gap = (
            max(0.0, observed_times[index + 1] - observed_times[index])
            if index + 1 < len(rows)
            else evidence_budgets[index]
        )
        observed_for = min(gap, evidence_budgets[index])
        rate = min(t.u_cap, emissions.learned_evidence_rate(profile, row))
        lam = belief.clamp(
            belief.advance(lam, engine.lam_prior, rate, observed_for, t.tau_decay),
            engine.lam_min,
            engine.lam_max,
        )
        maximum = max(maximum, lam)
        silent_for = gap - observed_for
        if silent_for > 0.0:
            lam = belief.clamp(
                belief.advance(lam, engine.lam_prior, 0.0, silent_for, t.tau_decay),
                engine.lam_min,
                engine.lam_max,
            )
    return maximum


def _fail(
    engine: ConductorEngine,
    zone_id: str,
    plan: Plan,
    reason: str,
    *,
    metrics=None,
) -> None:
    zst = engine.state.zones.get(zone_id)
    if zst is not None:
        session = zst.guided_calibration
        baseline_committed = session is not None and session.status != "recording_baseline"
        previous_had_profile = (
            session is not None
            and session.previous is not None
            and session.previous.profile is not None
        )
        if previous_had_profile or not baseline_committed:
            _restore_previous(zst)
        elif session is not None:
            # First-time Full failure still keeps its independently useful
            # committed-in-memory empty baseline.
            plan.persist_calibration = True
            plan.persist_calibration_zones.add(zone_id)
        zst.recording = None
        zst.guided_calibration = None
        zst.last_calibration_result = "failed"
        zst.last_calibration_error = reason
    plan.cancel_timer(timers.baseline_end(zone_id))
    plan.cancel_timer(timers.guided_phase_end(zone_id))
    zone = engine.config.zone_or_none(zone_id)
    if zone is not None:
        _release_sensor(engine, zone.sensor_id, plan)
    plan.emit_control(FullCalibrationRecorded(zone_id, False, metrics=metrics, reason=reason))


def _pin_sensor_empty(engine: ConductorEngine, sensor_id: str, plan: Plan) -> None:
    """Keep intentional calibration movement out of consumer automations."""

    for zone in engine.config.zones_for_sensor(sensor_id):
        zst = engine.state.zones[zone.zone_id]
        zst.lam = engine.lam_prior
        zst.occupied = False
        zst.motion = False
        zst.activity = Activity.EMPTY
        zst.occupied_since = None
        zst.dwell_seconds = 0.0
        zst.peak_confidence = 0.0
        zst.attack_candidate = False
        zst.attack_count = 0
        zst.attack_last = None
        zst.attack_path = None
        plan.cancel_timer(timers.motion_off(zone.zone_id))


def _snapshot(zst) -> ZoneCalibrationSnapshot:
    return ZoneCalibrationSnapshot(
        move_mu=zst.move_baseline.mu,
        move_sigma=zst.move_baseline.sigma,
        still_mu=zst.still_baseline.mu,
        still_sigma=zst.still_baseline.sigma,
        gate_move=tuple(
            (index, floor.mu, floor.sigma)
            for index, floor in sorted(zst.gate_move_baselines.items())
        ),
        gate_still=tuple(
            (index, floor.mu, floor.sigma)
            for index, floor in sorted(zst.gate_still_baselines.items())
        ),
        gate_move_ready=zst.gate_move_ready,
        gate_still_ready=zst.gate_still_ready,
        statistics=dict(zst.stat_cal),
        profile=zst.occupied_profile,
        validation=zst.last_validation,
    )


def _restore_previous(zst) -> None:
    session = zst.guided_calibration
    previous = session.previous if session is not None else None
    if previous is None:
        return
    zst.move_baseline = ChannelStats(previous.move_mu, previous.move_sigma)
    zst.still_baseline = ChannelStats(previous.still_mu, previous.still_sigma)
    zst.gate_move_baselines = {
        index: ChannelStats(mu, sigma) for index, mu, sigma in previous.gate_move
    }
    zst.gate_still_baselines = {
        index: ChannelStats(mu, sigma) for index, mu, sigma in previous.gate_still
    }
    zst.gate_move_ready = previous.gate_move_ready
    zst.gate_still_ready = previous.gate_still_ready
    zst.stat_cal = dict(previous.statistics)
    zst.occupied_profile = previous.profile
    zst.last_validation = previous.validation


def _release_sensor(engine: ConductorEngine, sensor_id: str, plan: Plan) -> None:
    """Drop scripted held evidence before normal inference resumes."""
    sensor = engine.state.sensors[sensor_id]
    sensor.move_obs_at = None
    sensor.still_obs_at = None
    sensor.move_energy_fresh = False
    _pin_sensor_empty(engine, sensor_id, plan)
    for zone in engine.config.zones_for_sensor(sensor_id):
        zst = engine.state.zones[zone.zone_id]
        zst.z_move = zst.z_still = 0.0
        zst.move_gated = zst.still_gated = False
        zst.move_from_gates = zst.still_from_gates = False


def profile_fingerprint(config, zone, owned, zst, path: str) -> str:
    """Bind a profile to exact geometry, floors, and statistic transforms."""
    suffix = "gate" if path == "gate" else "agg"
    if path == "gate":
        floor_values = ";".join(
            f"{index}:{zst.gate_move_baselines[index].mu:.17g},"
            f"{zst.gate_move_baselines[index].sigma:.17g},"
            f"{zst.gate_still_baselines[index].mu:.17g},"
            f"{zst.gate_still_baselines[index].sigma:.17g}"
            for index in owned
        )
    else:
        floor_values = (
            f"{zst.move_baseline.mu:.17g},{zst.move_baseline.sigma:.17g},"
            f"{zst.still_baseline.mu:.17g},{zst.still_baseline.sigma:.17g}"
        )

    def stat_context(key: str) -> str:
        calibration = zst.stat_cal.get(key)
        if calibration is None:
            return f"analytic:{stats.calibration_fingerprint(key, owned, config.tunables)}"
        return (
            f"empirical:{calibration.mu:.17g},{calibration.sigma:.17g},"
            f"{calibration.clip_mu:.17g},{calibration.tau:.17g},"
            f"{calibration.decorrelation_seconds!r},{calibration.fingerprint!r}"
        )

    return emissions.emission_fingerprint(
        path=path,
        gate_indices=owned,
        floor_fingerprint=(
            f"{stats.floor_calibration_fingerprint(config.tunables)}|{floor_values}"
        ),
        move_stat_fingerprint=stat_context(f"move_{suffix}"),
        still_stat_fingerprint=stat_context(f"still_{suffix}"),
        sensor_id=zone.sensor_id,
        zone_geometry=(
            f"{zone.near_cm:.17g},{zone.far_cm:.17g},{int(zone.fallback)},"
            f"{config.tunables.margin_cm:.17g},{config.tunables.distance_hold:.17g};"
            f"temporal={config.tunables.obs_budget:.17g},"
            f"{config.tunables.u_cap:.17g},{config.tunables.tau_decay:.17g},"
            f"{config.tunables.p_prior:.17g},{config.tunables.theta_on:.17g},"
            f"{config.tunables.p_min:.17g},{config.tunables.p_max:.17g}"
        ),
    )
