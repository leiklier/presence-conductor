"""Guided occupied-emission calibration and validation."""

from __future__ import annotations

from custom_components.presence_conductor.core import emissions, evidence, guided, stats, timers
from custom_components.presence_conductor.core.engine import ConductorEngine
from custom_components.presence_conductor.core.events import (
    AdvanceFullCalibration,
    CancelCalibration,
    RecordBaseline,
    StartFullCalibration,
)
from custom_components.presence_conductor.core.model import (
    GuidedPhase,
    InitialSnapshot,
    ZoneBaselines,
)
from custom_components.presence_conductor.core.plan import FullCalibrationRecorded
from tests.core.harness import DESK, KONTOR, Harness, make_config


def _finish_baseline(h: Harness) -> None:
    for _ in range(3):
        h.send_frame(KONTOR, move_e=5, still_e=5)
        h.step_to(h.now + 1)
    h.fire_timer(timers.baseline_end(DESK), at=h.now + 1)


def _phase_frame(phase: GuidedPhase, *, bad_validation: bool = False) -> dict:
    if phase in {GuidedPhase.TRAIN_EMPTY, GuidedPhase.VALIDATE_EMPTY} or (
        bad_validation and phase.value.startswith("validate_")
    ):
        return {"move_e": 5, "still_e": 5}
    if phase in {GuidedPhase.TRAIN_MOVING, GuidedPhase.VALIDATE_MOVING}:
        return {"move_d": 100, "move_e": 30, "still_e": 5, "moving": True}
    if phase in {GuidedPhase.TRAIN_STANDING, GuidedPhase.VALIDATE_STANDING}:
        return {"still_d": 100, "move_e": 7, "still_e": 22, "still": True}
    return {"still_d": 100, "move_e": 5, "still_e": 16, "still": True}


def _record_all_phases(h: Harness, *, bad_validation: bool = False) -> None:
    for phase in guided.PHASES:
        session = h.zone(DESK).guided_calibration
        assert session is not None
        assert guided.PHASES[session.next_phase] is phase
        h.submit(AdvanceFullCalibration(DESK, duration=4))
        for _ in range(3):
            h.send_frame(KONTOR, **_phase_frame(phase, bad_validation=bad_validation))
            h.step_to(h.now + 1)
        h.fire_timer(timers.guided_phase_end(DESK), at=h.now + 1)


def test_full_calibration_commits_only_after_held_out_validation(monkeypatch) -> None:
    monkeypatch.setattr(guided, "MIN_PHASE_SAMPLES", 3)
    h = Harness(make_config(stat_min_rows=3))
    h.submit(StartFullCalibration(DESK, baseline_duration=4))
    assert h.zone(DESK).guided_calibration is not None
    assert not h.zone(DESK).occupied

    _finish_baseline(h)
    assert h.zone(DESK).guided_calibration.status == "waiting"
    assert h.zone(DESK).occupied_profile is None

    _record_all_phases(h)

    zone = h.zone(DESK)
    assert zone.guided_calibration is None
    assert zone.occupied_profile is not None
    assert zone.last_calibration_result == "full_ready"
    assert zone.last_validation is not None
    assert zone.last_validation.confusion.sensitivity == 1.0
    assert zone.last_validation.confusion.specificity == 1.0
    assert h.persist_count == 1  # Full is one atomic persistence transaction
    outcomes = [event for _, event in h.emitted if isinstance(event, FullCalibrationRecorded)]
    assert outcomes[-1].success

    persisted = ZoneBaselines(
        zone.move_baseline.mu,
        zone.move_baseline.sigma,
        zone.still_baseline.mu,
        zone.still_baseline.sigma,
        stats=dict(zone.stat_cal),
        gate_indices=h.engine.owned_gates[DESK],
        sensor_id=KONTOR,
        floor_fingerprint=stats.floor_calibration_fingerprint(h.config.tunables),
        gate_size_cm=h.config.sensor(KONTOR).gate_size_cm,
        occupied_profile=zone.occupied_profile,
    )
    reloaded = ConductorEngine(h.config, InitialSnapshot(baselines={DESK: persisted}))
    assert reloaded.state.zones[DESK].occupied_profile == zone.occupied_profile

    changed = make_config(stat_min_rows=3, theta_on=0.9)
    invalidated = ConductorEngine(changed, InitialSnapshot(baselines={DESK: persisted}))
    assert invalidated.state.zones[DESK].occupied_profile is None


def test_failed_validation_keeps_empty_only_fallback(monkeypatch) -> None:
    monkeypatch.setattr(guided, "MIN_PHASE_SAMPLES", 3)
    h = Harness(make_config(stat_min_rows=3))
    h.submit(StartFullCalibration(DESK, baseline_duration=4))
    _finish_baseline(h)
    _record_all_phases(h, bad_validation=True)

    zone = h.zone(DESK)
    assert zone.guided_calibration is None
    assert zone.occupied_profile is None
    assert zone.last_calibration_result == "failed"
    assert "sensitivity" in zone.last_calibration_error
    assert h.persist_count == 1  # only the independently useful empty baseline


def test_rejected_full_baseline_preserves_previous_state(monkeypatch) -> None:
    monkeypatch.setattr(guided, "MIN_PHASE_SAMPLES", 3)
    h = Harness(make_config(stat_min_rows=3))
    before = (
        h.zone(DESK).move_baseline.mu,
        h.zone(DESK).move_baseline.sigma,
        h.zone(DESK).still_baseline.mu,
        h.zone(DESK).still_baseline.sigma,
    )
    h.submit(StartFullCalibration(DESK, baseline_duration=5))
    h.fire_timer(timers.baseline_end(DESK), at=h.now + 5)

    zone = h.zone(DESK)
    assert zone.guided_calibration is None
    assert zone.last_calibration_result == "failed"
    assert (
        zone.move_baseline.mu,
        zone.move_baseline.sigma,
        zone.still_baseline.mu,
        zone.still_baseline.sigma,
    ) == before
    assert h.persist_count == 0


def test_failed_recalibration_restores_existing_full_profile(monkeypatch) -> None:
    monkeypatch.setattr(guided, "MIN_PHASE_SAMPLES", 3)
    h = Harness(make_config(stat_min_rows=3))
    h.submit(StartFullCalibration(DESK, baseline_duration=4))
    _finish_baseline(h)
    _record_all_phases(h)
    zone = h.zone(DESK)
    previous = (
        zone.move_baseline.mu,
        zone.move_baseline.sigma,
        zone.still_baseline.mu,
        zone.still_baseline.sigma,
        dict(zone.stat_cal),
        zone.occupied_profile,
        zone.last_validation,
    )
    persisted_before = h.persist_count

    h.submit(StartFullCalibration(DESK, baseline_duration=4))
    _finish_baseline(h)
    _record_all_phases(h, bad_validation=True)

    assert (
        zone.move_baseline.mu,
        zone.move_baseline.sigma,
        zone.still_baseline.mu,
        zone.still_baseline.sigma,
        zone.stat_cal,
        zone.occupied_profile,
        zone.last_validation,
    ) == previous
    assert h.persist_count == persisted_before


def test_cancel_discards_in_memory_rows_and_timers(monkeypatch) -> None:
    monkeypatch.setattr(guided, "MIN_PHASE_SAMPLES", 3)
    h = Harness(make_config(stat_min_rows=3))
    h.submit(StartFullCalibration(DESK, baseline_duration=4))
    _finish_baseline(h)
    h.submit(AdvanceFullCalibration(DESK, duration=30))
    h.send_frame(KONTOR, move_e=5, still_e=5)
    h.tick()

    h.submit(CancelCalibration(DESK))

    zone = h.zone(DESK)
    assert zone.guided_calibration is None
    assert zone.last_calibration_result == "cancelled"
    assert timers.guided_phase_end(DESK) not in h.deadlines


def test_same_sensor_cannot_run_overlapping_calibrations() -> None:
    h = Harness(make_config(stat_min_rows=3))
    h.submit(StartFullCalibration(DESK, baseline_duration=4))

    # Reissuing on the same sensor is rejected without replacing the session.
    first = h.zone(DESK).guided_calibration
    h.submit(StartFullCalibration(DESK, baseline_duration=4))
    assert h.zone(DESK).guided_calibration is first


def test_simple_baselines_cannot_overlap_on_sensor_global_energies() -> None:
    h = Harness(make_config(stat_min_rows=3))
    h.submit(RecordBaseline(DESK, duration=10))
    h.submit(RecordBaseline("kontor_door", duration=10))

    assert h.zone(DESK).recording is not None
    assert h.zone("kontor_door").recording is None


def test_full_session_and_profile_freeze_background_adaptation() -> None:
    config = make_config(stat_min_rows=3, t_background=0, tau_background=1)
    h = Harness(config)
    desk = h.zone(DESK)
    before = desk.move_baseline.mu
    h.submit(StartFullCalibration(DESK, baseline_duration=30))
    h.send_frame(KONTOR, move_e=40, still_e=40)
    assert desk.move_baseline.mu == before

    h.submit(CancelCalibration(DESK))
    desk.occupied_profile = emissions.fit_occupied_profile(
        [(-1.0, -1.0)] * 3,
        [(3.0, 0.0)] * 3,
        [(0.0, 3.0)] * 3,
    )
    desk.below_since = 0
    desk.last_adapt_at = 0
    h.send_frame(KONTOR, move_e=40, still_e=40, at=10)
    assert desk.move_baseline.mu == before


def test_mixed_gate_fallback_never_uses_profile() -> None:
    h = Harness()
    zone = h.zone(DESK)
    profile = emissions.fit_occupied_profile(
        [(-1.0, -1.0)] * 3,
        [(3.0, 0.0)] * 3,
        [(0.0, 3.0)] * 3,
    )
    zone.occupied_profile = profile
    zone.z_move = 3.0
    zone.z_still = -1.0
    zone.move_from_gates = True
    zone.still_from_gates = False

    with_profile = evidence.evidence_rate(zone, h.config.tunables, 0.0, 0.0)
    zone.occupied_profile = None
    assert with_profile == evidence.evidence_rate(zone, h.config.tunables, 0.0, 0.0)


def test_same_path_profile_requires_both_channels_fresh() -> None:
    h = Harness()
    zone = h.zone(DESK)
    zone.z_move = 3.0
    zone.z_still = 3.0
    zone.occupied_profile = emissions.fit_occupied_profile(
        [(-1.0, -1.0)] * 3,
        [(3.0, 0.0)] * 3,
        [(0.0, 3.0)] * 3,
    )
    learned = evidence.evidence_rate(zone, h.config.tunables, 0.0, 0.0)
    profile = zone.occupied_profile
    zone.occupied_profile = None
    legacy = evidence.evidence_rate(zone, h.config.tunables, 0.0, 0.0)
    zone.occupied_profile = profile

    assert learned != legacy
    zone.occupied_profile = None
    partial_legacy = evidence.evidence_rate(zone, h.config.tunables, 0.0, None)
    expired_legacy = evidence.evidence_rate(
        zone, h.config.tunables, 0.0, h.config.tunables.obs_budget + 0.1
    )
    zone.occupied_profile = profile
    assert evidence.evidence_rate(zone, h.config.tunables, 0.0, None) == partial_legacy
    assert (
        evidence.evidence_rate(zone, h.config.tunables, 0.0, h.config.tunables.obs_budget + 0.1)
        == expired_legacy
    )
