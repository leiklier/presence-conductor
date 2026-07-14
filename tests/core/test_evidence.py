"""Rule 3: noise floors, evidence scores, calibration, background drift."""

from __future__ import annotations

import pytest

from custom_components.presence_conductor.core import timers
from custom_components.presence_conductor.core.belief import advance
from custom_components.presence_conductor.core.events import (
    RecordBaseline,
    SensorAvailability,
    SetEnabled,
)
from custom_components.presence_conductor.core.evidence import robust_stats
from custom_components.presence_conductor.core.model import Coverage, StatBaseline

from .harness import (
    DESK,
    DOOR,
    KONTOR,
    MU,
    SOFAKROK,
    Z_EMPTY,
    Harness,
    centered_of_raw,
    make_config,
    make_snapshot,
)


class TestRule31RobustStats:
    def test_rule_3_1_median_shrugs_off_outliers(self) -> None:
        # A realistically sized window (>= stat_min_rows): the rank UCB
        # stays inside the bulk and the two outliers never reach it.
        samples = [0.08, 0.09, 0.10, 0.11, 0.12] * 24 + [0.90, 0.95]
        mu, sigma = robust_stats(samples, sigma_min=0.001, quantum=0.0)
        assert mu == pytest.approx(0.10)
        assert sigma <= 1.4826 * 0.021  # bulk deviation, not an outlier

    def test_rule_3_1_sigma_is_floored(self) -> None:
        mu, sigma = robust_stats([0.1] * 20, sigma_min=0.02, quantum=0.0)
        assert mu == pytest.approx(0.1)
        assert sigma == 0.02  # constant samples: zero deviations, floor wins

    def test_rule_3_1_half_quantum_guards_grid_bias(self) -> None:
        # All deviations exactly one grid step: the half-quantum term keeps
        # the scale from trusting the grid (3.1).
        _mu, sigma = robust_stats([0.1, 0.11] * 30, sigma_min=0.001, quantum=0.01)
        assert sigma == pytest.approx(1.4826 * (0.005 + 0.005))

    def test_rule_3_1_ucb_covers_the_true_scale(self) -> None:
        """The reviewer's failure mechanism: quantized-MAD point estimates
        fitted 0.59-0.89x the true scale on ~120-row windows. The UCB +
        half-quantum floor must cover the true scale in the overwhelming
        majority of fits (seeded)."""
        import random

        rng = random.Random(2)
        true_sigma = 0.05  # raw sigma 5, normalized
        under = 0
        for _ in range(300):
            samples = [max(0, min(100, round(rng.gauss(20, 5)))) / 100.0 for _ in range(119)]
            _, sigma = robust_stats(samples, sigma_min=0.02, quantum=0.01)
            if sigma < true_sigma:
                under += 1
        assert under <= 6  # ~2% vs the ~50% of a point estimate


class TestRule32EvidenceScore:
    def test_rule_3_2_z_capped_at_z_cap(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, still_d=100, still_e=100, still=True)
        assert h.zone(DESK).z_still == pytest.approx(centered_of_raw(100))  # z_cap

    def test_rule_3_2_weights_shape_the_evidence_rate(self) -> None:
        h = Harness()
        h.send_frame(SOFAKROK, still_d=100, still_e=35, still=True)  # z_still = 6
        h.tick()
        # 4.1: exact constant-input update over dt = 1 with
        # u = k_still * 6 + k_move * Z_EMPTY - k_bias.
        t = h.config.tunables
        u = min(
            t.u_cap,
            t.k_still * (centered_of_raw(35) - t.k_bias) + t.k_move * (Z_EMPTY - t.k_bias),
        )
        expected = advance(h.engine.lam_prior, h.engine.lam_prior, u, 1.0, 90.0)
        assert h.zone("sofakrok_zone").lam == pytest.approx(expected)

    def test_rule_3_2_bias_applies_while_observed(self) -> None:
        # A freshly observed channel at scores 0 (the empty mean) still
        # integrates -k_bias: expected empty evidence is negative. With no
        # observation at all the rate is 0 (3.8) and the belief only
        # relaxes toward the prior.
        h = Harness()
        h.tick()
        assert h.zone(DESK).lam == pytest.approx(h.engine.lam_prior)  # 3.8 silence
        h.send_frame(KONTOR, move_e=10.001, still_e=10.001)  # observed, near defaults
        h.tick()
        assert h.zone(DESK).lam < h.engine.lam_prior  # bias flowed

    def test_rule_3_2_clear_still_evidence_outweighs_the_bias(self) -> None:
        h = Harness()
        h.send_frame(SOFAKROK, still_d=100, still_e=20, still=True)  # z = 4.45
        h.tick()
        assert h.zone("sofakrok_zone").lam > h.engine.lam_prior

    def test_rule_3_2_marginal_still_evidence_is_net_negative(self) -> None:
        # Raw z = 1 is barely above what empty noise produces (E[max(0,Z)]
        # = 0.4): after centering it no longer holds a zone up on its own.
        h = Harness()
        h.send_frame(SOFAKROK, still_d=100, still_e=10, still=True)
        h.tick()
        assert h.zone("sofakrok_zone").lam < h.engine.lam_prior


class TestRule33BaselineCalibration:
    def test_rule_3_3_record_baseline_replaces_stats_robustly(self) -> None:
        h = Harness()
        h.submit(RecordBaseline(DESK, duration=80.0))
        assert h.deadlines[timers.baseline_end(DESK)] == pytest.approx(80.0)
        raw = [8, 9, 10, 11, 12] * 16
        raw[3] = 90  # a person walks through: brief violations don't poison
        raw[11] = 95
        for value in raw:
            # 3.3: rows are sampled on the tick clock from the held values,
            # so each 1 Hz frame is captured by the following tick.
            h.send_frame(KONTOR, move_e=value, still_e=value)
            h.step_to(h.now + 1.0)
        h.run(5)  # window closed at t=80
        desk = h.zone(DESK)
        assert desk.recording is None
        assert desk.move_baseline.mu == pytest.approx(0.10, abs=0.011)
        # 3.1: UCB of the deviations (0.02) + half quantum, times 1.4826.
        assert desk.move_baseline.sigma == pytest.approx(1.4826 * 0.025, abs=0.002)
        assert desk.still_baseline.mu == pytest.approx(0.10, abs=0.011)
        assert h.persist_count == 1  # 3.3: baselines persist in the config entry
        events = h.baseline_events()
        assert len(events) == 1
        assert events[0].zone_id == DESK
        assert events[0].move_mu == desk.move_baseline.mu

    def test_rule_3_3_default_duration_is_300s(self) -> None:
        # 3.3: sized to the measured ~2.5 s empty cadence — 120 s yields
        # only ~48 fresh observations, below stat_min_rows.
        h = Harness()
        h.submit(RecordBaseline(DESK))
        assert h.deadlines[timers.baseline_end(DESK)] == pytest.approx(300.0)

    def test_rule_3_3_empty_window_keeps_old_floor(self) -> None:
        h = Harness()
        h.submit(RecordBaseline(DESK, duration=10.0))
        h.run(15)  # no frames at all during the window
        desk = h.zone(DESK)
        assert desk.move_baseline.mu == pytest.approx(MU)
        assert h.persist_count == 0
        # 3.3 observability: even a void window reports its outcome —
        # nothing materialized, so the commit is a failure, not a success.
        (event,) = h.baseline_events()
        assert event.success is False
        assert event.frame_count == 0

    def test_rule_3_3_unknown_zone_ignored(self) -> None:
        h = Harness()
        plan = h.submit(RecordBaseline("nope", duration=10.0))
        assert plan.timer_starts == []


class TestRule33TransactionalCalibration:
    """Round-5 review regressions (rule 3.3): a window computes candidate
    + coverage before mutating anything, commits atomically, and must
    never look successful when a required channel was not calibrated."""

    @staticmethod
    def drive(h: Harness, duration: float | None, *, seed: int = 1) -> None:
        """The measured production shape: one real sensor frame every
        ~2.5 s (quantized N(20, 5) still energy), move energy quiescent,
        and an independent one-second tick clock."""
        import random

        rng = random.Random(seed)
        h.submit(RecordBaseline(DESK) if duration is None else RecordBaseline(DESK, duration))
        zone = h.zone(DESK)
        still_val, next_still = 20.0, 0.0
        start = h.now
        while zone.recording is not None:
            fresh_still = h.now - start + 1e-9 >= next_still
            if fresh_still:
                still_val = float(max(0, min(100, round(rng.gauss(20, 5)))))
                next_still += 2.5
            if fresh_still:
                h.send_frame(
                    KONTOR,
                    move_e=3.0,
                    still_e=still_val,
                    fresh_move=False,
                    fresh_still=True,
                    fresh_move_energy=False,
                )
            h.step_to(h.now + 1.0)

    def test_short_window_at_measured_cadence_rejects_atomically(self) -> None:
        """The round-5 blocker: 120 s at the 2.5 s cadence yields ~48 fresh
        still observations — the still path is rejected, and the rejection
        must preserve the *entire* previous calibration, persist nothing
        and report failure (not silently clear stat_cal and claim success)."""
        h = Harness()
        zone = h.zone(DESK)
        good = {
            "move_agg": StatBaseline(0.4, 0.6, 0.0, 1.2),
            "still_agg": StatBaseline(0.4, 0.6, 0.0, 1.4),
        }
        zone.stat_cal = dict(good)
        self.drive(h, 120.0)
        assert zone.recording is None
        (event,) = h.baseline_events()
        assert event.success is False
        cov = event.coverage
        assert cov["still_agg"].status is Coverage.REJECTED
        assert cov["still_agg"].fresh < 60  # ~48 at the measured cadence
        assert "need 60" in cov["still_agg"].reason
        # Neither the changing still channel nor the quiescent move channel
        # has enough real sensor observations in 120 seconds.
        assert cov["move_agg"].status is Coverage.REJECTED
        assert cov["move_gate"].status is Coverage.NO_DATA
        # Atomic: nothing half-applied, nothing lost, nothing persisted.
        assert zone.move_baseline.mu == pytest.approx(MU)
        assert zone.still_baseline.mu == pytest.approx(MU)
        assert zone.stat_cal == good
        assert h.persist_count == 0

    def test_default_window_at_measured_cadence_commits(self) -> None:
        """The 300 s default yields ~120 fresh still observations: the
        window commits, the quiescent move channel is reported as such,
        and only the committed window persists."""
        h = Harness()
        zone = h.zone(DESK)
        self.drive(h, None)  # the default baseline_duration
        (event,) = h.baseline_events()
        assert event.success is True
        cov = event.coverage
        assert cov["still_agg"].status is Coverage.CALIBRATED
        assert cov["still_agg"].fresh >= 60
        assert cov["move_agg"].status is Coverage.QUIESCENT  # reported as such
        assert cov["move_gate"].status is Coverage.NO_DATA
        assert zone.still_baseline.mu == pytest.approx(0.20, abs=0.02)
        assert zone.still_baseline.sigma >= 0.05  # 3.1: UCB covers the truth
        assert zone.move_baseline.mu == pytest.approx(0.03)
        assert zone.move_baseline.sigma == pytest.approx(0.02)  # sigma_min
        assert "still_agg" in zone.stat_cal  # empirical statistic (3.7)
        cal = zone.stat_cal["still_agg"]
        assert cal.decorrelation_seconds is not None
        assert cal.decorrelation_seconds > cal.tau  # ~2.5 s observation cadence
        assert "move_agg" not in zone.stat_cal  # quiescent: analytic fallback
        assert h.persist_count == 1

    def test_tick_only_cache_cannot_commit_as_quiescent(self) -> None:
        """A cached value plus ticks is not a measured plateau. Round 5
        incorrectly committed this as QUIESCENT and sharpened both scales
        to sigma_min despite zero post-command sensor observations."""
        h = Harness()
        h.send_frame(KONTOR, move_e=20.0, still_e=30.0)
        zone = h.zone(DESK)
        old_move = (zone.move_baseline.mu, zone.move_baseline.sigma)
        old_still = (zone.still_baseline.mu, zone.still_baseline.sigma)
        old_stats = dict(zone.stat_cal)

        h.submit(RecordBaseline(DESK, duration=70.0))
        h.run(70.0)  # ticks only — no post-command SensorFrame

        (event,) = h.baseline_events()
        assert event.success is False
        assert event.coverage["move_agg"].status is Coverage.REJECTED
        assert event.coverage["still_agg"].status is Coverage.REJECTED
        assert event.coverage["move_agg"].fresh == 0
        assert event.coverage["move_agg"].observed == 0
        assert (zone.move_baseline.mu, zone.move_baseline.sigma) == old_move
        assert (zone.still_baseline.mu, zone.still_baseline.sigma) == old_still
        assert zone.stat_cal == old_stats
        assert h.persist_count == 0

    def test_front_loaded_observations_then_silence_reject(self) -> None:
        h = Harness()
        h.submit(RecordBaseline(DESK, duration=300.0))
        for i in range(60):
            h.send_frame(KONTOR, move_e=float(i), still_e=float(i))
            h.step_to(h.now + 1.0)
        h.run(240.0)

        (event,) = h.baseline_events()
        assert event.success is False
        assert "maximum sensor-observation gap" in event.coverage["move_agg"].reason
        assert h.persist_count == 0

    def test_sensor_unavailable_at_close_rejects(self) -> None:
        h = Harness()
        h.submit(RecordBaseline(DESK, duration=70.0))
        for i in range(60):
            h.send_frame(KONTOR, move_e=float(i), still_e=float(i))
            h.step_to(h.now + 1.0)
        h.submit(SensorAvailability(KONTOR, available=False))
        h.run(10.0)

        (event,) = h.baseline_events()
        assert event.success is False
        assert event.coverage["move_agg"].reason == "sensor unavailable when calibration closed"

    def test_same_value_sensor_observations_certify_quiescence(self) -> None:
        """Real same-value publications remain useful: the sensor-wide
        observation clock distinguishes them from a tick-held cache."""
        h = Harness()
        h.submit(RecordBaseline(DESK, duration=70.0))
        for _ in range(70):
            h.send_frame(
                KONTOR,
                move_e=20.0,
                still_e=30.0,
                fresh_move=False,
                fresh_still=False,
                fresh_move_energy=False,
            )
            h.step_to(h.now + 1.0)

        (event,) = h.baseline_events()
        assert event.success is True
        assert event.coverage["move_agg"].status is Coverage.QUIESCENT
        assert event.coverage["move_agg"].fresh == 0
        assert event.coverage["move_agg"].observed >= 60

    def test_disabled_calibration_still_emits_control_outcome(self) -> None:
        """Calibration persistence is a control-plane action while disabled;
        its matching outcome must remain observable too."""
        h = Harness()
        h.submit(SetEnabled(False))
        h.submit(RecordBaseline(DESK, duration=70.0))
        for _ in range(70):
            h.send_frame(KONTOR, move_e=20.0, still_e=30.0)
            h.step_to(h.now + 1.0)

        assert h.persist_count == 1
        (event,) = h.baseline_events()
        assert event.success is True
        assert h.state.enabled is False


class TestRule34BackgroundAdaptation:
    def make(self) -> Harness:
        config = make_config(t_background=60.0, tau_background=600.0)
        return Harness(config, make_snapshot(config))

    def test_rule_3_4_floor_follows_downward_drift_when_quiet(self) -> None:
        h = self.make()
        h.run(61)  # posterior below p_background since start: now eligible
        for _ in range(100):
            h.send_frame(KONTOR, move_e=0, still_e=0)
            h.step_to(h.now + 1.0)
        desk = h.zone(DESK)
        assert desk.move_baseline.mu < MU - 0.005  # drifted toward observed 0
        assert desk.still_baseline.mu < MU - 0.005
        assert desk.move_baseline.sigma >= 0.02  # 3.1 floor holds

    def test_rule_3_4_cannot_erode_calibrated_scale_confidence_bound(self) -> None:
        config = make_config(t_background=0.0, tau_background=1.0)
        h = Harness(config, make_snapshot(config, sigma=0.08))
        desk = h.zone(DESK)
        statistic = StatBaseline(0.4, 0.6)
        desk.stat_cal["move_agg"] = statistic

        # Lower point variance may move the mean, but cannot sharpen the
        # conservative window-fitted scale paired with this statistic.
        for _ in range(20):
            h.send_frame(KONTOR, move_e=5.0, still_e=5.0)
            h.step_to(h.now + 1.0)
        assert desk.move_baseline.sigma == pytest.approx(0.08)
        assert desk.stat_cal["move_agg"] is statistic

        # A larger empty deviation may only make the scale more conservative.
        h.send_frame(KONTOR, move_e=20.0, still_e=5.0)
        h.step_to(h.now + 1.0)
        assert desk.move_baseline.sigma >= 0.08
        assert not desk.occupied

    def test_rule_3_4_no_adaptation_before_t_background(self) -> None:
        h = self.make()
        h.run(30)  # quiet since t=0, but not for t_background yet
        for _ in range(20):  # frames end at t=50 < 60: never eligible
            h.send_frame(KONTOR, move_e=0, still_e=0)
            h.step_to(h.now + 1.0)
        assert h.zone(DESK).move_baseline.mu == MU

    def test_rule_3_4_adaptation_freezes_the_moment_posterior_rises(self) -> None:
        h = self.make()
        h.run(61)
        h.occupy(KONTOR)  # posterior jumps to p_attack
        desk = h.zone(DESK)
        assert desk.below_since is None  # clock reset immediately
        mu_before = desk.move_baseline.mu
        for _ in range(30):
            h.send_frame(KONTOR, move_d=100, move_e=35, moving=True)
            h.step_to(h.now + 1.0)
        assert desk.move_baseline.mu == mu_before  # frozen while elevated

    def test_rule_3_4_elevated_sibling_zone_freezes_the_whole_sensor(self) -> None:
        # Energies are per sensor: while the desk sees a person, the door
        # must not learn those energies as background noise.
        h = self.make()
        h.run(61)  # both kontor zones eligible
        h.occupy(KONTOR, distance=100)  # desk elevated, door still empty
        door_mu = h.zone(DOOR).move_baseline.mu
        for _ in range(30):
            h.send_frame(KONTOR, move_d=100, move_e=35, moving=True)
            h.step_to(h.now + 1.0)
        assert h.zone(DOOR).move_baseline.mu == door_mu
