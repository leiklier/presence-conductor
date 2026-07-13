"""Rule 3: noise floors, evidence scores, calibration, background drift."""

from __future__ import annotations

import pytest

from custom_components.presence_conductor.core import timers
from custom_components.presence_conductor.core.belief import advance
from custom_components.presence_conductor.core.events import RecordBaseline
from custom_components.presence_conductor.core.evidence import robust_stats

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

    def test_rule_3_3_default_duration_is_120s(self) -> None:
        h = Harness()
        h.submit(RecordBaseline(DESK))
        assert h.deadlines[timers.baseline_end(DESK)] == pytest.approx(120.0)

    def test_rule_3_3_empty_window_keeps_old_floor(self) -> None:
        h = Harness()
        h.submit(RecordBaseline(DESK, duration=10.0))
        h.run(15)  # no frames at all during the window
        desk = h.zone(DESK)
        assert desk.move_baseline.mu == pytest.approx(MU)
        assert h.persist_count == 0
        assert h.baseline_events() == []

    def test_rule_3_3_unknown_zone_ignored(self) -> None:
        h = Harness()
        plan = h.submit(RecordBaseline("nope", duration=10.0))
        assert plan.timer_starts == []


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
