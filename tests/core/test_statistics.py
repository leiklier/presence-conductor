"""Statistical regression tests (spec §3, rules 3.2/3.7/4.1/2.7).

These encode the failure modes of the pre-3.7 evidence model, found in an
external statistical review: one-sided anomaly scores have positive
expectation under symmetric empty noise (``E[max(0, Z)] ~ 0.4``), a
max-over-gates statistic grows with the gate count, and tick-boundary
integration applied new evidence retroactively. Every test here fails on
that model.

Randomness lives in the *tests* (seeded, deterministic); the engine itself
never draws a random number (7.3).
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist

import pytest

from custom_components.presence_conductor.core import evidence
from custom_components.presence_conductor.core.events import RecordBaseline
from custom_components.presence_conductor.core.model import (
    ConductorConfig,
    GateBaselines,
    RoomConfig,
    SensorConfig,
    StatBaseline,
    ZoneBaselines,
    ZoneConfig,
)
from custom_components.presence_conductor.core.stats import (
    attack_threshold,
    clipped_mean,
    onesided_max_stats,
)

from .harness import (
    DESK,
    KONTOR,
    SOFA,
    SOFAKROK,
    Z_EMPTY,
    Harness,
    InitialSnapshot,
    Tunables,
    centered_of_raw,
    gate_tuple,
    make_config,
    make_snapshot,
)

#: Empty-room noise in raw 0-100 energy units: integer-quantized Gaussian,
#: matching how the LD2410 reports energies.
NOISE_MU = 5.0
NOISE_SIGMA = 2.0


def _noisy_energy(rng: random.Random) -> float:
    return float(max(0, min(100, round(rng.gauss(NOISE_MU, NOISE_SIGMA)))))


def _gate_zone_config(far_cm: float) -> ConductorConfig:
    """One sensor, one zone owning the gates that cover ``[0, far_cm]``."""
    return ConductorConfig(
        sensors=(SensorConfig(KONTOR, "Kontor"),),
        zones=(ZoneConfig("z", "Z", KONTOR, room_id="r", near_cm=0, far_cm=far_cm),),
        rooms=(RoomConfig("r", "R"),),
    )


def _noise_tuple(rng: random.Random, owned: tuple[int, ...]) -> tuple[float | None, ...]:
    return gate_tuple({index: _noisy_energy(rng) for index in owned}, fill=None)


@pytest.mark.parametrize("far_cm", [40.0, 150.0, 600.0])  # owns 1 / 3 / 9 gates
def test_calibrated_empty_gate_noise_never_occupies(far_cm: float) -> None:
    """The review's headline reproduction, inverted: seeded, quantized
    Gaussian gate noise on every owned gate — move and still — for 15
    minutes after a RecordBaseline over the same process. The pre-3.7 model
    went occupied in a median of 4 seconds (500/500 trials); the centered
    model must never leave empty, regardless of the owned-gate count."""
    config = _gate_zone_config(far_cm)
    h = Harness(config, InitialSnapshot())
    owned = h.engine.owned_gates["z"]
    assert len(owned) == {40.0: 1, 150.0: 3, 600.0: 9}[far_cm]
    rng = random.Random(4)

    # Calibrate floors + the 3.7 statistic over the real noise process.
    h.submit(RecordBaseline("z", duration=120.0))
    for _ in range(121):
        h.send_frame(
            KONTOR,
            gate_move=_noise_tuple(rng, owned),
            gate_still=_noise_tuple(rng, owned),
        )
        h.step_to(h.now + 1.0)
    zone = h.zone("z")
    assert zone.recording is None
    assert "move_gate" in zone.stat_cal  # 3.7: empirical statistic recorded

    # 15 minutes of the same empty noise.
    rate_sum = 0.0
    for second in range(900):
        h.send_frame(
            KONTOR,
            gate_move=_noise_tuple(rng, owned),
            gate_still=_noise_tuple(rng, owned),
        )
        rate_sum += evidence.evidence_rate(zone, config.tunables)
        h.step_to(h.now + 1.0)
        assert not zone.occupied, f"false occupancy at t={second} with m={len(owned)}"
    # §3: the expected evidence rate in a calibrated empty room is negative.
    assert rate_sum / 900 < 0.0


def test_analytic_fallback_empty_gate_noise_never_occupies() -> None:
    """Same process without RecordBaseline: exact floors, no empirical
    statistic — the analytic Gaussian fallback (3.7) must also hold the
    zone empty."""
    config = _gate_zone_config(150.0)
    floors = {
        index: GateBaselines(NOISE_MU / 100, NOISE_SIGMA / 100, NOISE_MU / 100, NOISE_SIGMA / 100)
        for index in range(9)
    }
    snapshot = InitialSnapshot(baselines={"z": ZoneBaselines(0.05, 0.02, 0.05, 0.02, gates=floors)})
    h = Harness(config, snapshot)
    owned = h.engine.owned_gates["z"]
    rng = random.Random(11)
    zone = h.zone("z")
    for second in range(900):
        h.send_frame(
            KONTOR,
            gate_move=_noise_tuple(rng, owned),
            gate_still=_noise_tuple(rng, owned),
        )
        h.step_to(h.now + 1.0)
        assert not zone.occupied, f"false occupancy at t={second}"


def test_ghost_gated_aggregate_noise_never_occupies() -> None:
    """Aggregate-path worst case: the radar false-flags a target inside the
    zone every second while both energies are pure floor noise. Exact
    floors, analytic centering — the zone must stay empty."""
    config = make_config()
    h = Harness(config, make_snapshot(config, mu=NOISE_MU / 100, sigma=NOISE_SIGMA / 100))
    rng = random.Random(7)
    desk = h.zone(DESK)
    for second in range(900):
        h.send_frame(
            KONTOR,
            move_d=100.0,
            still_d=100.0,
            move_e=_noisy_energy(rng),
            still_e=_noisy_energy(rng),
            moving=True,
            still=True,
        )
        h.step_to(h.now + 1.0)
        assert not desk.occupied, f"false occupancy at t={second}"


@pytest.mark.parametrize(
    ("seed", "far_cm"),
    [(13, 40.0), (14, 40.0), (17, 40.0), (21, 600.0)],
)
def test_review_seeds_stay_empty_for_an_hour(seed: int, far_cm: float) -> None:
    """Fixed-seed regressions from the external review's stress sweep: these
    seeds produced empirical calibrations whose underestimated scale (e.g.
    seed 13: s0 = 0.405 vs the analytic 0.584) inflated every later score
    and falsely occupied within the hour. The 3.7 shrinkage — analytic
    scale floor, minimum rows, clip-mean recentering — plus the analytic
    attack tail (4.2) must hold them empty."""
    config = _gate_zone_config(far_cm)
    h = Harness(config, InitialSnapshot())
    owned = h.engine.owned_gates["z"]
    rng = random.Random(seed)
    h.submit(RecordBaseline("z", duration=120.0))
    for _ in range(121):
        h.send_frame(
            KONTOR,
            gate_move=_noise_tuple(rng, owned),
            gate_still=_noise_tuple(rng, owned),
        )
        h.step_to(h.now + 1.0)
    zone = h.zone("z")
    reference_s0 = onesided_max_stats(len(owned))[1]
    assert zone.stat_cal["move_gate"].sigma >= reference_s0  # 3.7 floor
    for second in range(3600):
        h.send_frame(
            KONTOR,
            gate_move=_noise_tuple(rng, owned),
            gate_still=_noise_tuple(rng, owned),
        )
        h.step_to(h.now + 1.0)
        assert not zone.occupied, f"seed {seed}: false occupancy at t={second}"


def _noise_at(rng: random.Random, mu: float, sigma: float) -> float:
    return float(max(0, min(100, round(rng.gauss(mu, sigma)))))


def _tuple_at(
    rng: random.Random, owned: tuple[int, ...], mu: float, sigma: float
) -> tuple[float | None, ...]:
    return gate_tuple({i: _noise_at(rng, mu, sigma) for i in owned}, fill=None)


@pytest.mark.parametrize(
    ("seed", "far_cm", "mu", "sigma"),
    [(11, 600.0, 20.0, 5.0), (21, 600.0, 20.0, 5.0), (21, 600.0, 50.0, 10.0)],
)
def test_round3_seeds_with_uncalibrated_scale_noise(
    seed: int, far_cm: float, mu: float, sigma: float
) -> None:
    """Round-3 review regressions: at these noise scales sigma_min no
    longer masks per-gate scale underestimation, and the quantized-MAD
    point estimate fitted gates as low as 0.59x the true scale — voiding
    the analytic attack tail (seed 11: false attack 64 s in) and inflating
    the evidence walk (seed 21: crossings ~250 s in). The 3.1 UCB +
    half-quantum floor must hold them empty for the measured hour, after a
    60 s post-calibration burn-in."""
    config = _gate_zone_config(far_cm)
    h = Harness(config, InitialSnapshot())
    owned = h.engine.owned_gates["z"]
    rng = random.Random(seed)
    h.submit(RecordBaseline("z", duration=120.0))
    for _ in range(121):
        h.send_frame(
            KONTOR,
            gate_move=_tuple_at(rng, owned, mu, sigma),
            gate_still=_tuple_at(rng, owned, mu, sigma),
        )
        h.step_to(h.now + 1.0)
    zone = h.zone("z")
    # 3.1: the fitted per-gate scales must essentially cover the true scale
    # (the UCB is a 95% bound per fit; the old point estimates sat at
    # 0.59-0.89x, which is what voided the tails).
    for floor in zone.gate_move_baselines.values():
        assert floor.sigma >= 0.9 * sigma / 100.0
    for _ in range(60 + 3600):  # burn-in, then the measured hour
        h.send_frame(
            KONTOR,
            gate_move=_tuple_at(rng, owned, mu, sigma),
            gate_still=_tuple_at(rng, owned, mu, sigma),
        )
        h.step_to(h.now + 1.0)
        assert not zone.occupied, f"seed {seed}: false occupancy at t={h.now}"


def test_calibration_cannot_manufacture_occupancy() -> None:
    """Round-3 blocker 2: an asserted-empty room with background well
    above the default floors must not become occupied *by its own
    calibration* — previously the window scored frames against the old
    floors (occupied 11 s in, confidence 0.998 at close) and ratcheted
    anyone_home for ~30 minutes (3.3 lifecycle)."""
    config = _gate_zone_config(40.0)
    h = Harness(config, InitialSnapshot())
    owned = h.engine.owned_gates["z"]
    rng = random.Random(1)
    lam_home_before = h.state.lam_home
    h.submit(RecordBaseline("z", duration=120.0))
    zone = h.zone("z")
    for _ in range(121):
        h.send_frame(
            KONTOR,
            gate_move=_tuple_at(rng, owned, 20.0, 5.0),
            gate_still=_tuple_at(rng, owned, 20.0, 5.0),
        )
        h.step_to(h.now + 1.0)
        assert not zone.occupied  # 3.3: suspended, pinned at the prior
        assert not zone.motion
        if zone.recording is not None:  # window still open: pinned
            assert zone.confidence == pytest.approx(0.02)
        assert h.room("r").occupied is False
        assert h.state.anyone_home is False
    assert h.pass_bys() == []  # no synthetic traversal either
    assert h.state.lam_home == pytest.approx(lam_home_before, abs=0.01)  # no ratchet
    assert zone.recording is None  # window closed
    assert zone.gate_move_baselines[0].mu == pytest.approx(0.20, abs=0.02)
    # And with the *correct* floors installed, the same background stays empty.
    for _ in range(300):
        h.send_frame(
            KONTOR,
            gate_move=_tuple_at(rng, owned, 20.0, 5.0),
            gate_still=_tuple_at(rng, owned, 20.0, 5.0),
        )
        h.step_to(h.now + 1.0)
        assert not zone.occupied


def test_calibration_voids_current_occupancy_without_pass_by() -> None:
    """Pressing record-baseline on an occupied zone publishes the operator's
    assertion — empty at the prior — with no pass_by (3.3)."""
    h = Harness()
    h.occupy(SOFAKROK)
    assert h.zone(SOFA).occupied
    h.submit(RecordBaseline(SOFA, duration=30.0))
    zone = h.zone(SOFA)
    assert not zone.occupied
    assert zone.confidence == pytest.approx(0.02)
    assert h.pass_bys() == []


class TestRule37Shrinkage:
    def test_empirical_scale_is_floored_by_the_analytic_reference(self) -> None:
        """A near-constant window would fit a tiny s0; the stored scale must
        never fall below the analytic reference for the path (3.7)."""
        config = _gate_zone_config(150.0)  # owns 3 gates
        h = Harness(config, InitialSnapshot())
        owned = h.engine.owned_gates["z"]
        h.submit(RecordBaseline("z", duration=120.0))
        for _ in range(121):
            h.send_frame(
                KONTOR,
                gate_move=gate_tuple(dict.fromkeys(owned, 5.0), fill=None),
                gate_still=gate_tuple(dict.fromkeys(owned, 5.0), fill=None),
            )
            h.step_to(h.now + 1.0)
        cal = h.zone("z").stat_cal["move_gate"]
        assert cal.sigma == pytest.approx(onesided_max_stats(3)[1])

    def test_short_windows_replace_nothing(self) -> None:
        """Fewer than stat_min_rows rows cannot certify a scale (3.1, 3.3):
        neither the floors nor the statistic calibration are replaced."""
        config = _gate_zone_config(150.0)
        h = Harness(config, InitialSnapshot())
        owned = h.engine.owned_gates["z"]
        rng = random.Random(3)
        h.submit(RecordBaseline("z", duration=30.0))
        for _ in range(31):
            h.send_frame(KONTOR, gate_move=_noise_tuple(rng, owned))
            h.step_to(h.now + 1.0)
        zone = h.zone("z")
        assert zone.gate_move_baselines == {}  # floors kept (3.3 coverage)
        assert zone.stat_cal == {}  # statistic kept too (3.7)


class TestRule42AttackTail:
    def test_thresholds_equalize_the_tail_across_gate_counts(self) -> None:
        """P_H0(S >= threshold) is attack_tail for every m (4.2): the
        candidate rate no longer depends 10x on the owned-gate count."""
        tail = 1e-4
        for m in (1, 3, 9):
            threshold = attack_threshold(m, tail)
            p = 1.0 - NormalDist().cdf(threshold) ** m
            assert p == pytest.approx(tail, rel=1e-6), f"m={m}"

    def test_clipped_mean_recenters_exactly(self) -> None:
        """Monte Carlo: after subtracting clipped_mean, the clamped centered
        score of a 3-gate max is mean-zero under H0 (3.2)."""
        rng = random.Random(5)
        m0, s0 = onesided_max_stats(3)
        c0 = clipped_mean(3, 1.0, 6.0)
        assert c0 > 0.03  # the asymmetric-clamp bias is real for m = 3
        total = 0.0
        n = 200_000
        for _ in range(n):
            s_raw = max(0.0, max(rng.gauss(0, 1) for _ in range(3)))
            total += min(6.0, max(-1.0, (s_raw - m0) / s0)) - c0
        assert total / n == pytest.approx(0.0, abs=0.01)


class TestRule41Chronology:
    def test_new_evidence_is_never_applied_retroactively(self) -> None:
        """The review's exact reproduction: last tick at t=0, a strong still
        frame at t=9.9, tick at t=10. The old model integrated the new
        evidence over the full 10 s (confidence 0.02 -> 0.89); the evidence
        existed for 0.1 s and must count for 0.1 s."""
        h = Harness()
        h.send_frame(SOFAKROK, still_d=100, still_e=35, still=True, at=9.9)
        h.tick(at=10.0)
        sofa = h.zone(SOFA)
        assert sofa.confidence < 0.05
        assert not sofa.occupied

    def test_the_same_evidence_integrates_forward(self) -> None:
        """The mirror image: the frame arriving just after a tick counts
        for the full following interval."""
        h = Harness()
        h.tick(at=10.0)
        h.send_frame(SOFAKROK, still_d=100, still_e=35, still=True, at=10.1)
        h.tick(at=20.0)
        assert h.zone(SOFA).occupied

    @staticmethod
    def _drive(tick_times: list[float]) -> float:
        """A fixed frame history under an arbitrary tick schedule; returns
        the final belief."""
        h = Harness()
        frames = [
            (1.5, {"still_d": 100, "still_e": 35, "still": True}),
            (13.7, {"move_e": 5, "still_e": 5}),  # back at the floor
        ]
        events = sorted(
            [(t, kw) for t, kw in frames] + [(t, None) for t in tick_times if t <= 40.0],
            key=lambda event: event[0],
        )
        for t, kw in events:
            while True:  # fire due timers at their exact deadlines
                due = [(when, key) for key, when in h.deadlines.items() if when <= t]
                if not due:
                    break
                when, key = min(due)
                h.fire_timer(key, at=max(h.now, when))
            if kw is None:
                h.tick(at=t)
            else:
                h.send_frame(SOFAKROK, at=t, **kw)
        h.tick(at=40.0)
        return h.zone(SOFA).lam

    def test_outputs_are_invariant_to_tick_cadence(self) -> None:
        """Rule 4.1: 0.1 s, 1 s and 10 s tick schedules produce the same
        belief for the same frame history."""
        fine = self._drive([round(0.1 * i, 10) for i in range(1, 400)])
        normal = self._drive([float(i) for i in range(1, 40)])
        coarse = self._drive([10.0, 20.0, 30.0])
        assert fine == pytest.approx(normal, rel=1e-9)
        assert coarse == pytest.approx(normal, rel=1e-9)

    def test_outputs_survive_a_scheduler_pause(self) -> None:
        """Rule 4.1: ticks silently stopping for 20 s and resuming changes
        nothing — the next event integrates the full gap exactly."""
        paused = self._drive([float(i) for i in range(1, 40) if not 15 <= i <= 35])
        normal = self._drive([float(i) for i in range(1, 40)])
        assert paused == pytest.approx(normal, rel=1e-9)


class TestRule27DistanceHold:
    def test_frozen_distance_gates_within_the_hold(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=100, move_e=5, moving=True)  # flag epoch
        # Flag off, distance frozen at 100: within distance_hold (30 s) the
        # energy margin still gates there (3.5 bridging).
        h.send_frame(KONTOR, move_d=100, move_e=35, at=10.0)
        assert h.zone(DESK).move_gated
        assert h.zone(DESK).z_move == pytest.approx(centered_of_raw(35))

    def test_frozen_distance_expires_after_the_hold(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=100, move_e=5, moving=True)  # flag epoch
        # 40 s later an energy blip arrives with the frozen distance still
        # cached: it must not be attributed to wherever someone last stood.
        h.send_frame(KONTOR, move_d=100, move_e=35, at=40.0)
        assert not h.zone(DESK).move_gated
        assert h.zone(DESK).z_move == pytest.approx(Z_EMPTY)


class TestRule37Statistic:
    def test_analytic_constants_match_the_closed_form(self) -> None:
        mean, std = onesided_max_stats(1)
        assert mean == pytest.approx(1 / math.sqrt(2 * math.pi), abs=1e-6)
        assert std == pytest.approx(math.sqrt(0.5 - 1 / (2 * math.pi)), abs=1e-6)

    def test_analytic_mean_grows_with_the_gate_count(self) -> None:
        means = [onesided_max_stats(m)[0] for m in range(1, 10)]
        assert means == sorted(means)  # multiple comparisons, quantified

    def test_analytic_values_match_monte_carlo(self) -> None:
        rng = random.Random(99)
        samples = [max(0.0, max(rng.gauss(0, 1) for _ in range(3))) for _ in range(200_000)]
        mean = sum(samples) / len(samples)
        var = sum((s - mean) ** 2 for s in samples) / len(samples)
        a_mean, a_std = onesided_max_stats(3)
        assert a_mean == pytest.approx(mean, abs=0.01)
        assert a_std == pytest.approx(math.sqrt(var), abs=0.01)

    def test_persisted_stats_seed_the_centering(self) -> None:
        config = make_config()
        baselines = {
            zone.zone_id: ZoneBaselines(
                0.05, 0.05, 0.05, 0.05, stats={"still_agg": StatBaseline(0.2, 0.5)}
            )
            for zone in config.zones
        }
        h = Harness(config, InitialSnapshot(baselines=baselines))
        h.send_frame(KONTOR, still_d=100, still_e=10, still=True)  # raw S = 1.0
        assert h.zone(DESK).z_still == pytest.approx((1.0 - 0.2) / 0.5)

    def test_record_baseline_produces_a_centered_statistic(self) -> None:
        """After calibration over a noisy window, the empty process scores
        an evidence rate whose mean is ~ -k_bias (3.2): the statistic is
        centered by construction."""
        config = make_config()
        h = Harness(config, make_snapshot(config))
        rng = random.Random(21)
        h.submit(RecordBaseline(DESK, duration=120.0))
        for _ in range(121):
            h.send_frame(
                KONTOR,
                move_d=100.0,
                still_d=100.0,
                move_e=_noisy_energy(rng),
                still_e=_noisy_energy(rng),
                moving=True,
                still=True,
            )
            h.step_to(h.now + 1.0)
        desk = h.zone(DESK)
        assert set(desk.stat_cal) >= {"move_agg", "still_agg"}
        rates = []
        for _ in range(600):
            h.send_frame(
                KONTOR,
                move_d=100.0,
                still_d=100.0,
                move_e=_noisy_energy(rng),
                still_e=_noisy_energy(rng),
                moving=True,
                still=True,
            )
            rates.append(evidence.evidence_rate(desk, config.tunables))
            h.step_to(h.now + 1.0)
        mean_rate = sum(rates) / len(rates)
        t = Tunables()
        assert mean_rate < 0.0
        assert mean_rate == pytest.approx(-t.k_bias, abs=0.35)
