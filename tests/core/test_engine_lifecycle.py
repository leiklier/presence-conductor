"""Rule 7: restart adoption, the enabled switch, determinism."""

from __future__ import annotations

import pytest

from custom_components.presence_conductor.core import timers
from custom_components.presence_conductor.core.events import (
    RecordBaseline,
    SensorAvailability,
    SetEnabled,
)
from custom_components.presence_conductor.core.model import Activity

from .harness import (
    BORD,
    DESK,
    DOOR,
    KONTOR,
    SOFA,
    SOFAKROK,
    SPISEBORD,
    Harness,
    frame,
    make_config,
    make_snapshot,
)

STRONG_MOVE = {"move_d": 100.0, "move_e": 35.0, "moving": True}


class TestRule71Seeding:
    def test_rule_7_1_posteriors_start_at_the_prior(self) -> None:
        h = Harness()
        for zone_id in (DESK, DOOR, SOFA, BORD):
            zone = h.zone(zone_id)
            assert zone.lam == pytest.approx(h.engine.lam_prior)
            assert not zone.occupied
            assert zone.activity is Activity.EMPTY
        assert h.state.anyone_home is False

    def test_rule_7_1_gated_target_seeds_at_theta_on(self) -> None:
        config = make_config()
        snapshot = make_snapshot(
            config,
            frames={SOFAKROK: frame(SOFAKROK, still_d=100, still_e=40, still=True)},
        )
        h = Harness(config, snapshot)
        zone = h.zone(SOFA)
        assert zone.lam == pytest.approx(h.engine.lam_on)  # not p_attack: 7.1
        assert zone.occupied
        assert zone.activity is Activity.PASSING
        assert h.room("stue").occupied is True
        assert h.state.anyone_home is True

    def test_rule_7_1_ungated_target_stays_at_the_prior(self) -> None:
        config = make_config()
        snapshot = make_snapshot(
            config,
            frames={SOFAKROK: frame(SOFAKROK, still_d=400, still_e=40, still=True)},
        )
        h = Harness(config, snapshot)
        assert h.zone(SOFA).lam == pytest.approx(h.engine.lam_prior)
        assert not h.zone(SOFA).occupied

    def test_rule_7_1_flag_without_distance_seeds_the_fallback_zone(self) -> None:
        config = make_config()
        snapshot = make_snapshot(
            config,
            frames={KONTOR: frame(KONTOR, move_e=40, moving=True)},  # 2.3 at seed
        )
        h = Harness(config, snapshot)
        assert h.zone(DESK).occupied
        assert h.zone(DESK).lam == pytest.approx(h.engine.lam_on)
        assert not h.zone(DOOR).occupied

    def test_rule_7_1_unavailable_sensor_seeds_unknown(self) -> None:
        config = make_config()
        snapshot = make_snapshot(config, available={KONTOR: False})
        h = Harness(config, snapshot)
        assert h.room("kontor").occupied is None  # 6.3
        assert timers.sensor_stale(KONTOR) not in h.deadlines


class TestRule72Enabled:
    def test_rule_7_2_disabled_suppresses_outputs_and_events(self) -> None:
        h = Harness()
        plan = h.submit(SetEnabled(False))
        assert plan.suppress_outputs
        # A full walk-through while disabled: state updates, nothing emitted.
        h.occupy(SOFAKROK)
        assert h.zone(SOFA).occupied  # 7.2: the engine keeps updating
        h.sustain_quiet(SOFAKROK, 25)
        assert not h.zone(SOFA).occupied
        assert h.pass_bys() == []  # 7.2: emits no events

    def test_rule_7_2_reenable_is_warm(self) -> None:
        h = Harness()
        h.submit(SetEnabled(False))
        h.occupy(SOFAKROK)
        h.sustain(SOFAKROK, 5, **STRONG_MOVE)
        plan = h.submit(SetEnabled(True))
        assert not plan.suppress_outputs
        assert h.zone(SOFA).occupied  # adopted instantly, no re-learning
        assert h.state.anyone_home is True

    def test_rule_7_2_snapshot_seeds_the_switch(self) -> None:
        config = make_config()
        h = Harness(config, make_snapshot(config, enabled=False))
        plan = h.occupy(SOFAKROK)
        assert plan.suppress_outputs
        assert h.zone(SOFA).occupied  # still estimating underneath

    def test_rule_7_2_events_flow_again_after_reenable(self) -> None:
        h = Harness()
        h.submit(SetEnabled(False))
        h.submit(SetEnabled(True))
        h.occupy(SOFAKROK)
        h.sustain(SOFAKROK, 5, **STRONG_MOVE)
        h.sustain_quiet(SOFAKROK, 25)
        assert len(h.pass_bys()) == 1


class TestRule73Determinism:
    def _scenario(self, h: Harness) -> list[tuple]:
        prints: list[tuple] = [h.fingerprint()]
        h.occupy(KONTOR)
        prints.append(h.fingerprint())
        h.sustain(KONTOR, 10, **STRONG_MOVE)
        prints.append(h.fingerprint())
        h.submit(RecordBaseline(DOOR, duration=15.0))
        h.sustain_quiet(KONTOR, 20)  # baseline window closes along the way
        prints.append(h.fingerprint())
        h.submit(SensorAvailability(SPISEBORD, available=False))
        h.run(40)  # kontor goes stale while draining
        prints.append(h.fingerprint())
        h.submit(SetEnabled(False))
        h.occupy(SOFAKROK)
        h.submit(SetEnabled(True))
        h.run(10)
        prints.append(h.fingerprint())
        return prints

    def test_rule_7_3_same_sequence_same_trajectory(self) -> None:
        assert self._scenario(Harness()) == self._scenario(Harness())


class TestRobustness:
    def test_unknown_sensor_frame_is_ignored(self) -> None:
        h = Harness()
        plan = h.send_frame("nope", move_d=100, move_e=35, moving=True)
        assert plan.timer_starts == []

    def test_stale_timer_key_is_ignored_when_not_pending(self) -> None:
        h = Harness()
        before = h.fingerprint()
        h.fire_timer(timers.motion_off(DESK))  # never started
        assert h.fingerprint() == before

    def test_unknown_timer_prefix_is_ignored(self) -> None:
        h = Harness()
        h.engine._pending_timers.add("mystery:desk")
        h.fire_timer("mystery:desk")  # dispatch falls through, no crash
