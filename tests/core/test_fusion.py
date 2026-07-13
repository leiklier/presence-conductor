"""Rule 6: room fusion and home presence."""

from __future__ import annotations

import pytest

from custom_components.presence_conductor.core.events import SensorAvailability
from custom_components.presence_conductor.core.model import Activity

from .harness import BORD, KONTOR, SOFA, SOFAKROK, SPISEBORD, Harness

STRONG_STILL = {"still_d": 100.0, "still_e": 35.0, "still": True, "move_e": 5.0}


class TestRule61RoomOccupancy:
    def test_rule_6_1_any_healthy_member_occupies_the_room(self) -> None:
        h = Harness()
        assert h.room("stue").occupied is False
        h.occupy(SOFAKROK)
        assert h.room("stue").occupied is True
        assert not h.zone(BORD).occupied  # the other sensor stays out of it

    def test_rule_6_1_probability_is_noisy_or(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        h.occupy(SPISEBORD)
        p1 = h.zone(SOFA).probability
        p2 = h.zone(BORD).probability
        assert h.room("stue").probability == pytest.approx(1 - (1 - p1) * (1 - p2))


class TestRule62RoomActivity:
    def test_rule_6_2_activity_is_max_severity_and_settled_flag(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        h.sustain(SOFAKROK, 35, **STRONG_STILL)  # sofa: SETTLED
        h.occupy(SPISEBORD)  # bord: PASSING
        assert h.zone(SOFA).activity is Activity.SETTLED
        assert h.zone(BORD).activity is Activity.PASSING
        assert h.room("stue").activity is Activity.SETTLED  # settled > passing
        assert h.room("stue").settled is True

    def test_rule_6_2_settled_false_when_no_member_settled(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        assert h.room("stue").settled is False

    def test_rule_6_2_motion_follows_any_member_zone(self) -> None:
        h = Harness()
        assert h.room("stue").motion is False
        h.occupy(SOFAKROK)  # strong gated move: motion on (4.4)
        assert h.zone(SOFA).motion is True
        assert not h.zone(BORD).motion  # the other member stays quiet
        assert h.room("stue").motion is True

    def test_rule_6_2_motion_releases_with_the_member_channel(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        assert h.room("stue").motion is True
        h.sustain_quiet(SOFAKROK, 10)  # motion_hold (5 s) expires within this
        assert h.zone(SOFA).motion is False
        assert h.room("stue").motion is False


class TestRule63HealthExclusion:
    def test_rule_6_3_unknown_zone_is_excluded_from_fusion(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        assert h.room("stue").occupied is True
        h.submit(SensorAvailability(SOFAKROK, available=False))
        assert h.zone(SOFA).occupied  # 1.3: output holds its last state
        assert h.room("stue").occupied is False  # healthy member (bord) is empty

    def test_rule_6_3_all_members_unknown_publishes_unknown(self) -> None:
        h = Harness()
        h.submit(SensorAvailability(KONTOR, available=False))
        room = h.room("kontor")
        assert room.occupied is None
        assert room.motion is None
        assert room.probability is None
        assert room.activity is None
        assert room.settled is None


class TestRule64Monotone:
    def test_rule_6_4_a_zone_never_vetoes_another(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        p_sofa = h.zone(SOFA).probability
        h.sustain_quiet(SPISEBORD, 10)  # strong absence at the other sensor
        assert h.room("stue").occupied is True
        assert h.room("stue").probability >= h.zone(SOFA).probability * 0.99
        assert h.zone(SOFA).probability == pytest.approx(p_sofa, abs=0.3)


class TestRule65HomePresence:
    def test_rule_6_5_anyone_home_rises_immediately_with_any_zone(self) -> None:
        h = Harness()
        assert h.state.anyone_home is False
        h.occupy(SOFAKROK)
        assert h.state.anyone_home is True  # same submit, no tick needed
        assert h.state.home_probability >= 0.8

    def test_rule_6_5_home_decays_much_slower_than_zones(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        h.sustain_quiet(SOFAKROK, 25)  # zone drains in ~10 s
        assert not h.zone(SOFA).occupied
        assert h.state.anyone_home is True  # tau_home = 20 min
        h.run(300)
        assert h.state.anyone_home is True  # still holding after 5 min
        h.run(1300)
        assert h.state.anyone_home is False  # ~20+ min later: away

    def test_rule_6_5_home_holds_while_any_zone_is_occupied(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        h.sustain(SOFAKROK, 120, **STRONG_STILL)
        assert h.state.anyone_home is True
        assert h.state.home_probability == pytest.approx(0.999, abs=0.002)

    def test_rule_6_5_all_zones_unhealthy_publishes_unknown(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        for sensor in (KONTOR, SOFAKROK, SPISEBORD):
            h.submit(SensorAvailability(sensor, available=False))
        assert h.state.anyone_home is None
        assert h.state.home_probability is None
        h.occupy(SOFAKROK)  # a frame recovers health immediately (1.3)
        assert h.state.anyone_home is True
