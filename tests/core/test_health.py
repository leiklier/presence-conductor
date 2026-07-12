"""Rule 1.3: staleness, availability, UNKNOWN health, recovery."""

from __future__ import annotations

from custom_components.presence_conductor.core import timers
from custom_components.presence_conductor.core.events import SensorAvailability
from custom_components.presence_conductor.core.model import Health

from .harness import BORD, DESK, DOOR, KONTOR, SOFA, SOFAKROK, Harness

STRONG_STILL = {"still_d": 100.0, "still_e": 35.0, "still": True}


class TestRule13Staleness:
    def test_rule_1_3_silence_while_occupied_goes_unknown(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)  # frame at t=0 arms the watchdog
        h.run(35)  # no further frames; watchdog fires at t=30
        zone = h.zone(SOFA)
        assert zone.health is Health.UNKNOWN
        assert zone.occupied  # outputs hold their last state
        assert h.room("stue").occupied is False  # 6.3: excluded; bord is empty

    def test_rule_1_3_outputs_hold_while_unknown(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        h.run(35)
        zone = h.zone(SOFA)
        held = (zone.lam, zone.dwell_seconds, zone.activity, zone.motion)
        h.run(20)
        assert (zone.lam, zone.dwell_seconds, zone.activity, zone.motion) == held

    def test_rule_1_3_recovery_is_immediate_on_the_next_frame(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        h.run(35)
        assert h.zone(SOFA).health is Health.UNKNOWN
        h.send_frame(SOFAKROK, **STRONG_STILL)
        assert h.zone(SOFA).health is Health.OK
        assert h.room("stue").occupied is True  # back in fusion right away

    def test_rule_1_3_silence_while_empty_stays_healthy(self) -> None:
        h = Harness()
        h.run(90)  # nobody home, sensors deduplicate into silence
        for zone_id in (DESK, DOOR, SOFA, BORD):
            assert h.zone(zone_id).health is Health.OK
        assert h.room("stue").occupied is False

    def test_rule_1_3_watchdog_restarts_on_every_frame(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        for _ in range(40):  # a frame every second: never stale
            h.send_frame(SOFAKROK, **STRONG_STILL)
            h.step_to(h.now + 1.0)
        assert h.zone(SOFA).health is Health.OK


class TestRule13Availability:
    def test_rule_1_3_unavailable_marks_unknown_immediately(self) -> None:
        h = Harness()
        h.submit(SensorAvailability(KONTOR, available=False))
        assert h.zone(DESK).health is Health.UNKNOWN
        assert h.zone(DOOR).health is Health.UNKNOWN
        assert timers.sensor_stale(KONTOR) not in h.deadlines  # watchdog off

    def test_rule_1_3_available_alone_does_not_recover(self) -> None:
        h = Harness()
        h.submit(SensorAvailability(KONTOR, available=False))
        h.submit(SensorAvailability(KONTOR, available=True))
        assert h.zone(DESK).health is Health.UNKNOWN  # no data yet
        h.send_frame(KONTOR, move_e=5, still_e=5)
        assert h.zone(DESK).health is Health.OK  # the frame recovers (1.3)

    def test_rule_1_3_unknown_sensor_availability_ignored(self) -> None:
        h = Harness()
        h.submit(SensorAvailability("nope", available=False))  # no crash
