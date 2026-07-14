"""Rule 1.3: staleness, availability, UNKNOWN health, recovery."""

from __future__ import annotations

from custom_components.presence_conductor.core import timers
from custom_components.presence_conductor.core.events import SensorAvailability
from custom_components.presence_conductor.core.model import Health, StatBaseline

from .harness import BORD, DESK, DOOR, KONTOR, SOFA, SOFAKROK, Harness, frame

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

    def test_cached_frame_does_not_rearm_or_recover_health(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        sensor = h.state.sensors[SOFAKROK]
        deadline = h.deadlines[timers.sensor_stale(SOFAKROK)]

        # Same observation epochs model an attribute-only/invalid HA event:
        # the adapter may submit its cache, but this is not sensor liveness.
        h.submit(
            frame(
                SOFAKROK,
                move_obs=sensor.move_obs,
                still_obs=sensor.still_obs,
                frame_obs=sensor.frame_obs,
                move_energy_obs=sensor.move_energy_obs,
            ),
            at=h.now + 10.0,
        )
        assert h.deadlines[timers.sensor_stale(SOFAKROK)] == deadline
        h.step_to(deadline)
        assert h.zone(SOFA).health is Health.UNKNOWN

        h.submit(
            frame(
                SOFAKROK,
                move_d=100,
                move_e=35,
                moving=True,
                move_obs=sensor.move_obs,
                still_obs=sensor.still_obs,
                frame_obs=sensor.frame_obs,
                move_energy_obs=sensor.move_energy_obs,
            ),
            at=h.now + 1.0,
        )
        assert h.zone(SOFA).health is Health.UNKNOWN
        assert not h.zone(SOFA).motion
        assert timers.motion_off(SOFA) not in h.deadlines

        # A genuine quiet observation recovers health without leaking a
        # motion hold created from the cached strong frame above.
        h.send_frame(SOFAKROK, move_e=5, still_e=5, at=h.now + 1.0)
        assert h.zone(SOFA).health is Health.OK
        assert not h.zone(SOFA).motion

    def test_empty_staleness_breaks_long_attack_chain(self) -> None:
        h = Harness()
        zone = h.zone(DESK)
        zone.stat_cal["move_agg"] = StatBaseline(0.4, 0.6, decorrelation_seconds=62.5)
        h.send_frame(KONTOR, move_d=100, move_e=35, moving=True)
        assert zone.attack_count == 1

        # Empty sensors remain healthy under deduplicated silence, but each
        # watchdog firing still proves a blind interval and clears attack.
        h.step_to(62.5)
        assert zone.attack_count == 0
        h.send_frame(KONTOR, move_d=100, move_e=36, moving=True, at=62.5)
        assert zone.attack_count == 1
        assert not zone.occupied


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

    def test_unavailability_breaks_attack_confirmation_chain(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=100, move_e=35, moving=True)
        zone = h.zone(DESK)
        assert zone.attack_count == 1

        h.submit(SensorAvailability(KONTOR, available=False), at=h.now + 0.1)
        assert zone.attack_count == 0
        h.submit(SensorAvailability(KONTOR, available=True), at=h.now + 0.1)
        h.send_frame(KONTOR, move_d=100, move_e=36, moving=True, at=h.now + 0.2)

        assert zone.attack_count == 1
        assert not zone.occupied

    def test_rule_1_3_unknown_sensor_availability_ignored(self) -> None:
        h = Harness()
        h.submit(SensorAvailability("nope", available=False))  # no crash
