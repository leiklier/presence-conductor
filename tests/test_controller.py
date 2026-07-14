"""Controller tests: frame coalescing, ticks, timers, persistence, suppression.

The engine is a scripted :class:`tests.fake_engine.FakeEngine`; the real
``ConductorEngine`` is exercised end-to-end in test_e2e.py. Time is driven
with the ``freezer`` fixture (freezegun patches ``time.monotonic``, which
both the controller's ``now`` values and HA's timer wheel run on) plus
``async_fire_time_changed``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_capture_events,
    async_fire_time_changed,
)

from custom_components.presence_conductor.config import baselines_from_options
from custom_components.presence_conductor.const import DOMAIN, GATE_COUNT
from custom_components.presence_conductor.core.engine import ConductorEngine
from custom_components.presence_conductor.core.events import (
    SensorAvailability,
    SensorFrame,
    Tick,
)
from custom_components.presence_conductor.core.model import (
    ChannelStats,
    GateBaselines,
    InitialSnapshot,
)
from custom_components.presence_conductor.core.plan import PassBy
from tests.fake_engine import FakeEngine

KONTOR = "kontor_radar"
SOFAKROK = "sofakrok_radar"

KONTOR_ENTITIES = {
    "move_energy": "sensor.kontor_move_energy",
    "still_energy": "sensor.kontor_still_energy",
    "moving_distance": "sensor.kontor_moving_distance",
    "still_distance": "sensor.kontor_still_distance",
    "target": "binary_sensor.kontor_target",
    "moving_target": "binary_sensor.kontor_moving_target",
    "still_target": "binary_sensor.kontor_still_target",
}
SOFAKROK_ENTITIES = {
    "move_energy": "sensor.sofakrok_move_energy",
    "still_energy": "sensor.sofakrok_still_energy",
    "moving_distance": "sensor.sofakrok_moving_distance",
    "still_distance": "sensor.sofakrok_still_distance",
    # Not consumed by the v1 estimator: changes must not produce frames.
    "detection_distance": "sensor.sofakrok_detection_distance",
    "target": "binary_sensor.sofakrok_target",
}

OPTIONS: dict[str, Any] = {
    "sensors": [
        {"sensor_id": KONTOR, "name": "Kontor radar", "entities": KONTOR_ENTITIES},
        {"sensor_id": SOFAKROK, "name": "Sofakrok radar", "entities": SOFAKROK_ENTITIES},
    ],
    "zones": [
        {
            "zone_id": "kontor_pult",
            "name": "Kontor pult",
            "sensor": KONTOR,
            "room": "kontor",
            "near_cm": 0.0,
            "far_cm": 150.0,
            "fallback": True,
        },
        {
            "zone_id": "kontor_dor",
            "name": "Kontor dør",
            "sensor": KONTOR,
            "room": "kontor",
            "near_cm": 150.0,
            "far_cm": 400.0,
            "fallback": False,
        },
        {
            "zone_id": "sofakrok",
            "name": "Sofakrok",
            "sensor": SOFAKROK,
            "room": "stue",
            "near_cm": 0.0,
            "far_cm": 250.0,
            "fallback": False,
        },
    ],
    "rooms": [
        {"room_id": "kontor", "name": "Kontor"},
        {"room_id": "stue", "name": "Stue"},
    ],
    "baselines": {
        "sofakrok": {"move_mu": 0.02, "move_sigma": 0.02, "still_mu": 0.3, "still_sigma": 0.05}
    },
}


def seed_world(hass: HomeAssistant) -> None:
    """Idle radar states for every configured entity."""
    for entities in (KONTOR_ENTITIES, SOFAKROK_ENTITIES):
        for role, entity_id in entities.items():
            if entity_id.startswith("binary_sensor."):
                hass.states.async_set(entity_id, "off")
            elif "energy" in role:
                hass.states.async_set(entity_id, "2.0")
            else:
                hass.states.async_set(entity_id, "0.0")


async def setup_conductor(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any] | None = None,
    seed: bool = True,
):
    """Set up the integration with FakeEngine and a seeded fake world."""
    import custom_components.presence_conductor as integration

    monkeypatch.setattr(integration, "ConductorEngine", FakeEngine)
    if seed:
        seed_world(hass)
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Presence Conductor",
        unique_id=DOMAIN,
        data={},
        options=options or OPTIONS,
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    controller = hass.data[DOMAIN][entry.entry_id]
    assert controller is not None
    return entry, controller, controller.engine


async def advance(hass: HomeAssistant, freezer, seconds: float) -> None:
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------


async def test_engine_started_with_monotonic_now(hass: HomeAssistant, monkeypatch) -> None:
    _, _controller, fake = await setup_conductor(hass, monkeypatch)
    assert len(fake.start_calls) == 1
    assert fake.snapshot.available == {KONTOR: True, SOFAKROK: True}
    frame = fake.snapshot.frames[KONTOR]
    assert frame == SensorFrame(
        sensor_id=KONTOR,
        moving_distance_cm=0.0,
        still_distance_cm=0.0,
        move_energy=2.0,
        still_energy=2.0,
        move_obs=frame.move_obs,
        still_obs=frame.still_obs,
        frame_obs=frame.frame_obs,
        move_energy_obs=frame.move_energy_obs,
    )
    assert fake.snapshot.baselines["sofakrok"].still_mu == 0.3


async def test_snapshot_marks_unavailable_sensors(hass: HomeAssistant, monkeypatch) -> None:
    seed_world(hass)
    hass.states.async_set(KONTOR_ENTITIES["move_energy"], "unavailable")
    _, _controller, fake = await setup_conductor(hass, monkeypatch, seed=False)
    assert fake.snapshot.available == {KONTOR: False, SOFAKROK: True}
    assert fake.snapshot.frames[KONTOR] is None  # no state worth adopting


# ---------------------------------------------------------------------------
# frame coalescing (rule 1.1)
# ---------------------------------------------------------------------------


async def test_any_entity_change_produces_complete_frame(hass: HomeAssistant, monkeypatch) -> None:
    _, _controller, fake = await setup_conductor(hass, monkeypatch)

    hass.states.async_set(KONTOR_ENTITIES["move_energy"], "42.5")
    await hass.async_block_till_done()
    frame = fake.events_of(SensorFrame)[-1]
    assert frame == SensorFrame(
        sensor_id=KONTOR,
        moving_distance_cm=0.0,
        still_distance_cm=0.0,
        move_energy=42.5,
        still_energy=2.0,
        move_obs=frame.move_obs,
        still_obs=frame.still_obs,
        frame_obs=frame.frame_obs,
        move_energy_obs=frame.move_energy_obs,
    )
    first_move_obs = frame.move_obs
    first_still_obs = frame.still_obs
    first_energy_obs = frame.move_energy_obs

    # The next change reuses the cached view: the energy sticks.
    hass.states.async_set(KONTOR_ENTITIES["moving_target"], "on")
    await hass.async_block_till_done()
    frame = fake.events_of(SensorFrame)[-1]
    # 1.1 observation clock: the flag observed the move channel (verified
    # firmware guarantee — an unchanged energy state is the current
    # measurement), but the attack counter accepts nothing except an
    # actual energy publication (4.2).
    assert frame.move_obs == first_move_obs + 1
    assert frame.still_obs == first_still_obs
    assert frame.move_energy_obs == first_energy_obs
    assert frame == SensorFrame(
        sensor_id=KONTOR,
        moving_distance_cm=0.0,
        still_distance_cm=0.0,
        move_energy=42.5,
        still_energy=2.0,
        has_moving_target=True,
        move_obs=frame.move_obs,
        still_obs=frame.still_obs,
        frame_obs=frame.frame_obs,
        move_energy_obs=frame.move_energy_obs,
    )

    # Distance churn observes its own channel only — and never the attack
    # counter (rule 1.1 / 4.2).
    hass.states.async_set(KONTOR_ENTITIES["still_distance"], "123.0")
    await hass.async_block_till_done()
    frame = fake.events_of(SensorFrame)[-1]
    assert frame.move_obs == first_move_obs + 1
    assert frame.still_obs == first_still_obs + 1
    assert frame.move_energy_obs == first_energy_obs


async def test_unknown_fields_are_none(hass: HomeAssistant, monkeypatch) -> None:
    """A non-required entity going unavailable blanks its field only."""
    _, _controller, fake = await setup_conductor(hass, monkeypatch)

    hass.states.async_set(KONTOR_ENTITIES["still_distance"], "123.0")
    await hass.async_block_till_done()
    before = fake.events_of(SensorFrame)[-1]
    hass.states.async_set(KONTOR_ENTITIES["still_distance"], "unavailable")
    await hass.async_block_till_done()
    frame = fake.events_of(SensorFrame)[-1]
    assert frame.still_distance_cm is None
    assert frame.move_energy == 2.0
    assert frame.still_obs == before.still_obs
    assert frame.frame_obs == before.frame_obs
    assert fake.events_of(SensorAvailability) == []  # distances are not required


async def test_attribute_only_state_event_is_not_a_measurement(
    hass: HomeAssistant, monkeypatch
) -> None:
    _, _controller, fake = await setup_conductor(hass, monkeypatch)
    entity = KONTOR_ENTITIES["move_energy"]
    hass.states.async_set(entity, "42.0", {"marker": 1})
    await hass.async_block_till_done()
    before = fake.events_of(SensorFrame)[-1]

    hass.states.async_set(entity, "42.0", {"marker": 2})
    await hass.async_block_till_done()
    after = fake.events_of(SensorFrame)[-1]
    assert after.move_obs == before.move_obs
    assert after.move_energy_obs == before.move_energy_obs
    assert after.frame_obs == before.frame_obs


async def test_same_value_state_reported_is_a_measurement(hass: HomeAssistant, monkeypatch) -> None:
    _, _controller, fake = await setup_conductor(hass, monkeypatch)
    entity = KONTOR_ENTITIES["move_energy"]
    hass.states.async_set(entity, "42.0")
    await hass.async_block_till_done()
    before = fake.events_of(SensorFrame)[-1]

    # Exact same state + attributes emits state_reported in HA, even when
    # the source entity does not opt into force_update.
    hass.states.async_set(entity, "42.0")
    await hass.async_block_till_done()
    after = fake.events_of(SensorFrame)[-1]
    assert after.move_obs == before.move_obs + 1
    assert after.move_energy_obs == before.move_energy_obs + 1
    assert after.frame_obs == before.frame_obs + 1


async def test_non_numeric_state_is_none(hass: HomeAssistant, monkeypatch) -> None:
    _, _controller, fake = await setup_conductor(hass, monkeypatch)

    hass.states.async_set(KONTOR_ENTITIES["moving_distance"], "123.0")
    await hass.async_block_till_done()
    before = fake.events_of(SensorFrame)[-1]
    hass.states.async_set(KONTOR_ENTITIES["moving_distance"], "not-a-number")
    await hass.async_block_till_done()
    after = fake.events_of(SensorFrame)[-1]
    assert after.moving_distance_cm is None
    assert after.move_obs == before.move_obs
    assert after.frame_obs == before.frame_obs


async def test_detection_distance_produces_no_frame(hass: HomeAssistant, monkeypatch) -> None:
    """The v1 estimator does not consume detection_distance (const.py)."""
    _, _controller, fake = await setup_conductor(hass, monkeypatch)

    hass.states.async_set(SOFAKROK_ENTITIES["detection_distance"], "123")
    await hass.async_block_till_done()
    assert fake.events_of(SensorFrame) == []


async def test_unrelated_entities_are_ignored(hass: HomeAssistant, monkeypatch) -> None:
    _, _controller, fake = await setup_conductor(hass, monkeypatch)

    hass.states.async_set("sensor.unrelated", "99")
    await hass.async_block_till_done()
    assert fake.events == []


# ---------------------------------------------------------------------------
# availability (rule 1.3)
# ---------------------------------------------------------------------------


async def test_required_entity_unavailable_flips_availability(
    hass: HomeAssistant, monkeypatch
) -> None:
    _, _controller, fake = await setup_conductor(hass, monkeypatch)

    hass.states.async_set(KONTOR_ENTITIES["move_energy"], "unavailable")
    await hass.async_block_till_done()
    assert fake.events_of(SensorAvailability) == [SensorAvailability(KONTOR, False)]
    assert fake.events_of(SensorFrame) == []  # a frame would count as recovery

    # Changes on other entities of a blind sensor still produce no frames.
    hass.states.async_set(KONTOR_ENTITIES["moving_distance"], "120")
    await hass.async_block_till_done()
    assert fake.events_of(SensorFrame) == []

    # Recovery: availability first, then the coalesced frame.
    hass.states.async_set(KONTOR_ENTITIES["move_energy"], "5.5")
    await hass.async_block_till_done()
    assert fake.events_of(SensorAvailability) == [
        SensorAvailability(KONTOR, False),
        SensorAvailability(KONTOR, True),
    ]
    frame = fake.events_of(SensorFrame)[-1]
    assert frame.move_energy == 5.5
    assert frame.moving_distance_cm == 120.0  # cached while blind


async def test_unknown_state_counts_as_unavailable(hass: HomeAssistant, monkeypatch) -> None:
    _, _controller, fake = await setup_conductor(hass, monkeypatch)

    hass.states.async_set(SOFAKROK_ENTITIES["still_energy"], "unknown")
    await hass.async_block_till_done()
    assert fake.events_of(SensorAvailability) == [SensorAvailability(SOFAKROK, False)]


# ---------------------------------------------------------------------------
# per-gate entities (rules 2.4-2.6)
# ---------------------------------------------------------------------------

GATED = "gated_radar"
GATED_ENTITIES = {
    "move_energy": "sensor.gated_move_energy",
    "still_energy": "sensor.gated_still_energy",
    "moving_distance": "sensor.gated_moving_distance",
    "still_distance": "sensor.gated_still_distance",
    **{f"g{i}_move": f"sensor.gated_g{i}_move_energy" for i in range(GATE_COUNT)},
    **{f"g{i}_still": f"sensor.gated_g{i}_still_energy" for i in range(GATE_COUNT)},
}
GATED_OPTIONS: dict[str, Any] = {
    "sensors": [{"sensor_id": GATED, "name": "Gated radar", "entities": GATED_ENTITIES}],
    "zones": [
        {
            "zone_id": "gated_zone",
            "name": "Gated zone",
            "sensor": GATED,
            "room": "stue",
            "near_cm": 0.0,
            "far_cm": 300.0,
            "fallback": True,
        }
    ],
    "rooms": [{"room_id": "stue", "name": "Stue"}],
}


def seed_gated_world(hass: HomeAssistant) -> None:
    for role, entity_id in GATED_ENTITIES.items():
        if role.startswith("g"):
            hass.states.async_set(entity_id, "3.0")
        elif "energy" in role:
            hass.states.async_set(entity_id, "2.0")
        else:
            hass.states.async_set(entity_id, "0.0")


async def test_gate_entities_feed_the_frame(hass: HomeAssistant, monkeypatch) -> None:
    """Configured gate entities ride along in the snapshot and every frame."""
    seed_gated_world(hass)
    _, _controller, fake = await setup_conductor(
        hass, monkeypatch, options=GATED_OPTIONS, seed=False
    )
    snapshot_frame = fake.snapshot.frames[GATED]
    assert snapshot_frame.gate_move == (3.0,) * GATE_COUNT  # 7.1 seeding
    assert snapshot_frame.gate_still == (3.0,) * GATE_COUNT

    hass.states.async_set(GATED_ENTITIES["g3_move"], "42.5")
    await hass.async_block_till_done()
    frame = fake.events_of(SensorFrame)[-1]
    assert frame.gate_move == (3.0, 3.0, 3.0, 42.5, 3.0, 3.0, 3.0, 3.0, 3.0)
    assert frame.gate_still == (3.0,) * GATE_COUNT
    assert frame.move_energy == 2.0  # aggregates unchanged


async def test_unknown_gate_is_none_without_availability_impact(
    hass: HomeAssistant, monkeypatch
) -> None:
    """Engineering mode off: unknown gates go None per gate, while sensor
    availability stays keyed to the aggregate energy roles (rule 1.3)."""
    seed_gated_world(hass)
    _, _controller, fake = await setup_conductor(
        hass, monkeypatch, options=GATED_OPTIONS, seed=False
    )
    hass.states.async_set(GATED_ENTITIES["g0_move"], "unknown")
    await hass.async_block_till_done()
    frame = fake.events_of(SensorFrame)[-1]
    assert frame.gate_move[0] is None
    assert frame.gate_move[1] == 3.0
    assert fake.events_of(SensorAvailability) == []  # gates are not required


async def test_sensor_without_gate_entities_carries_no_gate_tuples(
    hass: HomeAssistant, monkeypatch
) -> None:
    _, _controller, fake = await setup_conductor(hass, monkeypatch)
    hass.states.async_set(KONTOR_ENTITIES["move_energy"], "42.5")
    await hass.async_block_till_done()
    frame = fake.events_of(SensorFrame)[-1]
    assert frame.gate_move is None
    assert frame.gate_still is None


# ---------------------------------------------------------------------------
# tick cadence (rule 1.2)
# ---------------------------------------------------------------------------


async def test_tick_cadence(hass: HomeAssistant, monkeypatch, freezer) -> None:
    _, _controller, fake = await setup_conductor(hass, monkeypatch)
    assert fake.events_of(Tick) == []

    await advance(hass, freezer, 1.05)
    assert len(fake.events_of(Tick)) == 1
    await advance(hass, freezer, 1.05)
    await advance(hass, freezer, 1.05)
    assert len(fake.events_of(Tick)) == 3


# ---------------------------------------------------------------------------
# timers
# ---------------------------------------------------------------------------

STALE = "sensor_stale:kontor_radar"


async def test_timer_fires_after_delay(hass: HomeAssistant, monkeypatch, freezer) -> None:
    _, controller, fake = await setup_conductor(hass, monkeypatch)

    fake.script(fake.plan(starts=[(STALE, 30.0)]))
    controller.submit(Tick())
    await hass.async_block_till_done()

    await advance(hass, freezer, 20.0)
    assert fake.timer_fires == []
    await advance(hass, freezer, 10.1)
    assert fake.timer_fires == [STALE]


async def test_timer_restart_resets_delay(hass: HomeAssistant, monkeypatch, freezer) -> None:
    _, controller, fake = await setup_conductor(hass, monkeypatch)

    fake.script(fake.plan(starts=[(STALE, 30.0)]))
    controller.submit(Tick())
    await hass.async_block_till_done()
    await advance(hass, freezer, 20.0)

    fake.script(fake.plan(starts=[(STALE, 30.0)]))  # restart resets the clock
    controller.submit(Tick())
    await hass.async_block_till_done()

    await advance(hass, freezer, 20.0)  # 40 s after first start, 20 s after restart
    assert fake.timer_fires == []
    await advance(hass, freezer, 10.1)
    assert fake.timer_fires == [STALE]


async def test_timer_cancel_prevents_firing(hass: HomeAssistant, monkeypatch, freezer) -> None:
    _, controller, fake = await setup_conductor(hass, monkeypatch)

    fake.script(fake.plan(starts=[(STALE, 30.0)]))
    controller.submit(Tick())
    await hass.async_block_till_done()

    fake.script(fake.plan(cancels=[STALE]))
    controller.submit(Tick())
    await hass.async_block_till_done()

    await advance(hass, freezer, 60.0)
    assert fake.timer_fires == []


# ---------------------------------------------------------------------------
# calibration persistence (rule 3.3)
# ---------------------------------------------------------------------------


async def test_persist_calibration_writes_options_without_reload(
    hass: HomeAssistant, monkeypatch
) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)

    zst = fake.state.zones["kontor_pult"]
    zst.move_baseline.mu = 0.031
    zst.move_baseline.sigma = 0.021
    zst.still_baseline.mu = 0.062
    zst.still_baseline.sigma = 0.02
    fake.script(fake.plan(persist=True))
    controller.submit(Tick())
    await hass.async_block_till_done()

    assert entry.options["baselines"]["kontor_pult"] == {
        "move_mu": 0.031,
        "move_sigma": 0.021,
        "still_mu": 0.062,
        "still_sigma": 0.02,
        "sensor_id": KONTOR,
        "floor_fingerprint": "v1|floor_min=0.02|quantum=0.01",
        "gate_size_cm": 75.0,
    }
    # Every configured zone is captured; stored keys survive the merge.
    assert set(entry.options["baselines"]) == {"kontor_pult", "kontor_dor", "sofakrok"}
    # The baselines-only write must not reload the entry (no loop).
    assert hass.data[DOMAIN][entry.entry_id] is controller
    assert len(fake.start_calls) == 1


async def test_persist_calibration_includes_gate_floors(hass: HomeAssistant, monkeypatch) -> None:
    """Rule 3.6: per-gate floors persist under "gates" (string keys, JSON);
    zones without any keep the v0.1.0 schema exactly."""
    entry, controller, fake = await setup_conductor(hass, monkeypatch)

    zst = fake.state.zones["kontor_pult"]
    owned = fake.owned_gates["kontor_pult"]
    zst.gate_move_baselines = {index: ChannelStats(0.011 + index / 1000, 0.021) for index in owned}
    zst.gate_move_ready = True
    zst.gate_still_baselines = {}  # rejected/absent optional counterpart
    zst.gate_still_ready = False
    fake.script(fake.plan(persist=True))
    controller.submit(Tick())
    await hass.async_block_till_done()

    stored = entry.options["baselines"]["kontor_pult"]
    assert stored["gate_indices"] == list(owned)
    assert set(stored["gates"]) == {str(index) for index in owned}
    assert all(gate["has_move"] and not gate["has_still"] for gate in stored["gates"].values())
    reloaded = ConductorEngine(
        controller.config,
        InitialSnapshot(baselines=baselines_from_options(entry.options)),
    ).state.zones["kontor_pult"]
    assert reloaded.gate_move_ready
    assert not reloaded.gate_still_ready
    assert reloaded.gate_still_baselines == {}
    assert "gates" not in entry.options["baselines"]["kontor_dor"]  # v0.1.0 schema


async def test_baselines_with_gates_round_trip_into_the_snapshot(
    hass: HomeAssistant, monkeypatch
) -> None:
    """Stored "gates" floors reach the engine through the snapshot (3.6)."""
    gate_floor = {"move_mu": 0.01, "move_sigma": 0.02, "still_mu": 0.03, "still_sigma": 0.04}
    options = {
        **OPTIONS,
        "baselines": {"sofakrok": {**OPTIONS["baselines"]["sofakrok"], "gates": {"1": gate_floor}}},
    }
    _, _controller, fake = await setup_conductor(hass, monkeypatch, options=options)
    assert fake.snapshot.baselines["sofakrok"].gates == {
        1: GateBaselines(move_mu=0.01, move_sigma=0.02, still_mu=0.03, still_sigma=0.04)
    }


async def test_persist_preserves_stale_zone_keys(hass: HomeAssistant, monkeypatch) -> None:
    """Baseline keys of removed zones survive the merge (ignored on read)."""
    gone = {"move_mu": 1, "move_sigma": 1, "still_mu": 1, "still_sigma": 1}
    options = {**OPTIONS, "baselines": {"gone_zone": gone}}
    entry, controller, fake = await setup_conductor(hass, monkeypatch, options=options)

    fake.script(fake.plan(persist=True))
    controller.submit(Tick())
    await hass.async_block_till_done()
    assert "gone_zone" in entry.options["baselines"]
    assert "sofakrok" in entry.options["baselines"]


async def test_real_options_change_reloads(hass: HomeAssistant, monkeypatch) -> None:
    entry, controller, _fake = await setup_conductor(hass, monkeypatch)

    new_zones = [dict(zone, far_cm=500.0) for zone in OPTIONS["zones"]]
    hass.config_entries.async_update_entry(entry, options={**entry.options, "zones": new_zones})
    await hass.async_block_till_done()
    new_controller = hass.data[DOMAIN][entry.entry_id]
    assert new_controller is not None
    assert new_controller is not controller  # reloaded
    assert entry.state.value == "loaded"


# ---------------------------------------------------------------------------
# pass-by (rule 5.2)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("entity_registry_enabled_by_default")
async def test_pass_by_fires_bus_event_and_event_entities(hass: HomeAssistant, monkeypatch) -> None:
    _, controller, fake = await setup_conductor(hass, monkeypatch)
    captured = async_capture_events(hass, "presence_conductor_pass_by")

    fake.script(fake.plan(events=[PassBy("sofakrok", 0.9312, 3.217)]))
    controller.submit(Tick())
    await hass.async_block_till_done()

    assert len(captured) == 1
    assert captured[0].data == {
        "zone_id": "sofakrok",
        "peak_confidence": 0.9312,
        "duration": 3.22,
    }
    entity_state = hass.states.get("event.presence_conductor_sofakrok_pass_by")
    assert entity_state.state != "unknown"  # a timestamp: the event fired
    assert entity_state.attributes["event_type"] == "pass_by"
    assert entity_state.attributes["peak_confidence"] == 0.9312
    assert entity_state.attributes["duration"] == 3.22
    # The zone's room fired too (§6 membership), carrying the zone id.
    room_state = hass.states.get("event.presence_conductor_stue_room_pass_by")
    assert room_state.state != "unknown"
    assert room_state.attributes["event_type"] == "pass_by"
    assert room_state.attributes["zone_id"] == "sofakrok"
    assert room_state.attributes["peak_confidence"] == 0.9312
    assert room_state.attributes["duration"] == 3.22
    # The other room saw nothing.
    assert hass.states.get("event.presence_conductor_kontor_room_pass_by").state == "unknown"


# ---------------------------------------------------------------------------
# suppression (rule 7.2)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("entity_registry_enabled_by_default")
async def test_suppress_outputs_freezes_state_entities(hass: HomeAssistant, monkeypatch) -> None:
    _, controller, fake = await setup_conductor(hass, monkeypatch)
    occupancy = "binary_sensor.presence_conductor_sofakrok_occupancy"
    assert hass.states.get(occupancy).state == "off"

    # Engine state changes underneath, but the plan says suppress: frozen.
    fake.state.enabled = False
    fake.state.zones["sofakrok"].occupied = True
    fake.script(fake.plan(suppress=True))
    controller.submit(Tick())
    await hass.async_block_till_done()
    assert hass.states.get(occupancy).state == "off"  # frozen for consumers
    # Control surfaces keep updating: operators must see "disabled".
    assert hass.states.get("sensor.presence_conductor_state").state == "disabled"
    assert hass.states.get("switch.presence_conductor_enabled").state == "off"

    # Re-enable: the first non-suppressed plan publishes everything once.
    fake.state.enabled = True
    fake.script(fake.plan(suppress=False))
    controller.submit(Tick())
    await hass.async_block_till_done()
    assert hass.states.get(occupancy).state == "on"
    assert hass.states.get("sensor.presence_conductor_state").state == "enabled"


# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------


async def test_unload_stops_ticks_and_timers(hass: HomeAssistant, monkeypatch, freezer) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)

    fake.script(fake.plan(starts=[(STALE, 30.0)]))
    controller.submit(Tick())
    await hass.async_block_till_done()
    submitted = len(fake.events)

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state.value == "not_loaded"

    await advance(hass, freezer, 60.0)  # neither ticks nor the timer fire
    assert len(fake.events) == submitted
    assert fake.timer_fires == []

    # State changes no longer reach the (stopped) engine either.
    hass.states.async_set(KONTOR_ENTITIES["move_energy"], "50")
    await hass.async_block_till_done()
    assert len(fake.events) == submitted
    hass.states.async_set(KONTOR_ENTITIES["move_energy"], "50")  # state_reported too
    await hass.async_block_till_done()
    assert len(fake.events) == submitted
