"""Entity platform tests: engine-state mirroring and command routing."""

from __future__ import annotations

import pytest
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_send
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    mock_restore_cache,
)

from custom_components.presence_conductor.const import DOMAIN
from custom_components.presence_conductor.core.events import RecordBaseline, SetEnabled
from custom_components.presence_conductor.core.model import Activity, Health
from tests.test_controller import setup_conductor


def entity_id_for(hass: HomeAssistant, platform: str, unique_id: str) -> str:
    registry = er.async_get(hass)
    entity_id = registry.async_get_entity_id(platform, DOMAIN, unique_id)
    assert entity_id is not None, f"no {platform} entity with unique_id {unique_id}"
    return entity_id


# ---------------------------------------------------------------------------
# zone binary sensors
# ---------------------------------------------------------------------------

# Zone state entities ship disabled (rooms and home are the consumer
# surface; tests/test_devices.py covers the defaults). Tests exercising
# zone entity behavior opt back in with entity_registry_enabled_by_default.


@pytest.mark.usefixtures("entity_registry_enabled_by_default")
async def test_zone_occupancy_mirrors_engine_state(hass: HomeAssistant, monkeypatch) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    occupancy = entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_zone_sofakrok_occupancy")
    assert occupancy == "binary_sensor.presence_conductor_sofakrok_occupancy"

    state = hass.states.get(occupancy)
    assert state.state == "off"
    assert state.attributes["device_class"] == "occupancy"
    assert state.attributes["zone_id"] == "sofakrok"
    assert state.attributes["room"] == "stue"

    fake.state.zones["sofakrok"].occupied = True
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(occupancy).state == "on"


@pytest.mark.usefixtures("entity_registry_enabled_by_default")
async def test_zone_motion_mirrors_engine_state(hass: HomeAssistant, monkeypatch) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    motion = entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_zone_kontor_pult_motion")

    state = hass.states.get(motion)
    assert state.state == "off"
    assert state.attributes["device_class"] == "motion"

    fake.state.zones["kontor_pult"].motion = True
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(motion).state == "on"


@pytest.mark.usefixtures("entity_registry_enabled_by_default")
async def test_zone_entities_unavailable_while_health_unknown(
    hass: HomeAssistant, monkeypatch
) -> None:
    """Rule 1.3: stale probability maps to unavailable, recovery restores."""
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    entity_ids = [
        entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_zone_sofakrok_occupancy"),
        entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_zone_sofakrok_motion"),
        entity_id_for(hass, "sensor", f"{entry.entry_id}_zone_sofakrok_activity"),
        entity_id_for(hass, "sensor", f"{entry.entry_id}_zone_sofakrok_probability"),
        entity_id_for(hass, "sensor", f"{entry.entry_id}_zone_sofakrok_dwell"),
    ]

    fake.state.zones["sofakrok"].health = Health.UNKNOWN
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    for entity_id in entity_ids:
        assert hass.states.get(entity_id).state == "unavailable", entity_id

    # Sibling zones on other sensors are untouched.
    other = entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_zone_kontor_pult_occupancy")
    assert hass.states.get(other).state == "off"

    fake.state.zones["sofakrok"].health = Health.OK
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    for entity_id in entity_ids:
        assert hass.states.get(entity_id).state != "unavailable", entity_id


# ---------------------------------------------------------------------------
# zone sensors
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("entity_registry_enabled_by_default")
async def test_zone_activity_enum(hass: HomeAssistant, monkeypatch) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    activity = entity_id_for(hass, "sensor", f"{entry.entry_id}_zone_sofakrok_activity")

    state = hass.states.get(activity)
    assert state.state == "empty"
    assert state.attributes["options"] == ["empty", "passing", "active", "settled"]
    assert state.attributes["device_class"] == "enum"

    fake.state.zones["sofakrok"].activity = Activity.SETTLED
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(activity).state == "settled"


@pytest.mark.usefixtures("entity_registry_enabled_by_default")
async def test_zone_probability_and_dwell(hass: HomeAssistant, monkeypatch) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    probability = entity_id_for(hass, "sensor", f"{entry.entry_id}_zone_sofakrok_probability")
    dwell = entity_id_for(hass, "sensor", f"{entry.entry_id}_zone_sofakrok_dwell")

    assert float(hass.states.get(probability).state) == pytest.approx(2.0)  # the prior
    assert hass.states.get(probability).attributes["unit_of_measurement"] == "%"
    assert hass.states.get(dwell).state == "0.0"
    assert hass.states.get(dwell).attributes["unit_of_measurement"] == "s"

    fake.state.zones["sofakrok"].lam = 0.0  # p = 0.5
    fake.state.zones["sofakrok"].dwell_seconds = 12.34
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert float(hass.states.get(probability).state) == pytest.approx(50.0)
    assert hass.states.get(dwell).state == "12.3"


async def test_diagnostic_and_config_entity_categories(hass: HomeAssistant, monkeypatch) -> None:
    entry, _controller, _fake = await setup_conductor(hass, monkeypatch)
    registry = er.async_get(hass)

    diagnostic = er.EntityCategory.DIAGNOSTIC

    def category(platform: str, unique_id: str):
        return registry.async_get(entity_id_for(hass, platform, unique_id)).entity_category

    assert category("sensor", f"{entry.entry_id}_zone_sofakrok_probability") == diagnostic
    assert category("sensor", f"{entry.entry_id}_zone_sofakrok_dwell") == diagnostic
    assert category("sensor", f"{entry.entry_id}_room_stue_probability") == diagnostic
    assert category("sensor", f"{entry.entry_id}_home_probability") == diagnostic
    assert category("sensor", f"{entry.entry_id}_state") == diagnostic
    assert category("switch", f"{entry.entry_id}_enabled") == er.EntityCategory.CONFIG
    assert (
        category("button", f"{entry.entry_id}_zone_sofakrok_record_baseline")
        == er.EntityCategory.CONFIG
    )
    # Primary outputs stay uncategorized.
    assert category("binary_sensor", f"{entry.entry_id}_zone_sofakrok_occupancy") is None
    assert category("sensor", f"{entry.entry_id}_zone_sofakrok_activity") is None


# ---------------------------------------------------------------------------
# room entities
# ---------------------------------------------------------------------------


async def test_room_entities_mirror_fusion(hass: HomeAssistant, monkeypatch) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    occupancy = entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_room_kontor_occupancy")
    motion = entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_room_kontor_motion")
    settled = entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_room_kontor_settled")
    activity = entity_id_for(hass, "sensor", f"{entry.entry_id}_room_kontor_activity")
    probability = entity_id_for(hass, "sensor", f"{entry.entry_id}_room_kontor_probability")
    assert occupancy == "binary_sensor.presence_conductor_kontor_room_occupancy"
    assert motion == "binary_sensor.presence_conductor_kontor_room_motion"

    assert hass.states.get(occupancy).state == "off"
    assert hass.states.get(occupancy).attributes["zones"] == ["kontor_pult", "kontor_dor"]
    assert hass.states.get(motion).state == "off"
    assert hass.states.get(motion).attributes["device_class"] == "motion"
    assert hass.states.get(settled).state == "off"
    assert hass.states.get(activity).state == "empty"

    room = fake.state.rooms["kontor"]
    room.occupied = True
    room.motion = True
    room.settled = True
    room.activity = Activity.SETTLED
    room.probability = 0.987
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(occupancy).state == "on"
    assert hass.states.get(motion).state == "on"
    assert hass.states.get(settled).state == "on"
    assert hass.states.get(activity).state == "settled"
    assert float(hass.states.get(probability).state) == pytest.approx(98.7)


async def test_room_entities_unavailable_when_fusion_unknown(
    hass: HomeAssistant, monkeypatch
) -> None:
    """Rule 6.3: a blind room publishes unknown, not off."""
    entry, controller, fake = await setup_conductor(hass, monkeypatch)

    room = fake.state.rooms["kontor"]
    room.occupied = None
    room.motion = None
    room.settled = None
    room.activity = None
    room.probability = None
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    for unique_suffix, platform in (
        ("occupancy", "binary_sensor"),
        ("motion", "binary_sensor"),
        ("settled", "binary_sensor"),
        ("activity", "sensor"),
        ("probability", "sensor"),
    ):
        entity_id = entity_id_for(hass, platform, f"{entry.entry_id}_room_kontor_{unique_suffix}")
        assert hass.states.get(entity_id).state == "unavailable", entity_id


# ---------------------------------------------------------------------------
# home entities
# ---------------------------------------------------------------------------


async def test_anyone_home_and_probability(hass: HomeAssistant, monkeypatch) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    anyone = entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_anyone_home")
    probability = entity_id_for(hass, "sensor", f"{entry.entry_id}_home_probability")
    assert anyone == "binary_sensor.presence_conductor_anyone_home"

    state = hass.states.get(anyone)
    assert state.state == "off"
    assert state.attributes["device_class"] == "presence"

    fake.state.anyone_home = True
    fake.state.home_probability = 0.964
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(anyone).state == "on"
    assert float(hass.states.get(probability).state) == pytest.approx(96.4)

    # Rule 6.5: all zones unhealthy -> anyone_home publishes unknown.
    fake.state.anyone_home = None
    fake.state.home_probability = None
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(anyone).state == "unavailable"
    assert hass.states.get(probability).state == "unavailable"


# ---------------------------------------------------------------------------
# enabled switch (rule 7.2)
# ---------------------------------------------------------------------------


async def test_enabled_switch_mirrors_and_submits(hass: HomeAssistant, monkeypatch) -> None:
    entry, _controller, fake = await setup_conductor(hass, monkeypatch)
    switch = entity_id_for(hass, "switch", f"{entry.entry_id}_enabled")
    assert switch == "switch.presence_conductor_enabled"
    assert hass.states.get(switch).state == "on"

    await hass.services.async_call("switch", "turn_off", {"entity_id": switch}, blocking=True)
    await hass.async_block_till_done()
    assert fake.events_of(SetEnabled) == [SetEnabled(False)]
    # The switch is a control surface: it reflects "off" even though the
    # resulting plan suppressed the state signal.
    assert hass.states.get(switch).state == "off"

    await hass.services.async_call("switch", "turn_on", {"entity_id": switch}, blocking=True)
    await hass.async_block_till_done()
    assert fake.events_of(SetEnabled) == [SetEnabled(False), SetEnabled(True)]
    assert hass.states.get(switch).state == "on"


async def test_enabled_switch_restores_off(hass: HomeAssistant, monkeypatch) -> None:
    """A restored 'off' is pushed back into the engine as SetEnabled(False)."""
    mock_restore_cache(hass, (State("switch.presence_conductor_enabled", "off"),))
    _entry, _controller, fake = await setup_conductor(hass, monkeypatch)
    assert fake.events_of(SetEnabled) == [SetEnabled(False)]
    assert fake.state.enabled is False
    assert hass.states.get("switch.presence_conductor_enabled").state == "off"


async def test_enabled_switch_restore_matching_state_is_silent(
    hass: HomeAssistant, monkeypatch
) -> None:
    mock_restore_cache(hass, (State("switch.presence_conductor_enabled", "on"),))
    _entry, _controller, fake = await setup_conductor(hass, monkeypatch)
    assert fake.events_of(SetEnabled) == []


async def test_enabled_switch_ignores_invalid_restore(hass: HomeAssistant, monkeypatch) -> None:
    mock_restore_cache(hass, (State("switch.presence_conductor_enabled", "unavailable"),))
    _entry, _controller, fake = await setup_conductor(hass, monkeypatch)
    assert fake.events_of(SetEnabled) == []
    assert hass.states.get("switch.presence_conductor_enabled").state == "on"


# ---------------------------------------------------------------------------
# record-baseline button + service (rule 3.3)
# ---------------------------------------------------------------------------


async def test_record_baseline_button(hass: HomeAssistant, monkeypatch) -> None:
    entry, _controller, fake = await setup_conductor(hass, monkeypatch)
    button = entity_id_for(hass, "button", f"{entry.entry_id}_zone_sofakrok_record_baseline")
    assert button == "button.presence_conductor_sofakrok_record_baseline"

    await hass.services.async_call("button", "press", {"entity_id": button}, blocking=True)
    await hass.async_block_till_done()
    assert fake.events_of(RecordBaseline) == [RecordBaseline("sofakrok", None)]


async def test_record_baseline_service(hass: HomeAssistant, monkeypatch) -> None:
    _entry, _controller, fake = await setup_conductor(hass, monkeypatch)

    await hass.services.async_call(
        DOMAIN,
        "record_baseline",
        {"zone_id": "kontor_pult", "duration": 60},
        blocking=True,
    )
    await hass.async_block_till_done()
    assert fake.events_of(RecordBaseline) == [RecordBaseline("kontor_pult", 60.0)]

    # Duration is optional: the engine falls back to the tunable default.
    await hass.services.async_call(
        DOMAIN, "record_baseline", {"zone_id": "sofakrok"}, blocking=True
    )
    await hass.async_block_till_done()
    assert fake.events_of(RecordBaseline)[-1] == RecordBaseline("sofakrok", None)


async def test_record_baseline_service_rejects_unknown_zone(
    hass: HomeAssistant, monkeypatch
) -> None:
    _entry, _controller, fake = await setup_conductor(hass, monkeypatch)

    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN, "record_baseline", {"zone_id": "nope"}, blocking=True
        )
    assert fake.events_of(RecordBaseline) == []


# ---------------------------------------------------------------------------
# pass-by event entities
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("entity_registry_enabled_by_default")
async def test_pass_by_event_entities_exist_per_zone_and_room(
    hass: HomeAssistant, monkeypatch
) -> None:
    entry, _controller, _fake = await setup_conductor(hass, monkeypatch)
    unique_ids = [f"{entry.entry_id}_zone_{z}_pass_by" for z in ("kontor_pult", "kontor_dor")]
    unique_ids += [f"{entry.entry_id}_room_{r}_pass_by" for r in ("kontor", "stue")]
    for unique_id in unique_ids:
        event = entity_id_for(hass, "event", unique_id)
        state = hass.states.get(event)
        assert state.state == "unknown"  # nothing traversed yet
        assert state.attributes["event_types"] == ["pass_by"]


# ---------------------------------------------------------------------------
# diagnostics sensor
# ---------------------------------------------------------------------------


async def test_diagnostics_sensor(hass: HomeAssistant, monkeypatch) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    sensor = entity_id_for(hass, "sensor", f"{entry.entry_id}_state")

    state = hass.states.get(sensor)
    assert state.state == "enabled"
    assert state.attributes["enabled"] is True
    assert state.attributes["anyone_home"] is False
    sofakrok = state.attributes["zones"]["sofakrok"]
    assert sofakrok["health"] == "ok"
    assert sofakrok["activity"] == "empty"
    assert sofakrok["occupied"] is False
    assert "lambda" in sofakrok
    assert state.attributes["rooms"]["stue"]["occupied"] is False
    assert state.attributes["rooms"]["stue"]["motion"] is False
    assert state.attributes["sensors"]["sofakrok_radar"] == {"available": True}
    assert "home_lambda" in state.attributes

    fake.state.enabled = False
    async_dispatcher_send(hass, controller.signal_control)
    await hass.async_block_till_done()
    assert hass.states.get(sensor).state == "disabled"


# ---------------------------------------------------------------------------
# entity inventory + unconfigured entry
# ---------------------------------------------------------------------------


async def test_entity_inventory_spans_hub_and_room_devices(
    hass: HomeAssistant, monkeypatch
) -> None:
    entry, _controller, _fake = await setup_conductor(hass, monkeypatch)
    registry = er.async_get(hass)
    entries = er.async_entries_for_config_entry(registry, entry.entry_id)
    # 3 zones x (occupancy, motion, activity, probability, dwell, pass-by,
    # record baseline) + 2 rooms x (occupancy, motion, settled, activity,
    # probability, pass-by) + anyone_home + home probability + enabled +
    # state. Disabled-by-default zone entities are registered like the rest.
    assert len(entries) == 3 * 7 + 2 * 6 + 4
    # Hub + one device per room; the layout itself is tests/test_devices.py.
    assert len({e.device_id for e in entries}) == 3


async def test_unconfigured_entry_creates_no_entities(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN, title="Presence Conductor", unique_id=DOMAIN, data={}, options={}
    )
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state.value == "loaded"
    assert hass.data[DOMAIN][entry.entry_id] is None

    registry = er.async_get(hass)
    assert er.async_entries_for_config_entry(registry, entry.entry_id) == []

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state.value == "not_loaded"
