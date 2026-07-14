"""Device layout and visibility defaults.

Rooms and home are the consumer surface: the hub device carries the
home-level outputs, controls and diagnostics, and every configured room is
its own device (via_device -> hub) carrying the room's fused outputs plus
its member zones' outputs. Zone state entities ship disabled by default —
estimator internals, opt-in per entity — while each zone's record-baseline
button stays enabled: calibration is a first-class operator action.
"""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er

from custom_components.presence_conductor.const import DOMAIN
from tests.test_controller import OPTIONS, setup_conductor

# ---------------------------------------------------------------------------
# device layout
# ---------------------------------------------------------------------------


async def test_hub_and_one_device_per_room(hass: HomeAssistant, monkeypatch) -> None:
    entry, _controller, _fake = await setup_conductor(hass, monkeypatch)
    registry = dr.async_get(hass)
    devices = dr.async_entries_for_config_entry(registry, entry.entry_id)
    assert len(devices) == 3  # hub + kontor + stue

    hub = registry.async_get_device(identifiers={(DOMAIN, entry.entry_id)})
    assert hub is not None
    assert hub.name == "Presence Conductor"
    assert hub.via_device_id is None

    for room_id, room_name in (("kontor", "Kontor"), ("stue", "Stue")):
        device = registry.async_get_device(
            identifiers={(DOMAIN, f"{entry.entry_id}_room_{room_id}")}
        )
        assert device is not None, room_id
        assert device.name == f"{room_name} presence"
        assert device.via_device_id == hub.id
        # Presentation consistent with the hub.
        assert device.manufacturer == hub.manufacturer
        assert device.model == hub.model
        assert device.entry_type == hub.entry_type
        # suggested_area placed the device in an area named after the room.
        assert device.area_id is not None
        area = ar.async_get(hass).async_get_area(device.area_id)
        assert area.name == room_name


async def test_entities_live_on_their_room_device(hass: HomeAssistant, monkeypatch) -> None:
    entry, _controller, _fake = await setup_conductor(hass, monkeypatch)
    device_registry = dr.async_get(hass)
    entity_registry = er.async_get(hass)
    hub = device_registry.async_get_device(identifiers={(DOMAIN, entry.entry_id)})
    stue = device_registry.async_get_device(identifiers={(DOMAIN, f"{entry.entry_id}_room_stue")})
    kontor = device_registry.async_get_device(
        identifiers={(DOMAIN, f"{entry.entry_id}_room_kontor")}
    )

    def device_of(platform: str, unique_id: str) -> str:
        entity_id = entity_registry.async_get_entity_id(platform, DOMAIN, unique_id)
        assert entity_id is not None, unique_id
        return entity_registry.async_get(entity_id).device_id

    # Zone entities live on the device of the room their zone belongs to.
    for platform, suffix in (
        ("binary_sensor", "occupancy"),
        ("binary_sensor", "motion"),
        ("sensor", "activity"),
        ("sensor", "confidence"),
        ("sensor", "dwell"),
        ("sensor", "calibration_status"),
        ("event", "pass_by"),
        ("button", "record_baseline"),
    ):
        assert device_of(platform, f"{entry.entry_id}_zone_sofakrok_{suffix}") == stue.id
        assert device_of(platform, f"{entry.entry_id}_zone_kontor_pult_{suffix}") == kontor.id

    # Room entities live on their room device.
    for platform, suffix in (
        ("binary_sensor", "occupancy"),
        ("binary_sensor", "motion"),
        ("binary_sensor", "settled"),
        ("sensor", "activity"),
        ("sensor", "confidence"),
        ("event", "pass_by"),
    ):
        assert device_of(platform, f"{entry.entry_id}_room_stue_{suffix}") == stue.id

    # Home outputs, controls and diagnostics stay on the hub.
    assert device_of("binary_sensor", f"{entry.entry_id}_anyone_home") == hub.id
    assert device_of("sensor", f"{entry.entry_id}_home_confidence") == hub.id
    assert device_of("switch", f"{entry.entry_id}_enabled") == hub.id
    assert device_of("sensor", f"{entry.entry_id}_state") == hub.id


async def test_removed_room_releases_its_device(hass: HomeAssistant, monkeypatch) -> None:
    """Reconfiguring away a room releases its device from the entry."""
    entry, _controller, _fake = await setup_conductor(hass, monkeypatch)

    # Move every zone into kontor: the stue room disappears.
    new_zones = [dict(zone, room="kontor") for zone in OPTIONS["zones"]]
    new_rooms = [{"room_id": "kontor", "name": "Kontor"}]
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, "zones": new_zones, "rooms": new_rooms}
    )
    await hass.async_block_till_done()

    registry = dr.async_get(hass)
    devices = dr.async_entries_for_config_entry(registry, entry.entry_id)
    assert len(devices) == 2  # hub + kontor
    assert registry.async_get_device(identifiers={(DOMAIN, f"{entry.entry_id}_room_stue")}) is None


# ---------------------------------------------------------------------------
# visibility defaults
# ---------------------------------------------------------------------------


async def test_zone_state_entities_are_opt_in(hass: HomeAssistant, monkeypatch) -> None:
    """Zone outputs ship disabled: rooms and home are the consumer surface
    (spec §0); zones are opt-in per-entity diagnostics."""
    entry, _controller, _fake = await setup_conductor(hass, monkeypatch)
    registry = er.async_get(hass)

    def registry_entry(platform: str, unique_id: str) -> er.RegistryEntry:
        entity_id = registry.async_get_entity_id(platform, DOMAIN, unique_id)
        assert entity_id is not None, unique_id
        return registry.async_get(entity_id)

    for platform, suffix in (
        ("binary_sensor", "occupancy"),
        ("binary_sensor", "motion"),
        ("sensor", "activity"),
        ("sensor", "confidence"),
        ("sensor", "dwell"),
        ("event", "pass_by"),
    ):
        entity = registry_entry(platform, f"{entry.entry_id}_zone_sofakrok_{suffix}")
        assert entity.disabled_by is er.RegistryEntryDisabler.INTEGRATION, entity.entity_id
        assert hass.states.get(entity.entity_id) is None  # not added while disabled

    # The record-baseline button stays enabled: calibration is a first-class
    # operator action, not a diagnostic.
    button = registry_entry("button", f"{entry.entry_id}_zone_sofakrok_record_baseline")
    assert button.disabled_by is None
    assert hass.states.get(button.entity_id) is not None
    calibration = registry_entry("sensor", f"{entry.entry_id}_zone_sofakrok_calibration_status")
    assert calibration.disabled_by is None
    assert hass.states.get(calibration.entity_id) is not None


async def test_room_and_home_entities_stay_enabled(hass: HomeAssistant, monkeypatch) -> None:
    entry, _controller, _fake = await setup_conductor(hass, monkeypatch)
    registry = er.async_get(hass)
    unique_ids = [
        f"{entry.entry_id}_room_stue_{suffix}"
        for suffix in ("occupancy", "motion", "settled", "activity", "confidence", "pass_by")
    ]
    unique_ids += [f"{entry.entry_id}_anyone_home", f"{entry.entry_id}_home_confidence"]
    for unique_id in unique_ids:
        entries = [
            registry.async_get(entity_id)
            for platform in ("binary_sensor", "sensor", "event")
            if (entity_id := registry.async_get_entity_id(platform, DOMAIN, unique_id))
        ]
        assert entries, unique_id
        for entity in entries:
            assert entity.disabled_by is None, entity.entity_id
            assert hass.states.get(entity.entity_id) is not None, entity.entity_id
