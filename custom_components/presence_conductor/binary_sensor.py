"""Binary outputs: zone occupancy/motion, room occupancy/motion/settled,
anyone home.

All values are read straight from the engine's published state (spec §0);
nothing is re-derived here. Zone entities report unavailable while their
zone's health is UNKNOWN (rule 1.3: outputs hold, confidence is stale —
unavailable is the honest HA mapping). Room and home entities report
unavailable when fusion publishes ``None`` (rules 6.3, 6.5): "blind" must
stay distinguishable from "nobody there".
"""

from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DOMAIN
from .controller import PresenceConductorController
from .core.model import Health, RoomState, ZoneConfig, ZoneState
from .entity import ConductorEntity


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up the binary presence outputs."""
    controller: PresenceConductorController | None = hass.data[DOMAIN][entry.entry_id]
    if controller is None:
        return
    entities: list[BinarySensorEntity] = []
    for zone in controller.config.zones:
        entities.append(ZoneOccupancySensor(controller, zone))
        entities.append(ZoneMotionSensor(controller, zone))
    for room_id in controller.config.room_ids():
        entities.append(RoomOccupancySensor(controller, room_id))
        entities.append(RoomMotionSensor(controller, room_id))
        entities.append(RoomSettledSensor(controller, room_id))
    entities.append(AnyoneHomeSensor(controller))
    async_add_entities(entities)


class ZoneBinarySensor(ConductorEntity, BinarySensorEntity):
    """Base for per-zone binaries: health-gated availability (rule 1.3).

    Lives on the zone's room device; disabled by default — rooms and home
    are the consumer surface (spec §0), zone outputs are the estimator's
    internals, kept available as opt-in per-entity diagnostics.
    """

    _attr_entity_registry_enabled_default = False

    def __init__(self, controller: PresenceConductorController, zone: ZoneConfig) -> None:
        super().__init__(controller, room_id=zone.room_id)
        self._zone = zone

    @property
    def zone_state(self) -> ZoneState:
        return self.engine_state.zones[self._zone.zone_id]

    @property
    def available(self) -> bool:
        return self.zone_state.health is Health.OK

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {"zone_id": self._zone.zone_id, "room": self._zone.room_id}


class ZoneOccupancySensor(ZoneBinarySensor):
    """Robust per-zone occupancy (rules 4.3, 5.3: includes PASSING)."""

    _attr_device_class = BinarySensorDeviceClass.OCCUPANCY
    _attr_translation_key = "zone_occupancy"

    def __init__(self, controller: PresenceConductorController, zone: ZoneConfig) -> None:
        super().__init__(controller, zone)
        self._attr_unique_id = f"{controller.entry.entry_id}_zone_{zone.zone_id}_occupancy"
        self._attr_name = f"{zone.name} occupancy"

    @property
    def is_on(self) -> bool:
        return self.zone_state.occupied


class ZoneMotionSensor(ZoneBinarySensor):
    """The gated, undamped fast channel (rule 4.4): flicker by design."""

    _attr_device_class = BinarySensorDeviceClass.MOTION
    _attr_translation_key = "zone_motion"

    def __init__(self, controller: PresenceConductorController, zone: ZoneConfig) -> None:
        super().__init__(controller, zone)
        self._attr_unique_id = f"{controller.entry.entry_id}_zone_{zone.zone_id}_motion"
        self._attr_name = f"{zone.name} motion"

    @property
    def is_on(self) -> bool:
        return self.zone_state.motion


class RoomBinarySensor(ConductorEntity, BinarySensorEntity):
    """Base for per-room binaries: unavailable while fusion is blind (6.3)."""

    def __init__(self, controller: PresenceConductorController, room_id: str) -> None:
        super().__init__(controller, room_id=room_id)
        self._room_id = room_id

    @property
    def room_state(self) -> RoomState:
        return self.engine_state.rooms[self._room_id]

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {
            "room_id": self._room_id,
            "zones": [z.zone_id for z in self.controller.config.zones_in_room(self._room_id)],
        }


class RoomOccupancySensor(RoomBinarySensor):
    """Any healthy member zone occupied (rule 6.1)."""

    _attr_device_class = BinarySensorDeviceClass.OCCUPANCY
    _attr_translation_key = "room_occupancy"

    def __init__(self, controller: PresenceConductorController, room_id: str) -> None:
        super().__init__(controller, room_id)
        self._attr_unique_id = f"{controller.entry.entry_id}_room_{room_id}_occupancy"
        self._attr_name = f"{controller.room_name(room_id)} room occupancy"

    @property
    def available(self) -> bool:
        return self.room_state.occupied is not None

    @property
    def is_on(self) -> bool | None:
        return self.room_state.occupied


class RoomMotionSensor(RoomBinarySensor):
    """Any healthy member zone's motion channel (rule 6.2): flicker by design."""

    _attr_device_class = BinarySensorDeviceClass.MOTION
    _attr_translation_key = "room_motion"

    def __init__(self, controller: PresenceConductorController, room_id: str) -> None:
        super().__init__(controller, room_id)
        self._attr_unique_id = f"{controller.entry.entry_id}_room_{room_id}_motion"
        self._attr_name = f"{controller.room_name(room_id)} room motion"

    @property
    def available(self) -> bool:
        return self.room_state.motion is not None

    @property
    def is_on(self) -> bool | None:
        return self.room_state.motion


class RoomSettledSensor(RoomBinarySensor):
    """Any member zone SETTLED (rule 6.2) — the audio-zone-grade signal."""

    _attr_translation_key = "room_settled"

    def __init__(self, controller: PresenceConductorController, room_id: str) -> None:
        super().__init__(controller, room_id)
        self._attr_unique_id = f"{controller.entry.entry_id}_room_{room_id}_settled"
        self._attr_name = f"{controller.room_name(room_id)} room settled"

    @property
    def available(self) -> bool:
        return self.room_state.settled is not None

    @property
    def is_on(self) -> bool | None:
        return self.room_state.settled


class AnyoneHomeSensor(ConductorEntity, BinarySensorEntity):
    """Home-level presence (rule 6.5)."""

    _attr_device_class = BinarySensorDeviceClass.PRESENCE
    _attr_translation_key = "anyone_home"
    _attr_name = "Anyone home"

    def __init__(self, controller: PresenceConductorController) -> None:
        super().__init__(controller)
        self._attr_unique_id = f"{controller.entry.entry_id}_anyone_home"

    @property
    def available(self) -> bool:
        return self.engine_state.anyone_home is not None

    @property
    def is_on(self) -> bool | None:
        return self.engine_state.anyone_home
