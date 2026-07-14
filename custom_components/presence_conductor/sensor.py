"""Sensor outputs: activity enums, confidences, dwell, and diagnostics.

Activity is the consumer-facing split (rule 5.3): automations that must not
react to walk-throughs key on ``active``/``settled`` instead of occupancy.
Confidence and dwell are diagnostic surfaces; the diagnostics sensor
mirrors the whole engine state at a glance.
"""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DOMAIN
from .controller import (
    CalibrationStatus,
    ConductorEntity,
    PresenceConductorController,
)
from .core.model import Activity, Health, RoomState, ZoneConfig, ZoneState

ACTIVITY_OPTIONS = [activity.value for activity in Activity]
CALIBRATION_OPTIONS = [status.value for status in CalibrationStatus]


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up the sensor outputs."""
    controller: PresenceConductorController | None = hass.data[DOMAIN][entry.entry_id]
    if controller is None:
        return
    entities: list[SensorEntity] = []
    for zone in controller.config.zones:
        entities.append(ZoneActivitySensor(controller, zone))
        entities.append(ZoneConfidenceSensor(controller, zone))
        entities.append(ZoneDwellSensor(controller, zone))
        entities.append(ZoneCalibrationStatusSensor(controller, zone))
    for room_id in controller.config.room_ids():
        entities.append(RoomActivitySensor(controller, room_id))
        entities.append(RoomConfidenceSensor(controller, room_id))
    entities.append(HomeConfidenceSensor(controller))
    entities.append(ConductorStateSensor(controller))
    async_add_entities(entities)


class ZoneSensor(ConductorEntity, SensorEntity):
    """Base for per-zone sensors: health-gated availability (rule 1.3).

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


class ZoneActivitySensor(ZoneSensor):
    """The per-zone FSM state (rule 5.1)."""

    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = ACTIVITY_OPTIONS
    _attr_translation_key = "zone_activity"

    def __init__(self, controller: PresenceConductorController, zone: ZoneConfig) -> None:
        super().__init__(controller, zone)
        self._attr_unique_id = f"{controller.entry.entry_id}_zone_{zone.zone_id}_activity"
        self._attr_name = f"{zone.name} activity"

    @property
    def native_value(self) -> str:
        return self.zone_state.activity.value


class ZoneConfidenceSensor(ZoneSensor):
    """Zone occupancy confidence (§0), as a percentage — a monotone
    score, not a calibrated probability (rule 8.7)."""

    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_suggested_display_precision = 1
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "zone_confidence"

    def __init__(self, controller: PresenceConductorController, zone: ZoneConfig) -> None:
        super().__init__(controller, zone)
        self._attr_unique_id = f"{controller.entry.entry_id}_zone_{zone.zone_id}_confidence"
        self._attr_name = f"{zone.name} confidence"

    @property
    def native_value(self) -> float:
        return round(self.zone_state.confidence * 100.0, 2)


class ZoneDwellSensor(ZoneSensor):
    """Continuous occupancy of the zone, in seconds (rule 5.4)."""

    _attr_device_class = SensorDeviceClass.DURATION
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfTime.SECONDS
    _attr_suggested_display_precision = 0
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "zone_dwell"

    def __init__(self, controller: PresenceConductorController, zone: ZoneConfig) -> None:
        super().__init__(controller, zone)
        self._attr_unique_id = f"{controller.entry.entry_id}_zone_{zone.zone_id}_dwell"
        self._attr_name = f"{zone.name} dwell"

    @property
    def native_value(self) -> float:
        return round(self.zone_state.dwell_seconds, 1)


class ZoneCalibrationStatusSensor(ConductorEntity, SensorEntity):
    """Always-visible calibration provenance/readiness for one zone."""

    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = CALIBRATION_OPTIONS
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "calibration_status"
    _control_surface = True

    def __init__(self, controller: PresenceConductorController, zone: ZoneConfig) -> None:
        super().__init__(controller, room_id=zone.room_id)
        self._zone = zone
        self._attr_unique_id = f"{controller.entry.entry_id}_zone_{zone.zone_id}_calibration_status"
        self._attr_name = f"{zone.name} calibration status"

    @property
    def native_value(self) -> str:
        return self.controller.calibration_diagnostic(self._zone.zone_id).status.value

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {
            "zone_id": self._zone.zone_id,
            "room": self._zone.room_id,
            **self.controller.calibration_diagnostic(self._zone.zone_id).attributes(),
        }


class RoomSensor(ConductorEntity, SensorEntity):
    """Base for per-room sensors: unavailable while fusion is blind (6.3)."""

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


class RoomActivitySensor(RoomSensor):
    """Maximum-severity member activity (rule 6.2)."""

    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = ACTIVITY_OPTIONS
    _attr_translation_key = "room_activity"

    def __init__(self, controller: PresenceConductorController, room_id: str) -> None:
        super().__init__(controller, room_id)
        self._attr_unique_id = f"{controller.entry.entry_id}_room_{room_id}_activity"
        self._attr_name = f"{controller.room_name(room_id)} room activity"

    @property
    def available(self) -> bool:
        return self.room_state.activity is not None

    @property
    def native_value(self) -> str | None:
        activity = self.room_state.activity
        return activity.value if activity is not None else None


class RoomConfidenceSensor(RoomSensor):
    """Maximum member confidence (rule 6.1), as a percentage."""

    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_suggested_display_precision = 1
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "room_confidence"

    def __init__(self, controller: PresenceConductorController, room_id: str) -> None:
        super().__init__(controller, room_id)
        self._attr_unique_id = f"{controller.entry.entry_id}_room_{room_id}_confidence"
        self._attr_name = f"{controller.room_name(room_id)} room confidence"

    @property
    def available(self) -> bool:
        return self.room_state.confidence is not None

    @property
    def native_value(self) -> float | None:
        confidence = self.room_state.confidence
        return round(confidence * 100.0, 2) if confidence is not None else None


class HomeConfidenceSensor(ConductorEntity, SensorEntity):
    """Home-level occupancy confidence (rule 6.5), as a percentage."""

    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_suggested_display_precision = 1
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "home_confidence"
    _attr_name = "Home confidence"

    def __init__(self, controller: PresenceConductorController) -> None:
        super().__init__(controller)
        self._attr_unique_id = f"{controller.entry.entry_id}_home_confidence"

    @property
    def available(self) -> bool:
        return self.engine_state.home_confidence is not None

    @property
    def native_value(self) -> float | None:
        confidence = self.engine_state.home_confidence
        return round(confidence * 100.0, 2) if confidence is not None else None


class ConductorStateSensor(ConductorEntity, SensorEntity):
    """Engine state at a glance.

    A control surface: refreshed on every plan, even while outputs are
    suppressed (rule 7.2) — diagnosing a disabled engine must stay possible.
    """

    _attr_name = "State"
    _attr_translation_key = "state"
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _control_surface = True

    def __init__(self, controller: PresenceConductorController) -> None:
        super().__init__(controller)
        self._attr_unique_id = f"{controller.entry.entry_id}_state"

    @property
    def native_value(self) -> str:
        return "enabled" if self.engine_state.enabled else "disabled"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        state = self.engine_state
        return {
            "enabled": state.enabled,
            "home_lambda": round(state.lam_home, 4),
            "home_confidence": state.home_confidence,
            "anyone_home": state.anyone_home,
            "zones": {
                zone_id: {
                    "lambda": round(zst.lam, 4),
                    "confidence": round(zst.confidence, 4),
                    "health": zst.health.value,
                    "activity": zst.activity.value,
                    "occupied": zst.occupied,
                    "motion": zst.motion,
                    "dwell_seconds": round(zst.dwell_seconds, 1),
                    "move_baseline": [
                        round(zst.move_baseline.mu, 4),
                        round(zst.move_baseline.sigma, 4),
                    ],
                    "still_baseline": [
                        round(zst.still_baseline.mu, 4),
                        round(zst.still_baseline.sigma, 4),
                    ],
                    "calibration": {
                        "status": self.controller.calibration_diagnostic(zone_id).status.value,
                        **self.controller.calibration_diagnostic(zone_id).attributes(),
                    },
                }
                for zone_id, zst in state.zones.items()
            },
            "rooms": {
                room_id: {
                    "occupied": room.occupied,
                    "motion": room.motion,
                    "activity": room.activity.value if room.activity is not None else None,
                    "settled": room.settled,
                    "confidence": room.confidence,
                }
                for room_id, room in state.rooms.items()
            },
            "sensors": {
                sensor_id: {"available": sensor.available}
                for sensor_id, sensor in state.sensors.items()
            },
        }
