"""Sensor outputs: activity enums, confidences, dwell, and diagnostics.

Activity is the consumer-facing split (rule 5.3): automations that must not
react to walk-throughs key on ``active``/``settled`` instead of occupancy.
Confidence and dwell are diagnostic surfaces; the state sensor keeps a
discrete engine summary at a glance.

Recorder discipline: entity states and attributes only carry values that
change at natural intervals — discrete transitions, bucketed confidence
behind a publish interval, coarse dwell buckets. Per-frame numerics
(lambdas, baselines, runtime evidence paths) live in the diagnostics
download (``diagnostics.py``), never on entities, because every attribute
change writes a recorder row regardless of recorder-side attribute
exclusion.
"""

from __future__ import annotations

import math
import time
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, UnitOfTime
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .calibration import CalibrationStatus
from .const import DOMAIN
from .controller import PresenceConductorController
from .core.model import Activity, Health, RoomState, ZoneConfig, ZoneState
from .entity import ConductorEntity

ACTIVITY_OPTIONS = [activity.value for activity in Activity]
CALIBRATION_OPTIONS = [status.value for status in CalibrationStatus]

#: Confidence quantization step, in percent points. Whole-percent steps
#: still wrote a recorder row per frame during belief sweeps (measured
#: live: 47 -> 0 in five seconds, one row per percent); a sweep now
#: crosses ~20 buckets, and the interval below collapses that to 1-2
#: rows. The 5-minute long-term statistics keep the fine trend.
CONFIDENCE_STEP = 5
#: Minimum seconds between confidence publishes per sensor. Suppressed
#: values are not lost: the 1 Hz tick keeps dispatching, so the settled
#: value lands at most one interval late. Availability transitions
#: bypass the interval — "blind" must surface immediately (rule 6.3).
CONFIDENCE_PUBLISH_INTERVAL = 10.0

#: Monotonic clock hook; tests patch this to script the interval.
_monotonic = time.monotonic


def quantized_confidence(confidence: float) -> int:
    """Confidence as a percentage in :data:`CONFIDENCE_STEP` buckets."""
    return CONFIDENCE_STEP * round(confidence * 100.0 / CONFIDENCE_STEP)


class ConfidencePublishGate:
    """Mixin for confidence sensors: a rate-limited diagnostic trend.

    Confidence is a monotone score wired to per-frame belief, so raw
    publishes follow the radar's cadence no matter how the value is
    quantized. This gate republishes only when the bucketed value (or
    availability) actually changed, and value-only changes at most once
    per :data:`CONFIDENCE_PUBLISH_INTERVAL`.
    """

    _published: tuple[bool, int | None] | None = None
    _published_at: float | None = None

    @callback
    def _on_controller_update(self) -> None:
        now = _monotonic()
        snapshot = (self.available, self.native_value if self.available else None)
        if self._published is not None:
            if snapshot == self._published:
                return  # unchanged — skip even the state_reported write
            if (
                snapshot[0] == self._published[0]
                and self._published_at is not None
                and now - self._published_at < CONFIDENCE_PUBLISH_INTERVAL
            ):
                return  # inside the interval; a later tick lands the value
        self._published = snapshot
        self._published_at = now
        super()._on_controller_update()


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


class ZoneConfidenceSensor(ConfidencePublishGate, ZoneSensor):
    """Zone occupancy confidence (§0), as a percentage — a monotone
    score, not a calibrated probability (rule 8.7). Bucketed and
    rate-limited by :class:`ConfidencePublishGate`; long-term
    MEASUREMENT statistics keep the 5-minute trend."""

    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_suggested_display_precision = 0
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "zone_confidence"

    def __init__(self, controller: PresenceConductorController, zone: ZoneConfig) -> None:
        super().__init__(controller, zone)
        self._attr_unique_id = f"{controller.entry.entry_id}_zone_{zone.zone_id}_confidence"
        self._attr_name = f"{zone.name} confidence"

    @property
    def native_value(self) -> int:
        return quantized_confidence(self.zone_state.confidence)


#: Dwell resolution in seconds: a per-tick counter would write a recorder
#: row every second while occupied; 10 s buckets keep dwell thresholds
#: useful (they fire at most one bucket late, never early — floor).
DWELL_RESOLUTION = 10


class ZoneDwellSensor(ZoneSensor):
    """Continuous occupancy of the zone, in seconds (rule 5.4),
    quantized to :data:`DWELL_RESOLUTION` buckets."""

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
    def native_value(self) -> int:
        return math.floor(self.zone_state.dwell_seconds / DWELL_RESOLUTION) * DWELL_RESOLUTION


class ZoneCalibrationStatusSensor(ConductorEntity, SensorEntity):
    """Always-visible calibration provenance/readiness for one zone.

    Attributes carry stable provenance only: the per-frame runtime
    evidence paths (gate/aggregate, empirical/analytic) flip with gate
    readiness and would write a recorder row per flip — they live in the
    diagnostics download instead.
    """

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
            **self.controller.calibration_diagnostic(self._zone.zone_id).provenance_attributes(),
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


class RoomConfidenceSensor(ConfidencePublishGate, RoomSensor):
    """Maximum member confidence (rule 6.1), bucketed and rate-limited."""

    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_suggested_display_precision = 0
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
    def native_value(self) -> int | None:
        confidence = self.room_state.confidence
        return quantized_confidence(confidence) if confidence is not None else None


class HomeConfidenceSensor(ConfidencePublishGate, ConductorEntity, SensorEntity):
    """Home-level occupancy confidence (rule 6.5), bucketed and
    rate-limited."""

    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_suggested_display_precision = 0
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
    def native_value(self) -> int | None:
        confidence = self.engine_state.home_confidence
        return quantized_confidence(confidence) if confidence is not None else None


class ConductorStateSensor(ConductorEntity, SensorEntity):
    """Engine state at a glance.

    A control surface: refreshed on every plan, even while outputs are
    suppressed (rule 7.2) — diagnosing a disabled engine must stay possible.

    Attributes are a discrete summary that only changes on real
    transitions; the per-frame numerics (lambdas, confidences, baselines,
    dwell) live in the diagnostics download. Even so, the summary
    duplicates dedicated entities, so it is kept out of the recorder —
    only the enabled/disabled state is stored.
    """

    _attr_name = "State"
    _attr_translation_key = "state"
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _control_surface = True
    _unrecorded_attributes = frozenset({"anyone_home", "zones", "rooms", "sensors"})

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
            "anyone_home": state.anyone_home,
            "zones": {
                zone_id: {
                    "health": zst.health.value,
                    "activity": zst.activity.value,
                    "occupied": zst.occupied,
                    "calibration": self.controller.calibration_diagnostic(zone_id).status.value,
                }
                for zone_id, zst in state.zones.items()
            },
            "rooms": {
                room_id: {
                    "occupied": room.occupied,
                    "activity": room.activity.value if room.activity is not None else None,
                    "settled": room.settled,
                }
                for room_id, room in state.rooms.items()
            },
            "sensors": {
                sensor_id: {"available": sensor.available}
                for sensor_id, sensor in state.sensors.items()
            },
        }
