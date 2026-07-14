"""Per-zone baseline calibration buttons (spec rule 3.3).

Pressing a button opens a RecordBaseline window with the default duration
(``Tunables.baseline_duration``, 300 s); the operator asserts the zone is
empty for the window. The ``presence_conductor.record_baseline`` service is
the parameterized alternative (custom duration).
"""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import CALIBRATION_MODE_FULL, CONF_CALIBRATION_MODE, DOMAIN
from .controller import PresenceConductorController
from .core.events import (
    AdvanceFullCalibration,
    CancelCalibration,
    RecordBaseline,
    StartFullCalibration,
)
from .core.model import ZoneConfig
from .entity import ConductorEntity


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up one record-baseline button per zone."""
    controller: PresenceConductorController | None = hass.data[DOMAIN][entry.entry_id]
    if controller is None:
        return
    entities: list[ButtonEntity] = [
        entity
        for zone in controller.config.zones
        for entity in (
            ZoneRecordBaselineButton(controller, zone),
            ZoneCancelCalibrationButton(controller, zone),
        )
    ]
    if entry.options.get(CONF_CALIBRATION_MODE) == CALIBRATION_MODE_FULL:
        for zone in controller.config.zones:
            entities.extend(
                (
                    ZoneStartFullCalibrationButton(controller, zone),
                    ZoneNextCalibrationPhaseButton(controller, zone),
                )
            )
    async_add_entities(entities)


class ZoneRecordBaselineButton(ConductorEntity, ButtonEntity):
    """Record the empty-room noise floor of one zone (rule 3.3).

    Lives on the zone's room device and — unlike the zone state entities —
    stays enabled by default: calibration is a first-class operator action,
    not a diagnostic.
    """

    _attr_entity_category = EntityCategory.CONFIG
    _attr_translation_key = "record_baseline"

    def __init__(self, controller: PresenceConductorController, zone: ZoneConfig) -> None:
        super().__init__(controller, room_id=zone.room_id)
        self._zone = zone
        self._attr_unique_id = f"{controller.entry.entry_id}_zone_{zone.zone_id}_record_baseline"
        self._attr_name = f"{zone.name} record baseline"

    @property
    def available(self) -> bool:
        return all(
            self.controller.state.zones[sibling.zone_id].recording is None
            and self.controller.state.zones[sibling.zone_id].guided_calibration is None
            for sibling in self.controller.config.zones_for_sensor(self._zone.sensor_id)
        )

    async def async_press(self) -> None:
        self.controller.submit(RecordBaseline(self._zone.zone_id))


class _GuidedCalibrationButton(ConductorEntity, ButtonEntity):
    _attr_entity_category = EntityCategory.CONFIG
    _control_surface = True

    def __init__(self, controller: PresenceConductorController, zone: ZoneConfig) -> None:
        super().__init__(controller, room_id=zone.room_id)
        self._zone = zone

    @property
    def zone_state(self):
        return self.controller.state.zones[self._zone.zone_id]


class ZoneStartFullCalibrationButton(_GuidedCalibrationButton):
    """Start the empty-baseline stage of the guided full workflow."""

    _attr_translation_key = "start_full_calibration"

    def __init__(self, controller: PresenceConductorController, zone: ZoneConfig) -> None:
        super().__init__(controller, zone)
        self._attr_unique_id = f"{controller.entry.entry_id}_{zone.zone_id}_start_full_calibration"
        self._attr_name = f"{zone.name} start full calibration"

    @property
    def available(self) -> bool:
        return self.controller.state.sensors[self._zone.sensor_id].available and all(
            self.controller.state.zones[sibling.zone_id].recording is None
            and self.controller.state.zones[sibling.zone_id].guided_calibration is None
            for sibling in self.controller.config.zones_for_sensor(self._zone.sensor_id)
        )

    async def async_press(self) -> None:
        self.controller.submit(StartFullCalibration(self._zone.zone_id))


class ZoneNextCalibrationPhaseButton(_GuidedCalibrationButton):
    """Start the next phase after the operator follows the status instruction."""

    _attr_translation_key = "next_calibration_phase"

    def __init__(self, controller: PresenceConductorController, zone: ZoneConfig) -> None:
        super().__init__(controller, zone)
        self._attr_unique_id = f"{controller.entry.entry_id}_{zone.zone_id}_next_calibration_phase"
        self._attr_name = f"{zone.name} record next calibration phase"

    @property
    def available(self) -> bool:
        session = self.zone_state.guided_calibration
        return session is not None and session.status == "waiting"

    async def async_press(self) -> None:
        self.controller.submit(AdvanceFullCalibration(self._zone.zone_id))


class ZoneCancelCalibrationButton(_GuidedCalibrationButton):
    """Cancel the active baseline or full session."""

    _attr_translation_key = "cancel_calibration"

    def __init__(self, controller: PresenceConductorController, zone: ZoneConfig) -> None:
        super().__init__(controller, zone)
        self._attr_unique_id = f"{controller.entry.entry_id}_{zone.zone_id}_cancel_calibration"
        self._attr_name = f"{zone.name} cancel calibration"

    @property
    def available(self) -> bool:
        return (
            self.zone_state.recording is not None or self.zone_state.guided_calibration is not None
        )

    async def async_press(self) -> None:
        self.controller.submit(CancelCalibration(self._zone.zone_id))
