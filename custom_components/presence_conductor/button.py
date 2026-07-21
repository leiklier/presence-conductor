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

from .const import DOMAIN
from .controller import PresenceConductorController
from .core.events import RecordBaseline
from .core.model import ZoneConfig
from .entity import ConductorEntity


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up one record-baseline button per zone."""
    controller: PresenceConductorController | None = hass.data[DOMAIN][entry.entry_id]
    if controller is None:
        return
    async_add_entities(
        ZoneRecordBaselineButton(controller, zone) for zone in controller.config.zones
    )


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

    async def async_press(self) -> None:
        self.controller.submit(RecordBaseline(self._zone.zone_id))
