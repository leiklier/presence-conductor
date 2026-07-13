"""Per-zone pass-by event entities (spec rule 5.2).

A zone traversed without dwelling emits ``pass_by`` with the zone's peak
probability and traversal duration. The same payload is fired on the HA bus
as ``presence_conductor_pass_by`` for automations; the entity is the
dashboard/logbook surface.
"""

from __future__ import annotations

from typing import ClassVar

from homeassistant.components.event import EventEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DOMAIN
from .controller import ConductorEntity, PresenceConductorController
from .core.model import ZoneConfig
from .core.plan import PassBy

EVENT_TYPE_PASS_BY = "pass_by"


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up one pass-by event entity per zone."""
    controller: PresenceConductorController | None = hass.data[DOMAIN][entry.entry_id]
    if controller is None:
        return
    async_add_entities(ZonePassByEvent(controller, zone) for zone in controller.config.zones)


class ZonePassByEvent(ConductorEntity, EventEntity):
    """A zone was traversed without dwelling (rule 5.2)."""

    _attr_event_types: ClassVar[list[str]] = [EVENT_TYPE_PASS_BY]
    _attr_translation_key = "pass_by"

    def __init__(self, controller: PresenceConductorController, zone: ZoneConfig) -> None:
        super().__init__(controller)
        self._zone = zone
        self._attr_unique_id = f"{controller.entry.entry_id}_zone_{zone.zone_id}_pass_by"
        self._attr_name = f"{zone.name} pass-by"

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                self.controller.pass_by_signal(self._zone.zone_id),
                self._on_pass_by,
            )
        )

    @callback
    def _on_pass_by(self, event: PassBy) -> None:
        self._trigger_event(
            EVENT_TYPE_PASS_BY,
            {
                "peak_probability": round(event.peak_probability, 4),
                "duration": round(event.duration, 2),
            },
        )
        self.async_write_ha_state()
