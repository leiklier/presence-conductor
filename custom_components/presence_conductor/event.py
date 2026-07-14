"""Event entities: pass-by (spec rule 5.2) and calibration outcomes (3.3).

A zone traversed without dwelling emits ``pass_by`` with the zone's peak
confidence and traversal duration. The same payload is fired on the HA bus
as ``presence_conductor_pass_by`` for automations; the entities are the
dashboard/logbook surface. Each room carries one aggregating entity — it
fires for every member zone's pass-by, carrying the zone id — and each zone
keeps its own, opt-in like the rest of the zone surface.

Each zone also carries a **calibration outcome** entity (rule 3.3): every
RecordBaseline window close fires ``recorded`` or ``rejected`` with the
per-path coverage verdicts — the button must never look successful when a
required channel was not calibrated, so this surface is enabled by default.
"""

from __future__ import annotations

from typing import ClassVar

from homeassistant.components.event import EventEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .calibration import baseline_payload
from .const import DOMAIN
from .controller import PresenceConductorController
from .core.model import ZoneConfig
from .core.plan import BaselineRecorded, PassBy
from .entity import ConductorEntity

EVENT_TYPE_PASS_BY = "pass_by"
EVENT_TYPE_BASELINE_RECORDED = "recorded"
EVENT_TYPE_BASELINE_REJECTED = "rejected"


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up pass-by event entities per zone/room and one calibration
    outcome entity per zone."""
    controller: PresenceConductorController | None = hass.data[DOMAIN][entry.entry_id]
    if controller is None:
        return
    entities: list[EventEntity] = [
        ZonePassByEvent(controller, zone) for zone in controller.config.zones
    ]
    entities += [ZoneBaselineEvent(controller, zone) for zone in controller.config.zones]
    entities += [RoomPassByEvent(controller, room_id) for room_id in controller.config.room_ids()]
    async_add_entities(entities)


class ZonePassByEvent(ConductorEntity, EventEntity):
    """A zone was traversed without dwelling (rule 5.2).

    Lives on the zone's room device; disabled by default — rooms and home
    are the consumer surface (spec §0), zone outputs are the estimator's
    internals, kept available as opt-in per-entity diagnostics.
    """

    _attr_event_types: ClassVar[list[str]] = [EVENT_TYPE_PASS_BY]
    _attr_translation_key = "pass_by"
    _attr_entity_registry_enabled_default = False

    def __init__(self, controller: PresenceConductorController, zone: ZoneConfig) -> None:
        super().__init__(controller, room_id=zone.room_id)
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
                "peak_confidence": round(event.peak_confidence, 4),
                "duration": round(event.duration, 2),
            },
        )
        self.async_write_ha_state()


class ZoneBaselineEvent(ConductorEntity, EventEntity):
    """The outcome of a RecordBaseline window for this zone (rule 3.3).

    Fires ``recorded`` on an atomic commit and ``rejected`` when a
    required path failed coverage (previous calibration kept, nothing
    persisted), with the per-path verdicts in the payload. Enabled by
    default, unlike the rest of the zone surface: calibration feedback is
    the whole point — the button must never look silently successful.
    """

    _attr_event_types: ClassVar[list[str]] = [
        EVENT_TYPE_BASELINE_RECORDED,
        EVENT_TYPE_BASELINE_REJECTED,
    ]
    _attr_translation_key = "baseline_outcome"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, controller: PresenceConductorController, zone: ZoneConfig) -> None:
        super().__init__(controller, room_id=zone.room_id)
        self._zone = zone
        self._attr_unique_id = f"{controller.entry.entry_id}_zone_{zone.zone_id}_baseline"
        self._attr_name = f"{zone.name} calibration"

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                self.controller.baseline_signal(self._zone.zone_id),
                self._on_baseline,
            )
        )

    @callback
    def _on_baseline(self, event: BaselineRecorded) -> None:
        payload = baseline_payload(event)
        payload.pop("zone_id")  # the entity IS the zone
        self._trigger_event(
            EVENT_TYPE_BASELINE_RECORDED if event.success else EVENT_TYPE_BASELINE_REJECTED,
            payload,
        )
        self.async_write_ha_state()


class RoomPassByEvent(ConductorEntity, EventEntity):
    """Any member zone of the room was traversed without dwelling (rule 5.2).

    The room-level consumer surface for pass-bys: fires for every member
    zone's pass-by (§6 membership) and carries the zone id, so a consumer
    watching one room device sees which slice was crossed.
    """

    _attr_event_types: ClassVar[list[str]] = [EVENT_TYPE_PASS_BY]
    _attr_translation_key = "room_pass_by"

    def __init__(self, controller: PresenceConductorController, room_id: str) -> None:
        super().__init__(controller, room_id=room_id)
        self._room_id = room_id
        self._attr_unique_id = f"{controller.entry.entry_id}_room_{room_id}_pass_by"
        self._attr_name = f"{controller.room_name(room_id)} room pass-by"

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                self.controller.room_pass_by_signal(self._room_id),
                self._on_pass_by,
            )
        )

    @callback
    def _on_pass_by(self, event: PassBy) -> None:
        self._trigger_event(
            EVENT_TYPE_PASS_BY,
            {
                "zone_id": event.zone_id,
                "peak_confidence": round(event.peak_confidence, 4),
                "duration": round(event.duration, 2),
            },
        )
        self.async_write_ha_state()
