"""Shared Home Assistant device and dispatcher-driven entity plumbing."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import EntityPlatform
from homeassistant.util import slugify

from .const import DOMAIN
from .core.model import EngineState

if TYPE_CHECKING:
    from .controller import PresenceConductorController


def conductor_device_info(entry: ConfigEntry) -> DeviceInfo:
    """Return the hub device used by home-level controls and diagnostics."""
    return DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name="Presence Conductor",
        manufacturer="Presence Conductor",
        model="Calibrated mmWave occupancy estimator",
        entry_type=DeviceEntryType.SERVICE,
    )


def room_device_info(controller: PresenceConductorController, room_id: str) -> DeviceInfo:
    """Return the room device containing room and member-zone signals."""
    room_name = controller.room_name(room_id)
    return DeviceInfo(
        identifiers={(DOMAIN, f"{controller.entry.entry_id}_room_{room_id}")},
        name=f"{room_name} presence",
        manufacturer="Presence Conductor",
        model="Calibrated mmWave occupancy estimator",
        suggested_area=room_name,
        via_device=(DOMAIN, controller.entry.entry_id),
        entry_type=DeviceEntryType.SERVICE,
    )


class ConductorEntity(Entity):
    """Base for conductor entities: dispatcher-driven, never polled."""

    _attr_should_poll = False
    _attr_has_entity_name = True
    # Controls and diagnostics override this so they remain honest while
    # ordinary estimator outputs are suppressed (ENGINE_SPEC rule 7.2).
    _control_surface = False

    def __init__(self, controller: PresenceConductorController, room_id: str | None = None) -> None:
        self.controller = controller
        self._attr_device_info = (
            conductor_device_info(controller.entry)
            if room_id is None
            else room_device_info(controller, room_id)
        )

    @callback
    def add_to_platform_start(
        self,
        hass: HomeAssistant,
        platform: EntityPlatform,
        parallel_updates: asyncio.Semaphore | None,
    ) -> None:
        """Pin the language-stable ``presence_conductor_*`` object id."""
        if self.entity_id is None:
            self.entity_id = f"{platform.domain}.{slugify(f'Presence Conductor {self._attr_name}')}"
        super().add_to_platform_start(hass, platform, parallel_updates)

    @property
    def engine_state(self) -> EngineState:
        return self.controller.engine.state

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        signal = self.controller.signal_control if self._control_surface else self.controller.signal
        self.async_on_remove(
            async_dispatcher_connect(self.hass, signal, self._on_controller_update)
        )

    @callback
    def _on_controller_update(self) -> None:
        self.async_write_ha_state()
