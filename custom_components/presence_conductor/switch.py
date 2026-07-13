"""The global enabled switch (spec rule 7.2).

While off, the engine keeps ingesting frames and updating state (re-enable
is warm) but publishes no transitions and emits no events. The switch itself
is a control surface: it refreshes on the always-sent control signal, so it
never freezes with the outputs it suppresses.
"""

from __future__ import annotations

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .const import DOMAIN
from .controller import ConductorEntity, PresenceConductorController
from .core.events import SetEnabled


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up the enabled switch."""
    controller: PresenceConductorController | None = hass.data[DOMAIN][entry.entry_id]
    if controller is None:
        return
    async_add_entities([ConductorEnabledSwitch(controller)])


class ConductorEnabledSwitch(ConductorEntity, SwitchEntity, RestoreEntity):
    """The engine's enabled flag, restored across restarts.

    The engine seeds ``enabled=True`` (its model default); a differing
    restored state is pushed back through the controller as ``SetEnabled``,
    like any user command. Unknown/unavailable restored states leave the
    engine default untouched.
    """

    _attr_name = "Enabled"
    _attr_translation_key = "enabled"
    _attr_entity_category = EntityCategory.CONFIG
    _control_surface = True

    def __init__(self, controller: PresenceConductorController) -> None:
        super().__init__(controller)
        self._attr_unique_id = f"{controller.entry.entry_id}_enabled"

    @property
    def is_on(self) -> bool:
        return self.engine_state.enabled

    async def async_turn_on(self, **kwargs: object) -> None:
        self.controller.submit(SetEnabled(True))

    async def async_turn_off(self, **kwargs: object) -> None:
        self.controller.submit(SetEnabled(False))

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last is None or last.state not in (STATE_ON, STATE_OFF):
            return  # nothing restored (or invalid): keep the engine default
        enabled = last.state == STATE_ON
        if enabled != self.engine_state.enabled:
            self.controller.submit(SetEnabled(enabled))
