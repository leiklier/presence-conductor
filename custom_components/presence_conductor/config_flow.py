"""Config flow for Presence Conductor.

Single-instance flow with no fields yet: the scaffold only creates the
singleton entry. The real setup wizard — mapping sensors to zones and rooms
per docs/ENGINE_SPEC.md §0 — lands together with the engine in a later PR.
"""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigFlow, ConfigFlowResult

from .const import DOMAIN


class PresenceConductorConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the (single-instance) setup flow."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Confirm and create the singleton entry."""
        await self.async_set_unique_id(DOMAIN)
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")
        if user_input is None:
            return self.async_show_form(step_id="user", data_schema=vol.Schema({}))
        return self.async_create_entry(title="Presence Conductor", data={})
