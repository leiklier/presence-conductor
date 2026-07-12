"""Presence Conductor integration setup.

Scaffold only: the entry loads and unloads, nothing more. The controller
adapter and the entity platforms that publish engine state (zone occupancy,
motion, activity, room fusion, anyone-home — docs/ENGINE_SPEC.md §0) land in
subsequent PRs.
"""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Presence Conductor from a config entry."""
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return True
