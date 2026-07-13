"""Presence Conductor integration setup.

The entry builds one :class:`~.controller.PresenceConductorController`
around one :class:`~.core.engine.ConductorEngine`, forwards the entity
platforms that publish engine state (docs/ENGINE_SPEC.md §0), and registers
the ``record_baseline`` service (rule 3.3).
"""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv

from .config import build_config
from .const import CONF_BASELINES, CONF_SENSORS, DOMAIN
from .controller import PresenceConductorController, build_initial_snapshot
from .core.engine import ConductorEngine
from .core.events import RecordBaseline

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [
    Platform.BINARY_SENSOR,
    Platform.BUTTON,
    Platform.EVENT,
    Platform.SENSOR,
    Platform.SWITCH,
]

#: hass.data key holding, per entry_id, the options that warrant a reload.
DATA_RELOAD_BASELINE = f"{DOMAIN}_reload_baseline"

SERVICE_RECORD_BASELINE = "record_baseline"
ATTR_ZONE_ID = "zone_id"
ATTR_DURATION = "duration"

SERVICE_RECORD_BASELINE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ZONE_ID): cv.string,
        vol.Optional(ATTR_DURATION): vol.All(vol.Coerce(float), vol.Range(min=5, max=3600)),
    }
)


def _reload_relevant(options: Any) -> dict[str, Any]:
    """Options minus the keys the controller itself writes at runtime."""
    return {k: v for k, v in dict(options).items() if k != CONF_BASELINES}


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Presence Conductor from a config entry."""
    controller: PresenceConductorController | None = None
    if entry.options.get(CONF_SENSORS):
        config = build_config(entry.options)
        snapshot = build_initial_snapshot(hass, entry.options)
        controller = PresenceConductorController(
            hass, entry, config, snapshot, engine_factory=ConductorEngine
        )
    else:
        _LOGGER.debug("No sensors configured; loading %s without a controller", entry.entry_id)

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = controller
    hass.data.setdefault(DATA_RELOAD_BASELINE, {})[entry.entry_id] = _reload_relevant(entry.options)
    entry.async_on_unload(entry.add_update_listener(_async_options_updated))
    _async_register_services(hass)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    if controller is not None:
        controller.async_start()
    return True


async def _async_options_updated(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload on options changes — except controller-written baselines.

    The controller persists calibration into ``options["baselines"]`` (rule
    3.3); reloading on that write would cause a reload loop.
    """
    baseline = hass.data.get(DATA_RELOAD_BASELINE, {}).get(entry.entry_id)
    if baseline is not None and _reload_relevant(entry.options) == baseline:
        return
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    controller: PresenceConductorController | None = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if controller is not None:
        await controller.async_stop()
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
        hass.data.get(DATA_RELOAD_BASELINE, {}).pop(entry.entry_id, None)
        if not hass.data.get(DOMAIN):
            hass.services.async_remove(DOMAIN, SERVICE_RECORD_BASELINE)
    return unload_ok


@callback
def _async_register_services(hass: HomeAssistant) -> None:
    """Register the ``record_baseline`` service (rule 3.3) once."""
    if hass.services.has_service(DOMAIN, SERVICE_RECORD_BASELINE):
        return

    @callback
    def _handle_record_baseline(call: ServiceCall) -> None:
        zone_id: str = call.data[ATTR_ZONE_ID]
        duration: float | None = call.data.get(ATTR_DURATION)
        for controller in hass.data.get(DOMAIN, {}).values():
            if controller is not None and controller.config.zone_or_none(zone_id) is not None:
                controller.submit(RecordBaseline(zone_id, duration))
                return
        raise ServiceValidationError(
            f"Unknown zone '{zone_id}'. Use the zone's ID: the slug of its zone name."
        )

    hass.services.async_register(
        DOMAIN,
        SERVICE_RECORD_BASELINE,
        _handle_record_baseline,
        schema=SERVICE_RECORD_BASELINE_SCHEMA,
    )
