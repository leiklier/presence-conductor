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
from homeassistant.helpers import device_registry as dr

from .config import build_config
from .const import CONF_BASELINES, CONF_SENSORS, DOMAIN
from .controller import PresenceConductorController
from .core.engine import ConductorEngine
from .core.events import RecordBaseline
from .entity import conductor_device_info, room_device_info
from .observation import build_initial_snapshot

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
    if controller is not None:
        _async_register_devices(hass, entry, controller)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    if controller is not None:
        controller.async_start()
    return True


@callback
def _async_register_devices(
    hass: HomeAssistant, entry: ConfigEntry, controller: PresenceConductorController
) -> None:
    """Register the hub and one device per configured room.

    Done before platform forwarding so the room devices' ``via_device``
    reference always resolves, regardless of the order entities register
    in. Devices of rooms that no longer exist are released from the entry.
    """
    registry = dr.async_get(hass)
    registry.async_get_or_create(config_entry_id=entry.entry_id, **conductor_device_info(entry))
    expected = {(DOMAIN, entry.entry_id)}
    for room_id in controller.config.room_ids():
        info = room_device_info(controller, room_id)
        registry.async_get_or_create(config_entry_id=entry.entry_id, **info)
        expected |= info["identifiers"]
    for device in dr.async_entries_for_config_entry(registry, entry.entry_id):
        if not device.identifiers & expected:
            registry.async_update_device(device.id, remove_config_entry_id=entry.entry_id)


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
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        if controller is not None:
            await controller.async_stop()
            controller.clear_calibration_issue()
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
