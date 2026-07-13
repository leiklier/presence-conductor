"""Registry-driven discovery of candidate mmWave sensor devices.

Pure read-only helpers over the entity/device/area registries. The config
flow uses these to prefill its forms with installation-specific defaults;
nothing in here mutates anything.

A candidate device is one that exposes the LD2410-style entity cluster
(Apollo MSR-2 / ESPHome ld2410 naming, e.g.
``sensor.apollo_msr_2_f79794_radar_move_energy``). Matching is per device:
entities whose entity id or unique id ends with a known suffix are mapped to
roles, and the device qualifies if it carries both energy sensors and at
least one distance sensor. Users who name their entities differently assign
them manually in the config flow instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
)
from homeassistant.helpers import (
    device_registry as dr,
)
from homeassistant.helpers import (
    entity_registry as er,
)

from .const import (
    DISTANCE_ROLES,
    ROLE_DETECTION_DISTANCE,
    ROLE_MOVE_ENERGY,
    ROLE_MOVING_DISTANCE,
    ROLE_MOVING_TARGET,
    ROLE_STILL_DISTANCE,
    ROLE_STILL_ENERGY,
    ROLE_STILL_TARGET,
    ROLE_TARGET,
)

#: Entity-id / unique-id suffix -> (entity domain, role). This is the Apollo
#: MSR-2 / ESPHome ld2410 naming; other firmwares extend this table with
#: their own suffixes.
SUFFIX_ROLES: dict[str, tuple[str, str]] = {
    "_radar_move_energy": ("sensor", ROLE_MOVE_ENERGY),
    "_radar_still_energy": ("sensor", ROLE_STILL_ENERGY),
    "_radar_moving_distance": ("sensor", ROLE_MOVING_DISTANCE),
    "_radar_still_distance": ("sensor", ROLE_STILL_DISTANCE),
    "_radar_detection_distance": ("sensor", ROLE_DETECTION_DISTANCE),
    "_radar_target": ("binary_sensor", ROLE_TARGET),
    "_radar_moving_target": ("binary_sensor", ROLE_MOVING_TARGET),
    "_radar_still_target": ("binary_sensor", ROLE_STILL_TARGET),
}

#: Longest suffix first, so e.g. ``_radar_moving_target`` can never be
#: shadowed by a shorter suffix another firmware might add.
_SUFFIXES_BY_LENGTH: tuple[tuple[str, tuple[str, str]], ...] = tuple(
    sorted(SUFFIX_ROLES.items(), key=lambda item: len(item[0]), reverse=True)
)


@dataclass(frozen=True, slots=True)
class DiscoveredSensor:
    """A qualifying mmWave device found in the registries."""

    device_id: str
    name: str
    area_id: str | None
    area_name: str | None
    #: role -> entity_id for every matched entity of the device.
    entities: dict[str, str] = field(default_factory=dict)


def _usable(entry: er.RegistryEntry) -> bool:
    return entry.disabled_by is None and entry.hidden_by is None


def _match_role(entry: er.RegistryEntry) -> str | None:
    """The role of a registry entry, by entity-id or unique-id suffix."""
    for suffix, (domain, role) in _SUFFIXES_BY_LENGTH:
        if entry.domain != domain:
            continue
        if entry.entity_id.endswith(suffix) or (entry.unique_id or "").endswith(suffix):
            return role
    return None


def discover_sensors(hass: HomeAssistant) -> list[DiscoveredSensor]:
    """All qualifying mmWave devices, sorted by name.

    A device qualifies if its matched entities cover both energy roles and at
    least one distance role (the minimum the estimator can gate on, rule 2.1).
    """
    ent_reg = er.async_get(hass)
    dev_reg = dr.async_get(hass)
    area_reg = ar.async_get(hass)

    by_device: dict[str, dict[str, str]] = {}
    for entry in list(ent_reg.entities.values()):
        if entry.device_id is None or not _usable(entry):
            continue
        role = _match_role(entry)
        if role is not None:
            by_device.setdefault(entry.device_id, {}).setdefault(role, entry.entity_id)

    sensors: list[DiscoveredSensor] = []
    for device_id, roles in by_device.items():
        if ROLE_MOVE_ENERGY not in roles or ROLE_STILL_ENERGY not in roles:
            continue
        if not any(role in roles for role in DISTANCE_ROLES):
            continue
        device = dev_reg.async_get(device_id)
        if device is None:
            continue
        area = area_reg.async_get_area(device.area_id) if device.area_id else None
        sensors.append(
            DiscoveredSensor(
                device_id=device_id,
                name=device.name_by_user or device.name or device_id,
                area_id=device.area_id,
                area_name=area.name if area else None,
                entities=roles,
            )
        )
    sensors.sort(key=lambda sensor: sensor.name)
    return sensors


def area_names(hass: HomeAssistant) -> list[str]:
    """All area names, sorted — room-name suggestions for the zone forms."""
    return sorted(area.name for area in ar.async_get(hass).async_list_areas())
