"""Tests for registry-driven mmWave device discovery.

The registries are staged to mirror the real installation: four Apollo MSR-2
devices (ESPHome ld2410 naming, ``sensor.apollo_msr_2_<mac>_radar_*``) in
Kjøkken, Kontor, Sofakrok and Spisebord, plus devices that must NOT be
discovered: incomplete clusters, disabled entities and unrelated platforms.
"""

from __future__ import annotations

from collections.abc import Iterable

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
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.presence_conductor import discovery
from custom_components.presence_conductor.const import (
    ALL_ROLES,
    ROLE_DETECTION_DISTANCE,
    ROLE_MOVE_ENERGY,
    ROLE_STILL_ENERGY,
    ROLE_TARGET,
)

#: role -> (entity domain, entity-id suffix), inverted from the module table.
ROLE_TO_SUFFIX: dict[str, tuple[str, str]] = {
    role: (domain, suffix) for suffix, (domain, role) in discovery.SUFFIX_ROLES.items()
}

KJOKKEN_PREFIX = "apollo_msr_2_29abc4"
KONTOR_PREFIX = "apollo_msr_2_f77c08"
SOFAKROK_PREFIX = "apollo_msr_2_f79794"
SPISEBORD_PREFIX = "apollo_msr_2_fadea8"

#: The Spisebord unit exposes a reduced cluster: energies + one distance
#: (detection) + one binary. Still qualifies.
SPISEBORD_ROLES: tuple[str, ...] = (
    ROLE_MOVE_ENERGY,
    ROLE_STILL_ENERGY,
    ROLE_DETECTION_DISTANCE,
    ROLE_TARGET,
)


def cluster_entities(prefix: str, roles: Iterable[str] = ALL_ROLES) -> dict[str, str]:
    """role -> entity_id for an Apollo-named cluster."""
    return {role: f"{ROLE_TO_SUFFIX[role][0]}.{prefix}{ROLE_TO_SUFFIX[role][1]}" for role in roles}


async def build_installation(hass: HomeAssistant) -> dict[str, str]:
    """Stage registries like the real installation. Returns device ids."""
    area_reg = ar.async_get(hass)
    dev_reg = dr.async_get(hass)
    ent_reg = er.async_get(hass)

    areas = {
        name: area_reg.async_create(name).id
        for name in ("Kjøkken", "Kontor", "Sofakrok", "Spisebord")
    }

    esphome = MockConfigEntry(domain="esphome")
    esphome.add_to_hass(hass)

    def add_cluster(
        identifier: str,
        device_name: str,
        prefix: str,
        area: str | None,
        roles: Iterable[str] = ALL_ROLES,
        disabled: bool = False,
    ) -> str:
        device = dev_reg.async_get_or_create(
            config_entry_id=esphome.entry_id,
            identifiers={("esphome", identifier)},
            name=device_name,
        )
        if area:
            dev_reg.async_update_device(device.id, area_id=areas[area])
        for role in roles:
            domain, suffix = ROLE_TO_SUFFIX[role]
            ent_reg.async_get_or_create(
                domain,
                "esphome",
                f"{identifier}{suffix}",
                suggested_object_id=f"{prefix}{suffix}",
                device_id=device.id,
                disabled_by=er.RegistryEntryDisabler.USER if disabled else None,
            )
        return device.id

    devices = {
        "kjokken": add_cluster("29abc4", "Apollo MSR-2 Kjøkken", KJOKKEN_PREFIX, "Kjøkken"),
        "kontor": add_cluster("f77c08", "Apollo MSR-2 Kontor", KONTOR_PREFIX, "Kontor"),
        "sofakrok": add_cluster("f79794", "Apollo MSR-2 Sofakrok", SOFAKROK_PREFIX, "Sofakrok"),
        "spisebord": add_cluster(
            "fadea8", "Apollo MSR-2 Spisebord", SPISEBORD_PREFIX, "Spisebord", SPISEBORD_ROLES
        ),
    }

    # One energy + one binary only: must NOT qualify.
    add_cluster(
        "bad", "Apollo MSR-2 Bad", "apollo_msr_2_bad", None, (ROLE_MOVE_ENERGY, ROLE_TARGET)
    )
    # Both energies but no distance: must NOT qualify.
    add_cluster(
        "bod", "Radar Bod", "radar_bod", None, (ROLE_MOVE_ENERGY, ROLE_STILL_ENERGY, ROLE_TARGET)
    )
    # A full cluster whose entities are all disabled: invisible.
    add_cluster(
        "vaskerom", "Apollo MSR-2 Vaskerom", "apollo_msr_2_vaskerom", None, ALL_ROLES, disabled=True
    )
    # Unrelated entities that must never be matched.
    ent_reg.async_get_or_create(
        "media_player", "sonos", "sonos_kjokken", suggested_object_id="kjokken_sonos_move"
    )
    hass.states.async_set("binary_sensor.kjokken_occupancy", "off")

    return devices


async def stage_renamed_device(hass: HomeAssistant) -> str:
    """A cluster whose entity ids were renamed; unique ids keep the suffixes."""
    dev_reg = dr.async_get(hass)
    ent_reg = er.async_get(hass)
    esphome = MockConfigEntry(domain="esphome")
    esphome.add_to_hass(hass)
    device = dev_reg.async_get_or_create(
        config_entry_id=esphome.entry_id,
        identifiers={("esphome", "gang")},
        name="Apollo MSR-2 Gang",
    )
    for role in (ROLE_MOVE_ENERGY, ROLE_STILL_ENERGY, "moving_distance", "still_distance"):
        domain, suffix = ROLE_TO_SUFFIX[role]
        ent_reg.async_get_or_create(
            domain,
            "esphome",
            f"gang{suffix}",  # unique_id keeps the ld2410 suffix
            suggested_object_id=f"gangen_{role}",  # entity_id does not
            device_id=device.id,
        )
    return device.id


async def test_discover_sensors(hass: HomeAssistant) -> None:
    """Qualifying devices only, sorted by name, with roles and areas mapped."""
    await build_installation(hass)
    sensors = discovery.discover_sensors(hass)

    assert [sensor.name for sensor in sensors] == [
        "Apollo MSR-2 Kjøkken",
        "Apollo MSR-2 Kontor",
        "Apollo MSR-2 Sofakrok",
        "Apollo MSR-2 Spisebord",
    ]

    kjokken, kontor, sofakrok, spisebord = sensors
    assert kjokken.area_name == "Kjøkken"
    assert kjokken.entities == cluster_entities(KJOKKEN_PREFIX)
    assert kontor.area_name == "Kontor"
    assert sofakrok.area_name == "Sofakrok"
    assert sofakrok.entities == cluster_entities(SOFAKROK_PREFIX)
    # Reduced cluster: qualifies through the detection distance.
    assert spisebord.area_name == "Spisebord"
    assert spisebord.entities == cluster_entities(SPISEBORD_PREFIX, SPISEBORD_ROLES)


async def test_incomplete_clusters_not_discovered(hass: HomeAssistant) -> None:
    """Missing an energy or every distance -> not a candidate."""
    await build_installation(hass)
    names = [sensor.name for sensor in discovery.discover_sensors(hass)]

    assert "Apollo MSR-2 Bad" not in names  # one energy only
    assert "Radar Bod" not in names  # no distance
    assert "Apollo MSR-2 Vaskerom" not in names  # all entities disabled


async def test_discover_empty_registry(hass: HomeAssistant) -> None:
    """No registries staged -> nothing discovered."""
    assert discovery.discover_sensors(hass) == []


async def test_unique_id_suffix_match(hass: HomeAssistant) -> None:
    """Renamed entity ids are still matched through their unique ids."""
    await stage_renamed_device(hass)
    sensors = discovery.discover_sensors(hass)

    assert [sensor.name for sensor in sensors] == ["Apollo MSR-2 Gang"]
    assert sensors[0].entities == {
        "move_energy": "sensor.gangen_move_energy",
        "still_energy": "sensor.gangen_still_energy",
        "moving_distance": "sensor.gangen_moving_distance",
        "still_distance": "sensor.gangen_still_distance",
    }
    assert sensors[0].area_name is None


async def test_area_names(hass: HomeAssistant) -> None:
    """Sorted area names feed the room-name suggestions."""
    await build_installation(hass)
    assert discovery.area_names(hass) == ["Kjøkken", "Kontor", "Sofakrok", "Spisebord"]
