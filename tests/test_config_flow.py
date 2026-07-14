"""Tests for the Presence Conductor config and options flows."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.presence_conductor.const import (
    CALIBRATION_MODE_FULL,
    CALIBRATION_MODE_SIMPLE,
    CALIBRATION_MODE_SKIP,
    CONF_CALIBRATION_MODE,
    DOMAIN,
)
from custom_components.presence_conductor.core.model import Tunables
from tests.test_discovery import (
    KJOKKEN_PREFIX,
    KONTOR_PREFIX,
    SOFAKROK_PREFIX,
    SPISEBORD_PREFIX,
    SPISEBORD_ROLES,
    build_installation,
    cluster_entities,
    stage_renamed_device,
)

TUNABLE_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Tunables)}

COMPONENT_DIR = Path(__file__).parent.parent / "custom_components" / "presence_conductor"


def _suggested(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the suggested values from a form result's schema."""
    values: dict[str, Any] = {}
    for key in result["data_schema"].schema:
        if isinstance(key, vol.Marker) and key.description:
            values[str(key)] = key.description.get("suggested_value")
    return values


def _zone_input(
    name: str,
    room: str,
    near: float = 0,
    far: float = 600,
    fallback: bool = False,
    add_another: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "room": room,
        "near_cm": near,
        "far_cm": far,
        "fallback": fallback,
        "add_another_zone": add_another,
    }


def _base_options() -> dict[str, Any]:
    """Stored options mirroring the contract, incl. runtime-written keys."""
    return {
        "sensors": [
            {
                "sensor_id": "apollo_msr_2_kjokken",
                "name": "Apollo MSR-2 Kjøkken",
                "entities": cluster_entities(KJOKKEN_PREFIX),
            },
            {
                "sensor_id": "apollo_msr_2_kontor",
                "name": "Apollo MSR-2 Kontor",
                "entities": cluster_entities(KONTOR_PREFIX),
            },
            {
                "sensor_id": "apollo_msr_2_sofakrok",
                "name": "Apollo MSR-2 Sofakrok",
                "entities": cluster_entities(SOFAKROK_PREFIX),
            },
            {
                "sensor_id": "apollo_msr_2_spisebord",
                "name": "Apollo MSR-2 Spisebord",
                "entities": cluster_entities(SPISEBORD_PREFIX, SPISEBORD_ROLES),
            },
        ],
        "zones": [
            {
                "zone_id": "kjokken",
                "name": "Kjøkken",
                "sensor": "apollo_msr_2_kjokken",
                "room": "kjokken",
                "near_cm": 0.0,
                "far_cm": 600.0,
                "fallback": False,
            },
            {
                "zone_id": "kontor",
                "name": "Kontor",
                "sensor": "apollo_msr_2_kontor",
                "room": "kontor",
                "near_cm": 0.0,
                "far_cm": 400.0,
                "fallback": False,
            },
            {
                "zone_id": "sofakrok",
                "name": "Sofakrok",
                "sensor": "apollo_msr_2_sofakrok",
                "room": "stue",
                "near_cm": 0.0,
                "far_cm": 250.0,
                "fallback": False,
            },
            {
                "zone_id": "spisebord",
                "name": "Spisebord",
                "sensor": "apollo_msr_2_spisebord",
                "room": "stue",
                "near_cm": 260.0,
                "far_cm": 450.0,
                "fallback": False,
            },
        ],
        "rooms": [
            {"room_id": "kjokken", "name": "Kjøkken"},
            {"room_id": "kontor", "name": "Kontor"},
            {"room_id": "stue", "name": "Stue"},
        ],
        "tunables": dict(TUNABLE_DEFAULTS),
        # Runtime-written calibration (rule 3.3): the flows never touch it.
        "baselines": {
            "kjokken": {
                "move_mu": 0.05,
                "move_sigma": 0.02,
                "still_mu": 0.12,
                "still_sigma": 0.03,
            },
            "sofakrok": {
                "move_mu": 0.02,
                "move_sigma": 0.02,
                "still_mu": 0.30,
                "still_sigma": 0.05,
            },
        },
    }


async def _add_entry(hass: HomeAssistant) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Presence Conductor",
        unique_id=DOMAIN,
        data={},
        options=_base_options(),
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


# ---------------------------------------------------------------------------
# Setup flow
# ---------------------------------------------------------------------------


async def test_full_happy_path(hass: HomeAssistant) -> None:
    """Discovery -> zones per sensor (incl. a two-zone sensor) -> exact options."""
    devices = await build_installation(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    # All discovered devices are preselected (schema default).
    markers = {str(key): key for key in result["data_schema"].schema}
    assert set(markers["sensors"].default()) == set(devices.values())

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {"sensors": list(devices.values()), "add_manual": False},
    )

    # Zone 1: Kjøkken. Suggestions come from the device's area.
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "zones"
    assert result["description_placeholders"] == {
        "sensor": "Apollo MSR-2 Kjøkken",
        "sensor_number": "1",
        "sensor_count": "4",
        "zone_number": "1",
    }
    suggested = _suggested(result)
    assert suggested["name"] == "Kjøkken"
    assert suggested["room"] == "Kjøkken"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _zone_input("Kjøkken", "Kjøkken", fallback=True)
    )

    # Kontor gets two zones (fallback on the first; per-sensor rule 2.3).
    assert result["step_id"] == "zones"
    assert result["description_placeholders"]["sensor"] == "Apollo MSR-2 Kontor"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        _zone_input("Kontor pult", "Kontor", 0, 150, fallback=True, add_another=True),
    )
    assert result["step_id"] == "zones"
    assert result["description_placeholders"] == {
        "sensor": "Apollo MSR-2 Kontor",
        "sensor_number": "2",
        "sensor_count": "4",
        "zone_number": "2",
    }
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _zone_input("Kontor dør", "Kontor", 150, 400)
    )

    # Sofakrok and Spisebord share the acoustic room "Stue" without overlap.
    assert result["description_placeholders"]["sensor"] == "Apollo MSR-2 Sofakrok"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _zone_input("Sofakrok", "Stue", 0, 250)
    )
    assert result["description_placeholders"]["sensor"] == "Apollo MSR-2 Spisebord"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _zone_input("Spisebord", "Stue", 260, 450)
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "calibration_mode"
    markers = {str(key): key for key in result["data_schema"].schema}
    assert markers[CONF_CALIBRATION_MODE].default() == CALIBRATION_MODE_SIMPLE
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_CALIBRATION_MODE: CALIBRATION_MODE_FULL}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Presence Conductor"
    assert result["data"] == {}
    assert dict(result["result"].options) == {
        "sensors": [
            {
                "sensor_id": "apollo_msr_2_kjokken",
                "name": "Apollo MSR-2 Kjøkken",
                "entities": cluster_entities(KJOKKEN_PREFIX),
            },
            {
                "sensor_id": "apollo_msr_2_kontor",
                "name": "Apollo MSR-2 Kontor",
                "entities": cluster_entities(KONTOR_PREFIX),
            },
            {
                "sensor_id": "apollo_msr_2_sofakrok",
                "name": "Apollo MSR-2 Sofakrok",
                "entities": cluster_entities(SOFAKROK_PREFIX),
            },
            {
                "sensor_id": "apollo_msr_2_spisebord",
                "name": "Apollo MSR-2 Spisebord",
                "entities": cluster_entities(SPISEBORD_PREFIX, SPISEBORD_ROLES),
            },
        ],
        "zones": [
            {
                "zone_id": "kjokken",
                "name": "Kjøkken",
                "sensor": "apollo_msr_2_kjokken",
                "room": "kjokken",
                "near_cm": 0.0,
                "far_cm": 600.0,
                "fallback": True,
            },
            {
                "zone_id": "kontor_pult",
                "name": "Kontor pult",
                "sensor": "apollo_msr_2_kontor",
                "room": "kontor",
                "near_cm": 0.0,
                "far_cm": 150.0,
                "fallback": True,
            },
            {
                "zone_id": "kontor_dor",
                "name": "Kontor dør",
                "sensor": "apollo_msr_2_kontor",
                "room": "kontor",
                "near_cm": 150.0,
                "far_cm": 400.0,
                "fallback": False,
            },
            {
                "zone_id": "sofakrok",
                "name": "Sofakrok",
                "sensor": "apollo_msr_2_sofakrok",
                "room": "stue",
                "near_cm": 0.0,
                "far_cm": 250.0,
                "fallback": False,
            },
            {
                "zone_id": "spisebord",
                "name": "Spisebord",
                "sensor": "apollo_msr_2_spisebord",
                "room": "stue",
                "near_cm": 260.0,
                "far_cm": 450.0,
                "fallback": False,
            },
        ],
        "rooms": [
            {"room_id": "kjokken", "name": "Kjøkken"},
            {"room_id": "kontor", "name": "Kontor"},
            {"room_id": "stue", "name": "Stue"},
        ],
        CONF_CALIBRATION_MODE: CALIBRATION_MODE_FULL,
    }


async def test_overlap_warning_path(hass: HomeAssistant) -> None:
    """Same-room cross-sensor overlap warns (rule 2.2) but does not block."""
    devices = await build_installation(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {"sensors": [devices["sofakrok"], devices["spisebord"]], "add_manual": False},
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _zone_input("Sofakrok", "Stue", 0, 300)
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _zone_input("Spisebord", "Stue", 200, 450)
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "overlap"
    overlaps = result["description_placeholders"]["overlaps"]
    assert "Sofakrok" in overlaps
    assert "Spisebord" in overlaps
    assert "Stue" in overlaps

    # Non-blocking: submitting proceeds to the calibration choice.
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "calibration_mode"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_CALIBRATION_MODE: CALIBRATION_MODE_SKIP}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    zones = result["result"].options["zones"]
    assert [zone["zone_id"] for zone in zones] == ["sofakrok", "spisebord"]
    assert result["result"].options[CONF_CALIBRATION_MODE] == CALIBRATION_MODE_SKIP


async def test_manual_sensor_path(hass: HomeAssistant) -> None:
    """No discovery: assign two sensors by hand, then their zones."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    # Nothing discovered: the form only offers manual assignment (defaulting on).
    keys = [str(key) for key in result["data_schema"].schema]
    assert keys == ["add_manual"]

    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"add_manual": True})
    assert result["step_id"] == "manual_sensor"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "name": "Gang radar",
            "move_energy": "sensor.gang_move_energy",
            "still_energy": "sensor.gang_still_energy",
            "moving_distance": "sensor.gang_moving_distance",
            "still_distance": "sensor.gang_still_distance",
            "target": "binary_sensor.gang_target",
            "add_another": True,
        },
    )

    assert result["step_id"] == "manual_sensor"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "name": "Bad radar",
            "move_energy": "sensor.bad_move_energy",
            "still_energy": "sensor.bad_still_energy",
            "moving_distance": "sensor.bad_moving_distance",
            "still_distance": "sensor.bad_still_distance",
            "add_another": False,
        },
    )

    # Zones: no area -> the sensor name seeds the suggestions.
    assert result["step_id"] == "zones"
    suggested = _suggested(result)
    assert suggested["name"] == "Gang radar"
    assert suggested["room"] == "Gang radar"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _zone_input("Gang", "Gang")
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _zone_input("Bad", "Bad")
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "calibration_mode"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_CALIBRATION_MODE: CALIBRATION_MODE_SIMPLE}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    options = dict(result["result"].options)
    assert options["sensors"] == [
        {
            "sensor_id": "gang_radar",
            "name": "Gang radar",
            "entities": {
                "move_energy": "sensor.gang_move_energy",
                "still_energy": "sensor.gang_still_energy",
                "moving_distance": "sensor.gang_moving_distance",
                "still_distance": "sensor.gang_still_distance",
                "target": "binary_sensor.gang_target",
            },
        },
        {
            "sensor_id": "bad_radar",
            "name": "Bad radar",
            "entities": {
                "move_energy": "sensor.bad_move_energy",
                "still_energy": "sensor.bad_still_energy",
                "moving_distance": "sensor.bad_moving_distance",
                "still_distance": "sensor.bad_still_distance",
            },
        },
    ]
    assert [zone["zone_id"] for zone in options["zones"]] == ["gang", "bad"]
    assert options[CONF_CALIBRATION_MODE] == CALIBRATION_MODE_SIMPLE


async def test_abort_single_instance(hass: HomeAssistant) -> None:
    """A second flow aborts immediately."""
    MockConfigEntry(domain=DOMAIN, unique_id=DOMAIN, data={}).add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "single_instance_allowed"


async def test_no_sensors_selected_error(hass: HomeAssistant) -> None:
    """Deselecting everything without manual assignment re-shows the form."""
    await build_installation(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"sensors": [], "add_manual": False}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {"base": "no_sensors_selected"}


async def test_duplicate_zone_name_rejected(hass: HomeAssistant) -> None:
    """Zone names must slugify to unique zone ids across sensors."""
    devices = await build_installation(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {"sensors": [devices["sofakrok"], devices["spisebord"]], "add_manual": False},
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _zone_input("Stue", "Stue")
    )
    # "STUE" slugifies to the same zone_id as "Stue".
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _zone_input("STUE", "Stue")
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "zones"
    assert result["errors"] == {"name": "duplicate_zone_name"}


async def test_second_fallback_same_sensor_rejected(hass: HomeAssistant) -> None:
    """At most one fallback zone per sensor (rule 2.3)."""
    devices = await build_installation(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"sensors": [devices["kontor"]], "add_manual": False}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        _zone_input("Kontor pult", "Kontor", 0, 150, fallback=True, add_another=True),
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _zone_input("Kontor dør", "Kontor", 150, 400, fallback=True)
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"fallback": "multiple_fallback_zones"}

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _zone_input("Kontor dør", "Kontor", 150, 400)
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "calibration_mode"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_CALIBRATION_MODE: CALIBRATION_MODE_SIMPLE}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY


async def test_invalid_zone_range_rejected(hass: HomeAssistant) -> None:
    """far_cm must exceed near_cm."""
    devices = await build_installation(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"sensors": [devices["kjokken"]], "add_manual": False}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _zone_input("Kjøkken", "Kjøkken", 300, 300)
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"far_cm": "invalid_zone_range"}


# ---------------------------------------------------------------------------
# Options flow
# ---------------------------------------------------------------------------


async def test_options_zones_roundtrip_preserves_baselines(hass: HomeAssistant) -> None:
    """Edit one zone's far_cm; everything else — notably baselines — survives."""
    entry = await _add_entry(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.MENU
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "zones"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "zones"
    assert result["description_placeholders"] == {
        "zone": "Kjøkken",
        "zone_number": "1",
        "zone_count": "4",
    }
    # Seeded from storage: room shows its display name.
    suggested = _suggested(result)
    assert suggested["room"] == "Kjøkken"
    assert suggested["near_cm"] == 0.0
    assert suggested["far_cm"] == 600.0
    assert suggested["fallback"] is False

    # Zone 1: shrink 600 -> 550; keep the rest.
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {"room": "Kjøkken", "near_cm": 0, "far_cm": 550, "fallback": False},
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {"room": "Kontor", "near_cm": 0, "far_cm": 400, "fallback": False},
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {"room": "Stue", "near_cm": 0, "far_cm": 250, "fallback": False},
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {"room": "Stue", "near_cm": 260, "far_cm": 450, "fallback": False},
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY

    expected = _base_options()
    expected["zones"][0]["far_cm"] = 550.0
    assert dict(entry.options) == expected
    assert entry.options["baselines"] == _base_options()["baselines"]


async def test_options_zones_overlap_warning(hass: HomeAssistant) -> None:
    """Edits that introduce same-room overlap warn but still save."""
    entry = await _add_entry(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "zones"}
    )
    for zone_input in (
        {"room": "Kjøkken", "near_cm": 0, "far_cm": 600, "fallback": False},
        {"room": "Kontor", "near_cm": 0, "far_cm": 400, "fallback": False},
        # Sofakrok now reaches into Spisebord's slice of the shared room.
        {"room": "Stue", "near_cm": 0, "far_cm": 400, "fallback": False},
        {"room": "Stue", "near_cm": 260, "far_cm": 450, "fallback": False},
    ):
        result = await hass.config_entries.options.async_configure(result["flow_id"], zone_input)

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "overlap"
    assert "Sofakrok" in result["description_placeholders"]["overlaps"]

    result = await hass.config_entries.options.async_configure(result["flow_id"], {})
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options["zones"][2]["far_cm"] == 400.0
    assert entry.options["baselines"] == _base_options()["baselines"]


async def test_options_tunables_roundtrip(hass: HomeAssistant) -> None:
    """Tunables are seeded from storage and merged back."""
    entry = await _add_entry(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "tunables"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "tunables"
    assert _suggested(result)["theta_on"] == 0.80

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {**TUNABLE_DEFAULTS, "theta_on": 0.9, "tau_decay": 120.0}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options["tunables"] == {**TUNABLE_DEFAULTS, "theta_on": 0.9, "tau_decay": 120.0}
    assert entry.options["baselines"] == _base_options()["baselines"]
    assert entry.options["sensors"] == _base_options()["sensors"]


async def test_options_calibration_defaults_to_simple_and_preserves_options(
    hass: HomeAssistant,
) -> None:
    """Legacy entries default to simple; changing mode touches no other key."""
    entry = await _add_entry(hass)
    assert CONF_CALIBRATION_MODE not in entry.options

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.MENU
    assert "calibration" in result["menu_options"]
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "calibration"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "calibration"
    markers = {str(key): key for key in result["data_schema"].schema}
    assert markers[CONF_CALIBRATION_MODE].default() == CALIBRATION_MODE_SIMPLE

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_CALIBRATION_MODE: CALIBRATION_MODE_FULL}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options == {
        **_base_options(),
        CONF_CALIBRATION_MODE: CALIBRATION_MODE_FULL,
    }
    assert entry.options["baselines"] == _base_options()["baselines"]


async def test_options_reject_inverted_attack_gap(hass: HomeAssistant) -> None:
    entry = await _add_entry(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "tunables"}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {**TUNABLE_DEFAULTS, "attack_gap_min": 5.0, "attack_gap_max": 0.5},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "invalid_attack_gap"}
    assert entry.options["tunables"] == _base_options()["tunables"]


async def test_options_remove_sensor(hass: HomeAssistant) -> None:
    """Deselecting a sensor drops it and its zones; baselines stay untouched."""
    entry = await _add_entry(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "sensors"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "sensors"

    kept = ["apollo_msr_2_kjokken", "apollo_msr_2_kontor", "apollo_msr_2_sofakrok"]
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"sensors": kept, "add_manual": False}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY

    assert [s["sensor_id"] for s in entry.options["sensors"]] == kept
    assert [z["zone_id"] for z in entry.options["zones"]] == ["kjokken", "kontor", "sofakrok"]
    # "stue" is still referenced by the sofakrok zone, so it survives.
    assert entry.options["rooms"] == _base_options()["rooms"]
    # Baselines are preserved verbatim — including the (now stale) spisebord-less
    # zone keys; stale keys are ignored on read.
    assert entry.options["baselines"] == _base_options()["baselines"]
    assert entry.options["tunables"] == _base_options()["tunables"]


async def test_options_add_manual_sensor(hass: HomeAssistant) -> None:
    """Adding a manual sensor chains into its zone form, then saves."""
    entry = await _add_entry(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "sensors"}
    )
    stored_ids = [s["sensor_id"] for s in _base_options()["sensors"]]
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"sensors": stored_ids, "add_manual": True}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "manual_sensor"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "name": "Gang radar",
            "move_energy": "sensor.gang_move_energy",
            "still_energy": "sensor.gang_still_energy",
            "moving_distance": "sensor.gang_moving_distance",
            "still_distance": "sensor.gang_still_distance",
            "add_another": False,
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "new_zones"
    assert result["description_placeholders"]["sensor"] == "Gang radar"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], _zone_input("Gang", "Gang")
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY

    assert [s["sensor_id"] for s in entry.options["sensors"]] == [
        *stored_ids,
        "gang_radar",
    ]
    assert entry.options["zones"][-1] == {
        "zone_id": "gang",
        "name": "Gang",
        "sensor": "gang_radar",
        "room": "gang",
        "near_cm": 0.0,
        "far_cm": 600.0,
        "fallback": False,
    }
    assert {"room_id": "gang", "name": "Gang"} in entry.options["rooms"]
    assert entry.options["baselines"] == _base_options()["baselines"]


async def test_options_add_discovered_sensor(hass: HomeAssistant) -> None:
    """A newly discovered (unconfigured) device can be added from options."""
    await build_installation(hass)
    gang_device = await stage_renamed_device(hass)
    entry = await _add_entry(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "sensors"}
    )
    stored_ids = [s["sensor_id"] for s in _base_options()["sensors"]]
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"sensors": [*stored_ids, gang_device], "add_manual": False}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "new_zones"
    assert result["description_placeholders"]["sensor"] == "Apollo MSR-2 Gang"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], _zone_input("Gang", "Gang")
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY

    added = entry.options["sensors"][-1]
    assert added["sensor_id"] == "apollo_msr_2_gang"
    assert added["entities"] == {
        "move_energy": "sensor.gangen_move_energy",
        "still_energy": "sensor.gangen_still_energy",
        "moving_distance": "sensor.gangen_moving_distance",
        "still_distance": "sensor.gangen_still_distance",
    }
    assert entry.options["zones"][-1]["sensor"] == "apollo_msr_2_gang"
    assert entry.options["baselines"] == _base_options()["baselines"]


async def test_options_zones_abort_without_zones(hass: HomeAssistant) -> None:
    """The zones section needs zones to edit."""
    entry = MockConfigEntry(domain=DOMAIN, unique_id=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "zones"}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "no_zones_configured"


# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------


def _key_paths(obj: Any, prefix: str = "") -> set[str]:
    if not isinstance(obj, dict):
        return {prefix}
    return {
        path
        for key, value in obj.items()
        for path in _key_paths(value, f"{prefix}.{key}" if prefix else key)
    }


def test_translations_in_sync() -> None:
    """en.json is byte-identical to strings.json; nb.json has the same keys."""
    strings = (COMPONENT_DIR / "strings.json").read_bytes()
    assert (COMPONENT_DIR / "translations" / "en.json").read_bytes() == strings

    base_keys = _key_paths(json.loads(strings))
    nb_keys = _key_paths(json.loads((COMPONENT_DIR / "translations" / "nb.json").read_text()))
    assert nb_keys == base_keys
