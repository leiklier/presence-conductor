"""Config and options flow for Presence Conductor.

The setup wizard walks discovered mmWave sensors (discovery.py) -> optional
manually assigned sensors -> zones (one form per zone, iterating over the
sensors) -> a non-blocking overlap warning (rule 2.2) -> entry creation.
Everything is stored in ``entry.options`` (``entry.data`` stays empty) using
the contract documented in const.py. The options flow edits three sections —
sensors (add/remove), zones, tunables — and merges its result back,
preserving every key it does not own (notably the runtime-written
``baselines``, rule 3.3).
"""

from __future__ import annotations

import dataclasses
from itertools import combinations
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    BooleanSelector,
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
)
from homeassistant.util import slugify

from . import discovery
from .const import (
    ALL_ROLES,
    BINARY_ROLES,
    CONF_ROOMS,
    CONF_SENSORS,
    CONF_TUNABLES,
    CONF_ZONES,
    DOMAIN,
    ROLE_DETECTION_DISTANCE,
    ROLE_MOVE_ENERGY,
    ROLE_MOVING_DISTANCE,
    ROLE_STILL_DISTANCE,
    ROLE_STILL_ENERGY,
)
from .core.model import Tunables

#: Zone distance defaults: full LD2410 range (far 600 cm is the radar's max).
DEFAULT_NEAR_CM = 0.0
DEFAULT_FAR_CM = 600.0
#: Selector ceiling, leaving headroom above the LD2410 range.
MAX_RANGE_CM = 800.0

#: UI metadata (min, max, step, unit) per Tunables field. Defaults come from
#: the dataclass; unlisted (future) fields get _TUNABLE_FALLBACK_UI.
_TUNABLE_UI: dict[str, tuple[float, float, float, str | None]] = {
    "margin_cm": (0, 200, 5, "cm"),
    "stale_after": (5, 300, 5, "s"),
    "tick_interval": (0.1, 10, 0.1, "s"),
    "sigma_min": (0.001, 0.2, 0.001, None),
    "energy_quantum": (0, 0.1, 0.001, None),
    "default_mu": (0, 1, 0.01, None),
    "default_sigma": (0.001, 1, 0.001, None),
    "z_cap": (1, 20, 0.5, None),
    "z_neg_cap": (0, 5, 0.1, None),
    "stat_sigma_min": (0.05, 2, 0.05, None),
    "stat_min_rows": (2, 600, 1, None),
    "tau_int_max": (1, 100, 1, None),
    "obs_budget": (0.1, 10, 0.1, "s"),
    "obs_hold": (0.5, 60, 0.5, "s"),
    "k_move": (0, 5, 0.05, None),
    "k_still": (0, 5, 0.05, None),
    "k_bias": (0, 5, 0.05, None),
    "u_cap": (0.5, 20, 0.5, None),
    "tau_decay": (5, 600, 5, "s"),
    "p_prior": (0.001, 0.5, 0.001, None),
    "attack_tail_ppm": (1, 10000, 1, "ppm"),
    "attack_confirm": (1, 5, 1, None),
    "attack_gap_min": (0.1, 5, 0.1, "s"),
    "attack_gap_max": (0.5, 10, 0.1, "s"),
    "p_attack": (0.5, 0.999, 0.001, None),
    "theta_on": (0.5, 0.999, 0.01, None),
    "theta_off": (0.001, 0.5, 0.01, None),
    "z_motion": (0, 10, 0.1, None),
    "motion_hold": (0, 60, 0.5, "s"),
    "p_min": (0.0001, 0.1, 0.001, None),
    "p_max": (0.5, 0.9999, 0.001, None),
    "distance_hold": (0, 300, 5, "s"),
    "p_background": (0.001, 0.5, 0.001, None),
    "t_background": (10, 3600, 10, "s"),
    "tau_background": (60, 86400, 60, "s"),
    "baseline_duration": (10, 600, 10, "s"),
    "t_dwell": (1, 600, 1, "s"),
    "t_settle": (1, 600, 1, "s"),
    "tau_home": (60, 7200, 60, "s"),
    "theta_home_on": (0.5, 0.999, 0.01, None),
    "theta_home_off": (0.001, 0.5, 0.01, None),
}
_TUNABLE_FALLBACK_UI: tuple[float, float, float, str | None] = (0, 86400, 0.001, None)


def _number(minimum: float, maximum: float, step: float, unit: str | None = None) -> NumberSelector:
    config = NumberSelectorConfig(min=minimum, max=maximum, step=step, mode=NumberSelectorMode.BOX)
    if unit is not None:
        config["unit_of_measurement"] = unit
    return NumberSelector(config)


def _tunables_schema() -> vol.Schema:
    schema: dict[Any, Any] = {}
    for field in dataclasses.fields(Tunables):
        if isinstance(field.default, bool):
            schema[vol.Required(field.name, default=field.default)] = BooleanSelector()
            continue
        minimum, maximum, step, unit = _TUNABLE_UI.get(field.name, _TUNABLE_FALLBACK_UI)
        schema[vol.Required(field.name, default=field.default)] = _number(
            minimum, maximum, step, unit
        )
    return vol.Schema(schema)


def _tunables_from_input(user_input: dict[str, Any]) -> dict[str, float]:
    return {f.name: user_input[f.name] for f in dataclasses.fields(Tunables)}


def _sensor_schema() -> vol.Schema:
    """One manually assigned sensor: a name and per-role entity pickers."""
    sensor = EntitySelector(EntitySelectorConfig(domain="sensor"))
    binary = EntitySelector(EntitySelectorConfig(domain="binary_sensor"))
    return vol.Schema(
        {
            vol.Required("name"): TextSelector(),
            vol.Required(ROLE_MOVE_ENERGY): sensor,
            vol.Required(ROLE_STILL_ENERGY): sensor,
            vol.Required(ROLE_MOVING_DISTANCE): sensor,
            vol.Required(ROLE_STILL_DISTANCE): sensor,
            vol.Optional(ROLE_DETECTION_DISTANCE): sensor,
            **{vol.Optional(role): binary for role in BINARY_ROLES},
            vol.Required("add_another", default=False): BooleanSelector(),
        }
    )


def _zone_schema(
    room_options: list[str], include_name: bool, include_add_another: bool
) -> vol.Schema:
    """Schema for one zone (setup and options; options keeps the name fixed)."""
    schema: dict[Any, Any] = {}
    if include_name:
        schema[vol.Required("name")] = TextSelector()
    schema[vol.Required("room")] = SelectSelector(
        SelectSelectorConfig(
            options=room_options, custom_value=True, mode=SelectSelectorMode.DROPDOWN
        )
    )
    schema[vol.Required("near_cm", default=DEFAULT_NEAR_CM)] = _number(0, MAX_RANGE_CM, 1, "cm")
    schema[vol.Required("far_cm", default=DEFAULT_FAR_CM)] = _number(0, MAX_RANGE_CM, 1, "cm")
    schema[vol.Required("fallback", default=False)] = BooleanSelector()
    if include_add_another:
        schema[vol.Required("add_another_zone", default=False)] = BooleanSelector()
    return vol.Schema(schema)


def _validate_zone(
    user_input: dict[str, Any],
    zones: list[dict[str, Any]],
    sensor_id: str,
    *,
    check_name: bool = True,
) -> dict[str, str]:
    """Field errors for a submitted zone form (against already-collected zones)."""
    errors: dict[str, str] = {}
    if check_name:
        zone_id = slugify(user_input["name"])
        if not zone_id:
            errors["name"] = "invalid_zone_name"
        elif any(zone["zone_id"] == zone_id for zone in zones):
            errors["name"] = "duplicate_zone_name"
    if not slugify(user_input["room"]):
        errors["room"] = "invalid_room_name"
    if float(user_input["far_cm"]) <= float(user_input["near_cm"]):
        errors["far_cm"] = "invalid_zone_range"
    # Rule 2.3: at most one fallback zone per sensor.
    if user_input["fallback"] and any(
        zone["fallback"] for zone in zones if zone["sensor"] == sensor_id
    ):
        errors["fallback"] = "multiple_fallback_zones"
    return errors


def _overlapping_pairs(
    zones: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Zone pairs of *different* sensors in the *same* room whose [near, far]
    intervals overlap (rule 2.2). Touching intervals (far == near) don't count:
    the warning is about the configured intervals, not the runtime margin."""
    return [
        (a, b)
        for a, b in combinations(zones, 2)
        if a["sensor"] != b["sensor"]
        and a["room"] == b["room"]
        and a["near_cm"] < b["far_cm"]
        and b["near_cm"] < a["far_cm"]
    ]


def _describe_overlaps(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    sensors: list[dict[str, Any]],
    room_names: dict[str, str],
) -> str:
    """Human-readable one-line-per-pair summary for the warning step."""
    sensor_names = {sensor["sensor_id"]: sensor["name"] for sensor in sensors}

    def zone_label(zone: dict[str, Any]) -> str:
        sensor = sensor_names.get(zone["sensor"], zone["sensor"])
        return f"{zone['name']} ({sensor}, {zone['near_cm']:g}-{zone['far_cm']:g} cm)"

    return "\n".join(
        f"- {zone_label(a)} + {zone_label(b)} in {room_names.get(a['room'], a['room'])}"
        for a, b in pairs
    )


def _rooms_option(zones: list[dict[str, Any]], room_names: dict[str, str]) -> list[dict[str, str]]:
    """The CONF_ROOMS list: every room referenced by a zone, in first-seen order."""
    ordered = dict.fromkeys(zone["room"] for zone in zones)
    return [{"room_id": room_id, "name": room_names.get(room_id, room_id)} for room_id in ordered]


def _unique_sensor_id(name: str, sensors: list[dict[str, Any]]) -> str:
    """Slug of the device name, suffixed on collision with an existing sensor."""
    base = slugify(name)
    taken = {sensor["sensor_id"] for sensor in sensors}
    if base not in taken:
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


def _suggest_zone_name(candidates: list[str | None], zones: list[dict[str, Any]]) -> str:
    """First candidate whose slug is still free, else a numbered variant."""
    taken = {zone["zone_id"] for zone in zones}
    fallback = ""
    for candidate in candidates:
        if not candidate:
            continue
        fallback = fallback or candidate
        if slugify(candidate) not in taken:
            return candidate
    n = 2
    while slugify(f"{fallback} {n}") in taken:
        n += 1
    return f"{fallback} {n}"


class PresenceConductorConfigFlow(ConfigFlow, domain=DOMAIN):
    """Multi-step setup wizard driven by registry discovery."""

    VERSION = 1

    def __init__(self) -> None:
        self._discovered: list[discovery.DiscoveredSensor] = []
        self._sensors: list[dict[str, Any]] = []
        self._areas: dict[str, str | None] = {}
        self._sensor_index = 0
        self._zones: list[dict[str, Any]] = []
        self._rooms: dict[str, str] = {}

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> PresenceConductorOptionsFlow:
        """Options flow for editing sensors, zones and tunables."""
        return PresenceConductorOptionsFlow()

    def _add_sensor(self, name: str, entities: dict[str, str], area_name: str | None) -> None:
        sensor_id = _unique_sensor_id(name, self._sensors)
        self._sensors.append({"sensor_id": sensor_id, "name": name, "entities": entities})
        self._areas[sensor_id] = area_name

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Intro: pick discovered sensors and/or opt into manual assignment."""
        await self.async_set_unique_id(DOMAIN)
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        if not self._discovered:
            self._discovered = discovery.discover_sensors(self.hass)

        errors: dict[str, str] = {}
        if user_input is not None:
            selected = set(user_input.get("sensors", []))
            if not selected and not user_input["add_manual"]:
                errors["base"] = "no_sensors_selected"
            else:
                for found in self._discovered:
                    if found.device_id in selected:
                        self._add_sensor(found.name, dict(found.entities), found.area_name)
                if user_input["add_manual"]:
                    return await self.async_step_manual_sensor()
                return await self.async_step_zones()

        schema: dict[Any, Any] = {}
        if self._discovered:
            options = [
                SelectOptionDict(
                    value=found.device_id,
                    label=f"{found.name} ({found.area_name})" if found.area_name else found.name,
                )
                for found in self._discovered
            ]
            schema[
                vol.Required("sensors", default=[found.device_id for found in self._discovered])
            ] = SelectSelector(
                SelectSelectorConfig(options=options, multiple=True, mode=SelectSelectorMode.LIST)
            )
        schema[vol.Required("add_manual", default=not self._discovered)] = BooleanSelector()
        return self.async_show_form(step_id="user", data_schema=vol.Schema(schema), errors=errors)

    async def async_step_manual_sensor(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Assign one sensor's entities by hand (loops while "add another")."""
        errors: dict[str, str] = {}
        if user_input is not None:
            name = user_input["name"]
            if not slugify(name):
                errors["name"] = "invalid_sensor_name"
            elif any(s["sensor_id"] == slugify(name) for s in self._sensors):
                errors["name"] = "duplicate_sensor_name"
            if not errors:
                entities = {role: user_input[role] for role in ALL_ROLES if user_input.get(role)}
                self._add_sensor(name, entities, None)
                if user_input["add_another"]:
                    return await self.async_step_manual_sensor()
                return await self.async_step_zones()

        schema = _sensor_schema()
        if user_input is not None:
            schema = self.add_suggested_values_to_schema(schema, user_input)
        return self.async_show_form(step_id="manual_sensor", data_schema=schema, errors=errors)

    async def async_step_zones(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """One form per zone, iterating over the configured sensors."""
        sensor = self._sensors[self._sensor_index]
        errors: dict[str, str] = {}

        if user_input is not None:
            errors = _validate_zone(user_input, self._zones, sensor["sensor_id"])
            if not errors:
                room_id = slugify(user_input["room"])
                self._rooms.setdefault(room_id, user_input["room"].strip())
                self._zones.append(
                    {
                        "zone_id": slugify(user_input["name"]),
                        "name": user_input["name"],
                        "sensor": sensor["sensor_id"],
                        "room": room_id,
                        "near_cm": float(user_input["near_cm"]),
                        "far_cm": float(user_input["far_cm"]),
                        "fallback": user_input["fallback"],
                    }
                )
                if not user_input["add_another_zone"]:
                    self._sensor_index += 1
                if self._sensor_index < len(self._sensors):
                    return await self.async_step_zones()
                return await self.async_step_overlap()

        area = self._areas.get(sensor["sensor_id"])
        if user_input is not None:
            suggested = user_input  # redisplay after a validation error
        else:
            suggested = {
                "name": _suggest_zone_name([area, sensor["name"]], self._zones),
                "room": area or sensor["name"],
            }
        room_options = sorted(set(discovery.area_names(self.hass)) | set(self._rooms.values()))
        schema = self.add_suggested_values_to_schema(
            _zone_schema(room_options, include_name=True, include_add_another=True), suggested
        )
        zone_number = sum(1 for z in self._zones if z["sensor"] == sensor["sensor_id"]) + 1
        return self.async_show_form(
            step_id="zones",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "sensor": sensor["name"],
                "sensor_number": str(self._sensor_index + 1),
                "sensor_count": str(len(self._sensors)),
                "zone_number": str(zone_number),
            },
        )

    async def async_step_overlap(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Non-blocking warning on same-room cross-sensor overlap (rule 2.2)."""
        pairs = _overlapping_pairs(self._zones)
        if user_input is not None or not pairs:
            return self.async_create_entry(
                title="Presence Conductor",
                data={},
                options={
                    CONF_SENSORS: self._sensors,
                    CONF_ZONES: self._zones,
                    CONF_ROOMS: _rooms_option(self._zones, self._rooms),
                },
            )
        return self.async_show_form(
            step_id="overlap",
            data_schema=vol.Schema({}),
            description_placeholders={
                "overlaps": _describe_overlaps(pairs, self._sensors, self._rooms)
            },
        )


class PresenceConductorOptionsFlow(OptionsFlow):
    """Edit sensors, zones or tunables; merge back preserving other keys."""

    def __init__(self) -> None:
        self._sensors: list[dict[str, Any]] = []
        self._areas: dict[str, str | None] = {}
        self._pending: list[dict[str, Any]] = []
        self._pending_index = 0
        self._added_zones: list[dict[str, Any]] = []
        self._zones: list[dict[str, Any]] = []
        self._zone_index = 0
        self._rooms: dict[str, str] = {}
        self._pending_save: dict[str, Any] = {}

    def _save(self, updates: dict[str, Any]) -> ConfigFlowResult:
        """Merge our sections into the options, preserving everything else.

        Notably ``baselines`` (written at runtime when a RecordBaseline window
        completes, rule 3.3) and any future keys survive untouched.
        """
        new_options = {**self.config_entry.options, **updates}
        return self.async_create_entry(title="", data=new_options)

    def _stored(self, key: str) -> list[dict[str, Any]]:
        return list(self.config_entry.options.get(key, []))

    def _stored_room_names(self) -> dict[str, str]:
        return {room["room_id"]: room["name"] for room in self._stored(CONF_ROOMS)}

    def _add_sensor(self, name: str, entities: dict[str, str], area_name: str | None) -> None:
        sensor_id = _unique_sensor_id(name, self._sensors)
        sensor = {"sensor_id": sensor_id, "name": name, "entities": entities}
        self._sensors.append(sensor)
        self._pending.append(sensor)
        self._areas[sensor_id] = area_name

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Section menu."""
        return self.async_show_menu(step_id="init", menu_options=["sensors", "zones", "tunables"])

    # ------------------------------------------------------------------
    # Sensors section: add (discovered or manual) / remove.
    # ------------------------------------------------------------------

    async def async_step_sensors(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Keep/remove configured sensors; offer newly discovered devices."""
        stored = self._stored(CONF_SENSORS)
        configured_entities = {
            entity_id for sensor in stored for entity_id in sensor["entities"].values()
        }
        discovered = [
            found
            for found in discovery.discover_sensors(self.hass)
            if not (set(found.entities.values()) & configured_entities)
        ]

        errors: dict[str, str] = {}
        if user_input is not None:
            selected = set(user_input["sensors"])
            kept = [sensor for sensor in stored if sensor["sensor_id"] in selected]
            added = [found for found in discovered if found.device_id in selected]
            if not kept and not added and not user_input["add_manual"]:
                errors["base"] = "no_sensors_selected"
            else:
                self._sensors = kept
                self._rooms = self._stored_room_names()
                for found in added:
                    self._add_sensor(found.name, dict(found.entities), found.area_name)
                if user_input["add_manual"]:
                    return await self.async_step_manual_sensor()
                return await self._async_continue_sensors()

        options = [
            SelectOptionDict(value=sensor["sensor_id"], label=sensor["name"]) for sensor in stored
        ] + [
            SelectOptionDict(
                value=found.device_id,
                label=f"{found.name} ({found.area_name})" if found.area_name else found.name,
            )
            for found in discovered
        ]
        schema = vol.Schema(
            {
                vol.Required(
                    "sensors", default=[sensor["sensor_id"] for sensor in stored]
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=options, multiple=True, mode=SelectSelectorMode.LIST
                    )
                ),
                vol.Required("add_manual", default=False): BooleanSelector(),
            }
        )
        return self.async_show_form(step_id="sensors", data_schema=schema, errors=errors)

    async def async_step_manual_sensor(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Assign one sensor's entities by hand (loops while "add another")."""
        errors: dict[str, str] = {}
        if user_input is not None:
            name = user_input["name"]
            if not slugify(name):
                errors["name"] = "invalid_sensor_name"
            elif any(s["sensor_id"] == slugify(name) for s in self._sensors):
                errors["name"] = "duplicate_sensor_name"
            if not errors:
                entities = {role: user_input[role] for role in ALL_ROLES if user_input.get(role)}
                self._add_sensor(name, entities, None)
                if user_input["add_another"]:
                    return await self.async_step_manual_sensor()
                return await self._async_continue_sensors()

        schema = _sensor_schema()
        if user_input is not None:
            schema = self.add_suggested_values_to_schema(schema, user_input)
        return self.async_show_form(step_id="manual_sensor", data_schema=schema, errors=errors)

    def _kept_zones(self) -> list[dict[str, Any]]:
        kept_ids = {sensor["sensor_id"] for sensor in self._sensors}
        return [zone for zone in self._stored(CONF_ZONES) if zone["sensor"] in kept_ids]

    async def _async_continue_sensors(self) -> ConfigFlowResult:
        """Zones for newly added sensors, then save the sensors section."""
        if self._pending_index < len(self._pending):
            return await self.async_step_new_zones()
        zones = self._kept_zones() + self._added_zones
        self._pending_save = {
            CONF_SENSORS: self._sensors,
            CONF_ZONES: zones,
            CONF_ROOMS: _rooms_option(zones, self._rooms),
        }
        if self._added_zones and _overlapping_pairs(zones):
            return await self.async_step_overlap()
        return self._save(self._pending_save)

    async def async_step_new_zones(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Zone forms for sensors added in this options session."""
        sensor = self._pending[self._pending_index]
        existing = self._kept_zones() + self._added_zones
        errors: dict[str, str] = {}

        if user_input is not None:
            errors = _validate_zone(user_input, existing, sensor["sensor_id"])
            if not errors:
                room_id = slugify(user_input["room"])
                self._rooms.setdefault(room_id, user_input["room"].strip())
                self._added_zones.append(
                    {
                        "zone_id": slugify(user_input["name"]),
                        "name": user_input["name"],
                        "sensor": sensor["sensor_id"],
                        "room": room_id,
                        "near_cm": float(user_input["near_cm"]),
                        "far_cm": float(user_input["far_cm"]),
                        "fallback": user_input["fallback"],
                    }
                )
                if not user_input["add_another_zone"]:
                    self._pending_index += 1
                return await self._async_continue_sensors()

        area = self._areas.get(sensor["sensor_id"])
        if user_input is not None:
            suggested = user_input
        else:
            suggested = {
                "name": _suggest_zone_name([area, sensor["name"]], existing),
                "room": area or sensor["name"],
            }
        room_options = sorted(set(discovery.area_names(self.hass)) | set(self._rooms.values()))
        schema = self.add_suggested_values_to_schema(
            _zone_schema(room_options, include_name=True, include_add_another=True), suggested
        )
        zone_number = sum(1 for z in self._added_zones if z["sensor"] == sensor["sensor_id"]) + 1
        return self.async_show_form(
            step_id="new_zones",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "sensor": sensor["name"],
                "sensor_number": str(self._pending_index + 1),
                "sensor_count": str(len(self._pending)),
                "zone_number": str(zone_number),
            },
        )

    # ------------------------------------------------------------------
    # Zones section: edit every stored zone, seeded from the options.
    # ------------------------------------------------------------------

    async def async_step_zones(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Re-run the zone form for every stored zone (names stay fixed)."""
        stored_zones = self._stored(CONF_ZONES)
        if not stored_zones:
            return self.async_abort(reason="no_zones_configured")
        if not self._rooms:
            self._rooms = self._stored_room_names()
        zone = dict(stored_zones[self._zone_index])

        errors: dict[str, str] = {}
        if user_input is not None:
            errors = _validate_zone(user_input, self._zones, zone["sensor"], check_name=False)
            if not errors:
                room_id = slugify(user_input["room"])
                self._rooms[room_id] = user_input["room"].strip()
                zone.update(
                    {
                        "room": room_id,
                        "near_cm": float(user_input["near_cm"]),
                        "far_cm": float(user_input["far_cm"]),
                        "fallback": user_input["fallback"],
                    }
                )
                self._zones.append(zone)
                self._zone_index += 1
                if self._zone_index < len(stored_zones):
                    return await self.async_step_zones()
                self._pending_save = {
                    CONF_ZONES: self._zones,
                    CONF_ROOMS: _rooms_option(self._zones, self._rooms),
                }
                if _overlapping_pairs(self._zones):
                    return await self.async_step_overlap()
                return self._save(self._pending_save)

        if user_input is not None:
            suggested = user_input
        else:
            suggested = {
                "room": self._rooms.get(zone["room"], zone["room"]),
                "near_cm": zone["near_cm"],
                "far_cm": zone["far_cm"],
                "fallback": zone["fallback"],
            }
        room_options = sorted(set(discovery.area_names(self.hass)) | set(self._rooms.values()))
        schema = self.add_suggested_values_to_schema(
            _zone_schema(room_options, include_name=False, include_add_another=False), suggested
        )
        return self.async_show_form(
            step_id="zones",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "zone": zone["name"],
                "zone_number": str(self._zone_index + 1),
                "zone_count": str(len(stored_zones)),
            },
        )

    async def async_step_overlap(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Non-blocking warning on same-room cross-sensor overlap (rule 2.2)."""
        if user_input is not None:
            return self._save(self._pending_save)
        zones = self._pending_save[CONF_ZONES]
        sensors = self._pending_save.get(CONF_SENSORS) or self._stored(CONF_SENSORS)
        return self.async_show_form(
            step_id="overlap",
            data_schema=vol.Schema({}),
            description_placeholders={
                "overlaps": _describe_overlaps(_overlapping_pairs(zones), sensors, self._rooms)
            },
        )

    # ------------------------------------------------------------------
    # Tunables section.
    # ------------------------------------------------------------------

    async def async_step_tunables(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Every Tunables field, seeded from storage (defaults otherwise)."""
        if user_input is not None:
            if user_input["attack_gap_max"] > user_input["attack_gap_min"]:
                return self._save({CONF_TUNABLES: _tunables_from_input(user_input)})
            schema = self.add_suggested_values_to_schema(_tunables_schema(), user_input)
            return self.async_show_form(
                step_id="tunables", data_schema=schema, errors={"base": "invalid_attack_gap"}
            )
        stored = self.config_entry.options.get(CONF_TUNABLES, {})
        schema = self.add_suggested_values_to_schema(_tunables_schema(), stored)
        return self.async_show_form(step_id="tunables", data_schema=schema)
