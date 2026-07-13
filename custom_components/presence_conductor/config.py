"""Options -> core config translation (the adapter side of the contract).

The config flow writes ``entry.options`` using the contract documented in
const.py; this module turns those options into the frozen core dataclasses
the engine consumes (rule 7.3). The controller PR is the consumer; this
module has no Home Assistant dependencies so it is unit-testable directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from .const import CONF_BASELINES, CONF_ROOMS, CONF_SENSORS, CONF_TUNABLES, CONF_ZONES
from .core.model import (
    ConductorConfig,
    RoomConfig,
    SensorConfig,
    Tunables,
    ZoneBaselines,
    ZoneConfig,
)

_TUNABLE_FIELDS = frozenset(f.name for f in fields(Tunables))


def build_tunables(options: Mapping[str, Any]) -> Tunables:
    """Stored tunables over the dataclass defaults; unknown keys ignored."""
    stored = options.get(CONF_TUNABLES) or {}
    return Tunables(**{k: float(v) for k, v in stored.items() if k in _TUNABLE_FIELDS})


def build_config(options: Mapping[str, Any]) -> ConductorConfig:
    """Translate config-entry options into the frozen core config."""
    sensors = tuple(
        SensorConfig(sensor_id=s["sensor_id"], name=s["name"])
        for s in options.get(CONF_SENSORS) or []
    )
    zones = tuple(
        ZoneConfig(
            zone_id=z["zone_id"],
            name=z["name"],
            sensor_id=z["sensor"],
            room_id=z["room"],
            near_cm=float(z["near_cm"]),
            far_cm=float(z["far_cm"]),
            fallback=bool(z.get("fallback", False)),
        )
        for z in options.get(CONF_ZONES) or []
    )
    rooms = tuple(
        RoomConfig(room_id=r["room_id"], name=r["name"]) for r in options.get(CONF_ROOMS) or []
    )
    return ConductorConfig(
        sensors=sensors, zones=zones, rooms=rooms, tunables=build_tunables(options)
    )


def sensor_entities(options: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """The stored entity mapping: ``sensor_id -> role -> entity_id``."""
    return {s["sensor_id"]: dict(s["entities"]) for s in options.get(CONF_SENSORS) or []}


def baselines_from_options(options: Mapping[str, Any]) -> dict[str, ZoneBaselines]:
    """Persisted per-zone calibration (rule 3.3) as core dataclasses.

    Passthrough by design: baseline keys for zones that no longer exist are
    kept — the engine seeds from ``InitialSnapshot.baselines`` by zone id, so
    stale keys are simply never read.
    """
    raw = options.get(CONF_BASELINES) or {}
    return {
        zone_id: ZoneBaselines(
            move_mu=float(b["move_mu"]),
            move_sigma=float(b["move_sigma"]),
            still_mu=float(b["still_mu"]),
            still_sigma=float(b["still_sigma"]),
        )
        for zone_id, b in raw.items()
    }
