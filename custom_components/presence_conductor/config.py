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

from .const import (
    CONF_BASELINES,
    CONF_GATE_SIZE_CM,
    CONF_ROOMS,
    CONF_SENSORS,
    CONF_TUNABLES,
    CONF_ZONES,
)
from .core.model import (
    DEFAULT_GATE_SIZE_CM,
    ConductorConfig,
    GateBaselines,
    RoomConfig,
    SensorConfig,
    StatBaseline,
    Tunables,
    ZoneBaselines,
    ZoneConfig,
)

_TUNABLE_TYPES = {f.name: type(f.default) for f in fields(Tunables)}


def build_tunables(options: Mapping[str, Any]) -> Tunables:
    """Stored tunables over the dataclass defaults; unknown keys ignored.

    Values are cast to the field's default type (``attack_confirm`` is an
    int; number selectors hand back floats).
    """
    stored = options.get(CONF_TUNABLES) or {}
    return Tunables(**{k: _TUNABLE_TYPES[k](v) for k, v in stored.items() if k in _TUNABLE_TYPES})


def build_config(options: Mapping[str, Any]) -> ConductorConfig:
    """Translate config-entry options into the frozen core config."""
    sensors = tuple(
        SensorConfig(
            sensor_id=s["sensor_id"],
            name=s["name"],
            # Rule 2.4: per-sensor gate size; absent means the radar's
            # default 0.75 m range resolution.
            gate_size_cm=float(s.get(CONF_GATE_SIZE_CM, DEFAULT_GATE_SIZE_CM)),
        )
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
            sensor_id=str(b["sensor_id"]) if "sensor_id" in b else None,
            gate_size_cm=float(b["gate_size_cm"]) if "gate_size_cm" in b else None,
            # Rule 3.6: optional per-gate floors. Baselines stored before
            # per-gate evidence existed (v0.1.0) have no "gates" key; gate
            # indices are stored as strings because options are JSON.
            gates={
                int(index): GateBaselines(
                    move_mu=float(g["move_mu"]),
                    move_sigma=float(g["move_sigma"]),
                    still_mu=float(g["still_mu"]),
                    still_sigma=float(g["still_sigma"]),
                    has_move=bool(g.get("has_move", True)),
                    has_still=bool(g.get("has_still", True)),
                )
                for index, g in (b.get("gates") or {}).items()
            },
            gate_indices=(
                tuple(int(index) for index in b["gate_indices"]) if "gate_indices" in b else ()
            ),
            # Rule 3.7: optional statistic calibration. Baselines stored
            # before 3.7 have no "stats" key and fall back to the analytic
            # values.
            stats={
                key: StatBaseline(
                    mu=float(s["mu"]),
                    sigma=float(s["sigma"]),
                    clip_mu=float(s.get("clip_mu", 0.0)),
                    tau=float(s.get("tau", 1.0)),
                    decorrelation_seconds=(
                        float(s["decorrelation_seconds"])
                        if s.get("decorrelation_seconds") is not None
                        else None
                    ),
                    # Missing v0.4 metadata is deliberately invalid rather
                    # than silently trusted under a changed transform.
                    fingerprint=str(s.get("fingerprint", "")),
                )
                for key, s in (b.get("stats") or {}).items()
            },
        )
        for zone_id, b in raw.items()
    }
