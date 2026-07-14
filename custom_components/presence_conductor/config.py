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
from .core import emissions
from .core.model import (
    DEFAULT_GATE_SIZE_CM,
    ConductorConfig,
    ConfusionMatrix,
    EmissionValidationMetrics,
    GateBaselines,
    LinearDiscriminant,
    OccupiedEmissionProfile,
    RoomConfig,
    ScenarioEmissionMetrics,
    SensorConfig,
    StatBaseline,
    Tunables,
    ZoneBaselines,
    ZoneConfig,
)

_TUNABLE_TYPES = {f.name: type(f.default) for f in fields(Tunables)}


def _confusion(raw: Mapping[str, Any]) -> ConfusionMatrix:
    if not isinstance(raw, Mapping):
        raise TypeError("confusion matrix must be a mapping")
    return ConfusionMatrix(
        true_positive=int(raw.get("true_positive", 0)),
        false_positive=int(raw.get("false_positive", 0)),
        true_negative=int(raw.get("true_negative", 0)),
        false_negative=int(raw.get("false_negative", 0)),
    )


def _occupied_profile(raw: Any) -> OccupiedEmissionProfile | None:
    """Parse a persisted profile defensively; malformed data means fallback."""
    if not isinstance(raw, Mapping):
        return None
    try:
        validation_raw = raw.get("validation")
        validation = None
        if isinstance(validation_raw, Mapping):
            validation = EmissionValidationMetrics(
                threshold=float(validation_raw.get("threshold", 0.0)),
                confusion=_confusion(validation_raw.get("confusion", {})),
                scenarios=tuple(
                    ScenarioEmissionMetrics(
                        name=str(item["name"]),
                        expected_occupied=bool(item["expected_occupied"]),
                        confusion=_confusion(item.get("confusion", {})),
                        mean_discriminant=float(item["mean_discriminant"]),
                        minimum_discriminant=float(item["minimum_discriminant"]),
                        maximum_discriminant=float(item["maximum_discriminant"]),
                    )
                    for item in validation_raw.get("scenarios", [])
                ),
                empty_mean_rate=(
                    float(validation_raw["empty_mean_rate"])
                    if validation_raw.get("empty_mean_rate") is not None
                    else None
                ),
                occupied_mean_rates={
                    str(key): float(value)
                    for key, value in (validation_raw.get("occupied_mean_rates") or {}).items()
                },
            )

        def discriminant(key: str) -> LinearDiscriminant:
            item = raw[key]
            return LinearDiscriminant(
                move_weight=float(item["move_weight"]),
                still_weight=float(item["still_weight"]),
                intercept=float(item["intercept"]),
            )

        profile = OccupiedEmissionProfile(
            active=discriminant("active"),
            settled=discriminant("settled"),
            path=str(raw.get("path", "aggregate")),
            active_weight=float(raw.get("active_weight", 0.5)),
            evidence_scale=float(raw.get("evidence_scale", 1.0)),
            evidence_min=float(raw.get("evidence_min", -3.0)),
            evidence_max=float(raw.get("evidence_max", 3.0)),
            fingerprint=str(raw.get("fingerprint", "")),
            empty_rows=int(raw.get("empty_rows", 0)),
            active_rows=int(raw.get("active_rows", 0)),
            settled_rows=int(raw.get("settled_rows", 0)),
            validation=validation,
        )
        if profile.path not in {"aggregate", "gate"} or not profile.fingerprint:
            return None
        emissions.occupied_discriminant(profile, (0.0, 0.0))
        emissions.learned_evidence_rate(profile, (0.0, 0.0))
        emissions.validate_persisted_profile(profile)
        return profile
    except AttributeError, KeyError, OverflowError, TypeError, ValueError:
        return None


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
            floor_fingerprint=(str(b["floor_fingerprint"]) if "floor_fingerprint" in b else None),
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
            occupied_profile=_occupied_profile(b.get("occupied_profile")),
        )
        for zone_id, b in raw.items()
    }
