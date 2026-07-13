"""Tests for the options -> core config builder (config.py).

Pure unit tests: build_config consumes a plain mapping, so no Home Assistant
instance is involved.
"""

from __future__ import annotations

from typing import Any

from custom_components.presence_conductor.config import (
    baselines_from_options,
    build_config,
    build_tunables,
    sensor_entities,
)
from custom_components.presence_conductor.core.model import (
    GateBaselines,
    RoomConfig,
    SensorConfig,
    Tunables,
    ZoneBaselines,
    ZoneConfig,
)


def _options() -> dict[str, Any]:
    return {
        "sensors": [
            {
                "sensor_id": "apollo_msr_2_sofakrok",
                "name": "Apollo MSR-2 Sofakrok",
                "entities": {
                    "move_energy": "sensor.apollo_msr_2_f79794_radar_move_energy",
                    "still_energy": "sensor.apollo_msr_2_f79794_radar_still_energy",
                    "moving_distance": "sensor.apollo_msr_2_f79794_radar_moving_distance",
                    "still_distance": "sensor.apollo_msr_2_f79794_radar_still_distance",
                },
            },
            {
                "sensor_id": "apollo_msr_2_spisebord",
                "name": "Apollo MSR-2 Spisebord",
                "entities": {
                    "move_energy": "sensor.apollo_msr_2_fadea8_radar_move_energy",
                    "still_energy": "sensor.apollo_msr_2_fadea8_radar_still_energy",
                    "detection_distance": "sensor.apollo_msr_2_fadea8_radar_detection_distance",
                },
            },
        ],
        "zones": [
            {
                "zone_id": "sofakrok",
                "name": "Sofakrok",
                "sensor": "apollo_msr_2_sofakrok",
                "room": "stue",
                "near_cm": 0.0,
                "far_cm": 250.0,
                "fallback": True,
            },
            {
                "zone_id": "spisebord",
                "name": "Spisebord",
                "sensor": "apollo_msr_2_spisebord",
                "room": "stue",
                "near_cm": 260,  # ints coerce to float
                "far_cm": 450,
                "fallback": False,
            },
        ],
        "rooms": [{"room_id": "stue", "name": "Stue"}],
        "baselines": {
            "sofakrok": {
                "move_mu": 0.02,
                "move_sigma": 0.02,
                "still_mu": 0.30,
                "still_sigma": 0.05,
            },
            # A zone that no longer exists: passed through, ignored on read.
            "kontor": {"move_mu": 0.1, "move_sigma": 0.1, "still_mu": 0.1, "still_sigma": 0.1},
        },
    }


def test_build_config_full() -> None:
    """Options map 1:1 onto the core dataclasses."""
    config = build_config(_options())

    assert config.sensors == (
        SensorConfig(sensor_id="apollo_msr_2_sofakrok", name="Apollo MSR-2 Sofakrok"),
        SensorConfig(sensor_id="apollo_msr_2_spisebord", name="Apollo MSR-2 Spisebord"),
    )
    assert config.zones == (
        ZoneConfig(
            zone_id="sofakrok",
            name="Sofakrok",
            sensor_id="apollo_msr_2_sofakrok",
            room_id="stue",
            near_cm=0.0,
            far_cm=250.0,
            fallback=True,
        ),
        ZoneConfig(
            zone_id="spisebord",
            name="Spisebord",
            sensor_id="apollo_msr_2_spisebord",
            room_id="stue",
            near_cm=260.0,
            far_cm=450.0,
            fallback=False,
        ),
    )
    assert config.rooms == (RoomConfig(room_id="stue", name="Stue"),)
    # No tunables stored -> pure dataclass defaults.
    assert config.tunables == Tunables()
    # Convenience accessors from the core still work on the built config.
    assert config.zones_in_room("stue") == config.zones
    assert config.room_ids() == ("stue",)


def test_build_config_empty_options() -> None:
    """A fresh (or sensorless) entry builds an empty but valid config."""
    config = build_config({})
    assert config.sensors == ()
    assert config.zones == ()
    assert config.rooms == ()
    assert config.tunables == Tunables()


def test_tunables_partial_merge_and_unknown_keys() -> None:
    """Stored values win; missing fields default; unknown keys are ignored."""
    tunables = build_tunables({"tunables": {"theta_on": 0.9, "tau_decay": 120, "not_a_field": 1.0}})
    assert tunables.theta_on == 0.9
    assert tunables.tau_decay == 120.0
    assert tunables.theta_off == Tunables().theta_off  # untouched default

    config = build_config({**_options(), "tunables": {"z_attack": 2.5}})
    assert config.tunables.z_attack == 2.5
    assert config.tunables.k_move == Tunables().k_move


def test_baselines_passthrough() -> None:
    """Stored baselines map to ZoneBaselines, stale zone keys included."""
    baselines = baselines_from_options(_options())
    assert baselines == {
        "sofakrok": ZoneBaselines(move_mu=0.02, move_sigma=0.02, still_mu=0.30, still_sigma=0.05),
        "kontor": ZoneBaselines(move_mu=0.1, move_sigma=0.1, still_mu=0.1, still_sigma=0.1),
    }
    assert baselines_from_options({}) == {}


def test_sensor_entities_mapping() -> None:
    """sensor_id -> role -> entity_id, straight from the options."""
    entities = sensor_entities(_options())
    assert entities["apollo_msr_2_sofakrok"]["move_energy"] == (
        "sensor.apollo_msr_2_f79794_radar_move_energy"
    )
    assert entities["apollo_msr_2_spisebord"]["detection_distance"] == (
        "sensor.apollo_msr_2_fadea8_radar_detection_distance"
    )
    assert sensor_entities({}) == {}


def test_gate_size_cm_defaults_and_overrides() -> None:
    """Rule 2.4: per-sensor gate size; absent means the 0.75 m resolution."""
    options = _options()
    options["sensors"][0]["gate_size_cm"] = 20
    config = build_config(options)
    assert config.sensors[0].gate_size_cm == 20.0
    assert config.sensors[1].gate_size_cm == 75.0


def test_gate_entities_pass_through_the_sensor_mapping() -> None:
    """Gate roles are ordinary entries in the entities map (rules 2.4-2.6)."""
    options = _options()
    options["sensors"][0]["entities"]["g0_move"] = "sensor.apollo_msr_2_f79794_g0_move_energy"
    entities = sensor_entities(options)
    assert entities["apollo_msr_2_sofakrok"]["g0_move"] == (
        "sensor.apollo_msr_2_f79794_g0_move_energy"
    )


def test_baselines_gates_round_trip() -> None:
    """Rule 3.6: the optional "gates" mapping (string keys, JSON) becomes
    int-keyed GateBaselines; entries without it load with none (v0.1.0)."""
    options = _options()
    options["baselines"]["sofakrok"]["gates"] = {
        "2": {"move_mu": 0.01, "move_sigma": 0.02, "still_mu": 0.30, "still_sigma": 0.04},
        "3": {"move_mu": 0.05, "move_sigma": 0.05, "still_mu": 0.05, "still_sigma": 0.05},
    }
    baselines = baselines_from_options(options)
    assert baselines["sofakrok"].gates == {
        2: GateBaselines(move_mu=0.01, move_sigma=0.02, still_mu=0.30, still_sigma=0.04),
        3: GateBaselines(move_mu=0.05, move_sigma=0.05, still_mu=0.05, still_sigma=0.05),
    }
    assert baselines["kontor"].gates == {}  # backward compatible (3.6)
