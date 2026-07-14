"""The offline replay harness (spec 8.5): history export -> metrics table."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import replay as replay_mod

BASE = datetime(2026, 7, 11, 18, 0, 0, tzinfo=UTC)


def _at(seconds: float) -> str:
    return (BASE + timedelta(seconds=seconds)).isoformat()


def make_replay_config() -> dict:
    return {
        "sensors": {
            "kontor": {
                "moving_distance": "sensor.kontor_moving_distance",
                "still_distance": "sensor.kontor_still_distance",
                "move_energy": "sensor.kontor_move_energy",
                "still_energy": "sensor.kontor_still_energy",
                "has_moving_target": "binary_sensor.kontor_moving_target",
                "has_still_target": "binary_sensor.kontor_still_target",
            }
        },
        "zones": [
            {
                "zone_id": "kontor",
                "sensor_id": "kontor",
                "room_id": "kontor",
                "near_cm": 0,
                "far_cm": 300,
                "fallback": True,
            }
        ],
        "baselines": {
            "kontor": {
                "move_mu": 0.05,
                "move_sigma": 0.05,
                "still_mu": 0.05,
                "still_sigma": 0.05,
            }
        },
    }


def make_history() -> list[dict]:
    """Quiet start, one 60 s visit (strong move then still), quiet end."""
    history = [
        {"entity_id": "sensor.kontor_move_energy", "state": "5", "last_changed": _at(0)},
        {"entity_id": "sensor.kontor_still_energy", "state": "5", "last_changed": _at(0)},
    ]
    # t=20: person walks in - strong move evidence (fast attack, rule 4.2).
    moving_target = "binary_sensor.kontor_moving_target"
    history += [
        {"entity_id": moving_target, "state": "on", "last_changed": _at(20)},
        {"entity_id": "sensor.kontor_moving_distance", "state": "150", "last_changed": _at(20)},
        {"entity_id": "sensor.kontor_move_energy", "state": "60", "last_changed": _at(20)},
    ]
    # t=21: a second move reading confirms the attack (4.2).
    history.append(
        {"entity_id": "sensor.kontor_move_energy", "state": "61", "last_changed": _at(21)}
    )
    # t=25..80: sitting still - breathing wobbles the energy every 2 s
    # (real occupied readings churn; only truly empty streams freeze, 3.8).
    for t in range(25, 81, 2):
        history += [
            {
                "entity_id": "sensor.kontor_still_energy",
                "state": str(40 + (t % 4) // 2),
                "last_changed": _at(t),
            },
        ]
    history += [
        {"entity_id": "sensor.kontor_still_distance", "state": "150", "last_changed": _at(25)},
        {"entity_id": "binary_sensor.kontor_still_target", "state": "on", "last_changed": _at(25)},
    ]
    # t=81: gone - everything back at the baseline.
    history += [
        {"entity_id": moving_target, "state": "off", "last_changed": _at(81)},
        {"entity_id": "binary_sensor.kontor_still_target", "state": "off", "last_changed": _at(81)},
        {"entity_id": "sensor.kontor_move_energy", "state": "5", "last_changed": _at(81)},
        {"entity_id": "sensor.kontor_still_energy", "state": "5", "last_changed": _at(81)},
    ]
    # The empty stream keeps churning at the floor (~3 s cadence, as
    # measured on real hardware) — that carries the observed-absence drive.
    for t in range(84, 120, 3):
        history.append(
            {
                "entity_id": "sensor.kontor_still_energy",
                "state": str(4 + (t % 6) // 3),
                "last_changed": _at(t),
            }
        )
    # Quiet tail so the OFF interval closes long after the visit.
    history.append(
        {"entity_id": "sensor.kontor_move_energy", "state": "4", "last_changed": _at(310)}
    )
    return history


def test_replay_produces_one_clean_visit(tmp_path: Path) -> None:
    config, entity_map, baselines = _load(tmp_path)
    traces = replay_mod.replay(config, entity_map, make_history(), baselines)
    trace = traces["kontor"]
    # ON at ~t=10 (fast attack), OFF once the posterior drains after t=70:
    # exactly one OFF->ON->OFF cycle, no flapping in between.
    assert trace.transitions == 2
    assert len(trace.on_durations) == 1
    assert 60 <= trace.on_durations[0] <= 120  # visit + decay tail, no blips
    assert all(d >= 15 for d in trace.off_durations)  # no short OFF gaps


def test_replay_handles_unavailability(tmp_path: Path) -> None:
    config, entity_map, baselines = _load(tmp_path)
    history = make_history()
    history.append(
        {
            "entity_id": "sensor.kontor_move_energy",
            "state": "unavailable",
            "last_changed": _at(400),
        }
    )
    history.append(
        {"entity_id": "sensor.kontor_move_energy", "state": "5", "last_changed": _at(500)}
    )
    traces = replay_mod.replay(config, entity_map, history, baselines)
    assert traces["kontor"].transitions == 2  # unavailability adds no flaps


def test_format_table_matches_decision_layout(tmp_path: Path) -> None:
    config, entity_map, baselines = _load(tmp_path)
    traces = replay_mod.replay(config, entity_map, make_history(), baselines)
    table = replay_mod.format_table(traces)
    assert "| zone | transitions | median ON | median OFF" in table
    assert "| kontor | 2 |" in table


def make_gated_replay_config() -> dict:
    """The kontor config plus per-gate entities and calibrated gate floors."""
    config = make_replay_config()
    sensor = config["sensors"]["kontor"]
    sensor["gates_move"] = [f"sensor.kontor_g{i}_move_energy" for i in range(9)]
    sensor["gates_still"] = [f"sensor.kontor_g{i}_still_energy" for i in range(9)]
    sensor["gate_size_cm"] = 75.0
    config["tunables"] = {"use_gate_evidence": True}  # 2.6: experimental opt-in
    config["baselines"]["kontor"]["gate_indices"] = list(range(5))
    config["baselines"]["kontor"]["gates"] = {
        str(i): {"move_mu": 0.05, "move_sigma": 0.05, "still_mu": 0.05, "still_sigma": 0.05}
        for i in range(5)
    }
    return config


def make_gate_history() -> list[dict]:
    """One gate-driven visit at gate 2, then an engineering-mode dropout."""
    history = [
        {"entity_id": "sensor.kontor_move_energy", "state": "5", "last_changed": _at(0)},
        {"entity_id": "sensor.kontor_still_energy", "state": "5", "last_changed": _at(0)},
    ]
    history += [
        {"entity_id": f"sensor.kontor_g{i}_move_energy", "state": "5", "last_changed": _at(1)}
        for i in range(9)
    ]
    # t=20/21: strong move at gate 2, twice - the confirmed fast attack
    # rides the gate path (2.5, 4.2).
    history.append(
        {"entity_id": "sensor.kontor_g2_move_energy", "state": "60", "last_changed": _at(20)}
    )
    history.append(
        {"entity_id": "sensor.kontor_g2_move_energy", "state": "61", "last_changed": _at(21)}
    )
    # t=25..80: still returns at gate 2 hold the zone (breathing wobble -
    # real occupied streams churn, 3.8).
    for t in range(25, 81, 2):
        history.append(
            {
                "entity_id": "sensor.kontor_g2_still_energy",
                "state": str(40 + (t % 4) // 2),
                "last_changed": _at(t),
            }
        )
    # t=81: gone - the gates back at the floor.
    history += [
        {"entity_id": "sensor.kontor_g2_move_energy", "state": "5", "last_changed": _at(81)},
        {"entity_id": "sensor.kontor_g2_still_energy", "state": "5", "last_changed": _at(81)},
    ]
    # t=120: engineering mode drops - every gate reads unavailable. Gates go
    # None (per-frame fallback, rule 2.6) with no availability flap (1.3).
    history += [
        {
            "entity_id": f"sensor.kontor_g{i}_move_energy",
            "state": "unavailable",
            "last_changed": _at(120),
        }
        for i in range(9)
    ]
    # The empty stream keeps churning at the floor (~3 s cadence, as
    # measured on real hardware) — that carries the observed-absence drive.
    for t in range(84, 120, 3):
        history.append(
            {
                "entity_id": "sensor.kontor_still_energy",
                "state": str(4 + (t % 6) // 3),
                "last_changed": _at(t),
            }
        )
    # Quiet tail so the OFF interval closes long after the visit.
    history.append(
        {"entity_id": "sensor.kontor_move_energy", "state": "4", "last_changed": _at(310)}
    )
    return history


def test_replay_consumes_gate_history(tmp_path: Path) -> None:
    """Rules 2.4-2.6/3.6 through the replay harness: gate entities drive the
    visit and the engineering-mode dropout causes no extra transitions."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(make_gated_replay_config()))
    config, entity_map, baselines = replay_mod.load_config(config_path)
    assert config.sensors[0].gate_size_cm == 75.0
    assert entity_map["sensor.kontor_g2_move_energy"] == ("kontor", "gate_move:2")

    traces = replay_mod.replay(config, entity_map, make_gate_history(), baselines)
    trace = traces["kontor"]
    assert trace.transitions == 2  # one clean gate-driven visit, nothing else
    assert len(trace.on_durations) == 1
    assert 55 <= trace.on_durations[0] <= 130  # visit + decay tail
    assert all(d >= 15 for d in trace.off_durations)  # dropout added no flaps


def test_main_prints_table(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "config.json"
    history_path = tmp_path / "history.json"
    config_path.write_text(json.dumps(make_replay_config()))
    history_path.write_text(json.dumps(make_history()))
    assert replay_mod.main(["--config", str(config_path), "--history", str(history_path)]) == 0
    out = capsys.readouterr().out
    assert "| kontor |" in out


def _load(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(make_replay_config()))
    return replay_mod.load_config(config_path)
