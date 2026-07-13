"""Replay recorded Home Assistant history through the estimation core.

Offline harness (spec 8.5): feeds a history export to the engine with real
timestamps and prints per-zone transition counts and ON/OFF duration
statistics in the format of the docs/DECISION.md baseline table, so tuning
changes are judged against recorded reality.

Usage:
    python tools/replay.py --config replay_config.json --history history.json

``history.json`` is a flat JSON list of state changes as returned by HA's
``/api/history/period`` with ``minimal_response`` (flattened across
entities), each item shaped ``{"entity_id": ..., "state": ...,
"last_changed": "2026-07-11T18:00:00.000000+00:00"}``.

``replay_config.json`` maps entities to sensors and declares zones:

    {
      "sensors": {
        "kontor": {
          "moving_distance": "sensor.kontor_msr2_moving_distance",
          "still_distance": "sensor.kontor_msr2_still_distance",
          "move_energy": "sensor.kontor_msr2_move_energy",
          "still_energy": "sensor.kontor_msr2_still_energy",
          "has_moving_target": "binary_sensor.kontor_msr2_moving_target",
          "has_still_target": "binary_sensor.kontor_msr2_still_target",
          "gates_move": ["sensor.kontor_msr2_g0_move_energy", "..."],
          "gates_still": ["sensor.kontor_msr2_g0_still_energy", "..."],
          "gate_size_cm": 75.0
        }
      },
      "zones": [
        {"zone_id": "kontor", "sensor_id": "kontor", "room_id": "kontor",
         "near_cm": 0, "far_cm": 300, "fallback": true}
      ],
      "baselines": {
        "kontor": {"move_mu": 0.05, "move_sigma": 0.05,
                   "still_mu": 0.05, "still_sigma": 0.05,
                   "gates": {"2": {"move_mu": 0.05, "move_sigma": 0.05,
                                   "still_mu": 0.05, "still_sigma": 0.05}}}
      },
      "tunables": {"tau_decay": 90}
    }

``baselines``, ``tunables`` and the gate keys are optional. ``gates_move``/
``gates_still`` list per-gate energy entities by gate index (spec rules
2.4-2.6); ``gate_size_cm`` (default 75.0, rule 2.4) and per-zone baseline
``"gates"`` floors (rule 3.6) ride along. Distance entities are treated
as ``None`` (rule 1.1) while the matching ``has_*_target`` binary is off,
mirroring the adapter's frame contract; an ``unavailable``/``unknown``
state on any of a sensor's aggregate entities marks the sensor unavailable
until the next parseable update, while an unknown *gate* entity only blanks
that gate — engineering mode dropping is normal, not sensor failure (rule
2.6). Ticks are synthesized every ``tick_interval`` between recorded
changes, and engine timers fire at their deadlines.

Pure Python: imports only the core package and the standard library.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Import only the core package: pre-register the parent packages as plain
# namespace modules so the HA adapter's __init__ (which imports
# homeassistant) never executes. The replay tool stays runnable with a bare
# python3 and no Home Assistant install.
for _name, _path in (
    ("custom_components", _ROOT / "custom_components"),
    ("custom_components.presence_conductor", _ROOT / "custom_components" / "presence_conductor"),
):
    if _name not in sys.modules:
        _module = types.ModuleType(_name)
        _module.__path__ = [str(_path)]
        sys.modules[_name] = _module

from custom_components.presence_conductor.core.engine import ConductorEngine  # noqa: E402
from custom_components.presence_conductor.core.events import (  # noqa: E402
    GATE_COUNT,
    SensorAvailability,
    SensorFrame,
    Tick,
)
from custom_components.presence_conductor.core.model import (  # noqa: E402
    DEFAULT_GATE_SIZE_CM,
    ConductorConfig,
    GateBaselines,
    InitialSnapshot,
    RoomConfig,
    SensorConfig,
    Tunables,
    ZoneBaselines,
    ZoneConfig,
)

FRAME_ROLES = (
    "moving_distance",
    "still_distance",
    "move_energy",
    "still_energy",
    "has_target",
    "has_moving_target",
    "has_still_target",
)

#: Internal role prefix for per-gate entities: ``gate_move:3`` etc.
_GATE_ROLE_PREFIX = "gate_"


# ---------------------------------------------------------------------
# Config / history loading
# ---------------------------------------------------------------------


def load_config(path: str | Path) -> tuple[ConductorConfig, dict[str, tuple[str, str]], dict]:
    """Returns (core config, entity_id -> (sensor_id, role), raw baselines)."""
    raw = json.loads(Path(path).read_text())
    sensors = tuple(
        SensorConfig(
            sensor_id,
            sensor_id,
            gate_size_cm=float(roles.get("gate_size_cm", DEFAULT_GATE_SIZE_CM)),  # 2.4
        )
        for sensor_id, roles in raw["sensors"].items()
    )
    zones = tuple(
        ZoneConfig(
            zone_id=z["zone_id"],
            name=z.get("name", z["zone_id"]),
            sensor_id=z["sensor_id"],
            room_id=z.get("room_id", z["zone_id"]),
            near_cm=float(z["near_cm"]),
            far_cm=float(z["far_cm"]),
            fallback=bool(z.get("fallback", False)),
        )
        for z in raw["zones"]
    )
    room_ids = dict.fromkeys(z.room_id for z in zones)
    rooms = tuple(RoomConfig(room_id, room_id) for room_id in room_ids)
    config = ConductorConfig(
        sensors=sensors,
        zones=zones,
        rooms=rooms,
        tunables=Tunables(**raw.get("tunables", {})),
    )
    entity_map: dict[str, tuple[str, str]] = {}
    for sensor_id, roles in raw["sensors"].items():
        for role, entity_id in roles.items():
            if role in FRAME_ROLES:
                entity_map[entity_id] = (sensor_id, role)
        # Per-gate energy entities, listed by gate index (rules 2.4-2.6).
        for channel in ("move", "still"):
            for index, entity_id in enumerate(roles.get(f"gates_{channel}") or []):
                entity_map[entity_id] = (sensor_id, f"{_GATE_ROLE_PREFIX}{channel}:{index}")
    return config, entity_map, raw.get("baselines", {})


def load_history(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def _parse_time(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _parse_state(role: str, state: str) -> tuple[bool, float | bool | None]:
    """Returns (parseable, value). Unparseable means unavailable/unknown."""
    if state in ("unavailable", "unknown", "", None):
        return False, None
    if role.startswith("has_"):
        return True, state == "on"
    try:
        return True, float(state)
    except ValueError:
        return False, None


# ---------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------


@dataclass
class ZoneTrace:
    """Occupied-transition log for one zone."""

    transitions: int = 0
    on_durations: list[float] = field(default_factory=list)
    off_durations: list[float] = field(default_factory=list)
    _state: bool = False
    _since: float | None = None

    def observe(self, occupied: bool, now: float) -> None:
        if self._since is None:
            self._state, self._since = occupied, now
            return
        if occupied == self._state:
            return
        duration = now - self._since
        (self.on_durations if self._state else self.off_durations).append(duration)
        self.transitions += 1
        self._state, self._since = occupied, now

    def close(self, now: float) -> None:
        if self._since is not None and now > self._since:
            duration = now - self._since
            (self.on_durations if self._state else self.off_durations).append(duration)


class _SensorShadow:
    """Latest raw values of one sensor's entities."""

    def __init__(self) -> None:
        self.values: dict[str, float | bool | None] = dict.fromkeys(FRAME_ROLES)
        #: Per-gate values by channel; a channel stays ``None`` until its
        #: first recorded gate update, mirroring the adapter's "configured
        #: and reported" contract (rule 1.1).
        self.gates: dict[str, list[float | None] | None] = {"move": None, "still": None}
        self.available = True

    def set_gate(self, role: str, value: float | None) -> None:
        """Fold one per-gate update (``gate_move:3``) into the shadow."""
        channel, index = role.removeprefix(_GATE_ROLE_PREFIX).split(":")
        values = self.gates[channel]
        if values is None:
            values = self.gates[channel] = [None] * GATE_COUNT
        values[int(index)] = value

    def frame(self, sensor_id: str) -> SensorFrame:
        v = self.values

        def _flag(role: str) -> bool:
            return bool(v.get(role) or False)

        def _distance(role: str, flag_role: str) -> float | None:
            distance = v.get(role)
            if distance is None:
                return None
            # Rule 1.1: no target of that kind -> no distance. Only enforced
            # when the flag entity is actually mapped and reporting False.
            if v.get(flag_role) is False:
                return None
            return float(distance)

        gate_move = self.gates["move"]
        gate_still = self.gates["still"]
        return SensorFrame(
            sensor_id=sensor_id,
            moving_distance_cm=_distance("moving_distance", "has_moving_target"),
            still_distance_cm=_distance("still_distance", "has_still_target"),
            move_energy=None if v.get("move_energy") is None else float(v["move_energy"]),
            still_energy=None if v.get("still_energy") is None else float(v["still_energy"]),
            has_target=_flag("has_target")
            or _flag("has_moving_target")
            or _flag("has_still_target"),
            has_moving_target=_flag("has_moving_target"),
            has_still_target=_flag("has_still_target"),
            gate_move=None if gate_move is None else tuple(gate_move),
            gate_still=None if gate_still is None else tuple(gate_still),
        )


def replay(
    config: ConductorConfig,
    entity_map: dict[str, tuple[str, str]],
    history: list[dict],
    baselines: dict | None = None,
) -> dict[str, ZoneTrace]:
    """Run the recorded history through a fresh engine; returns zone traces."""
    changes = sorted(
        (
            (_parse_time(item["last_changed"]), item["entity_id"], item["state"])
            for item in history
            if item.get("entity_id") in entity_map
        ),
        key=lambda item: item[0],
    )
    if not changes:
        return {}
    snapshot = InitialSnapshot(
        baselines={
            zone_id: ZoneBaselines(
                move_mu=b["move_mu"],
                move_sigma=b["move_sigma"],
                still_mu=b["still_mu"],
                still_sigma=b["still_sigma"],
                # Rule 3.6: optional per-gate floors, keyed like the config
                # entry (string indices, JSON).
                gates={
                    int(index): GateBaselines(
                        move_mu=g["move_mu"],
                        move_sigma=g["move_sigma"],
                        still_mu=g["still_mu"],
                        still_sigma=g["still_sigma"],
                    )
                    for index, g in (b.get("gates") or {}).items()
                },
            )
            for zone_id, b in (baselines or {}).items()
        }
    )
    engine = ConductorEngine(config, snapshot)
    shadows = {s.sensor_id: _SensorShadow() for s in config.sensors}
    traces = {z.zone_id: ZoneTrace() for z in config.zones}
    deadlines: dict[str, float] = {}
    tick_interval = config.tunables.tick_interval

    start = changes[0][0]
    now = start

    def absorb(plan) -> None:
        for timer_start in plan.timer_starts:
            deadlines[timer_start.key] = now + timer_start.delay
        for cancel in plan.timer_cancels:
            deadlines.pop(cancel.key, None)

    def observe() -> None:
        for zone_id, trace in traces.items():
            trace.observe(bool(engine.state.zones[zone_id].occupied), now)

    def advance_to(target: float) -> None:
        """Fire due timers and synthesize ticks, chronologically, to ``target``."""
        nonlocal now, next_tick
        while True:
            due = min(
                ((when, key) for key, when in deadlines.items() if when <= target),
                default=None,
            )
            tick_due = next_tick <= target
            if due is not None and (not tick_due or due[0] <= next_tick):
                now = max(now, due[0])
                deadlines.pop(due[1])
                absorb(engine.on_timer(due[1], now))
                observe()
            elif tick_due:
                now = max(now, next_tick)
                next_tick += tick_interval
                absorb(engine.submit(Tick(), now))
                observe()
            else:
                now = max(now, target)
                return

    absorb(engine.start(now))
    observe()
    next_tick = start + tick_interval

    for when, entity_id, state in changes:
        advance_to(when)
        sensor_id, role = entity_map[entity_id]
        shadow = shadows[sensor_id]
        parseable, value = _parse_state(role, state)
        if role.startswith(_GATE_ROLE_PREFIX):
            # An unknown gate is normal (engineering mode off): the gate
            # goes None and sensor availability is untouched — rule 1.3
            # stays keyed to the aggregate entities; the engine falls back
            # to the aggregate path per frame (rule 2.6).
            shadow.set_gate(role, float(value) if parseable else None)
            if shadow.available:
                absorb(engine.submit(shadow.frame(sensor_id), now))
                observe()
            continue
        if not parseable:
            shadow.values[role] = None
            if shadow.available:
                shadow.available = False
                absorb(engine.submit(SensorAvailability(sensor_id, available=False), now))
                observe()
            continue
        shadow.values[role] = value
        if not shadow.available:
            shadow.available = True
            absorb(engine.submit(SensorAvailability(sensor_id, available=True), now))
        absorb(engine.submit(shadow.frame(sensor_id), now))
        observe()

    for trace in traces.values():
        trace.close(now)
    return traces


# ---------------------------------------------------------------------
# Reporting (docs/DECISION.md baseline table format)
# ---------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.1f} m"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def format_table(traces: dict[str, ZoneTrace]) -> str:
    header = (
        "| zone | transitions | median ON | median OFF | OFF gaps <15 s "
        "| ON blips <10 s | p90 ON | p90 OFF |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for zone_id, trace in traces.items():
        median_on = statistics.median(trace.on_durations) if trace.on_durations else 0.0
        median_off = statistics.median(trace.off_durations) if trace.off_durations else 0.0
        lines.append(
            f"| {zone_id} | {trace.transitions} | {_fmt_duration(median_on)} "
            f"| {_fmt_duration(median_off)} "
            f"| {sum(1 for d in trace.off_durations if d < 15)} "
            f"| {sum(1 for d in trace.on_durations if d < 10)} "
            f"| {_fmt_duration(_percentile(trace.on_durations, 0.9))} "
            f"| {_fmt_duration(_percentile(trace.off_durations, 0.9))} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="replay config JSON")
    parser.add_argument("--history", required=True, help="HA history export JSON")
    args = parser.parse_args(argv)
    config, entity_map, baselines = load_config(args.config)
    history = load_history(args.history)
    traces = replay(config, entity_map, history, baselines)
    print(format_table(traces))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
