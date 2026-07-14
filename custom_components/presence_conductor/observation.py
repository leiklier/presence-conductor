"""Home Assistant sensor reports folded into engine-ready frames.

This module owns the observation-clock boundary from ENGINE_SPEC rule 1.1:
HA entities are a torn, cached view of one radar frame, while the core accepts
one complete :class:`SensorFrame`.  Keeping that conversion separate makes it
possible to review freshness semantics without the controller's lifecycle,
timers, publishing, or calibration persistence in view.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, State

from .config import baselines_from_options, sensor_entities
from .const import (
    ENERGY_ROLES,
    GATE_COUNT,
    GATE_MOVE_ROLES,
    GATE_ROLES,
    GATE_STILL_ROLES,
    ROLE_MOVE_ENERGY,
    ROLE_MOVING_DISTANCE,
    ROLE_MOVING_TARGET,
    ROLE_STILL_DISTANCE,
    ROLE_STILL_ENERGY,
    ROLE_STILL_TARGET,
    ROLE_TARGET,
)
from .core.events import SensorFrame
from .core.model import InitialSnapshot

UNAVAILABLE_STATES = (STATE_UNAVAILABLE, STATE_UNKNOWN)

# ``detection_distance`` belongs to the LD2410 cluster but is not consumed
# by the estimator, so its churn deliberately produces no frame.
FRAME_ROLES: tuple[str, ...] = (
    ROLE_MOVE_ENERGY,
    ROLE_STILL_ENERGY,
    ROLE_MOVING_DISTANCE,
    ROLE_STILL_DISTANCE,
    ROLE_TARGET,
    ROLE_MOVING_TARGET,
    ROLE_STILL_TARGET,
)

_GATE_ROLE_INDEX: dict[str, tuple[str, int]] = {
    **{role: ("move", index) for index, role in enumerate(GATE_MOVE_ROLES)},
    **{role: ("still", index) for index, role in enumerate(GATE_STILL_ROLES)},
}

# Distance and flag reports observe their channel under the verified atomic
# radar-frame guarantee in rule 1.1. Fast attack remains energy-only.
_MOVE_ROLES = frozenset(
    {ROLE_MOVE_ENERGY, ROLE_MOVING_DISTANCE, ROLE_MOVING_TARGET, ROLE_TARGET, *GATE_MOVE_ROLES}
)
_STILL_ROLES = frozenset(
    {ROLE_STILL_ENERGY, ROLE_STILL_DISTANCE, ROLE_STILL_TARGET, ROLE_TARGET, *GATE_STILL_ROLES}
)
_MOVE_ENERGY_ROLES = frozenset({ROLE_MOVE_ENERGY, *GATE_MOVE_ROLES})


def _as_float(state: State | None) -> float | None:
    """Return a numeric HA state, or ``None`` when it is not usable."""
    if state is None or state.state in UNAVAILABLE_STATES:
        return None
    try:
        return float(state.state)
    except ValueError:
        return None


def _is_on(state: State | None) -> bool:
    return state is not None and state.state == STATE_ON


@dataclass(slots=True)
class SensorView:
    """Last observed values and observation epochs for one radar cluster."""

    move_energy: float | None = None
    still_energy: float | None = None
    moving_distance: float | None = None
    still_distance: float | None = None
    has_target: bool = False
    has_moving_target: bool = False
    has_still_target: bool = False
    gate_move: list[float | None] | None = None
    gate_still: list[float | None] | None = None
    available: bool = False
    move_obs: int = 0
    still_obs: int = 0
    frame_obs: int = 0
    move_energy_obs: int = 0

    def update(self, role: str, state: State | None, *, measurement: bool = True) -> None:
        """Fold one entity state into the view and advance certified epochs."""
        parseable = state is not None and (
            state.state in {STATE_ON, STATE_OFF}
            if role in {ROLE_TARGET, ROLE_MOVING_TARGET, ROLE_STILL_TARGET}
            else _as_float(state) is not None
        )
        valid_measurement = (
            measurement
            and parseable
            and state is not None
            and state.state not in UNAVAILABLE_STATES
        )
        if valid_measurement:
            self.frame_obs += 1
        if valid_measurement and role in _MOVE_ROLES:
            self.move_obs += 1
        if valid_measurement and role in _STILL_ROLES:
            self.still_obs += 1
        if valid_measurement and role in _MOVE_ENERGY_ROLES:
            self.move_energy_obs += 1

        if (gate := _GATE_ROLE_INDEX.get(role)) is not None:
            channel, index = gate
            values = self.gate_move if channel == "move" else self.gate_still
            if values is None:
                values = [None] * GATE_COUNT
                if channel == "move":
                    self.gate_move = values
                else:
                    self.gate_still = values
            values[index] = _as_float(state)
            return

        if role == ROLE_MOVE_ENERGY:
            self.move_energy = _as_float(state)
        elif role == ROLE_STILL_ENERGY:
            self.still_energy = _as_float(state)
        elif role == ROLE_MOVING_DISTANCE:
            self.moving_distance = _as_float(state)
        elif role == ROLE_STILL_DISTANCE:
            self.still_distance = _as_float(state)
        elif role == ROLE_TARGET:
            self.has_target = _is_on(state)
        elif role == ROLE_MOVING_TARGET:
            self.has_moving_target = _is_on(state)
        elif role == ROLE_STILL_TARGET:
            self.has_still_target = _is_on(state)

    def frame(self, sensor_id: str) -> SensorFrame:
        """Build the complete cached frame consumed by the core."""
        return SensorFrame(
            sensor_id=sensor_id,
            moving_distance_cm=self.moving_distance,
            still_distance_cm=self.still_distance,
            move_energy=self.move_energy,
            still_energy=self.still_energy,
            has_target=self.has_target,
            has_moving_target=self.has_moving_target,
            has_still_target=self.has_still_target,
            gate_move=None if self.gate_move is None else tuple(self.gate_move),
            gate_still=None if self.gate_still is None else tuple(self.gate_still),
            move_obs=self.move_obs,
            still_obs=self.still_obs,
            frame_obs=self.frame_obs,
            move_energy_obs=self.move_energy_obs,
        )


def required_available(hass: HomeAssistant, roles: Mapping[str, str]) -> bool:
    """Whether every required aggregate-energy entity is live (rule 1.3)."""
    for role in ENERGY_ROLES:
        entity_id = roles.get(role)
        if entity_id is None:
            return False
        state = hass.states.get(entity_id)
        if state is None or state.state in UNAVAILABLE_STATES:
            return False
    return True


def build_view(hass: HomeAssistant, roles: Mapping[str, str]) -> SensorView:
    """Build one sensor view from Home Assistant's current state machine."""
    view = SensorView()
    for role in (*FRAME_ROLES, *GATE_ROLES):
        if (entity_id := roles.get(role)) is not None:
            view.update(role, hass.states.get(entity_id))
    view.available = required_available(hass, roles)
    return view


def build_initial_snapshot(hass: HomeAssistant, options: Mapping[str, Any]) -> InitialSnapshot:
    """Snapshot current HA states and persisted baselines for core startup."""
    frames: dict[str, SensorFrame | None] = {}
    available: dict[str, bool] = {}
    for sensor_id, roles in sensor_entities(options).items():
        view = build_view(hass, roles)
        available[sensor_id] = view.available
        frames[sensor_id] = view.frame(sensor_id) if view.available else None
    return InitialSnapshot(
        frames=frames,
        available=available,
        baselines=baselines_from_options(options),
        enabled=True,
    )
