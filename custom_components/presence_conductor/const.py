"""Constants for the Presence Conductor integration.

Options contract
----------------
``entry.data`` stays empty; ALL configuration lives in ``entry.options``.
The config/options flows are the single writers of these structures; the
controller (a later PR) is the single reader, via :func:`.config.build_config`.

- ``CONF_SENSORS`` — ``list[dict]``, one per physical mmWave device:
  ``{"sensor_id", "name", "entities": {role: entity_id}}``. ``sensor_id`` is a
  slug of the device name taken at configuration time and then *stored*, so
  later device/entity renames never change it. The ``entities`` map uses the
  ``ROLE_*``/gate-role constants below; energies and both plain distances are
  always present, ``detection_distance``, the binary target roles and the
  per-gate energy roles are optional. Gate roles (``g0_move``..``g8_still``)
  are filled by discovery when the device exposes engineering-mode gate
  entities; the manual sensor form deliberately omits them (18 selectors
  would drown it), so gate consumption is discovered-only by design. An
  optional ``"gate_size_cm"`` key (default 75.0, spec rule 2.4) covers the
  radar's 0.2 m resolution mode; it has no UI and is set by editing the
  stored options.
- ``CONF_ZONES`` — ``list[dict]``: ``{"zone_id", "name", "sensor", "room",
  "near_cm", "far_cm", "fallback"}``. ``sensor`` references a ``sensor_id``,
  ``room`` a ``room_id``; distances are floats in cm (spec §0, rule 2.1).
- ``CONF_ROOMS`` — ``list[dict]``: ``{"room_id", "name"}``. Display names for
  the rooms referenced by zones (``room_id`` = slug of the typed room name).
- ``CONF_TUNABLES`` — ``dict`` mirroring ``core.model.Tunables`` fields.
  Written only when the options flow saves the tunables section; missing keys
  (or the whole missing dict) fall back to the dataclass defaults in
  :func:`.config.build_config`.
- ``CONF_CALIBRATION_MODE`` — the operator-selected calibration workflow:
  ``skip``, ``simple`` (the backward-compatible default), or ``full``. The
  choice is configuration intent only; runtime capture is started separately.
- ``CONF_BASELINES`` — ``dict[zone_id, {"move_mu", "move_sigma", "still_mu",
  "still_sigma"}]``, optionally extended with ``"gates": {gate_index_str:
  {"move_mu", "move_sigma", "still_mu", "still_sigma"}}`` (spec rule 3.6;
  keys are strings because options are JSON). Written at *runtime* by the
  controller when a RecordBaseline window completes (spec rule 3.3:
  baselines persist in the config entry) — never by the flows. Baselines
  stored without ``"gates"`` (v0.1.0) load unchanged, and zones without
  per-gate floors are written without it. The options flow merges its
  sections into the stored options, so reconfiguring preserves calibration;
  baseline keys for zones that no longer exist are simply ignored on read.
"""

from __future__ import annotations

from .core.events import GATE_COUNT

DOMAIN = "presence_conductor"

CONF_SENSORS = "sensors"
CONF_ZONES = "zones"
CONF_ROOMS = "rooms"
CONF_TUNABLES = "tunables"
CONF_BASELINES = "baselines"
CONF_CALIBRATION_MODE = "calibration_mode"

CALIBRATION_MODE_SKIP = "skip"
CALIBRATION_MODE_SIMPLE = "simple"
CALIBRATION_MODE_FULL = "full"
DEFAULT_CALIBRATION_MODE = CALIBRATION_MODE_SIMPLE
CALIBRATION_MODES: tuple[str, ...] = (
    CALIBRATION_MODE_SKIP,
    CALIBRATION_MODE_SIMPLE,
    CALIBRATION_MODE_FULL,
)

# Entity roles: the per-sensor ``entities`` map keys, mirroring the fields of
# ``core.events.SensorFrame`` (rule 1.1). ``detection_distance`` is part of
# the LD2410 cluster but not consumed by the v1 estimator.
ROLE_MOVE_ENERGY = "move_energy"
ROLE_STILL_ENERGY = "still_energy"
ROLE_MOVING_DISTANCE = "moving_distance"
ROLE_STILL_DISTANCE = "still_distance"
ROLE_DETECTION_DISTANCE = "detection_distance"
ROLE_TARGET = "target"
ROLE_MOVING_TARGET = "moving_target"
ROLE_STILL_TARGET = "still_target"

ENERGY_ROLES: tuple[str, ...] = (ROLE_MOVE_ENERGY, ROLE_STILL_ENERGY)
DISTANCE_ROLES: tuple[str, ...] = (
    ROLE_MOVING_DISTANCE,
    ROLE_STILL_DISTANCE,
    ROLE_DETECTION_DISTANCE,
)
BINARY_ROLES: tuple[str, ...] = (ROLE_TARGET, ROLE_MOVING_TARGET, ROLE_STILL_TARGET)
#: The manual-form roles. Gate roles are deliberately NOT included: they are
#: discovered-only (see the options contract above).
ALL_ROLES: tuple[str, ...] = ENERGY_ROLES + DISTANCE_ROLES + BINARY_ROLES


def gate_move_role(index: int) -> str:
    """Role key of one per-gate move-energy entity (spec rules 2.4-2.6)."""
    return f"g{index}_move"


def gate_still_role(index: int) -> str:
    """Role key of one per-gate still-energy entity (spec rules 2.4-2.6)."""
    return f"g{index}_still"


GATE_MOVE_ROLES: tuple[str, ...] = tuple(gate_move_role(i) for i in range(GATE_COUNT))
GATE_STILL_ROLES: tuple[str, ...] = tuple(gate_still_role(i) for i in range(GATE_COUNT))
GATE_ROLES: tuple[str, ...] = GATE_MOVE_ROLES + GATE_STILL_ROLES

#: Per-sensor override of the radar's distance-gate size (spec rule 2.4).
CONF_GATE_SIZE_CM = "gate_size_cm"
