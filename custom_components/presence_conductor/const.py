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
  ``ROLE_*`` constants below; energies and both plain distances are always
  present, ``detection_distance`` and the binary target roles are optional.
- ``CONF_ZONES`` — ``list[dict]``: ``{"zone_id", "name", "sensor", "room",
  "near_cm", "far_cm", "fallback"}``. ``sensor`` references a ``sensor_id``,
  ``room`` a ``room_id``; distances are floats in cm (spec §0, rule 2.1).
- ``CONF_ROOMS`` — ``list[dict]``: ``{"room_id", "name"}``. Display names for
  the rooms referenced by zones (``room_id`` = slug of the typed room name).
- ``CONF_TUNABLES`` — ``dict`` mirroring ``core.model.Tunables`` fields.
  Written only when the options flow saves the tunables section; missing keys
  (or the whole missing dict) fall back to the dataclass defaults in
  :func:`.config.build_config`.
- ``CONF_BASELINES`` — ``dict[zone_id, {"move_mu", "move_sigma", "still_mu",
  "still_sigma"}]``. Written at *runtime* by the controller when a
  RecordBaseline window completes (spec rule 3.3: baselines persist in the
  config entry) — never by the flows. The options flow merges its sections
  into the stored options, so reconfiguring preserves calibration; baseline
  keys for zones that no longer exist are simply ignored on read.
"""

from __future__ import annotations

DOMAIN = "presence_conductor"

CONF_SENSORS = "sensors"
CONF_ZONES = "zones"
CONF_ROOMS = "rooms"
CONF_TUNABLES = "tunables"
CONF_BASELINES = "baselines"

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
ALL_ROLES: tuple[str, ...] = ENERGY_ROLES + DISTANCE_ROLES + BINARY_ROLES
