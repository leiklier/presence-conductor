"""Distance gating and unit hygiene (spec rules 1.4, 2).

Gating keys on the reported *distance* alone, not the radar's binary target
verdict: a sub-threshold still person keeps a reported distance while
``has_still_target`` is off, and rule 3.5 requires that margin to keep
counting as evidence. The binary flags matter only for rule 2.3 (flag set,
distance omitted) and rule 4.4's motion channel.

Same-room separation (rule 2.2) is config-only: the mask is the *only*
mechanism, and non-overlapping intervals between sensors sharing a room are
the operator's contract. There is no cross-sensor arbitration here.
"""

from __future__ import annotations

from .events import GATE_COUNT, SensorFrame
from .model import ConductorConfig, ZoneConfig


def normalize_energy(value: float | None) -> float | None:
    """Energies are normalized to [0, 1] on ingest; out-of-range values are
    clamped, never rejected (rule 1.4)."""
    if value is None:
        return None
    return min(100.0, max(0.0, value)) / 100.0


def normalize_gates(
    values: tuple[float | None, ...] | None,
) -> tuple[float | None, ...] | None:
    """Rule 1.4 for per-gate energies: unknown gates stay ``None``."""
    if values is None:
        return None
    return tuple(normalize_energy(value) for value in values)


def owned_gates(zone: ZoneConfig, gate_size_cm: float, margin_cm: float) -> tuple[int, ...]:
    """The gate indices a zone owns (rule 2.4): every gate whose interval
    ``[i * gate_size, (i + 1) * gate_size)`` overlaps the zone's masked
    interval ``[near - margin, far + margin]`` (the 2.1 mask). Adjacent
    zones may share a boundary gate — ownership is a mask, not a partition.
    """
    lo = zone.near_cm - margin_cm
    hi = zone.far_cm + margin_cm
    return tuple(
        index
        for index in range(GATE_COUNT)
        if index * gate_size_cm <= hi and (index + 1) * gate_size_cm > lo
    )


def clamp_distance(value: float | None) -> float | None:
    """Distances stay in cm; out-of-range values are clamped (rule 1.4)."""
    if value is None:
        return None
    return max(0.0, value)


def in_zone(zone: ZoneConfig, distance_cm: float, margin_cm: float) -> bool:
    """Zone mask: ``distance in [near - margin, far + margin]`` (rule 2.1)."""
    return zone.near_cm - margin_cm <= distance_cm <= zone.far_cm + margin_cm


def default_zone(config: ConductorConfig, sensor_id: str) -> ZoneConfig | None:
    """The sensor's default zone for distance-less evidence (rule 2.3):
    the zone flagged ``fallback``, else the nearest zone."""
    zones = config.zones_for_sensor(sensor_id)
    if not zones:
        return None
    for zone in zones:
        if zone.fallback:
            return zone
    return min(zones, key=lambda z: z.near_cm)


def gate_frame(config: ConductorConfig, frame: SensorFrame) -> dict[str, tuple[bool, bool]]:
    """Per zone of the frame's sensor: ``(move_gated, still_gated)``.

    A frame with a distance outside every zone of its sensor contributes
    nothing — the target belongs to another zone, another sensor's
    territory, or is a ghost at an implausible range (rule 2.1).
    """
    margin = config.tunables.margin_cm
    move_d = clamp_distance(frame.moving_distance_cm)  # 1.4
    still_d = clamp_distance(frame.still_distance_cm)  # 1.4
    gates: dict[str, list[bool]] = {}
    for zone in config.zones_for_sensor(frame.sensor_id):
        gates[zone.zone_id] = [
            move_d is not None and in_zone(zone, move_d, margin),  # 2.1
            still_d is not None and in_zone(zone, still_d, margin),  # 2.1
        ]
    # 2.3: a target flag without a distance is attributed to the default
    # zone, keeping single-zone sensors working when the device momentarily
    # omits distance.
    fallback = default_zone(config, frame.sensor_id)
    if fallback is not None:
        if frame.has_moving_target and move_d is None:
            gates[fallback.zone_id][0] = True  # 2.3
        if frame.has_still_target and still_d is None:
            gates[fallback.zone_id][1] = True  # 2.3
    return {zone_id: (g[0], g[1]) for zone_id, g in gates.items()}
