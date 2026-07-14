"""Structured timer identifiers shared by the engine and its tests.

The adapter owns the actual clocks: it starts/cancels timers as requested by
the plan and calls ``ConductorEngine.on_timer(key, now)`` when one fires.
Starting an already-pending key restarts it.
"""

from __future__ import annotations

SENSOR_STALE_PREFIX = "sensor_stale:"
MOTION_OFF_PREFIX = "motion_off:"
BASELINE_END_PREFIX = "baseline_end:"


def sensor_stale(sensor_id: str) -> str:
    """Staleness watchdog (rule 1.3): restarted on every frame."""
    return f"{SENSOR_STALE_PREFIX}{sensor_id}"


def motion_off(zone_id: str) -> str:
    """Motion hold (rule 4.4): fires ``motion_hold`` after the last motion
    evidence."""
    return f"{MOTION_OFF_PREFIX}{zone_id}"


def baseline_end(zone_id: str) -> str:
    """End of a RecordBaseline collection window (rule 3.3)."""
    return f"{BASELINE_END_PREFIX}{zone_id}"


GUIDED_PHASE_END_PREFIX = "guided_phase_end:"


def guided_phase_end(zone_id: str) -> str:
    """Timer key closing one labeled full-calibration phase."""

    return f"{GUIDED_PHASE_END_PREFIX}{zone_id}"
