"""Diagnostics download: the full engine state, on demand.

This is where the per-frame numerics live — lambdas, confidences,
baselines, dwell, runtime evidence paths. Entities deliberately do not
carry them (every attribute change writes a recorder row); the download
gives operators the complete picture exactly when they ask for it.
"""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .controller import PresenceConductorController


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return the full engine state for one config entry."""
    controller: PresenceConductorController | None = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if controller is None:
        return {"configured": False}
    state = controller.state
    return {
        "configured": True,
        "enabled": state.enabled,
        "home_lambda": round(state.lam_home, 4),
        "home_confidence": state.home_confidence,
        "anyone_home": state.anyone_home,
        "zones": {
            zone_id: {
                "lambda": round(zst.lam, 4),
                "confidence": round(zst.confidence, 4),
                "health": zst.health.value,
                "activity": zst.activity.value,
                "occupied": zst.occupied,
                "motion": zst.motion,
                "dwell_seconds": round(zst.dwell_seconds, 1),
                "move_baseline": [
                    round(zst.move_baseline.mu, 4),
                    round(zst.move_baseline.sigma, 4),
                ],
                "still_baseline": [
                    round(zst.still_baseline.mu, 4),
                    round(zst.still_baseline.sigma, 4),
                ],
                "calibration": {
                    "status": controller.calibration_diagnostic(zone_id).status.value,
                    **controller.calibration_diagnostic(zone_id).attributes(),
                },
            }
            for zone_id, zst in state.zones.items()
        },
        "rooms": {
            room_id: {
                "occupied": room.occupied,
                "motion": room.motion,
                "activity": room.activity.value if room.activity is not None else None,
                "settled": room.settled,
                "confidence": room.confidence,
            }
            for room_id, room in state.rooms.items()
        },
        "sensors": {
            sensor_id: {"available": sensor.available}
            for sensor_id, sensor in state.sensors.items()
        },
    }
