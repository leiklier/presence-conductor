"""Sensor staleness and availability (spec rule 1.3).

Each sensor carries a staleness watchdog timer, restarted on every frame.
When it fires while any of the sensor's zones is occupied — or when the
sensor's entities become unavailable — the zones enter UNKNOWN health:
their outputs hold their last state, the filter and FSM freeze, and room
fusion ignores them (6.3). Recovery is immediate on the next frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import timers
from .events import SensorAvailability
from .model import Health

if TYPE_CHECKING:
    from .engine import ConductorEngine
    from .plan import Plan


def on_frame(engine: ConductorEngine, sensor_id: str, now: float, plan: Plan) -> None:
    """A frame arrived: recover health and re-arm the watchdog (1.3)."""
    sensor = engine.state.sensors[sensor_id]
    sensor.last_frame_at = now
    sensor.available = True  # a frame implies live entities
    for zone in engine.config.zones_for_sensor(sensor_id):
        zst = engine.state.zones[zone.zone_id]
        if zst.health is Health.UNKNOWN:
            zst.health = Health.OK  # 1.3: recovery is immediate on the next frame
    plan.start_timer(timers.sensor_stale(sensor_id), engine.config.tunables.stale_after)  # 1.3


def on_availability(
    engine: ConductorEngine, event: SensorAvailability, now: float, plan: Plan
) -> None:
    """Entity availability changed (1.3)."""
    sensor = engine.state.sensors.get(event.sensor_id)
    if sensor is None:  # unknown sensor: ignore
        return
    sensor.available = event.available
    if not event.available:
        # 1.3: entities unavailable -> UNKNOWN immediately (whether or not
        # any zone is occupied — the sensor is plainly blind).
        for zone in engine.config.zones_for_sensor(event.sensor_id):
            engine.state.zones[zone.zone_id].health = Health.UNKNOWN
        plan.cancel_timer(timers.sensor_stale(event.sensor_id))  # no frames expected
    # available=True alone carries no data: recovery waits for the next
    # frame (1.3), which restarts the watchdog.


def on_stale(engine: ConductorEngine, sensor_id: str, now: float, plan: Plan) -> None:
    """The staleness watchdog fired: no frame for ``stale_after`` (1.3)."""
    zones = engine.config.zones_for_sensor(sensor_id)
    if any(engine.state.zones[z.zone_id].occupied for z in zones):
        # 1.3: silence *while occupied* means we are blind, not that the
        # occupant left — hold outputs, exclude from fusion (6.3).
        for zone in zones:
            engine.state.zones[zone.zone_id].health = Health.UNKNOWN
    else:
        # An empty sensor may be legitimately silent (deduplicated,
        # unchanged readings). Re-arm so late-blooming occupancy without
        # fresh frames is still caught within stale_after.
        plan.start_timer(timers.sensor_stale(sensor_id), engine.config.tunables.stale_after)
