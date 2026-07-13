"""Input events for the estimation engine.

The adapter is responsible for:

- coalescing a sensor's entity states into one :class:`SensorFrame` and
  submitting it whenever any underlying entity changes (rule 1.1);
- delivering a periodic :class:`Tick` (rule 1.2, default every 1.0 s);
- monotonic time: every call to ``ConductorEngine.submit`` passes ``now``
  (monotonic seconds). Events carry no timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Number of LD2410 distance gates (``g0..g8``, rules 1.1, 2.4). Fixed by
#: the radar in both resolution modes; the resolution changes the gate
#: *size* (``SensorConfig.gate_size_cm``), never the count.
GATE_COUNT = 9


@dataclass(frozen=True, slots=True)
class Event:
    """Base class for all engine input events."""


@dataclass(frozen=True, slots=True)
class SensorFrame(Event):
    """One coalesced snapshot of a sensor's entity states (rule 1.1).

    Distances are ``None`` when the device reports no target of that kind.
    Energies are raw 0-100 (``None`` when unknown); the core normalizes them
    to [0, 1] on ingest (rule 1.4).
    """

    sensor_id: str
    moving_distance_cm: float | None = None
    still_distance_cm: float | None = None
    move_energy: float | None = None
    still_energy: float | None = None
    has_target: bool = False
    has_moving_target: bool = False
    has_still_target: bool = False
    #: Per-gate energies from engineering mode (rule 1.1), consumed by rules
    #: 2.4-2.6: ``GATE_COUNT`` elements when present, raw 0-100, individual
    #: gates ``None`` when unknown. The whole tuple is ``None`` when the
    #: sensor has no gate entities configured or has never reported them.
    gate_move: tuple[float | None, ...] | None = None
    gate_still: tuple[float | None, ...] | None = None
    #: Observation clock (rule 1.1): monotonic counters the adapter advances
    #: whenever an entity of the channel receives a reported update —
    #: including same-value forced re-publications. The engine keys held-
    #: value handling (3.8) and attack freshness (4.2) on counter advances
    #: (inequality with the last seen value), never on value comparison.
    move_obs: int = 0
    still_obs: int = 0
    #: Move-energy observations only (move_energy or gate-move entities);
    #: consumed by the fast-attack confirmation (4.2).
    move_energy_obs: int = 0


@dataclass(frozen=True, slots=True)
class Tick(Event):
    """Periodic clock event (rule 1.2). All time-driven behavior advances
    on ticks and event timestamps; the core never reads a clock."""


@dataclass(frozen=True, slots=True)
class SensorAvailability(Event):
    """A sensor's entities became (un)available (rule 1.3)."""

    sensor_id: str
    available: bool


@dataclass(frozen=True, slots=True)
class SetEnabled(Event):
    """Global enable switch (rule 7.2)."""

    enabled: bool


@dataclass(frozen=True, slots=True)
class RecordBaseline(Event):
    """Start a baseline calibration window for one zone (rule 3.3).

    ``duration`` is the collection window in seconds; ``None`` uses
    ``Tunables.baseline_duration`` (default 120 s). The operator asserts the
    zone is empty for the window.
    """

    zone_id: str
    duration: float | None = None
