"""Configuration and state models for the estimation core (spec §0).

All identifiers (``sensor_id``, ``zone_id``, ``room_id``) are opaque strings
to the core. The Home Assistant adapter uses entity/device ids, but the core
never inspects them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from .belief import sigmoid
from .events import SensorFrame


class Activity(StrEnum):
    """Per-zone activity classification (rule 5.1)."""

    EMPTY = "empty"
    PASSING = "passing"
    ACTIVE = "active"
    SETTLED = "settled"


#: Severity order for room fusion: ``settled > active > passing > empty``
#: (rule 6.2).
ACTIVITY_SEVERITY: dict[Activity, int] = {
    Activity.EMPTY: 0,
    Activity.PASSING: 1,
    Activity.ACTIVE: 2,
    Activity.SETTLED: 3,
}


class Health(StrEnum):
    """Zone health (rule 1.3). While UNKNOWN, outputs hold their last state
    and room fusion ignores the zone (rule 6.3)."""

    OK = "ok"
    UNKNOWN = "unknown"


#: LD2410 gate size at the default 0.75 m range resolution (rule 2.4). The
#: 0.2 m resolution mode makes it 20.
DEFAULT_GATE_SIZE_CM = 75.0


@dataclass(frozen=True, slots=True)
class SensorConfig:
    """One physical mmWave device (§0)."""

    sensor_id: str
    name: str
    #: Distance-gate size of this sensor's radar, in cm (rule 2.4).
    gate_size_cm: float = DEFAULT_GATE_SIZE_CM


@dataclass(frozen=True, slots=True)
class ZoneConfig:
    """A spatial slice of one sensor's beam (§0): the estimation unit."""

    zone_id: str
    name: str
    sensor_id: str
    #: Room this zone fuses into (§6). Rooms may span sensors.
    room_id: str
    near_cm: float
    far_cm: float
    #: Default zone for evidence whose target flag is set but whose distance
    #: is None (rule 2.3). At most one per sensor; without a flagged zone the
    #: sensor's nearest zone (smallest ``near_cm``) is the default.
    fallback: bool = False


@dataclass(frozen=True, slots=True)
class RoomConfig:
    """A named set of zones, possibly from different sensors (§0)."""

    room_id: str
    name: str


@dataclass(frozen=True, slots=True)
class Tunables:
    """Behavioral tuning knobs. Durations in seconds, distances in cm,
    energies/probabilities in [0, 1] (normalized, rule 1.4).

    All defaults come from docs/ENGINE_SPEC.md; nothing reads config at
    update time (rule 7.3).
    """

    #: Zone mask slack absorbing interpolated-distance smoothing (rule 2.1).
    margin_cm: float = 30.0
    #: Silence window before an occupied sensor's zones go UNKNOWN (rule 1.3).
    stale_after: float = 30.0
    #: Expected Tick cadence (rule 1.2). Informational for the adapter; the
    #: core integrates with actual event timestamps.
    tick_interval: float = 1.0
    #: Noise-floor sigma is never below this (rule 3.1), in normalized
    #: energy units. The spec fixes no default; 0.02 = two raw energy points.
    sigma_min: float = 0.02
    #: Baseline used for zones with no persisted calibration (spec is
    #: silent; deliberately wide so uncalibrated zones are conservative).
    default_mu: float = 0.10
    default_sigma: float = 0.10
    #: Evidence-score cap (rule 3.2).
    z_cap: float = 6.0
    #: Per-second evidence weights (rule 3.2), scaled by tick interval.
    k_move: float = 1.0
    k_still: float = 0.6
    k_absence: float = 0.4
    #: Departure-hazard relaxation time constant (rule 4.1).
    tau_decay: float = 90.0
    #: Empty-state prior probability (rule 4.1).
    p_prior: float = 0.02
    #: Fast-attack trigger and floor (rule 4.2).
    z_attack: float = 3.0
    p_attack: float = 0.95
    #: Occupied hysteresis thresholds (rule 4.3), as probabilities.
    theta_on: float = 0.80
    theta_off: float = 0.20
    #: Motion channel trigger and hold (rule 4.4).
    z_motion: float = 1.5
    motion_hold: float = 5.0
    #: Posterior clamp (rule 4.5), as probabilities.
    p_min: float = 0.001
    p_max: float = 0.999
    #: Background adaptation (rule 3.4).
    p_background: float = 0.05
    t_background: float = 600.0
    tau_background: float = 3600.0
    #: Default RecordBaseline collection window (rule 3.3).
    baseline_duration: float = 120.0
    #: Activity FSM timing (rule 5.1).
    t_dwell: float = 45.0
    t_settle: float = 30.0
    #: Home-presence decay and hysteresis (rule 6.5). Hysteresis thresholds
    #: follow rule 4.3's defaults.
    tau_home: float = 1200.0
    theta_home_on: float = 0.80
    theta_home_off: float = 0.20


@dataclass(frozen=True, slots=True)
class ConductorConfig:
    """Full static configuration handed to the engine (rule 7.3)."""

    sensors: tuple[SensorConfig, ...]
    zones: tuple[ZoneConfig, ...]
    #: Explicit room declarations (naming). Rooms referenced by zones but not
    #: declared here still fuse; they simply have no display name.
    rooms: tuple[RoomConfig, ...] = ()
    tunables: Tunables = field(default_factory=Tunables)

    def sensor(self, sensor_id: str) -> SensorConfig:
        return next(s for s in self.sensors if s.sensor_id == sensor_id)

    def zone(self, zone_id: str) -> ZoneConfig:
        return next(z for z in self.zones if z.zone_id == zone_id)

    def zone_or_none(self, zone_id: str) -> ZoneConfig | None:
        return next((z for z in self.zones if z.zone_id == zone_id), None)

    def zones_for_sensor(self, sensor_id: str) -> tuple[ZoneConfig, ...]:
        return tuple(z for z in self.zones if z.sensor_id == sensor_id)

    def zones_in_room(self, room_id: str) -> tuple[ZoneConfig, ...]:
        return tuple(z for z in self.zones if z.room_id == room_id)

    def room_ids(self) -> tuple[str, ...]:
        """Declared rooms first, then any room referenced only by zones."""
        ordered = [r.room_id for r in self.rooms]
        ordered += [z.room_id for z in self.zones if z.room_id not in ordered]
        return tuple(dict.fromkeys(ordered))


@dataclass(slots=True)
class ChannelStats:
    """Robust noise floor of one energy channel (rule 3.1), normalized units."""

    mu: float
    sigma: float


@dataclass(slots=True)
class BaselineRecording:
    """Samples collected during a RecordBaseline window (rules 3.3, 3.6).

    Per-gate samples are keyed by owned gate index (rule 2.4); a gate that
    reported nothing during the window simply has no key.
    """

    move_samples: list[float] = field(default_factory=list)
    still_samples: list[float] = field(default_factory=list)
    gate_move_samples: dict[int, list[float]] = field(default_factory=dict)
    gate_still_samples: dict[int, list[float]] = field(default_factory=dict)


@dataclass(slots=True)
class ZoneState:
    """Mutable runtime state of a zone (owned by the engine).

    The published outputs (§0) are ``occupied``, ``motion``, ``activity``,
    ``probability``, ``dwell_seconds`` and ``health``; the adapter reads
    them after every ``submit``/``on_timer`` call. The remaining fields are
    engine internals.
    """

    #: Occupancy posterior in log-odds (§0).
    lam: float
    #: Per-channel noise floors (rule 3.1); mutated by calibration (3.3)
    #: and background adaptation (3.4).
    move_baseline: ChannelStats
    still_baseline: ChannelStats
    #: Per-gate noise floors (rule 3.6), keyed by gate index. Entries are
    #: created on demand — persisted calibration (3.3), background
    #: adaptation (3.4) — so a gate without one scores against the
    #: ``Tunables`` defaults and zones that never see gate data stay empty.
    gate_move_baselines: dict[int, ChannelStats] = field(default_factory=dict)
    gate_still_baselines: dict[int, ChannelStats] = field(default_factory=dict)
    # -- published outputs (§0) ----------------------------------------
    occupied: bool = False
    motion: bool = False
    activity: Activity = Activity.EMPTY
    dwell_seconds: float = 0.0
    health: Health = Health.OK
    # -- evidence from the most recent frame (rules 2, 3.2). Values persist
    # between frames: the device deduplicates identical publishes, so the
    # last report remains the best estimate of the current signal.
    z_move: float = 0.0
    z_still: float = 0.0
    move_gated: bool = False
    still_gated: bool = False
    #: Whether the channel's current z came from gate evidence (rule 2.6);
    #: the motion channel (4.4) keys off this to ignore the sensor-global
    #: ``has_moving_target`` flag while gates say where the mover is.
    move_from_gates: bool = False
    still_from_gates: bool = False
    # -- activity FSM internals (rule 5) --------------------------------
    occupied_since: float | None = None
    peak_probability: float = 0.0
    still_dominant_since: float | None = None
    move_dominant_since: float | None = None
    # -- background adaptation internals (rule 3.4) ----------------------
    below_since: float | None = None
    last_adapt_at: float | None = None
    #: Active RecordBaseline collection, if any (rule 3.3).
    recording: BaselineRecording | None = None

    @property
    def probability(self) -> float:
        """Sigmoid of the posterior (§0). Stale while health is UNKNOWN
        (rule 1.3); the adapter marks it accordingly."""
        return sigmoid(self.lam)


@dataclass(slots=True)
class SensorState:
    """Mutable runtime state of a sensor (owned by the engine)."""

    available: bool = True
    last_frame_at: float | None = None


@dataclass(slots=True)
class RoomState:
    """Published room fusion outputs (§6). ``None`` means unknown: every
    member zone is in UNKNOWN health (rule 6.3)."""

    occupied: bool | None = None
    #: Any healthy member zone's motion channel (rule 6.2).
    motion: bool | None = None
    probability: float | None = None
    activity: Activity | None = None
    settled: bool | None = None


@dataclass(slots=True)
class EngineState:
    """Aggregate engine state. The adapter reads this to publish entities;
    it never re-derives outputs (§0)."""

    enabled: bool = True
    zones: dict[str, ZoneState] = field(default_factory=dict)
    sensors: dict[str, SensorState] = field(default_factory=dict)
    rooms: dict[str, RoomState] = field(default_factory=dict)
    #: Home-level log-odds of "someone is in the apartment" (rule 6.5).
    lam_home: float = 0.0
    #: Published home presence; ``None`` when all zones are unhealthy (6.5).
    anyone_home: bool | None = False
    home_probability: float | None = None


@dataclass(frozen=True, slots=True)
class GateBaselines:
    """Persisted calibration for one gate of a zone (rule 3.6), normalized
    units."""

    move_mu: float
    move_sigma: float
    still_mu: float
    still_sigma: float


@dataclass(frozen=True, slots=True)
class ZoneBaselines:
    """Persisted calibration for one zone (rule 3.3), normalized units."""

    move_mu: float
    move_sigma: float
    still_mu: float
    still_sigma: float
    #: Optional per-gate floors (rule 3.6), keyed by gate index. Baselines
    #: persisted before per-gate evidence existed simply have no gates —
    #: the schema is backward compatible in both directions.
    gates: Mapping[int, GateBaselines] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InitialSnapshot:
    """Point-in-time world state used to seed the engine at startup (7.1).

    The adapter gathers this from current entity states so the engine can
    adopt reality instead of cold-starting every restart.
    """

    #: Current coalesced frame per sensor; ``None`` when no state is known.
    frames: Mapping[str, SensorFrame | None] = field(default_factory=dict)
    #: Sensor availability; missing sensors default to available.
    available: Mapping[str, bool] = field(default_factory=dict)
    #: Persisted calibration per zone (rule 3.3); missing zones use the
    #: ``Tunables`` defaults.
    baselines: Mapping[str, ZoneBaselines] = field(default_factory=dict)
    enabled: bool = True
