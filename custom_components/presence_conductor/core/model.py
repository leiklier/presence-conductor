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
    #: Device energy quantization step, normalized (rules 3.1, 3.4): LD2410
    #: energies are integers 0-100, so one step is 0.01. Half a step guards
    #: the scale estimates against grid bias.
    energy_quantum: float = 0.01
    #: Baseline used for zones with no persisted calibration (spec is
    #: silent; deliberately wide so uncalibrated zones are conservative).
    default_mu: float = 0.10
    default_sigma: float = 0.10
    #: Centered-score clamp (rule 3.2): ``z_cap`` above, ``z_neg_cap`` below
    #: (absence evidence is bounded so odd baselines cannot instantly unlatch).
    z_cap: float = 6.0
    z_neg_cap: float = 1.0
    #: Floor for the statistic calibration's deviation ``s0`` (rule 3.7);
    #: the analytic reference deviation for the path's gate count floors it
    #: as well — a short window may recentre but never sharpen the score.
    stat_sigma_min: float = 0.3
    #: Minimum calibration rows before an empirical statistic is accepted
    #: (rule 3.7); shorter windows keep the analytic fallback. Counted over
    #: *distinct observations* (3.1) / *fresh rows* (3.7), never raw ticks.
    stat_min_rows: int = 60
    #: Clamp on the estimated integrated autocorrelation time (rule 3.7).
    tau_int_max: float = 25.0
    #: Observation-clock windows (rule 3.8): positive evidence integrates
    #: for at most obs_budget after its observation; non-positive evidence
    #: and the absence bias for at most obs_hold. Past both, silence.
    obs_budget: float = 1.0
    obs_hold: float = 5.0
    #: Gate evidence (2.5-2.6) is experimental until real gate-path timing
    #: is captured and validated (2.6); the aggregate path is the
    #: field-validated default.
    use_gate_evidence: bool = False
    #: Per-second evidence weights and the always-subtracted absence bias
    #: (rule 3.2): ``E[u | calibrated empty] <= -k_bias < 0``. The gains
    #: bound the *variance* of the empty walk (see 3.2); genuine-entry
    #: latching speed is set by ``u_cap``, not the gains.
    k_move: float = 0.5
    k_still: float = 0.3
    #: Absence margin subtracted from every observed centered score
    #: (3.2/3.8): E[score] = 0 on calibrated empty noise, so the expected
    #: observed rate is exactly -(k_move + k_still) * k_bias = -0.4/s for
    #: any gate count.
    k_bias: float = 0.5
    #: Upward cap on the evidence rate (rule 3.2): one wild sample held for
    #: a second must not out-accumulate a genuine entry.
    u_cap: float = 3.0
    #: Departure-hazard relaxation time constant (rule 4.1).
    tau_decay: float = 90.0
    #: Empty-state prior confidence (rule 4.1).
    p_prior: float = 0.02
    #: Fast-attack tail probability (parts per million), confirmation and
    #: floor (rule 4.2). Candidacy thresholds on the analytic tail of the
    #: raw statistic — per-observation ``P_H0(S >= threshold) =
    #: attack_tail_ppm * 1e-6`` — never on a window-estimated tail.
    attack_tail_ppm: float = 100.0
    attack_confirm: int = 2
    attack_gap_min: float = 0.3
    attack_gap_max: float = 3.0
    p_attack: float = 0.95
    #: Occupied hysteresis thresholds (rule 4.3), as confidences.
    theta_on: float = 0.80
    theta_off: float = 0.20
    #: Motion channel trigger (centered units) and hold (rule 4.4).
    z_motion: float = 2.0
    motion_hold: float = 5.0
    #: Belief clamp (rule 4.5), as confidences.
    p_min: float = 0.001
    p_max: float = 0.999
    #: How long a frozen distance stays usable after its target flag was
    #: last on (rule 2.7).
    distance_hold: float = 30.0
    #: Background adaptation (rule 3.4).
    p_background: float = 0.05
    t_background: float = 600.0
    tau_background: float = 3600.0
    #: Default RecordBaseline collection window (rule 3.3), sized to the
    #: measured ~2.5 s empty reporting cadence: 300 s yields ~120 fresh
    #: observations per channel, comfortably above ``stat_min_rows``
    #: (120 s yielded ~48 and rejected the still path).
    baseline_duration: float = 300.0
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


@dataclass(frozen=True, slots=True)
class BaselineRow:
    """One tick-aligned calibration sample (rule 3.3): the sensor's cached
    normalized energies, aggregate and per-gate, as a single snapshot."""

    move_e: float | None
    still_e: float | None
    gate_move: tuple[float | None, ...] | None
    gate_still: tuple[float | None, ...] | None
    #: Whether each channel's observation counter advanced since the
    #: previous row (rules 3.1, 3.7): held/deduplicated rows repeat one
    #: measurement and are excluded from the statistics.
    move_fresh: bool = True
    still_fresh: bool = True
    #: Whether any supported entity from this sensor was observed since
    #: the previous tick-aligned row. Unlike the row itself, this proves
    #: that the cached plateau was re-observed rather than merely held.
    frame_fresh: bool = True


#: Calibration coverage statuses (rule 3.3).
class Coverage(StrEnum):
    """Per-path verdict of one RecordBaseline window (rule 3.3)."""

    #: Enough distinct observations for a floor and fresh rows for a
    #: statistic: the path's calibration is in the committed candidate.
    CALIBRATED = "calibrated"
    #: The channel's values span at most one quantum: the empty signal
    #: never moves over enough certified sensor observations. Floor
    #: ``(median, sigma_min)``, analytic statistic.
    QUIESCENT = "quiescent"
    #: The channel never reported during the window: nothing to calibrate,
    #: previous values kept. Optional paths preserve their old family;
    #: required paths cannot commit with no data.
    NO_DATA = "no_data"
    #: Data present but too few fresh/distinct observations: blocks the
    #: whole commit when the path is required.
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ChannelCoverage:
    """Coverage accounting for one evidence path (rule 3.3): the verdict
    plus the counts that produced it."""

    status: Coverage
    #: Tick-aligned rows carrying data for this path (3.3).
    rows: int
    #: Rows on which the channel's observation counter advanced (1.1).
    fresh: int
    #: Distinct observations after collapsing consecutive duplicates (3.1).
    distinct: int
    #: Tick rows certified by at least one real sensor-frame observation
    #: during the row interval (3.3). Ticks alone never increment this.
    observed: int = 0
    #: Human-readable rejection reason; ``None`` unless REJECTED.
    reason: str | None = None


@dataclass(slots=True)
class BaselineRecording:
    """Rows collected during a RecordBaseline window (rules 3.3, 3.6, 3.7).

    Rows are sampled on the tick clock, never per entity change (3.3):
    entity-change sampling weights samples by publish frequency and tears
    gate tuples across radar frames. Aligned rows are also what lets 3.7
    score the post-aggregation statistic per sample.
    """

    rows: list[BaselineRow] = field(default_factory=list)
    #: Observation counters at the previous row, for freshness tagging.
    last_move_obs: int = 0
    last_still_obs: int = 0
    last_frame_obs: int = 0


@dataclass(slots=True)
class ZoneState:
    """Mutable runtime state of a zone (owned by the engine).

    The published outputs (§0) are ``occupied``, ``motion``, ``activity``,
    ``confidence``, ``dwell_seconds`` and ``health``; the adapter reads
    them after every ``submit``/``on_timer`` call. The remaining fields are
    engine internals.
    """

    #: Occupancy belief accumulator (§0), log-odds form.
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
    #: Statistic calibration (rule 3.7): empty-room ``(m0, s0, c0)`` of the
    #: raw statistic per channel + path, keyed ``move_agg`` / ``move_gate``
    #: / ``still_agg`` / ``still_gate``. A missing key means the analytic
    #: Gaussian fallback applies.
    stat_cal: dict[str, StatBaseline] = field(default_factory=dict)
    # -- published outputs (§0) ----------------------------------------
    occupied: bool = False
    motion: bool = False
    activity: Activity = Activity.EMPTY
    dwell_seconds: float = 0.0
    health: Health = Health.OK
    # -- evidence from the most recent frame (rules 2, 3.2): centered
    # scores, negative when the channel sits below its empty-room mean.
    # Values persist between frames: the device deduplicates identical
    # publishes, so the last report remains the best estimate of the
    # current signal.
    z_move: float = 0.0
    z_still: float = 0.0
    move_gated: bool = False
    still_gated: bool = False
    #: Whether the channel's current score came from gate evidence (rule
    #: 2.6); the motion channel (4.4) keys off this to ignore the sensor-
    #: global ``has_moving_target`` flag while gates say where the mover is.
    move_from_gates: bool = False
    still_from_gates: bool = False
    #: Whether this frame's raw move statistic exceeds the analytic attack
    #: tail threshold (rule 4.2 candidacy; set by evidence ingest).
    attack_candidate: bool = False
    #: Fast-attack confirmation chain (rule 4.2): fresh qualifying move
    #: observations counted so far, and when the last one arrived.
    attack_count: int = 0
    attack_last: float | None = None
    #: Evidence path that started the current confirmation chain. Gate and
    #: aggregate tail events may not confirm one another (rule 4.2).
    attack_path: str | None = None
    # -- activity FSM internals (rule 5) --------------------------------
    occupied_since: float | None = None
    peak_confidence: float = 0.0
    still_dominant_since: float | None = None
    move_dominant_since: float | None = None
    # -- background adaptation internals (rule 3.4) ----------------------
    below_since: float | None = None
    last_adapt_at: float | None = None
    #: Active RecordBaseline collection, if any (rule 3.3).
    recording: BaselineRecording | None = None

    @property
    def confidence(self) -> float:
        """Sigmoid of the belief (§0) — a monotone score, not a calibrated
        probability (rule 8.7). Stale while health is UNKNOWN (rule 1.3);
        the adapter marks it accordingly."""
        return sigmoid(self.lam)


@dataclass(slots=True)
class SensorState:
    """Mutable runtime state of a sensor (owned by the engine)."""

    available: bool = True
    last_frame_at: float | None = None
    #: Last normalized energies seen from this sensor, cached for the
    #: tick-aligned calibration rows (rule 3.3).
    last_move_e: float | None = None
    last_still_e: float | None = None
    last_gate_move: tuple[float | None, ...] | None = None
    last_gate_still: tuple[float | None, ...] | None = None
    #: When each target flag was last observed on (rule 2.7 distance hold).
    move_flag_at: float | None = None
    still_flag_at: float | None = None
    #: Observation clock (rules 1.1, 3.8): last seen frame counters, the
    #: engine time each channel was last observed, and the per-frame
    #: move-energy freshness flag (4.2).
    move_obs: int = 0
    still_obs: int = 0
    frame_obs: int = 0
    move_energy_obs: int = 0
    move_obs_at: float | None = None
    still_obs_at: float | None = None
    move_energy_fresh: bool = False


@dataclass(slots=True)
class RoomState:
    """Published room fusion outputs (§6). ``None`` means unknown: every
    member zone is in UNKNOWN health (rule 6.3)."""

    occupied: bool | None = None
    #: Any healthy member zone's motion channel (rule 6.2).
    motion: bool | None = None
    confidence: float | None = None
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
    #: Home-level belief that someone is in the apartment (rule 6.5).
    lam_home: float = 0.0
    #: Published home presence; ``None`` when all zones are unhealthy (6.5).
    anyone_home: bool | None = False
    home_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class GateBaselines:
    """Persisted calibration for one gate of a zone (rule 3.6), normalized
    units."""

    move_mu: float
    move_sigma: float
    still_mu: float
    still_sigma: float


@dataclass(frozen=True, slots=True)
class StatBaseline:
    """Statistic calibration (rule 3.7) for one channel + path: empty-room
    mean and deviation of the raw statistic ``S``, plus the residual mean
    of the clamped centered score (``c0``, rule 3.2)."""

    mu: float
    sigma: float
    clip_mu: float = 0.0
    #: Estimated integrated autocorrelation time of the calibrated empty
    #: process (rule 3.7); runtime scores divide by it (3.2).
    tau: float = 1.0


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
    #: Optional statistic calibration (rule 3.7), keyed ``move_agg`` /
    #: ``move_gate`` / ``still_agg`` / ``still_gate``. Missing keys (and
    #: pre-3.7 baselines) fall back to the analytic values.
    stats: Mapping[str, StatBaseline] = field(default_factory=dict)


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
