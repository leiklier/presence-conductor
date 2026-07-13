"""Test harness for the pure engine: live-shaped config + drive helpers.

The default fixture mirrors the live apartment (docs/DECISION.md): three
MSR-2 sensors, where kontor carries two zones (desk + door) and the stue
room fuses two single-zone sensors (sofakrok + spisebord). All zones share
a calibrated noise floor of ``mu = 0.05, sigma = 0.05`` (normalized units)
and no statistic calibration, so the analytic fallback applies (rule 3.7,
``m0 = 0.3989, s0 = 0.5838`` for m = 1): raw energies map to raw scores as
``S = max(0, (raw/100 - 0.05) / 0.05)`` and to centered scores (rule 3.2)
as ``z = clamp((S - 0.3989) / 0.5838, -1, 6)``:

====  =====  =========================================
raw       z  meaning with default tunables
====  =====  =========================================
   5  -0.68  at baseline: absence evidence (rule 3.2)
  10   1.03  weak evidence
  13   2.06  motion trigger (z_motion = 2.0, rule 4.4)
  25   6.0   saturated (z_cap); attack candidate (4.2)
  35   6.0   saturated
====  =====  =========================================

The centered score of a baseline frame is Z_EMPTY = -0.683; the evidence
rate of a fully quiet zone is ``(k_move + k_still) * Z_EMPTY - k_bias``.
"""

from __future__ import annotations

from custom_components.presence_conductor.core.engine import ConductorEngine
from custom_components.presence_conductor.core.events import GATE_COUNT, Event, SensorFrame, Tick
from custom_components.presence_conductor.core.model import (
    ConductorConfig,
    GateBaselines,
    InitialSnapshot,
    RoomConfig,
    RoomState,
    SensorConfig,
    Tunables,
    ZoneBaselines,
    ZoneConfig,
    ZoneState,
)
from custom_components.presence_conductor.core.plan import BaselineRecorded, PassBy, Plan
from custom_components.presence_conductor.core.stats import onesided_max_stats

KONTOR = "kontor"
SOFAKROK = "sofakrok"
SPISEBORD = "spisebord"

DESK = "kontor_desk"
DOOR = "kontor_door"
SOFA = "sofakrok_zone"
BORD = "spisebord_zone"

#: Calibrated noise floor used by the default snapshot (normalized units).
MU = 0.05
SIGMA = 0.05

#: Analytic fallback statistic (rule 3.7) for m = 1, used by every zone of
#: the default snapshot (no empirical statistic calibration).
STAT_M0 = 0.39894228040139595
STAT_S0 = 0.5838193701035719


def centered_of_raw(raw_energy: float, *, mu: float = MU, sigma: float = SIGMA) -> float:
    """The centered score (rule 3.2) a gated frame with this raw 0-100
    energy produces under the default snapshot's floors."""
    s = max(0.0, (raw_energy / 100.0 - mu) / sigma)
    return min(6.0, max(-1.0, (s - STAT_M0) / STAT_S0))


#: Centered score of a channel sitting exactly at the noise floor (S = 0).
Z_EMPTY = (0.0 - STAT_M0) / STAT_S0


def centered_of(raw_s: float, m: int = 1) -> float:
    """Centered score (rule 3.2) of a raw statistic ``S`` under the analytic
    fallback for ``m`` owned gates (rule 3.7) and default caps."""
    m0, s0 = onesided_max_stats(m)
    return min(6.0, max(-1.0, (raw_s - m0) / s0))


DEFAULT_ZONES = (
    ZoneConfig(DESK, "Desk", KONTOR, room_id="kontor", near_cm=30, far_cm=150, fallback=True),
    ZoneConfig(DOOR, "Door", KONTOR, room_id="kontor", near_cm=220, far_cm=300),
    ZoneConfig(SOFA, "Sofakrok", SOFAKROK, room_id="stue", near_cm=30, far_cm=200, fallback=True),
    ZoneConfig(BORD, "Spisebord", SPISEBORD, room_id="stue", near_cm=50, far_cm=250, fallback=True),
)


def make_config(
    *,
    zones: tuple[ZoneConfig, ...] = DEFAULT_ZONES,
    **tunable_overrides: float,
) -> ConductorConfig:
    return ConductorConfig(
        sensors=(
            SensorConfig(KONTOR, "Kontor MSR-2"),
            SensorConfig(SOFAKROK, "Sofakrok MSR-2"),
            SensorConfig(SPISEBORD, "Spisebord MSR-2"),
        ),
        zones=zones,
        rooms=(RoomConfig("kontor", "Kontor"), RoomConfig("stue", "Stue")),
        tunables=Tunables(**tunable_overrides),
    )


def make_snapshot(
    config: ConductorConfig,
    *,
    frames: dict[str, SensorFrame] | None = None,
    available: dict[str, bool] | None = None,
    enabled: bool = True,
    mu: float = MU,
    sigma: float = SIGMA,
    gate_floors: dict[str, dict[int, GateBaselines]] | None = None,
) -> InitialSnapshot:
    return InitialSnapshot(
        frames=frames or {},
        available=available or {},
        baselines={
            z.zone_id: ZoneBaselines(
                mu, sigma, mu, sigma, gates=(gate_floors or {}).get(z.zone_id, {})
            )
            for z in config.zones
        },
        enabled=enabled,
    )


def calibrated_gate_floors(
    config: ConductorConfig, *, mu: float = MU, sigma: float = SIGMA
) -> dict[str, dict[int, GateBaselines]]:
    """Every gate of every zone calibrated like the aggregates (3.6), so the
    raw-energy-to-z table above applies to gate energies too."""
    gates = {index: GateBaselines(mu, sigma, mu, sigma) for index in range(GATE_COUNT)}
    return {zone.zone_id: gates for zone in config.zones}


def gate_tuple(
    peaks: dict[int, float | None] | None = None, *, fill: float | None = 100 * MU
) -> tuple[float | None, ...]:
    """A 9-gate raw-energy tuple: ``fill`` everywhere except the ``peaks``."""
    values: list[float | None] = [fill] * GATE_COUNT
    for index, value in (peaks or {}).items():
        values[index] = value
    return tuple(values)


def frame(
    sensor_id: str,
    *,
    move_d: float | None = None,
    still_d: float | None = None,
    move_e: float | None = None,
    still_e: float | None = None,
    moving: bool = False,
    still: bool = False,
    gate_move: tuple[float | None, ...] | None = None,
    gate_still: tuple[float | None, ...] | None = None,
) -> SensorFrame:
    """A coalesced frame (rule 1.1). Energies are raw 0-100."""
    return SensorFrame(
        sensor_id=sensor_id,
        moving_distance_cm=move_d,
        still_distance_cm=still_d,
        move_energy=move_e,
        still_energy=still_e,
        has_target=moving or still,
        has_moving_target=moving,
        has_still_target=still,
        gate_move=gate_move,
        gate_still=gate_still,
    )


def quiet(sensor_id: str) -> SensorFrame:
    """Everything back at the noise floor: energies == mu, no targets."""
    return frame(sensor_id, move_e=100 * MU, still_e=100 * MU)


class Harness:
    """Drives the engine with explicit time and mirrors the adapter's timers."""

    def __init__(
        self,
        config: ConductorConfig | None = None,
        snapshot: InitialSnapshot | None = None,
        *,
        now: float = 0.0,
        auto_start: bool = True,
    ) -> None:
        self.config = config if config is not None else make_config()
        self.snapshot = snapshot if snapshot is not None else make_snapshot(self.config)
        self.engine = ConductorEngine(self.config, self.snapshot)
        self.now = now
        #: timer key -> absolute deadline, mirroring the adapter's timers.
        self.deadlines: dict[str, float] = {}
        #: (timestamp, event) for every event any plan emitted.
        self.emitted: list[tuple[float, object]] = []
        #: plans that asked for calibration persistence.
        self.persist_count = 0
        if auto_start:
            self._absorb(self.engine.start(now))

    @property
    def state(self):
        return self.engine.state

    def zone(self, zone_id: str) -> ZoneState:
        return self.engine.state.zones[zone_id]

    def room(self, room_id: str) -> RoomState:
        return self.engine.state.rooms[room_id]

    # -- driving --------------------------------------------------------

    def submit(self, event: Event, at: float | None = None) -> Plan:
        if at is not None:
            assert at >= self.now, "monotonic time must not go backwards"
            self.now = at
        plan = self.engine.submit(event, self.now)
        self._absorb(plan)
        return plan

    def send_frame(self, sensor_id: str, *, at: float | None = None, **frame_kw) -> Plan:
        return self.submit(frame(sensor_id, **frame_kw), at=at)

    def occupy(self, sensor_id: str, distance: float = 100.0, *, at: float | None = None) -> Plan:
        """Strong gated move evidence: the confirmed fast attack (4.2)
        flips occupied — two qualifying frames ``attack_gap_min`` apart."""
        gap = self.config.tunables.attack_gap_min
        self.send_frame(sensor_id, move_d=distance, move_e=35.0, moving=True, at=at)
        return self.send_frame(
            sensor_id, move_d=distance, move_e=35.0, moving=True, at=self.now + gap
        )

    def tick(self, at: float | None = None) -> Plan:
        return self.submit(Tick(), at=self.now + 1.0 if at is None else at)

    def fire_timer(self, key: str, at: float | None = None) -> Plan:
        """Fire a timer as the adapter would (at its deadline by default)."""
        if at is None:
            at = max(self.now, self.deadlines.get(key, self.now))
        self.deadlines.pop(key, None)
        assert at >= self.now
        self.now = at
        plan = self.engine.on_timer(key, at)
        self._absorb(plan)
        return plan

    def run(self, seconds: float) -> None:
        """Advance time in 1 s ticks, firing due timers at their deadlines."""
        end = self.now + seconds
        while self.now + 1e-9 < end:
            self.step_to(min(end, self.now + 1.0))

    def sustain(self, sensor_id: str, seconds: float, **frame_kw) -> None:
        """One frame + one tick per second for ``seconds`` seconds."""
        for _ in range(round(seconds)):
            self.send_frame(sensor_id, **frame_kw)
            self.step_to(self.now + 1.0)

    def sustain_quiet(self, sensor_id: str, seconds: float) -> None:
        """Baseline frames (z = 0): absence evidence per rule 3.2."""
        for _ in range(round(seconds)):
            self.submit(quiet(sensor_id))
            self.step_to(self.now + 1.0)

    def step_to(self, target: float) -> None:
        """Fire timers due at/before ``target`` (earliest first), then tick."""
        while True:
            due = [(when, key) for key, when in self.deadlines.items() if when <= target + 1e-9]
            if not due:
                break
            when, key = min(due)
            self.fire_timer(key, at=max(self.now, when))
        self.tick(at=target)

    # -- inspection ------------------------------------------------------

    def pass_bys(self) -> list[PassBy]:
        return [e for _, e in self.emitted if isinstance(e, PassBy)]

    def baseline_events(self) -> list[BaselineRecorded]:
        return [e for _, e in self.emitted if isinstance(e, BaselineRecorded)]

    def fingerprint(self) -> tuple:
        """Full observable state, for determinism comparisons (rule 7.3)."""
        s = self.engine.state
        zones = tuple(
            (
                zone_id,
                z.lam,
                z.occupied,
                z.motion,
                str(z.activity),
                z.dwell_seconds,
                str(z.health),
                z.move_baseline.mu,
                z.move_baseline.sigma,
                z.still_baseline.mu,
                z.still_baseline.sigma,
                tuple(sorted((i, g.mu, g.sigma) for i, g in z.gate_move_baselines.items())),
                tuple(sorted((i, g.mu, g.sigma) for i, g in z.gate_still_baselines.items())),
            )
            for zone_id, z in s.zones.items()
        )
        rooms = tuple(
            (room_id, r.occupied, r.motion, r.confidence, str(r.activity), r.settled)
            for room_id, r in s.rooms.items()
        )
        return (
            zones,
            rooms,
            s.lam_home,
            s.anyone_home,
            s.home_confidence,
            tuple(self.emitted),
            self.persist_count,
        )

    def _absorb(self, plan: Plan) -> None:
        for start in plan.timer_starts:
            self.deadlines[start.key] = self.now + start.delay
        for cancel in plan.timer_cancels:
            self.deadlines.pop(cancel.key, None)
        self.emitted.extend((self.now, event) for event in plan.events)
        if plan.persist_calibration:
            self.persist_count += 1
