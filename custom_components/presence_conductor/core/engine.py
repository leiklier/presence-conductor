"""The estimation engine: a pure, synchronous event processor.

``submit(event, now)`` and ``on_timer(key, now)`` mutate
:class:`~.model.EngineState` and return a :class:`~.plan.Plan` carrying
emitted events, timer requests, and persistence flags. The engine never
performs I/O, never reads a clock, and never sleeps — docs/ENGINE_SPEC.md
is the behavioral contract this class implements.

This module owns state, seeding, and dispatch; the behavior lives in
feature modules that take the engine as their first parameter:
:mod:`.gating`/:mod:`.evidence` (rules 1.4, 2, 3), :mod:`.filter` (rule 4),
:mod:`.activity` (rule 5), :mod:`.fusion` (rule 6), and :mod:`.health`
(rule 1.3).

Determinism (7.3): the same event sequence with the same timestamps yields
the same outputs. All tunables are read from config at construction (the
log-odds thresholds below are precomputed); there is no module-level
mutable state, and every iteration follows config declaration order.
"""

from __future__ import annotations

from . import activity, emissions, evidence, fusion, gating, guided, health, stats, timers
from . import filter as filter_
from .belief import logit
from .events import (
    AdvanceFullCalibration,
    CancelCalibration,
    Event,
    RecordBaseline,
    SensorAvailability,
    SensorFrame,
    SetEnabled,
    StartFullCalibration,
    Tick,
)
from .model import (
    ChannelStats,
    ConductorConfig,
    EngineState,
    Health,
    InitialSnapshot,
    RoomState,
    SensorState,
    ZoneState,
)
from .plan import Plan


class ConductorEngine:
    """Deterministic core of Presence Conductor."""

    def __init__(self, config: ConductorConfig, snapshot: InitialSnapshot) -> None:
        self.config = config
        t = config.tunables
        # 7.3: thresholds derived from tunables once, at construction.
        self.lam_prior = logit(t.p_prior)  # 4.1
        self.lam_attack = logit(t.p_attack)  # 4.2
        self.lam_on = logit(t.theta_on)  # 4.3
        self.lam_off = logit(t.theta_off)  # 4.3
        self.lam_min = logit(t.p_min)  # 4.5
        self.lam_max = logit(t.p_max)  # 4.5
        self.lam_home_on = logit(t.theta_home_on)  # 6.5
        self.lam_home_off = logit(t.theta_home_off)  # 6.5
        #: Gate ownership per zone (rule 2.4), derived from config once at
        #: construction (7.3) like the thresholds above.
        self.owned_gates: dict[str, tuple[int, ...]] = {
            zone.zone_id: gating.owned_gates(
                zone, config.sensor(zone.sensor_id).gate_size_cm, t.margin_cm
            )
            for zone in config.zones
        }
        #: Fast-attack candidacy thresholds on the raw statistic (rule 4.2):
        #: per zone, ``(gate path, aggregate path)`` — analytic tail values,
        #: never taken from a calibration window.
        tail = t.attack_tail_ppm * 1e-6
        self.attack_thresholds: dict[str, tuple[float, float]] = {
            zone.zone_id: (
                stats.attack_threshold(len(self.owned_gates[zone.zone_id]), tail),
                stats.attack_threshold(1, tail),
            )
            for zone in config.zones
        }
        #: Timer keys the engine believes are pending at the adapter.
        self._pending_timers: set[str] = set()
        #: Home-presence hysteresis latch (6.5).
        self._home_on: bool = False
        #: Timestamp of the last chronological advance (rule 4.1's dt).
        self._last_advance: float | None = None
        self.state = EngineState()
        self._seed(snapshot)

    # ------------------------------------------------------------------
    # Startup (rule 7.1)
    # ------------------------------------------------------------------

    def _seed(self, snapshot: InitialSnapshot) -> None:
        """Adopt current entity states (rule 7.1). Timeless: timestamps and
        timers are stamped by :meth:`start`."""
        t = self.config.tunables
        sigma_floor = evidence.floor_sigma_min(t.sigma_min, t.energy_quantum)
        state = self.state
        state.enabled = snapshot.enabled  # 7.2
        state.lam_home = self.lam_prior  # 6.5
        for sensor in self.config.sensors:
            state.sensors[sensor.sensor_id] = SensorState(
                available=snapshot.available.get(sensor.sensor_id, True)
            )
        for zone in self.config.zones:
            persisted = snapshot.baselines.get(zone.zone_id)
            calibration_context_valid = (
                persisted is not None
                and (persisted.sensor_id is None or persisted.sensor_id == zone.sensor_id)
                and (
                    persisted.floor_fingerprint is None
                    or persisted.floor_fingerprint == stats.floor_calibration_fingerprint(t)
                )
            )
            trusted = persisted if calibration_context_valid else None
            zst = ZoneState(
                lam=self.lam_prior,  # 7.1: beliefs start at the prior
                move_baseline=ChannelStats(trusted.move_mu, max(sigma_floor, trusted.move_sigma))
                if trusted
                else ChannelStats(t.default_mu, max(sigma_floor, t.default_sigma)),
                still_baseline=ChannelStats(trusted.still_mu, max(sigma_floor, trusted.still_sigma))
                if trusted
                else ChannelStats(t.default_mu, max(sigma_floor, t.default_sigma)),
            )
            if trusted is not None:
                persisted = trusted
                owned = self.owned_gates[zone.zone_id]
                gate_context_valid = (
                    persisted.gate_indices is None or persisted.gate_indices == owned
                ) and (
                    persisted.gate_size_cm is None
                    or persisted.gate_size_cm == self.config.sensor(zone.sensor_id).gate_size_cm
                )
                # 3.6: persisted per-gate floors; zones stored before
                # compatibility metadata existed simply fall back to the
                # aggregate path until a current complete family commits.
                move_indices = {index for index, gate in persisted.gates.items() if gate.has_move}
                still_indices = {index for index, gate in persisted.gates.items() if gate.has_still}
                move_complete = (
                    set(owned).issubset(move_indices)
                    if persisted.gate_indices is None
                    else move_indices == set(owned)
                )
                still_complete = (
                    set(owned).issubset(still_indices)
                    if persisted.gate_indices is None
                    else still_indices == set(owned)
                )
                if gate_context_valid and (persisted.gate_indices is None or move_complete):
                    for index in sorted(move_indices) if persisted.gate_indices is None else owned:
                        gate = persisted.gates[index]
                        zst.gate_move_baselines[index] = ChannelStats(
                            gate.move_mu, max(sigma_floor, gate.move_sigma)
                        )
                    zst.gate_move_ready = bool(owned) and move_complete
                if gate_context_valid and (persisted.gate_indices is None or still_complete):
                    for index in sorted(still_indices) if persisted.gate_indices is None else owned:
                        gate = persisted.gates[index]
                        zst.gate_still_baselines[index] = ChannelStats(
                            gate.still_mu, max(sigma_floor, gate.still_sigma)
                        )
                    zst.gate_still_ready = bool(owned) and still_complete
                # 3.7: persisted statistic calibration; missing keys (and
                # pre-3.7 baselines) fall back to the analytic values.
                for key in sorted(persisted.stats):
                    cal = persisted.stats[key]
                    expected = stats.calibration_fingerprint(key, owned, t)
                    path_ready = not key.endswith("_gate") or (
                        zst.gate_move_ready if key == "move_gate" else zst.gate_still_ready
                    )
                    if path_ready and (cal.fingerprint is None or cal.fingerprint == expected):
                        zst.stat_cal[key] = cal
                profile = persisted.occupied_profile
                if profile is not None and profile.path in {"aggregate", "gate"}:
                    try:
                        # Adapter parsing enforces the production sample
                        # minimum; the pure core also rejects structurally
                        # invalid programmatic snapshots.
                        emissions.validate_persisted_profile(profile, min_rows=1)
                    except ValueError:
                        profile = None
                if profile is not None and profile.path in {"aggregate", "gate"}:
                    profile_ready = profile.path == "aggregate" or (
                        zst.gate_move_ready and zst.gate_still_ready
                    )
                    expected = guided.profile_fingerprint(
                        self.config, zone, owned, zst, profile.path
                    )
                    if profile_ready and profile.fingerprint == expected:
                        zst.occupied_profile = profile
                        zst.last_validation = profile.validation
            if not state.sensors[zone.sensor_id].available:
                zst.health = Health.UNKNOWN  # 1.3
            state.zones[zone.zone_id] = zst
        for room_id in self.config.room_ids():
            state.rooms[room_id] = RoomState()
        # Ingest the snapshot frames as initial evidence so the first ticks
        # integrate reality, then apply the 7.1 exception: zones whose
        # sensor currently reports a gated target start at theta_on -
        # someone plainly there should not wait out a cold start. Snapshot
        # frames may carry gate data; under gate precedence (2.6) the gated
        # flags are spatial, so only zones whose own gates are elevated adopt.
        for sensor in self.config.sensors:
            frame = snapshot.frames.get(sensor.sensor_id)
            if frame is None:
                continue
            # now=None (2.7): flag recency unknown at seed, so only flag-on
            # distances gate. Observation counters adopt without recency
            # (1.1): channels stay silent until the first live frame.
            sensor_state = state.sensors[sensor.sensor_id]
            sensor_state.move_obs = frame.move_obs
            sensor_state.still_obs = frame.still_obs
            sensor_state.frame_obs = frame.frame_obs
            sensor_state.move_energy_obs = frame.move_energy_obs
            evidence.ingest_frame(self, frame, None)
            for zone in self.config.zones_for_sensor(sensor.sensor_id):
                zst = state.zones[zone.zone_id]
                gated_target = (frame.has_moving_target and zst.move_gated) or (
                    frame.has_still_target and zst.still_gated
                )
                if gated_target and zst.health is Health.OK:
                    zst.lam = self.lam_on  # 7.1
                    zst.occupied = True  # 4.3: at theta_on

    def start(self, now: float) -> Plan:
        """Stamp startup time, arm watchdogs, publish the seeded state."""
        plan = self._plan()
        self._last_advance = now
        for sensor in self.config.sensors:
            sst = self.state.sensors[sensor.sensor_id]
            if sst.available:
                sst.last_frame_at = now
                # 1.3: arm the staleness watchdog.
                plan.start_timer(
                    timers.sensor_stale(sensor.sensor_id), self.config.tunables.stale_after
                )
        for zone in self.config.zones:
            zst = self.state.zones[zone.zone_id]
            if zst.occupied:
                # 7.1 adoption enters the FSM at PASSING; t_dwell promotes
                # it if they stay (5.1).
                activity.on_occupied(zst, now)
            evidence.update_background_clock(self, zst, now)  # 3.4
        fusion.refresh(self, now)  # 6
        return self._finish(plan)

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def submit(self, event: Event, now: float) -> Plan:
        """Process one event and return the resulting plan."""
        plan = self._plan()
        dt = self._advance(now, plan)  # 4.1: chronology before the event's effect
        match event:
            case SensorFrame():
                self._on_frame(event, now, plan)
            case Tick():
                # 4.1 made ticks pure clock: integration already happened in
                # _advance. What remains is the tick-aligned calibration
                # sampling (3.3).
                evidence.collect_baseline_rows(self, now)
                guided.collect_rows(self, now)
            case SensorAvailability():
                health.on_availability(self, event, now, plan)  # 1.3
            case SetEnabled():
                # 7.2: while off the engine keeps ingesting and updating
                # (re-enable is warm); only publication is suppressed, which
                # the plan's suppress_outputs flag carries.
                self.state.enabled = event.enabled
            case RecordBaseline():
                evidence.on_record_baseline(self, event, now, plan)  # 3.3
            case StartFullCalibration():
                guided.start_full(self, event, now, plan)
            case AdvanceFullCalibration():
                guided.advance(self, event, now, plan)
            case CancelCalibration():
                guided.cancel(self, event, plan)
            case _:  # unknown event types are ignored
                pass
        fusion.refresh(self, now, dt)  # 6 (home decay over dt, 6.5)
        return self._finish(plan)

    def on_timer(self, key: str, now: float) -> Plan:
        """A timer previously requested through a plan fired."""
        plan = self._plan()
        if key not in self._pending_timers:
            return self._finish(plan)  # stale or unknown timer key
        self._pending_timers.discard(key)
        dt = self._advance(now, plan)  # 4.1: chronology before the timer's effect
        if key.startswith(timers.SENSOR_STALE_PREFIX):
            health.on_stale(self, key.removeprefix(timers.SENSOR_STALE_PREFIX), now, plan)  # 1.3
        elif key.startswith(timers.MOTION_OFF_PREFIX):
            # 4.4: motion hold expiry.
            filter_.on_motion_off(self, key.removeprefix(timers.MOTION_OFF_PREFIX), now, plan)
        elif key.startswith(timers.BASELINE_END_PREFIX):
            # 3.3: calibration window closes.
            evidence.on_baseline_end(self, key.removeprefix(timers.BASELINE_END_PREFIX), now, plan)
            guided.on_baseline_closed(self, key.removeprefix(timers.BASELINE_END_PREFIX), plan)
        elif key.startswith(timers.GUIDED_PHASE_END_PREFIX):
            guided.end_phase(self, key.removeprefix(timers.GUIDED_PHASE_END_PREFIX), now, plan)
        fusion.refresh(self, now, dt)  # 6 / 6.3 / 6.5
        return self._finish(plan)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _advance(self, now: float, plan: Plan) -> float:
        """Rule 4.1: integrate every zone from the last advance to ``now``
        using the evidence that was in force during that interval — before
        the current input installs anything new. Returns the elapsed dt."""
        last = self._last_advance if self._last_advance is not None else now
        dt = max(0.0, now - last)
        self._last_advance = max(last, now)
        if dt > 0.0:
            for zone in self.config.zones:  # config order (7.3)
                zst = self.state.zones[zone.zone_id]
                filter_.advance_zone(self, zone, zst, dt, now, plan)  # 4.1
                activity.tick_zone(self, zone, zst, now)  # 5
        return dt

    def _on_frame(self, frame: SensorFrame, now: float, plan: Plan) -> None:
        if frame.sensor_id not in self.state.sensors:
            return  # unknown sensor: ignore
        sensor = self.state.sensors[frame.sensor_id]
        # A cached frame emitted for an attribute-only/invalid HA event is
        # useful for no new evidence, but it is not proof that the radar is
        # alive. Only a real observation epoch may recover/re-arm health.
        measurement_fresh = frame.frame_obs != sensor.frame_obs
        health.on_frame(self, frame.sensor_id, now, plan, measurement_fresh)  # 1.3
        evidence.apply_frame(self, frame, now)  # 1.4, 2, 3.2-3.4
        filter_.on_frame(self, frame, now, plan)  # 4.2, 4.4
        # 6.5: submit()'s fusion refresh runs after this, so occupancy
        # raised by fast attack lifts home presence on the same frame.

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _plan(self) -> Plan:
        return Plan(self._pending_timers, suppress_outputs=not self.state.enabled)  # 7.2

    def _finish(self, plan: Plan) -> Plan:
        # 7.2: reflect the post-event enabled state, so SetEnabled(False)
        # suppresses its own plan and SetEnabled(True) publishes again.
        plan.suppress_outputs = not self.state.enabled
        return plan
