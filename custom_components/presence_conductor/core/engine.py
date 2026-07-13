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

from . import activity, evidence, fusion, gating, health, timers
from . import filter as filter_
from .belief import logit
from .events import (
    Event,
    RecordBaseline,
    SensorAvailability,
    SensorFrame,
    SetEnabled,
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
        #: Timer keys the engine believes are pending at the adapter.
        self._pending_timers: set[str] = set()
        #: Home-presence hysteresis latch (6.5).
        self._home_on: bool = False
        #: Timestamp of the last tick integration (rule 4.1's dt).
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
        state = self.state
        state.enabled = snapshot.enabled  # 7.2
        state.lam_home = self.lam_prior  # 6.5
        for sensor in self.config.sensors:
            state.sensors[sensor.sensor_id] = SensorState(
                available=snapshot.available.get(sensor.sensor_id, True)
            )
        for zone in self.config.zones:
            persisted = snapshot.baselines.get(zone.zone_id)
            zst = ZoneState(
                lam=self.lam_prior,  # 7.1: posteriors start at the prior
                move_baseline=ChannelStats(persisted.move_mu, persisted.move_sigma)
                if persisted
                else ChannelStats(t.default_mu, t.default_sigma),
                still_baseline=ChannelStats(persisted.still_mu, persisted.still_sigma)
                if persisted
                else ChannelStats(t.default_mu, t.default_sigma),
            )
            if persisted is not None:
                # 3.6: persisted per-gate floors; zones stored before
                # per-gate evidence existed simply have none.
                for index in sorted(persisted.gates):
                    gate = persisted.gates[index]
                    zst.gate_move_baselines[index] = ChannelStats(gate.move_mu, gate.move_sigma)
                    zst.gate_still_baselines[index] = ChannelStats(gate.still_mu, gate.still_sigma)
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
            evidence.ingest_frame(self, frame)
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
        match event:
            case SensorFrame():
                self._on_frame(event, now, plan)
            case Tick():
                self._on_tick(now, plan)
            case SensorAvailability():
                health.on_availability(self, event, now, plan)
                fusion.refresh(self, now)  # 6.3 exclusion may change fusion
            case SetEnabled():
                # 7.2: while off the engine keeps ingesting and updating
                # (re-enable is warm); only publication is suppressed, which
                # the plan's suppress_outputs flag carries.
                self.state.enabled = event.enabled
            case RecordBaseline():
                evidence.on_record_baseline(self, event, now, plan)  # 3.3
            case _:  # unknown event types are ignored
                pass
        return self._finish(plan)

    def on_timer(self, key: str, now: float) -> Plan:
        """A timer previously requested through a plan fired."""
        plan = self._plan()
        if key not in self._pending_timers:
            return self._finish(plan)  # stale or unknown timer key
        self._pending_timers.discard(key)
        if key.startswith(timers.SENSOR_STALE_PREFIX):
            health.on_stale(self, key.removeprefix(timers.SENSOR_STALE_PREFIX), now, plan)  # 1.3
            fusion.refresh(self, now)  # 6.3
        elif key.startswith(timers.MOTION_OFF_PREFIX):
            # 4.4: motion hold expiry.
            filter_.on_motion_off(self, key.removeprefix(timers.MOTION_OFF_PREFIX), now, plan)
        elif key.startswith(timers.BASELINE_END_PREFIX):
            # 3.3: calibration window closes.
            evidence.on_baseline_end(self, key.removeprefix(timers.BASELINE_END_PREFIX), now, plan)
        return self._finish(plan)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_frame(self, frame: SensorFrame, now: float, plan: Plan) -> None:
        if frame.sensor_id not in self.state.sensors:
            return  # unknown sensor: ignore
        health.on_frame(self, frame.sensor_id, now, plan)  # 1.3 first: recovery
        evidence.apply_frame(self, frame, now)  # 1.4, 2, 3.2-3.4
        filter_.on_frame(self, frame, now, plan)  # 4.2, 4.4
        # 6.5: occupancy raised by fast attack lifts home presence on the
        # same frame (dt=0: events never decay it).
        fusion.refresh(self, now)

    def _on_tick(self, now: float, plan: Plan) -> None:
        # 1.2: all time-driven behavior advances on ticks and event
        # timestamps; integration uses the actual elapsed time.
        last = self._last_advance if self._last_advance is not None else now
        dt = max(0.0, now - last)
        self._last_advance = now
        if dt > 0.0:
            for zone in self.config.zones:  # config order (7.3)
                zst = self.state.zones[zone.zone_id]
                filter_.tick_zone(self, zone, zst, dt, now, plan)  # 4.1
                activity.tick_zone(self, zone, zst, now)  # 5
        fusion.refresh(self, now, dt)  # 6 (home decay, 6.5)

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
