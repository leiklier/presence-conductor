"""A scriptable engine double for controller/entity tests.

Duck-types the real ``ConductorEngine`` surface — ``state``, ``start(now)``,
``submit(event, now)``, ``on_timer(key, now)`` — so the adapter can be
tested in isolation. It records every event and timer fire it receives; the
plans it returns are scripted per-call via :meth:`script`, defaulting to an
empty plan whose ``suppress_outputs`` mirrors ``state.enabled`` (rule 7.2).

The one behavioral concession: ``SetEnabled`` mutates ``state.enabled``,
because the enabled switch and the publish gate read it back.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from custom_components.presence_conductor.core.belief import logit, sigmoid
from custom_components.presence_conductor.core.events import Event, SetEnabled
from custom_components.presence_conductor.core.model import (
    Activity,
    ChannelStats,
    ConductorConfig,
    EngineState,
    Health,
    InitialSnapshot,
    RoomState,
    SensorState,
    ZoneState,
)
from custom_components.presence_conductor.core.plan import EmittedEvent, Plan


class FakeEngine:
    """Records inputs, returns scripted plans, seeds state from config."""

    def __init__(self, config: ConductorConfig, snapshot: InitialSnapshot) -> None:
        self.config = config
        self.snapshot = snapshot
        t = config.tunables
        self.state = EngineState()
        self.state.lam_home = logit(t.p_prior)
        self.state.home_probability = sigmoid(self.state.lam_home)
        for sensor in config.sensors:
            self.state.sensors[sensor.sensor_id] = SensorState(
                available=snapshot.available.get(sensor.sensor_id, True)
            )
        for zone in config.zones:
            zst = ZoneState(
                lam=logit(t.p_prior),
                move_baseline=ChannelStats(t.default_mu, t.default_sigma),
                still_baseline=ChannelStats(t.default_mu, t.default_sigma),
            )
            if not self.state.sensors[zone.sensor_id].available:
                zst.health = Health.UNKNOWN
            self.state.zones[zone.zone_id] = zst
        for room_id in config.room_ids():
            room = self.state.rooms[room_id] = RoomState()
            room.occupied = False
            room.probability = t.p_prior
            room.activity = Activity.EMPTY
            room.settled = False

        #: Every event received, in order.
        self.events: list[Event] = []
        #: Every timer key fired through ``on_timer``, in order.
        self.timer_fires: list[str] = []
        #: ``now`` values passed to start().
        self.start_calls: list[float] = []
        self._pending: set[str] = set()
        self._scripted: deque[Plan] = deque()

    # -- scripting -----------------------------------------------------------

    def plan(
        self,
        *,
        events: Iterable[EmittedEvent] = (),
        starts: Iterable[tuple[str, float]] = (),
        cancels: Iterable[str] = (),
        persist: bool = False,
        suppress: bool | None = None,
    ) -> Plan:
        """Build a plan bound to this engine's pending-timer registry."""
        suppress_outputs = (not self.state.enabled) if suppress is None else suppress
        plan = Plan(self._pending, suppress_outputs=suppress_outputs)
        for event in events:
            plan.emit(event)
        for key, delay in starts:
            plan.start_timer(key, delay)
        for key in cancels:
            plan.cancel_timer(key)
        plan.persist_calibration = persist
        return plan

    def script(self, plan: Plan) -> None:
        """Queue the plan to return from the next engine call."""
        self._scripted.append(plan)

    def events_of(self, event_type: type) -> list[Event]:
        return [event for event in self.events if isinstance(event, event_type)]

    def _next_plan(self) -> Plan:
        if self._scripted:
            return self._scripted.popleft()
        return self.plan()

    # -- engine surface --------------------------------------------------------

    def start(self, now: float) -> Plan:
        self.start_calls.append(now)
        return self._next_plan()

    def submit(self, event: Event, now: float) -> Plan:
        self.events.append(event)
        if isinstance(event, SetEnabled):
            self.state.enabled = event.enabled  # 7.2 (read back by the switch)
        return self._next_plan()

    def on_timer(self, key: str, now: float) -> Plan:
        self.timer_fires.append(key)
        return self._next_plan()
