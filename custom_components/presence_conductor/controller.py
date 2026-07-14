"""The Presence Conductor controller: the adapter between HA and the engine.

Single-writer strategy: the engine is pure and synchronous, and every input
— entity state changes, ticks, timer expiries, entity commands — reaches it
through :meth:`PresenceConductorController.submit` (or the timer path), which
runs entirely inside one event-loop callback with no awaits between the
engine call and the plan application. Unlike sonos-conductor there is no
queue/actor: applying a plan needs no service calls, so nothing can suspend
mid-event and re-enter the engine.

Responsibilities (docs/ENGINE_SPEC.md):

- Coalesce each sensor's entity states into ``SensorFrame`` events and
  submit one whenever any underlying entity changes (rule 1.1). Frames are
  cheap; there is no debouncing.
- Deliver a periodic ``Tick`` every ``tick_interval`` seconds (rule 1.2)
  and run the engine's timers via ``async_call_later`` (restart semantics:
  starting a pending key cancels the old clock first).
- Track availability of the *required* entities — the two energy roles, the
  minimum evidence path — and submit ``SensorAvailability`` on transitions
  (rule 1.3).
- Persist calibration into ``entry.options[CONF_BASELINES]`` when a plan
  asks for it (rule 3.3). The options listener in ``__init__.py`` ignores
  baselines-only diffs, so this write never causes a reload loop.
- Publish: entities read ``engine.state`` and are nudged over the
  dispatcher. While the engine is disabled (rule 7.2) plans carry
  ``suppress_outputs`` and the state signal is withheld — zone/room/home
  entities freeze for consumers while the engine keeps updating underneath;
  re-enabling publishes once. A second, always-sent control signal keeps
  the enabled switch and the diagnostics sensor honest.
- Engine time is ``time.monotonic()`` everywhere; the engine never reads a
  clock (rule 7.3).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Protocol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_STATE_REPORTED
from homeassistant.core import (
    CALLBACK_TYPE,
    EventStateChangedData,
    EventStateReportedData,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.core import (
    Event as HAEvent,
)
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.start import async_at_started

from .calibration import (
    CalibrationDiagnostic,
    CalibrationManager,
    baseline_payload,
    full_calibration_payload,
)
from .config import sensor_entities
from .const import DOMAIN, GATE_ROLES
from .core.events import Event, SensorAvailability, Tick
from .core.model import ConductorConfig, EngineState, InitialSnapshot
from .core.plan import (
    BaselineRecorded,
    FullCalibrationProgress,
    FullCalibrationRecorded,
    PassBy,
    Plan,
)
from .observation import FRAME_ROLES, SensorView, build_view, required_available

_LOGGER = logging.getLogger(__name__)

#: HA bus event fired for every engine pass-by (rule 5.2).
EVENT_PASS_BY = f"{DOMAIN}_pass_by"
#: HA bus event fired for every RecordBaseline outcome (rule 3.3) —
#: success or rejection, with the per-path coverage verdicts.
EVENT_BASELINE_RECORDED = f"{DOMAIN}_baseline_recorded"
#: HA bus event for guided phase changes and held-out validation results.
EVENT_FULL_CALIBRATION = f"{DOMAIN}_full_calibration"


class EngineProtocol(Protocol):
    """The engine surface the controller drives (real or test double)."""

    state: EngineState
    owned_gates: dict[str, tuple[int, ...]]

    def start(self, now: float) -> Plan: ...

    def submit(self, event: Event, now: float) -> Plan: ...

    def on_timer(self, key: str, now: float) -> Plan: ...


class PresenceConductorController:
    """Single-writer adapter: HA inputs -> engine -> plan application."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        config: ConductorConfig,
        snapshot: InitialSnapshot,
        engine_factory: type[EngineProtocol] | Any,
    ) -> None:
        self.hass = hass
        self.entry = entry
        self.config = config
        #: Zone/room/home entity refresh; withheld while suppressed (7.2).
        self.signal = f"{DOMAIN}_{entry.entry_id}_updated"
        #: Enabled switch + diagnostics refresh; always sent, so operators can
        #: see (and undo) the disabled state — 7.2 governs published outputs,
        #: not the controls that govern the engine itself.
        self.signal_control = f"{DOMAIN}_{entry.entry_id}_control"
        # The engine is created eagerly so entities added during platform
        # forwarding can always read ``engine.state``.
        self.engine: EngineProtocol = engine_factory(config, snapshot)

        self._started = False
        self._unsub_start: CALLBACK_TYPE | None = None
        self._unsubs: list[CALLBACK_TYPE] = []
        self._tick_unsub: CALLBACK_TYPE | None = None
        self._timers: dict[str, CALLBACK_TYPE] = {}

        #: Room display names (rooms referenced only by zones keep their id).
        self._room_names: dict[str, str] = {r.room_id: r.name for r in config.rooms}

        # Entity maps derived from the options contract.
        self._roles_by_sensor: dict[str, dict[str, str]] = sensor_entities(entry.options)
        self._sensor_role_by_entity: dict[str, tuple[str, str]] = {}
        for sensor_id, roles in self._roles_by_sensor.items():
            for role in (*FRAME_ROLES, *GATE_ROLES):
                if (entity_id := roles.get(role)) is not None:
                    self._sensor_role_by_entity[entity_id] = (sensor_id, role)

        #: Cached per-sensor views (the frame source, rule 1.1).
        self._views: dict[str, SensorView] = {
            sensor_id: build_view(hass, roles) for sensor_id, roles in self._roles_by_sensor.items()
        }
        self._calibration = CalibrationManager(
            hass,
            entry,
            config,
            self.engine,
            self._roles_by_sensor,
            snapshot.baselines,
        )

    # -- public API for entities ---------------------------------------------

    @property
    def state(self) -> EngineState:
        return self.engine.state

    def room_name(self, room_id: str) -> str:
        return self._room_names.get(room_id, room_id)

    def calibration_diagnostic(self, zone_id: str) -> CalibrationDiagnostic:
        """Current calibration readiness for one zone."""
        return self._calibration.diagnostic(zone_id)

    def pass_by_signal(self, zone_id: str) -> str:
        """Dispatcher signal carrying :class:`PassBy` events for one zone."""
        return f"{DOMAIN}_{self.entry.entry_id}_pass_by_{zone_id}"

    def room_pass_by_signal(self, room_id: str) -> str:
        """Dispatcher signal carrying :class:`PassBy` events of every member
        zone of one room (rule 5.2 routed through §6 membership)."""
        return f"{DOMAIN}_{self.entry.entry_id}_room_pass_by_{room_id}"

    def baseline_signal(self, zone_id: str) -> str:
        """Dispatcher signal carrying :class:`BaselineRecorded` outcomes
        for one zone (rule 3.3 observability)."""
        return f"{DOMAIN}_{self.entry.entry_id}_baseline_{zone_id}"

    def clear_calibration_issue(self) -> None:
        """Remove this entry's warning after a successful entry unload."""
        self._calibration.clear_issue()

    @callback
    def submit(self, event: Event) -> None:
        """Feed one event through the engine and apply the resulting plan.

        Event-loop only. Safe before :meth:`async_start` (entity restore
        during platform forwarding submits commands); the engine mutates
        state and the plan is applied like any other.
        """
        self._apply_plan(self.engine.submit(event, time.monotonic()))

    # -- lifecycle -------------------------------------------------------------

    @callback
    def async_start(self) -> None:
        """Start the engine once HA is running (immediately if it already is).

        Startup order (7.1): refresh the cached views right before
        ``engine.start`` so the frame source matches the world the engine
        adopts, then subscribe and start the tick clock.
        """
        self._unsub_start = async_at_started(self.hass, self._on_started)

    @callback
    def _on_started(self, _hass: HomeAssistant) -> None:
        self._unsub_start = None
        self._started = True
        for sensor_id, roles in self._roles_by_sensor.items():
            self._views[sensor_id] = build_view(self.hass, roles)
        self._subscribe()
        self._tick_unsub = async_track_time_interval(
            self.hass,
            self._on_tick_interval,
            timedelta(seconds=self.config.tunables.tick_interval),  # 1.2
        )
        self._apply_plan(self.engine.start(time.monotonic()))
        self._calibration.sync_issue()

    async def async_stop(self) -> None:
        """Cancel subscriptions, the tick clock and all engine timers."""
        self._started = False
        if self._unsub_start is not None:
            self._unsub_start()
            self._unsub_start = None
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()
        if self._tick_unsub is not None:
            self._tick_unsub()
            self._tick_unsub = None
        for unsub in self._timers.values():
            unsub()
        self._timers.clear()

    # -- HA inputs ---------------------------------------------------------------

    @callback
    def _subscribe(self) -> None:
        if self._sensor_role_by_entity:
            self._unsubs.append(
                async_track_state_change_event(
                    self.hass, sorted(self._sensor_role_by_entity), self._on_entity_event
                )
            )
            # Exact same-state/same-attribute writes are EVENT_STATE_REPORTED,
            # not EVENT_STATE_CHANGED. HA requires an event filter for this
            # high-volume event type; using the tracked-entity map keeps the
            # listener O(1) and independent of ESPHome force_update settings.
            self._unsubs.append(
                self.hass.bus.async_listen(
                    EVENT_STATE_REPORTED,
                    self._on_entity_reported,
                    event_filter=self._is_tracked_state_report,
                )
            )

    @callback
    def _is_tracked_state_report(self, data: EventStateReportedData) -> bool:
        """Filter the high-volume state_reported bus at dispatch time."""
        return data["entity_id"] in self._sensor_role_by_entity

    @callback
    def _on_entity_event(self, event: HAEvent[EventStateChangedData]) -> None:
        """Rule 1.1: fold a changed entity and classify sample freshness."""
        old_state = event.data["old_state"]
        new_state = event.data["new_state"]
        # A same-value write with changed attributes is ambiguous metadata,
        # not proof that the radar value was sampled again. Exact repeated
        # reports arrive on state_reported; force_update arrives here with
        # equal attributes. Both are explicit measurements.
        measurement = new_state is not None and (
            old_state is None
            or new_state.state != old_state.state
            or new_state.attributes == old_state.attributes
        )
        self._on_entity_measurement(event.data["entity_id"], new_state, measurement=measurement)

    @callback
    def _on_entity_reported(self, event: HAEvent[EventStateReportedData]) -> None:
        """Rule 1.1: an unchanged state write is still a measurement."""
        self._on_entity_measurement(
            event.data["entity_id"], event.data["new_state"], measurement=True
        )

    @callback
    def _on_entity_measurement(
        self, entity_id: str, new_state: State | None, *, measurement: bool
    ) -> None:
        """Fold one tracked HA report and submit the sensor's complete view."""
        mapped = self._sensor_role_by_entity.get(entity_id)
        if mapped is None:
            return
        sensor_id, role = mapped
        view = self._views[sensor_id]
        view.update(role, new_state, measurement=measurement)
        available = required_available(self.hass, self._roles_by_sensor[sensor_id])
        if available != view.available:
            view.available = available
            self.submit(SensorAvailability(sensor_id, available))  # 1.3
        if available:
            # While the required entities are down the sensor is blind; a
            # frame would count as recovery in the engine (1.3), so none is
            # submitted until availability returns.
            self.submit(view.frame(sensor_id))

    @callback
    def _on_tick_interval(self, _now: datetime) -> None:
        self.submit(Tick())  # 1.2

    # -- plan application ----------------------------------------------------------

    @callback
    def _apply_plan(self, plan: Plan) -> None:
        """Apply one engine plan: timers, persistence, events, publish."""
        for start in plan.timer_starts:
            self._start_timer(start.key, start.delay)
        for cancel in plan.timer_cancels:
            self._cancel_timer(cancel.key)
        if plan.persist_calibration:
            # Honored even while disabled: calibration is operator-requested
            # data, not a published transition (rule 7.2).
            changed_zones = plan.persist_calibration_zones or {
                zone.zone_id for zone in self.config.zones
            }
            self._calibration.persist(changed_zones)
        for event in plan.events:
            # The plan is empty while suppressed (Plan.emit drops events,
            # rule 7.2); everything here is meant to be published.
            if isinstance(event, PassBy):
                self.hass.bus.async_fire(
                    EVENT_PASS_BY,
                    {
                        "zone_id": event.zone_id,
                        "peak_confidence": round(event.peak_confidence, 4),
                        "duration": round(event.duration, 2),
                    },
                )
                async_dispatcher_send(self.hass, self.pass_by_signal(event.zone_id), event)
                # Every pass-by also reaches the zone's room (§6 membership):
                # the room event entity is the consumer surface, the zone
                # entity the opt-in diagnostic.
                room_id = self.config.zone(event.zone_id).room_id
                async_dispatcher_send(self.hass, self.room_pass_by_signal(room_id), event)
            elif isinstance(event, BaselineRecorded):
                # 3.3 observability: the calibration outcome — success or
                # rejection — reaches the bus, the per-zone event entity
                # and the log; persistence rode persist_calibration above
                # (set only on a committed window).
                payload = baseline_payload(event)
                self.hass.bus.async_fire(EVENT_BASELINE_RECORDED, payload)
                async_dispatcher_send(self.hass, self.baseline_signal(event.zone_id), event)
                if event.success:
                    _LOGGER.info(
                        "Baseline calibration for %s committed: %s",
                        event.zone_id,
                        payload["coverage"],
                    )
                else:
                    _LOGGER.warning(
                        "Baseline calibration for %s REJECTED — previous calibration kept, "
                        "nothing persisted: %s",
                        event.zone_id,
                        payload["coverage"],
                    )
            elif isinstance(event, (FullCalibrationProgress, FullCalibrationRecorded)):
                payload = full_calibration_payload(event)
                self.hass.bus.async_fire(EVENT_FULL_CALIBRATION, payload)
                if isinstance(event, FullCalibrationRecorded):
                    level = logging.INFO if event.success else logging.WARNING
                    _LOGGER.log(level, "Full calibration for %s: %s", event.zone_id, payload)
        self._publish(plan.suppress_outputs)

    @callback
    def _publish(self, suppressed: bool) -> None:
        async_dispatcher_send(self.hass, self.signal_control)
        if not suppressed:
            # 7.2: while disabled, zone/room/home entities are not notified —
            # their state freezes for consumers. The first non-suppressed
            # plan after re-enable notifies everything once.
            async_dispatcher_send(self.hass, self.signal)

    # -- timers -----------------------------------------------------------------

    @callback
    def _start_timer(self, key: str, delay: float) -> None:
        """(Re)start an engine timer: starting a pending key restarts it."""
        self._cancel_timer(key)

        @callback
        def _fire(_now: datetime) -> None:
            self._timers.pop(key, None)
            self._apply_plan(self.engine.on_timer(key, time.monotonic()))

        self._timers[key] = async_call_later(self.hass, delay, _fire)

    @callback
    def _cancel_timer(self, key: str) -> None:
        unsub = self._timers.pop(key, None)
        if unsub is not None:
            unsub()
