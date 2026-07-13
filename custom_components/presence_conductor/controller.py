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

import asyncio
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import (
    CALLBACK_TYPE,
    EventStateChangedData,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.core import (
    Event as HAEvent,
)
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import EntityPlatform
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.start import async_at_started
from homeassistant.util import slugify

from .config import baselines_from_options, sensor_entities
from .const import (
    CONF_BASELINES,
    DOMAIN,
    ENERGY_ROLES,
    GATE_COUNT,
    GATE_MOVE_ROLES,
    GATE_ROLES,
    GATE_STILL_ROLES,
    ROLE_MOVE_ENERGY,
    ROLE_MOVING_DISTANCE,
    ROLE_MOVING_TARGET,
    ROLE_STILL_DISTANCE,
    ROLE_STILL_ENERGY,
    ROLE_STILL_TARGET,
    ROLE_TARGET,
)
from .core.events import Event, SensorAvailability, SensorFrame, Tick
from .core.model import ChannelStats, ConductorConfig, EngineState, InitialSnapshot
from .core.plan import PassBy, Plan

_LOGGER = logging.getLogger(__name__)

#: HA bus event fired for every engine pass-by (rule 5.2).
EVENT_PASS_BY = f"{DOMAIN}_pass_by"

UNAVAILABLE_STATES = (STATE_UNAVAILABLE, STATE_UNKNOWN)

#: Frame roles the controller subscribes to. ``detection_distance`` is part
#: of the LD2410 cluster but not consumed by the estimator (const.py), so
#: its churn never produces frames.
FRAME_ROLES: tuple[str, ...] = (
    ROLE_MOVE_ENERGY,
    ROLE_STILL_ENERGY,
    ROLE_MOVING_DISTANCE,
    ROLE_STILL_DISTANCE,
    ROLE_TARGET,
    ROLE_MOVING_TARGET,
    ROLE_STILL_TARGET,
)

#: Gate role -> (channel, gate index), e.g. ``"g3_move" -> ("move", 3)``.
#: Gate entities are optional enrichment (spec rules 2.4-2.6): subscribed
#: and folded into frames when configured, but never part of availability
#: (rule 1.3 stays keyed to the two aggregate energy roles) — engineering
#: mode may drop at any radar power-cycle and the engine falls back to the
#: aggregate path per frame.
_GATE_ROLE_INDEX: dict[str, tuple[str, int]] = {
    **{role: ("move", index) for index, role in enumerate(GATE_MOVE_ROLES)},
    **{role: ("still", index) for index, role in enumerate(GATE_STILL_ROLES)},
}

#: Observation-clock role sets (rule 1.1). ``target`` flips when any kind
#: of target appears or disappears: it observes both channels.
_MOVE_ROLES = frozenset(
    {ROLE_MOVE_ENERGY, ROLE_MOVING_DISTANCE, ROLE_MOVING_TARGET, ROLE_TARGET, *GATE_MOVE_ROLES}
)
_STILL_ROLES = frozenset(
    {ROLE_STILL_ENERGY, ROLE_STILL_DISTANCE, ROLE_STILL_TARGET, ROLE_TARGET, *GATE_STILL_ROLES}
)
_MOVE_ENERGY_ROLES = frozenset({ROLE_MOVE_ENERGY, *GATE_MOVE_ROLES})


class EngineProtocol(Protocol):
    """The engine surface the controller drives (real or test double)."""

    state: EngineState

    def start(self, now: float) -> Plan: ...

    def submit(self, event: Event, now: float) -> Plan: ...

    def on_timer(self, key: str, now: float) -> Plan: ...


def _as_float(state: State | None) -> float | None:
    """A state as a float; ``None`` when absent, unavailable or non-numeric."""
    if state is None or state.state in UNAVAILABLE_STATES:
        return None
    try:
        return float(state.state)
    except ValueError:
        return None


def _is_on(state: State | None) -> bool:
    return state is not None and state.state == STATE_ON


@dataclass(slots=True)
class _SensorView:
    """Last observed HA-side values of one sensor's entity cluster.

    The cached view is the frame source: every entity change updates one
    field and the *complete* frame is rebuilt from the cache (rule 1.1).
    Fields are ``None`` where unknown.
    """

    move_energy: float | None = None
    still_energy: float | None = None
    moving_distance: float | None = None
    still_distance: float | None = None
    has_target: bool = False
    has_moving_target: bool = False
    has_still_target: bool = False
    #: Per-gate energies (rules 2.4-2.6): ``None`` when the sensor has no
    #: gate entities configured for the channel; individual gates are
    #: ``None`` while unknown/unavailable (engineering mode off) — the
    #: engine falls back to the aggregate path per frame (rule 2.6).
    gate_move: list[float | None] | None = None
    gate_still: list[float | None] | None = None
    #: Availability over the required (energy) roles, rule 1.3.
    available: bool = False
    #: Observation clock (rule 1.1): counters advanced on every reported
    #: update of a channel's entities, including same-value forced
    #: re-publications — that is a new measurement, not a duplicate.
    move_obs: int = 0
    still_obs: int = 0
    move_energy_obs: int = 0

    def update(self, role: str, state: State | None) -> None:
        """Fold one entity state into the view."""
        if role in _MOVE_ROLES:
            self.move_obs += 1
        if role in _STILL_ROLES:
            self.still_obs += 1
        if role in _MOVE_ENERGY_ROLES:
            self.move_energy_obs += 1
        if (gate := _GATE_ROLE_INDEX.get(role)) is not None:
            channel, index = gate
            values = self.gate_move if channel == "move" else self.gate_still
            if values is None:
                values = [None] * GATE_COUNT
                if channel == "move":
                    self.gate_move = values
                else:
                    self.gate_still = values
            values[index] = _as_float(state)
            return
        match role:
            case "move_energy":
                self.move_energy = _as_float(state)
            case "still_energy":
                self.still_energy = _as_float(state)
            case "moving_distance":
                self.moving_distance = _as_float(state)
            case "still_distance":
                self.still_distance = _as_float(state)
            case "target":
                self.has_target = _is_on(state)
            case "moving_target":
                self.has_moving_target = _is_on(state)
            case "still_target":
                self.has_still_target = _is_on(state)

    def frame(self, sensor_id: str) -> SensorFrame:
        """The coalesced frame (rule 1.1). Energies stay raw 0-100; the
        core normalizes on ingest (rule 1.4)."""
        return SensorFrame(
            sensor_id=sensor_id,
            moving_distance_cm=self.moving_distance,
            still_distance_cm=self.still_distance,
            move_energy=self.move_energy,
            still_energy=self.still_energy,
            has_target=self.has_target,
            has_moving_target=self.has_moving_target,
            has_still_target=self.has_still_target,
            gate_move=None if self.gate_move is None else tuple(self.gate_move),
            gate_still=None if self.gate_still is None else tuple(self.gate_still),
            move_obs=self.move_obs,
            still_obs=self.still_obs,
            move_energy_obs=self.move_energy_obs,
        )


def _build_view(hass: HomeAssistant, roles: Mapping[str, str]) -> _SensorView:
    """A view of one sensor's current entity states."""
    view = _SensorView()
    for role in (*FRAME_ROLES, *GATE_ROLES):
        if (entity_id := roles.get(role)) is not None:
            view.update(role, hass.states.get(entity_id))
    view.available = _required_available(hass, roles)
    return view


def _required_available(hass: HomeAssistant, roles: Mapping[str, str]) -> bool:
    """Rule 1.3 availability: every required (energy) entity has a live state."""
    for role in ENERGY_ROLES:
        entity_id = roles.get(role)
        if entity_id is None:
            return False
        state = hass.states.get(entity_id)
        if state is None or state.state in UNAVAILABLE_STATES:
            return False
    return True


def build_initial_snapshot(hass: HomeAssistant, options: Mapping[str, Any]) -> InitialSnapshot:
    """Snapshot the current world from ``hass.states`` to seed the engine (7.1).

    ``enabled`` is always True here: the engine's model default. A persisted
    "off" is pushed back through the enabled switch's restore path, like any
    user command.
    """
    frames: dict[str, SensorFrame | None] = {}
    available: dict[str, bool] = {}
    for sensor_id, roles in sensor_entities(options).items():
        view = _build_view(hass, roles)
        available[sensor_id] = view.available
        frames[sensor_id] = view.frame(sensor_id) if view.available else None
    return InitialSnapshot(
        frames=frames,
        available=available,
        baselines=baselines_from_options(options),
        enabled=True,
    )


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
        self._views: dict[str, _SensorView] = {
            sensor_id: _build_view(hass, roles)
            for sensor_id, roles in self._roles_by_sensor.items()
        }

    # -- public API for entities ---------------------------------------------

    @property
    def state(self) -> EngineState:
        return self.engine.state

    def room_name(self, room_id: str) -> str:
        return self._room_names.get(room_id, room_id)

    def pass_by_signal(self, zone_id: str) -> str:
        """Dispatcher signal carrying :class:`PassBy` events for one zone."""
        return f"{DOMAIN}_{self.entry.entry_id}_pass_by_{zone_id}"

    def room_pass_by_signal(self, room_id: str) -> str:
        """Dispatcher signal carrying :class:`PassBy` events of every member
        zone of one room (rule 5.2 routed through §6 membership)."""
        return f"{DOMAIN}_{self.entry.entry_id}_room_pass_by_{room_id}"

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
            self._views[sensor_id] = _build_view(self.hass, roles)
        self._subscribe()
        self._tick_unsub = async_track_time_interval(
            self.hass,
            self._on_tick_interval,
            timedelta(seconds=self.config.tunables.tick_interval),  # 1.2
        )
        self._apply_plan(self.engine.start(time.monotonic()))

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

    @callback
    def _on_entity_event(self, event: HAEvent[EventStateChangedData]) -> None:
        """Rule 1.1: any underlying entity change produces a complete frame."""
        mapped = self._sensor_role_by_entity.get(event.data["entity_id"])
        if mapped is None:
            return
        sensor_id, role = mapped
        view = self._views[sensor_id]
        view.update(role, event.data["new_state"])
        available = _required_available(self.hass, self._roles_by_sensor[sensor_id])
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
            self._persist_baselines()
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
            # BaselineRecorded rides on persist_calibration above; entities
            # (diagnostics) refresh through the publish below.
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

    # -- baseline persistence ------------------------------------------------------

    @callback
    def _persist_baselines(self) -> None:
        """Write the engine's zone baselines into the config entry (rule 3.3).

        Merged over the stored map so baseline keys of removed zones survive
        (they are ignored on read); compared before writing so a no-op plan
        never touches the entry.
        """
        stored: dict[str, Any] = dict(self.entry.options.get(CONF_BASELINES) or {})
        default = ChannelStats(self.config.tunables.default_mu, self.config.tunables.default_sigma)
        current: dict[str, Any] = {}
        for zone in self.config.zones:
            zst = self.engine.state.zones.get(zone.zone_id)
            if zst is None:
                continue
            record: dict[str, Any] = {
                "move_mu": zst.move_baseline.mu,
                "move_sigma": zst.move_baseline.sigma,
                "still_mu": zst.still_baseline.mu,
                "still_sigma": zst.still_baseline.sigma,
            }
            # Rule 3.6: optional per-gate floors (string keys: options are
            # JSON). Zones without any stay schema-identical to v0.1.0. A
            # gate calibrated on one channel only persists the defaults for
            # the other, which is what the engine would score with anyway.
            gates = {
                str(index): {
                    "move_mu": (gm := zst.gate_move_baselines.get(index, default)).mu,
                    "move_sigma": gm.sigma,
                    "still_mu": (gs := zst.gate_still_baselines.get(index, default)).mu,
                    "still_sigma": gs.sigma,
                }
                for index in sorted(zst.gate_move_baselines.keys() | zst.gate_still_baselines)
            }
            if gates:
                record["gates"] = gates
            # Rule 3.7: optional statistic calibration. Zones calibrated
            # before 3.7 (or not at all) persist without it and score
            # against the analytic fallback.
            stats = {
                key: {"mu": cal.mu, "sigma": cal.sigma, "clip_mu": cal.clip_mu, "tau": cal.tau}
                for key, cal in sorted(zst.stat_cal.items())
            }
            if stats:
                record["stats"] = stats
            current[zone.zone_id] = record
        merged = {**stored, **current}
        if merged == stored:
            return
        # The options listener in __init__.py ignores baselines-only diffs,
        # so this write never causes a reload loop.
        self.hass.config_entries.async_update_entry(
            self.entry, options={**self.entry.options, CONF_BASELINES: merged}
        )


# ---------------------------------------------------------------------------
# Shared entity plumbing
# ---------------------------------------------------------------------------


def conductor_device_info(entry: ConfigEntry) -> DeviceInfo:
    """The hub device: home-level outputs, controls and diagnostics."""
    return DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name="Presence Conductor",
        manufacturer="Presence Conductor",
        model="Calibrated mmWave occupancy estimator",
        entry_type=DeviceEntryType.SERVICE,
    )


def room_device_info(controller: PresenceConductorController, room_id: str) -> DeviceInfo:
    """One device per configured room, hanging off the hub via ``via_device``.

    A room's presence signals — its own fused outputs (§6) and the outputs
    of its member zones — are self-contained on this device, so a consumer
    (a dashboard, a future sonos-conductor hookup) points at exactly one
    device per room.
    """
    room_name = controller.room_name(room_id)
    return DeviceInfo(
        identifiers={(DOMAIN, f"{controller.entry.entry_id}_room_{room_id}")},
        name=f"{room_name} presence",
        manufacturer="Presence Conductor",
        model="Calibrated mmWave occupancy estimator",
        suggested_area=room_name,
        via_device=(DOMAIN, controller.entry.entry_id),
        entry_type=DeviceEntryType.SERVICE,
    )


class ConductorEntity(Entity):
    """Base for conductor entities: dispatcher-driven, never polled.

    Entities with a ``room_id`` live on that room's device; the rest (home
    outputs, controls, diagnostics) live on the hub.

    ``_attr_name`` values are hardcoded English on purpose, and the object
    id is pinned to the ``presence_conductor_`` scheme those names produce
    (see :meth:`add_to_platform_start`), yielding language-stable object ids
    like ``binary_sensor.presence_conductor_sofakrok_occupancy``.
    """

    _attr_should_poll = False
    _attr_has_entity_name = True
    #: Which controller signal refreshes this entity. Control surfaces
    #: (enabled switch, diagnostics) override this with ``signal_control``
    #: so they keep updating while outputs are suppressed (rule 7.2).
    _control_surface = False

    def __init__(self, controller: PresenceConductorController, room_id: str | None = None) -> None:
        self.controller = controller
        self._attr_device_info = (
            conductor_device_info(controller.entry)
            if room_id is None
            else room_device_info(controller, room_id)
        )

    @callback
    def add_to_platform_start(
        self,
        hass: HomeAssistant,
        platform: EntityPlatform,
        parallel_updates: asyncio.Semaphore | None,
    ) -> None:
        """Suggest the object id before registration.

        Object ids are the stable consumer contract; device grouping is
        presentation. Without the explicit suggestion HA derives the object
        id from the device name, so moving an entity from the hub to its
        room device would rename it.
        """
        if self.entity_id is None:
            self.entity_id = f"{platform.domain}.{slugify(f'Presence Conductor {self._attr_name}')}"
        super().add_to_platform_start(hass, platform, parallel_updates)

    @property
    def engine_state(self) -> EngineState:
        return self.controller.engine.state

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        signal = self.controller.signal_control if self._control_surface else self.controller.signal
        self.async_on_remove(
            async_dispatcher_connect(self.hass, signal, self._on_controller_update)
        )

    @callback
    def _on_controller_update(self) -> None:
        self.async_write_ha_state()
