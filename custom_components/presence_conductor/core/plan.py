"""Output plan for one ``submit``/``on_timer`` call.

The core's outputs are split in two:

- **State** — the adapter reads :class:`~.model.EngineState` after every
  call and publishes zone/room/home entities from it. There is no dedicated
  "publish state" effect. While the engine is disabled (rule 7.2) the plan's
  ``suppress_outputs`` flag is True: state keeps updating (re-enable is
  warm) but the adapter must not publish transitions to its entities.
- **The plan** — emitted events (:class:`PassBy`, :class:`BaselineRecorded`),
  timer start/cancel requests, and ``persist_calibration``, set when
  baselines changed so the adapter saves them to the config entry.
  ``persist_calibration`` is honored even while disabled: calibration is
  operator-requested data, not a published transition (rule 7.2).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmittedEvent:
    """Base class for events the engine emits through the plan."""


@dataclass(frozen=True, slots=True)
class PassBy(EmittedEvent):
    """A zone was traversed without dwelling (rule 5.2)."""

    zone_id: str
    peak_confidence: float
    #: Seconds the zone was occupied (on -> off).
    duration: float


@dataclass(frozen=True, slots=True)
class BaselineRecorded(EmittedEvent):
    """A RecordBaseline window completed and replaced the noise floor (3.3)."""

    zone_id: str
    move_mu: float
    move_sigma: float
    still_mu: float
    still_sigma: float
    frame_count: int


@dataclass(frozen=True, slots=True)
class StartTimer:
    """Ask the adapter to (re)start a timer. See :mod:`.timers` for keys."""

    key: str
    delay: float


@dataclass(frozen=True, slots=True)
class CancelTimer:
    """Ask the adapter to cancel a timer (idempotent)."""

    key: str


class Plan:
    """Accumulates the outputs of one engine call.

    It shares the engine's pending-timer registry so cancellations are only
    emitted for timers the engine believes are running, and stale
    ``on_timer`` keys can be rejected.
    """

    __slots__ = (
        "_pending",
        "events",
        "persist_calibration",
        "suppress_outputs",
        "timer_cancels",
        "timer_starts",
    )

    def __init__(self, pending: set[str], *, suppress_outputs: bool = False) -> None:
        self._pending = pending
        self.events: list[EmittedEvent] = []
        self.timer_starts: list[StartTimer] = []
        self.timer_cancels: list[CancelTimer] = []
        self.persist_calibration: bool = False
        self.suppress_outputs: bool = suppress_outputs

    def emit(self, event: EmittedEvent) -> None:
        # 7.2: while disabled the engine emits no events.
        if not self.suppress_outputs:
            self.events.append(event)

    def start_timer(self, key: str, delay: float) -> None:
        # Starting a pending key restarts it (adapter contract).
        self._pending.add(key)
        self.timer_starts.append(StartTimer(key, delay))

    def cancel_timer(self, key: str) -> None:
        if key in self._pending:
            self._pending.discard(key)
            self.timer_cancels.append(CancelTimer(key))
