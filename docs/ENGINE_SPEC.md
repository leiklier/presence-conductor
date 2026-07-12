# Occupancy Conductor — Engine Specification

This document is the normative contract for the estimation core. Code
comments cite rules by number ("rule 4.2"). Changes to behavior land here
first, then in code, in the same PR.

The core is pure Python: no `homeassistant` imports (CI-enforced), no wall
clock, no I/O. The adapter feeds it events and monotonic timestamps; the core
returns state changes and timer requests through a plan object.

## 0. Model and definitions

- **Sensor** — one physical mmWave device (opaque `sensor_id`). Provides a
  stream of *frames* (rule 1.1).
- **Zone** — a spatial slice of one sensor's beam, defined by a distance
  interval `[near_cm, far_cm]` (opaque `zone_id`). A sensor may carry several
  zones; a zone belongs to exactly one sensor. Zones are the estimation unit.
- **Room** — a set of zones, possibly from different sensors
  (opaque `room_id`). Rooms are the fusion unit (§6).
- **Occupancy posterior** — per zone, the engine maintains `lambda` = log-odds
  of "zone is occupied". All evidence arithmetic happens in the log-odds
  domain.

The published outputs per zone are: `occupied` (robust binary), `motion`
(low-latency binary), `activity` (enum: `empty | passing | active |
settled`), `probability` (sigmoid of lambda), `dwell_seconds`, and a
`pass_by` event. Per room: `occupied`, `activity` (max-severity of member
zones), `settled`. Adapters read published state; they never re-derive it.

## 1. Inputs and conditioning

- **1.1 Frame.** The adapter coalesces a sensor's entity states into
  `SensorFrame(sensor_id, moving_distance_cm, still_distance_cm, move_energy,
  still_energy, has_target, has_moving_target, has_still_target)` and submits
  it whenever any underlying entity changes. Distances are `None` when the
  device reports no target of that kind. Energies are 0–100, `None` when
  unknown.
- **1.2 Tick.** The adapter delivers a periodic `Tick` (default every 1.0 s).
  All time-driven behavior (decay, dwell, holds) advances on ticks and event
  timestamps; the core never reads a clock.
- **1.3 Staleness.** If a sensor delivers no frame for
  `stale_after` (default 30 s) while any of its zones is occupied, or its
  entities become unavailable, its zones enter `UNKNOWN` health: outputs hold
  their last state, `probability` is marked stale, and room fusion ignores
  the zone (6.3). Recovery is immediate on the next frame.
- **1.4 Unit hygiene.** Energies are normalized to [0, 1] on ingest.
  Distances stay in cm. Out-of-range values are clamped, never rejected.

## 2. Distance gating

- **2.1 Zone mask.** A frame contributes *move evidence* to a zone iff
  `moving_distance ∈ [near_cm − margin, far_cm + margin]`; *still evidence*
  iff `still_distance` is in the same interval. `margin` (default 30 cm)
  absorbs the LD2410's interpolated-distance smoothing. A frame with a
  distance outside every zone of its sensor contributes nothing (the target
  belongs to another zone, another sensor's territory, or is a ghost at an
  implausible range).
- **2.2 Same-room separation.** Two sensors covering one room are separated
  by their zones' `far_cm` cutoffs. The mask is the *only* mechanism —
  there is no cross-sensor arbitration. Configuring non-overlapping
  intervals is the operator's contract; the config flow warns on overlap
  between zones of different sensors in the same room but does not forbid it.
- **2.3 No distance, no gate.** If a target flag is set but its distance is
  `None`, the evidence is attributed to the sensor's *default zone* (the zone
  flagged `fallback: true`, else the nearest zone). This keeps single-zone
  sensors working when the device momentarily omits distance.

## 3. Evidence model and calibration

- **3.1 Noise floor.** Per zone and per channel (move, still), calibration
  produces a robust baseline `(mu, sigma)` of the energy observed while the
  zone is empty (median / MAD over the calibration window, floored by
  `sigma_min` to avoid overconfidence).
- **3.2 Evidence score.** Per frame, per gated channel:
  `z = max(0, (energy − mu) / sigma)`, capped at `z_cap` (default 6).
  The per-frame log-likelihood ratio is
  `llr = k_move · z_move + k_still · z_still − k_absence`
  where `k_absence` applies only when both channels are un-gated or at
  baseline. Defaults: `k_move = 1.0`, `k_still = 0.6`, `k_absence = 0.4`
  (per second, scaled by tick interval).
- **3.3 Baseline calibration.** `RecordBaseline(zone_id, duration)` (service/
  button, default 120 s) collects empty-room frames and replaces `(mu,
  sigma)`. The operator asserts emptiness; the engine uses robust statistics
  so brief violations don't poison the baseline. Baselines persist in the
  config entry.
- **3.4 Background adaptation.** While a zone's posterior stays below
  `p_background` (default 0.05) for at least `t_background` (default 10 min),
  `(mu, sigma)` follow the observed energies with a slow EMA
  (`tau_background`, default 1 h). Adaptation freezes the moment the
  posterior rises. This tracks seasonal/furniture drift without learning a
  person as noise.
- **3.5 Still-margin recovery.** Rationale, not a rule: the radar's own
  binary output loses a still person whose energy sits *under* the gate
  threshold; rule 3.2 still credits that margin as evidence. This is the
  mechanism that bridges the measured dropout gaps.

## 4. Occupancy filter

- **4.1 Log-odds update.** Per tick: `lambda ← decay(lambda) + llr · dt`,
  where `decay` relaxes lambda toward the empty-state prior `lambda_0`
  (default corresponding to p = 0.02) with time constant `tau_decay`
  (default 90 s). The relaxation implements the hazard of departure; there is
  no fixed occupancy timeout anywhere in the engine.
- **4.2 Fast attack.** If `z_move ≥ z_attack` (default 3.0) on a gated frame,
  then `lambda ← max(lambda, lambda_attack)` (default corresponding to
  p = 0.95) immediately — not waiting for the next tick. This is the
  lights-on path; its latency is bounded by sensor publish latency alone.
- **4.3 Hysteresis.** `occupied` turns on at `lambda ≥ theta_on`
  (default p = 0.80) and off at `lambda ≤ theta_off` (default p = 0.20).
  Between thresholds the binary holds.
- **4.4 Motion output.** `motion` is the gated, undamped fast channel:
  on when a gated frame has `z_move ≥ z_motion` (default 1.5) or
  `has_moving_target` with a gated distance; off after `motion_hold`
  (default 5 s) without such evidence. It exists for automations that want
  raw responsiveness (hallway lights) and accepts flicker by design.
- **4.5 Clamp.** `lambda` is clamped to `[lambda_min, lambda_max]`
  (p = 0.001 / 0.999) so long occupation cannot build unbounded inertia.

## 5. Activity classification and pass-by

A per-zone FSM driven by the posterior and channel dominance:

- **5.1 States.** `EMPTY` (not occupied) → `PASSING` (occupied, since less
  than `t_dwell`, default 45 s, and no still-takeover yet) → `ACTIVE`
  (occupied past `t_dwell` with ongoing move evidence) / `SETTLED`
  (still evidence has dominated for `t_settle`, default 30 s — the seated /
  sleeping case). `ACTIVE ↔ SETTLED` follow channel dominance with the same
  `t_settle` smoothing.
- **5.2 Pass-by event.** `EMPTY` reached *from* `PASSING` emits `pass_by`
  with the zone's peak probability and traversal duration. Reached from
  `ACTIVE`/`SETTLED` it does not.
- **5.3 Consumer contract.** `occupied` includes `PASSING` (a person in the
  zone is in the zone); consumers that must not react to walk-throughs
  (dim-on-empty lighting, audio zones) key on `activity ∈ {active, settled}`
  or the room-level `settled`. This split — not suppression of short
  occupancy — is how "no flicker on walk-past" is achieved without adding
  latency for genuine entries.
- **5.4 Dwell.** `dwell_seconds` counts continuous occupancy of the zone,
  reset on `EMPTY`.

## 6. Room fusion

- **6.1 Occupancy.** A room is occupied iff any healthy member zone is
  occupied. Probability: noisy-OR over member posteriors
  (`P_room = 1 − Π(1 − p_i)`), published for diagnostics.
- **6.2 Activity.** Room activity is the maximum-severity member state
  (`settled > active > passing > empty`). Room `settled` is true iff any
  member zone is `SETTLED`.
- **6.3 Health.** Zones in `UNKNOWN` health (1.3) are excluded from fusion.
  A room with all members unknown publishes unknown, not off — downstream
  automations must be able to distinguish "nobody there" from "blind".
- **6.4 No cross-zone inhibition.** Fusion is monotone: a zone can only add
  occupancy to its room, never veto another zone. Separation is done at the
  gate (2.2), not at fusion.

## 7. Failure modes and lifecycle

- **7.1 Restart adoption.** On startup the engine seeds from an
  `InitialSnapshot` of current entity states; posteriors start at the prior,
  except zones whose sensor currently reports a gated target, which start at
  `theta_on` (someone plainly there should not wait out a cold start).
- **7.2 Disabled.** A global `enabled` switch: while off, the engine keeps
  ingesting frames and updating state (so re-enable is warm) but publishes no
  transitions and emits no events.
- **7.3 Determinism.** Same event sequence + timestamps ⇒ same outputs.
  All tunables live in one `Tunables` dataclass; nothing reads config at
  update time.

## 8. Non-goals (v1) and roadmap

- **8.1** Per-gate energies (`g0..g8`, engineering mode) are *not* consumed in
  v1. The frame model reserves optional fields; a phase-2 PR adds per-gate
  noise floors (spatial background subtraction) behind a capability flag,
  together with the ESPHome data-enablement overlay (engineering-mode
  re-assert, filter tuning, recorder-exclusion guidance).
- **8.2** No writes to device configuration (gate thresholds, radar
  timeout) in v1. A later calibration assistant may *suggest* thresholds;
  writing them stays a manual, operator-approved step.
- **8.3** No non-radar evidence (doors, media players, PIR). The evidence
  interface is a list of channels per frame, so additional sources can join
  without touching the filter; deliberately out of scope until the radar-only
  estimator is proven.
- **8.4** Multi-target tracking (LD2450 / MTR-1) is out of scope; the model's
  single-target assumption is documented where it bites (2.3).
- **8.5** An offline replay harness (`tools/replay.py`: HA history export →
  estimator → transition metrics vs. the DECISION.md baseline table) ships
  alongside the first estimator PR and gates tuning changes.
