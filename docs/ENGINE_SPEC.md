# Presence Conductor engine specification

This is the normative contract for the estimation core. Code comments cite
rules by number (for example, “rule 4.2”). Behavioral changes update this
document before code in the same pull request.

The core is deterministic, synchronous Python with no Home Assistant imports,
wall clock, or I/O. The adapter supplies events and monotonic timestamps; the
core returns state changes, emitted events, persistence requests, and timers in
a plan.

Supporting documents are deliberately non-normative:

- [Calibration operations](CALIBRATION.md) explains what an operator sees and
  how to record a trustworthy baseline.
- [Estimator rationale](ESTIMATOR_RATIONALE.md) explains the statistical and
  DSP choices, assumptions, and validation history.
- [Decision record](DECISION.md) contains the original deployment baseline.

## 0. Model and outputs

- **Sensor:** one physical mmWave device identified by `sensor_id`.
- **Zone:** one distance interval `[near_cm, far_cm]` of exactly one sensor.
  Zones are the estimation unit.
- **Room:** a set of zones. Rooms are the fusion unit.
- **Belief:** each zone stores a bounded accumulator `lambda`; `confidence =
  sigmoid(lambda)` is a monotone occupancy score, not a calibrated posterior
  probability. Simple calibration learns the empty distribution; optional
  Full calibration also learns and validates local occupied emissions (3.9).

Zone outputs are `occupied`, `motion`, `activity` (`empty`, `passing`,
`active`, `settled`), `confidence`, `dwell_seconds`, and `pass_by`. Room
outputs are `occupied`, `motion`, `activity`, `settled`, and `confidence`.
Home outputs are `anyone_home` and `confidence`. Adapters publish core state;
they never re-derive it.

## 1. Inputs and conditioning

- **1.1 Frame and observation clock.** The adapter coalesces all configured
  entities of one sensor into a complete `SensorFrame` and submits it whenever
  a tracked entity is reported. Exact unchanged writes arrive through an
  entity-filtered Home Assistant `state_reported` listener; changed values and
  force updates arrive through `state_changed`. Same-value writes whose only
  difference is attributes update the cache but do not certify a measurement.

  Cached distance and energy values remain present until replaced; `None`
  means never reported, unavailable, or unparseable. Freshness is explicit:
  `move_obs` and `still_obs` advance for certified reports of any entity in
  that channel, `frame_obs` advances for any certified sensor report, and
  `move_energy_obs` advances only for aggregate or gate move-energy reports.
  Values are never compared to infer freshness.

  Counting distance and target reports as channel observations relies on the
  deployed LD2410 firmware contract: fields originate in one atomic hardware
  frame and per-entity filters are throttle-only. Energy delta suppression,
  including Apollo `reduce_db_reporting`, is unsupported. Fast attack (4.2)
  always requires `move_energy_obs`, never a distance or flag report.

  Aggregate energies are raw 0–100 values. Gate tuples contain `g0..g8`, with
  `None` for a missing gate and a tuple-level `None` when no channel entities
  are configured. Gate loss does not affect sensor availability.
- **1.2 Tick.** The adapter emits `Tick` every `tick_interval` (default 1 s).
  Every event advances time chronologically (4.1), so results are invariant to
  tick cadence and scheduler pauses.
- **1.3 Health and staleness.** Required availability is the two aggregate
  energy entities. Unavailability, or no measured observation for
  `stale_after` (default 30 s) while any sensor zone is occupied, changes every
  sensor zone to `UNKNOWN`: published values hold, confidence is stale, and
  room fusion excludes the zones. A measured observation recovers immediately.
  Invalid, cached, and attribute-only events cannot recover health or re-arm
  the watchdog. Every watchdog firing clears fast-attack confirmation.
- **1.4 Units.** Aggregate and gate energies normalize to `[0, 1]` at ingest;
  distances remain centimetres. Out-of-range numeric values clamp.

## 2. Spatial gating

- **2.1 Zone mask.** A channel contributes aggregate evidence when its usable
  distance (2.7) lies in `[near_cm − margin_cm, far_cm + margin_cm]`.
  `margin_cm` defaults to 30 cm.
- **2.2 Same-room separation.** Zone masks are the only cross-sensor spatial
  separation. The config flow warns, but does not reject, overlapping zones
  belonging to different sensors in one room.
- **2.3 Missing distance.** A target flag with no usable distance assigns the
  channel to the sensor’s `fallback` zone, or its nearest zone when none is
  marked.
- **2.4 Gate ownership.** Gate `i` spans
  `[i·gate_size_cm, (i+1)·gate_size_cm)`. A zone owns every gate overlapping
  its masked interval. Adjacent zones may share a boundary gate. The default
  gate size is 75 cm; 20 cm mode is configured per sensor.
- **2.5 Gate statistic.** For each channel, the raw gate statistic is the
  maximum one-sided standardized deviation over owned gates. It is never a
  sum. Its empty distribution is calibrated for the same gate family (3.7).
- **2.6 Gate precedence.** When `use_gate_evidence` is enabled and at least one
  owned gate is present for a channel, gate evidence replaces aggregate
  distance-gated evidence for that channel and frame. Otherwise the aggregate
  path applies. Fallback is automatic, per frame, and per channel; there is no
  mode latch. Motion and fast attack use the path selected for the frame.
  Gate evidence is experimental and defaults off until synchronized production
  gate captures validate its temporal and cross-gate behavior.
- **2.7 Distance freshness.** A distance is usable while its target flag is on
  and for `distance_hold` (default 30 s) after the flag was last on. After the
  hold it is ignored. At startup, before recency exists, only flag-on distances
  are usable.

## 3. Evidence and calibration

The estimator is a calibrated empty-room anomaly score. Its required invariant
is negative expected drive in a calibrated empty room, independent of gate
count. See [Estimator rationale](ESTIMATOR_RATIONALE.md) for derivations.

- **3.1 Noise floor.** Each aggregate channel and owned gate has robust empty
  statistics `(mu, sigma)`. `mu` is the median. `sigma` is the MAD-based,
  one-sided upper-confidence scale plus half an `energy_quantum`, floored by
  `max(sigma_min, MAD_TO_SIGMA·energy_quantum/2)`. Aggregate fits use a 95%
  one-sided rank bound; gate fits use the Bonferroni family-wise bound for the
  number of owned gates. Dependence discounts the rank count through the upper
  bound of lag-1 sign agreement. Duplicate held values are collapsed.

  A channel spanning at most one quantum is `quiescent` only after at least
  `stat_min_rows` tick rows certified by real observations. A non-quiescent
  channel with insufficient distinct observations keeps its previous floor.
- **3.2 Centered evidence.** The raw statistic is
  `S=max(0,(energy−mu)/sigma)` for the active aggregate channel, or the maximum
  equivalent value over owned gates. The centered score is
  `ẑ=clamp((S−m0)/s0,−z_neg_cap,z_cap)−c0`, divided by the path’s integrated
  autocorrelation estimate `tau`. `(m0,s0,c0)` come from the accepted empirical
  calibration for the exact path, otherwise the analytic Gaussian max model.

  Over live channels, the evidence rate is
  `u=min(u_cap, k_move·(ẑ_move−k_bias)+k_still·(ẑ_still−k_bias))`.
  Defaults are `k_move=0.5`, `k_still=0.3`, `k_bias=0.5`, and `u_cap=3`.
  Thus calibrated empty observations have expected drive `−0.4/s`; gains and
  the upward cap bound the variance and impact of isolated excursions.
- **3.3 Transactional baseline.** `RecordBaseline(zone_id, duration)` opens an
  empty-room window, default 300 s. Rows are sampled once per tick from the
  coherent sensor cache, while freshness counts distinct observation epochs.
  The zone is suspended during collection: belief is pinned to the empty prior,
  occupancy and motion are off, activity is `EMPTY`, attack state is cleared,
  and incoming frames feed calibration only.

  The four paths (`move_agg`, `still_agg`, `move_gate`, `still_gate`) receive
  `calibrated`, `quiescent`, `no_data`, or `rejected` coverage verdicts before
  any state mutates. Aggregate paths configured for the sensor are required;
  a required rejection, sensor unavailability at close, or an observation gap
  beyond `stale_after` rejects the whole candidate and preserves all previous
  calibration. Gate channels are optional when absent, but a present family
  commits only when complete for every owned gate. A successful commit persists
  only the requested zone. Every outcome is emitted to the HA bus, event entity,
  and log. Detailed operator behavior lives in [Calibration operations](CALIBRATION.md).
- **3.4 Background adaptation.** When every sibling zone of a sensor has
  confidence below `p_background` for `t_background`, floors adapt with EMA
  time constant `tau_background`. Adaptation may move `mu` and increase
  `sigma`; it may not decrease the conservative calibrated scale. It stops
  while unavailable, occupied, explicitly calibrating, or while a Full
  occupied profile is installed. A Full profile is bound to its exact empty
  transform, so changing floors behind it would invalidate its features.
- **3.5 Still-margin recovery.** While the frozen still distance remains usable
  under 2.7, energy above its calibrated floor contributes positive evidence
  even when the radar's binary still-target flag has dropped. This margin
  bridges measured still-target dropout gaps. Energy at or below the floor, or
  a zone not selected by either ordinary distance gating or the 2.3 fallback,
  produces a zero raw statistic; while the channel is observationally live
  (3.8), centering makes that absence evidence.
- **3.6 Gate calibration.** Gate floors are independent per channel and gate.
  Runtime gate readiness requires a complete, context-compatible family.
  Partial, absent, legacy, or resolution-mismatched families fall back to the
  aggregate path. Background adaptation may create provisional floors but can
  never make a family runtime-ready.
- **3.7 Statistic calibration and compatibility.** Empirical `(m0,s0,c0,tau)`
  uses fresh rows only and requires `stat_min_rows`. Its scale cannot be sharper
  than the analytic reference or `stat_sigma_min`. The lag-1 dependence upper
  bound yields `tau=clamp((1+rho)/(1−rho),1,tau_int_max)`; the median physical
  observation interval converts this to the decorrelation time used by 4.2.
  Rare-tail attack thresholds always remain analytic.

  Persisted floors carry sensor identity, exact gate family and resolution,
  and a floor-fit fingerprint. Statistics carry a separate fingerprint for
  path, gate family, transform, floor settings, and dependence limits.
  Incompatible floors use defaults; incompatible statistics use the analytic
  fallback. Legacy metadata is safe but requires recalibration and is surfaced
  by diagnostics.
- **3.8 Held evidence.** A positive margin-shifted score contributes for at
  most `obs_budget` (default 1 s) after its observation. A non-positive score
  contributes for at most `obs_hold` (default 5 s). Thereafter that channel
  contributes nothing. Stale positive evidence never becomes absence; with all
  channels silent, `u=0` and only the departure hazard changes belief.

- **3.9 Calibration levels.** Operator intent is `skip`, `simple`, or `full`;
  missing legacy intent means `simple`. Skip suppresses calibration prompts;
  manual capture remains available and any compatible saved calibration stays active. Simple is
  the transactional empty baseline in 3.3. Full records that baseline, then
  eight explicit phases: training empty, moving, standing, seated; validation
  moving, standing, seated, empty (empty is last so the room is safe to re-enter).
  Each phase requires at least 15 fresh observation epochs and a live sensor at
  close. Raw labeled rows stay in memory and are discarded after completion,
  rejection, cancellation, unload, or restart.

  Full calibration suspends every zone of the selected sensor so intentional
  movement cannot reach zone, room, home, motion, or pass-by outputs. Only one
  calibration may run per sensor. Empty calibration requires the sensor's
  entire field of view—not merely one distance slice—to be empty because
  aggregate energies are sensor-global.

- **3.10 Occupied emission model and validation.** Full calibration fits a
  regularized shared-covariance two-feature LDA over the existing clipped,
  empty-standardized `(z_move,z_still)` feature. Moving is one occupied mode;
  standing and seated form the stationary mode. Classes and occupied modes
  receive equal weights, so recording duration cannot become a deployment
  prior. The two discriminants combine by normalized log-sum-exp into a
  two-sided rate bounded to `[-3,3]` and by `u_cap`.

  Training and validation phases are never mixed. The held-out result stores
  TP, FP, TN, FN, sensitivity, specificity and balanced accuracy plus
  per-scenario recall. The ordered held-out observations are also replayed
  through 4.1: every occupied scenario must cross `theta_on`, and empty must
  never cross it. A profile commits only with sensitivity at least 70%,
  specificity at least 80%, at least 50% recall in every occupied scenario,
  negative mean empty drive, and positive mean drive in every occupied
  scenario. Full is one atomic transaction: failure/cancellation restores a
  previous Full calibration, or keeps the newly useful empty baseline on the
  first Full attempt. Profiles bind to the exact aggregate/gate path, gate
  family, exact floors, floor settings, statistic transforms, geometry, and
  distance-hold context; mismatch or per-frame gate fallback uses 3.2 unchanged.

## 4. Occupancy filter

- **4.1 Chronological integration.** Every event first integrates the evidence
  that was in force over elapsed time, then installs its own effects. For
  constant `u`, the exact update is
  `lambda=lambda_eq+(lambda−lambda_eq)·exp(−dt/tau_decay)`, where
  `lambda_eq=lambda_prior+tau_decay·u`. `tau_decay` defaults to 90 s and the
  prior confidence to 0.02. No fixed occupancy timeout exists.
- **4.2 Fast attack.** A fresh move-energy observation is a candidate when its
  raw statistic exceeds the analytic Gaussian max threshold
  `Phi^-1((1−attack_tail)^(1/m))`; `attack_tail_ppm` defaults to 100. Attack
  raises belief to at least `p_attack` (default 0.95) after `attack_confirm`
  candidates (default 2) separated by `[g,G]`.

  Without empirical timing, `g=attack_gap_min` and `G=attack_gap_max`
  (defaults 0.3 and 3 s). With decorrelation estimate `d`,
  `g=max(attack_gap_min,d)` and the configured window width is preserved.
  Invalid legacy windows regain the default 2.7 s width; zero minimums receive
  a 0.1 s defensive floor. Candidates closer than `g` are ignored. A fresh
  non-candidate, stale/unavailable interval, or aggregate/gate path switch
  resets the chain. Non-fresh frames do not change it.

  With a compatible Full profile on the active path, the candidate must also
  have a non-negative occupied discriminant. A nuisance tail spike rejected by
  the validated model therefore cannot bypass it through fast attack.
- **4.3 Hysteresis.** Occupancy turns on at `theta_on` (default confidence
  0.80), off at `theta_off` (default 0.20), and otherwise holds.
- **4.4 Motion.** Motion turns on for gated `ẑ_move >= z_motion` (default 2)
  or a moving-target flag with usable aggregate distance, and turns off after
  `motion_hold` (default 5 s). While gate evidence is active, the global target
  flag is not zone evidence.
- **4.5 Clamp.** Belief is continuously clamped to confidences `[0.001,0.999]`
  so no history creates unbounded inertia.

## 5. Activity and pass-by

- **5.1 States.** `EMPTY` becomes `PASSING` on occupancy. After `t_dwell`
  (default 45 s) it becomes `ACTIVE`, while continuous still dominance for
  `t_settle` (default 30 s) produces `SETTLED`. `ACTIVE` and `SETTLED` switch
  only after continuous channel dominance; neutral or equal evidence resets
  both dominance clocks.
- **5.2 Pass-by.** Transition from `PASSING` to `EMPTY` emits one event with
  peak confidence and duration. Exits from `ACTIVE` or `SETTLED` do not.
- **5.3 Consumer contract.** `occupied` includes `PASSING`. Consumers that
  should ignore walk-throughs use `activity in {active,settled}` or room
  `settled`.
- **5.4 Dwell.** `dwell_seconds` counts continuous zone occupancy and resets
  on `EMPTY`.

## 6. Room and home fusion

- **6.1 Room occupancy.** A room is occupied when any healthy member is
  occupied. Confidence is the maximum healthy member confidence, not noisy-OR,
  because member evidence is dependent.
- **6.2 Activity and motion.** Activity is the maximum severity
  (`settled > active > passing > empty`); `settled` and `motion` are ORs over
  healthy members.
- **6.3 Health.** `UNKNOWN` zones are excluded. A room with no healthy members
  publishes unknown rather than empty.
- **6.4 Monotonicity.** A zone may add room occupancy but never veto another
  zone. Separation belongs to spatial gating, not fusion.
- **6.5 Home presence.** Any occupied healthy zone raises `lambda_home`
  immediately. With all zones empty it relaxes toward its prior with
  `tau_home` (default 20 minutes) and uses hysteresis. If every zone is
  unhealthy, home presence is unknown.

## 7. Lifecycle

- **7.1 Startup.** The engine adopts an `InitialSnapshot`. Beliefs start at the
  prior, except zones with a currently gated target start at `theta_on`.
- **7.2 Disabled.** While disabled the engine remains warm but ordinary state
  transitions and pass-by events are suppressed. Explicit control outcomes,
  including baseline success or rejection, remain observable.
- **7.3 Determinism.** Equal event sequences and timestamps produce equal
  outputs. All runtime parameters live in the immutable `Tunables` snapshot.

## 8. Scope and roadmap

- **8.1** Gate evidence is implemented but experimental and off by default.
  Device engineering-mode setup remains outside this repository.
- **8.2** The integration never writes radar thresholds or timeouts.
- **8.3** Door, PIR, media, and other non-radar evidence are out of scope.
- **8.4** Multi-target tracking is out of scope.
- **8.5** `tools/replay.py` is the offline regression harness for production
  history and transition metrics.
- **8.6** Audio and lighting behavior belongs in consumer integrations; this
  package publishes zone and room presence only.
- **8.7** The score is not a posterior probability. Full calibration learns a
  local likelihood-ratio map and validates binary classification on a short
  held-out capture; it does not estimate population priors, transition
  probabilities, or a reliability curve. Confidence remains a bounded
  monotone control score.

## Tunable defaults

This table is the single documentation reference for `Tunables` defaults.
Rules above define their behavior.

| Area | Tunable | Default |
| --- | --- | ---: |
| Input | `margin_cm` | `30` |
| Input | `stale_after` | `30 s` |
| Input | `tick_interval` | `1 s` |
| Floor | `sigma_min` | `0.02` |
| Floor | `energy_quantum` | `0.01` |
| Floor | `default_mu`, `default_sigma` | `0.10`, `0.10` |
| Score | `z_cap`, `z_neg_cap` | `6`, `1` |
| Score | `stat_sigma_min`, `stat_min_rows` | `0.3`, `60` |
| Score | `tau_int_max` | `25` |
| Hold | `obs_budget`, `obs_hold` | `1 s`, `5 s` |
| Gate | `use_gate_evidence` | `false` |
| Evidence | `k_move`, `k_still`, `k_bias` | `0.5`, `0.3`, `0.5` |
| Evidence | `u_cap` | `3` |
| Filter | `tau_decay`, `p_prior` | `90 s`, `0.02` |
| Attack | `attack_tail_ppm`, `attack_confirm` | `100`, `2` |
| Attack | `attack_gap_min`, `attack_gap_max` | `0.3 s`, `3 s` |
| Attack | `p_attack` | `0.95` |
| Occupancy | `theta_on`, `theta_off` | `0.80`, `0.20` |
| Motion | `z_motion`, `motion_hold` | `2`, `5 s` |
| Clamp | `p_min`, `p_max` | `0.001`, `0.999` |
| Distance | `distance_hold` | `30 s` |
| Adaptation | `p_background` | `0.05` |
| Adaptation | `t_background`, `tau_background` | `600 s`, `3600 s` |
| Calibration | `baseline_duration` | `300 s` |
| Activity | `t_dwell`, `t_settle` | `45 s`, `30 s` |
| Home | `tau_home` | `1200 s` |
| Home | `theta_home_on`, `theta_home_off` | `0.80`, `0.20` |
