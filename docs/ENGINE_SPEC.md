# Presence Conductor — Engine Specification

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
- **Occupancy belief** — per zone, the engine maintains `lambda`, a bounded
  accumulator of calibrated evidence in log-odds *form*. `confidence =
  sigmoid(lambda)` is a monotone occupancy score in [0, 1]. It is **not** a
  calibrated posterior probability: the evidence model (§3) knows the empty
  distribution but no occupied distribution, so no Bayesian semantics are
  claimed anywhere in this spec (8.7). All evidence arithmetic happens in
  the lambda domain.

The published outputs per zone are: `occupied` (robust binary), `motion`
(low-latency binary), `activity` (enum: `empty | passing | active |
settled`), `confidence` (sigmoid of lambda), `dwell_seconds`, and a
`pass_by` event. Per room: `occupied`, `motion` (any member zone's motion),
`activity` (max-severity of member zones), `settled`, `confidence` (6.1).
Per home: `anyone_home` and its confidence (6.5).
Adapters read published state; they never re-derive it.

Zone outputs are a first-class consumer surface, not an internal detail:
external consumers (sonos-conductor audio zones, lighting automations)
subscribe to individual zones — sofakrok and spisebord each publish their
full output set even though they fuse into one room.

## 1. Inputs and conditioning

- **1.1 Frame.** The adapter coalesces a sensor's entity states into
  `SensorFrame(sensor_id, moving_distance_cm, still_distance_cm, move_energy,
  still_energy, has_target, has_moving_target, has_still_target, gate_move,
  gate_still)` and submits it whenever any underlying entity changes.
  Distance and energy fields carry the **last reported** value (`None` only
  when never reported / unparseable): the device deduplicates identical
  publishes, so a frozen value is normal and the adapter must not null a
  distance just because its target flag turned off — distance freshness is
  the core's job (2.7), and production and replay must feed frames by this
  same contract. Because a frame is re-emitted on *any* entity change, the
  values alone cannot say which measurement is new: the frame also carries
  an explicit **observation clock** — three monotonic counters,
  `move_obs` / `still_obs` (incremented whenever any entity of that
  channel receives a reported update, *including* a same-value forced
  re-publication such as the still-energy heartbeat — the radar
  re-measured and got the same number) and `move_energy_obs` (move-energy
  or gate-move entities only, consumed by 4.2). The engine keys held-value
  handling (3.8) and attack freshness (4.2) on these counters, never on
  value comparison. Energies are 0–100, `None` when unknown. `gate_move`/`gate_still` carry the
  per-gate energies of engineering mode (2.4–2.6): nine elements — one per
  LD2410 distance gate `g0..g8` — with `None` for gates the device does not
  currently report; the whole tuple is `None` when the sensor has no gate
  entities configured. The adapter fills them only when gate entities are
  configured and parseable; engineering mode dropping (it does not survive a
  radar power-cycle) blanks gates without touching availability (1.3).
- **1.2 Tick.** The adapter delivers a periodic `Tick` (default every 1.0 s).
  Ticks only guarantee a floor on how often time advances: integration is
  chronological on *every* event (4.1), so outputs are invariant to tick
  cadence and scheduler delays.
- **1.3 Staleness.** If a sensor delivers no frame for
  `stale_after` (default 30 s) while any of its zones is occupied, or its
  entities become unavailable, its zones enter `UNKNOWN` health: outputs hold
  their last state, `confidence` is marked stale, and room fusion ignores
  the zone (6.3). Recovery is immediate on the next frame.
- **1.4 Unit hygiene.** Energies — aggregate and per-gate — are normalized
  to [0, 1] on ingest. Distances stay in cm. Out-of-range values are
  clamped, never rejected.

## 2. Distance gating

- **2.1 Zone mask.** A frame contributes *move evidence* to a zone iff its
  *usable* (2.7) `moving_distance ∈ [near_cm − margin, far_cm + margin]`;
  *still evidence* iff the usable `still_distance` is in the same interval.
  `margin` (default 30 cm) absorbs the LD2410's interpolated-distance
  smoothing. A frame with a distance outside every zone of its sensor
  contributes nothing (the target belongs to another zone, another sensor's
  territory, or is a ghost at an implausible range).
- **2.2 Same-room separation.** Two sensors covering one room are separated
  by their zones' `far_cm` cutoffs. The mask is the *only* mechanism —
  there is no cross-sensor arbitration. Configuring non-overlapping
  intervals is the operator's contract; the config flow warns on overlap
  between zones of different sensors in the same room but does not forbid it.
- **2.3 No distance, no gate.** If a target flag is set but its distance is
  `None`, the evidence is attributed to the sensor's *default zone* (the zone
  flagged `fallback: true`, else the nearest zone). This keeps single-zone
  sensors working when the device momentarily omits distance.
- **2.4 Gate ownership.** The radar divides its beam into nine distance
  gates, gate `i` spanning `[i · gate_size, (i + 1) · gate_size)` cm from
  the sensor. `gate_size` is per-sensor configuration (`gate_size_cm`,
  default 75 — the 0.75 m range resolution; the 0.2 m mode makes it 20). A
  zone owns every gate whose interval overlaps its masked interval
  `[near_cm − margin, far_cm + margin]` (the same margin as 2.1). Adjacent
  zones of one sensor may share a boundary gate: ownership is a mask, not a
  partition. A zone whose masked interval lies beyond the last gate owns
  nothing and runs on the aggregate path alone (2.6).
- **2.5 Gate evidence.** Per owned gate and per channel, `z_i` is the
  one-sided deviation of that gate's energy from that gate's own noise
  floor (3.6). The zone's raw channel statistic is the **maximum** over its
  owned gates, not the sum: a person occupies one or two gates, and summing
  would dilute a strong local return with the noise of empty gates. The max
  also credits two simultaneous people in different zones of one sensor —
  impossible in the single-distance model (2.1). The max is a biased,
  gate-count-dependent statistic; it enters the filter only after centering
  against its own empty-room distribution (3.2, 3.7).
- **2.6 Gate precedence.** When a frame's gate data covers a zone's channel
  — the gate tuple is present and at least one owned gate reports a value —
  gate evidence (2.5) replaces the aggregate energy + distance path
  (2.1–2.3) for that channel of that frame: spatially it is strictly
  better. When it does not (engineering mode off, no gate entities, all
  owned gates unknown, or no owned gates), the aggregate path applies
  unchanged. The fallback is automatic and per frame and per channel; there
  is no mode latch. Fast attack (4.2) fires on whichever move score the
  frame produced. The motion channel (4.4) keys on the gate move score
  while gate evidence is in effect; the sensor-global `has_moving_target`
  flag is not zone evidence when the gates already say where the mover is.
  **Status: experimental, default off** (`use_gate_evidence`, default
  false — gate tuples are ignored for evidence while calibration still
  records them). The temporal model (3.8) is validated against synthetic
  held/AR processes and the aggregate path against 24 h of live history,
  but the *gate path's* real update cadence, run lengths and cross-gate
  correlation have never been captured (the recorder excludes gate
  entities by design); until a bounded empty-room gate capture validates
  them, production defaults to the field-validated aggregate path.
- **2.7 Distance freshness.** Measured on real MSR-2 history: when a target
  flag turns off, the distance entity simply stops updating and *freezes at
  the last target's location* (it neither zeroes nor clears). A frozen
  distance is trustworthy only briefly: a channel's reported distance is
  **usable** while its target flag is on, and for at most `distance_hold`
  (default 30 s) after the flag was last on — long enough to keep
  attributing sub-threshold energy to a person who just went still (3.5),
  short enough that a later energy blip is not attributed to wherever
  someone last stood. Past the hold the distance is not usable and the
  channel is un-gated (2.3 still applies when the flag itself is on with no
  distance ever reported). The engine tracks flag recency itself; at seed
  time (7.1), with no recency known, only flag-on distances are usable.

## 3. Evidence model and calibration

The evidence model is a **calibrated anomaly score**, not a likelihood
ratio: calibration learns the empty-room (H0) distribution only, and no
occupied-state distribution is modeled (8.7). The design requirement that
replaces Bayesian semantics is: **the expected evidence rate in a
calibrated empty room is strictly negative** — symmetric sensor noise must
never drift a zone toward occupied, regardless of how many gates it owns.

- **3.1 Noise floor.** Per zone and per channel (move, still), calibration
  produces a robust baseline `(mu, sigma)` of the energy observed while the
  zone is empty: `mu` is the window median; `sigma` is a **conservative
  out-of-sample scale**, not the point estimate. A window-fitted scale has
  sampling error, and the device reports integer energies so observed
  deviations sit on a grid (`energy_quantum`, default 0.01 normalized —
  one raw count): a MAD that quantizes one step low understates the scale
  by 25–40%, and *every* future score is inflated by that error — an
  underestimated scale is a hazard, not a calibration. So
  `sigma = MAD_TO_SIGMA · (UCB(median |deviation|) + energy_quantum / 2)`,
  floored by `sigma_min`, where `UCB` is the one-sided 95% upper
  confidence bound for a median: the k-th order statistic of the absolute
  deviations with `k = ceil(n/2 + 1.645 · n / (2 · sqrt(n_eff)))`, where
  `n_eff = n(1 − ρ̂)/(1 + ρ̂)` discounts the trial count for dependence
  (`ρ̂` at its own upper bound; independent samples give back
  `sqrt(n)/2`). The samples are the window's **distinct observations** —
  consecutive duplicate values are collapsed first, because
  held/deduplicated rows repeat one measurement and a rank bound computed
  over repeats overstates its confidence. A channel whose distinct samples span at most one quantum is
  *quiescent* and calibrates to `(value, sigma_min)` directly; a channel
  with fewer than `stat_min_rows` distinct samples that is not quiescent
  keeps its previous floor — re-run with a longer `duration` instead.
  Stated assumptions (this is **not** distribution-free end to end): the
  rank-based UCB assumes the distinct observations are independent draws
  (3.7's dependence estimate discounts them when they are not); the
  1.4826 MAD-to-sigma conversion and the analytic attack tail (4.2) assume
  approximately Gaussian empty noise.
- **3.2 Evidence score.** Per frame and channel, the **raw statistic** `S`
  is the one-sided deviation from the noise floor: on the gate path,
  `S = max over owned gates of max(0, (energy_g − mu_g) / sigma_g)` (2.5);
  on the aggregate path, `S = max(0, (energy − mu) / sigma)` when the
  channel is gated (2.1–2.3), else `S = 0`. `S` is *biased by
  construction* — `E[S] > 0` under symmetric empty noise, and grows with
  the number of owned gates (a max over m gates is a multiple-comparison
  statistic). It is therefore never used directly: the **centered score**
  is `ẑ = clamp((S − m0) / s0, −z_neg_cap, z_cap) − c0` (caps default 1.0 /
  6.0), where `(m0, s0)` are the mean and deviation of `S` itself in a
  calibrated empty room (3.7) — for the *same* aggregation the channel used
  (per path, per owned-gate count) — and `c0` is the empty-room mean of the
  *clamped* score: asymmetric clamping alone would leave a positive
  residual mean for multi-gate maxima (≈ +0.05 to +0.07), so the final
  score is recentered after clamping, and finally divided by the path's
  estimated **integrated autocorrelation time** `τ̂` (3.7): correlated
  observations carry proportionally less independent information per
  second, and integrating them at full rate is how autocorrelated empty
  noise walks a zone occupied. `E[ẑ | empty] = 0` up to estimation error,
  by construction — measured for empirical calibrations, derived
  numerically for the analytic fallback (which assumes `τ̂ = 1`).
  The per-second evidence rate is
  `u = min(u_cap, k_move · (ẑ_move − k_bias) + k_still · (ẑ_still −
  k_bias))` over the observationally live channels (3.8); defaults
  `k_move = 0.5`, `k_still = 0.3`, `k_bias = 0.5`, `u_cap = 3.0`.
  `k_bias` is the **absence margin**: `E[ẑ] = 0` on calibrated empty
  noise for *any* gate count (3.7), so subtracting the margin from every
  observed score makes the expected observed rate exactly
  `−(k_move + k_still) · k_bias = −0.4/s` — what drives an empty zone
  down, with no conditional bias machinery to mis-tune per topology. There is no discontinuous "any z > 0 ⇒ positive
  evidence" rule. Two guards bound the *variance* of the empty process,
  not just its mean — centering fixes the expectation, but a mean-zero
  random walk still crosses any threshold given enough variance. First,
  the channel gains: with fresh samples every second, the per-sample gain
  is what sets the probability that a lucky noise run walks `lambda` from
  the empty clamp up to `theta_on` (a ruin problem: the exponent scales
  with `k_bias / k²`); the defaults hold that probability below ~1e-6 per
  excursion in the seeded null simulations while `u_cap` (not the gains)
  is what bounds genuine-entry latching at ~2 s. Second, `u_cap` bounds
  the upward rate only: one wild sample held for a second must not
  out-accumulate what a genuine entry sustains — a spike is an attack
  candidate (4.2), and the attack path carries its own confirmation. The
  downward rate is already bounded by `z_neg_cap`.
- **3.3 Baseline calibration.** `RecordBaseline(zone_id, duration)` (service/
  button, default 120 s) opens a collection window. Collection is sampled at
  the tick clock — **one aligned row per tick** (aggregate energies plus the
  zone's owned-gate energies as one snapshot) — never per entity change:
  entity-change sampling weights samples by publish frequency and tears gate
  tuples across radar frames. The window replaces `(mu, sigma)` (3.1, 3.6)
  and the statistic calibration (3.7). The operator asserts emptiness; the
  engine uses robust statistics so brief violations don't poison the
  baseline. Baselines persist in the config entry.
  **Coverage:** a window with fewer than `stat_min_rows` rows replaces
  *nothing* — floors included — and an individual channel or gate whose
  column has fewer than `stat_min_rows` samples keeps its previous floor:
  a scale cannot be certified from a handful of points (3.1).
  **Lifecycle:** while the window is open the zone's estimator is
  **suspended** — the operator has asserted emptiness, so scoring the
  incoming frames against the old (possibly wrong) floors would let the
  calibration itself manufacture occupancy and ratchet home memory. The
  belief is pinned at the empty prior, `occupied`/`motion` are off, the
  activity FSM is `EMPTY` (entered *without* a `pass_by` — the estimate is
  declared void, nobody traversed), the attack chain is cleared, and
  frames feed only the calibration rows. Fusion sees an ordinary empty
  zone; other zones and legitimate home memory are untouched. On window
  close the new calibration installs and the zone resumes from that empty
  prior — the first post-window frame scores against the new floors.
- **3.4 Background adaptation.** While a zone's confidence stays below
  `p_background` (default 0.05) for at least `t_background` (default 10 min),
  `(mu, sigma)` follow the observed energies with a slow EMA
  (`tau_background`, default 1 h); the deviation target carries the same
  half-quantum guard as 3.1 (the EMA's sampling error is negligible at its
  time constant, the quantization bias is not). Adaptation freezes the
  moment the confidence rises. This tracks seasonal/furniture drift without learning a
  person as noise. Adaptation moves the floors only; the statistic
  calibration (3.7) refreshes only on RecordBaseline, so large drift
  warrants recalibration (a "stuck occupied" recovery story is roadmap,
  8.7).
- **3.5 Still-margin recovery.** Rationale, not a rule: the radar's own
  binary output loses a still person whose energy sits *under* the gate
  threshold; rule 3.2 still credits that margin as positive evidence while
  the frozen distance stays usable (2.7). This is the mechanism that
  bridges the measured dropout gaps.
- **3.6 Per-gate noise floors.** Per zone, per owned gate (2.4) and per
  channel, calibration maintains a `(mu, sigma)` floor — spatial background
  subtraction: a fan or appliance elevating one gate gets its own floor
  there and stops polluting the whole zone. `RecordBaseline` (3.3) collects
  per-gate samples alongside the aggregates and replaces the floor of every
  gate that produced samples; gates that reported nothing keep their
  previous floor. Background adaptation (3.4) extends per gate under the
  same eligibility, clock and freeze conditions. A gate without a recorded
  or adapted floor scores against the `default_mu`/`default_sigma`
  tunables. Persisted per-zone baselines gain an optional `gates` mapping
  (gate index → per-channel `(mu, sigma)`); baselines stored without it
  load unchanged, and zones without per-gate floors persist without it —
  the schema is backward compatible in both directions.
- **3.7 Statistic calibration.** Per zone, per channel and per evidence
  path (gate / aggregate), calibration produces `(m0, s0, c0)` — the mean
  and standard deviation of the raw statistic `S` (3.2) over the
  calibration rows, scored against the *new* floors, plus the residual
  mean of the clamped score (3.2). This calibrates the
  **post-aggregation** statistic: quantization, held/deduplicated values,
  gate correlation and the owned-gate count are all captured empirically,
  so adding gates to a zone cannot silently raise its false-alarm rate.
  A short window estimates a *scale* poorly, and an underestimated `s0`
  inflates every future score — a hazard, not a calibration — so the
  empirical calibration is **shrunk toward safety**: it requires at least
  `stat_min_rows` rows (default 60, else the path stays on the fallback),
  and `s0` is floored by the analytic reference deviation for the path's
  gate count (and by `stat_sigma_min`, default 0.3). Empirical calibration
  may *recentre* the score (deduplicated real traffic sits well below the
  Gaussian mean) but may never *sharpen* it beyond the analytic scale:
  a 120 s window simply cannot certify a smaller-than-Gaussian tail.
  Persisted alongside the floors (optional keys; older baselines load
  without them). When a path has no accepted empirical calibration the
  engine falls back to the **analytic** values for
  `max(0, max of m iid N(0,1))` with `m` = the zone's owned-gate count
  (gate path) or 1 (aggregate path): exact under ideal Gaussian noise,
  conservative (extra-negative `ẑ`) under real deduplicated traffic.
  Rare-tail behavior is never taken from the window at all: the fast
  attack (4.2) thresholds on the analytic tail regardless of calibration,
  until long-run empty recordings exist to certify anything sharper (8.7).
  Statistics are computed over the window's **fresh rows** only (rows on
  which the path's observation counter advanced, 1.1); the lag-1
  autocorrelation `ρ̂` of that fresh sequence — taken at its one-sided
  95% upper confidence bound, since a strongly dependent window holds few
  independent samples and an underestimated discount under-protects —
  yields `τ̂ = clamp((1 + ρ̂)/(1 − ρ̂), 1, tau_int_max)` (AR(1)
  assumption, `tau_int_max` default 25), stored with the statistic and
  applied at runtime (3.2). A path whose fresh rows fall below `stat_min_rows` keeps
  the analytic fallback — if that happens, the room's traffic is too
  deduplicated for a 120 s window: re-run with a longer `duration`.
- **3.8 Held evidence and the observation clock.** The integrator (4.1)
  consumes evidence only while it is *observationally live*, per channel:
  a **positive** centered score integrates for at most `obs_budget`
  (default 1.0 s — the nominal reporting period) after the channel's last
  observation (1.1); a **non-positive** score integrates for at most
  `obs_hold` (default 5 s); the positive/non-positive split is evaluated
  on the margin-shifted score (3.2). Occupied streams pausing between
  reports therefore never drain through the margin — a stale positive
  contributes nothing rather than turning into absence. Past the
  windows a channel contributes nothing, and with everything silent
  `u = 0` — the belief simply relaxes toward the prior (the departure
  hazard is the only thing silence licenses). Rationale: deduplication holds the last
  value in every cache, but a held value is *one* measurement — counting
  it every second lets a single empty-room excursion accumulate
  indefinitely (measured: exact floors, 9 gates, values held 5 s → 96%
  false-occupied hours before this rule). The asymmetry is deliberate and
  conservative in both directions: elevated evidence must be re-observed
  to keep accumulating (a settled person's wobble and the still-energy
  heartbeat both do this naturally), while at-floor evidence may drive
  the zone down a little longer. A latched zone whose observations stop
  entirely decays out over ~2 minutes (`tau_decay`) unless staleness
  (1.3) marks it UNKNOWN first — occupancy without observations is a
  claim the engine refuses to keep making.

## 4. Occupancy filter

- **4.1 Chronological update.** Time integration advances on **every**
  engine input — frame, tick, timer — before that input's own effect is
  applied: `lambda` integrates from the previous event's timestamp to `now`
  using the evidence rate that was in force *during* that interval, and
  only then does a frame install its new evidence. New evidence is never
  applied to past time; outputs are invariant to tick cadence, to a frame
  arriving just before versus just after a tick, and to scheduler pauses.
  The update is the exact constant-input solution of
  `dλ/dt = −(λ − λ_0)/tau_decay + u`:
  `lambda ← λ_eq + (lambda − λ_eq) · exp(−dt / tau_decay)` with
  `λ_eq = λ_0 + tau_decay · u`, where `λ_0` is the empty-state prior
  (default corresponding to a confidence of 0.02) and `tau_decay` defaults
  to 90 s. The relaxation implements the hazard of departure; there is no
  fixed occupancy timeout anywhere in the engine.
- **4.2 Fast attack.** Attack candidacy is a **tail-probability event on
  the raw statistic**, not a centered-score threshold: a gated move
  observation qualifies iff its raw `S` exceeds the analytic threshold
  `Φ⁻¹((1 − attack_tail)^(1/m))` for the path's gate count (`m` owned
  gates, or 1 on the aggregate path; `attack_tail_ppm` defaults to 100
  parts per million — 1e-4 per observation). Mean/std standardization does not equalize *tails* across
  gate counts — a centered threshold of 4.5 fires ~10× more often for one
  gate than for three — and a 120 s calibration window cannot estimate a
  1e-4 tail, so attack thresholds are always analytic (3.7), never
  empirical. The attack fires — `lambda ← max(lambda, lambda_attack)`
  (default corresponding to a confidence of 0.95), immediately, not
  waiting for a tick — once `attack_confirm` (default 2) qualifying
  **fresh move observations** have arrived, consecutive ones separated by
  at least `attack_gap_min` (default 0.3 s) and at most `attack_gap_max`
  (default 3 s). *Fresh* means the frame's `move_energy_obs` counter
  (1.1) advanced — a new move-energy or gate-move measurement was
  actually reported. The adapter re-emits its complete cached frame on
  **any** entity change, so an unrelated update (a still-energy
  heartbeat, a churning distance) re-presents a held move spike without
  any new measurement behind it — elapsed time alone proves nothing, and
  value comparison is the wrong proxy (a forced same-value
  re-publication IS a new measurement; a burst of per-gate entity
  updates from one radar frame is one). `attack_gap_min` additionally
  collapses that burst. Non-fresh
  frames leave the attack state untouched; a fresh non-qualifying move
  observation resets the count. One observation is a max-over-gates
  excursion that empty-room noise produces routinely (3.2); N fresh ones,
  separated by real radar intervals, are a mover. This is still the
  lights-on path — its latency is bounded by the sensor's publish cadence,
  and the tick integration of strong evidence latches occupancy within
  ~2 s regardless. Set `attack_confirm = 1` to restore single-observation
  attack.
- **4.3 Hysteresis.** `occupied` turns on at `lambda ≥ theta_on`
  (default confidence 0.80) and off at `lambda ≤ theta_off` (default
  confidence 0.20). Between thresholds the binary holds.
- **4.4 Motion output.** `motion` is the gated, undamped fast channel:
  on when a gated frame has `ẑ_move ≥ z_motion` (default 2.0, centered
  units) or `has_moving_target` with a gated distance; off after
  `motion_hold` (default 5 s) without such evidence. It exists for
  automations that want raw responsiveness (hallway lights) and accepts
  flicker by design; occupancy (4.2, 4.3) deliberately demands more.
- **4.5 Clamp.** `lambda` is clamped to `[lambda_min, lambda_max]`
  (confidence 0.001 / 0.999) so occupation cannot build unbounded inertia.
  Clearing the ceiling takes ~7–10 s of consecutively observed at-floor
  readings; measured empty streams keep reporting at a ~2.5 s cadence, so
  departures clear promptly, while sub-threshold dropout margins (above
  the floor, 3.5) never count as observed absence and bridge. The clamp is
  applied continuously (per integration segment), so trajectories are
  cadence-invariant.

## 5. Activity classification and pass-by

A per-zone FSM driven by the occupancy belief and channel dominance:

- **5.1 States.** `EMPTY` (not occupied) → `PASSING` (occupied, since less
  than `t_dwell`, default 45 s, and no still-takeover yet) → `ACTIVE`
  (occupied past `t_dwell` with ongoing move evidence) / `SETTLED`
  (still evidence has dominated for `t_settle`, default 30 s — the seated /
  sleeping case). `ACTIVE ↔ SETTLED` follow channel dominance with the same
  `t_settle` smoothing. Dominance must be **continuous**: a channel's
  dominance clock runs only while that channel's positive score exceeds the
  other's; when neither channel dominates (quiet or equal evidence) both
  clocks reset — a single dominant frame followed by silence must not
  mature into a takeover `t_settle` later.
- **5.2 Pass-by event.** `EMPTY` reached *from* `PASSING` emits `pass_by`
  with the zone's peak confidence and traversal duration. Reached from
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
  occupied. Room `confidence` is the **maximum** member confidence,
  published for diagnostics. Not noisy-OR: member zones are strongly
  dependent (shared boundary gates, the same person, cross-covering
  sensors), and independence-assuming combination overstates the result —
  four independent empty zones at 0.02 would already noisy-OR to 0.078.
  The max is the honest dependence-free upper bound of the members.
- **6.2 Activity and motion.** Room activity is the maximum-severity member
  state (`settled > active > passing > empty`). Room `settled` is true iff
  any member zone is `SETTLED`. Room `motion` is true iff any healthy member
  zone's motion (4.4) is on — the same undamped fast channel, fused with OR,
  and it inherits 4.4's flicker-by-design contract.
- **6.3 Health.** Zones in `UNKNOWN` health (1.3) are excluded from fusion.
  A room with all members unknown publishes unknown, not off — downstream
  automations must be able to distinguish "nobody there" from "blind".
- **6.4 No cross-zone inhibition.** Fusion is monotone: a zone can only add
  occupancy to its room, never veto another zone. Separation is done at the
  gate (2.2), not at fusion.
- **6.5 Home presence.** The engine maintains a home-level belief
  `lambda_home` ("someone is in the apartment"; its confidence is the
  sigmoid, with the same non-probabilistic caveat as §0). Any healthy zone
  being occupied drives it up immediately; with all zones empty it decays
  toward the empty prior with `tau_home` (default 20 min), advancing
  chronologically on every event like 4.1 —
  deliberately much slower than zone decay, because the sensors do not cover
  every room: all-zones-empty means "not seen lately", not "gone". Binary
  `anyone_home` follows hysteresis thresholds like 4.3. If all zones are
  unhealthy (6.3), `anyone_home` publishes unknown. Departure evidence
  (entrance door + no re-detection) is a planned refinement (8.3); until
  then `tau_home` is the honest ceiling on how fast "away" can be declared.

## 7. Failure modes and lifecycle

- **7.1 Restart adoption.** On startup the engine seeds from an
  `InitialSnapshot` of current entity states; beliefs start at the prior,
  except zones whose sensor currently reports a gated target, which start at
  `theta_on` (someone plainly there should not wait out a cold start).
- **7.2 Disabled.** A global `enabled` switch: while off, the engine keeps
  ingesting frames and updating state (so re-enable is warm) but publishes no
  transitions and emits no events.
- **7.3 Determinism.** Same event sequence + timestamps ⇒ same outputs.
  All tunables live in one `Tunables` dataclass; nothing reads config at
  update time.

## 8. Non-goals (v1) and roadmap

- **8.1** Per-gate energies (`g0..g8`, engineering mode) are consumed since
  phase 2: gate ownership (2.4), max-aggregated gate evidence (2.5),
  per-frame precedence over the aggregate path (2.6) and per-gate noise
  floors with backward-compatible persistence (3.6). There is no capability
  flag: gate consumption is implied by a sensor's configured gate entities,
  and the engine degrades to the aggregate path per frame whenever
  engineering mode drops (it does not survive a radar power-cycle; the
  ESPHome overlay re-asserts it within about a minute). The data-enablement
  overlay itself — engineering-mode re-assert, per-sensor filter tuning,
  recorder-exclusion guidance for the 18 gate entities — lives with the
  device configs (`homeassistant-bjaalands/esphome/*.yaml`), not in this
  repo. Still out of scope here: writing gate thresholds or using gate data
  for auto-threshold suggestions (8.2).
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
- **8.6 sonos-conductor integration path.** Zone outputs are designed to
  replace the template occupancy helpers feeding sonos-conductor's audio
  zones 1:1 (`occupied` as the drop-in, `activity ∈ {active, settled}` as
  the richer upgrade so a walk-through never wakes a speaker). Deeper
  integration — sonos-conductor consuming `activity`/`pass_by` natively, or
  a future lighting FSM — happens on the consumer side; this engine stays a
  presence estimator and grows no audio or lighting awareness.
- **8.7 Statistical status and roadmap.** The estimator is a **calibrated
  anomaly-score filter**, deliberately not a Bayesian one: only the empty
  (H0) distribution is learned, `u` (3.2) is not a log-likelihood ratio,
  and `confidence` is a monotone score, not a calibrated posterior — which
  is why nothing in this spec or the entity surface says "probability". The
  guarantees actually made are: negative expected evidence when calibrated
  and empty (3.2, 3.7), gate-count-invariant false-alarm behavior (3.7),
  and chronology-invariant integration (4.1). The statistically complete
  upgrade — a two-state HMM with transition hazards and an emission model
  learned from labeled empty/occupied data, plus reliability-curve
  validation before any output is again called a probability — is roadmap,
  as are: calibration quality gates (minimum coverage, contamination
  rejection, staleness diagnostics), sensor-scoped gate floors shared
  between zones, recovery from sustained upward background shifts, labeled
  replay metrics (false-occupied minutes per empty hour, entry/exit latency
  percentiles), and a bounded synchronized frame-capture facility for
  evaluating the production gate path (which also gates flipping
  `use_gate_evidence` on by default, 2.6). Temporal-model assumptions
  (3.7, 3.8): dependence is treated as AR(1)-like with a clamped
  integrated autocorrelation time estimated from the calibration window;
  the observation clock treats each reported update as one measurement.
  Cross-gate and move/still cross-correlation are not modeled beyond the
  max-statistic calibration.
