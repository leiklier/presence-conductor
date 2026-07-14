# Estimator rationale and validation

This document explains why the normative rules in [ENGINE_SPEC](ENGINE_SPEC.md)
look the way they do. It is evidence and design context, not an additional
behavioral contract.

## Why an empty-room anomaly score

The available calibration assertion is “this room is empty.” There is no
reliable labeled occupied corpus spanning people, poses, rooms, and radar
placements. The estimator therefore models `H0` (empty) and asks how unusual a
new return is under that distribution. Calling the output a posterior
probability would be statistically false; `confidence` is a bounded monotone
score.

The key safety invariant is negative expected empty-room drive. A one-sided
score `max(0,Z)` has positive mean even for perfectly symmetric Gaussian noise,
and the maximum over gates grows with gate count. Centering the exact aggregate
statistic, then subtracting an absence margin, prevents topology-dependent
upward drift.

## Conservative floor estimation

LD2410 energies are quantized integer values. A raw MAD can land one grid step
low, understating scale by 25–40% in short windows. Because every future score
divides by that estimate, underestimating uncertainty is much more dangerous
than overestimating it.

The fit therefore combines:

- a median location;
- a one-sided rank upper bound for median absolute deviation;
- half a reporting quantum;
- a hard minimum scale;
- a dependence-discounted effective sample count;
- a Bonferroni family-wise quantile for a max over gates.

The dependence adjustment is a conservative engineering approximation for the
supported IID, held, and quantized AR(1) process family. It is validated by
seeded null sweeps; it is not claimed as a distribution-free coverage theorem.

## Observation time is not scheduler time

A cached radar value is one measurement, even when the engine receives ticks
for several seconds. Re-integrating it on each tick caused held or correlated
empty excursions to accumulate like independent evidence. The observation
clock records explicit sensor reports, while held-evidence budgets limit how
long one report can contribute.

Positive evidence receives the shorter budget because repeated high returns are
needed to keep making an occupancy claim. At-floor evidence may contribute a
little longer, but an occupied reporting gap never turns stale positive energy
into absence. With complete silence, only the departure hazard relaxes belief.

Home Assistant’s `state_reported` event is essential here: exact same-value
reports are real observations even without `force_update`. Conversely, an
attribute-only change is ambiguous metadata and cannot certify a new radar
sample or confirm a cached spike.

## Fast attack and dependence

Entry latency needs a path faster than the bounded accumulator. Its candidate
threshold comes from the analytic tail of the raw maximum statistic, not from a
minutes-long empirical window that cannot estimate a 1e-4 tail.

Confirmation only reduces false alarms when observations are separated enough
to contain new information. Empty streams with lag-1 correlation around 0.9 can
turn two one-second exceedances into effectively one event. Calibration stores
both dimensionless integrated autocorrelation time for score discounting and a
physical decorrelation interval for attack spacing. Aggregate and gate
candidates never confirm each other, and only actual move-energy reports may
advance the chain.

## Gate evidence remains experimental

The maximum over per-gate energy is spatially better than a sensor-global
distance and supports simultaneous people in different zones. It also has a
more difficult temporal distribution: gate update cadence, run lengths,
cross-gate dependence, and engineering-mode loss have not yet been captured in
a bounded synchronized production dataset. Aggregate evidence therefore
remains the default, with gate evidence opt-in and automatic per-frame fallback.

## Empirical findings that shaped the design

- Distance freezes at the previous target location when the target flag drops;
  it does not reliably clear. This led to core-owned distance freshness.
- Empty still energy reports roughly every 2.5 s. A 120 s calibration often
  provided only about 48 fresh observations; 300 s provides roughly 120.
- Many distance reports have no nearby energy state change. Under the verified
  atomic-frame firmware contract these are re-measured energy plateaus, not
  silence; energy-only epochs starved real occupancy in replay.
- Held five-second gate values and strongly autocorrelated empty simulations
  falsely occupied nearly every tested hour before held-evidence and dependence
  handling.
- Per-fit 95% gate scales were not simultaneous: across nine gates the minimum
  fitted scale was often too small. Family-wise bounds and decorrelated attack
  confirmation removed the seeded failures.

The offline replay harness checks production history for transition regressions;
seeded simulation checks false-occupied hours, timing invariance, attack chains,
and calibration coverage. Neither replaces labeled occupancy metrics.

## Roadmap

The statistically complete upgrade is a two-state model with transition hazards
and occupied/empty emission distributions learned from labeled data, followed by
reliability-curve validation. Other useful work includes:

- calibration contamination detection and long-run drift diagnostics;
- sensor-scoped gate floors shared safely between zones;
- recovery from sustained upward background shifts;
- labeled false-occupied minutes and entry/exit latency percentiles;
- bounded synchronized aggregate/gate frame capture;
- explicit cross-gate and move/still dependence models;
- door or other departure evidence for home-level presence.
