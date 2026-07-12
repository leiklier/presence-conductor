# Architecture decision: where does the estimation run?

**Status:** accepted (2026-07-12)
**Decision:** Home Assistant custom integration with a pure-Python estimation
core, fed by the sensor entities the Apollo MSR-2 firmware already exposes.
On-device (ESPHome) changes are limited to a thin, optional *data-enablement*
overlay in a later phase — never estimation logic.

## Problem

Four Apollo MSR-2 (ESP32-C3 + HLK-LD2410B, 24 GHz FMCW) sensors drive
presence-based lighting and audio (sonos-conductor) automations. The LD2410's
internal detection is binary per-gate thresholding plus a hold timer; even
carefully hand-tuned it flaps. Measured over 24 h on the live instance
(2026-07-12):

| zone sensor | transitions/24h | median ON | median OFF | OFF gaps <15 s | ON blips <10 s |
|---|---|---|---|---|---|
| kontor     | 2558 | 4 s  | 2 s  | 1191 | 865 |
| sofakrok   | 480  | 10 s | 17 s | 109  | 129 |
| kjøkken    | 422  | 12 s | 2 s  | 157  | 96  |
| spisebord  | 96   | 1.1 m| 3.0 m| 6    | 2   |

Two failure modes dominate: **dropouts** (a still person falls below the
still-energy threshold for seconds at a time — the sofakrok TV-evening
pattern) and **ghost blips** (short spurious detections). Downstream
automations compensate with fixed delays, which trades latency for stability
in both directions at once.

A known environmental factor at kontor: a laptop sometimes sits between the
occupant and the sensor, attenuating returns — partial occlusion presents as
exactly the dropout pattern above. This favors an estimator that credits
weakened sub-threshold energy as evidence over anything that trusts the
radar's binary verdict.

Additional requirements: multiple sensors may cover one room (stue =
sofakrok + spisebord) and must not claim each other's areas
(distance cutoffs); outputs should distinguish *passing through* from
*occupying*; calibration should be easier and more principled than manual
per-gate threshold tuning.

## Options considered

### A. On-device estimation (fork/extend the ESPHome config or component)

The LD2410 pushes frames at ~10–20 Hz over UART; ESPHome could run the
estimator at full rate with zero HA involvement and publish only derived
state.

Rejected as the primary architecture because:

- **Cross-device fusion is impossible on-device.** The stue requirement means
  the interesting layer is central no matter what; on-device estimation would
  split the algorithm across two runtimes.
- **Testability.** The quality bar here is a fully unit-tested estimator.
  ESPHome lambdas / C++ components have no comparable test story; a
  pure-Python core runs thousands of scenario tests in CI.
- **Iteration cost.** Every tuning change would be recompile + OTA × 4
  devices. HA-side, it is a config-entry update.
- **Fleet drift.** The devices pull Apollo's remote package (`MSR-2_BLE.yaml`
  from GitHub `main`, refresh 1 min); estimation logic embedded there would
  sit on a moving upstream.

### B. HA integration on existing entities (chosen)

The stock Apollo firmware already publishes, per device, at ~1 Hz (ESPHome
default `throttle_with_priority: 1000ms` / binary `settle: 1000ms`):
`moving_distance`, `still_distance`, `move_energy`, `still_energy`,
`detection_distance`, and the three target binaries. The 18 per-gate energy
sensors (`g0..g8` move/still) also already exist as entities and stream once
the `engineering mode` switch is on (additive: normal detection keeps
working).

The decisive observation: the flapping is caused by *binary thresholding*
inside the radar, but the **sub-threshold energy margins are visible in the
energy sensors**. A person sitting still whose still-energy hovers just under
the gate threshold is invisible to the radar's `has_target` — but plainly
visible to an estimator that tracks energies against a calibrated noise
floor. 1 Hz input is ample for occupancy dynamics (seconds-scale); the
latency-critical path (lights-on) is served by a fast-attack rule on strong
motion evidence.

### C. Hybrid with on-device preprocessing

Considered as "A for signal conditioning + B for fusion". Rejected *for now*
because phase 1 needs no firmware change at all — conditioning at 1 Hz in the
core is sufficient, and every on-device change costs an OTA cycle across the
fleet. Retained as the **phase-2 data-enablement overlay** (explicitly not
estimation logic):

- re-assert `engineering mode` after radar power-cycle (it does not persist);
- `filters: []` / `delta` + `throttle_average` tuning on selected sensors
  where the estimator wants faster or cleaner input than 1 Hz;
- recorder-exclusion guidance for the gate-energy entities (18 sensors/device
  at ~1 Hz would bloat the recorder DB; they should feed the estimator, not
  history).

The overlay composes as an additional ESPHome package on top of Apollo's
remote package in `homeassistant-bjaalands/esphome/*.yaml` — no fork of the
Apollo repo.

## Facts the decision rests on

Verified against primary sources (ESPHome `dev` source, Apollo MSR-2 repo,
HLK-LD2410B manual V1.04) on 2026-07-12:

1. Engineering mode is additive (frame type 0x01 carries the normal payload
   plus per-gate energies); it does not degrade detection, but does not
   survive a radar power-cycle.
2. ESPHome ≥ 2025.8 applies default 1 s per-sensor filters to all ld2410
   sensors; `filters: []` restores full frame rate (~10–20 Hz) per sensor.
   Identical consecutive values are never re-published (dedup).
3. LD2410B fw ≥ V2.44 (fleet is on 2.44.24073110) has an auto
   background-noise calibration command (0x000B) — not exposed by ESPHome;
   possible future external component, not needed for v1.
4. Gate masking (sensitivity 100) and the existing per-device distance-zone
   numbers are the radar-side tools for spatial cutoffs; the integration adds
   its own distance masks on top, which is what enables same-room sensor
   pairs.
5. Prior art (HA `bayesian`, Hankanman/Area-Occupancy-Detection,
   Everything Presence Lite, wasp-in-a-box) fuses *binary/derived* entities
   or relies on LD2450 coordinates. Nothing consumes LD2410 energies with
   multi-device room fusion and pass-by classification. The niche is open.

## Consequences

- Repo mirrors the sonos-conductor architecture: `core/` pure Python (no
  `homeassistant` imports, CI-enforced), a single-writer controller adapter,
  entities that publish engine state, `docs/ENGINE_SPEC.md` as the normative
  contract.
- Phase 1 ships with zero changes to the devices or the ESPHome configs.
- Phase 2 (overlay + gate energies) and phase 3 (auto-threshold external
  component) are additive and optional; the spec keeps gate-energy input
  behind a capability flag.
- An offline replay harness (history export → estimator → transition metrics)
  ships with the integration so tuning decisions are made against recorded
  reality, with the table above as the baseline to beat.
