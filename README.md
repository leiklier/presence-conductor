# Presence Conductor

Robust occupancy estimation for mmWave presence sensors, as a Home Assistant
custom integration (HACS).

mmWave radars like the HLK-LD2410B (Apollo MSR-2 and friends) detect people
with binary per-gate thresholds inside the module. The result flaps: a still
person drops out for seconds at a time, drafts and fans ghost in, and every
downstream automation compensates with fixed delays. This integration
consumes the radar's *energy and distance* streams instead of its binary
verdicts and runs a calibrated anomaly-score filter over them (scores are
centered against each zone's measured empty-room distribution, so sensor
noise can never accumulate into occupancy). Rooms and the home are the
consumer surface — one Home Assistant device per room, plus a hub —
publishing signals automations can actually use:

- **one device per room** — each room's full presence surface on one device:
  occupancy, motion, activity, settled, pass-by events and a live
  confidence, fused from every zone (of any sensor) covering the room, with
  distance cutoffs so sensors don't claim each other's areas
- **occupancy** — robust "someone is in the room", bridging dropouts and
  suppressing ghosts, with no fixed timeouts
- **motion** — the low-latency channel for instant lights-on
- **activity** — `empty / passing / active / settled`, so a walk-through
  never flickers the living-room lights and a seated reader never sits in
  the dark
- **anyone-home** — an apartment-wide presence estimate with a slow memory,
  for automations that only need to know whether anybody is around
- **zone-level diagnostics** — the per-zone estimator outputs behind every
  room (occupancy, motion, activity, confidence, dwell, pass-by) live on
  the room's device but ship disabled; enable individual entities when
  debugging the estimate or when an automation wants one specific distance
  slice. Each zone's record-baseline button stays enabled — calibration is
  a first-class operator action
- **per-gate spatial evidence** — when the radar's engineering mode streams
  per-gate energies, zones are scored gate-by-gate against per-gate noise
  floors (a fan ghosting at one distance gets its own floor instead of
  polluting the zone), falling back to the aggregate path per frame whenever
  the gate stream drops
- **guided calibration** — record an empty-room baseline per zone instead of
  hand tuning 18 gate thresholds

Status: deployed in production. The architecture decision is documented in
[docs/DECISION.md](docs/DECISION.md). The documentation map starts at
[docs/README.md](docs/README.md), and the normative engine contract lives in
[docs/ENGINE_SPEC.md](docs/ENGINE_SPEC.md).

Sibling project: [sonos-conductor](https://github.com/leiklier/sonos-conductor),
whose pure-core architecture this repo mirrors.
