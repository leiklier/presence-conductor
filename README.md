# Occupancy Conductor

Robust occupancy estimation for mmWave presence sensors, as a Home Assistant
custom integration (HACS).

mmWave radars like the HLK-LD2410B (Apollo MSR-2 and friends) detect people
with binary per-gate thresholds inside the module. The result flaps: a still
person drops out for seconds at a time, drafts and fans ghost in, and every
downstream automation compensates with fixed delays. This integration
consumes the radar's *energy and distance* streams instead of its binary
verdicts and runs a calibrated Bayesian filter over them, publishing signals
automations can actually use:

- **occupancy** — robust "someone is in the zone", bridging dropouts and
  suppressing ghosts, with no fixed timeouts
- **motion** — the low-latency channel for instant lights-on
- **activity** — `empty / passing / active / settled`, so a walk-through
  never flickers the living-room lights and a seated reader never sits in
  the dark
- **pass-by events**, dwell time, and a live probability per zone
- **room-level fusion** of multiple sensors covering one room, with distance
  cutoffs so they don't claim each other's areas
- **guided calibration** — record an empty-room baseline instead of hand
  tuning 18 gate thresholds

Status: design phase. The architecture decision is documented in
[docs/DECISION.md](docs/DECISION.md); the normative engine contract lives in
[docs/ENGINE_SPEC.md](docs/ENGINE_SPEC.md).

Sibling project: [sonos-conductor](https://github.com/leiklier/sonos-conductor),
whose pure-core architecture this repo mirrors.
