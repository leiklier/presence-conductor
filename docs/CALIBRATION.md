# Calibration operations

This guide describes the operator-facing calibration workflow. The normative
estimator rules are [ENGINE_SPEC §3](ENGINE_SPEC.md#3-evidence-and-calibration).

## What calibration measures

Calibration records the empty-room distribution after the same aggregation the
runtime estimator uses. It learns:

- aggregate move and still energy floors;
- optional per-gate move and still floors;
- empty-score center, scale, autocorrelation, and physical decorrelation time
  when enough fresh observations exist.

It does not learn an occupied distribution. The published confidence remains a
monotone score rather than an occupancy probability.

## Recording a baseline

1. Keep the selected zone empty.
2. Press its **Record baseline** button or call
   `presence_conductor.record_baseline`.
3. Leave the sensor and zone undisturbed for the full window. The default is
   300 seconds because a healthy empty still channel reports about every 2.5 s;
   the tick rate is not the sample rate.
4. Inspect the zone’s calibration-status sensor or calibration event.

The window starts when the button/service is invoked. It never uses samples
from before the request.

During recording, the selected zone is intentionally suspended: occupancy and
motion are off, belief stays at the empty prior, activity is `EMPTY`, and frames
feed the candidate calibration only. Other zones continue normally.

## Transactional result

Each path receives one verdict before anything is committed:

| Verdict | Meaning |
| --- | --- |
| `calibrated` | Enough independent variation and fresh observations were available. |
| `quiescent` | The signal remained within one reporting quantum, but the sensor repeatedly certified the plateau. |
| `no_data` | The optional channel was not configured or never appeared. |
| `rejected` | Coverage, freshness, family completeness, or availability was insufficient. |

Aggregate paths are required. Configured gate families are atomic across all
owned gates. If any required path fails, the entire candidate is rejected and
the previous calibration remains unchanged. Recalibrating one zone never
rewrites another zone’s stored provenance.

## Calibration-status sensor

Every zone exposes an enabled diagnostic enum:

- `ready` — the stored context is compatible;
- `uncalibrated` — no recorded baseline exists;
- `recalibration_required` — stored data is incompatible or incomplete;
- `calibrating` — a window is currently open.

Attributes explain the decision:

- `reason_codes` and `reasons` identify missing or incompatible context;
- `floor_source` is `recorded` or `default`;
- `action` gives the next operator step when one is required.

The per-frame runtime paths — `move_runtime` / `still_runtime` (the
aggregate or gate path actually used by the current frame) and
`move_statistic` / `still_statistic` (empirical or analytic centering) —
live in the config entry's diagnostics download, not on the entity: they
can flip frame to frame, and every attribute change would write a
recorder row.

Any non-ready zone creates one nonpersistent Home Assistant Repairs warning for
the config entry. It names every affected zone and disappears only when all are
ready. A failed entry unload retains the warning; a successful unload removes
it.

## Why recalibration may be required

Stored calibration is bound to its sensor, gate resolution, exact owned-gate
family, floor-fit settings, and score transform. Safe fallbacks are automatic:

| Change | Runtime behavior |
| --- | --- |
| Sensor identity changed | Use default aggregate floors. |
| Floor-fit settings changed | Use default floors. |
| Gate family or resolution changed | Disable the incompatible gate family and use aggregate evidence. |
| Gate evidence enabled without complete gate calibration | Use aggregate evidence. |
| Statistic transform changed | Keep compatible floors but use analytic score statistics. |
| Legacy metadata | Continue safely and request a fresh baseline. |

Background adaptation is not a substitute for explicit calibration: it may
track a quiet mean and increase uncertainty, but it cannot reduce the recorded
conservative scale or make a provisional gate family ready.
