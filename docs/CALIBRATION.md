# Calibration operations

This guide describes the operator-facing calibration workflow. The normative
estimator rules are [ENGINE_SPEC §3](ENGINE_SPEC.md#3-evidence-and-calibration).

## Choose a calibration level

Presence Conductor asks for one target during setup. Setup finishes
immediately; radar capture happens afterward on the room devices and can be
upgraded later through **Configure → Calibration**.

| Level | What it does | When to use it |
| --- | --- | --- |
| Skip | Skips prompts; manual capture remains available and compatible saved calibration stays active. | Evaluation, or a room that cannot be emptied yet. |
| Simple (recommended) | Learns empty-room floors, score centering, and timing. | The quickest useful and safest default. |
| Full | Adds moving and stationary occupied emissions, then verifies them on a second pass. | Rooms where Simple still produces nuisance occupancy or misses stationary people. |

The existing **Record baseline** button and `record_baseline` service remain
the quick Simple workflow regardless of the selected target.

## What calibration measures

Calibration records the empty-room distribution after the same aggregation the
runtime estimator uses. It learns:

- aggregate move and still energy floors;
- optional per-gate move and still floors;
- empty-score center, scale, autocorrelation, and physical decorrelation time
  when enough fresh observations exist.

Full calibration additionally learns how the normalized move/still feature pair
separates local empty, moving, standing, and seated observations. It stores only
model coefficients and held-out validation counts; raw labeled rows are
discarded. Published confidence remains a monotone score, not a probability.

## Recording a baseline

1. Keep the physical sensor's **entire field of view** empty. Aggregate LD2410
   energies are sensor-global, so a person in a sibling distance zone would
   contaminate the candidate.
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
- `move_runtime` / `still_runtime` show the aggregate or gate path actually
  used by the current frame;
- `move_statistic` / `still_statistic` show empirical or analytic centering;
- `action` gives the next operator step when one is required.

Any non-ready zone creates one nonpersistent Home Assistant Repairs warning,
except a zone explicitly set to Skip. A valid empty baseline with Full still
outstanding appears as `partial`.

## Full guided calibration

Set **Configure → Calibration** to Full. Each zone then exposes **Start full
calibration**, **Record next calibration phase**, and **Cancel calibration**.
Run one zone at a time:

1. Empty the whole sensor field of view and press **Start full calibration**.
   The normal transactional baseline runs first.
2. Watch the zone's calibration-status sensor. Its `action`, `phase`,
   `samples`, and `sample_progress` attributes give the next instruction.
3. Prepare the requested condition, press **Record next calibration phase**,
   and continue that behavior for the 60-second phase.
4. Training phases are empty, natural walking throughout the zone, quiet
   standing in typical positions, and seated stillness in typical positions.
5. Repeat moving, standing, and seated for held-out validation, varying paths
   and positions. The final validation phase is empty, leaving the room safe
   for normal operation when the workflow finishes.

The selected physical sensor is intentionally published empty throughout the
session, so automations do not react to scripted motion. Other sensors continue
normally. A phase rejects with fewer than 15 fresh observations, sensor loss,
or a stale close. Cancellation and restart discard in-memory rows.

The result reports TP, FP, TN, FN, sensitivity, specificity, balanced accuracy,
and recall for each occupied scenario. The profile commits only when aggregate
sensitivity reaches 70%, specificity 80%, and every occupied scenario 50%
recall. It also replays the ordered captures through the runtime belief filter:
all occupied scenarios must latch, empty must not latch, and mean drive must
have the correct sign. A failed first Full attempt keeps its useful empty
baseline; failed recalibration restores the previously committed Full profile.
Retry with more representative positions.

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
