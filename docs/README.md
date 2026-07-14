# Presence Conductor documentation

| Document | Purpose |
| --- | --- |
| [Engine specification](ENGINE_SPEC.md) | Normative numbered behavior implemented by the core and adapter. |
| [Calibration operations](CALIBRATION.md) | Operator workflow, coverage verdicts, diagnostics, Repairs, and fallback behavior. |
| [Estimator rationale](ESTIMATOR_RATIONALE.md) | Statistical/DSP reasoning, assumptions, empirical evidence, and roadmap. |
| [Decision record](DECISION.md) | Original architecture choice and deployment comparison. |

Start with the engine specification when changing behavior. Use the operations
guide when deploying or recalibrating, and the rationale when evaluating or
changing the estimator’s statistical model.

## Code ownership map

| Module | Responsibility |
| --- | --- |
| `controller.py` | Single-writer orchestration: lifecycle, HA subscriptions, plans, timers, and publication. |
| `observation.py` | Entity-state cache, observation epochs, availability, and `SensorFrame` construction. |
| `calibration.py` | Baseline serialization, compatibility diagnostics, calibration events, and Repairs. |
| `entity.py` | Shared device placement and dispatcher-driven entity base class. |
| `core/` | Deterministic estimator with no Home Assistant dependencies. |
