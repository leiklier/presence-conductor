"""Pure learned occupied-emission statistics.

The legacy estimator models empty-room anomalies.  A full guided calibration
can additionally fit this small discriminative layer over the estimator's
already clipped and empty-standardized ``(z_move, z_still)`` features.

The model is regularized two-dimensional linear discriminant analysis (LDA):
EMPTY, ACTIVE and SETTLED receive equal weight when estimating a shared
within-class covariance, then ACTIVE-vs-EMPTY and SETTLED-vs-EMPTY become two
linear log-density ratios.  Equal class and occupied-mode weights ensure that
operator-selected recording durations do not become accidental priors.

This module owns no state, clock or I/O and has no third-party dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .model import (
    ConfusionMatrix,
    EmissionScenario,
    EmissionValidationMetrics,
    LinearDiscriminant,
    OccupiedEmissionProfile,
    ScenarioEmissionMetrics,
)

type Feature = tuple[float, float]


def emission_fingerprint(
    *,
    path: str,
    gate_indices: tuple[int, ...],
    floor_fingerprint: str,
    move_stat_fingerprint: str,
    still_stat_fingerprint: str,
    feature_version: str = "z-move-still-v1",
    sensor_id: str = "",
    zone_geometry: str = "",
) -> str:
    """Return a deterministic compatibility key for a learned profile.

    Profiles are meaningful only under the feature transform, evidence path,
    exact gate family, and empty calibration transforms that produced them.
    Delimiters are safe because current fingerprints contain no newlines; the
    length-prefix makes the representation unambiguous even if future values
    contain ``|``.
    """

    fields = (
        "emission-v1",
        feature_version,
        path,
        (
            ",".join(str(index) for index in gate_indices)
            if path == "gate" or path.endswith("_gate")
            else "-"
        ),
        floor_fingerprint,
        move_stat_fingerprint,
        still_stat_fingerprint,
        sensor_id,
        zone_geometry,
    )
    return "|".join(f"{len(value)}:{value}" for value in fields)


def fit_occupied_profile(
    empty: Sequence[Feature],
    active: Sequence[Feature],
    settled: Sequence[Feature],
    *,
    shrinkage: float = 0.25,
    variance_floor: float = 0.01,
    active_weight: float = 0.5,
    evidence_scale: float = 1.0,
    evidence_min: float = -3.0,
    evidence_max: float = 3.0,
    path: str = "aggregate",
    fingerprint: str | None = None,
) -> OccupiedEmissionProfile:
    """Fit equal-class-weight regularized shared-covariance 2-D LDA.

    Features must already be finite, clipped empty-standardized scores.  Each
    class covariance uses its population normalization and the three class
    covariances are averaged equally.  Consequently repeating an entire
    class recording does not alter its weight or turn recording duration into
    a prior.  Shrinkage pulls covariance toward an isotropic matrix and
    ``variance_floor`` makes constant or collinear captures well-defined.

    The returned discriminants use equal EMPTY-vs-mode priors.  ACTIVE and
    SETTLED are combined later with ``active_weight``; their sample counts do
    not set that mixture.
    """

    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    if not math.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError("variance_floor must be finite and positive")
    if not 0.0 < active_weight < 1.0:
        raise ValueError("active_weight must be strictly between 0 and 1")
    if not math.isfinite(evidence_scale) or evidence_scale <= 0.0:
        raise ValueError("evidence_scale must be finite and positive")
    if not all(map(math.isfinite, (evidence_min, evidence_max))) or evidence_min >= evidence_max:
        raise ValueError("evidence bounds must be finite and increasing")
    if path not in {"aggregate", "gate"}:
        raise ValueError("path must be 'aggregate' or 'gate'")

    classes = tuple(
        _checked_features(name, rows)
        for name, rows in (
            ("empty", empty),
            ("active", active),
            ("settled", settled),
        )
    )
    means = tuple(_mean(rows) for rows in classes)
    covariances = tuple(
        _population_covariance(rows, mean) for rows, mean in zip(classes, means, strict=True)
    )

    # Equal class weighting is intentional: guided phase duration is an
    # operator choice, not a deployment prior.
    c00 = sum(cov[0] for cov in covariances) / len(covariances)
    c01 = sum(cov[1] for cov in covariances) / len(covariances)
    c11 = sum(cov[2] for cov in covariances) / len(covariances)
    isotropic = (c00 + c11) / 2.0
    c00 = (1.0 - shrinkage) * c00 + shrinkage * isotropic + variance_floor
    c01 = (1.0 - shrinkage) * c01
    c11 = (1.0 - shrinkage) * c11 + shrinkage * isotropic + variance_floor

    determinant = c00 * c11 - c01 * c01
    # Positive diagonal loading theoretically makes the matrix positive
    # definite.  This defensive check keeps crafted extreme floats total.
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise ValueError("regularized covariance is not positive definite")
    inverse = (c11 / determinant, -c01 / determinant, c00 / determinant)
    empty_mean, active_mean, settled_mean = means

    return OccupiedEmissionProfile(
        active=_discriminant(empty_mean, active_mean, inverse),
        settled=_discriminant(empty_mean, settled_mean, inverse),
        path=path,
        active_weight=active_weight,
        evidence_scale=evidence_scale,
        evidence_min=evidence_min,
        evidence_max=evidence_max,
        fingerprint=fingerprint,
        empty_rows=len(classes[0]),
        active_rows=len(classes[1]),
        settled_rows=len(classes[2]),
    )


def discriminant_score(discriminant: LinearDiscriminant, feature: Feature) -> float:
    """Evaluate one mode-vs-empty linear log-density ratio."""

    move, still = _checked_feature(feature)
    score = discriminant.move_weight * move + discriminant.still_weight * still
    score += discriminant.intercept
    if not math.isfinite(score):
        raise ValueError("discriminant produced a non-finite score")
    return score


def occupied_discriminant(profile: OccupiedEmissionProfile, feature: Feature) -> float:
    """Combine ACTIVE and SETTLED emissions as a normalized log mixture.

    This is ``log(w*exp(active) + (1-w)*exp(settled))``.  The stable
    log-sum-exp form remains finite for strongly separated calibrations.
    Because mixture weights sum to one, identical mode models return their
    original score rather than gaining an accidental multiple-comparison
    bonus.
    """

    if not 0.0 < profile.active_weight < 1.0:
        raise ValueError("profile active_weight must be strictly between 0 and 1")
    active = discriminant_score(profile.active, feature) + math.log(profile.active_weight)
    settled = discriminant_score(profile.settled, feature) + math.log1p(-profile.active_weight)
    high = max(active, settled)
    combined = high + math.log(math.exp(active - high) + math.exp(settled - high))
    if not math.isfinite(combined):
        raise ValueError("occupied mixture produced a non-finite score")
    return combined


def learned_evidence_rate(profile: OccupiedEmissionProfile, feature: Feature) -> float:
    """Return the profile's scaled discriminant, clamped to safe bounds."""

    if not math.isfinite(profile.evidence_scale) or profile.evidence_scale <= 0.0:
        raise ValueError("profile evidence_scale must be finite and positive")
    if (
        not math.isfinite(profile.evidence_min)
        or not math.isfinite(profile.evidence_max)
        or profile.evidence_min >= profile.evidence_max
    ):
        raise ValueError("profile evidence bounds must be finite and increasing")
    rate = occupied_discriminant(profile, feature) * profile.evidence_scale
    return min(profile.evidence_max, max(profile.evidence_min, rate))


def validate_profile(
    profile: OccupiedEmissionProfile,
    scenarios: Sequence[EmissionScenario],
    *,
    threshold: float = 0.0,
) -> EmissionValidationMetrics:
    """Classify independent validation scenarios and return exact metrics.

    The caller controls scenario boundaries so temporally adjacent training
    and validation rows are never accidentally shuffled together.  Counts
    use the unbounded occupied discriminant; ``threshold`` is therefore
    independent of evidence-rate clipping and scaling.
    """

    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not scenarios:
        raise ValueError("at least one validation scenario is required")

    all_counts = ConfusionMatrix()
    metrics: list[ScenarioEmissionMetrics] = []
    names: set[str] = set()
    for scenario in scenarios:
        if not scenario.name:
            raise ValueError("scenario names must be non-empty")
        if scenario.name in names:
            raise ValueError(f"duplicate scenario name: {scenario.name}")
        names.add(scenario.name)
        if not scenario.features:
            raise ValueError(f"scenario {scenario.name!r} has no observations")
        scores = tuple(occupied_discriminant(profile, feature) for feature in scenario.features)
        counts = _confusion(scores, scenario.expected_occupied, threshold)
        all_counts = _add_confusion(all_counts, counts)
        metrics.append(
            ScenarioEmissionMetrics(
                name=scenario.name,
                expected_occupied=scenario.expected_occupied,
                confusion=counts,
                mean_discriminant=sum(scores) / len(scores),
                minimum_discriminant=min(scores),
                maximum_discriminant=max(scores),
            )
        )
    return EmissionValidationMetrics(
        threshold=threshold,
        confusion=all_counts,
        scenarios=tuple(metrics),
    )


def validate_persisted_profile(
    profile: OccupiedEmissionProfile,
    *,
    min_rows: int = 15,
    min_sensitivity: float = 0.70,
    min_specificity: float = 0.80,
    min_scenario_recall: float = 0.50,
) -> None:
    """Reject incomplete or internally inconsistent persisted profiles.

    Persistence is an untrusted boundary: accepting finite coefficients is
    insufficient when validation metadata can be missing or fabricated
    inconsistently.  This verifies the complete schema and the same minimum
    held-out guarantees used by the guided workflow.  It deliberately raises
    ``ValueError`` so callers can fall back to the empty-only estimator.
    """

    if min_rows < 1:
        raise ValueError("min_rows must be positive")
    if min(profile.empty_rows, profile.active_rows, profile.settled_rows) < min_rows:
        raise ValueError("profile has too few training observations")
    validation = profile.validation
    if validation is None or validation.threshold != 0.0:
        raise ValueError("profile requires held-out validation at the runtime threshold")

    expected = {"empty": False, "moving": True, "standing": True, "seated": True}
    scenarios = {scenario.name: scenario for scenario in validation.scenarios}
    if set(scenarios) != set(expected) or len(scenarios) != len(validation.scenarios):
        raise ValueError("profile requires exactly empty, moving, standing, and seated validation")

    total = ConfusionMatrix()
    for name, expected_occupied in expected.items():
        scenario = scenarios[name]
        if scenario.expected_occupied is not expected_occupied:
            raise ValueError(f"validation label mismatch for {name}")
        _validate_confusion(scenario.confusion)
        if scenario.confusion.samples < min_rows:
            raise ValueError(f"validation scenario {name} has too few observations")
        if not all(
            math.isfinite(value)
            for value in (
                scenario.mean_discriminant,
                scenario.minimum_discriminant,
                scenario.maximum_discriminant,
            )
        ):
            raise ValueError(f"validation scenario {name} has non-finite metrics")
        if (
            not scenario.minimum_discriminant
            <= scenario.mean_discriminant
            <= scenario.maximum_discriminant
        ):
            raise ValueError(f"validation scenario {name} has inconsistent score bounds")
        if expected_occupied:
            if scenario.confusion.false_positive or scenario.confusion.true_negative:
                raise ValueError(f"occupied scenario {name} contains empty labels")
            if (scenario.confusion.sensitivity or 0.0) < min_scenario_recall:
                raise ValueError(f"occupied scenario {name} failed recall")
        elif scenario.confusion.true_positive or scenario.confusion.false_negative:
            raise ValueError("empty scenario contains occupied labels")
        total = _add_confusion(total, scenario.confusion)

    _validate_confusion(validation.confusion)
    if validation.confusion != total:
        raise ValueError("aggregate confusion matrix does not match scenarios")
    if (validation.confusion.sensitivity or 0.0) < min_sensitivity:
        raise ValueError("profile sensitivity is below the acceptance threshold")
    if (validation.confusion.specificity or 0.0) < min_specificity:
        raise ValueError("profile specificity is below the acceptance threshold")
    if validation.empty_mean_rate is None or not math.isfinite(validation.empty_mean_rate):
        raise ValueError("profile requires a finite empty mean drive")
    if validation.empty_mean_rate >= 0.0:
        raise ValueError("profile empty mean drive must be negative")
    if set(validation.occupied_mean_rates) != {"moving", "standing", "seated"}:
        raise ValueError("profile requires all occupied mean drives")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in validation.occupied_mean_rates.values()
    ):
        raise ValueError("profile occupied mean drives must be finite and positive")


def _validate_confusion(confusion: ConfusionMatrix) -> None:
    if any(
        value < 0
        for value in (
            confusion.true_positive,
            confusion.false_positive,
            confusion.true_negative,
            confusion.false_negative,
        )
    ):
        raise ValueError("confusion counts cannot be negative")


def _checked_features(name: str, rows: Sequence[Feature]) -> tuple[Feature, ...]:
    if not rows:
        raise ValueError(f"{name} calibration requires at least one observation")
    return tuple(_checked_feature(row) for row in rows)


def _checked_feature(feature: Feature) -> Feature:
    if len(feature) != 2:
        raise ValueError("emission features must contain exactly move and still scores")
    move, still = feature
    if not math.isfinite(move) or not math.isfinite(still):
        raise ValueError("emission features must be finite")
    return float(move), float(still)


def _mean(rows: Sequence[Feature]) -> Feature:
    n = len(rows)
    return sum(row[0] for row in rows) / n, sum(row[1] for row in rows) / n


def _population_covariance(rows: Sequence[Feature], mean: Feature) -> tuple[float, float, float]:
    n = len(rows)
    c00 = sum((row[0] - mean[0]) ** 2 for row in rows) / n
    c01 = sum((row[0] - mean[0]) * (row[1] - mean[1]) for row in rows) / n
    c11 = sum((row[1] - mean[1]) ** 2 for row in rows) / n
    return c00, c01, c11


def _discriminant(
    empty_mean: Feature,
    occupied_mean: Feature,
    inverse: tuple[float, float, float],
) -> LinearDiscriminant:
    i00, i01, i11 = inverse
    d0 = occupied_mean[0] - empty_mean[0]
    d1 = occupied_mean[1] - empty_mean[1]
    w0 = i00 * d0 + i01 * d1
    w1 = i01 * d0 + i11 * d1
    intercept = -0.5 * (
        (occupied_mean[0] + empty_mean[0]) * w0 + (occupied_mean[1] + empty_mean[1]) * w1
    )
    if not all(map(math.isfinite, (w0, w1, intercept))):
        raise ValueError("calibration produced a non-finite discriminant")
    return LinearDiscriminant(w0, w1, intercept)


def _confusion(
    scores: Sequence[float], expected_occupied: bool, threshold: float
) -> ConfusionMatrix:
    predicted_occupied = sum(score >= threshold for score in scores)
    predicted_empty = len(scores) - predicted_occupied
    if expected_occupied:
        return ConfusionMatrix(
            true_positive=predicted_occupied,
            false_negative=predicted_empty,
        )
    return ConfusionMatrix(
        false_positive=predicted_occupied,
        true_negative=predicted_empty,
    )


def _add_confusion(left: ConfusionMatrix, right: ConfusionMatrix) -> ConfusionMatrix:
    return ConfusionMatrix(
        true_positive=left.true_positive + right.true_positive,
        false_positive=left.false_positive + right.false_positive,
        true_negative=left.true_negative + right.true_negative,
        false_negative=left.false_negative + right.false_negative,
    )
