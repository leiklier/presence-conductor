"""Learned occupied-emission fitting, scoring and held-out validation."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from custom_components.presence_conductor.core.emissions import (
    discriminant_score,
    emission_fingerprint,
    fit_occupied_profile,
    learned_evidence_rate,
    occupied_discriminant,
    validate_profile,
)
from custom_components.presence_conductor.core.model import (
    EmissionScenario,
    LinearDiscriminant,
    OccupiedEmissionProfile,
)

EMPTY = ((-1.2, -1.0), (-0.8, -1.1), (-1.0, -0.9), (-1.1, -1.2))
ACTIVE = ((3.8, 0.8), (4.2, 1.2), (4.0, 1.0), (4.1, 0.9))
SETTLED = ((-0.2, 3.8), (0.2, 4.2), (0.0, 4.0), (0.1, 4.1))


def profile(**kwargs) -> OccupiedEmissionProfile:
    return fit_occupied_profile(EMPTY, ACTIVE, SETTLED, **kwargs)


class TestRegularizedLdaFit:
    def test_both_occupied_modes_separate_from_empty(self) -> None:
        fitted = profile()

        assert occupied_discriminant(fitted, (-1.0, -1.0)) < 0.0
        assert occupied_discriminant(fitted, (4.0, 1.0)) > 0.0
        assert occupied_discriminant(fitted, (0.0, 4.0)) > 0.0
        assert fitted.empty_rows == len(EMPTY)
        assert fitted.active_rows == len(ACTIVE)
        assert fitted.settled_rows == len(SETTLED)

    def test_recording_duration_does_not_set_class_weight(self) -> None:
        fitted = profile()
        repeated_active = fit_occupied_profile(EMPTY, ACTIVE * 5, SETTLED)

        assert repeated_active.active == pytest.approx(fitted.active)
        assert repeated_active.settled == pytest.approx(fitted.settled)
        assert repeated_active.active_weight == 0.5

    def test_constant_and_collinear_captures_are_regularized(self) -> None:
        fitted = fit_occupied_profile(
            empty=((0.0, 0.0),),
            active=((2.0, 2.0),),
            settled=((1.0, 1.0),),
            shrinkage=0.0,
            variance_floor=0.001,
        )

        fields = (
            fitted.active.move_weight,
            fitted.active.still_weight,
            fitted.active.intercept,
            fitted.settled.move_weight,
            fitted.settled.still_weight,
            fitted.settled.intercept,
        )
        assert all(map(math.isfinite, fields))
        assert occupied_discriminant(fitted, (2.0, 2.0)) > 0.0

    def test_shared_covariance_accounts_for_feature_correlation(self) -> None:
        # All classes vary mostly along move==still.  The discriminative
        # signal is perpendicular to that noisy direction.
        empty = ((-2.0, -2.0), (-1.0, -1.0), (1.0, 1.0), (2.0, 2.0))
        active = tuple((move + 1.0, still - 1.0) for move, still in empty)
        settled = tuple((move - 1.0, still + 1.0) for move, still in empty)
        fitted = fit_occupied_profile(empty, active, settled, shrinkage=0.05)

        assert fitted.active.move_weight > 0.0
        assert fitted.active.still_weight < 0.0
        assert fitted.settled.move_weight < 0.0
        assert fitted.settled.still_weight > 0.0
        assert occupied_discriminant(fitted, (1.0, -1.0)) > 0.0
        assert occupied_discriminant(fitted, (-1.0, 1.0)) > 0.0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"shrinkage": -0.1}, "shrinkage"),
            ({"shrinkage": 1.1}, "shrinkage"),
            ({"variance_floor": 0.0}, "variance_floor"),
            ({"active_weight": 0.0}, "active_weight"),
            ({"active_weight": 1.0}, "active_weight"),
            ({"evidence_scale": 0.0}, "evidence_scale"),
            ({"evidence_min": 1.0, "evidence_max": 1.0}, "bounds"),
        ],
    )
    def test_invalid_fit_settings_are_rejected(
        self, kwargs: dict[str, float], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            profile(**kwargs)

    @pytest.mark.parametrize(
        "rows",
        [(), ((math.nan, 0.0),), ((0.0, math.inf),)],
    )
    def test_missing_or_nonfinite_classes_are_rejected(self, rows) -> None:
        with pytest.raises(ValueError):
            fit_occupied_profile(rows, ACTIVE, SETTLED)


class TestScoring:
    def test_identical_modes_have_no_log_two_bonus(self) -> None:
        mode = LinearDiscriminant(1.0, -0.5, 0.25)
        fitted = OccupiedEmissionProfile(active=mode, settled=mode)
        feature = (3.0, 2.0)

        assert occupied_discriminant(fitted, feature) == pytest.approx(
            discriminant_score(mode, feature)
        )

    def test_log_mixture_is_stable_for_extreme_separation(self) -> None:
        fitted = OccupiedEmissionProfile(
            active=LinearDiscriminant(1e200, 0.0, 0.0),
            settled=LinearDiscriminant(-1e200, 0.0, 0.0),
        )

        score = occupied_discriminant(fitted, (1.0, 0.0))
        assert math.isfinite(score)
        assert score == pytest.approx(1e200)

    def test_evidence_rate_is_scaled_and_bounded_both_directions(self) -> None:
        mode = LinearDiscriminant(2.0, 0.0, 0.0)
        fitted = OccupiedEmissionProfile(
            active=mode,
            settled=mode,
            evidence_scale=2.0,
            evidence_min=-1.5,
            evidence_max=2.5,
        )

        assert learned_evidence_rate(fitted, (0.25, 0.0)) == pytest.approx(1.0)
        assert learned_evidence_rate(fitted, (100.0, 0.0)) == 2.5
        assert learned_evidence_rate(fitted, (-100.0, 0.0)) == -1.5

    def test_invalid_persisted_profile_is_rejected_at_scoring_boundary(self) -> None:
        fitted = profile()

        with pytest.raises(ValueError, match="active_weight"):
            occupied_discriminant(replace(fitted, active_weight=1.0), (0.0, 0.0))
        with pytest.raises(ValueError, match="bounds"):
            learned_evidence_rate(replace(fitted, evidence_min=3.0), (0.0, 0.0))
        with pytest.raises(ValueError, match="finite"):
            occupied_discriminant(fitted, (math.nan, 0.0))


class TestValidation:
    def test_exact_aggregate_and_per_scenario_confusion(self) -> None:
        mode = LinearDiscriminant(1.0, 0.0, 0.0)
        fitted = OccupiedEmissionProfile(active=mode, settled=mode)
        scenarios = (
            EmissionScenario("empty", False, ((-2.0, 0.0), (1.0, 0.0))),
            EmissionScenario("active", True, ((2.0, 0.0), (-1.0, 0.0))),
            EmissionScenario("settled", True, ((3.0, 0.0), (4.0, 0.0))),
        )

        result = validate_profile(fitted, scenarios)

        assert result.threshold == 0.0
        assert result.confusion.true_positive == 3
        assert result.confusion.false_positive == 1
        assert result.confusion.true_negative == 1
        assert result.confusion.false_negative == 1
        assert result.confusion.samples == 6
        assert result.confusion.sensitivity == pytest.approx(0.75)
        assert result.confusion.specificity == pytest.approx(0.5)
        assert result.confusion.balanced_accuracy == pytest.approx(0.625)
        assert [metric.name for metric in result.scenarios] == ["empty", "active", "settled"]
        assert result.scenarios[0].confusion.false_positive == 1
        assert result.scenarios[0].mean_discriminant == pytest.approx(-0.5)
        assert result.scenarios[2].confusion.true_positive == 2
        assert result.scenarios[2].minimum_discriminant == pytest.approx(3.0)
        assert result.scenarios[2].maximum_discriminant == pytest.approx(4.0)

    def test_threshold_is_applied_to_unbounded_discriminant(self) -> None:
        mode = LinearDiscriminant(1.0, 0.0, 0.0)
        fitted = OccupiedEmissionProfile(
            active=mode,
            settled=mode,
            evidence_scale=100.0,
            evidence_min=-0.1,
            evidence_max=0.1,
        )
        scenario = EmissionScenario("active", True, ((0.5, 0.0), (2.0, 0.0)))

        result = validate_profile(fitted, (scenario,), threshold=1.0)

        assert result.confusion.true_positive == 1
        assert result.confusion.false_negative == 1

    @pytest.mark.parametrize(
        "scenarios",
        [
            (),
            (EmissionScenario("", False, ((0.0, 0.0),)),),
            (EmissionScenario("empty", False, ()),),
            (
                EmissionScenario("same", False, ((0.0, 0.0),)),
                EmissionScenario("same", True, ((1.0, 0.0),)),
            ),
        ],
    )
    def test_invalid_scenario_sets_are_rejected(self, scenarios) -> None:
        with pytest.raises(ValueError):
            validate_profile(profile(), scenarios)


class TestFingerprint:
    def test_is_deterministic_and_context_sensitive(self) -> None:
        kwargs = {
            "path": "move_gate",
            "gate_indices": (1, 2, 3),
            "floor_fingerprint": "floor|v1",
            "move_stat_fingerprint": "move|v1",
            "still_stat_fingerprint": "still|v1",
        }
        first = emission_fingerprint(**kwargs)

        assert emission_fingerprint(**kwargs) == first
        assert emission_fingerprint(**{**kwargs, "gate_indices": (1, 2)}) != first
        assert emission_fingerprint(**{**kwargs, "path": "move_agg"}) != first
        assert emission_fingerprint(**{**kwargs, "feature_version": "z-v2"}) != first

    def test_aggregate_path_ignores_irrelevant_gate_family(self) -> None:
        kwargs = {
            "path": "move_agg",
            "floor_fingerprint": "floor",
            "move_stat_fingerprint": "move",
            "still_stat_fingerprint": "still",
        }

        assert emission_fingerprint(gate_indices=(1,), **kwargs) == emission_fingerprint(
            gate_indices=(7, 8), **kwargs
        )
