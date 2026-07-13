"""Rule 5: activity classification and pass-by."""

from __future__ import annotations

import pytest

from custom_components.presence_conductor.core.model import Activity

from .harness import SOFA, SOFAKROK, Harness, make_config, make_snapshot

STRONG_MOVE = {"move_d": 100.0, "move_e": 35.0, "moving": True}
STRONG_STILL = {"still_d": 100.0, "still_e": 35.0, "still": True, "move_e": 5.0}


class TestRule51States:
    def test_rule_5_1_entry_starts_passing(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        zone = h.zone(SOFA)
        assert zone.occupied
        assert zone.activity is Activity.PASSING
        assert zone.dwell_seconds == 0.0

    def test_rule_5_1_passing_becomes_active_at_t_dwell(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        h.sustain(SOFAKROK, 44, **STRONG_MOVE)
        assert h.zone(SOFA).activity is Activity.PASSING
        h.sustain(SOFAKROK, 2, **STRONG_MOVE)
        assert h.zone(SOFA).activity is Activity.ACTIVE

    def test_rule_5_1_still_takeover_settles_from_passing(self) -> None:
        """Walking in and sitting down settles before t_dwell ever elapses."""
        h = Harness()
        h.occupy(SOFAKROK)
        h.sustain(SOFAKROK, 29, **STRONG_STILL)
        assert h.zone(SOFA).activity is Activity.PASSING  # t_settle not yet
        h.sustain(SOFAKROK, 3, **STRONG_STILL)
        # Settled with dwell < t_dwell: PASSING -> SETTLED without ACTIVE.
        assert h.zone(SOFA).activity is Activity.SETTLED
        assert h.zone(SOFA).dwell_seconds < 45.0

    def test_rule_5_1_active_to_settled_and_back_with_t_settle_smoothing(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        h.sustain(SOFAKROK, 46, **STRONG_MOVE)
        assert h.zone(SOFA).activity is Activity.ACTIVE
        h.sustain(SOFAKROK, 29, **STRONG_STILL)
        assert h.zone(SOFA).activity is Activity.ACTIVE  # smoothing holds
        h.sustain(SOFAKROK, 3, **STRONG_STILL)
        assert h.zone(SOFA).activity is Activity.SETTLED
        h.sustain(SOFAKROK, 29, **STRONG_MOVE)
        assert h.zone(SOFA).activity is Activity.SETTLED  # smoothing again
        h.sustain(SOFAKROK, 3, **STRONG_MOVE)
        assert h.zone(SOFA).activity is Activity.ACTIVE


class TestRule51DominanceContinuity:
    def test_rule_5_1_single_still_impulse_never_matures(self) -> None:
        """A single still-dominant frame followed by quiet evidence must not
        promote to SETTLED t_settle later: dominance is continuous, and
        quiet/equal evidence resets both clocks (5.1)."""
        # z_neg_cap = 0 and k_bias = 0 keep the quiet zone occupied (the
        # belief holds near lam_attack), isolating the dominance logic.
        config = make_config(z_neg_cap=0.0, k_bias=0.0, tau_decay=3600.0)
        h = Harness(config, make_snapshot(config))
        h.occupy(SOFAKROK)
        h.send_frame(SOFAKROK, **STRONG_STILL)  # one still-dominant frame
        h.tick()  # dominance clocks advance with time (4.1)
        assert h.zone(SOFA).still_dominant_since is not None
        h.sustain_quiet(SOFAKROK, 40)  # > t_settle of quiet evidence
        zone = h.zone(SOFA)
        assert zone.occupied  # belief held; only dominance is under test
        assert zone.still_dominant_since is None  # 5.1: clock reset
        assert zone.activity is not Activity.SETTLED

    def test_rule_5_1_continuous_still_dominance_settles(self) -> None:
        """The contrast: the same window with the still channel genuinely
        dominant throughout does settle."""
        config = make_config(z_neg_cap=0.0, k_bias=0.0, tau_decay=3600.0)
        h = Harness(config, make_snapshot(config))
        h.occupy(SOFAKROK)
        h.sustain(SOFAKROK, 35, **STRONG_STILL)
        assert h.zone(SOFA).activity is Activity.SETTLED


class TestRule52PassBy:
    def test_rule_5_2_walkthrough_emits_pass_by_on_exit(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        h.sustain(SOFAKROK, 5, **STRONG_MOVE)
        h.sustain_quiet(SOFAKROK, 25)  # person is gone; posterior drains
        assert not h.zone(SOFA).occupied
        assert h.zone(SOFA).activity is Activity.EMPTY
        events = h.pass_bys()
        assert len(events) == 1
        assert events[0].zone_id == SOFA
        assert events[0].peak_confidence == pytest.approx(0.999, abs=0.002)
        assert 10 < events[0].duration < 45  # on -> off traversal time

    def test_rule_5_2_no_pass_by_after_dwelling(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        h.sustain(SOFAKROK, 50, **STRONG_MOVE)  # past t_dwell: ACTIVE
        h.sustain_quiet(SOFAKROK, 25)
        assert not h.zone(SOFA).occupied
        assert h.pass_bys() == []  # 5.2: only EMPTY-from-PASSING emits

    def test_rule_5_2_no_pass_by_from_settled(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        h.sustain(SOFAKROK, 35, **STRONG_STILL)  # SETTLED via still takeover
        assert h.zone(SOFA).activity is Activity.SETTLED
        h.sustain_quiet(SOFAKROK, 25)
        assert not h.zone(SOFA).occupied
        assert h.pass_bys() == []


class TestRule53ConsumerContract:
    def test_rule_5_3_occupied_includes_passing(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        zone = h.zone(SOFA)
        assert zone.activity is Activity.PASSING
        assert zone.occupied  # a person in the zone is in the zone
        assert h.room("stue").occupied is True
        # Walk-through-averse consumers key on activity instead:
        assert zone.activity not in (Activity.ACTIVE, Activity.SETTLED)


class TestRule54Dwell:
    def test_rule_5_4_dwell_counts_continuous_occupancy(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        h.sustain(SOFAKROK, 10, **STRONG_MOVE)
        assert h.zone(SOFA).dwell_seconds == pytest.approx(10.0)

    def test_rule_5_4_dwell_resets_on_empty(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        h.sustain(SOFAKROK, 10, **STRONG_MOVE)
        h.sustain_quiet(SOFAKROK, 25)
        assert not h.zone(SOFA).occupied
        assert h.zone(SOFA).dwell_seconds == 0.0
