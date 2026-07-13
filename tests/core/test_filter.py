"""Rule 4: the occupancy filter, plus the DECISION.md motivating cases."""

from __future__ import annotations

import math

import pytest

from custom_components.presence_conductor.core import timers

from .harness import DESK, KONTOR, SOFA, SOFAKROK, Harness, make_config, make_snapshot, quiet


class TestRule41LogOddsUpdate:
    def test_rule_4_1_decay_relaxes_toward_the_prior(self) -> None:
        # k_bias = 0 and z_neg_cap = 0 isolate the pure relaxation of 4.1
        # (a quiet channel's centered score is negative, 3.2).
        config = make_config(k_bias=0.0, z_neg_cap=0.0)
        h = Harness(config, make_snapshot(config))
        h.occupy(KONTOR)  # lambda -> lam_attack (4.2)
        h.submit(quiet(KONTOR))  # zero the stored evidence
        h.run(20)
        expected = h.engine.lam_prior + (h.engine.lam_attack - h.engine.lam_prior) * math.exp(
            -20 / 90
        )
        assert h.zone(DESK).lam == pytest.approx(expected, rel=1e-6)

    def test_rule_4_1_no_fixed_timeout_evidence_holds_indefinitely(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        # Real still evidence beats the decay pull: occupancy holds for
        # 5 minutes with no fixed timeout anywhere.
        h.sustain(SOFAKROK, 300, still_d=100, still_e=15, still=True)
        assert h.zone(SOFA).occupied


class TestRule42FastAttack:
    def test_rule_4_2_confirmed_attack_flips_without_a_tick(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=100, move_e=25, moving=True)  # candidate
        desk = h.zone(DESK)
        assert not desk.occupied  # 4.2: one frame is not an attack
        assert desk.motion  # 4.4 rides the first frame
        h.send_frame(KONTOR, move_d=100, move_e=25, moving=True, at=0.3)
        assert desk.occupied  # confirmed: still before any tick
        assert desk.lam >= h.engine.lam_attack
        assert h.state.anyone_home is True  # 6.5: rises immediately

    def test_rule_4_2_gap_bounds_the_confirmation(self) -> None:
        # Zero evidence weights make the belief inert, isolating the attack
        # path (it keys on the raw candidate condition, not the weights).
        config = make_config(k_move=0.0, k_still=0.0, k_bias=0.0)
        h = Harness(config, make_snapshot(config))
        h.send_frame(KONTOR, move_d=100, move_e=25, moving=True)
        # Too close: radar has not produced a fresh reading yet.
        h.send_frame(KONTOR, move_d=100, move_e=25, moving=True, at=0.1)
        assert not h.zone(DESK).occupied
        # Too late: the candidate expired and this frame re-arms instead.
        h.send_frame(KONTOR, move_d=100, move_e=25, moving=True, at=5.0)
        assert not h.zone(DESK).occupied
        # A non-qualifying frame clears the candidate outright.
        h.send_frame(KONTOR, move_d=100, move_e=5, moving=False, at=5.2)
        h.send_frame(KONTOR, move_d=100, move_e=25, moving=True, at=5.4)
        assert not h.zone(DESK).occupied  # 5.4 is a fresh candidate
        # And a well-spaced confirming frame fires it.
        h.send_frame(KONTOR, move_d=100, move_e=25, moving=True, at=5.8)
        assert h.zone(DESK).occupied

    def test_rule_4_2_attack_confirm_1_restores_single_frame(self) -> None:
        config = make_config(attack_confirm=1)
        h = Harness(config, make_snapshot(config))
        h.send_frame(KONTOR, move_d=100, move_e=25, moving=True)
        assert h.zone(DESK).occupied
        assert h.zone(DESK).lam == pytest.approx(h.engine.lam_attack)

    def test_rule_4_2_attack_is_a_floor_never_lowers(self) -> None:
        h = Harness()
        h.occupy(KONTOR)
        h.sustain(KONTOR, 3, move_d=100, move_e=35, still_d=100, still_e=35, moving=True)
        assert h.zone(DESK).lam == pytest.approx(h.engine.lam_max)  # 4.5 clamp
        h.occupy(KONTOR)
        assert h.zone(DESK).lam == pytest.approx(h.engine.lam_max)

    def test_rule_4_2_requires_a_gated_frame(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=400, move_e=35, moving=True)
        assert not h.zone(DESK).occupied
        assert h.zone(DESK).lam == pytest.approx(h.engine.lam_prior)

    def test_rule_4_2_below_z_attack_waits_for_ticks(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=100, move_e=19, moving=True)  # z = 2.8 < 3
        assert not h.zone(DESK).occupied
        assert h.zone(DESK).lam == pytest.approx(h.engine.lam_prior)


class TestRule43Hysteresis:
    def test_rule_4_3_binary_holds_between_thresholds(self) -> None:
        # z_neg_cap = 0 keeps the downward pull at the k_bias rate, so the
        # crossing timings below stay coarse.
        config = make_config(z_neg_cap=0.0)
        h = Harness(config, make_snapshot(config))
        h.occupy(KONTOR)  # lambda -> lam_attack: 2.944
        h.submit(quiet(KONTOR))
        h.run(5)  # absence + decay pull lambda under theta_on
        desk = h.zone(DESK)
        assert h.engine.lam_off < desk.lam < h.engine.lam_on
        assert desk.occupied  # holds between thresholds
        h.run(7)
        assert desk.lam <= h.engine.lam_off
        assert not desk.occupied  # off at theta_off


class TestRule44Motion:
    def test_rule_4_4_gated_z_motion_turns_motion_on(self) -> None:
        # A live flag is not needed: within distance_hold of the last
        # flag-on frame (2.7), the centered move score alone drives motion.
        h = Harness()
        h.send_frame(KONTOR, move_d=100, move_e=5, moving=True)  # flag epoch
        h.run(6)  # motion hold expires at t=5
        assert not h.zone(DESK).motion
        h.send_frame(KONTOR, move_d=100, move_e=13, at=6.0)  # z = 2.06 >= z_motion
        assert h.zone(DESK).motion
        assert not h.zone(DESK).occupied  # below z_attack

    def test_rule_4_4_flag_with_gated_distance_turns_motion_on(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=100, move_e=5, moving=True)  # z = 0, flag set
        assert h.zone(DESK).motion

    def test_rule_4_4_no_motion_without_gate(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=400, move_e=35, moving=True)
        assert not h.zone(DESK).motion

    def test_rule_4_4_motion_off_after_hold(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=100, move_e=13, moving=True)
        h.submit(quiet(KONTOR))
        assert timers.motion_off(DESK) in h.deadlines
        h.run(6)  # hold expires at t=5
        assert not h.zone(DESK).motion

    def test_rule_4_4_new_evidence_restarts_the_hold(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=100, move_e=13, moving=True)
        h.run(3)
        h.send_frame(KONTOR, move_d=100, move_e=13, moving=True)  # restart at t=3
        h.submit(quiet(KONTOR))
        h.run(3)  # t=6: original deadline passed, restarted one has not
        assert h.zone(DESK).motion
        h.run(3)  # t=9 > 8
        assert not h.zone(DESK).motion


class TestRule45Clamp:
    def test_rule_4_5_lambda_clamped_at_both_ends(self) -> None:
        h = Harness()
        h.sustain(KONTOR, 5, move_d=100, move_e=35, still_d=100, still_e=35, moving=True)
        assert h.zone(DESK).lam == pytest.approx(h.engine.lam_max)
        h.sustain_quiet(KONTOR, 60)
        assert h.zone(DESK).lam == pytest.approx(h.engine.lam_min)


class TestDecisionCases:
    """The two failure modes from docs/DECISION.md the engine must beat."""

    def test_dropout_bridged_by_sub_threshold_energy(self) -> None:
        """Rule 3.5: a still person under the radar's gate threshold keeps
        `occupied` through a 15 s dropout (the sofakrok TV-evening case)."""
        h = Harness()
        strong = {"move_d": 100, "move_e": 35, "still_d": 100, "still_e": 35, "moving": True}
        h.sustain(SOFAKROK, 5, still=True, **strong)
        assert h.zone(SOFA).occupied
        for _ in range(15):
            # Radar verdict drops (has_still_target False) but the energy
            # margin stays visible above the noise floor (z_still = 2).
            h.send_frame(SOFAKROK, still_d=100, still_e=15, move_e=2)
            h.step_to(h.now + 1.0)
            assert h.zone(SOFA).occupied  # bridged every second of the gap
        assert h.pass_bys() == []  # no spurious exit either

    def test_ghost_blip_never_reaches_theta_on(self) -> None:
        """A 5 s modest energy blip from baseline stays under theta_on."""
        h = Harness()
        for _ in range(5):
            h.send_frame(SOFAKROK, still_d=100, still_e=12)  # z = 1.4
            h.step_to(h.now + 1.0)
            assert not h.zone(SOFA).occupied
            assert not h.zone(SOFA).motion  # 1.4 < z_motion too
        h.sustain_quiet(SOFAKROK, 20)  # blip ends: back to baseline
        assert not h.zone(SOFA).occupied
        assert h.room("stue").occupied is False
        assert h.state.anyone_home is False
