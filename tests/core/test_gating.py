"""Rules 1.4 and 2: unit hygiene and distance gating."""

from __future__ import annotations

import pytest

from custom_components.presence_conductor.core.model import ZoneConfig

from .harness import (
    BORD,
    DESK,
    DOOR,
    KONTOR,
    SOFA,
    SOFAKROK,
    SPISEBORD,
    Harness,
    make_config,
    make_snapshot,
)


class TestRule21ZoneMask:
    def test_rule_2_1_distance_inside_zone_gates_evidence(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=100, move_e=35, moving=True)
        assert h.zone(DESK).move_gated
        assert h.zone(DESK).z_move == pytest.approx(6.0)  # (0.35-0.05)/0.05, capped
        assert not h.zone(DOOR).move_gated
        assert h.zone(DOOR).z_move == 0.0

    def test_rule_2_1_margin_extends_the_interval(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=175, move_e=35, moving=True)
        assert h.zone(DESK).move_gated  # 150 + margin 30 = 180
        h.send_frame(KONTOR, move_d=185, move_e=35, moving=True)
        assert not h.zone(DESK).move_gated
        assert not h.zone(DOOR).move_gated  # 220 - margin 30 = 190

    def test_rule_2_1_distance_outside_every_zone_contributes_nothing(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=400, move_e=35, moving=True)
        assert not h.zone(DESK).move_gated
        assert not h.zone(DOOR).move_gated
        assert not h.zone(DESK).occupied  # no fast attack either (4.2)
        lam_before = h.zone(DESK).lam
        h.tick()
        assert h.zone(DESK).lam < lam_before  # absence evidence applies (3.2)

    def test_rule_2_1_still_distance_gates_the_still_channel(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, still_d=100, still_e=35, still=True)
        assert h.zone(DESK).still_gated
        assert h.zone(DESK).z_still == pytest.approx(6.0)
        assert not h.zone(DESK).move_gated


class TestRule22SameRoomSeparation:
    def test_rule_2_2_evidence_at_one_sensor_never_moves_the_other_zone(self) -> None:
        h = Harness()
        h.occupy(SOFAKROK)
        assert h.zone(SOFA).occupied
        assert h.zone(BORD).lam == pytest.approx(h.engine.lam_prior)
        assert not h.zone(BORD).occupied
        assert h.room("stue").occupied is True  # 6.1: either sensor suffices

    def test_rule_2_2_zones_of_one_sensor_are_separated_by_the_mask(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=250, move_e=35, moving=True)
        assert h.zone(DOOR).move_gated
        assert not h.zone(DESK).move_gated
        assert h.zone(DOOR).occupied  # 4.2
        assert not h.zone(DESK).occupied


class TestRule23FallbackAttribution:
    def test_rule_2_3_flag_without_distance_goes_to_fallback_zone(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=None, move_e=35, moving=True)
        assert h.zone(DESK).move_gated  # desk is flagged fallback
        assert h.zone(DESK).occupied  # 4.2 rides on the attributed evidence
        assert not h.zone(DOOR).move_gated

    def test_rule_2_3_nearest_zone_when_no_fallback_flag(self) -> None:
        zones = (
            ZoneConfig(DESK, "Desk", KONTOR, room_id="kontor", near_cm=30, far_cm=150),
            ZoneConfig(DOOR, "Door", KONTOR, room_id="kontor", near_cm=220, far_cm=300),
            ZoneConfig(SOFA, "Sofakrok", SOFAKROK, room_id="stue", near_cm=30, far_cm=200),
            ZoneConfig(BORD, "Spisebord", SPISEBORD, room_id="stue", near_cm=50, far_cm=250),
        )
        config = make_config(zones=zones)
        h = Harness(config, make_snapshot(config))
        h.send_frame(KONTOR, still_d=None, still_e=35, still=True)
        assert h.zone(DESK).still_gated  # nearest: near_cm 30 < 220
        assert not h.zone(DOOR).still_gated

    def test_rule_2_3_no_flag_and_no_distance_contributes_nothing(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=None, move_e=35, moving=False)
        assert not h.zone(DESK).move_gated
        assert h.zone(DESK).z_move == 0.0


class TestRule14UnitHygiene:
    def test_rule_1_4_energies_clamped_then_normalized(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=100, move_e=150)  # clamped to 100 -> 1.0
        assert h.zone(DESK).z_move == pytest.approx(6.0)  # capped at z_cap (3.2)

    def test_rule_1_4_negative_energy_clamps_to_zero(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, move_d=100, move_e=-5)
        assert h.zone(DESK).z_move == 0.0

    def test_rule_1_4_negative_distance_clamps_to_zero(self) -> None:
        h = Harness()
        h.send_frame(KONTOR, still_d=-5, still_e=35)
        # clamped to 0 cm, inside desk's masked interval [30-30, 150+30]
        assert h.zone(DESK).still_gated
