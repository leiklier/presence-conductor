"""Rules 2.4-2.6 and 3.6: gate ownership, max-aggregated gate evidence,
per-frame precedence over the aggregate path, and per-gate noise floors."""

from __future__ import annotations

import pytest

from custom_components.presence_conductor.core import gating, timers
from custom_components.presence_conductor.core.events import RecordBaseline
from custom_components.presence_conductor.core.model import (
    ConductorConfig,
    GateBaselines,
    SensorConfig,
    Tunables,
    ZoneConfig,
)

from .harness import (
    DESK,
    DOOR,
    KONTOR,
    MU,
    SIGMA,
    Harness,
    calibrated_gate_floors,
    centered_of,
    centered_of_raw,
    frame,
    gate_tuple,
    make_config,
    make_snapshot,
)


def make_harness(**tunable_overrides: float) -> Harness:
    """Default config with every gate calibrated like the aggregates, so the
    harness's raw-energy-to-z table applies to gate energies too. Gate
    evidence is experimental (2.6): these tests opt in."""
    config = make_config(use_gate_evidence=True, **tunable_overrides)
    return Harness(config, make_snapshot(config, gate_floors=calibrated_gate_floors(config)))


class TestRule24GateOwnership:
    def test_rule_2_4_ownership_overlaps_the_masked_interval(self) -> None:
        desk = ZoneConfig(DESK, "Desk", KONTOR, room_id="kontor", near_cm=30, far_cm=150)
        # Masked interval [0, 180]: gates [0,75), [75,150), [150,225).
        assert gating.owned_gates(desk, 75.0, 30.0) == (0, 1, 2)

    def test_rule_2_4_margin_overlap_shares_a_boundary_gate(self) -> None:
        door = ZoneConfig(DOOR, "Door", KONTOR, room_id="kontor", near_cm=220, far_cm=300)
        # Masked interval [190, 330]: gate 2 is shared with the desk zone
        # above - ownership is a mask, not a partition (2.4).
        assert gating.owned_gates(door, 75.0, 30.0) == (2, 3, 4)

    def test_rule_2_4_gate_size_20_shifts_ownership(self) -> None:
        near = ZoneConfig("z", "Z", KONTOR, room_id="r", near_cm=30, far_cm=150)
        assert gating.owned_gates(near, 20.0, 30.0) == tuple(range(9))  # [0, 180] covers all
        far = ZoneConfig("z", "Z", KONTOR, room_id="r", near_cm=200, far_cm=400)
        # Only gate 8 ([160, 180)) reaches past the masked near edge (170).
        assert gating.owned_gates(far, 20.0, 30.0) == (8,)

    def test_rule_2_4_zone_beyond_the_last_gate_owns_nothing(self) -> None:
        beyond = ZoneConfig("z", "Z", KONTOR, room_id="r", near_cm=500, far_cm=600)
        assert gating.owned_gates(beyond, 20.0, 30.0) == ()

    def test_rule_2_4_gate_size_is_per_sensor_config(self) -> None:
        config = ConductorConfig(
            sensors=(SensorConfig(KONTOR, "Kontor", gate_size_cm=20.0),),
            zones=(ZoneConfig(DESK, "Desk", KONTOR, room_id="kontor", near_cm=30, far_cm=150),),
        )
        h = Harness(config, make_snapshot(config))
        assert h.engine.owned_gates[DESK] == tuple(range(9))


class TestRule25GateEvidence:
    def test_rule_2_5_zone_gate_evidence_is_the_max_not_the_sum(self) -> None:
        h = make_harness()
        h.send_frame(KONTOR, gate_move=gate_tuple({1: 10, 2: 12.5}))
        # Raw S = max(1.0, 1.5), not 2.5; centered for the desk's three
        # owned gates (3.7 analytic fallback).
        assert h.zone(DESK).z_move == pytest.approx(centered_of(1.5, m=3))

    def test_rule_2_5_each_gate_scores_against_its_own_floor(self) -> None:
        config = make_config(use_gate_evidence=True)
        floors = calibrated_gate_floors(config)
        floors[DOOR] = {**floors[DOOR], 4: GateBaselines(0.40, 0.05, 0.40, 0.05)}
        h = Harness(config, make_snapshot(config, gate_floors=floors))
        h.send_frame(KONTOR, gate_move=gate_tuple({4: 40}))
        # At gate 4's own elevated floor (3.6): raw S = 0 centers negative.
        assert h.zone(DOOR).z_move == pytest.approx(centered_of(0.0, m=3))
        h.send_frame(KONTOR, gate_move=gate_tuple({3: 40}))
        assert h.zone(DOOR).z_move == pytest.approx(centered_of(7.0, m=3))  # sibling

    def test_rule_2_5_two_people_in_two_zones_of_one_sensor(self) -> None:
        h = make_harness()
        # One frame carries a mover at gate 1 (desk) and one at gate 4
        # (door): both zones are credited at once - impossible in the
        # single-distance model (2.1), which reports one distance per kind.
        h.send_frame(KONTOR, gate_move=gate_tuple({1: 35, 4: 35}))
        h.send_frame(KONTOR, gate_move=gate_tuple({1: 36, 4: 36}), at=h.now + 0.3)
        assert h.zone(DESK).occupied  # 4.2 (confirmed) on the gate move score
        assert h.zone(DOOR).occupied

    def test_rule_2_5_uncalibrated_gates_use_the_tunables_defaults(self) -> None:
        config = make_config(use_gate_evidence=True)
        h = Harness(config, make_snapshot(config))  # no persisted gate floors
        h.send_frame(KONTOR, gate_move=gate_tuple({1: 15}, fill=10.0))
        # default_mu 0.10, default_sigma 0.10: raw 10 -> 0, raw 15 -> S 0.5.
        assert h.zone(DESK).z_move == pytest.approx(centered_of(0.5, m=3))


class TestRule26GatePrecedence:
    def test_rule_2_6_gate_data_replaces_the_aggregate_path(self) -> None:
        h = make_harness()
        # The aggregate path alone would saturate (raw 35 at a gated
        # distance); the frame's gates say the zone is quiet, and they win.
        h.send_frame(KONTOR, move_d=100, move_e=35, moving=True, gate_move=gate_tuple())
        desk = h.zone(DESK)
        assert desk.z_move == pytest.approx(centered_of(0.0, m=3))  # quiet gates
        assert not desk.move_gated
        assert not desk.occupied  # no fast attack either (4.2)
        lam_before = desk.lam
        h.tick()
        assert desk.lam < lam_before  # absence evidence applies (3.2)

    def test_rule_2_6_fallback_is_per_frame_with_no_latch(self) -> None:
        h = make_harness()
        h.send_frame(KONTOR, move_d=100, move_e=35, moving=True, gate_move=gate_tuple())
        assert not h.zone(DESK).occupied  # quiet gates win this frame
        # Engineering mode drops: the very next frame has no gate data and
        # the aggregate path applies unchanged - no mode latch.
        h.send_frame(KONTOR, move_d=100, move_e=35, moving=True)
        assert h.zone(DESK).z_move == pytest.approx(centered_of_raw(35))
        h.send_frame(KONTOR, move_d=100, move_e=36, moving=True, at=h.now + 0.3)
        assert h.zone(DESK).occupied  # 4.2 (confirmed, fresh)

    def test_rule_2_6_all_owned_gates_unknown_falls_back_per_zone(self) -> None:
        h = make_harness()
        # Desk's owned gates (0-2) are all unknown -> aggregate path; door
        # still has known gates (3, 4) -> gate path. One frame, two paths.
        h.send_frame(
            KONTOR,
            move_d=100,
            move_e=35,
            moving=True,
            gate_move=gate_tuple({0: None, 1: None, 2: None, 4: 35.0}),
        )
        desk, door = h.zone(DESK), h.zone(DOOR)
        assert not desk.move_from_gates
        assert desk.z_move == pytest.approx(centered_of_raw(35))  # aggregate
        assert door.move_from_gates
        assert door.z_move == pytest.approx(centered_of(6.0, m=3))  # gate 4 (2.5)

    def test_rule_2_6_channels_fall_back_independently(self) -> None:
        h = make_harness()
        h.send_frame(KONTOR, still_d=100, still_e=35, still=True, gate_move=gate_tuple())
        desk = h.zone(DESK)
        assert desk.z_move == pytest.approx(centered_of(0.0, m=3))  # quiet gates
        assert desk.z_still == pytest.approx(centered_of_raw(35))  # aggregate

    def test_rule_2_6_zone_without_owned_gates_stays_on_the_aggregate_path(self) -> None:
        config = ConductorConfig(
            sensors=(SensorConfig(KONTOR, "Kontor", gate_size_cm=20.0),),
            zones=(ZoneConfig("far_zone", "Far", KONTOR, room_id="r", near_cm=500, far_cm=600),),
            tunables=Tunables(use_gate_evidence=True),
        )
        h = Harness(config, make_snapshot(config))
        assert h.engine.owned_gates["far_zone"] == ()  # 2.4
        h.send_frame(KONTOR, move_d=550, move_e=35, moving=True, gate_move=gate_tuple({1: 35}))
        assert h.zone("far_zone").z_move == pytest.approx(centered_of_raw(35))  # 2.6

    def test_rule_2_6_fast_attack_rides_the_gate_move_z(self) -> None:
        h = make_harness()
        h.send_frame(KONTOR, gate_move=gate_tuple({1: 35}))  # raw 6 >= tail threshold
        h.send_frame(KONTOR, gate_move=gate_tuple({1: 36}), at=h.now + 0.3)  # fresh
        assert h.zone(DESK).occupied  # 4.2: confirmed, no tick needed

    def test_rule_2_6_motion_keys_on_gate_z_not_the_global_flag(self) -> None:
        h = make_harness()
        # has_moving_target with a gated distance but quiet owned gates: the
        # sensor-global flag is not zone evidence under gate precedence (4.4).
        h.send_frame(KONTOR, move_d=100, moving=True, gate_move=gate_tuple())
        assert not h.zone(DESK).motion
        # An owned gate above z_motion turns motion on without any flag.
        h.send_frame(KONTOR, gate_move=gate_tuple({1: 20}))  # centered 3.1
        assert h.zone(DESK).motion

    def test_rule_2_6_startup_adoption_is_spatial_with_gates(self) -> None:
        config = make_config(use_gate_evidence=True)
        snapshot = make_snapshot(
            config,
            gate_floors=calibrated_gate_floors(config),
            frames={KONTOR: frame(KONTOR, moving=True, move_d=100, gate_move=gate_tuple({4: 35}))},
        )
        h = Harness(config, snapshot)
        # 7.1: the mover is plainly at gate 4, so the door adopts occupancy;
        # the desk does not, although the aggregate distance points at it.
        assert h.zone(DOOR).occupied
        assert not h.zone(DESK).occupied


class TestRule36PerGateFloors:
    def test_rule_3_6_record_baseline_replaces_per_gate_floors(self) -> None:
        h = make_harness()
        h.submit(RecordBaseline(DESK, duration=80.0))
        assert h.deadlines[timers.baseline_end(DESK)] == pytest.approx(80.0)
        raw = [8, 9, 10, 11, 12] * 16
        raw[3] = 90  # a person walks through: brief violations don't poison
        for value in raw:
            h.send_frame(
                KONTOR,
                gate_move=gate_tuple({1: value, 2: None}),
                gate_still=gate_tuple({1: value, 2: None}),
            )
            h.step_to(h.now + 1.0)
        h.run(5)  # window closes at t=80 during this run
        desk = h.zone(DESK)
        assert desk.recording is None
        assert desk.gate_move_baselines[1].mu == pytest.approx(0.10, abs=0.011)
        # 3.1: UCB of the deviations (0.01 here — one outlier, not two) +
        # half quantum, times 1.4826.
        assert desk.gate_move_baselines[1].sigma == pytest.approx(1.4826 * 0.015, abs=0.002)
        assert desk.gate_still_baselines[1].mu == pytest.approx(0.10, abs=0.011)
        # Gate 2 reported nothing during the window: previous floor kept.
        assert desk.gate_move_baselines[2].mu == pytest.approx(MU)
        assert desk.gate_move_baselines[2].sigma == pytest.approx(SIGMA)
        assert h.persist_count == 1  # 3.3: baselines persist in the config entry

    def test_rule_3_6_adaptation_extends_per_gate_for_owned_gates(self) -> None:
        config = make_config(use_gate_evidence=True, t_background=60.0, tau_background=600.0)
        h = Harness(config, make_snapshot(config, gate_floors=calibrated_gate_floors(config)))
        h.run(61)  # posterior below p_background since start: now eligible
        for _ in range(100):
            h.send_frame(KONTOR, gate_move=gate_tuple(fill=0.0), gate_still=gate_tuple(fill=0.0))
            h.step_to(h.now + 1.0)
        desk = h.zone(DESK)
        for index in h.engine.owned_gates[DESK]:
            assert desk.gate_move_baselines[index].mu < MU - 0.005  # drifted toward 0
            assert desk.gate_move_baselines[index].sigma >= 0.02  # 3.1 floor holds
        # Gate 5 is not owned by the desk (2.4): its floor never adapts.
        assert desk.gate_move_baselines[5].mu == MU

    def test_rule_3_6_adaptation_freezes_the_moment_the_posterior_rises(self) -> None:
        config = make_config(use_gate_evidence=True, t_background=60.0, tau_background=600.0)
        h = Harness(config, make_snapshot(config, gate_floors=calibrated_gate_floors(config)))
        h.run(61)
        h.send_frame(KONTOR, gate_move=gate_tuple({1: 35}))  # 4.2 candidate
        h.send_frame(KONTOR, gate_move=gate_tuple({1: 36}), at=h.now + 0.3)  # confirmed
        desk = h.zone(DESK)
        assert desk.below_since is None  # clock reset immediately (3.4)
        mu_before = desk.gate_move_baselines[1].mu
        for _ in range(30):
            h.send_frame(KONTOR, gate_move=gate_tuple({1: 35}))
            h.step_to(h.now + 1.0)
        assert desk.gate_move_baselines[1].mu == mu_before  # never learn a person

    def test_rule_3_6_fan_at_one_gate_with_its_own_floor_causes_no_occupancy(self) -> None:
        """The headline improvement (3.6): a fan parked at door gate 4 whose
        elevated still floor was calibrated produces z = 0 at its own gate,
        so hours of fan energy never flip the zone."""
        config = make_config(use_gate_evidence=True)
        floors = calibrated_gate_floors(config)
        floors[DOOR] = {**floors[DOOR], 4: GateBaselines(MU, SIGMA, 0.40, 0.05)}
        h = Harness(config, make_snapshot(config, gate_floors=floors))
        for _ in range(120):  # two minutes of fan returns
            h.send_frame(
                KONTOR,
                still_d=250,
                still_e=40,
                still=True,
                gate_move=gate_tuple(),
                gate_still=gate_tuple({4: 40}),
            )
            h.step_to(h.now + 1.0)
        door = h.zone(DOOR)
        assert not door.occupied
        assert door.confidence < 0.05  # held at the empty prior

    def test_rule_3_6_same_fan_on_the_aggregate_path_would_false_positive(self) -> None:
        """The v0.1.0 contrast: identical energies without gate data run
        against the single zone floor (2.1-2.3) and flip the zone - the
        spatial floor is what removes this false positive."""
        h = make_harness()
        for _ in range(10):
            h.send_frame(KONTOR, still_d=250, still_e=40, still=True)
            h.step_to(h.now + 1.0)
        assert h.zone(DOOR).occupied  # z_still = 6 against the zone floor

    def test_rule_3_6_persisted_gate_floors_seed_the_engine(self) -> None:
        config = make_config(use_gate_evidence=True)
        floors = {DESK: {2: GateBaselines(0.2, 0.03, 0.3, 0.04)}}
        h = Harness(config, make_snapshot(config, gate_floors=floors))
        desk = h.zone(DESK)
        assert desk.gate_move_baselines.keys() == {2}
        assert desk.gate_move_baselines[2].mu == 0.2
        assert desk.gate_still_baselines[2].sigma == 0.04

    def test_rule_3_6_v0_1_0_baselines_without_gates_load_unchanged(self) -> None:
        h = Harness()  # default snapshot: ZoneBaselines carry no gates
        assert h.zone(DESK).gate_move_baselines == {}
        h.occupy(KONTOR)
        assert h.zone(DESK).occupied  # the aggregate estimator is untouched

    def test_rule_7_3_gate_frames_are_deterministic(self) -> None:
        def drive() -> tuple:
            h = make_harness()
            h.send_frame(KONTOR, gate_move=gate_tuple({1: 20}), at=1.0)
            h.run(30)
            h.send_frame(KONTOR, gate_move=gate_tuple({1: 8, 2: None}))
            h.run(30)
            return h.fingerprint()

        assert drive() == drive()
