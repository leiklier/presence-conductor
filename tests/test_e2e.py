"""End-to-end scenarios: real ConductorEngine through the real controller.

The engine and the adapter each have exhaustive isolated suites; these tests
prove the *seam* — a config entry shaped like the real installation (four
Apollo MSR-2 clusters, five zones, three rooms; the shape of
tests/test_config_flow.py), fake radar entity states, and observation
through real HA entities, bus events and config-entry options.

Time is driven with the ``freezer`` fixture: freezegun patches
``time.monotonic``, which both the engine's ``now`` values and HA's timer
wheel run on, so ticks, holds and watchdogs advance deterministically.
Baselines are calibrated tight (mu 0.02, sigma 0.02) so an energy of 80
saturates the evidence score and triggers the fast attack (rule 4.2).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_capture_events,
    async_fire_time_changed,
)

from custom_components.presence_conductor.const import ALL_ROLES, DOMAIN, GATE_ROLES
from custom_components.presence_conductor.controller import EVENT_PASS_BY
from tests.test_discovery import (
    KJOKKEN_PREFIX,
    KONTOR_PREFIX,
    SOFAKROK_PREFIX,
    SPISEBORD_PREFIX,
    SPISEBORD_ROLES,
    cluster_entities,
)

KJOKKEN = cluster_entities(KJOKKEN_PREFIX)
KONTOR = cluster_entities(KONTOR_PREFIX)
SOFAKROK = cluster_entities(SOFAKROK_PREFIX)
SPISEBORD = cluster_entities(SPISEBORD_PREFIX, SPISEBORD_ROLES)

TIGHT_BASELINE = {"move_mu": 0.02, "move_sigma": 0.02, "still_mu": 0.02, "still_sigma": 0.02}

OPTIONS: dict[str, Any] = {
    "sensors": [
        {"sensor_id": "apollo_msr_2_kjokken", "name": "Apollo MSR-2 Kjøkken", "entities": KJOKKEN},
        {"sensor_id": "apollo_msr_2_kontor", "name": "Apollo MSR-2 Kontor", "entities": KONTOR},
        {
            "sensor_id": "apollo_msr_2_sofakrok",
            "name": "Apollo MSR-2 Sofakrok",
            "entities": SOFAKROK,
        },
        {
            "sensor_id": "apollo_msr_2_spisebord",
            "name": "Apollo MSR-2 Spisebord",
            "entities": SPISEBORD,
        },
    ],
    "zones": [
        {
            "zone_id": "kjokken",
            "name": "Kjøkken",
            "sensor": "apollo_msr_2_kjokken",
            "room": "kjokken",
            "near_cm": 0.0,
            "far_cm": 600.0,
            "fallback": True,
        },
        {
            "zone_id": "kontor_pult",
            "name": "Kontor pult",
            "sensor": "apollo_msr_2_kontor",
            "room": "kontor",
            "near_cm": 0.0,
            "far_cm": 150.0,
            "fallback": True,
        },
        {
            "zone_id": "kontor_dor",
            "name": "Kontor dør",
            "sensor": "apollo_msr_2_kontor",
            "room": "kontor",
            "near_cm": 150.0,
            "far_cm": 400.0,
            "fallback": False,
        },
        {
            "zone_id": "sofakrok",
            "name": "Sofakrok",
            "sensor": "apollo_msr_2_sofakrok",
            "room": "stue",
            "near_cm": 0.0,
            "far_cm": 250.0,
            "fallback": False,
        },
        {
            "zone_id": "spisebord",
            "name": "Spisebord",
            "sensor": "apollo_msr_2_spisebord",
            "room": "stue",
            "near_cm": 260.0,
            "far_cm": 450.0,
            "fallback": False,
        },
    ],
    "rooms": [
        {"room_id": "kjokken", "name": "Kjøkken"},
        {"room_id": "kontor", "name": "Kontor"},
        {"room_id": "stue", "name": "Stue"},
    ],
    "baselines": {
        zone_id: dict(TIGHT_BASELINE)
        for zone_id in ("kjokken", "kontor_pult", "kontor_dor", "sofakrok", "spisebord")
    },
}

OCC_SOFAKROK = "binary_sensor.presence_conductor_sofakrok_occupancy"
OCC_KONTOR_PULT = "binary_sensor.presence_conductor_kontor_pult_occupancy"
OCC_KONTOR_DOR = "binary_sensor.presence_conductor_kontor_dor_occupancy"
OCC_ROOM_STUE = "binary_sensor.presence_conductor_stue_room_occupancy"
OCC_ROOM_KONTOR = "binary_sensor.presence_conductor_kontor_room_occupancy"
SETTLED_ROOM_STUE = "binary_sensor.presence_conductor_stue_room_settled"
SETTLED_ROOM_KONTOR = "binary_sensor.presence_conductor_kontor_room_settled"
ANYONE_HOME = "binary_sensor.presence_conductor_anyone_home"
ENABLED_SWITCH = "switch.presence_conductor_enabled"
STATE_SENSOR = "sensor.presence_conductor_state"


def seed_world(hass: HomeAssistant, overrides: dict[str, str] | None = None) -> None:
    """Idle radar states for every configured entity (nobody home)."""
    for cluster in (KJOKKEN, KONTOR, SOFAKROK, SPISEBORD):
        for role, entity_id in cluster.items():
            if entity_id.startswith("binary_sensor."):
                hass.states.async_set(entity_id, "off")
            elif "energy" in role:
                hass.states.async_set(entity_id, "2.0")
            else:
                hass.states.async_set(entity_id, "0.0")
    for entity_id, value in (overrides or {}).items():
        hass.states.async_set(entity_id, value)


async def setup_e2e(
    hass: HomeAssistant,
    overrides: dict[str, str] | None = None,
    options: dict[str, Any] | None = None,
):
    seed_world(hass, overrides)
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Presence Conductor",
        unique_id=DOMAIN,
        data={},
        options=options or OPTIONS,
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    controller = hass.data[DOMAIN][entry.entry_id]
    assert controller is not None
    return entry, controller


async def advance(hass: HomeAssistant, freezer, seconds: float) -> None:
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()


async def set_states(hass: HomeAssistant, states: dict[str, str]) -> None:
    for entity_id, value in states.items():
        hass.states.async_set(entity_id, value)
    await hass.async_block_till_done()


async def enter_zone(hass: HomeAssistant, cluster: dict[str, str], distance: str) -> None:
    """A person appears: strong move energy at a gated distance."""
    await set_states(
        hass,
        {
            cluster["move_energy"]: "80.0",
            cluster["moving_distance"]: distance,
            cluster["moving_target"]: "on",
            cluster["target"]: "on",
        },
    )


async def leave_zone(hass: HomeAssistant, cluster: dict[str, str]) -> None:
    """The person is gone: energies back at the noise floor."""
    await set_states(
        hass,
        {
            cluster["move_energy"]: "2.0",
            cluster["moving_distance"]: "0.0",
            cluster["moving_target"]: "off",
            cluster["target"]: "off",
        },
    )


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


async def test_walk_through_produces_pass_by_and_no_settle(hass: HomeAssistant, freezer) -> None:
    """Fast attack on entry; decay releases; PASSING exit emits pass_by."""
    _entry, _controller = await setup_e2e(hass)
    captured = async_capture_events(hass, EVENT_PASS_BY)
    assert hass.states.get(OCC_SOFAKROK).state == "off"

    await enter_zone(hass, SOFAKROK, "150.0")
    # Rule 4.2: the fast attack fires on the frame itself, no tick needed.
    assert hass.states.get(OCC_SOFAKROK).state == "on"
    assert hass.states.get("binary_sensor.presence_conductor_sofakrok_motion").state == "on"
    assert hass.states.get("sensor.presence_conductor_sofakrok_activity").state == "passing"
    assert hass.states.get(OCC_ROOM_STUE).state == "on"
    assert hass.states.get("sensor.presence_conductor_stue_room_activity").state == "passing"
    assert hass.states.get(ANYONE_HOME).state == "on"

    await leave_zone(hass, SOFAKROK)
    for _ in range(15):  # decay + absence evidence release well within this
        await advance(hass, freezer, 1.0)

    assert hass.states.get(OCC_SOFAKROK).state == "off"
    assert hass.states.get("sensor.presence_conductor_sofakrok_activity").state == "empty"
    assert hass.states.get(OCC_ROOM_STUE).state == "off"
    # Rule 5.2: EMPTY reached from PASSING emitted exactly one pass_by.
    assert len(captured) == 1
    payload = captured[0].data
    assert payload["zone_id"] == "sofakrok"
    assert payload["peak_probability"] >= 0.9
    assert 1.0 <= payload["duration"] <= 15.0
    event_state = hass.states.get("event.presence_conductor_sofakrok_pass_by")
    assert event_state.state != "unknown"
    assert event_state.attributes["event_type"] == "pass_by"
    assert event_state.attributes["peak_probability"] >= 0.9
    # Rule 5.3: a walk-through never settles the room.
    assert hass.states.get(SETTLED_ROOM_STUE).state == "off"
    # Rule 6.5: anyone_home outlives the walk-through (tau_home is slow).
    assert hass.states.get(ANYONE_HOME).state == "on"


async def test_still_person_settles_the_room(hass: HomeAssistant, freezer) -> None:
    """Still-channel dominance for t_settle promotes to SETTLED (rule 5.1)."""
    _entry, _controller = await setup_e2e(hass)

    await enter_zone(hass, KONTOR, "100.0")
    assert hass.states.get(OCC_KONTOR_PULT).state == "on"

    # They sit down: move evidence gone, still energy holds the zone.
    await set_states(
        hass,
        {
            KONTOR["move_energy"]: "2.0",
            KONTOR["moving_target"]: "off",
            KONTOR["still_energy"]: "40.0",
            KONTOR["still_distance"]: "100.0",
            KONTOR["still_target"]: "on",
        },
    )
    for i in range(35):  # t_settle = 30
        await advance(hass, freezer, 1.0)
        if i % 10 == 9:
            # The radar keeps reporting a (noisy) still target; without
            # frames the staleness watchdog would declare the sensor blind.
            await set_states(hass, {KONTOR["still_energy"]: f"4{i % 3}.0"})

    assert hass.states.get(OCC_KONTOR_PULT).state == "on"  # still occupied
    assert hass.states.get("sensor.presence_conductor_kontor_pult_activity").state == "settled"
    assert hass.states.get(SETTLED_ROOM_KONTOR).state == "on"
    assert hass.states.get("sensor.presence_conductor_kontor_room_activity").state == "settled"
    # The sibling zone saw nothing (distance-gated, rule 2.1).
    assert hass.states.get(OCC_KONTOR_DOR).state == "off"
    assert float(hass.states.get("sensor.presence_conductor_kontor_pult_dwell").state) >= 30.0


async def test_occupied_sensor_dropout_bridges_through_unknown(
    hass: HomeAssistant, freezer
) -> None:
    """Rule 1.3: silence while occupied -> UNKNOWN (unavailable), not off."""
    _entry, _controller = await setup_e2e(hass)

    await enter_zone(hass, KONTOR, "100.0")
    assert hass.states.get(OCC_KONTOR_PULT).state == "on"

    # The sensor falls silent while the zone is occupied. The posterior is
    # held by the persisted evidence, and after stale_after (30 s) the
    # watchdog declares the sensor blind.
    for _ in range(31):
        await advance(hass, freezer, 1.0)

    assert hass.states.get(OCC_KONTOR_PULT).state == "unavailable"
    assert hass.states.get(OCC_KONTOR_DOR).state == "unavailable"  # same sensor
    # Rule 6.3: the room publishes unknown, not off.
    assert hass.states.get(OCC_ROOM_KONTOR).state == "unavailable"
    assert hass.states.get("sensor.presence_conductor_kontor_room_activity").state == "unavailable"
    # Other sensors are healthy, so home presence stays published (6.5).
    assert hass.states.get(ANYONE_HOME).state == "on"

    # Recovery is immediate on the next frame (1.3) — outputs were held.
    await set_states(hass, {KONTOR["move_energy"]: "81.0"})
    assert hass.states.get(OCC_KONTOR_PULT).state == "on"
    assert hass.states.get(OCC_ROOM_KONTOR).state == "on"


async def test_unavailable_entities_at_startup_recover(hass: HomeAssistant, freezer) -> None:
    """Sensors born unavailable seed UNKNOWN zones; the first frame heals."""
    _entry, _controller = await setup_e2e(hass, overrides={KJOKKEN["move_energy"]: "unavailable"})
    occupancy = "binary_sensor.presence_conductor_kjokken_occupancy"
    assert hass.states.get(occupancy).state == "unavailable"
    assert hass.states.get("binary_sensor.presence_conductor_kjokken_room_occupancy").state == (
        "unavailable"
    )

    hass.states.async_set(KJOKKEN["move_energy"], "2.0")
    await hass.async_block_till_done()
    assert hass.states.get(occupancy).state == "off"


async def test_startup_adopts_present_person(hass: HomeAssistant) -> None:
    """Rule 7.1: a gated target at startup seeds occupancy at theta_on."""
    _entry, _controller = await setup_e2e(
        hass,
        overrides={
            SOFAKROK["move_energy"]: "60.0",
            SOFAKROK["moving_distance"]: "150.0",
            SOFAKROK["moving_target"]: "on",
            SOFAKROK["target"]: "on",
        },
    )
    assert hass.states.get(OCC_SOFAKROK).state == "on"
    assert hass.states.get("sensor.presence_conductor_sofakrok_activity").state == "passing"
    assert hass.states.get(OCC_ROOM_STUE).state == "on"


async def test_record_baseline_service_persists_into_options(hass: HomeAssistant, freezer) -> None:
    """Rule 3.3: the service records and the result lands in entry.options."""
    entry, controller = await setup_e2e(hass)

    await hass.services.async_call(
        DOMAIN, "record_baseline", {"zone_id": "kjokken", "duration": 5}, blocking=True
    )
    for value in ("3.0", "4.0", "3.5", "5.0", "4.5"):
        await set_states(hass, {KJOKKEN["move_energy"]: value})
        await advance(hass, freezer, 1.0)
    await advance(hass, freezer, 0.5)  # the 5 s window closes

    recorded = entry.options["baselines"]["kjokken"]
    assert recorded["move_mu"] == pytest.approx(0.04)  # median of the samples
    assert recorded["move_sigma"] == pytest.approx(0.02)  # floored at sigma_min
    assert recorded["still_mu"] == pytest.approx(0.02)  # unchanged channel
    # The baselines-only write must not reload the entry.
    assert hass.data[DOMAIN][entry.entry_id] is controller


async def test_disable_freezes_outputs_and_swallows_events(hass: HomeAssistant, freezer) -> None:
    """Rule 7.2: disabled = warm engine, frozen entities, no events."""
    _entry, controller = await setup_e2e(hass)
    captured = async_capture_events(hass, EVENT_PASS_BY)

    await hass.services.async_call(
        "switch", "turn_off", {"entity_id": ENABLED_SWITCH}, blocking=True
    )
    await hass.async_block_till_done()
    assert hass.states.get(ENABLED_SWITCH).state == "off"
    assert hass.states.get(STATE_SENSOR).state == "disabled"

    # A full walk-through while disabled: the engine tracks it underneath,
    # the entities never move, and the pass_by is swallowed.
    await enter_zone(hass, SOFAKROK, "150.0")
    assert controller.state.zones["sofakrok"].occupied is True  # warm underneath
    assert hass.states.get(OCC_SOFAKROK).state == "off"  # frozen for consumers
    await leave_zone(hass, SOFAKROK)
    for _ in range(15):
        await advance(hass, freezer, 1.0)
    assert captured == []
    assert hass.states.get("event.presence_conductor_sofakrok_pass_by").state == "unknown"

    # Someone is present when the operator re-enables: one publish catches
    # every entity up with reality.
    await enter_zone(hass, KONTOR, "100.0")
    assert hass.states.get(OCC_KONTOR_PULT).state == "off"  # still frozen
    await hass.services.async_call(
        "switch", "turn_on", {"entity_id": ENABLED_SWITCH}, blocking=True
    )
    await hass.async_block_till_done()
    assert hass.states.get(ENABLED_SWITCH).state == "on"
    assert hass.states.get(OCC_KONTOR_PULT).state == "on"  # caught up
    assert hass.states.get(STATE_SENSOR).state == "enabled"

    # And events flow again: a fresh walk-through emits its pass_by.
    await enter_zone(hass, SOFAKROK, "150.0")
    await leave_zone(hass, SOFAKROK)
    for _ in range(15):
        await advance(hass, freezer, 1.0)
    assert len(captured) == 1
    assert captured[0].data["zone_id"] == "sofakrok"


GATED_PREFIX = "apollo_msr_2_gates"
GATED_CLUSTER = cluster_entities(GATED_PREFIX, ALL_ROLES + GATE_ROLES)
GATED_OPTIONS: dict[str, Any] = {
    "sensors": [
        {"sensor_id": "apollo_gates", "name": "Apollo MSR-2 Gates", "entities": GATED_CLUSTER}
    ],
    "zones": [
        {
            "zone_id": "gates_near",
            "name": "Gates near",
            "sensor": "apollo_gates",
            "room": "gates",
            "near_cm": 0.0,
            "far_cm": 150.0,
            "fallback": True,
        },
        {
            "zone_id": "gates_far",
            "name": "Gates far",
            "sensor": "apollo_gates",
            "room": "gates",
            "near_cm": 220.0,
            "far_cm": 300.0,
            "fallback": False,
        },
    ],
    "rooms": [{"room_id": "gates", "name": "Gates"}],
    "baselines": {
        zone_id: {**TIGHT_BASELINE, "gates": {str(i): dict(TIGHT_BASELINE) for i in range(9)}}
        for zone_id in ("gates_near", "gates_far")
    },
}


def seed_gated_world(hass: HomeAssistant) -> None:
    """Idle states for the gated cluster (gate energies at the tight floor)."""
    for role, entity_id in GATED_CLUSTER.items():
        if entity_id.startswith("binary_sensor."):
            hass.states.async_set(entity_id, "off")
        elif role.startswith("g") or "energy" in role:
            hass.states.async_set(entity_id, "2.0")
        else:
            hass.states.async_set(entity_id, "0.0")


async def test_gate_evidence_end_to_end(hass: HomeAssistant) -> None:
    """Spec 2.4-2.6 through the real seam: gate entities drive occupancy
    spatially, and an engineering-mode dropout falls back per frame."""
    seed_gated_world(hass)
    _entry, _controller = await setup_e2e(hass, options=GATED_OPTIONS)
    occ_near = "binary_sensor.presence_conductor_gates_near_occupancy"
    occ_far = "binary_sensor.presence_conductor_gates_far_occupancy"
    assert hass.states.get(occ_near).state == "off"
    assert hass.states.get(occ_far).state == "off"

    # Strong move at gate 3 ([225, 300) cm): only the far zone owns it
    # (2.4), so only the far zone reacts - even though the aggregate
    # distance points squarely at the near zone. Gate precedence (2.6).
    await set_states(
        hass,
        {
            GATED_CLUSTER["g3_move"]: "80.0",
            GATED_CLUSTER["move_energy"]: "80.0",
            GATED_CLUSTER["moving_distance"]: "100.0",
            GATED_CLUSTER["moving_target"]: "on",
        },
    )
    assert hass.states.get(occ_far).state == "on"  # 2.5 + 4.2
    assert hass.states.get(occ_near).state == "off"  # aggregate path ignored

    # Engineering mode drops: every gate reads unknown. The very next frame
    # runs the aggregate path (2.6), which credits the near zone at 100 cm -
    # and nothing goes unavailable (1.3 keys on the aggregate energies).
    await set_states(
        hass,
        {GATED_CLUSTER[role]: "unknown" for role in GATE_ROLES},
    )
    assert hass.states.get(occ_near).state == "on"  # fallback, per frame
    assert hass.states.get(occ_far).state != "unavailable"


async def test_full_entity_inventory(hass: HomeAssistant) -> None:
    """5 zones, 3 rooms, home, controls: the complete §0 surface."""
    from homeassistant.helpers import entity_registry as er

    entry, _controller = await setup_e2e(hass)
    registry = er.async_get(hass)
    entries = er.async_entries_for_config_entry(registry, entry.entry_id)
    # Per zone: occupancy, motion, activity, probability, dwell, pass-by
    # event, record-baseline button. Per room: occupancy, settled, activity,
    # probability. Home: anyone_home + probability. Plus enabled + state.
    assert len(entries) == 5 * 7 + 3 * 4 + 4

    for zone_id in ("kjokken", "kontor_pult", "kontor_dor", "sofakrok", "spisebord"):
        for entity_id in (
            f"binary_sensor.presence_conductor_{zone_id}_occupancy",
            f"binary_sensor.presence_conductor_{zone_id}_motion",
            f"sensor.presence_conductor_{zone_id}_activity",
            f"sensor.presence_conductor_{zone_id}_probability",
            f"sensor.presence_conductor_{zone_id}_dwell",
            f"event.presence_conductor_{zone_id}_pass_by",
            f"button.presence_conductor_{zone_id}_record_baseline",
        ):
            assert hass.states.get(entity_id) is not None, entity_id
    for room_id in ("kjokken", "kontor", "stue"):
        for entity_id in (
            f"binary_sensor.presence_conductor_{room_id}_room_occupancy",
            f"binary_sensor.presence_conductor_{room_id}_room_settled",
            f"sensor.presence_conductor_{room_id}_room_activity",
            f"sensor.presence_conductor_{room_id}_room_probability",
        ):
            assert hass.states.get(entity_id) is not None, entity_id
    for entity_id in (
        ANYONE_HOME,
        "sensor.presence_conductor_home_probability",
        ENABLED_SWITCH,
        STATE_SENSOR,
    ):
        assert hass.states.get(entity_id) is not None, entity_id
