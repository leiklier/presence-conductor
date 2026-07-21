"""The recorder invariant, swept across every entity of the integration.

The v0.6.0-era regression: every mmWave frame triggered a global publish,
and any entity whose state or attributes carried per-frame numerics wrote
a recorder row per frame. The targeted fixes live next to their entities;
THIS test pins the invariant globally so a future entity (or a new
attribute on an existing one) cannot quietly reintroduce the bloat: a
burst of per-frame engine wiggle — belief drift, dwell inside one bucket,
baseline adaptation, gate-path flips — must not write a single state for
ANY registered entity.

If this test fails after adding an entity or attribute, the new value is
volatile: quantize it, move it to the diagnostics download, or derive it
from a genuine discrete transition.
"""

from __future__ import annotations

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_send

from custom_components.presence_conductor.core.model import ChannelStats
from tests.test_controller import setup_conductor

#: Entity inventory floor: 3 zones x 9 + 2 rooms x 6 + 4 hub entities.
#: Keep in sync with test_entity_inventory_spans_hub_and_room_devices —
#: the sweep is only meaningful if it really covers the whole surface.
EXPECTED_MINIMUM_ENTITIES = 3 * 9 + 2 * 6 + 4


@pytest.mark.usefixtures("entity_registry_enabled_by_default")
async def test_per_frame_wiggle_writes_no_entity_states(hass: HomeAssistant, monkeypatch) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    registry = er.async_get(hass)
    entity_ids = [
        entity.entity_id
        for entity in er.async_entries_for_config_entry(registry, entry.entry_id)
        if entity.disabled_by is None
    ]
    assert len(entity_ids) >= EXPECTED_MINIMUM_ENTITIES

    before = {}
    for entity_id in entity_ids:
        state = hass.states.get(entity_id)
        assert state is not None, entity_id
        before[entity_id] = state

    # 25 frames of realistic per-frame drift, each followed by both publish
    # signals (the state signal and the always-sent control signal), exactly
    # as the controller emits them after every engine plan. Discrete outputs
    # (occupancy, motion, activity) are deliberately NOT flipped — those are
    # genuine transitions and SHOULD write states.
    for frame in range(25):
        for zst in fake.state.zones.values():
            zst.lam += 0.0004  # sub-percent confidence drift
            zst.dwell_seconds += 0.3  # 7.5 s total: inside one 10 s bucket
            zst.move_baseline = ChannelStats(  # background EMA adaptation
                zst.move_baseline.mu + 0.0001, zst.move_baseline.sigma
            )
            zst.move_from_gates = frame % 2 == 0  # per-frame gate path flip
        for room in fake.state.rooms.values():
            room.confidence = 0.02 + 0.00004 * frame  # sub-percent
        fake.state.lam_home += 0.0004
        fake.state.home_confidence = 0.02 + 0.00004 * frame
        async_dispatcher_send(hass, controller.signal)
        async_dispatcher_send(hass, controller.signal_control)
    await hass.async_block_till_done()

    for entity_id in entity_ids:
        after = hass.states.get(entity_id)
        assert after.last_updated == before[entity_id].last_updated, (
            f"{entity_id} wrote a state during per-frame wiggle — "
            "volatile value on the entity surface (recorder bloat)"
        )
