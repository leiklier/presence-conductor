"""Diagnostics download: the per-frame numerics that entities must not carry."""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.presence_conductor.const import DOMAIN
from custom_components.presence_conductor.diagnostics import (
    async_get_config_entry_diagnostics,
)
from tests.test_controller import setup_conductor


async def test_diagnostics_dump_carries_full_engine_state(hass: HomeAssistant, monkeypatch) -> None:
    entry, _controller, fake = await setup_conductor(hass, monkeypatch)
    fake.state.zones["sofakrok"].dwell_seconds = 12.34

    dump = await async_get_config_entry_diagnostics(hass, entry)

    assert dump["configured"] is True
    assert dump["enabled"] is True
    assert "home_lambda" in dump
    assert dump["anyone_home"] is False
    sofakrok = dump["zones"]["sofakrok"]
    assert "lambda" in sofakrok
    assert "confidence" in sofakrok
    assert sofakrok["dwell_seconds"] == 12.3  # exact, unlike the bucketed sensor
    assert len(sofakrok["move_baseline"]) == 2
    assert all(isinstance(v, float) for v in sofakrok["move_baseline"])
    assert sofakrok["calibration"]["status"] == "recalibration_required"
    assert dump["rooms"]["stue"]["confidence"] is not None
    assert dump["sensors"]["sofakrok_radar"] == {"available": True}


async def test_diagnostics_track_runtime_evidence_path(hass: HomeAssistant, monkeypatch) -> None:
    """Gate dropout/recovery is observable here, not on entity attributes."""
    entry, controller, _fake = await setup_conductor(hass, monkeypatch)
    zone = controller.engine.state.zones["sofakrok"]

    zone.move_from_gates = True
    dump = await async_get_config_entry_diagnostics(hass, entry)
    assert dump["zones"]["sofakrok"]["calibration"]["move_runtime"] == "gate"

    zone.move_from_gates = False
    dump = await async_get_config_entry_diagnostics(hass, entry)
    assert dump["zones"]["sofakrok"]["calibration"]["move_runtime"] == "aggregate"


async def test_diagnostics_for_unconfigured_entry(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN, title="Presence Conductor", unique_id=DOMAIN, data={}, options={}
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert await async_get_config_entry_diagnostics(hass, entry) == {"configured": False}
