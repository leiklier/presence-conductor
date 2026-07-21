"""Entity platform tests: engine-state mirroring and command routing."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.dispatcher import async_dispatcher_send
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    mock_restore_cache,
)

from custom_components.presence_conductor import sensor as sensor_module
from custom_components.presence_conductor.calibration import CALIBRATION_ISSUE_ID_PREFIX
from custom_components.presence_conductor.const import DOMAIN
from custom_components.presence_conductor.core.events import RecordBaseline, SetEnabled
from custom_components.presence_conductor.core.model import Activity, Health, Tunables
from custom_components.presence_conductor.core.stats import floor_calibration_fingerprint
from custom_components.presence_conductor.sensor import (
    CONFIDENCE_PUBLISH_INTERVAL,
    ConductorStateSensor,
)
from tests.test_controller import KONTOR, OPTIONS, SOFAKROK, setup_conductor


def entity_id_for(hass: HomeAssistant, platform: str, unique_id: str) -> str:
    registry = er.async_get(hass)
    entity_id = registry.async_get_entity_id(platform, DOMAIN, unique_id)
    assert entity_id is not None, f"no {platform} entity with unique_id {unique_id}"
    return entity_id


@pytest.fixture
def confidence_clock(monkeypatch):
    """Script the confidence publish-gate clock (sensor._monotonic).

    Returns an ``advance(seconds)`` callable; use it between dispatches to
    step past CONFIDENCE_PUBLISH_INTERVAL. Without it, a value change
    dispatched right after setup lands inside the startup publish's
    interval and is (correctly) suppressed.
    """
    state = {"now": 1000.0}
    monkeypatch.setattr(sensor_module, "_monotonic", lambda: state["now"])

    def advance(seconds: float) -> None:
        state["now"] += seconds

    return advance


# ---------------------------------------------------------------------------
# zone binary sensors
# ---------------------------------------------------------------------------

# Zone state entities ship disabled (rooms and home are the consumer
# surface; tests/test_devices.py covers the defaults). Tests exercising
# zone entity behavior opt back in with entity_registry_enabled_by_default.


@pytest.mark.usefixtures("entity_registry_enabled_by_default")
async def test_zone_occupancy_mirrors_engine_state(hass: HomeAssistant, monkeypatch) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    occupancy = entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_zone_sofakrok_occupancy")
    assert occupancy == "binary_sensor.presence_conductor_sofakrok_occupancy"

    state = hass.states.get(occupancy)
    assert state.state == "off"
    assert state.attributes["device_class"] == "occupancy"
    assert state.attributes["zone_id"] == "sofakrok"
    assert state.attributes["room"] == "stue"

    fake.state.zones["sofakrok"].occupied = True
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(occupancy).state == "on"


@pytest.mark.usefixtures("entity_registry_enabled_by_default")
async def test_zone_motion_mirrors_engine_state(hass: HomeAssistant, monkeypatch) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    motion = entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_zone_kontor_pult_motion")

    state = hass.states.get(motion)
    assert state.state == "off"
    assert state.attributes["device_class"] == "motion"

    fake.state.zones["kontor_pult"].motion = True
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(motion).state == "on"


@pytest.mark.usefixtures("entity_registry_enabled_by_default")
async def test_zone_entities_unavailable_while_health_unknown(
    hass: HomeAssistant, monkeypatch
) -> None:
    """Rule 1.3: stale confidence maps to unavailable, recovery restores."""
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    entity_ids = [
        entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_zone_sofakrok_occupancy"),
        entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_zone_sofakrok_motion"),
        entity_id_for(hass, "sensor", f"{entry.entry_id}_zone_sofakrok_activity"),
        entity_id_for(hass, "sensor", f"{entry.entry_id}_zone_sofakrok_confidence"),
        entity_id_for(hass, "sensor", f"{entry.entry_id}_zone_sofakrok_dwell"),
    ]

    fake.state.zones["sofakrok"].health = Health.UNKNOWN
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    for entity_id in entity_ids:
        assert hass.states.get(entity_id).state == "unavailable", entity_id

    # Sibling zones on other sensors are untouched.
    other = entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_zone_kontor_pult_occupancy")
    assert hass.states.get(other).state == "off"

    fake.state.zones["sofakrok"].health = Health.OK
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    for entity_id in entity_ids:
        assert hass.states.get(entity_id).state != "unavailable", entity_id


# ---------------------------------------------------------------------------
# zone sensors
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("entity_registry_enabled_by_default")
async def test_zone_activity_enum(hass: HomeAssistant, monkeypatch) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    activity = entity_id_for(hass, "sensor", f"{entry.entry_id}_zone_sofakrok_activity")

    state = hass.states.get(activity)
    assert state.state == "empty"
    assert state.attributes["options"] == ["empty", "passing", "active", "settled"]
    assert state.attributes["device_class"] == "enum"

    fake.state.zones["sofakrok"].activity = Activity.SETTLED
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(activity).state == "settled"


@pytest.mark.usefixtures("entity_registry_enabled_by_default")
async def test_zone_confidence_and_dwell(
    hass: HomeAssistant, monkeypatch, confidence_clock
) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    confidence = entity_id_for(hass, "sensor", f"{entry.entry_id}_zone_sofakrok_confidence")
    dwell = entity_id_for(hass, "sensor", f"{entry.entry_id}_zone_sofakrok_dwell")

    # The 2% prior lands in the 0 bucket (5-point steps).
    assert hass.states.get(confidence).state == "0"
    assert hass.states.get(confidence).attributes["unit_of_measurement"] == "%"
    assert hass.states.get(dwell).state == "0"
    assert hass.states.get(dwell).attributes["unit_of_measurement"] == "s"

    fake.state.zones["sofakrok"].lam = 0.0  # p = 0.5
    fake.state.zones["sofakrok"].dwell_seconds = 12.34
    confidence_clock(CONFIDENCE_PUBLISH_INTERVAL + 1)
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(confidence).state == "50"
    # Dwell is floored to 10 s buckets so the recorder only sees a row
    # every bucket, not every tick.
    assert hass.states.get(dwell).state == "10"


async def test_diagnostic_and_config_entity_categories(hass: HomeAssistant, monkeypatch) -> None:
    entry, _controller, _fake = await setup_conductor(hass, monkeypatch)
    registry = er.async_get(hass)

    diagnostic = er.EntityCategory.DIAGNOSTIC

    def category(platform: str, unique_id: str):
        return registry.async_get(entity_id_for(hass, platform, unique_id)).entity_category

    assert category("sensor", f"{entry.entry_id}_zone_sofakrok_confidence") == diagnostic
    assert category("sensor", f"{entry.entry_id}_zone_sofakrok_calibration_status") == diagnostic
    assert category("sensor", f"{entry.entry_id}_zone_sofakrok_dwell") == diagnostic
    assert category("sensor", f"{entry.entry_id}_room_stue_confidence") == diagnostic
    assert category("sensor", f"{entry.entry_id}_home_confidence") == diagnostic
    assert category("sensor", f"{entry.entry_id}_state") == diagnostic
    assert category("switch", f"{entry.entry_id}_enabled") == er.EntityCategory.CONFIG
    assert (
        category("button", f"{entry.entry_id}_zone_sofakrok_record_baseline")
        == er.EntityCategory.CONFIG
    )
    # Primary outputs stay uncategorized.
    assert category("binary_sensor", f"{entry.entry_id}_zone_sofakrok_occupancy") is None
    assert category("sensor", f"{entry.entry_id}_zone_sofakrok_activity") is None


async def test_calibration_status_and_repairs_surface_startup_fallbacks(
    hass: HomeAssistant, monkeypatch
) -> None:
    entry, _controller, _fake = await setup_conductor(hass, monkeypatch)
    calibration = entity_id_for(
        hass, "sensor", f"{entry.entry_id}_zone_sofakrok_calibration_status"
    )
    state = hass.states.get(calibration)
    assert state.state == "recalibration_required"
    assert state.attributes["reason_codes"] == ["legacy_context"]
    assert state.attributes["floor_source"] == "recorded"
    # Per-frame runtime paths stay off the entity (recorder discipline);
    # they are asserted through diagnostics in tests/test_diagnostics.py.
    assert "move_statistic" not in state.attributes
    assert "move_runtime" not in state.attributes
    assert "record a new baseline" in state.attributes["action"]

    uncalibrated = entity_id_for(
        hass, "sensor", f"{entry.entry_id}_zone_kontor_pult_calibration_status"
    )
    assert hass.states.get(uncalibrated).state == "uncalibrated"
    issue = ir.async_get(hass).async_get_issue(
        DOMAIN, f"{CALIBRATION_ISSUE_ID_PREFIX}_{entry.entry_id}"
    )
    assert issue is not None
    assert issue.severity is ir.IssueSeverity.WARNING
    assert issue.is_fixable is False
    assert issue.is_persistent is False
    assert issue.translation_key == "calibration_required"
    assert "Sofakrok" in issue.translation_placeholders["details"]

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert (
        ir.async_get(hass).async_get_issue(
            DOMAIN, f"{CALIBRATION_ISSUE_ID_PREFIX}_{entry.entry_id}"
        )
        is None
    )


async def test_compatible_calibration_has_no_repairs_issue(
    hass: HomeAssistant, monkeypatch
) -> None:
    options = deepcopy(OPTIONS)
    options["baselines"] = {
        zone["zone_id"]: {
            "move_mu": 0.02,
            "move_sigma": 0.02,
            "still_mu": 0.02,
            "still_sigma": 0.02,
            "sensor_id": zone["sensor"],
            "floor_fingerprint": floor_calibration_fingerprint(Tunables()),
            "gate_size_cm": 75.0,
        }
        for zone in options["zones"]
    }
    entry, _controller, _fake = await setup_conductor(hass, monkeypatch, options=options)
    for zone_id in ("kontor_pult", "kontor_dor", "sofakrok"):
        calibration = entity_id_for(
            hass, "sensor", f"{entry.entry_id}_zone_{zone_id}_calibration_status"
        )
        assert hass.states.get(calibration).state == "ready"
    assert (
        ir.async_get(hass).async_get_issue(
            DOMAIN, f"{CALIBRATION_ISSUE_ID_PREFIX}_{entry.entry_id}"
        )
        is None
    )


async def test_enabled_gate_path_without_gate_calibration_is_visible(
    hass: HomeAssistant, monkeypatch
) -> None:
    options = deepcopy(OPTIONS)
    options["tunables"] = {"use_gate_evidence": True}
    options["sensors"][0]["entities"]["g0_move"] = "sensor.kontor_g0_move"
    options["baselines"] = {
        "kontor_pult": {
            "move_mu": 0.02,
            "move_sigma": 0.02,
            "still_mu": 0.02,
            "still_sigma": 0.02,
            "sensor_id": KONTOR,
            "floor_fingerprint": floor_calibration_fingerprint(Tunables()),
            "gate_size_cm": 100.0,
        },
        "kontor_dor": {
            "move_mu": 0.02,
            "move_sigma": 0.02,
            "still_mu": 0.02,
            "still_sigma": 0.02,
            "sensor_id": KONTOR,
            "floor_fingerprint": floor_calibration_fingerprint(Tunables()),
            "gate_size_cm": 75.0,
        },
        "sofakrok": {
            "move_mu": 0.02,
            "move_sigma": 0.02,
            "still_mu": 0.02,
            "still_sigma": 0.02,
            "sensor_id": SOFAKROK,
            "floor_fingerprint": floor_calibration_fingerprint(Tunables()),
            "gate_size_cm": 75.0,
        },
    }
    entry, _controller, _fake = await setup_conductor(hass, monkeypatch, options=options)
    calibration = entity_id_for(
        hass, "sensor", f"{entry.entry_id}_zone_kontor_pult_calibration_status"
    )
    state = hass.states.get(calibration)
    assert state.state == "recalibration_required"
    assert "gate_move_calibration_missing" in state.attributes["reason_codes"]
    assert "gate_resolution_changed" in state.attributes["reason_codes"]
    assert "Gate resolution changed from 100 cm to 75 cm." in state.attributes["reasons"]


async def test_changed_floor_fit_settings_require_recalibration(
    hass: HomeAssistant, monkeypatch
) -> None:
    options = deepcopy(OPTIONS)
    options["tunables"] = {"energy_quantum": 0.10}
    options["baselines"] = {
        "sofakrok": {
            "move_mu": 0.02,
            "move_sigma": 0.02,
            "still_mu": 0.02,
            "still_sigma": 0.02,
            "sensor_id": SOFAKROK,
            "floor_fingerprint": floor_calibration_fingerprint(Tunables()),
        }
    }
    entry, _controller, _fake = await setup_conductor(hass, monkeypatch, options=options)
    calibration = entity_id_for(
        hass, "sensor", f"{entry.entry_id}_zone_sofakrok_calibration_status"
    )
    state = hass.states.get(calibration)
    assert state.state == "recalibration_required"
    assert state.attributes["reason_codes"] == ["floor_settings_changed"]
    assert state.attributes["floor_source"] == "default"


async def test_runtime_source_flips_stay_off_the_calibration_entity(
    hass: HomeAssistant, monkeypatch
) -> None:
    """Per-frame gate dropout/recovery must not rewrite entity attributes —
    that wrote a recorder row per flip. The runtime path is tracked through
    diagnostics instead (tests/test_diagnostics.py)."""
    entry, controller, _fake = await setup_conductor(hass, monkeypatch)
    zone = controller.engine.state.zones["sofakrok"]
    calibration = entity_id_for(
        hass, "sensor", f"{entry.entry_id}_zone_sofakrok_calibration_status"
    )

    before = hass.states.get(calibration)
    zone.move_from_gates = True
    async_dispatcher_send(hass, controller.signal_control)
    await hass.async_block_till_done()
    after = hass.states.get(calibration)
    assert after.attributes == before.attributes
    assert after.last_updated == before.last_updated  # no state-machine write at all


async def test_failed_unload_keeps_repairs_warning(hass: HomeAssistant, monkeypatch) -> None:
    import custom_components.presence_conductor as integration

    entry, _controller, _fake = await setup_conductor(hass, monkeypatch)
    issue_id = f"{CALIBRATION_ISSUE_ID_PREFIX}_{entry.entry_id}"
    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id) is not None

    with patch.object(hass.config_entries, "async_unload_platforms", AsyncMock(return_value=False)):
        assert await integration.async_unload_entry(hass, entry) is False

    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id) is not None
    assert hass.data[DOMAIN][entry.entry_id] is not None


async def test_incompatible_statistic_reports_analytic_fallback(
    hass: HomeAssistant, monkeypatch
) -> None:
    options = deepcopy(OPTIONS)
    options["baselines"] = {
        zone["zone_id"]: {
            "move_mu": 0.02,
            "move_sigma": 0.02,
            "still_mu": 0.02,
            "still_sigma": 0.02,
            "sensor_id": zone["sensor"],
            "floor_fingerprint": floor_calibration_fingerprint(Tunables()),
            "gate_size_cm": 75.0,
        }
        for zone in options["zones"]
    }
    options["baselines"]["sofakrok"]["stats"] = {
        "move_agg": {"mu": 0.4, "sigma": 0.6, "fingerprint": "old-transform"}
    }
    entry, _controller, _fake = await setup_conductor(hass, monkeypatch, options=options)
    calibration = entity_id_for(
        hass, "sensor", f"{entry.entry_id}_zone_sofakrok_calibration_status"
    )
    state = hass.states.get(calibration)
    assert state.state == "recalibration_required"
    assert state.attributes["reason_codes"] == ["statistic_context_changed"]


# ---------------------------------------------------------------------------
# room entities
# ---------------------------------------------------------------------------


async def test_room_entities_mirror_fusion(
    hass: HomeAssistant, monkeypatch, confidence_clock
) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    occupancy = entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_room_kontor_occupancy")
    motion = entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_room_kontor_motion")
    settled = entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_room_kontor_settled")
    activity = entity_id_for(hass, "sensor", f"{entry.entry_id}_room_kontor_activity")
    confidence = entity_id_for(hass, "sensor", f"{entry.entry_id}_room_kontor_confidence")
    assert occupancy == "binary_sensor.presence_conductor_kontor_room_occupancy"
    assert motion == "binary_sensor.presence_conductor_kontor_room_motion"

    assert hass.states.get(occupancy).state == "off"
    assert hass.states.get(occupancy).attributes["zones"] == ["kontor_pult", "kontor_dor"]
    assert hass.states.get(motion).state == "off"
    assert hass.states.get(motion).attributes["device_class"] == "motion"
    assert hass.states.get(settled).state == "off"
    assert hass.states.get(activity).state == "empty"

    room = fake.state.rooms["kontor"]
    room.occupied = True
    room.motion = True
    room.settled = True
    room.activity = Activity.SETTLED
    room.confidence = 0.987
    confidence_clock(CONFIDENCE_PUBLISH_INTERVAL + 1)
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(occupancy).state == "on"
    assert hass.states.get(motion).state == "on"
    assert hass.states.get(settled).state == "on"
    assert hass.states.get(activity).state == "settled"
    # 5-point buckets (recorder discipline): 0.987 lands in 100.
    assert hass.states.get(confidence).state == "100"


async def test_sub_bucket_confidence_wiggle_writes_no_state(
    hass: HomeAssistant, monkeypatch, confidence_clock
) -> None:
    """The recorder-bloat regression: per-frame confidence wiggle inside one
    5-point bucket must not touch the state machine at all — even when the
    publish interval has long expired."""
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    confidence = entity_id_for(hass, "sensor", f"{entry.entry_id}_room_kontor_confidence")
    room = fake.state.rooms["kontor"]

    room.confidence = 0.502
    confidence_clock(CONFIDENCE_PUBLISH_INTERVAL + 1)
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    before = hass.states.get(confidence)
    assert before.state == "50"

    for wiggle in (0.49, 0.512, 0.4877):  # all land in the 50 bucket
        room.confidence = wiggle
        confidence_clock(CONFIDENCE_PUBLISH_INTERVAL + 1)  # interval is NOT the guard here
        async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    after = hass.states.get(confidence)
    assert after.state == "50"
    assert after.last_updated == before.last_updated  # no new recorder row


async def test_confidence_sweep_is_rate_limited(
    hass: HomeAssistant, monkeypatch, confidence_clock
) -> None:
    """The v0.5.2 live finding: a belief sweep wrote a row per frame (one
    per whole percent, ~10/s). Inside one publish interval a sweep must
    write at most one state; the settled value lands on a later tick."""
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    confidence = entity_id_for(hass, "sensor", f"{entry.entry_id}_room_kontor_confidence")
    room = fake.state.rooms["kontor"]

    room.confidence = 0.10
    confidence_clock(CONFIDENCE_PUBLISH_INTERVAL + 1)
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    first = hass.states.get(confidence)
    assert first.state == "10"

    # A full sweep dispatched frame-by-frame within the interval.
    for step in range(11, 90):
        room.confidence = step / 100.0
        async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    during = hass.states.get(confidence)
    assert during.last_updated == first.last_updated  # every frame suppressed

    # The next dispatch after the interval lands the settled value.
    confidence_clock(CONFIDENCE_PUBLISH_INTERVAL + 1)
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(confidence).state == "90"


async def test_confidence_availability_bypasses_publish_interval(
    hass: HomeAssistant, monkeypatch, confidence_clock
) -> None:
    """Rule 6.3: blind fusion must surface immediately, throttled or not."""
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    confidence = entity_id_for(hass, "sensor", f"{entry.entry_id}_room_kontor_confidence")
    room = fake.state.rooms["kontor"]

    room.confidence = 0.60
    confidence_clock(CONFIDENCE_PUBLISH_INTERVAL + 1)
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(confidence).state == "60"

    room.confidence = None  # fusion blind — inside the interval
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(confidence).state == "unavailable"

    room.confidence = 0.60  # recovery is availability too — also immediate
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(confidence).state == "60"


async def test_room_entities_unavailable_when_fusion_unknown(
    hass: HomeAssistant, monkeypatch
) -> None:
    """Rule 6.3: a blind room publishes unknown, not off."""
    entry, controller, fake = await setup_conductor(hass, monkeypatch)

    room = fake.state.rooms["kontor"]
    room.occupied = None
    room.motion = None
    room.settled = None
    room.activity = None
    room.confidence = None
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    for unique_suffix, platform in (
        ("occupancy", "binary_sensor"),
        ("motion", "binary_sensor"),
        ("settled", "binary_sensor"),
        ("activity", "sensor"),
        ("confidence", "sensor"),
    ):
        entity_id = entity_id_for(hass, platform, f"{entry.entry_id}_room_kontor_{unique_suffix}")
        assert hass.states.get(entity_id).state == "unavailable", entity_id


# ---------------------------------------------------------------------------
# home entities
# ---------------------------------------------------------------------------


async def test_anyone_home_and_confidence(
    hass: HomeAssistant, monkeypatch, confidence_clock
) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    anyone = entity_id_for(hass, "binary_sensor", f"{entry.entry_id}_anyone_home")
    confidence = entity_id_for(hass, "sensor", f"{entry.entry_id}_home_confidence")
    assert anyone == "binary_sensor.presence_conductor_anyone_home"

    state = hass.states.get(anyone)
    assert state.state == "off"
    assert state.attributes["device_class"] == "presence"

    fake.state.anyone_home = True
    fake.state.home_confidence = 0.964
    confidence_clock(CONFIDENCE_PUBLISH_INTERVAL + 1)
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(anyone).state == "on"
    # 5-point buckets (recorder discipline): 0.964 lands in 95.
    assert hass.states.get(confidence).state == "95"

    # Rule 6.5: all zones unhealthy -> anyone_home publishes unknown.
    fake.state.anyone_home = None
    fake.state.home_confidence = None
    async_dispatcher_send(hass, controller.signal)
    await hass.async_block_till_done()
    assert hass.states.get(anyone).state == "unavailable"
    assert hass.states.get(confidence).state == "unavailable"


# ---------------------------------------------------------------------------
# enabled switch (rule 7.2)
# ---------------------------------------------------------------------------


async def test_enabled_switch_mirrors_and_submits(hass: HomeAssistant, monkeypatch) -> None:
    entry, _controller, fake = await setup_conductor(hass, monkeypatch)
    switch = entity_id_for(hass, "switch", f"{entry.entry_id}_enabled")
    assert switch == "switch.presence_conductor_enabled"
    assert hass.states.get(switch).state == "on"

    await hass.services.async_call("switch", "turn_off", {"entity_id": switch}, blocking=True)
    await hass.async_block_till_done()
    assert fake.events_of(SetEnabled) == [SetEnabled(False)]
    # The switch is a control surface: it reflects "off" even though the
    # resulting plan suppressed the state signal.
    assert hass.states.get(switch).state == "off"

    await hass.services.async_call("switch", "turn_on", {"entity_id": switch}, blocking=True)
    await hass.async_block_till_done()
    assert fake.events_of(SetEnabled) == [SetEnabled(False), SetEnabled(True)]
    assert hass.states.get(switch).state == "on"


async def test_enabled_switch_restores_off(hass: HomeAssistant, monkeypatch) -> None:
    """A restored 'off' is pushed back into the engine as SetEnabled(False)."""
    mock_restore_cache(hass, (State("switch.presence_conductor_enabled", "off"),))
    _entry, _controller, fake = await setup_conductor(hass, monkeypatch)
    assert fake.events_of(SetEnabled) == [SetEnabled(False)]
    assert fake.state.enabled is False
    assert hass.states.get("switch.presence_conductor_enabled").state == "off"


async def test_enabled_switch_restore_matching_state_is_silent(
    hass: HomeAssistant, monkeypatch
) -> None:
    mock_restore_cache(hass, (State("switch.presence_conductor_enabled", "on"),))
    _entry, _controller, fake = await setup_conductor(hass, monkeypatch)
    assert fake.events_of(SetEnabled) == []


async def test_enabled_switch_ignores_invalid_restore(hass: HomeAssistant, monkeypatch) -> None:
    mock_restore_cache(hass, (State("switch.presence_conductor_enabled", "unavailable"),))
    _entry, _controller, fake = await setup_conductor(hass, monkeypatch)
    assert fake.events_of(SetEnabled) == []
    assert hass.states.get("switch.presence_conductor_enabled").state == "on"


# ---------------------------------------------------------------------------
# record-baseline button + service (rule 3.3)
# ---------------------------------------------------------------------------


async def test_record_baseline_button(hass: HomeAssistant, monkeypatch) -> None:
    entry, _controller, fake = await setup_conductor(hass, monkeypatch)
    button = entity_id_for(hass, "button", f"{entry.entry_id}_zone_sofakrok_record_baseline")
    assert button == "button.presence_conductor_sofakrok_record_baseline"

    await hass.services.async_call("button", "press", {"entity_id": button}, blocking=True)
    await hass.async_block_till_done()
    assert fake.events_of(RecordBaseline) == [RecordBaseline("sofakrok", None)]


async def test_record_baseline_service(hass: HomeAssistant, monkeypatch) -> None:
    _entry, _controller, fake = await setup_conductor(hass, monkeypatch)

    await hass.services.async_call(
        DOMAIN,
        "record_baseline",
        {"zone_id": "kontor_pult", "duration": 60},
        blocking=True,
    )
    await hass.async_block_till_done()
    assert fake.events_of(RecordBaseline) == [RecordBaseline("kontor_pult", 60.0)]

    # Duration is optional: the engine falls back to the tunable default.
    await hass.services.async_call(
        DOMAIN, "record_baseline", {"zone_id": "sofakrok"}, blocking=True
    )
    await hass.async_block_till_done()
    assert fake.events_of(RecordBaseline)[-1] == RecordBaseline("sofakrok", None)


async def test_record_baseline_service_rejects_unknown_zone(
    hass: HomeAssistant, monkeypatch
) -> None:
    _entry, _controller, fake = await setup_conductor(hass, monkeypatch)

    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN, "record_baseline", {"zone_id": "nope"}, blocking=True
        )
    assert fake.events_of(RecordBaseline) == []


# ---------------------------------------------------------------------------
# pass-by event entities
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("entity_registry_enabled_by_default")
async def test_pass_by_event_entities_exist_per_zone_and_room(
    hass: HomeAssistant, monkeypatch
) -> None:
    entry, _controller, _fake = await setup_conductor(hass, monkeypatch)
    unique_ids = [f"{entry.entry_id}_zone_{z}_pass_by" for z in ("kontor_pult", "kontor_dor")]
    unique_ids += [f"{entry.entry_id}_room_{r}_pass_by" for r in ("kontor", "stue")]
    for unique_id in unique_ids:
        event = entity_id_for(hass, "event", unique_id)
        state = hass.states.get(event)
        assert state.state == "unknown"  # nothing traversed yet
        assert state.attributes["event_types"] == ["pass_by"]


# ---------------------------------------------------------------------------
# diagnostics sensor
# ---------------------------------------------------------------------------


async def test_diagnostics_sensor(hass: HomeAssistant, monkeypatch) -> None:
    entry, controller, fake = await setup_conductor(hass, monkeypatch)
    sensor = entity_id_for(hass, "sensor", f"{entry.entry_id}_state")

    state = hass.states.get(sensor)
    assert state.state == "enabled"
    assert state.attributes["anyone_home"] is False
    sofakrok = state.attributes["zones"]["sofakrok"]
    assert sofakrok["health"] == "ok"
    assert sofakrok["activity"] == "empty"
    assert sofakrok["occupied"] is False
    assert sofakrok["calibration"] == "recalibration_required"
    assert state.attributes["rooms"]["stue"]["occupied"] is False
    assert state.attributes["rooms"]["stue"]["settled"] is False
    assert state.attributes["sensors"]["sofakrok_radar"] == {"available": True}
    # Recorder discipline: no per-frame numerics on the entity, and the
    # discrete summary duplicates dedicated entities — kept unrecorded.
    for volatile in ("home_lambda", "home_confidence", "enabled"):
        assert volatile not in state.attributes
    assert "lambda" not in sofakrok
    assert "dwell_seconds" not in sofakrok
    unrecorded = ConductorStateSensor._unrecorded_attributes
    assert unrecorded == frozenset({"anyone_home", "zones", "rooms", "sensors"})

    fake.state.enabled = False
    async_dispatcher_send(hass, controller.signal_control)
    await hass.async_block_till_done()
    assert hass.states.get(sensor).state == "disabled"


# ---------------------------------------------------------------------------
# entity inventory + unconfigured entry
# ---------------------------------------------------------------------------


async def test_entity_inventory_spans_hub_and_room_devices(
    hass: HomeAssistant, monkeypatch
) -> None:
    entry, _controller, _fake = await setup_conductor(hass, monkeypatch)
    registry = er.async_get(hass)
    entries = er.async_entries_for_config_entry(registry, entry.entry_id)
    # 3 zones x (occupancy, motion, activity, confidence, dwell, calibration
    # status, pass-by, calibration outcome, record baseline) + 2 rooms x (occupancy, motion,
    # settled, activity, confidence, pass-by) + anyone_home + home
    # confidence + enabled + state. Disabled-by-default zone entities are
    # registered like the rest.
    assert len(entries) == 3 * 9 + 2 * 6 + 4
    # Hub + one device per room; the layout itself is tests/test_devices.py.
    assert len({e.device_id for e in entries}) == 3


async def test_unconfigured_entry_creates_no_entities(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN, title="Presence Conductor", unique_id=DOMAIN, data={}, options={}
    )
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state.value == "loaded"
    assert hass.data[DOMAIN][entry.entry_id] is None

    registry = er.async_get(hass)
    assert er.async_entries_for_config_entry(registry, entry.entry_id) == []

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state.value == "not_loaded"
