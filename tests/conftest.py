"""Shared fixtures for Presence Conductor tests."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import PropertyMock, patch

import pytest


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Allow loading custom_components in every test."""
    yield


@pytest.fixture
def entity_registry_enabled_by_default() -> Generator[None]:
    """Register all entities enabled, overriding integration defaults.

    Zone state entities ship disabled (rooms and home are the consumer
    surface; zones are opt-in diagnostics — see tests/test_devices.py for
    the defaults themselves). Tests that exercise zone entity *behavior*
    opt in with this fixture, mirroring HA core's fixture of the same name.
    """
    with patch(
        "homeassistant.helpers.entity.Entity.entity_registry_enabled_default",
        new_callable=PropertyMock,
        return_value=True,
    ):
        yield
