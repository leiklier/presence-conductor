"""Calibration persistence, compatibility diagnostics, and Repairs state.

The core decides whether a calibration window commits. This adapter-side
manager owns everything that happens after that decision: serializing the
accepted state, retaining startup provenance, explaining safe fallbacks, and
maintaining the single Home Assistant Repairs warning for a config entry.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir

from .config import baselines_from_options
from .const import CONF_BASELINES, DOMAIN, GATE_MOVE_ROLES, GATE_STILL_ROLES
from .core import stats
from .core.model import ChannelStats, ConductorConfig, EngineState, ZoneBaselines
from .core.plan import BaselineRecorded

CALIBRATION_ISSUE_ID_PREFIX = "calibration_required"


def baseline_payload(event: BaselineRecorded) -> dict[str, Any]:
    """Return the JSON-safe HA representation of a calibration outcome."""
    return {
        "zone_id": event.zone_id,
        "success": event.success,
        "frame_count": event.frame_count,
        "move_mu": round(event.move_mu, 4),
        "move_sigma": round(event.move_sigma, 4),
        "still_mu": round(event.still_mu, 4),
        "still_sigma": round(event.still_sigma, 4),
        "coverage": {
            key: {
                "status": str(cov.status),
                "rows": cov.rows,
                "fresh": cov.fresh,
                "distinct": cov.distinct,
                "observed": cov.observed,
                **({"reason": cov.reason} if cov.reason else {}),
            }
            for key, cov in event.coverage.items()
        },
    }


class CalibrationEngine(Protocol):
    """Engine state needed by calibration persistence and diagnostics."""

    state: EngineState
    owned_gates: dict[str, tuple[int, ...]]


class CalibrationStatus(StrEnum):
    """Bounded operator-facing calibration readiness states."""

    READY = "ready"
    UNCALIBRATED = "uncalibrated"
    RECALIBRATION_REQUIRED = "recalibration_required"
    CALIBRATING = "calibrating"


@dataclass(frozen=True, slots=True)
class CalibrationDiagnostic:
    """Per-zone calibration provenance and active runtime paths."""

    status: CalibrationStatus
    reason_codes: tuple[str, ...]
    reasons: tuple[str, ...]
    floor_source: str
    move_statistic: str
    still_statistic: str
    move_runtime: str
    still_runtime: str

    @property
    def action(self) -> str | None:
        if self.status in (CalibrationStatus.READY, CalibrationStatus.CALIBRATING):
            return None
        return "Keep the zone empty and record a new baseline (default 300 seconds)."

    def provenance_attributes(self) -> dict[str, Any]:
        """Stable provenance for the calibration entity.

        Changes only on calibration lifecycle transitions — safe for the
        recorder. The per-frame runtime paths are deliberately absent:
        ``move_from_gates`` flips frame to frame, and every attribute
        change writes a recorder row.
        """
        return {
            "reason_codes": list(self.reason_codes),
            "reasons": list(self.reasons),
            "action": self.action,
            "floor_source": self.floor_source,
        }

    def attributes(self) -> dict[str, Any]:
        """Full JSON-safe dump, including the per-frame runtime paths —
        for the diagnostics download, never for entity attributes."""
        return {
            **self.provenance_attributes(),
            "move_statistic": self.move_statistic,
            "still_statistic": self.still_statistic,
            "move_runtime": self.move_runtime,
            "still_runtime": self.still_runtime,
        }


class CalibrationManager:
    """Own persisted calibration provenance and its operator-facing state."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        config: ConductorConfig,
        engine: CalibrationEngine,
        roles_by_sensor: Mapping[str, Mapping[str, str]],
        baselines: Mapping[str, ZoneBaselines],
    ) -> None:
        self._hass = hass
        self._entry = entry
        self._config = config
        self._engine = engine
        self._roles_by_sensor = roles_by_sensor
        # Preserve provenance before the core discards incompatible pieces.
        self._known_baselines = dict(baselines)
        self._diagnostics: dict[str, CalibrationDiagnostic] = {}
        self.refresh()

    @property
    def _issue_id(self) -> str:
        return f"{CALIBRATION_ISSUE_ID_PREFIX}_{self._entry.entry_id}"

    def diagnostic(self, zone_id: str) -> CalibrationDiagnostic:
        """Return current readiness plus the path actually used this frame."""
        diagnostic = self._diagnostics[zone_id]
        zst = self._engine.state.zones[zone_id]
        move_path = "gate" if zst.move_from_gates else "aggregate"
        still_path = "gate" if zst.still_from_gates else "aggregate"
        status = CalibrationStatus.CALIBRATING if zst.recording is not None else diagnostic.status
        return replace(
            diagnostic,
            status=status,
            move_runtime=move_path,
            still_runtime=still_path,
            move_statistic=(
                "empirical"
                if f"move_{'gate' if zst.move_from_gates else 'agg'}" in zst.stat_cal
                else "analytic"
            ),
            still_statistic=(
                "empirical"
                if f"still_{'gate' if zst.still_from_gates else 'agg'}" in zst.stat_cal
                else "analytic"
            ),
        )

    def _diagnose_zone(self, zone_id: str) -> CalibrationDiagnostic:
        zone = self._config.zone(zone_id)
        sensor = self._config.sensor(zone.sensor_id)
        zst = self._engine.state.zones[zone_id]
        persisted = self._known_baselines.get(zone_id)
        owned = self._engine.owned_gates[zone_id]
        roles = self._roles_by_sensor.get(zone.sensor_id, {})
        reasons: list[str] = []
        reason_codes: list[str] = []

        def add(code: str, reason: str) -> None:
            if code not in reason_codes:
                reason_codes.append(code)
                reasons.append(reason)

        floor_source = "default"
        context_valid = False
        if persisted is None:
            add("no_recorded_baseline", "No recorded baseline is available.")
        elif persisted.sensor_id not in (None, zone.sensor_id):
            add(
                "sensor_changed",
                f"Sensor changed from {persisted.sensor_id} to {zone.sensor_id}.",
            )
        elif (
            persisted.floor_fingerprint is not None
            and persisted.floor_fingerprint
            != stats.floor_calibration_fingerprint(self._config.tunables)
        ):
            add(
                "floor_settings_changed",
                "The empty-channel floor-fit settings changed after this baseline was recorded.",
            )
        else:
            context_valid = True
            floor_source = "recorded"
            if persisted.sensor_id is None or persisted.floor_fingerprint is None:
                add(
                    "legacy_context",
                    "The recorded baseline predates calibration compatibility metadata.",
                )

        move_gates_configured = any(role in roles for role in GATE_MOVE_ROLES)
        still_gates_configured = any(role in roles for role in GATE_STILL_ROLES)
        gate_requested = self._config.tunables.use_gate_evidence and bool(owned)
        if gate_requested:
            if not move_gates_configured and not still_gates_configured:
                add(
                    "gate_entities_missing",
                    "Gate evidence is enabled, but no gate energy entities are configured.",
                )
            if context_valid and persisted is not None:
                if (
                    persisted.gate_size_cm is not None
                    and persisted.gate_size_cm != sensor.gate_size_cm
                ):
                    add(
                        "gate_resolution_changed",
                        "Gate resolution changed from "
                        f"{persisted.gate_size_cm:g} cm to {sensor.gate_size_cm:g} cm.",
                    )
                if persisted.gate_indices is not None and persisted.gate_indices != owned:
                    add(
                        "gate_family_changed",
                        "Owned gates changed from "
                        f"{list(persisted.gate_indices)} to {list(owned)}.",
                    )
            if move_gates_configured and not zst.gate_move_ready:
                add(
                    "gate_move_calibration_missing",
                    "Move-gate calibration is missing, incomplete, or incompatible.",
                )
            if still_gates_configured and not zst.gate_still_ready:
                add(
                    "gate_still_calibration_missing",
                    "Still-gate calibration is missing, incomplete, or incompatible.",
                )

        active_stat_paths = ["move_agg", "still_agg"]
        if gate_requested and move_gates_configured:
            active_stat_paths.append("move_gate")
        if gate_requested and still_gates_configured:
            active_stat_paths.append("still_gate")
        if context_valid and persisted is not None:
            for path in active_stat_paths:
                cal = persisted.stats.get(path)
                if cal is None:
                    continue
                expected = stats.calibration_fingerprint(path, owned, self._config.tunables)
                if cal.fingerprint != expected:
                    add(
                        "statistic_context_changed",
                        f"The saved {path} statistic is incompatible with current settings.",
                    )

        move_runtime = "gate" if zst.move_from_gates else "aggregate"
        still_runtime = "gate" if zst.still_from_gates else "aggregate"
        move_stat_path = "move_gate" if move_runtime == "gate" else "move_agg"
        still_stat_path = "still_gate" if still_runtime == "gate" else "still_agg"
        status = (
            CalibrationStatus.UNCALIBRATED
            if persisted is None
            else (CalibrationStatus.RECALIBRATION_REQUIRED if reasons else CalibrationStatus.READY)
        )
        return CalibrationDiagnostic(
            status=status,
            reason_codes=tuple(reason_codes),
            reasons=tuple(reasons),
            floor_source=floor_source,
            move_statistic="empirical" if move_stat_path in zst.stat_cal else "analytic",
            still_statistic="empirical" if still_stat_path in zst.stat_cal else "analytic",
            move_runtime=move_runtime,
            still_runtime=still_runtime,
        )

    @callback
    def refresh(self, zone_ids: set[str] | None = None) -> None:
        """Re-evaluate bounded per-zone readiness from known provenance."""
        selected = {zone.zone_id for zone in self._config.zones} if zone_ids is None else zone_ids
        for zone_id in selected:
            if self._config.zone_or_none(zone_id) is not None:
                self._diagnostics[zone_id] = self._diagnose_zone(zone_id)

    @callback
    def sync_issue(self) -> None:
        """Maintain one nonpersistent Repairs warning for this entry."""
        failing = [
            (zone, self._diagnostics[zone.zone_id])
            for zone in self._config.zones
            if self._diagnostics[zone.zone_id].status is not CalibrationStatus.READY
        ]
        if not failing:
            ir.async_delete_issue(self._hass, DOMAIN, self._issue_id)
            return
        details = " | ".join(
            f"{zone.name}: {'; '.join(diagnostic.reasons)}" for zone, diagnostic in failing
        )
        ir.async_create_issue(
            self._hass,
            DOMAIN,
            self._issue_id,
            is_fixable=False,
            is_persistent=False,
            severity=ir.IssueSeverity.WARNING,
            translation_key="calibration_required",
            translation_placeholders={"details": details},
        )

    @callback
    def clear_issue(self) -> None:
        """Remove this entry's warning after a successful unload."""
        ir.async_delete_issue(self._hass, DOMAIN, self._issue_id)

    @callback
    def persist(self, zone_ids: set[str]) -> None:
        """Persist committed zones, then refresh provenance and Repairs."""
        stored: dict[str, Any] = dict(self._entry.options.get(CONF_BASELINES) or {})
        t = self._config.tunables
        default = ChannelStats(t.default_mu, t.default_sigma)
        current: dict[str, Any] = {}
        for zone in self._config.zones:
            if zone.zone_id not in zone_ids:
                continue
            zst = self._engine.state.zones.get(zone.zone_id)
            if zst is None:
                continue
            record: dict[str, Any] = {
                "move_mu": zst.move_baseline.mu,
                "move_sigma": zst.move_baseline.sigma,
                "still_mu": zst.still_baseline.mu,
                "still_sigma": zst.still_baseline.sigma,
                "sensor_id": zone.sensor_id,
                "floor_fingerprint": stats.floor_calibration_fingerprint(t),
                "gate_size_cm": self._config.sensor(zone.sensor_id).gate_size_cm,
            }
            move_gates = zst.gate_move_baselines if zst.gate_move_ready else {}
            still_gates = zst.gate_still_baselines if zst.gate_still_ready else {}
            gates = {
                str(index): {
                    "move_mu": (gm := move_gates.get(index, default)).mu,
                    "move_sigma": gm.sigma,
                    "still_mu": (gs := still_gates.get(index, default)).mu,
                    "still_sigma": gs.sigma,
                    "has_move": index in move_gates,
                    "has_still": index in still_gates,
                }
                for index in sorted(move_gates.keys() | still_gates)
            }
            if gates:
                record["gates"] = gates
                record["gate_indices"] = list(self._engine.owned_gates[zone.zone_id])
            statistic_records = {
                key: {
                    "mu": cal.mu,
                    "sigma": cal.sigma,
                    "clip_mu": cal.clip_mu,
                    "tau": cal.tau,
                    "decorrelation_seconds": cal.decorrelation_seconds,
                    "fingerprint": cal.fingerprint,
                }
                for key, cal in sorted(zst.stat_cal.items())
                if cal.fingerprint
            }
            if statistic_records:
                record["stats"] = statistic_records
            current[zone.zone_id] = record

        merged = {**stored, **current}
        if merged != stored:
            # The options listener in __init__.py ignores baselines-only
            # diffs, so persisting calibration cannot cause a reload loop.
            self._hass.config_entries.async_update_entry(
                self._entry, options={**self._entry.options, CONF_BASELINES: merged}
            )
        parsed = baselines_from_options({CONF_BASELINES: merged})
        for zone_id in zone_ids:
            if (baseline := parsed.get(zone_id)) is not None:
                self._known_baselines[zone_id] = baseline
        self.refresh(zone_ids)
        self.sync_issue()
