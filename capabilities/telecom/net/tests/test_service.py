"""Service-level tests for telecom_net."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name, path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def test_describe_returns_contract():
	mod = _load("svc_desc_net", PACKAGE_DIR / "service.py")
	svc = mod.TelecomNetService()
	c = svc.describe("t1")
	assert c["capability"] == "telecom_net"


def test_alarm_lifecycle_full():
	mod = _load("svc_alm_net", PACKAGE_DIR / "service.py")
	svc = mod.TelecomNetService()
	alarm = svc.raise_alarm("a1", "t1", "BTS-X", "major", "link_down", "desc", "2026-01-01")
	assert alarm["status"] == "raised"
	svc.update_alarm_status("a1", "t1", "acknowledged")
	cleared = svc.update_alarm_status("a1", "t1", "cleared", "2026-01-01T12:00:00")
	assert cleared["cleared_at"] == "2026-01-01T12:00:00"


def test_performance_threshold_audit():
	mod = _load("svc_perf_net", PACKAGE_DIR / "service.py")
	svc = mod.TelecomNetService()
	svc.record_performance("p1", "t1", "ne-1", "availability", 105.0, 100.0, "core", "2026-01-01")
	assert any(e["event_type"] == "performance_threshold_breached" for e in svc.audit_events)


def test_sla_compliant_not_audited_as_breach():
	mod = _load("svc_sla_net", PACKAGE_DIR / "service.py")
	svc = mod.TelecomNetService()
	sla = svc.record_sla("sla-ok", "t1", "availability", None, 99.0, 99.5, "2026-01")
	assert sla["status"] == "compliant"
	assert not any(e["event_type"] == "sla_breach_detected" for e in svc.audit_events)


def test_noc_handover_requires_notes():
	mod = _load("svc_noc_net", PACKAGE_DIR / "service.py")
	svc = mod.TelecomNetService()
	with pytest.raises(PermissionError, match="handover_notes_required"):
		svc.record_noc_handover("ho", "t1", "night", "alice", "bob", "", 2, "2026-01-01T22:00:00")


def test_config_change_requires_ne_reference():
	mod = _load("svc_chg_net", PACKAGE_DIR / "service.py")
	svc = mod.TelecomNetService()
	with pytest.raises(PermissionError, match="ne_reference_required"):
		svc.raise_alarm("a", "t1", "", "minor", "clock_failure", "desc", "2026-01-01")


def test_multi_tenant_alarm_isolation():
	mod = _load("svc_iso_net", PACKAGE_DIR / "service.py")
	svc = mod.TelecomNetService()
	svc.raise_alarm("a1", "tenant-a", "BTS-A", "critical", "hardware_failure", "desc", "2026-01-01")
	svc.raise_alarm("a1", "tenant-b", "BTS-B", "major", "link_down", "desc", "2026-01-01")
	assert svc.dashboard_summary("tenant-a")["alarm_count"] == 1
	assert svc.dashboard_summary("tenant-b")["alarm_count"] == 1
	assert svc.alarms[("tenant-a", "a1")].severity == "critical"
	assert svc.alarms[("tenant-b", "a1")].severity == "major"


def test_escalation_level_must_be_supported():
	mod = _load("svc_esc_net", PACKAGE_DIR / "service.py")
	svc = mod.TelecomNetService()
	svc.raise_alarm("a1", "t1", "BTS-001", "critical", "hardware_failure", "desc", "2026-01-01")
	svc.open_fault_ticket("tkt-1", "t1", "a1", "Title", "critical", "tier1")
	with pytest.raises(PermissionError, match="escalation_level_not_supported"):
		svc.escalate_fault("tkt-1", "t1", "tier99")


def test_agent_registration_validates_role():
	mod = _load("svc_agt_net", PACKAGE_DIR / "service.py")
	svc = mod.TelecomNetService()
	with pytest.raises(PermissionError, match="net_agent_role_not_supported"):
		svc.register_agent("agt", "t1", "Agent", "codex", "pizza_deliverer", "operations")


def test_validate_batch_positive_only():
	mod = _load("svc_batch_net", PACKAGE_DIR / "service.py")
	svc = mod.TelecomNetService()
	with pytest.raises(ValueError):
		svc.validate_batch("t1", -1)
