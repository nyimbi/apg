"""Tests for telecom_net capability contract and service."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def test_contract_shape():
	mod = _load("cc_net", PACKAGE_DIR / "capability_contract.py")
	c = mod.get_capability_contract("t1")
	assert c["capability"] == "telecom_net"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "6px"
	assert "fault_management_workflow" in c["provides"]
	assert "sla_monitoring_workflow" in c["provides"]
	assert len(c["ui"]["routes"]) >= 8
	assert len(c["rule_engine"]["rules"]) >= 20


def test_rule_engine():
	mod = _load("re_net", PACKAGE_DIR / "capability_contract.py")
	assert mod.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "suppress_alarm", "approval_present": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "submit_config_change", "change_type_supported": True, "approval_present": False, "in_freeze_period": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "net_batch", "event_stream": "kafka"})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True})["decision"] == "allow"


def test_network_management_lifecycle():
	mod = _load("svc_net", PACKAGE_DIR / "service.py")
	svc = mod.TelecomNetService()

	alarm = svc.raise_alarm("alm-1", "t1", "BTS-001", "critical", "hardware_failure", "Power unit failed", "2026-01-01T08:00:00")
	acknowledged = svc.update_alarm_status("alm-1", "t1", "acknowledged")
	ticket = svc.open_fault_ticket("tkt-1", "t1", alarm["id"], "Power Failure BTS-001", "critical", "tier1")
	escalated = svc.escalate_fault("tkt-1", "t1", "tier2")
	perf = svc.record_performance("perf-1", "t1", "BTS-001", "availability", 95.5, 99.9, "ran", "2026-01-01T08:00:00")
	change = svc.submit_config_change("chg-1", "t1", "BTS-001", "parameter_change", "Increase tx power", "approval-ref-1", "engineer-1", "2026-01-01T09:00:00")
	completed_change = svc.complete_config_change("chg-1", "t1")
	sla = svc.record_sla("sla-1", "t1", "availability", "cust-enterprise-1", 99.9, 95.5, "2026-01")
	handover = svc.record_noc_handover("ho-1", "t1", "night", "op-alice", "op-bob", "3 critical alarms open", 3, "2026-01-01T22:00:00")
	resolved = svc.resolve_fault_ticket("tkt-1", "t1", "2026-01-01T14:00:00")
	cleared = svc.update_alarm_status("alm-1", "t1", "cleared", "2026-01-01T14:00:00")
	agent = svc.register_agent("agt-1", "t1", "NET Agent", "codex", "fault_analyst", "fault management")
	batch = svc.validate_batch("t1", 5)
	summary = svc.dashboard_summary("t1")

	assert alarm["severity"] == "critical"
	assert acknowledged["status"] == "acknowledged"
	assert escalated["escalation_level"] == "tier2"
	assert perf["metric_type"] == "availability"
	assert completed_change["status"] == "completed"
	assert sla["status"] == "breached"
	assert handover["shift"] == "night"
	assert resolved["status"] == "resolved"
	assert cleared["status"] == "cleared"
	assert batch["processor"] == "bytewax"
	assert summary["alarm_count"] == 1
	assert summary["audit_event_count"] >= 10


def test_freeze_period_blocks_change():
	mod = _load("svc_freeze_net", PACKAGE_DIR / "service.py")
	svc = mod.TelecomNetService()

	with pytest.raises(PermissionError, match="change_freeze_period_active"):
		svc.submit_config_change("chg-freeze", "t1", "BTS-001", "parameter_change", "Change in freeze", "approval-ref", "engineer", "2026-12-25T10:00:00", in_freeze_period=True)


def test_guardrails():
	mod = _load("svc_guard_net", PACKAGE_DIR / "service.py")
	svc = mod.TelecomNetService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.raise_alarm("a", "", "ne-1", "critical", "hardware_failure", "desc", "2026-01-01")
	with pytest.raises(PermissionError, match="fault_severity_not_supported"):
		svc.raise_alarm("a", "t1", "ne-1", "apocalyptic", "hardware_failure", "desc", "2026-01-01")
	with pytest.raises(PermissionError, match="config_change_approval_required"):
		svc.submit_config_change("c", "t1", "ne-1", "parameter_change", "desc", "", "engineer", "2026-01-01")
	with pytest.raises(PermissionError, match="alarm_suppression_approval_required"):
		svc.suppress_alarm("alm-x", "t1", "")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 1, event_stream="amqp")


def test_api_and_views():
	api = _load("api_net", PACKAGE_DIR / "api.py")
	views = _load("views_net", PACKAGE_DIR / "views.py")

	alarm = api.raise_alarm({"tenant_id": "t-api", "alarm_id": "alm-api", "ne_reference": "BTS-API-001", "severity": "major", "category": "link_down", "description": "Link down"})
	ticket = api.open_fault_ticket({"tenant_id": "t-api", "ticket_id": "tkt-api", "alarm_id": alarm["id"], "title": "Link Down BTS-API", "severity": "major"})
	batch = api.validate_batch({"tenant_id": "t-api", "item_count": 3})
	db = views.dashboard_model(api.service(), "t-api")
	alarm_view = views.alarm_console_model(api.service(), "t-api")
	noc = views.noc_view_model(api.service(), "t-api")

	assert alarm["severity"] == "major"
	assert ticket["alarm_id"] == alarm["id"]
	assert batch["processor"] == "bytewax"
	assert db["summary"]["alarm_count"] == 1
	assert len(alarm_view["active_alarms"]) == 1
	assert noc["active_alarms_count"] == 1
