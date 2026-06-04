"""Tests for telecom_sec capability contract and service."""

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
	mod = _load("cc_sec", PACKAGE_DIR / "capability_contract.py")
	c = mod.get_capability_contract("t1")
	assert c["capability"] == "telecom_sec"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "6px"
	assert "fraud_management_workflow" in c["provides"]
	assert "lawful_intercept_workflow" in c["provides"]
	assert len(c["ui"]["routes"]) >= 8
	assert len(c["rule_engine"]["rules"]) >= 20
	assert "comp" in c["requires"]


def test_rule_engine():
	mod = _load("re_sec", PACKAGE_DIR / "capability_contract.py")
	assert mod.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "activate_intercept", "warrant_present": False, "regulatory_authority_present": True, "intercept_type_supported": True})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "apply_fraud_block", "evidence_present": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "sec_batch", "event_stream": "kafka"})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True})["decision"] == "allow"


def test_security_lifecycle():
	mod = _load("svc_sec", PACKAGE_DIR / "service.py")
	svc = mod.TelecomSecService()

	fraud = svc.raise_fraud_case("fraud-1", "t1", "wangiri", "+254700000001", 0.95, "cdr-evidence-ref", "2026-01-01T08:00:00")
	blocked = svc.apply_fraud_block("fraud-1", "t1", "cdr-evidence-ref")
	ss7 = svc.record_ss7_attack("ss7-1", "t1", "location_tracking", "SS7-PROBE-001", "+254700000001", "pcap-ref", "2026-01-01T09:00:00")
	diameter = svc.record_diameter_attack("dia-1", "t1", "identity_spoofing", "malicious.realm.com", "operator.ke", "pcap-ref-2", "2026-01-01T09:30:00")
	intercept = svc.activate_intercept("int-1", "t1", "voice_call", "+254700000002", "WARRANT-2026-001", "DCI Kenya", "2026-01-10T00:00:00", "2026-04-10T00:00:00")
	suspended = svc.update_intercept_status("int-1", "t1", "suspended")
	incident = svc.open_incident("inc-1", "t1", "fraud_detection", "critical", "Large scale WANGIRI campaign", "fraud-evidence-ref", "2026-01-01T10:00:00")
	contained = svc.update_incident_status("inc-1", "t1", "contained")
	intel = svc.record_threat_intel("ti-1", "t1", "gsma_fraud_forum", "msisdn", "+254700000001", "amber", "2026-01-01", "2026-07-01", True)
	agent = svc.register_agent("agt-1", "t1", "SEC Agent", "codex", "fraud_analyst", "fraud management")
	batch = svc.validate_batch("t1", 5)
	summary = svc.dashboard_summary("t1")

	assert fraud["fraud_type"] == "wangiri"
	assert blocked["status"] == "blocked"
	assert ss7["attack_type"] == "location_tracking"
	assert diameter["attack_type"] == "identity_spoofing"
	assert intercept["status"] == "active"
	assert suspended["status"] == "suspended"
	assert incident["severity"] == "critical"
	assert contained["status"] == "contained"
	assert intel["shared"] is True
	assert agent["role"] == "fraud_analyst"
	assert batch["processor"] == "bytewax"
	assert summary["fraud_case_count"] == 1
	assert summary["active_intercept_count"] == 0


def test_intercept_without_warrant_denied():
	mod = _load("svc_warrant_sec", PACKAGE_DIR / "service.py")
	svc = mod.TelecomSecService()
	with pytest.raises(PermissionError, match="intercept_warrant_required"):
		svc.activate_intercept("int-x", "t1", "voice_call", "+254700000003", "", "DCI Kenya", "2026-01-01", "2026-04-01")


def test_guardrails():
	mod = _load("svc_guard_sec", PACKAGE_DIR / "service.py")
	svc = mod.TelecomSecService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.raise_fraud_case("f", "", "wangiri", "+254", 0.9, "ev", "2026-01-01")
	with pytest.raises(PermissionError, match="fraud_type_not_supported"):
		svc.raise_fraud_case("f", "t1", "unicorn_fraud", "+254700000001", 0.9, "ev", "2026-01-01")
	with pytest.raises(PermissionError, match="fraud_block_requires_evidence"):
		svc.raise_fraud_case("f2", "t1", "wangiri", "+254700000001", 0.9, "ev-2", "2026-01-01")
		svc.apply_fraud_block("f2", "t1", "")
	with pytest.raises(PermissionError, match="incident_severity_not_supported"):
		svc.open_incident("i", "t1", "fraud_detection", "catastrophic", "desc", "ev", "2026-01-01")
	with pytest.raises(PermissionError, match="evidence_fabrication_denied"):
		svc.validate_agent_action("t1", False, False, evidence_fabrication_scope=True)
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 1, event_stream="rabbitmq")


def test_api_and_views():
	api = _load("api_sec", PACKAGE_DIR / "api.py")
	views = _load("views_sec", PACKAGE_DIR / "views.py")

	fraud = api.raise_fraud_case({"tenant_id": "t-api", "case_id": "fraud-api", "fraud_type": "sim_swap_fraud", "msisdn": "+254700000099", "confidence_score": 0.92, "evidence_reference": "ev-api"})
	incident = api.open_incident({"tenant_id": "t-api", "incident_id": "inc-api", "incident_type": "fraud_detection", "severity": "major", "description": "SIM swap campaign", "evidence_reference": "inc-ev-api"})
	intel = api.record_threat_intel({"tenant_id": "t-api", "intel_id": "ti-api", "source": "internal", "ioc_type": "msisdn", "ioc_value": "+254700000099", "tlp_level": "red", "valid_from": "2026-01-01", "shared": False})
	batch = api.validate_batch({"tenant_id": "t-api", "item_count": 3})
	db = views.dashboard_model(api.service(), "t-api")
	fraud_view = views.fraud_console_model(api.service(), "t-api")
	incident_view = views.incident_queue_model(api.service(), "t-api")

	assert fraud["fraud_type"] == "sim_swap_fraud"
	assert incident["severity"] == "major"
	assert intel["source"] == "internal"
	assert batch["processor"] == "bytewax"
	assert db["summary"]["fraud_case_count"] == 1
	assert len(fraud_view["open_fraud_cases"]) == 1
	assert len(incident_view["open_incidents"]) == 1
