"""Tests for telecom_qos capability contract and service."""

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
	mod = _load("cc_qos", PACKAGE_DIR / "capability_contract.py")
	c = mod.get_capability_contract("t1")
	assert c["capability"] == "telecom_qos"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "8px"
	assert "qos_policy_management_workflow" in c["provides"]
	assert "degradation_detection_workflow" in c["provides"]
	assert len(c["ui"]["routes"]) >= 8
	assert len(c["rule_engine"]["rules"]) >= 20


def test_rule_engine():
	mod = _load("re_qos", PACKAGE_DIR / "capability_contract.py")
	assert mod.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "create_qos_policy", "policy_type_supported": True, "qos_class_supported": True, "approval_present": False, "conflict_checked": True})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "qos_batch", "event_stream": "kafka"})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "change_qos_policy", "is_downgrade": True, "approval_present": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True})["decision"] == "allow"


def test_qos_lifecycle():
	mod = _load("svc_qos", PACKAGE_DIR / "service.py")
	svc = mod.TelecomQosService()

	policy = svc.create_qos_policy("pol-1", "t1", "bearer_qos", "conversational", "VoLTE Bearer", '{"gbr": "128kbps"}', "approval-ref-1", "engineer")
	classification = svc.classify_traffic("cls-1", "t1", "voice", "VoLTE", policy["id"], "flow-ref-1", "2026-01-01T10:00:00")
	enforcement = svc.update_enforcement_status("enf-1", "t1", policy["id"], "PCRF-01", "active", "2026-01-01T10:01:00", "2026-01-01T10:01:00")
	sla = svc.record_sla_measurement("sla-1", "t1", "max_latency_ms", 15.0, 20.0, "cust-1", "2026-01-01T10:05:00")
	degradation = svc.record_degradation("deg-1", "t1", "congestion", 0.91, "Cell congestion detected", "BTS-001", "evidence-ref", "2026-01-01T10:10:00")
	rca = svc.record_root_cause("rca-1", "t1", degradation["id"], "Heavy traffic on 1800MHz band during peak hour", 0.88, "pcap-evidence", "2026-01-01T10:15:00")
	remediation = svc.trigger_remediation("rem-1", "t1", degradation["id"], "load_balancing", False, None, "2026-01-01T10:16:00")
	completed = svc.complete_remediation("rem-1", "t1", "2026-01-01T10:20:00")
	agent = svc.register_agent("agt-1", "t1", "QOS Agent", "codex", "qos_policy_manager", "qos management")
	batch = svc.validate_batch("t1", 5)
	summary = svc.dashboard_summary("t1")

	assert policy["qos_class"] == "conversational"
	assert classification["traffic_type"] == "voice"
	assert enforcement["status"] == "active"
	assert sla["is_breach"] is False
	assert degradation["cause"] == "congestion"
	assert rca["confidence_score"] == 0.88
	assert completed["status"] == "completed"
	assert agent["role"] == "qos_policy_manager"
	assert batch["processor"] == "bytewax"
	assert summary["open_degradation_count"] == 0


def test_sla_breach_detection():
	mod = _load("svc_sla_qos", PACKAGE_DIR / "service.py")
	svc = mod.TelecomQosService()
	sla = svc.record_sla_measurement("sla-breach", "t1", "max_latency_ms", 45.0, 20.0, None, "2026-01-01")
	assert sla["is_breach"] is True


def test_guardrails():
	mod = _load("svc_guard_qos", PACKAGE_DIR / "service.py")
	svc = mod.TelecomQosService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.create_qos_policy("p", "", "bearer_qos", "conversational", "name", "{}", "approval", "engineer")
	with pytest.raises(PermissionError, match="qos_policy_type_not_supported"):
		svc.create_qos_policy("p", "t1", "magic_qos", "conversational", "name", "{}", "approval", "engineer")
	with pytest.raises(PermissionError, match="qos_policy_approval_required"):
		svc.create_qos_policy("p", "t1", "bearer_qos", "streaming", "name", "{}", "", "engineer")
	with pytest.raises(PermissionError, match="degradation_cause_not_supported"):
		svc.record_degradation("d", "t1", "alien_signal", 0.9, "desc", "res", "ev", "2026-01-01")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 1, event_stream="amqp")
	with pytest.raises(PermissionError, match="cross_tenant_qos_denied"):
		svc.validate_agent_action("t1", False, False, cross_tenant_qos_scope=True)


def test_api_and_views():
	api = _load("api_qos", PACKAGE_DIR / "api.py")
	views = _load("views_qos", PACKAGE_DIR / "views.py")

	policy = api.create_qos_policy({"tenant_id": "t-api", "policy_id": "pol-api", "policy_type": "bearer_qos", "qos_class": "streaming", "name": "Video QoS", "approval_reference": "approval-api", "created_by": "engineer"})
	degradation = api.record_degradation({"tenant_id": "t-api", "degradation_id": "deg-api", "cause": "congestion", "confidence_score": 0.85, "description": "Cell congestion", "affected_resource": "BTS-API", "evidence_reference": "ev-ref"})
	batch = api.validate_batch({"tenant_id": "t-api", "item_count": 2})
	db = views.dashboard_model(api.service(), "t-api")
	deg_view = views.degradation_console_model(api.service(), "t-api")

	assert policy["qos_class"] == "streaming"
	assert degradation["cause"] == "congestion"
	assert batch["processor"] == "bytewax"
	assert db["summary"]["policy_count"] == 1
	assert len(deg_view["open_degradations"]) == 1
