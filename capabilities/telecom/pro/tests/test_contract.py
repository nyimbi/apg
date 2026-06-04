"""Tests for telecom_pro capability contract and service."""

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
	mod = _load("cc_pro", PACKAGE_DIR / "capability_contract.py")
	c = mod.get_capability_contract("t1")
	assert c["capability"] == "telecom_pro"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "8px"
	assert "service_activation_workflow" in c["provides"]
	assert "rollback_workflow" in c["provides"]
	assert len(c["ui"]["routes"]) >= 8
	assert len(c["rule_engine"]["rules"]) >= 20


def test_rule_engine():
	mod = _load("re_pro", PACKAGE_DIR / "capability_contract.py")
	assert mod.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "push_config", "push_method_supported": True, "ne_health_checked": True, "dry_run_bypassed": True})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "pro_batch", "event_stream": "kafka"})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True})["decision"] == "allow"


def test_provisioning_lifecycle():
	mod = _load("svc_pro", PACKAGE_DIR / "service.py")
	svc = mod.TelecomProService()

	workflow = svc.start_workflow("wf-1", "t1", "service_activation", "ORD-001", "2026-01-01T09:00:00")
	svc.update_workflow_status("wf-1", "t1", "in_progress")
	reservation = svc.reserve_resource("res-1", "t1", "wf-1", "msisdn", "+254700000001", "2026-01-01T09:01:00", "2026-01-01T09:31:00")
	push = svc.push_config("push-1", "t1", "wf-1", "HLR-01", "restconf", "template-001", "2026-01-01T09:05:00")
	activation = svc.confirm_activation("act-1", "t1", "wf-1", "SVC-001", "2026-01-01T09:10:00", "engineer-1")
	svc.update_workflow_status("wf-1", "t1", "completed", "2026-01-01T09:10:00")
	released = svc.release_resource("res-1", "t1")
	bulk = svc.start_bulk_provisioning("bulk-1", "t1", "sim_provisioning", 500, "bulk-approval-1", "batch-user", "2026-01-05")
	agent = svc.register_agent("agt-1", "t1", "PRO Agent", "codex", "workflow_orchestrator", "provisioning operations")
	batch = svc.validate_batch("t1", 5)
	summary = svc.dashboard_summary("t1")

	assert workflow["workflow_type"] == "service_activation"
	assert reservation["resource_type"] == "msisdn"
	assert push["push_method"] == "restconf"
	assert activation["status"] == "activated"
	assert released["released"] is True
	assert bulk["item_count"] == 500
	assert agent["role"] == "workflow_orchestrator"
	assert batch["processor"] == "bytewax"
	assert summary["activation_count"] == 1


def test_rollback_workflow():
	mod = _load("svc_rollback_pro", PACKAGE_DIR / "service.py")
	svc = mod.TelecomProService()

	svc.start_workflow("wf-r1", "t1", "service_activation", "ORD-FAIL-001", "2026-01-01")
	rollback = svc.trigger_rollback("rb-1", "t1", "wf-r1", "provisioning_failure", "NE config push failed", "2026-01-01T10:00:00")
	assert rollback["trigger"] == "provisioning_failure"
	completed = svc.complete_rollback("rb-1", "t1", "2026-01-01T10:15:00")
	assert completed["status"] == "completed"


def test_guardrails():
	mod = _load("svc_guard_pro", PACKAGE_DIR / "service.py")
	svc = mod.TelecomProService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.start_workflow("w", "", "service_activation", "ORD-001", "2026-01-01")
	with pytest.raises(PermissionError, match="workflow_type_not_supported"):
		svc.start_workflow("w", "t1", "quantum_provisioning", "ORD-001", "2026-01-01")
	with pytest.raises(PermissionError, match="order_reference_required"):
		svc.start_workflow("w", "t1", "service_activation", "", "2026-01-01")
	with pytest.raises(PermissionError, match="bulk_provisioning_approval_required"):
		svc.start_bulk_provisioning("b", "t1", "sim_provisioning", 100, "", "user", "2026-01-01")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 1, event_stream="redis")
	with pytest.raises(PermissionError, match="human_approval_required"):
		svc.validate_agent_action("t1", True, False)


def test_api_and_views():
	api = _load("api_pro", PACKAGE_DIR / "api.py")
	views = _load("views_pro", PACKAGE_DIR / "views.py")

	wf = api.start_workflow({"tenant_id": "t-api", "workflow_id": "wf-api", "workflow_type": "service_activation", "order_reference": "ORD-API-001"})
	res = api.reserve_resource({"tenant_id": "t-api", "reservation_id": "res-api", "workflow_id": "wf-api", "resource_type": "msisdn", "resource_value": "+254700000099"})
	batch = api.validate_batch({"tenant_id": "t-api", "item_count": 2})
	db = views.dashboard_model(api.service(), "t-api")
	resource_view = views.resource_console_model(api.service(), "t-api")

	assert wf["workflow_type"] == "service_activation"
	assert res["resource_type"] == "msisdn"
	assert batch["processor"] == "bytewax"
	assert db["summary"]["workflow_count"] == 1
	assert len(resource_view["active_reservations"]) == 1
