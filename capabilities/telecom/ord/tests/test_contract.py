"""Tests for telecom_ord capability contract and service."""

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
	mod = _load("cc_ord", PACKAGE_DIR / "capability_contract.py")
	c = mod.get_capability_contract("t1")
	assert c["capability"] == "telecom_ord"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "8px"
	assert "order_capture_workflow" in c["provides"]
	assert "fallout_management_workflow" in c["provides"]
	assert len(c["ui"]["routes"]) >= 8
	assert len(c["rule_engine"]["rules"]) >= 20


def test_rule_engine():
	mod = _load("re_ord", PACKAGE_DIR / "capability_contract.py")
	assert mod.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "submit_order", "order_type_supported": True, "is_duplicate": True})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "ord_batch", "event_stream": "sqs"})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True})["decision"] == "allow"


def test_order_lifecycle():
	mod = _load("svc_ord", PACKAGE_DIR / "service.py")
	svc = mod.TelecomOrdService()

	order = svc.submit_order("ord-1", "t1", "new_service", "cust-1", "web_self_service", "normal", "2026-01-01T10:00:00")
	validated = svc.validate_order("ord-1", "t1")
	decomposed = svc.decompose_order("ord-1", "t1")
	task1 = svc.create_task("tsk-1", "t1", "ord-1", "customer_verification")
	task2 = svc.create_task("tsk-2", "t1", "ord-1", "network_provisioning", depends_on="tsk-1")
	completed_task = svc.complete_task("tsk-1", "t1", "2026-01-01T10:05:00")
	completed_order = svc.complete_order("ord-1", "t1", "2026-01-01T10:30:00")
	port = svc.submit_portability_request("port-1", "t1", "ord-2", "+254700000001", "Safaricom", "Airtel", "2026-01-05")
	bulk = svc.submit_bulk_order("bulk-1", "t1", "new_service", 100, "bulk-approval-ref", "batch-user", "2026-01-05")
	agent = svc.register_agent("agt-1", "t1", "ORD Agent", "codex", "order_validator", "order management")
	batch = svc.validate_batch("t1", 5)
	summary = svc.dashboard_summary("t1")

	assert order["order_type"] == "new_service"
	assert validated["status"] == "validated"
	assert decomposed["status"] == "decomposed"
	assert task2["depends_on"] == "tsk-1"
	assert completed_task["status"] == "completed"
	assert completed_order["status"] == "completed"
	assert port["msisdn"] == "+254700000001"
	assert bulk["item_count"] == 100
	assert agent["role"] == "order_validator"
	assert batch["processor"] == "bytewax"
	assert summary["order_count"] == 1


def test_fallout_workflow():
	mod = _load("svc_fallout_ord", PACKAGE_DIR / "service.py")
	svc = mod.TelecomOrdService()

	order = svc.submit_order("ord-f1", "t1", "new_service", "cust-f1", "web_self_service", "high", "2026-01-01")
	svc.validate_order("ord-f1", "t1")
	svc.decompose_order("ord-f1", "t1")
	fallout = svc.record_fallout("fall-1", "t1", "ord-f1", "provisioning_failure", "NE timeout")
	assert fallout["fallout_category"] == "provisioning_failure"
	retried = svc.retry_fallout("fall-1", "t1")
	assert retried["retry_count"] == 1
	resolved = svc.resolve_fallout("fall-1", "t1", "Manual intervention applied", "2026-01-02")
	assert resolved["status"] == "resolved"


def test_guardrails():
	mod = _load("svc_guard_ord", PACKAGE_DIR / "service.py")
	svc = mod.TelecomOrdService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.submit_order("o", "", "new_service", "cust", "web_self_service", "normal", "2026-01-01")
	with pytest.raises(PermissionError, match="order_type_not_supported"):
		svc.submit_order("o", "t1", "quantum_service", "cust", "web_self_service", "normal", "2026-01-01")
	with pytest.raises(PermissionError, match="duplicate_order_detected"):
		svc.submit_order("o", "t1", "new_service", "cust", "web_self_service", "normal", "2026-01-01", is_duplicate=True)
	with pytest.raises(PermissionError, match="order_must_be_valid_for_decomposition"):
		svc.submit_order("o2", "t1", "new_service", "cust2", "call_centre", "normal", "2026-01-01")
		svc.decompose_order("o2", "t1")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 1, event_stream="rabbitmq")
	with pytest.raises(PermissionError, match="bulk_order_approval_required"):
		svc.submit_bulk_order("b", "t1", "new_service", 50, "", "user", "2026-01-01")


def test_api_and_views():
	api = _load("api_ord", PACKAGE_DIR / "api.py")
	views = _load("views_ord", PACKAGE_DIR / "views.py")

	order = api.submit_order({"tenant_id": "t-api", "order_id": "ord-api", "order_type": "new_service", "customer_id": "cust-api"})
	api.validate_order({"tenant_id": "t-api", "order_id": "ord-api"})
	api.decompose_order({"tenant_id": "t-api", "order_id": "ord-api"})
	task = api.create_task({"tenant_id": "t-api", "task_id": "tsk-api", "order_id": "ord-api", "task_type": "customer_verification"})
	batch = api.validate_batch({"tenant_id": "t-api", "item_count": 1})
	db = views.dashboard_model(api.service(), "t-api")
	task_queue = views.task_queue_model(api.service(), "t-api")

	assert order["order_type"] == "new_service"
	assert task["task_type"] == "customer_verification"
	assert batch["processor"] == "bytewax"
	assert db["summary"]["order_count"] == 1
	assert len(task_queue["queued_tasks"]) == 1
