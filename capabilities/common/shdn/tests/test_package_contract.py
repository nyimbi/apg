"""SHDN package contract and deterministic runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.shdn import api, views
from capabilities.common.shdn.service import ShdnService


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_is_valid():
	module = _load_module("materialized_contract_shdn", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "shdn"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_app_entrypoint_is_publishable():
	module = _load_module("materialized_app_shdn", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "shdn" in model["capabilities"]


def test_shutdown_lifecycle_executes_with_recovery_evidence():
	service = ShdnService()

	target = service.register_service(
		tenant_id="tenant-a",
		target_id="billing-api",
		target_type="service",
		owner="platform-owner",
		environment="production",
		dependencies=["payments-db", "invoice-queue"],
		criticality="critical",
		health_gate_ref="health://billing-api/pre",
	)
	plan = service.create_shutdown_plan(
		tenant_id="tenant-a",
		name="Billing maintenance",
		owner="platform-owner",
		target_ids=[target["id"]],
		reason="Patch database driver",
		rollback_plan_ref="runbook://rollback/billing",
		restart_sequence=["payments-db", "invoice-queue", "billing-api"],
		approved_by="ops-director",
		maintenance_window_ref="window://mw-2026-05-29",
	)
	drain = service.start_drain(
		tenant_id="tenant-a",
		plan_id=plan["id"],
		target_id=target["id"],
		active_sessions=0,
		queue_depth=0,
	)
	snapshot = service.record_backup_snapshot(
		tenant_id="tenant-a",
		plan_id=plan["id"],
		target_id=target["id"],
		evidence_ref="backup://billing-api/1",
		restore_test_ref="restore-test://billing-api/1",
	)
	execution = service.execute_shutdown(
		tenant_id="tenant-a",
		plan_id=plan["id"],
		target_id=target["id"],
		actor="operator-1",
		health_gate_ref="health://billing-api/pre",
	)
	recovery = service.record_recovery(
		tenant_id="tenant-a",
		plan_id=plan["id"],
		target_id=target["id"],
		actor="operator-1",
		evidence_ref="incident://change/123",
		post_shutdown_health_check_ref="health://billing-api/post",
	)
	summary = service.dashboard_summary("tenant-a")

	assert target["criticality"] == "critical"
	assert plan["status"] == "approved"
	assert drain["status"] == "quiesced"
	assert snapshot["verified"] is True
	assert execution["status"] == "completed"
	assert recovery["status"] == "recovered"
	assert summary["target_count"] == 1
	assert summary["snapshot_count"] == 1
	assert summary["shutdown_count"] == 0
	assert summary["recovery_count"] == 1


def test_guardrails_require_tenant_owner_plan_evidence_drain_and_recovery_evidence():
	service = ShdnService()

	try:
		service.register_service("", "no-tenant", "service", "owner")
	except PermissionError as exc:
		assert str(exc) == "tenant_context_required"
	else:
		raise AssertionError("missing tenant was accepted")

	try:
		service.register_service("tenant-a", "no-owner", "service", "")
	except PermissionError as exc:
		assert str(exc) == "service_owner_required"
	else:
		raise AssertionError("missing owner was accepted")

	target = service.register_service("tenant-a", "orders-api", "service", "owner", criticality="critical")

	try:
		service.create_shutdown_plan(
			tenant_id="tenant-a",
			name="No approval",
			owner="owner",
			target_ids=[target["id"]],
			reason="maintenance",
			rollback_plan_ref="rollback://orders",
			restart_sequence=["orders-api"],
			approved_by=None,
			maintenance_window_ref="window://mw",
		)
	except PermissionError as exc:
		assert str(exc) == "production_approval_required"
	else:
		raise AssertionError("unapproved production shutdown was accepted")

	plan = service.create_shutdown_plan(
		tenant_id="tenant-a",
		name="Approved",
		owner="owner",
		target_ids=[target["id"]],
		reason="maintenance",
		rollback_plan_ref="rollback://orders",
		restart_sequence=["orders-api"],
		approved_by="approver",
		maintenance_window_ref="window://mw",
	)

	try:
		service.execute_shutdown("tenant-a", plan["id"], target["id"], "operator", "health://pre")
	except PermissionError as exc:
		assert str(exc) == "drain_not_recorded"
	else:
		raise AssertionError("shutdown without drain was accepted")

	service.start_drain("tenant-a", plan["id"], target["id"], active_sessions=1, queue_depth=0)
	service.record_backup_snapshot("tenant-a", plan["id"], target["id"], "backup://orders", "restore://orders")

	try:
		service.execute_shutdown("tenant-a", plan["id"], target["id"], "operator", "health://pre")
	except PermissionError as exc:
		assert str(exc) == "drain_not_quiesced"
	else:
		raise AssertionError("shutdown with active sessions was accepted")

	service.start_drain("tenant-a", plan["id"], target["id"], active_sessions=0, queue_depth=0)

	try:
		service.execute_shutdown("tenant-a", plan["id"], target["id"], "operator", "")
	except PermissionError as exc:
		assert str(exc) == "health_gate_required"
	else:
		raise AssertionError("shutdown without health gate was accepted")

	execution = service.execute_shutdown("tenant-a", plan["id"], target["id"], "operator", "health://pre", force_shutdown=True)
	assert execution["status"] == "blocked"
	assert execution["required_actions"] == ["review_force_shutdown"]

	try:
		service.record_recovery("tenant-a", plan["id"], target["id"], "operator", "", "health://post")
	except PermissionError as exc:
		assert str(exc) == "incident_link_required"
	else:
		raise AssertionError("recovery without incident link was accepted")


def test_api_and_view_models_expose_lifecycle_control_surfaces():
	local_service = ShdnService()
	api.SERVICE = local_service

	target = api.register_service({
		"tenant_id": "tenant-b",
		"target_id": "etl-worker",
		"target_type": "worker",
		"owner": "data-platform",
		"environment": "staging",
		"criticality": "normal",
	})
	plan = api.create_shutdown_plan({
		"tenant_id": "tenant-b",
		"name": "ETL drain",
		"owner": "data-platform",
		"target_ids": [target["id"]],
		"reason": "schema migration",
		"rollback_plan_ref": "rollback://etl",
		"restart_sequence": ["etl-worker"],
		"approved_by": "lead",
		"maintenance_window_ref": "window://staging",
	})
	api.start_drain({
		"tenant_id": "tenant-b",
		"plan_id": plan["id"],
		"target_id": target["id"],
	})
	api.record_backup_snapshot({
		"tenant_id": "tenant-b",
		"plan_id": plan["id"],
		"target_id": target["id"],
		"evidence_ref": "backup://etl",
		"restore_test_ref": "restore://etl",
	})
	api.execute_shutdown({
		"tenant_id": "tenant-b",
		"plan_id": plan["id"],
		"target_id": target["id"],
		"actor": "operator",
		"health_gate_ref": "health://etl/pre",
	})
	api.record_recovery({
		"tenant_id": "tenant-b",
		"plan_id": plan["id"],
		"target_id": target["id"],
		"actor": "operator",
		"evidence_ref": "incident://change/456",
		"post_shutdown_health_check_ref": "health://etl/post",
	})

	status = api.capability_status("tenant-b")
	lifecycle = api.list_lifecycle_control("tenant-b")
	dashboard = views.dashboard_model(local_service, "tenant-b")
	services = views.service_console_model(local_service, "tenant-b")
	plans = views.plan_builder_model(local_service, "tenant-b")
	executions = views.execution_monitor_model(local_service, "tenant-b")
	approvals = views.approvals_model(local_service, "tenant-b")
	recovery = views.recovery_center_model(local_service, "tenant-b")
	audit = views.audit_model(local_service, "tenant-b")
	settings = views.settings_model("tenant-b")

	assert status["target_count"] == 1
	assert lifecycle["summary"]["recovery_count"] == 1
	assert dashboard["summary"]["snapshot_count"] == 1
	assert services["route"] == "/shdn/services"
	assert plans["approval_required"] is True
	assert executions["statuses"] == ["pending", "draining", "quiesced", "completed", "blocked"]
	assert approvals["force_shutdown_review_required"] is True
	assert recovery["required_evidence"] == ["backup_snapshot", "restore_test", "post_shutdown_health_check", "incident_link"]
	assert audit["events"]
	assert settings["theme"]["name"] == "shdn_lifecycle_control"
