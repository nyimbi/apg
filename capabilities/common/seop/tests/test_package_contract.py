"""SEOP package contract and deterministic runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.seop import api, views
from capabilities.common.seop.service import SeopService


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
	module = _load_module("materialized_contract_seop", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "seop"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_app_entrypoint_is_publishable():
	module = _load_module("materialized_app_seop", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "seop" in model["capabilities"]


def test_security_operations_lifecycle_executes():
	service = SeopService()

	detection = service.create_detection(
		tenant_id="tenant-a",
		title="Privileged anomaly",
		alert_source="siem",
		anomaly_confidence=0.95,
		severity="high",
		signal_refs=["alert-1"],
	)
	incident = service.open_incident(
		tenant_id="tenant-a",
		title="Privileged compromise",
		owner="secops-lead",
		severity="critical",
		detection_ids=[detection["id"]],
		escalation_recorded=True,
		evidence_refs=["case://evidence/1"],
	)
	playbook = service.approve_playbook(
		tenant_id="tenant-a",
		name="Isolate privileged account",
		owner="secops-lead",
		steps=["disable token", "isolate endpoint", "notify owner"],
		approved_by="ciso",
	)
	response = service.execute_response(
		tenant_id="tenant-a",
		incident_id=incident["id"],
		playbook_id=playbook["id"],
		action="isolate endpoint",
		actor="analyst-1",
	)
	posture = service.record_posture_control(
		tenant_id="tenant-a",
		control_id="SOC-IR-01",
		domain="incident_response",
		coverage=0.55,
		owner="secops-lead",
	)
	closed = service.close_incident(
		tenant_id="tenant-a",
		incident_id=incident["id"],
		closure_evidence="case://closure/1",
		actor="analyst-1",
	)
	summary = service.dashboard_summary("tenant-a")

	assert detection["status"] == "review_required"
	assert detection["required_actions"] == ["review_anomaly"]
	assert incident["status"] == "escalated"
	assert response["status"] == "executed"
	assert posture["status"] == "gap"
	assert closed["status"] == "closed"
	assert summary["detection_count"] == 1
	assert summary["incident_count"] == 1
	assert summary["response_count"] == 1
	assert summary["posture_gap_count"] == 1


def test_guardrails_require_tenant_source_owner_escalation_and_playbook_approval():
	service = SeopService()

	try:
		service.create_detection("", "No tenant", "siem", 0.1)
	except PermissionError as exc:
		assert str(exc) == "tenant_context_required"
	else:
		raise AssertionError("missing tenant was accepted")

	try:
		service.create_detection("tenant-a", "No source", "", 0.1)
	except PermissionError as exc:
		assert str(exc) == "alert_source_required"
	else:
		raise AssertionError("missing alert source was accepted")

	try:
		service.open_incident("tenant-a", "No owner", "", "high")
	except PermissionError as exc:
		assert str(exc) == "incident_owner_required"
	else:
		raise AssertionError("missing incident owner was accepted")

	try:
		service.open_incident("tenant-a", "Critical", "owner", "critical", escalation_recorded=False)
	except PermissionError as exc:
		assert str(exc) == "critical_escalation_required"
	else:
		raise AssertionError("un-escalated critical incident was accepted")

	try:
		service.approve_playbook("tenant-a", "No approval", "owner", ["step"], "")
	except PermissionError as exc:
		assert str(exc) == "playbook_approval_required"
	else:
		raise AssertionError("unapproved playbook was accepted")

	try:
		service.create_detection("tenant-a", "Bad confidence", "siem", 1.5)
	except ValueError as exc:
		assert str(exc) == "anomaly_confidence_out_of_range:1.5"
	else:
		raise AssertionError("invalid confidence was accepted")


def test_api_and_view_models_expose_security_operations_surfaces():
	local_service = SeopService()
	api.SERVICE = local_service

	detection = api.create_detection({
		"tenant_id": "tenant-b",
		"title": "Suspicious egress",
		"alert_source": "network_sensor",
		"anomaly_confidence": 0.4,
		"severity": "medium",
	})
	incident = api.open_incident({
		"tenant_id": "tenant-b",
		"title": "Suspicious egress investigation",
		"owner": "analyst",
		"severity": "high",
		"detection_ids": [detection["id"]],
	})
	playbook = api.approve_playbook({
		"tenant_id": "tenant-b",
		"name": "Contain egress",
		"owner": "analyst",
		"steps": ["block destination"],
		"approved_by": "manager",
	})
	api.execute_response({
		"tenant_id": "tenant-b",
		"incident_id": incident["id"],
		"playbook_id": playbook["id"],
		"action": "block destination",
		"actor": "analyst",
	})
	api.record_posture_control({
		"tenant_id": "tenant-b",
		"control_id": "SOC-MON-01",
		"coverage": 0.95,
		"owner": "analyst",
	})

	status = api.capability_status("tenant-b")
	ops = api.list_security_operations("tenant-b")
	dashboard = views.dashboard_model(local_service, "tenant-b")
	detections = views.detection_console_model(local_service, "tenant-b")
	incidents = views.incident_queue_model(local_service, "tenant-b")
	playbooks = views.playbook_manager_model(local_service, "tenant-b")
	responses = views.response_actions_model(local_service, "tenant-b")
	posture = views.posture_model(local_service, "tenant-b")
	triage = views.triage_model(local_service, "tenant-b")
	settings = views.settings_model("tenant-b")

	assert status["detection_count"] == 1
	assert ops["summary"]["response_count"] == 1
	assert dashboard["summary"]["approved_playbook_count"] == 1
	assert detections["route"] == "/seop/detections"
	assert incidents["state_filters"][0] == "open"
	assert playbooks["approval_required"] is True
	assert responses["statuses"] == ["planned", "executed", "blocked"]
	assert posture["coverage_bands"] == ["gap", "partial", "covered"]
	assert triage["review_required"] == []
	assert settings["theme"]["name"] == "seop_security_ops"
