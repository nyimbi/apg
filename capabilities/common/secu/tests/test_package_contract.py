"""SECU package contract and deterministic runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.secu import api, views
from capabilities.common.secu.service import SecuService


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
	module = _load_module("materialized_contract_secu", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "secu"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_app_entrypoint_is_publishable():
	module = _load_module("materialized_app_secu", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "secu" in model["capabilities"]


def test_security_lifecycle_records_policy_device_threat_assessment_and_compliance():
	service = SecuService()

	policy = service.create_policy(
		tenant_id="tenant-a",
		name="Privileged access",
		owner="security-admin",
		security_level="restricted",
		required_controls=["mfa", "device_trust"],
		applies_to=["admin_console"],
		tags=["Privileged Access"],
	)
	device = service.record_device_posture(
		tenant_id="tenant-a",
		device_id="macbook-01",
		user_id="u-123",
		trust_state="trusted",
		risk_score=20,
	)
	threat = service.register_threat_indicator(
		tenant_id="tenant-a",
		name="Known hostile ASN",
		indicator_type="asn",
		value="AS64512",
		severity="high",
		source="manual",
	)
	assessment = service.assess_access(
		tenant_id="tenant-a",
		subject_id="u-123",
		subject_type="user",
		risk_score=75,
		device_id="macbook-01",
		challenge_completed=False,
	)
	control = service.record_compliance_control(
		tenant_id="tenant-a",
		framework="iso_27001",
		control_id="A.5.15",
		owner="security-admin",
		compliant=False,
	)
	summary = service.dashboard_summary("tenant-a")

	assert policy["security_level"] == "restricted"
	assert device["quarantined"] is False
	assert threat["severity"] == "high"
	assert assessment["decision"] == "challenge"
	assert assessment["required_actions"] == ["complete_security_challenge"]
	assert control["status"] == "evidence_required"
	assert summary["policy_count"] == 1
	assert summary["assessment_count"] == 1
	assert summary["compliance_gap_count"] == 1


def test_rule_guardrails_deny_quarantine_and_require_tenant_context():
	service = SecuService()

	try:
		service.create_policy("", "No tenant", "owner")
	except PermissionError as exc:
		assert str(exc) == "tenant_context_required"
	else:
		raise AssertionError("missing tenant context was accepted")

	try:
		service.create_policy("tenant-a", "No owner", "")
	except ValueError as exc:
		assert str(exc) == "policy_owner_required"
	else:
		raise AssertionError("missing policy owner was accepted")

	try:
		service.record_device_posture("tenant-a", "device-1", "user-1", trust_state="rooted")
	except ValueError as exc:
		assert str(exc) == "unsupported_device_trust:rooted"
	else:
		raise AssertionError("unsupported device trust state was accepted")

	service.record_device_posture("tenant-a", "device-2", "user-1", trust_state="compromised")
	quarantined = service.assess_access("tenant-a", "user-1", "user", 60, device_id="device-2")
	denied = service.assess_access("tenant-a", "user-1", "user", 95)
	malicious = service.assess_access("tenant-a", "user-1", "user", 10, is_known_malicious=True)

	assert quarantined["decision"] == "quarantine"
	assert "compromised_device_quarantined" in quarantined["matched_rules"]
	assert denied["decision"] == "deny"
	assert "critical_risk_denied" in denied["matched_rules"]
	assert malicious["decision"] == "deny"
	assert "known_malicious_network_denied" in malicious["matched_rules"]


def test_api_and_view_models_expose_security_posture_surfaces():
	local_api_service = SecuService()
	api.SERVICE = local_api_service

	api.create_policy({"tenant_id": "tenant-b", "name": "Data access", "owner": "secops"})
	api.record_device_posture({"tenant_id": "tenant-b", "device_id": "device-1", "user_id": "user-1"})
	api.register_threat_indicator({
		"tenant_id": "tenant-b",
		"name": "Suspicious host",
		"indicator_type": "host",
		"value": "example.invalid",
		"severity": "medium",
	})
	api.assess_access({"tenant_id": "tenant-b", "subject_id": "user-1", "risk_score": 30})
	api.record_compliance_control({
		"tenant_id": "tenant-b",
		"control_id": "CC6.1",
		"owner": "secops",
		"compliant": True,
		"evidence_ref": "audit://evidence/1",
	})

	status = api.capability_status("tenant-b")
	posture = api.list_security_posture("tenant-b")
	dashboard = views.dashboard_model(local_api_service, "tenant-b")
	risk = views.risk_console_model(local_api_service, "tenant-b")
	threats = views.threat_console_model(local_api_service, "tenant-b")
	policies = views.policy_workbench_model(local_api_service, "tenant-b")
	compliance = views.compliance_console_model(local_api_service, "tenant-b")
	rules = views.rule_workbench_model("tenant-b")
	settings = views.settings_model("tenant-b")

	assert status["policy_count"] == 1
	assert posture["summary"]["active_threat_count"] == 1
	assert dashboard["summary"]["assessment_count"] == 1
	assert risk["route"] == "/secu/risk"
	assert threats["severity_filters"] == ["info", "low", "medium", "high", "critical"]
	assert policies["security_levels"][-1] == "critical"
	assert compliance["controls"][0]["status"] == "implemented"
	assert rules["decision_order"] == ["deny", "quarantine", "challenge", "allow"]
	assert settings["theme"]["name"] == "secu_zero_trust"
