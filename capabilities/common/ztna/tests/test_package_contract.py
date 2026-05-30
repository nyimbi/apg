"""Zero Trust Network Access package runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.ztna import views
from capabilities.common.ztna.service import ZtnaService


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def _build_accessible_resource(service: ZtnaService) -> tuple[dict, dict, dict]:
	identity = service.register_identity(
		identity_key="analyst",
		tenant_id="tenant-ztna",
		subject_id="user-1",
		display_name="Analyst",
		verified=True,
		mfa_completed=True,
	)
	device = service.register_device(
		device_key="laptop",
		tenant_id="tenant-ztna",
		identity_id=identity["id"],
		name="Managed Laptop",
		trust_score=0.92,
		posture_present=True,
		managed=True,
		attested=True,
	)
	resource = service.register_resource(
		resource_key="crm",
		tenant_id="tenant-ztna",
		name="CRM Console",
		access_level="standard",
		policy_attached=True,
		policy_id="crm-policy",
	)
	return identity, device, resource


def test_package_contract_shape_and_entrypoint_are_publishable():
	contract_module = _load_module("ztna_contract_runtime", PACKAGE_DIR / "capability_contract.py")
	app_module = _load_module("ztna_app_runtime", PACKAGE_DIR / "app.py")

	contract = contract_module.get_capability_contract("tenant-test")
	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")

	self_test = app_module.self_test()
	manifest = app_module.component_manifest()
	model = app_module.semantic_model()

	assert contract["capability"] == "ztna"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert model["format"] == "apg.semantic-model.v1"
	assert "ztna" in model["capabilities"]
	assert model["capabilities"]["ztna"]["streaming"]["engine"] == "bytewax"
	assert model["capabilities"]["ztna"]["runtime"]["service"] == "service.ZtnaService"


def test_identity_device_resource_access_session_lifecycle_executes():
	service = ZtnaService()
	identity, device, resource = _build_accessible_resource(service)
	request = service.request_access(identity["id"], device["id"], resource["id"], requested_by="user-1")
	session = service.start_session(request["id"], actor_id="broker")
	closed = service.close_session(session["id"], actor_id="broker")
	summary = service.dashboard_summary("tenant-ztna")

	assert request["status"] == "approved"
	assert session["status"] == "active"
	assert closed["status"] == "closed"
	assert summary["identity_count"] == 1
	assert summary["trusted_device_count"] == 1
	assert summary["resource_count"] == 1
	assert summary["active_session_count"] == 0


def test_generic_record_compatibility_creates_protected_resources():
	service = ZtnaService()
	record = service.create_record(
		"vpn-admin",
		"tenant-ztna",
		metadata={"name": "VPN Admin", "access_level": "privileged", "policy_id": "vpn-policy"},
	)
	records = service.list_records("tenant-ztna")

	assert record["id"] == records[0]["id"]
	assert records[0]["name"] == "VPN Admin"
	assert records[0]["access_level"] == "privileged"
	assert records[0]["policy_attached"] is True


def test_tenant_and_identity_verification_are_required():
	service = ZtnaService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_identity("missing", "", "user-x", "Missing")

	identity = service.register_identity("pending", "tenant-ztna", "user-2", "Pending", verified=False)
	device = service.register_device("laptop", "tenant-ztna", identity["id"], "Laptop", trust_score=0.9)
	resource = service.register_resource("crm", "tenant-ztna", "CRM", policy_attached=True)

	with pytest.raises(PermissionError, match="identity_verification_required"):
		service.request_access(identity["id"], device["id"], resource["id"], requested_by="user-2")


def test_device_posture_and_resource_policy_are_required():
	service = ZtnaService()
	identity = service.register_identity("analyst", "tenant-ztna", "user-1", "Analyst", verified=True)
	device = service.register_device("unknown", "tenant-ztna", identity["id"], "Unknown Device", trust_score=0.9, posture_present=False)
	resource = service.register_resource("admin", "tenant-ztna", "Admin Console", policy_attached=False)

	with pytest.raises(PermissionError, match="device_posture_required"):
		service.request_access(identity["id"], device["id"], resource["id"], requested_by="user-1")

	device = service.update_device_posture(device["id"], trust_score=0.9, posture_present=True)
	with pytest.raises(PermissionError, match="resource_policy_required"):
		service.request_access(identity["id"], device["id"], resource["id"], requested_by="user-1")

	resource = service.attach_resource_policy(resource["id"], "admin-policy", actor_id="secops")
	request = service.request_access(identity["id"], device["id"], resource["id"], requested_by="user-1")

	assert device["status"] == "trusted"
	assert resource["policy_attached"] is True
	assert request["status"] == "approved"


def test_untrusted_devices_are_registered_for_remediation_but_denied_access():
	service = ZtnaService()
	identity = service.register_identity("analyst", "tenant-ztna", "user-1", "Analyst", verified=True)
	device = service.register_device("risky", "tenant-ztna", identity["id"], "Risky Device", trust_score=0.2, compliant=False)
	resource = service.register_resource("crm", "tenant-ztna", "CRM", policy_attached=True)

	assert device["status"] == "quarantined"
	with pytest.raises(PermissionError, match="device_trust_too_low"):
		service.request_access(identity["id"], device["id"], resource["id"], requested_by="user-1")


def test_privileged_access_requires_mfa():
	service = ZtnaService()
	identity = service.register_identity("admin", "tenant-ztna", "admin-1", "Admin", verified=True, privileged=True, mfa_completed=False)
	device = service.register_device("admin-laptop", "tenant-ztna", identity["id"], "Admin Laptop", trust_score=0.95, managed=True)
	resource = service.register_resource("root", "tenant-ztna", "Root Console", access_level="privileged", policy_attached=True)

	with pytest.raises(PermissionError, match="privileged_mfa_required"):
		service.request_access(identity["id"], device["id"], resource["id"], requested_by="admin-1")

	identity = service.verify_identity(identity["id"], actor_id="mfau", mfa_completed=True)
	request = service.request_access(identity["id"], device["id"], resource["id"], requested_by="admin-1", mfa_completed=True)
	approved = service.approve_access_request(request["id"], reviewer_id="reviewer-1")

	assert identity["mfa_completed"] is True
	assert request["status"] == "review_required"
	assert approved["status"] == "approved"


def test_high_risk_access_requires_review_before_session():
	service = ZtnaService()
	identity, device, resource = _build_accessible_resource(service)
	request = service.request_access(
		identity["id"],
		device["id"],
		resource["id"],
		requested_by="user-1",
		access_risk_score=0.95,
		access_review_recorded=False,
	)

	assert request["status"] == "review_required"
	assert request["required_actions"] == ["review_access_request"]
	with pytest.raises(PermissionError, match="access_request_not_approved"):
		service.start_session(request["id"], actor_id="broker")

	approved = service.approve_access_request(request["id"], reviewer_id="reviewer-1")
	session = service.start_session(approved["id"], actor_id="broker")

	assert approved["status"] == "approved"
	assert session["status"] == "active"


def test_request_specific_guardrails_require_review_or_block_duplicates():
	service = ZtnaService()
	identity, device, resource = _build_accessible_resource(service)
	scoped = service.request_access(
		identity["id"],
		device["id"],
		resource["id"],
		requested_by="user-1",
		least_privilege_scope_present=False,
	)

	assert scoped["status"] == "review_required"
	assert "narrow_access_scope" in scoped["required_actions"]
	with pytest.raises(PermissionError, match="duplicate_access_review"):
		service.request_access(
			identity["id"],
			device["id"],
			resource["id"],
			requested_by="user-1",
			least_privilege_scope_present=False,
		)


def test_continuous_verification_can_require_reauth_or_revoke_session():
	service = ZtnaService()
	identity, device, resource = _build_accessible_resource(service)
	request = service.request_access(identity["id"], device["id"], resource["id"], requested_by="user-1")
	session = service.start_session(request["id"], actor_id="broker")
	review = service.reevaluate_session(session["id"], risk_score=0.95, actor_id="risk-engine")
	revoked = service.reevaluate_session(session["id"], risk_score=0.2, identity_verified=False, actor_id="risk-engine")

	assert review["status"] == "review_required"
	assert review["reauth_required"] is True
	assert revoked["status"] == "revoked"
	assert revoked["ended_at"] is not None


def test_api_helpers_expose_zero_trust_lifecycle():
	from capabilities.common.ztna import api

	api.SERVICE = ZtnaService()
	identity = api.register_identity({
		"identity_key": "api-user",
		"tenant_id": "tenant-api",
		"subject_id": "api-1",
		"display_name": "API User",
		"verified": True,
		"mfa_completed": True,
	})
	device = api.register_device({
		"device_key": "api-laptop",
		"tenant_id": "tenant-api",
		"identity_id": identity["id"],
		"name": "API Laptop",
		"trust_score": 0.91,
		"managed": True,
	})
	resource = api.register_resource({
		"resource_key": "api-resource",
		"tenant_id": "tenant-api",
		"name": "API Resource",
		"policy_attached": True,
	})
	request = api.request_access({"identity_id": identity["id"], "device_id": device["id"], "resource_id": resource["id"], "requested_by": "api-1"})
	session = api.start_session(request["id"], actor_id="broker")
	status = api.capability_status("tenant-api")
	listing = api.list_zero_trust_access("tenant-api")

	assert session["status"] == "active"
	assert status["identity_count"] == 1
	assert status["active_session_count"] == 1
	assert listing["access_requests"][0]["id"] == request["id"]


def test_view_models_match_routes_theme_and_runtime_state():
	service = ZtnaService()
	identity, device, resource = _build_accessible_resource(service)
	request = service.request_access(identity["id"], device["id"], resource["id"], requested_by="user-1")
	session = service.start_session(request["id"], actor_id="broker")

	dashboard = views.dashboard_model(service, "tenant-ztna")
	policies = views.policy_console_model(service, "tenant-ztna")
	identities = views.identity_console_model(service, "tenant-ztna")
	devices = views.device_posture_model(service, "tenant-ztna")
	resources = views.resource_map_model(service, "tenant-ztna")
	access = views.access_requests_model(service, "tenant-ztna")
	sessions = views.session_monitor_model(service, "tenant-ztna")
	risk = views.risk_console_model(service, "tenant-ztna")
	reviews = views.review_queue_model(service, "tenant-ztna")
	audit = views.audit_model(service, "tenant-ztna")
	settings = views.settings_model("tenant-ztna")

	assert dashboard["summary"]["identity_count"] == 1
	assert policies["policy_required"] == []
	assert identities["identities"][0]["id"] == identity["id"]
	assert devices["devices"][0]["id"] == device["id"]
	assert resources["segments"] == ["default"]
	assert access["access_requests"][0]["id"] == request["id"]
	assert sessions["sessions"][0]["id"] == session["id"]
	assert risk["signals"]["revocation_rate"] == 0.0
	assert reviews["review_required"] == []
	assert audit["audit_events"]
	assert settings["theme"]["name"] == "ztna_zero_trust_ops"
