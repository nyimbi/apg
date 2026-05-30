"""USRM package contract and deterministic runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.usrm import api, views
from capabilities.common.usrm.service import UsrmService


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
	module = _load_module("package_contract_usrm", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "usrm"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert "usrm_agents" in contract["provides"]


def test_app_entrypoint_is_publishable():
	module = _load_module("package_app_usrm", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "usrm" in model["capabilities"]


def test_user_profile_invite_access_review_and_deprovision_lifecycle_executes():
	service = UsrmService()

	user = service.create_user(
		tenant_id="tenant-a",
		identity="jane@example.com",
		display_name="Jane Doe",
		email="jane@example.com",
		owner="identity-owner",
		profile_validated=True,
		privileged_user=False,
		mfa_enabled=False,
	)
	profile = service.update_profile(
		tenant_id="tenant-a",
		user_id=user["id"],
		attributes={"department": "finance", "title": "Controller"},
		privacy_preferences={"analytics": "limited"},
		consent_notice_ref="consent://notice/1",
		updated_by="profile-admin",
	)
	invitation = service.invite_user(
		tenant_id="tenant-a",
		user_id=user["id"],
		channel="email",
		consent_notice_ref="consent://notice/1",
		invited_by="identity-owner",
	)
	role = service.assign_role(
		tenant_id="tenant-a",
		user_id=user["id"],
		role="finance-admin",
		scope="tenant",
		privileged=True,
		mfa_enabled=True,
		approved_by="access-owner",
	)
	review = service.record_access_review(
		tenant_id="tenant-a",
		user_id=user["id"],
		reviewer="access-reviewer",
		decision="approve",
		findings=["mfa-enabled"],
	)
	deprovision = service.deprovision_user(
		tenant_id="tenant-a",
		user_id=user["id"],
		actor="identity-owner",
		access_revoked=True,
		evidence_ref="evidence://deprovision/1",
	)
	summary = service.dashboard_summary("tenant-a")

	assert user["status"] == "active"
	assert profile["privacy_preferences"]["analytics"] == "limited"
	assert invitation["status"] == "sent"
	assert role["privileged"] is True
	assert review["decision"] == "approve"
	assert deprovision["status"] == "completed"
	assert summary["user_count"] == 1
	assert summary["deprovisioned_user_count"] == 1
	assert summary["role_assignment_count"] == 1
	assert summary["access_review_count"] == 1
	assert summary["streaming"]["processor"] == "bytewax"


def test_user_guardrails_require_tenant_identity_consent_mfa_revocation_and_bulk_review():
	service = UsrmService()

	try:
		service.create_user("", "no-tenant", "No Tenant", "no@example.com", "owner")
	except PermissionError as exc:
		assert str(exc) == "tenant_context_required"
	else:
		raise AssertionError("missing tenant was accepted")

	try:
		service.create_user("tenant-a", "", "No Identity", "no@example.com", "owner")
	except PermissionError as exc:
		assert str(exc) == "unique_identity_required"
	else:
		raise AssertionError("missing identity was accepted")

	try:
		service.create_user("tenant-a", "no-owner", "No Owner", "owner@example.com", "")
	except PermissionError as exc:
		assert str(exc) == "user_owner_required"
	else:
		raise AssertionError("missing owner was accepted")

	try:
		service.create_user("tenant-a", "bad-profile", "Bad Profile", "bad@example.com", "owner", profile_validated=False)
	except PermissionError as exc:
		assert str(exc) == "profile_validation_required"
	else:
		raise AssertionError("unvalidated profile was accepted")

	user = service.create_user("tenant-a", "user-1", "User One", "one@example.com", "owner")

	try:
		service.create_user("tenant-a", "user-1", "User Duplicate", "dup@example.com", "owner")
	except ValueError as exc:
		assert str(exc) == "unique_identity_required"
	else:
		raise AssertionError("duplicate identity was accepted")

	try:
		service.invite_user("tenant-a", user["id"], "email", "", "owner")
	except PermissionError as exc:
		assert str(exc) == "consent_notice_required"
	else:
		raise AssertionError("invite without consent notice was accepted")

	try:
		service.invite_user("tenant-a", user["id"], "email", "consent://notice", "owner", event_stream="local")
	except PermissionError as exc:
		assert str(exc) == "bytewax_event_stream_required"
	else:
		raise AssertionError("invite without Bytewax was accepted")

	try:
		service.assign_role("tenant-a", user["id"], "admin", "tenant", True, False, "access-owner")
	except PermissionError as exc:
		assert str(exc) == "mfa_required"
	else:
		raise AssertionError("privileged role without MFA was accepted")

	try:
		service.assign_role("tenant-a", user["id"], "admin", "tenant", True, True, "")
	except PermissionError as exc:
		assert str(exc) == "role_assignment_approval_required"
	else:
		raise AssertionError("privileged role without approval was accepted")

	try:
		service.update_profile("tenant-a", user["id"], {}, {}, "consent://notice", "owner", privacy_sync_recorded=False)
	except PermissionError as exc:
		assert str(exc) == "privacy_preference_sync_required"
	else:
		raise AssertionError("profile update without privacy sync was accepted")

	try:
		service.record_access_review("tenant-a", user["id"], "", "approve")
	except PermissionError as exc:
		assert str(exc) == "access_reviewer_required"
	else:
		raise AssertionError("access review without reviewer was accepted")

	try:
		service.deprovision_user("tenant-a", user["id"], "owner", False, "evidence://revocation")
	except PermissionError as exc:
		assert str(exc) == "access_revocation_required"
	else:
		raise AssertionError("deprovision without access revocation was accepted")

	try:
		service.deprovision_user("tenant-a", user["id"], "owner", True, "")
	except PermissionError as exc:
		assert str(exc) == "deprovision_evidence_required"
	else:
		raise AssertionError("deprovision without evidence was accepted")

	try:
		service.deprovision_user("tenant-a", user["id"], "owner", True, "evidence://revocation", event_stream="local")
	except PermissionError as exc:
		assert str(exc) == "bytewax_event_stream_required"
	else:
		raise AssertionError("deprovision without Bytewax was accepted")

	users = [
		service.create_user("tenant-a", f"user-{index}", f"User {index}", f"user{index}@example.com", "owner")
		for index in range(2, 29)
	]
	bulk = service.bulk_suspend_users("tenant-a", [item["id"] for item in users], "owner")
	assert bulk["status"] == "review_required"
	assert bulk["required_actions"] == ["review_bulk_user_action"]

	try:
		service.bulk_suspend_users("tenant-a", [users[0]["id"]], "owner", event_stream="local")
	except PermissionError as exc:
		assert str(exc) == "bytewax_event_stream_required"
	else:
		raise AssertionError("bulk action without Bytewax was accepted")


def test_usrm_agents_and_batch_lifecycle_guardrails_execute():
	service = UsrmService()

	agent = service.register_usrm_agent(
		tenant_id="tenant-a",
		name="Access reviewer",
		runtime="codex",
		role="access_reviewer",
		scope="review roles, MFA, and deprovision evidence",
	)
	privileged = service.validate_agent_user_action(
		tenant_id="tenant-a",
		agent_id=agent["id"],
		action="assign_role",
		privileged_scope=True,
	)
	approved = service.validate_agent_user_action(
		tenant_id="tenant-a",
		agent_id=agent["id"],
		action="assign_role",
		privileged_scope=True,
		human_approval_ref="approval://agent/user",
	)
	batch_block = service.validate_batch_user_lifecycle("tenant-a", 30, event_stream="local")

	assert agent["runtime"] == "codex"
	assert privileged["decision"] == "deny"
	assert privileged["matched_rules"] == ["privileged_agent_user_action_requires_human_approval"]
	assert approved["decision"] == "allow"
	assert batch_block["decision"] == "deny"
	assert "bulk_user_action_requires_bytewax" in batch_block["matched_rules"]

	try:
		service.register_usrm_agent("tenant-a", "Unsupported", "unknown", "access_reviewer", "review")
	except PermissionError as exc:
		assert str(exc) == "usrm_agent_runtime_not_supported"
	else:
		raise AssertionError("unsupported user-management agent runtime was accepted")


def test_api_and_view_models_expose_user_management_surfaces():
	local_service = UsrmService()
	api.SERVICE = local_service

	user = api.create_user({
		"tenant_id": "tenant-b",
		"identity": "alex@example.com",
		"display_name": "Alex Example",
		"email": "alex@example.com",
		"owner": "identity-owner",
		"mfa_enabled": True,
	})
	api.update_profile({
		"tenant_id": "tenant-b",
		"user_id": user["id"],
		"attributes": {"department": "operations"},
		"privacy_preferences": {"analytics": "opt-in"},
		"consent_notice_ref": "consent://ops",
		"updated_by": "profile-admin",
	})
	api.invite_user({
		"tenant_id": "tenant-b",
		"user_id": user["id"],
		"channel": "email",
		"consent_notice_ref": "consent://ops",
		"invited_by": "identity-owner",
	})
	api.assign_role({
		"tenant_id": "tenant-b",
		"user_id": user["id"],
		"role": "operations-reviewer",
		"scope": "tenant",
		"privileged": False,
		"mfa_enabled": True,
		"approved_by": "access-owner",
	})
	api.record_access_review({
		"tenant_id": "tenant-b",
		"user_id": user["id"],
		"reviewer": "access-reviewer",
		"decision": "approve",
	})
	agent = api.register_usrm_agent({
		"tenant_id": "tenant-b",
		"name": "Lifecycle reviewer",
		"runtime": "claude_code",
		"role": "lifecycle_reviewer",
		"scope": "review onboarding and deprovisioning",
	})

	status = api.capability_status("tenant-b")
	system = api.list_user_management("tenant-b")
	dashboard = views.dashboard_model(local_service, "tenant-b")
	directory = views.user_directory_model(local_service, "tenant-b")
	profiles = views.profile_manager_model(local_service, "tenant-b")
	lifecycle = views.lifecycle_queue_model(local_service, "tenant-b")
	access = views.access_review_model(local_service, "tenant-b")
	privacy = views.privacy_preferences_model(local_service, "tenant-b")
	deprovisioning = views.deprovisioning_model(local_service, "tenant-b")
	agents = views.agent_workbench_model(local_service, "tenant-b")
	policy = views.policy_center_model(local_service, "tenant-b")
	settings = views.settings_model("tenant-b")

	assert status["user_count"] == 1
	assert status["usrm_agent_count"] == 1
	assert system["summary"]["profile_count"] == 1
	assert system["usrm_agents"][0]["id"] == agent["id"]
	assert dashboard["summary"]["role_assignment_count"] == 1
	assert directory["route"] == "/usrm/users"
	assert profiles["privacy_sync_required"] is True
	assert lifecycle["invitations"]
	assert access["mfa_required_for_privileged"] is True
	assert privacy["consent_notice_required"] is True
	assert deprovisioning["access_revocation_required"] is True
	assert agents["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert policy["streaming"]["processor"] == "bytewax"
	assert settings["theme"]["name"] == "usrm_user_lifecycle"
	assert settings["streaming"]["processor"] == "bytewax"
