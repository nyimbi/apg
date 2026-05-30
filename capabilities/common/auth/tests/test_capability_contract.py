"""Regression coverage for the AUTH executable capability contract."""

import pytest

from .. import api_helpers, view_models
from .. import get_capability_info, register_capability
from ..capability_contract import evaluate_capability_rules, get_capability_contract
from ..service import AuthService


def _grant_actor(service: AuthService, tenant_id: str, actor_id: str, permission: str) -> None:
	"""Grant one service-test actor a tenant permission through normal role state."""
	if not any(identity["id"] == actor_id for identity in service.list_identities(tenant_id)):
		service.register_identity(
			user_id=actor_id,
			tenant_id=tenant_id,
			email=f"{actor_id}@example.com",
			display_name=actor_id.replace("-", " ").title(),
			mfa_enabled=True,
		)
	role_id = f"role-{actor_id}-{permission.replace(':', '-')}"
	if not any(role["id"] == role_id for role in service.list_roles(tenant_id)):
		service.define_role(
			role_id=role_id,
			tenant_id=tenant_id,
			name=f"{permission} actor",
			permissions=[permission],
		)
		service.assign_role(
			assignment_id=f"assign-{role_id}",
			tenant_id=tenant_id,
			user_id=actor_id,
			role_id=role_id,
			assigned_by="system",
		)


def _grant_api_actor(tenant_id: str, actor_id: str, permission: str) -> None:
	api_helpers.register_identity({
		"id": actor_id,
		"tenant_id": tenant_id,
		"email": f"{actor_id}@example.com",
		"display_name": actor_id.replace("-", " ").title(),
		"mfa_enabled": True,
	})
	role = api_helpers.define_role({
		"id": f"role-{actor_id}-{permission.replace(':', '-')}",
		"tenant_id": tenant_id,
		"name": f"{permission} actor",
		"permissions": [permission],
	})
	api_helpers.assign_role({
		"id": f"assign-{role['id']}",
		"tenant_id": tenant_id,
		"user_id": actor_id,
		"role_id": role["id"],
		"assigned_by": "system",
	})


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-a",
		{"sessions": {"idle_timeout_minutes": 45}}
	)

	assert contract["capability"] == "auth"
	assert contract["configuration"]["tenant_id"] == "tenant-a"
	assert contract["configuration"]["sessions"]["idle_timeout_minutes"] == 45
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"authentication",
		"authorization",
		"sessions",
		"federation",
		"privacy",
		"security_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme"
	]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["configuration"]["security_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert set(contract["provides"]) >= {"identity_registry", "role_governance", "security_agents"}
	assert contract["requires"] == ["audl", "mten", "keym", "secu"]
	assert len(contract["rule_engine"]["rules"]) >= 9
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"login",
		"dashboard",
		"role_workbench",
		"role_approvals",
		"sessions",
		"access_decisions",
		"biometric_management",
		"quantum_keys",
		"behavioral_analysis",
		"privacy_analytics",
		"privacy_reviews",
		"security_agents",
		"audit",
		"analytics",
		"federated_mesh",
		"metrics"
	}
	assert contract["ui"]["api_prefix"] == "/api"
	assert contract["theme"]["tokens"]["border.radius"] == "12px"
	assert "risk_posture_meter" in contract["theme"]["components"]
	assert "role_approval_queue" in contract["theme"]["components"]
	assert "privacy_approval_queue" in contract["theme"]["components"]


def test_rule_engine_denies_unsafe_privileged_access():
	result = evaluate_capability_rules({
		"user_locked": True,
		"requested_permission_tier": "privileged",
		"mfa_verified": False,
		"risk_level": "high",
		"step_up_completed": False,
		"requested_operation": "assign_role",
		"role_tier": "admin",
		"approval_recorded": False,
		"auth_source": "federated",
		"issuer_trusted": False,
		"tenant_mismatch": True,
		"tenant_membership_confirmed": False,
	})
	approval_result = evaluate_capability_rules({
		"requested_operation": "approve_role_assignment",
		"reviewer_same_as_requester": True,
	})
	privacy_approval_result = evaluate_capability_rules({
		"requested_operation": "approve_privacy_budget",
		"reviewer_same_as_requester": True,
	})
	batch_result = evaluate_capability_rules({
		"requested_operation": "batch_auth_mutation",
		"event_stream": "memory",
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"locked_accounts_denied",
		"privileged_access_requires_mfa",
		"high_risk_sessions_require_step_up",
		"elevated_role_assignment_requires_approval",
		"untrusted_federation_denied",
		"cross_tenant_access_requires_membership",
	}
	assert approval_result["matched_rules"] == ["role_assignment_approval_requires_independent_reviewer"]
	assert privacy_approval_result["matched_rules"] == ["privacy_budget_approval_requires_independent_reviewer"]
	assert batch_result["matched_rules"] == ["batch_auth_mutation_requires_bytewax"]


def test_capability_info_and_registration_include_manifest_and_theme():
	info = get_capability_info()
	registration = register_capability()

	assert info["metadata"]["capability_id"] == "common/auth"
	assert info["metadata"]["aliases"] == ["auth_rbac"]
	assert info["configuration"]["tenant_id"] == "default"
	assert info["ui_manifest"]["requires_theme"] is True
	assert info["theme"]["name"] == "auth_trust_fabric"
	assert registration["aliases"] == ["auth_rbac"]
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_components"]["metrics"] == "/auth/metrics/overview"
	assert registration["ui_components"]["role_approvals"] == "/auth/roles/approvals"
	assert registration["ui_components"]["privacy_reviews"] == "/auth/privacy/reviews"
	assert "auth:approve_roles" in registration["permissions"]
	assert "auth:approve_privacy" in registration["permissions"]


def test_auth_service_runs_identity_role_session_access_and_privacy_lifecycle():
	service = AuthService()

	identity = service.register_identity(
		user_id="user-1",
		tenant_id="tenant-a",
		email="ada@example.com",
		display_name="Ada Lovelace",
		mfa_enabled=True,
		behavioral_trust_score=0.92,
		biometric_enrolled=True,
		privacy_budget=0.75,
	)
	role = service.define_role(
		role_id="role-admin",
		tenant_id="tenant-a",
		name="Tenant Administrator",
		permissions=["auth:view", "auth:admin"],
		tier="admin",
	)
	_grant_actor(service, "tenant-a", "security-requester", "auth:manage_roles")
	_grant_actor(service, "tenant-a", "security-reviewer", "auth:approve_roles")
	_grant_actor(service, "tenant-a", "security-owner", "auth:manage_roles")
	approval_request = service.request_role_assignment_approval(
		approval_id="approval-1",
		tenant_id="tenant-a",
		user_id="user-1",
		role_id="role-admin",
		requested_by="security-requester",
		justification="Tenant administrator setup.",
	)
	approval = service.decide_role_assignment_approval(
		approval_id=approval_request["id"],
		tenant_id="tenant-a",
		reviewer="security-reviewer",
		decision="approved",
		notes="Approved with ticket AUTH-1.",
	)
	assignment = service.assign_role(
		assignment_id="assign-1",
		tenant_id="tenant-a",
		user_id="user-1",
		role_id="role-admin",
		assigned_by="security-owner",
		approval_id=approval["id"],
	)
	session = service.start_session(
		session_id="session-1",
		tenant_id="tenant-a",
		user_id="user-1",
		device_id="device-1",
		mfa_verified=True,
		risk_level="medium",
	)
	decision = service.evaluate_access(
		decision_id="decision-1",
		tenant_id="tenant-a",
		user_id="user-1",
		permission="auth:admin",
		session_id="session-1",
		requested_permission_tier="privileged",
	)
	privacy_query = service.run_privacy_query(
		query_id="query-1",
		tenant_id="tenant-a",
		user_id="user-1",
		query_type="risk_histogram",
		epsilon_cost=0.25,
	)

	assert identity["mfa_enabled"] is True
	assert role["tier"] == "admin"
	assert approval["status"] == "approved"
	assert assignment["approval_recorded"] is True
	assert assignment["approval_id"] == "approval-1"
	assert session["trust_score"] > 0.7
	assert decision["decision"] == "allow"
	assert decision["role_ids"] == ["role-admin"]
	assert privacy_query["status"] == "completed"
	assert privacy_query["remaining_budget"] == 0.5
	assert service.dashboard_summary("tenant-a") == {
		"tenant_id": "tenant-a",
		"identity_count": 4,
		"active_session_count": 1,
		"role_count": 4,
		"admin_assignment_count": 1,
		"role_approval_count": 1,
		"pending_role_approval_count": 0,
		"privacy_approval_count": 0,
		"pending_privacy_approval_count": 0,
		"security_agent_count": 0,
		"denied_decision_count": 0,
		"privacy_review_count": 0,
		"average_trust_score": session["trust_score"],
	}
	assert any(record["id"] == "user-1" for record in service.list_records("tenant-a"))
	assert service.list_audit_events("tenant-a")


def test_auth_service_enforces_contract_guardrails():
	service = AuthService()
	service.register_identity(
		user_id="user-1",
		tenant_id="tenant-a",
		email="ada@example.com",
		display_name="Ada Lovelace",
		status="locked",
		tenant_memberships=["tenant-b"],
	)
	service.define_role(
		role_id="role-admin",
		tenant_id="tenant-a",
		name="Tenant Administrator",
		permissions=["auth:admin"],
		tier="admin",
	)
	_grant_actor(service, "tenant-a", "security-owner", "auth:manage_roles")
	_grant_actor(service, "tenant-a", "security-owner", "auth:approve_roles")
	_grant_actor(service, "tenant-a", "security-reviewer", "auth:approve_roles")

	with pytest.raises(PermissionError, match="account_locked"):
		service.start_session(
			session_id="session-locked",
			tenant_id="tenant-a",
			user_id="user-1",
			device_id="device-1",
		)

	with pytest.raises(PermissionError, match="approval_required_for_admin_role_assignment"):
		service.assign_role(
			assignment_id="assign-denied",
			tenant_id="tenant-a",
			user_id="user-1",
			role_id="role-admin",
			assigned_by="security-owner",
			approval_recorded=True,
		)

	approval_request = service.request_role_assignment_approval(
		approval_id="approval-rejected",
		tenant_id="tenant-a",
		user_id="user-1",
		role_id="role-admin",
		requested_by="security-owner",
		justification="Emergency admin assignment.",
	)
	with pytest.raises(PermissionError, match="independent_role_approval_reviewer_required"):
		service.decide_role_assignment_approval(
			approval_id=approval_request["id"],
			tenant_id="tenant-a",
			reviewer="security-owner",
			decision="approved",
			notes="Self approved.",
		)
	with pytest.raises(ValueError, match="role_approval_notes_required"):
		service.decide_role_assignment_approval(
			approval_id=approval_request["id"],
			tenant_id="tenant-a",
			reviewer="security-reviewer",
			decision="approved",
			notes="",
		)
	rejected = service.decide_role_assignment_approval(
		approval_id=approval_request["id"],
		tenant_id="tenant-a",
		reviewer="security-reviewer",
		decision="rejected",
		notes="No valid business owner evidence.",
	)
	with pytest.raises(ValueError, match="role_approval_already_decided"):
		service.decide_role_assignment_approval(
			approval_id=approval_request["id"],
			tenant_id="tenant-a",
			reviewer="security-reviewer",
			decision="approved",
			notes="Changed after rejection.",
		)
	with pytest.raises(PermissionError, match="role_assignment_approval_not_approved"):
		service.assign_role(
			assignment_id="assign-rejected",
			tenant_id="tenant-a",
			user_id="user-1",
			role_id="role-admin",
			assigned_by="security-owner",
			approval_id=rejected["id"],
		)

	service.register_identity(
		user_id="user-2",
		tenant_id="tenant-a",
		email="grace@example.com",
		display_name="Grace Hopper",
		tenant_memberships=["tenant-b"],
	)
	with pytest.raises(PermissionError, match="trusted_issuer_required"):
		service.start_session(
			session_id="session-untrusted",
			tenant_id="tenant-a",
			user_id="user-2",
			device_id="device-2",
			auth_source="federated",
			issuer_trusted=False,
		)
	with pytest.raises(PermissionError, match="tenant_membership_required"):
		service.start_session(
			session_id="session-cross",
			tenant_id="tenant-c",
			user_id="user-2",
			device_id="device-2",
		)


def test_auth_service_keeps_duplicate_ids_isolated_by_tenant():
	service = AuthService()
	for tenant_id, email in [("tenant-a", "same-a@example.com"), ("tenant-b", "same-b@example.com")]:
		service.register_identity(
			user_id="same-user",
			tenant_id=tenant_id,
			email=email,
			display_name=f"Same User {tenant_id}",
			mfa_enabled=True,
		)
		service.define_role(
			role_id="same-role",
			tenant_id=tenant_id,
			name="Same Role",
			permissions=["auth:view"],
		)
		service.assign_role(
			assignment_id="same-assignment",
			tenant_id=tenant_id,
			user_id="same-user",
			role_id="same-role",
			assigned_by="system",
		)
		service.start_session(
			session_id="same-session",
			tenant_id=tenant_id,
			user_id="same-user",
			device_id=f"device-{tenant_id}",
			mfa_verified=True,
		)

	assert service.list_identities("tenant-a")[0]["email"] == "same-a@example.com"
	assert service.list_identities("tenant-b")[0]["email"] == "same-b@example.com"
	assert service.list_sessions("tenant-a")[0]["device_id"] == "device-tenant-a"
	assert service.list_sessions("tenant-b")[0]["device_id"] == "device-tenant-b"
	revoked = service.revoke_session("same-session", actor="tenant-owner", tenant_id="tenant-a")
	assert revoked["status"] == "revoked"
	assert service.list_sessions("tenant-a")[0]["status"] == "revoked"
	assert service.list_sessions("tenant-b")[0]["status"] == "active"

	with pytest.raises(ValueError, match="identity already exists"):
		service.register_identity(
			user_id="same-user",
			tenant_id="tenant-a",
			email="duplicate@example.com",
			display_name="Duplicate",
		)


def test_auth_security_agents_and_bytewax_guardrails():
	service = AuthService()
	agent = service.register_security_agent(
		agent_id="security-agent-1",
		tenant_id="tenant-auth-agent",
		name="Role Review Assistant",
		runtime="claude-code",
		role="role-reviewer",
		scope="privileged role review summaries",
		contribution_disclosed=True,
		policy_ref="auth-agent-policy",
	)
	batch = service.validate_batch_auth_mutation(
		tenant_id="tenant-auth-agent",
		event_stream="bytewax",
		mutation_count=2,
	)
	dashboard = view_models.dashboard_model(service, "tenant-auth-agent")
	agents = view_models.security_agents_model(service, "tenant-auth-agent")
	analytics = view_models.analytics_model(service, "tenant-auth-agent")
	settings = view_models.settings_model("tenant-auth-agent")

	assert agent["runtime"] == "claude_code"
	assert agent["role"] == "role_reviewer"
	assert batch["accepted"] is True
	assert dashboard["security_agents"][0]["id"] == "security-agent-1"
	assert agents["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert analytics["summary"]["security_agent_count"] == 1
	assert settings["streaming"]["processor"] == "bytewax"

	with pytest.raises(PermissionError, match="security_agent_runtime_not_supported"):
		service.register_security_agent(
			agent_id="bad-runtime",
			tenant_id="tenant-auth-agent",
			name="Bad Runtime",
			runtime="unsupported",
			role="role_reviewer",
			scope="role review",
		)

	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch_auth_mutation(
			tenant_id="tenant-auth-agent",
			event_stream="memory",
			mutation_count=1,
		)


def test_api_helpers_and_view_models_expose_auth_lifecycle():
	identity = api_helpers.register_identity({
		"id": "api-user",
		"tenant_id": "tenant-api-auth",
		"email": "api@example.com",
		"display_name": "API User",
		"mfa_enabled": "true",
		"privacy_budget": 0.5,
	})
	role = api_helpers.define_role({
		"id": "api-role",
		"tenant_id": identity["tenant_id"],
		"name": "API Admin",
		"permissions": ["auth:admin"],
		"tier": "admin",
	})
	_grant_api_actor(identity["tenant_id"], "api-requester", "auth:manage_roles")
	_grant_api_actor(identity["tenant_id"], "api-reviewer", "auth:approve_roles")
	_grant_api_actor(identity["tenant_id"], "api-owner", "auth:manage_roles")
	_grant_api_actor(identity["tenant_id"], "api-privacy-requester", "auth:manage_privacy")
	_grant_api_actor(identity["tenant_id"], "api-privacy-reviewer", "auth:approve_privacy")
	approval_request = api_helpers.request_role_assignment_approval({
		"id": "api-approval",
		"tenant_id": identity["tenant_id"],
		"user_id": identity["id"],
		"role_id": role["id"],
		"requested_by": "api-requester",
		"justification": "Provision API admin.",
	})
	approval = api_helpers.decide_role_assignment_approval({
		"id": approval_request["id"],
		"tenant_id": identity["tenant_id"],
		"reviewer": "api-reviewer",
		"decision": "approved",
		"notes": "Approved for test.",
	})
	assignment = api_helpers.assign_role({
		"id": "api-assignment",
		"tenant_id": identity["tenant_id"],
		"user_id": identity["id"],
		"role_id": role["id"],
		"assigned_by": "api-owner",
		"approval_id": approval["id"],
	})
	session = api_helpers.start_session({
		"id": "api-session",
		"tenant_id": identity["tenant_id"],
		"user_id": identity["id"],
		"device_id": "api-device",
		"mfa_verified": "true",
	})
	decision = api_helpers.evaluate_access({
		"id": "api-decision",
		"tenant_id": identity["tenant_id"],
		"user_id": identity["id"],
		"permission": "auth:admin",
		"session_id": session["id"],
		"requested_permission_tier": "privileged",
	})
	privacy = api_helpers.run_privacy_query({
		"id": "api-privacy",
		"tenant_id": identity["tenant_id"],
		"user_id": identity["id"],
		"query_type": "risk_histogram",
		"epsilon_cost": 0.75,
	})
	privacy_approval_request = api_helpers.request_privacy_budget_approval({
		"id": "api-privacy-approval",
		"tenant_id": identity["tenant_id"],
		"user_id": identity["id"],
		"query_type": "risk_histogram",
		"epsilon_cost": 0.75,
		"requested_by": "api-privacy-requester",
		"justification": "Approved analytics override.",
	})
	privacy_approval = api_helpers.decide_privacy_budget_approval({
		"id": privacy_approval_request["id"],
		"tenant_id": identity["tenant_id"],
		"reviewer": "api-privacy-reviewer",
		"decision": "approved",
		"notes": "Budget override is acceptable for test.",
	})
	approved_privacy = api_helpers.run_privacy_query({
		"id": "api-privacy-approved",
		"tenant_id": identity["tenant_id"],
		"user_id": identity["id"],
		"query_type": "risk_histogram",
		"epsilon_cost": 0.75,
		"approval_id": privacy_approval["id"],
	})
	dashboard = view_models.dashboard_model(api_helpers.SERVICE, identity["tenant_id"])
	approval_queue = view_models.approval_queue_model(api_helpers.SERVICE, identity["tenant_id"])
	privacy_center = view_models.privacy_center_model(api_helpers.SERVICE, identity["tenant_id"])

	assert assignment["approval_recorded"] is True
	assert decision["decision"] == "allow"
	assert privacy["status"] == "review_required"
	assert approved_privacy["status"] == "completed"
	assert approved_privacy["approval_recorded"] is True
	assert api_helpers.capability_status(identity["tenant_id"])["role_approval_count"] == 1
	assert dashboard["summary"]["admin_assignment_count"] == 1
	assert dashboard["privacy_approvals"][0]["id"] == "api-privacy-approval"
	assert approval_queue["decided_approvals"][0]["id"] == "api-approval"
	assert privacy_center["decided_approvals"][0]["id"] == "api-privacy-approval"
	assert view_models.dashboard_model(tenant_id=identity["tenant_id"])["summary"]["admin_assignment_count"] == 1


def test_api_helpers_expose_security_agents_and_batch_guardrail():
	agent = api_helpers.register_security_agent({
		"id": "api-security-agent",
		"tenant_id": "tenant-api-security-agent",
		"name": "API Security Agent",
		"runtime": "opencode",
		"role": "identity_reviewer",
		"scope": "identity risk review",
		"contribution_disclosed": True,
	})
	batch = api_helpers.validate_batch_auth_mutation({
		"tenant_id": "tenant-api-security-agent",
		"event_stream": "bytewax",
		"mutation_count": 1,
	})

	assert agent["runtime"] == "opencode"
	assert api_helpers.list_security_agents("tenant-api-security-agent")[0]["id"] == "api-security-agent"
	assert batch["accepted"] is True


def test_auth_service_infers_privileged_access_tier_from_permission_and_role():
	service = AuthService()
	service.register_identity(
		user_id="admin-user",
		tenant_id="tenant-tier",
		email="admin-user@example.com",
		display_name="Admin User",
	)
	service.define_role(
		role_id="role-admin",
		tenant_id="tenant-tier",
		name="Tenant Administrator",
		permissions=["auth:admin"],
		tier="admin",
	)
	_grant_actor(service, "tenant-tier", "tier-requester", "auth:manage_roles")
	_grant_actor(service, "tenant-tier", "tier-reviewer", "auth:approve_roles")
	_grant_actor(service, "tenant-tier", "tier-owner", "auth:manage_roles")
	approval_request = service.request_role_assignment_approval(
		approval_id="tier-approval",
		tenant_id="tenant-tier",
		user_id="admin-user",
		role_id="role-admin",
		requested_by="tier-requester",
		justification="Admin access for tier inference.",
	)
	approval = service.decide_role_assignment_approval(
		approval_id=approval_request["id"],
		tenant_id="tenant-tier",
		reviewer="tier-reviewer",
		decision="approved",
		notes="Approved with separation of duties.",
	)
	service.assign_role(
		assignment_id="tier-assignment",
		tenant_id="tenant-tier",
		user_id="admin-user",
		role_id="role-admin",
		assigned_by="tier-owner",
		approval_id=approval["id"],
	)
	session = service.start_session(
		session_id="tier-session",
		tenant_id="tenant-tier",
		user_id="admin-user",
		device_id="tier-device",
		mfa_verified=False,
	)
	decision = service.evaluate_access(
		decision_id="tier-decision",
		tenant_id="tenant-tier",
		user_id="admin-user",
		permission="auth:admin",
		session_id=session["id"],
	)

	assert decision["decision"] == "deny"
	assert "mfa_required" in decision["reasons"]


def test_auth_service_tracks_privacy_review_and_denied_access():
	service = AuthService()
	service.register_identity(
		user_id="user-1",
		tenant_id="tenant-a",
		email="ada@example.com",
		display_name="Ada Lovelace",
		privacy_budget=0.1,
	)
	service.define_role(
		role_id="role-viewer",
		tenant_id="tenant-a",
		name="Viewer",
		permissions=["auth:view"],
	)
	service.assign_role(
		assignment_id="assign-1",
		tenant_id="tenant-a",
		user_id="user-1",
		role_id="role-viewer",
		assigned_by="system",
	)
	_grant_actor(service, "tenant-a", "privacy-requester", "auth:manage_privacy")
	_grant_actor(service, "tenant-a", "privacy-requester", "auth:approve_privacy")
	_grant_actor(service, "tenant-a", "privacy-reviewer", "auth:approve_privacy")
	session = service.start_session(
		session_id="session-1",
		tenant_id="tenant-a",
		user_id="user-1",
		device_id="device-1",
	)
	decision = service.evaluate_access(
		decision_id="decision-1",
		tenant_id="tenant-a",
		user_id="user-1",
		permission="auth:admin",
		session_id=session["id"],
	)
	raw_bypass = service.run_privacy_query(
		query_id="query-1",
		tenant_id="tenant-a",
		user_id="user-1",
		query_type="risk_histogram",
		epsilon_cost=0.2,
		approval_recorded=True,
	)
	approval_request = service.request_privacy_budget_approval(
		approval_id="privacy-approval-1",
		tenant_id="tenant-a",
		user_id="user-1",
		query_type="risk_histogram",
		epsilon_cost=0.2,
		requested_by="privacy-requester",
		justification="Budget override for governed analytics.",
	)
	with pytest.raises(PermissionError, match="independent_privacy_budget_reviewer_required"):
		service.decide_privacy_budget_approval(
			approval_id=approval_request["id"],
			tenant_id="tenant-a",
			reviewer="privacy-requester",
			decision="approved",
			notes="Self approved.",
		)
	approval = service.decide_privacy_budget_approval(
		approval_id=approval_request["id"],
		tenant_id="tenant-a",
		reviewer="privacy-reviewer",
		decision="approved",
		notes="Approved with privacy ticket.",
	)
	query = service.run_privacy_query(
		query_id="query-2",
		tenant_id="tenant-a",
		user_id="user-1",
		query_type="risk_histogram",
		epsilon_cost=0.2,
		approval_id=approval["id"],
	)
	service.register_identity(
		user_id="tenant-a-only",
		tenant_id="tenant-a",
		email="tenant-a-only@example.com",
		display_name="Tenant A Only",
	)
	with pytest.raises(PermissionError, match="tenant_membership_required"):
		service.run_privacy_query(
			query_id="query-cross",
			tenant_id="tenant-b",
			user_id="tenant-a-only",
			query_type="risk_histogram",
			epsilon_cost=0.1,
		)

	assert decision["decision"] == "deny"
	assert decision["reasons"] == ["mfa_required"]
	assert raw_bypass["status"] == "review_required"
	assert raw_bypass["approval_recorded"] is False
	assert raw_bypass["reasons"] == ["privacy_budget_exhausted"]
	assert query["status"] == "completed"
	assert query["approval_id"] == "privacy-approval-1"
	summary = service.dashboard_summary("tenant-a")
	assert summary["denied_decision_count"] == 1
	assert summary["privacy_review_count"] == 1
	assert summary["privacy_approval_count"] == 1
	assert summary["pending_privacy_approval_count"] == 0
