"""Regression coverage for the AUTH executable capability contract."""

import pytest

from .. import get_capability_info, register_capability
from ..capability_contract import evaluate_capability_rules, get_capability_contract
from ..service import AuthService


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
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 7
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"login",
		"dashboard",
		"biometric_management",
		"quantum_keys",
		"behavioral_analysis",
		"privacy_analytics",
		"federated_mesh",
		"metrics"
	}
	assert contract["ui"]["api_prefix"] == "/api"
	assert contract["theme"]["tokens"]["border.radius"] == "12px"
	assert "risk_posture_meter" in contract["theme"]["components"]


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

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"locked_accounts_denied",
		"privileged_access_requires_mfa",
		"high_risk_sessions_require_step_up",
		"elevated_role_assignment_requires_approval",
		"untrusted_federation_denied",
		"cross_tenant_access_requires_membership",
	}


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
		approval_recorded=True,
	)
	assignment = service.assign_role(
		assignment_id="assign-1",
		tenant_id="tenant-a",
		user_id="user-1",
		role_id="role-admin",
		assigned_by="security-owner",
		approval_recorded=True,
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
	assert assignment["approval_recorded"] is True
	assert session["trust_score"] > 0.7
	assert decision["decision"] == "allow"
	assert decision["role_ids"] == ["role-admin"]
	assert privacy_query["status"] == "completed"
	assert privacy_query["remaining_budget"] == 0.5
	assert service.dashboard_summary("tenant-a") == {
		"tenant_id": "tenant-a",
		"identity_count": 1,
		"active_session_count": 1,
		"role_count": 1,
		"admin_assignment_count": 1,
		"denied_decision_count": 0,
		"privacy_review_count": 0,
		"average_trust_score": session["trust_score"],
	}
	assert service.list_records("tenant-a")[0]["id"] == "user-1"
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
		assigned_by="security-owner",
	)
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
	query = service.run_privacy_query(
		query_id="query-1",
		tenant_id="tenant-a",
		user_id="user-1",
		query_type="risk_histogram",
		epsilon_cost=0.2,
	)

	assert decision["decision"] == "deny"
	assert decision["reasons"] == ["permission_not_granted"]
	assert query["status"] == "review_required"
	assert query["reasons"] == ["privacy_budget_exhausted"]
	summary = service.dashboard_summary("tenant-a")
	assert summary["denied_decision_count"] == 1
	assert summary["privacy_review_count"] == 1
