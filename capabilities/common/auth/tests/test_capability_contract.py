"""Regression coverage for the AUTH executable capability contract."""

from .. import get_capability_info, register_capability
from ..capability_contract import evaluate_capability_rules, get_capability_contract


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
