"""Regression coverage for the USRM executable capability contract."""

from capabilities.common.usrm import register_capability
from capabilities.common.usrm.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract,
	streaming_manifest,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-usrm", {"lifecycle": {"bulk_action_review_threshold": 10}})

	assert contract["capability"] == "usrm"
	assert contract["configuration"]["tenant_id"] == "tenant-usrm"
	assert contract["configuration"]["lifecycle"]["bulk_action_review_threshold"] == 10
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"users",
		"lifecycle",
		"access",
		"usrm_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert contract["requires"] == ["auth", "mfau", "cons", "audl", "idfd"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "users", "profiles", "lifecycle", "access", "privacy", "deprovisioning", "agents", "policy", "settings"}
	assert contract["theme"]["name"] == "usrm_user_lifecycle"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "usrm_agents" in contract["provides"]


def test_rule_engine_enforces_usrm_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_user", "unique_identity_present": False, "user_owner_assigned": False, "profile_validated": False, "privileged_user": True, "mfa_enabled": False, "affected_user_count": 40, "bulk_review_recorded": False})
	invite_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "invite_user", "consent_notice_attached": False, "event_stream": "local"})
	deprovision_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "deprovision_user", "access_revoked": False, "deprovision_evidence_present": False, "event_stream": "local"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "user_requires_identity", "user_requires_owner", "user_requires_profile_validation", "privileged_user_requires_mfa", "bulk_user_action_requires_review"}
	assert invite_result["matched_rules"] == ["invite_requires_consent_notice", "invite_requires_bytewax_stream"]
	assert deprovision_result["matched_rules"] == ["deprovision_requires_access_revocation", "deprovision_requires_evidence", "deprovision_requires_bytewax_stream"]


def test_agent_and_streaming_rules_are_exposed():
	agent_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "register_usrm_agent", "agent_runtime_supported": False, "agent_role_supported": False})
	privileged_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "agent_user_action", "privileged_scope": True, "human_approval_recorded": False})
	batch_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "bulk_user_action", "event_stream": "local", "affected_user_count": 30, "bulk_review_recorded": False})

	assert streaming_manifest()["stream"] == "apg.usrm.lifecycle"
	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) == {"usrm_agent_runtime_supported", "usrm_agent_role_supported"}
	assert privileged_result["matched_rules"] == ["privileged_agent_user_action_requires_human_approval"]
	assert set(batch_result["matched_rules"]) == {"bulk_user_action_requires_review", "bulk_user_action_requires_bytewax"}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "usrm"
	assert "audl" in registration["dependencies"]
	assert registration["ui_components"]["access"] == "/usrm/access"
	assert registration["ui_components"]["agents"] == "/usrm/agents"
	assert registration["streaming"]["processor"] == "bytewax"
	assert "usrm:review_access" in registration["permissions"]
