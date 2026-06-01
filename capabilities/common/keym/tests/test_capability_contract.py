"""Regression coverage for the KEYM executable capability contract."""

from __future__ import annotations

import pytest

from capabilities.common.keym import register_capability
from capabilities.common.keym.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-vault",
		{"lifecycle": {"default_rotation_days": 60}},
	)

	assert contract["capability"] == "keym"
	assert contract["configuration"]["tenant_id"] == "tenant-vault"
	assert contract["configuration"]["lifecycle"]["default_rotation_days"] == 60
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"key_domains",
		"lifecycle",
		"access",
		"hsm",
		"compliance",
		"automation",
		"operation_governance",
		"ui",
		"theme",
		"agents",
		"streaming",
	]
	assert len(contract["rule_engine"]["rules"]) >= 14
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"inventory",
		"lifecycle",
		"export_approvals",
		"rotation_exceptions",
		"compromise",
		"policies",
		"hsm",
		"audit",
		"analytics",
		"agents",
		"settings",
	}
	assert contract["ui"]["api_prefix"] == "/keym/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert {
		"key_inventory_row",
		"export_approval_queue",
		"rotation_exception_queue",
		"compromise_response_panel",
		"key_audit_timeline",
		"key_agent_roster",
		"bytewax_stream_indicator",
	} <= set(contract["theme"]["components"])
	assert contract["agents"]["first_class"] is True
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["streaming"]["engine"] == "bytewax"
	assert "review_evidence" in contract["provides"]
	assert contract["review_evidence"]["pending_queues"] == [
		"operations",
		"export_approvals",
		"rotation_exceptions",
		"rotations",
		"key_agents",
		"key_lifecycle_batches",
	]
	assert "policy_decision" in contract["review_evidence"]["policy_fields"]


def test_rule_engine_enforces_key_governance_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "export_key",
		"policy_attached": False,
		"key_class": "root",
		"hsm_attested": False,
		"dual_control_approved": False,
		"rotation_age_days": 120,
		"rotation_exception_recorded": False,
		"key_status": "compromised",
		"operation_is_cryptographic": True,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"root_key_requires_hsm_attestation",
		"export_requires_dual_control",
		"overdue_rotation_requires_review",
		"compromised_key_blocks_use",
	}


@pytest.mark.parametrize(
	"context, reason",
	[
		({"reviewer_same_as_requester": True, "review_notes_attached": True}, "independent_reviewer_required"),
		({"reviewer_same_as_requester": False, "review_notes_attached": False}, "review_notes_required"),
		({"operation": "complete_rotation", "key_rotation_evidence_attached": False}, "key_rotation_evidence_required"),
		(
			{
				"operation": "register_key_agent",
				"key_agent_runtime_supported": False,
				"key_agent_role_supported": True,
				"key_agent_scope_attached": True,
				"key_agent_privileged_role": False,
				"human_approval_required": False,
			},
			"key_agent_runtime_not_supported",
		),
		(
			{
				"operation": "register_key_agent",
				"key_agent_runtime_supported": True,
				"key_agent_role_supported": True,
				"key_agent_scope_attached": True,
				"key_agent_privileged_role": True,
				"human_approval_required": False,
			},
			"key_agent_privileged_role_requires_human_approval",
		),
		(
			{
				"operation": "validate_key_lifecycle_batch",
				"event_stream": "legacy_queue",
			},
			"bytewax_key_stream_required",
		),
	],
)
def test_rule_engine_enforces_review_and_rotation_guardrails(context, reason):
	result = evaluate_capability_rules(context)

	expected_decision = "require_review" if reason == "key_agent_privileged_role_requires_human_approval" else "deny"
	assert result["decision"] == expected_decision
	assert result["actions"][0]["reason"] == reason


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "keym_vault_console"
	assert registration["ui_components"]["inventory"] == "/keym/keys"
	assert registration["ui_components"]["export_approvals"] == "/keym/export-approvals"
	assert registration["ui_components"]["rotation_exceptions"] == "/keym/rotation-exceptions"
	assert registration["ui_components"]["agents"] == "/keym/agents"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["engine"] == "bytewax"
	assert registration["review_evidence"]["deny_behavior"] == "Denied key lifecycle batches persist evidence before PermissionError"
	assert "secu" in registration["dependencies"]
	assert "keym.approve_export" in registration["permissions"]
	assert "keym.respond_compromise" in registration["permissions"]
