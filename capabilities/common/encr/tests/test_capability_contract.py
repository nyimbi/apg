"""Regression coverage for the ENCR executable capability contract."""

from __future__ import annotations

import pytest

from capabilities.common.encr import register_capability
from capabilities.common.encr.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-crypto",
		{"cryptography": {"minimum_entropy_quality": 0.98}},
	)

	assert contract["capability"] == "encr"
	assert contract["configuration"]["tenant_id"] == "tenant-crypto"
	assert contract["configuration"]["cryptography"]["minimum_entropy_quality"] == 0.98
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"cryptography",
		"key_lifecycle",
		"policy",
		"threat_adaptive",
		"operation_governance",
		"compliance",
		"ui",
		"theme",
		"agents",
		"streaming",
	]
	assert len(contract["rule_engine"]["rules"]) >= 14
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"operations",
		"keys",
		"policies",
		"entropy",
		"exceptions",
		"rotations",
		"homomorphic",
		"analytics",
		"audit",
		"agents",
		"settings",
	}
	assert contract["ui"]["api_prefix"] == "/encr/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert {
		"entropy_quality_meter",
		"crypto_operation_queue",
		"crypto_exception_queue",
		"key_rotation_timeline",
		"crypto_audit_timeline",
		"crypto_agent_roster",
		"bytewax_stream_indicator",
	} <= set(contract["theme"]["components"])
	assert contract["agents"]["first_class"] is True
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["streaming"]["engine"] == "bytewax"
	assert "review_evidence" in contract["provides"]
	assert contract["review_evidence"]["pending_queues"] == [
		"operations",
		"exception_reviews",
		"rotations",
		"crypto_agents",
		"crypto_lifecycle_batches",
	]
	assert "policy_decision" in contract["review_evidence"]["policy_fields"]


def test_rule_engine_enforces_crypto_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"data_classification": "restricted",
		"algorithm_quantum_safe": False,
		"plaintext_export_requested": True,
		"entropy_quality": 0.8,
		"operation": "generate_key",
		"algorithm_family": "legacy",
		"security_review_recorded": False,
		"active_threat_signal": True,
		"key_rotation_completed": False,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"restricted_data_requires_quantum_safe_algorithm",
		"plaintext_export_blocked",
		"low_entropy_blocks_key_generation",
		"legacy_algorithm_requires_review",
		"active_threat_requires_key_rotation",
	}


@pytest.mark.parametrize(
	"context, reason",
	[
		(
			{
				"operation": "decide_crypto_exception",
				"crypto_exception_reviewer_same_as_requester": True,
				"crypto_exception_notes_attached": True,
			},
			"independent_crypto_reviewer_required",
		),
		(
			{
				"operation": "decide_crypto_exception",
				"crypto_exception_reviewer_same_as_requester": False,
				"crypto_exception_notes_attached": False,
			},
			"crypto_exception_notes_required",
		),
		(
			{
				"operation": "complete_key_rotation",
				"key_rotation_evidence_attached": False,
			},
			"key_rotation_evidence_required",
		),
		(
			{
				"operation": "register_crypto_agent",
				"crypto_agent_runtime_supported": False,
				"crypto_agent_role_supported": True,
				"crypto_agent_scope_attached": True,
				"crypto_agent_privileged_role": False,
				"human_approval_required": False,
			},
			"crypto_agent_runtime_not_supported",
		),
		(
			{
				"operation": "register_crypto_agent",
				"crypto_agent_runtime_supported": True,
				"crypto_agent_role_supported": True,
				"crypto_agent_scope_attached": True,
				"crypto_agent_privileged_role": True,
				"human_approval_required": False,
			},
			"crypto_agent_privileged_role_requires_human_approval",
		),
		(
			{
				"operation": "validate_crypto_lifecycle_batch",
				"event_stream": "legacy_queue",
			},
			"bytewax_crypto_stream_required",
		),
	],
)
def test_rule_engine_enforces_exception_and_rotation_guardrails(context, reason):
	result = evaluate_capability_rules(context)

	expected_decision = "require_review" if reason == "crypto_agent_privileged_role_requires_human_approval" else "deny"
	assert result["decision"] == expected_decision
	assert result["actions"][0]["reason"] == reason


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "encr_quantum_guard"
	assert registration["ui_components"]["homomorphic"] == "/encr/homomorphic"
	assert registration["ui_components"]["exceptions"] == "/encr/exceptions"
	assert registration["ui_components"]["rotations"] == "/encr/rotations"
	assert registration["ui_components"]["agents"] == "/encr/agents"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["engine"] == "bytewax"
	assert registration["review_evidence"]["deny_behavior"] == "Denied crypto lifecycle batches persist evidence before PermissionError"
	assert "secu" in registration["dependencies"]
	assert "encr:review" in registration["permissions"]
	assert "encr:rotate" in registration["permissions"]
