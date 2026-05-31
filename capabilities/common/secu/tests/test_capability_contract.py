"""Regression coverage for the SECU executable capability contract."""

from __future__ import annotations

import pytest

from .. import get_capability_info, register_capability
from ..capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-a", {"risk": {"critical_threshold": 95}})

	assert contract["capability"] == "secu"
	assert contract["configuration"]["tenant_id"] == "tenant-a"
	assert contract["configuration"]["risk"]["critical_threshold"] == 95
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"zero_trust",
		"risk",
		"threat_detection",
		"compliance",
		"incident_response",
		"agents",
		"streaming",
		"ui",
		"theme",
	]
	assert len(contract["rule_engine"]["rules"]) >= 14
	assert contract["agents"]["first_class"] is True
	assert contract["streaming"]["engine"] == "bytewax"
	assert "security_agent_privileged_role_requires_human_approval" in contract["agents"]["guardrails"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"risk",
		"threats",
		"policies",
		"exceptions",
		"incidents",
		"quarantine",
		"compliance",
		"audit",
		"agents",
		"rules",
		"settings",
	}
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert {
		"risk_score_meter",
		"policy_exception_queue",
		"incident_response_panel",
		"device_quarantine_list",
		"security_audit_timeline",
		"security_agent_roster",
		"bytewax_stream_indicator",
	} <= set(contract["theme"]["components"])


def test_rule_engine_denies_high_risk_context():
	result = evaluate_capability_rules({
		"is_known_malicious": True,
		"device_trust": "compromised",
		"risk_score": 92,
		"challenge_completed": False,
		"compliance_violation": True,
		"audit_evidence_attached": False,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"known_malicious_network_denied",
		"compromised_device_quarantined",
		"critical_risk_denied",
		"high_risk_requires_challenge",
		"compliance_violation_alert",
	}


@pytest.mark.parametrize(
	"context, reason",
	[
		(
			{
				"operation": "register_security_agent",
				"agent_runtime_supported": False,
				"agent_role_supported": True,
				"agent_scope_present": True,
			},
			"security_agent_runtime_unsupported",
		),
		(
			{
				"operation": "register_security_agent",
				"agent_runtime_supported": True,
				"agent_role_supported": True,
				"agent_scope_present": True,
				"agent_privileged_role": True,
				"human_approval_required": False,
			},
			"security_agent_human_approval_required",
		),
		(
			{
				"operation": "security_lifecycle_batch",
				"event_stream": "memory",
			},
			"bytewax_security_stream_required",
		),
		(
			{
				"operation": "approve_policy_exception",
				"exception_reviewer_same_as_requester": True,
				"policy_exception_expired": False,
			},
			"independent_exception_reviewer_required",
		),
		(
			{
				"operation": "approve_policy_exception",
				"exception_reviewer_same_as_requester": False,
				"policy_exception_expired": True,
			},
			"policy_exception_expired",
		),
		(
			{
				"operation": "open_incident",
				"incident_severity": "critical",
				"containment_plan_attached": False,
			},
			"critical_incident_containment_required",
		),
		(
			{
				"operation": "resolve_incident",
				"containment_evidence_attached": False,
			},
			"incident_containment_evidence_required",
		),
	],
)
def test_rule_engine_enforces_exception_and_incident_guardrails(context, reason):
	result = evaluate_capability_rules(context)

	assert result["decision"] == "deny"
	assert result["actions"][0]["reason"] == reason


def test_capability_info_and_registration_include_manifest_theme_and_permissions():
	info = get_capability_info()
	registration = register_capability()

	assert info["metadata"]["capability_name"] == "secu"
	assert info["configuration"]["tenant_id"] == "default"
	assert info["ui_manifest"]["requires_theme"] is True
	assert info["agents"]["first_class"] is True
	assert info["streaming"]["engine"] == "bytewax"
	assert info["theme"]["name"] == "secu_zero_trust"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_components"]["policies"] == "/secu/policies"
	assert registration["ui_components"]["exceptions"] == "/secu/exceptions"
	assert registration["ui_components"]["incidents"] == "/secu/incidents"
	assert registration["ui_components"]["agents"] == "/secu/agents"
	assert registration["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert "secu:approve_exception" in registration["permissions"]
	assert "secu:respond" in registration["permissions"]
