"""Regression coverage for the MQEB executable capability contract."""

from capabilities.common.mqeb import register_capability
from capabilities.common.mqeb.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-events",
		{"broker": {"max_topics_per_tenant": 250}}
	)

	assert contract["capability"] == "mqeb"
	assert contract["configuration"]["tenant_id"] == "tenant-events"
	assert contract["configuration"]["broker"]["max_topics_per_tenant"] == 250
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"broker",
		"delivery",
		"routing",
		"security",
		"compliance",
		"scaling",
		"operation_governance",
		"agents",
		"streaming",
		"ui",
		"theme"
	]
	assert contract["configuration"]["operation_governance"]["bytewax_first_runtime"] is True
	assert contract["configuration"]["operation_governance"]["broker_core_dependency_allowed"] is False
	assert contract["agents"]["first_class"] is True
	assert "codex" in contract["agents"]["supported_runtimes"]
	assert "claude_code" in contract["agents"]["supported_runtimes"]
	assert "bytewax_topology_reviewer" in contract["agents"]["privileged_roles"]
	assert contract["streaming"]["engine"] == "bytewax"
	assert contract["streaming"]["broker_core_dependency_allowed"] is False
	assert len(contract["rule_engine"]["rules"]) >= 22
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"topics",
		"publish",
		"subscriptions",
		"delivery",
		"dead_letters",
		"quota_exceptions",
		"replays",
		"agents",
		"bytewax",
		"routing",
		"scaling",
		"monitoring",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/mqeb/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "message_flow_map" in contract["theme"]["components"]
	assert "bytewax_bridge_panel" in contract["theme"]["components"]
	assert "event_agent_roster" in contract["theme"]["components"]


def test_rule_engine_enforces_message_governance_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "publish",
		"topic_exists": False,
		"topic_classification": "restricted",
		"message_encrypted": False,
		"cross_tenant_publish": True,
		"delivery_mode": "exactly_once",
		"dead_letter_queue_configured": False,
		"idempotency_key_present": False,
		"topic_status": "disabled",
		"priority_messages_per_minute": 15000,
		"quota_exception_recorded": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"publish_requires_topic",
		"restricted_topic_requires_encryption",
		"cross_tenant_publish_denied",
		"guaranteed_delivery_requires_dead_letter_queue",
		"exactly_once_requires_idempotency_key",
		"disabled_topic_blocks_publish",
		"priority_quota_exhaustion_requires_review"
	}


def test_rule_engine_enforces_review_replay_and_delivery_guardrails():
	replay = evaluate_capability_rules({
		"operation": "replay",
		"replay_range_bounded": False,
		"replay_reason_present": False,
	})
	delivery = evaluate_capability_rules({
		"operation": "deliver",
		"subscription_status": "paused",
	})
	review = evaluate_capability_rules({
		"reviewer_same_as_requester": True,
		"review_notes_attached": False,
	})

	assert replay["decision"] == "deny"
	assert {action["reason"] for action in replay["actions"]} == {"replay_range_required", "replay_reason_required"}
	assert delivery["decision"] == "deny"
	assert delivery["actions"][0]["reason"] == "subscription_paused"
	assert review["decision"] == "deny"
	assert {action["reason"] for action in review["actions"]} == {"independent_reviewer_required", "review_notes_required"}


def test_rule_engine_enforces_regulated_topic_encryption_and_schema():
	result = evaluate_capability_rules({
		"operation": "publish",
		"topic_classification": "regulated",
		"message_encrypted": False,
		"schema_ref_present": False,
	})

	assert result["decision"] == "deny"
	assert {action["reason"] for action in result["actions"]} == {
		"message_encryption_required",
		"schema_reference_required",
	}


def test_rule_engine_enforces_event_agent_and_bytewax_guardrails():
	agent = evaluate_capability_rules({
		"operation": "register_event_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"agent_scope_present": False,
		"agent_owner_present": False,
		"agent_purpose_present": False,
		"contribution_disclosed": False,
		"privileged_agent_role": True,
		"human_approval_required": False,
	})
	stream = evaluate_capability_rules({
		"operation": "validate_event_lifecycle_batch",
		"event_stream": "custom-broker",
	})

	assert agent["decision"] == "deny"
	assert {action["reason"] for action in agent["actions"]} == {
		"unsupported_event_agent_runtime",
		"unsupported_event_agent_role",
		"event_agent_scope_required",
		"event_agent_owner_required",
		"event_agent_purpose_required",
		"event_agent_contribution_disclosure_required",
		"event_agent_human_approval_required",
	}
	assert stream["decision"] == "deny"
	assert stream["actions"][0]["reason"] == "bytewax_event_stream_required"


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "mqeb_event_fabric"
	assert registration["ui_components"]["routing"] == "/mqeb/routing"
	assert registration["ui_components"]["bytewax"] == "/mqeb/bytewax"
	assert registration["ui_components"]["agents"] == "/mqeb/agents"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["engine"] == "bytewax"
	assert "mqeb:review_quota" in registration["permissions"]
	assert "mqeb:manage_bytewax" in registration["permissions"]
	assert "mqeb:manage_agents" in registration["permissions"]
	assert "auth_rbac" in registration["dependencies"]
