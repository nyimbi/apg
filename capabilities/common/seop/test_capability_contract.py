"""Regression coverage for the SEOP executable capability contract."""

from capabilities.common.seop import register_capability
from capabilities.common.seop.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-seop", {"detection": {"confidence_threshold": 0.8}})

	assert contract["capability"] == "seop"
	assert contract["configuration"]["tenant_id"] == "tenant-seop"
	assert contract["configuration"]["detection"]["confidence_threshold"] == 0.8
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"detection",
		"incidents",
		"response",
		"seop_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert contract["requires"] == ["secu", "anom", "moni", "logt", "audl"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert "codex" in contract["configuration"]["seop_agents"]["supported_runtimes"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "detections", "incidents", "triage", "playbooks", "responses", "posture", "agents", "audit", "settings"}
	assert contract["theme"]["name"] == "seop_security_ops"


def test_rule_engine_enforces_seop_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "open_incident", "incident_owner_assigned": False, "incident_severity": "critical", "escalation_recorded": False, "evidence_attached": False, "anomaly_confidence": 0.95, "triage_review_recorded": False})
	detection_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "create_detection", "alert_source_present": False, "event_stream": "bytewax"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "incident_requires_owner", "critical_incident_requires_escalation", "incident_requires_evidence", "high_confidence_anomaly_requires_review"}
	assert detection_result["matched_rules"] == ["detection_requires_alert_source"]


def test_rules_enforce_bytewax_and_agent_guardrails():
	stream_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "create_detection", "alert_source_present": True, "event_stream": "memory"})
	agent_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "register_seop_agent", "agent_runtime_supported": False, "agent_role_supported": False})
	critical_action = evaluate_capability_rules({"tenant_context_present": True, "operation": "agent_response_action", "incident_severity": "critical", "human_approval_recorded": False})

	assert stream_result["decision"] == "deny"
	assert stream_result["matched_rules"] == ["detection_requires_bytewax_stream"]
	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) == {"seop_agent_runtime_supported", "seop_agent_role_supported"}
	assert critical_action["matched_rules"] == ["critical_agent_action_requires_human_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "seop"
	assert "anom" in registration["dependencies"]
	assert "audl" in registration["dependencies"]
	assert registration["streaming"]["processor"] == "bytewax"
	assert registration["ui_components"]["incidents"] == "/seop/incidents"
	assert registration["ui_components"]["agents"] == "/seop/agents"
	assert "seop:respond" in registration["permissions"]
