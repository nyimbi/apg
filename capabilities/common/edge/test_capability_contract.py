"""Regression coverage for the EDGE executable capability contract."""

from capabilities.common.edge import register_capability
from capabilities.common.edge.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-edge", {"sync": {"max_offline_hours": 24}})

	assert contract["capability"] == "edge"
	assert contract["configuration"]["tenant_id"] == "tenant-edge"
	assert contract["configuration"]["sync"]["max_offline_hours"] == 24
	assert contract["configuration_schema"]["required"] == ["tenant_id", "nodes", "workloads", "sync", "governance", "ui", "theme"]
	assert contract["theme"]["name"] == "edge_operations_console"


def test_rule_engine_enforces_edge_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "register_node", "node_attested": False, "edge_connection": True, "secure_transport": False, "offline_hours": 100, "offline_review_recorded": False})
	deploy_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "deploy_workload", "artifact_signed": False})
	sync_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "sync_state", "conflict_policy_attached": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "node_requires_attestation", "edge_transport_requires_security", "long_offline_window_requires_review"}
	assert deploy_result["matched_rules"] == ["workload_requires_signed_artifact"]
	assert sync_result["matched_rules"] == ["sync_requires_conflict_policy"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "edge"
	assert "dist" in registration["dependencies"]
	assert registration["ui_components"]["nodes"] == "/edge/nodes"
	assert "edge:deploy_workloads" in registration["permissions"]
