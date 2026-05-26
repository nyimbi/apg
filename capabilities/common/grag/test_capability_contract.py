"""Regression coverage for the GRAG executable capability contract."""

from capabilities.common.grag import register_capability
from capabilities.common.grag.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-grag", {"reasoning": {"max_hops": 2}})

	assert contract["capability"] == "grag"
	assert contract["configuration"]["tenant_id"] == "tenant-grag"
	assert contract["configuration"]["reasoning"]["max_hops"] == 2
	assert contract["configuration_schema"]["required"] == ["tenant_id", "retrieval", "reasoning", "curation", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "query", "reasoning", "graphs", "curation", "explanations", "settings"}
	assert contract["ui"]["api_prefix"] == "/grag/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "reasoning_path" in contract["theme"]["components"]


def test_rule_engine_enforces_graphrag_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"query_type": "hybrid",
		"vector_index_ready": False,
		"graph_index_ready": False,
		"operation": "generate_answer",
		"evidence_path_present": False,
		"hop_count": 7,
		"review_recorded": False,
		"provenance_attached": False
	})
	reason_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "reason",
		"evidence_path_present": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"hybrid_query_requires_vector_and_graph",
		"hybrid_query_requires_graph_index",
		"multi_hop_requires_review",
		"answer_requires_provenance"
	}
	assert reason_result["decision"] == "deny"
	assert reason_result["matched_rules"] == ["reasoning_requires_evidence_path"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "grag"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "grag_reasoning_console"
	assert registration["ui_components"]["reasoning"] == "/grag/reasoning"
	assert "ragn" in registration["dependencies"]
	assert "grag:reason" in registration["permissions"]
