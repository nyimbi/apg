"""Regression coverage for the ONTO executable capability contract."""

from capabilities.common.onto import register_capability
from capabilities.common.onto.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-onto", {"mapping": {"confidence_threshold": 0.9}})

	assert contract["capability"] == "onto"
	assert contract["configuration"]["tenant_id"] == "tenant-onto"
	assert contract["configuration"]["mapping"]["confidence_threshold"] == 0.9
	assert contract["configuration_schema"]["required"] == ["tenant_id", "ontology", "vocabulary", "mapping", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "ontologies", "terms", "mappings", "publication", "governance", "settings"}
	assert contract["ui"]["api_prefix"] == "/onto/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "taxonomy_tree" in contract["theme"]["components"]


def test_rule_engine_enforces_ontology_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "publish_ontology",
		"owner_assigned": False,
		"approval_recorded": False,
		"change_type": "breaking",
		"review_recorded": False,
		"mapping_confidence": 0.2,
		"duplicate_term_detected": True
	})
	term_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "create_term",
		"owner_assigned": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"publication_requires_approval",
		"breaking_change_requires_review",
		"low_confidence_mapping_requires_review",
		"duplicate_term_blocks_publication"
	}
	assert term_result["decision"] == "deny"
	assert term_result["matched_rules"] == ["term_requires_owner"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "onto"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "onto_vocabulary_workbench"
	assert registration["ui_components"]["mappings"] == "/onto/mappings"
	assert "kngr" in registration["dependencies"]
	assert "onto:publish" in registration["permissions"]
