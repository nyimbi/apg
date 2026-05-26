"""Regression coverage for the RECS executable capability contract."""

from capabilities.common.recs import register_capability
from capabilities.common.recs.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-recs", {"models": {"minimum_training_events": 2500}})

	assert contract["capability"] == "recs"
	assert contract["configuration"]["tenant_id"] == "tenant-recs"
	assert contract["configuration"]["models"]["minimum_training_events"] == 2500
	assert contract["configuration_schema"]["required"] == ["tenant_id", "models", "ranking", "experiments", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "recommendations", "models", "catalogs", "profiles", "experiments", "policies", "settings"}
	assert contract["ui"]["api_prefix"] == "/recs/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "experiment_board" in contract["theme"]["components"]


def test_rule_engine_enforces_recommendation_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "recommend",
		"profile_consent_recorded": False,
		"ranking_policy_attached": False,
		"impact_level": "high",
		"explanation_attached": False,
		"experiment_percent": 50,
		"experiment_review_recorded": False
	})
	train_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "train_model", "training_event_count": 25})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "profile_consent_required", "ranking_policy_required", "high_impact_requires_explainability", "large_experiment_requires_review"}
	assert train_result["matched_rules"] == ["model_training_requires_events"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "recs"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "recs_recommendation_console"
	assert registration["ui_components"]["recommendations"] == "/recs/recommendations"
	assert "pred" in registration["dependencies"]
	assert "recs:recommend" in registration["permissions"]
