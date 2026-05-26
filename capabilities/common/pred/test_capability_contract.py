"""Regression coverage for the PRED executable capability contract."""

from capabilities.common.pred import register_capability
from capabilities.common.pred.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-forecast", {"forecasting": {"horizon_limit": 90}})

	assert contract["capability"] == "pred"
	assert contract["configuration"]["tenant_id"] == "tenant-forecast"
	assert contract["configuration"]["forecasting"]["horizon_limit"] == 90
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"forecasting",
		"scoring",
		"models",
		"governance",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "forecasts", "scores", "scenarios", "models", "governance", "settings"}
	assert contract["ui"]["api_prefix"] == "/pred/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "forecast_chart" in contract["theme"]["components"]


def test_rule_engine_enforces_predictive_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "score",
		"history_points": 12,
		"environment": "production",
		"model_approved": False,
		"feature_lineage_present": False,
		"impact": "high",
		"explainability_attached": False,
		"forecast_horizon_days": 730,
		"review_recorded": False
	})
	forecast_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "create_forecast",
		"history_points": 12
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"production_score_requires_approved_model",
		"scoring_requires_feature_lineage",
		"high_impact_prediction_requires_explainability",
		"long_horizon_requires_review"
	}
	assert forecast_result["decision"] == "deny"
	assert forecast_result["matched_rules"] == ["forecast_requires_history"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "pred"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "pred_forecast_console"
	assert registration["ui_components"]["forecasts"] == "/pred/forecasts"
	assert "mlcm" in registration["dependencies"]
	assert "pred:forecast" in registration["permissions"]
