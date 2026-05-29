"""Regression coverage for the PRED executable capability contract."""

import pytest

from capabilities.common.pred import register_capability
from capabilities.common.pred.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)
from capabilities.common.pred.service import PredService
from capabilities.common.pred import views


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


def test_service_runs_model_forecast_score_scenario_and_drift_lifecycle():
	service = PredService()
	tenant_id = "tenant-forecast"
	model = service.register_model(
		model_id="model-demand",
		tenant_id=tenant_id,
		name="Demand Forecast",
		owner="analytics",
		algorithm="gradient_boosted_tree",
		target="daily_demand",
		environment="production",
		approved=True,
		explainability_attached=True,
		training_history_points=48,
		feature_names=["demand", "season", "promotion"],
	)
	feature_set = service.register_feature_set(
		feature_set_id="features-demand",
		tenant_id=tenant_id,
		name="Demand Features",
		owner="analytics",
		feature_names=["demand", "season", "promotion"],
		lineage_refs=["etlp://pipelines/demand/features"],
		source_system="etlp",
	)
	forecast = service.create_forecast(
		forecast_id="forecast-week",
		tenant_id=tenant_id,
		model_id=model["id"],
		series_name="daily demand",
		history_values=[100 + index for index in range(24)],
		horizon_days=7,
		actor="planner",
	)
	score = service.score_entity(
		score_id="score-order-1",
		tenant_id=tenant_id,
		model_id=model["id"],
		feature_set_id=feature_set["id"],
		entity_id="order-1",
		feature_values={"demand": 43, "season": 12, "promotion": True},
		environment="production",
		impact="high",
		explanation_ref="explain://score-order-1",
		actor="planner",
	)
	scenario = service.simulate_scenario(
		scenario_id="scenario-promo",
		tenant_id=tenant_id,
		model_id=model["id"],
		name="Promotion lift",
		baseline_score=score["score"],
		adjustments={"promotion": 5.0, "stockout": -2.0},
		assumptions=["promotion starts Monday"],
		actor="planner",
	)
	drift = service.record_drift(
		report_id="drift-demand",
		tenant_id=tenant_id,
		model_id=model["id"],
		metric_name="population_stability_index",
		drift_score=0.42,
		threshold=0.30,
		actor="monitor",
	)

	summary = service.dashboard_summary(tenant_id)
	dashboard = views.dashboard_model(service, tenant_id)
	score_monitor = views.score_monitor_model(service, tenant_id)
	governance = views.governance_model(service, tenant_id)

	assert forecast["history_points"] == 24
	assert len(forecast["forecast_values"]) == 7
	assert score["score"] > 0
	assert scenario["delta"] == 3.0
	assert drift["status"] == "review_required"
	assert summary["approved_model_count"] == 1
	assert dashboard["summary"]["forecast_count"] == 1
	assert score_monitor["scores"][0]["id"] == "score-order-1"
	assert governance["drift_reports"][0]["status"] == "review_required"


def test_service_enforces_predictive_governance_guardrails():
	service = PredService()
	tenant_id = "tenant-risk"

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_model(
			model_id="missing-tenant",
			tenant_id="",
			name="Missing Tenant",
			owner="analytics",
			algorithm="linear",
			target="risk",
		)

	with pytest.raises(PermissionError, match="model_owner_required"):
		service.register_model(
			model_id="missing-owner",
			tenant_id=tenant_id,
			name="Missing Owner",
			owner="",
			algorithm="linear",
			target="risk",
		)

	unapproved = service.register_model(
		model_id="model-risk",
		tenant_id=tenant_id,
		name="Risk Model",
		owner="analytics",
		algorithm="linear",
		target="risk",
		approved=False,
		explainability_attached=False,
		training_history_points=24,
	)
	lineage = service.register_feature_set(
		feature_set_id="features-risk",
		tenant_id=tenant_id,
		name="Risk Features",
		owner="analytics",
		feature_names=["risk", "age"],
		lineage_refs=["etlp://risk/features"],
		source_system="etlp",
	)
	no_lineage = service.register_feature_set(
		feature_set_id="features-no-lineage",
		tenant_id=tenant_id,
		name="No Lineage",
		owner="analytics",
		feature_names=["risk"],
		lineage_refs=[],
		source_system="manual",
	)

	with pytest.raises(PermissionError, match="insufficient_history"):
		service.create_forecast(
			forecast_id="short-history",
			tenant_id=tenant_id,
			model_id=unapproved["id"],
			series_name="risk",
			history_values=[1.0] * 23,
			horizon_days=7,
		)

	with pytest.raises(PermissionError, match="forecast_horizon_required"):
		service.create_forecast(
			forecast_id="missing-horizon",
			tenant_id=tenant_id,
			model_id=unapproved["id"],
			series_name="risk",
			history_values=[1.0] * 24,
			horizon_days=0,
		)

	with pytest.raises(PermissionError, match="long_horizon_review_required"):
		service.create_forecast(
			forecast_id="long-horizon",
			tenant_id=tenant_id,
			model_id=unapproved["id"],
			series_name="risk",
			history_values=[1.0] * 24,
			horizon_days=366,
			review_recorded=False,
		)

	with pytest.raises(PermissionError, match="approved_model_required"):
		service.score_entity(
			score_id="unapproved-score",
			tenant_id=tenant_id,
			model_id=unapproved["id"],
			feature_set_id=lineage["id"],
			entity_id="loan-1",
			feature_values={"risk": 20},
			environment="production",
		)

	service.approve_model(unapproved["id"], tenant_id, approver="governance")

	with pytest.raises(PermissionError, match="feature_lineage_required"):
		service.score_entity(
			score_id="missing-lineage-score",
			tenant_id=tenant_id,
			model_id=unapproved["id"],
			feature_set_id=no_lineage["id"],
			entity_id="loan-2",
			feature_values={"risk": 20},
			environment="production",
		)

	with pytest.raises(PermissionError, match="explainability_required"):
		service.score_entity(
			score_id="missing-explainability",
			tenant_id=tenant_id,
			model_id=unapproved["id"],
			feature_set_id=lineage["id"],
			entity_id="loan-3",
			feature_values={"risk": 20},
			environment="production",
			impact="high",
			explanation_ref="",
		)

	with pytest.raises(PermissionError, match="scenario_assumptions_required"):
		service.simulate_scenario(
			scenario_id="no-assumptions",
			tenant_id=tenant_id,
			model_id=unapproved["id"],
			name="No assumptions",
			baseline_score=50,
			adjustments={"risk": 1},
			assumptions=[],
		)


def test_service_preserves_compatibility_records_as_models():
	service = PredService()
	record = service.create_record(
		record_id="compat-model",
		tenant_id="tenant-compat",
		metadata={
			"owner": "compat",
			"algorithm": "deterministic",
			"target": "compatibility",
			"training_history_points": 24,
			"feature_names": ["compat_signal"],
		},
	)

	assert record["status"] == "approved"
	assert service.list_records("tenant-compat")[0]["id"] == "compat-model"
