"""Regression coverage for the PRED executable capability contract."""

from __future__ import annotations

import pytest

from capabilities.common.pred import register_capability, views
from capabilities.common.pred.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract,
)
from capabilities.common.pred.service import PredService


def test_contract_exposes_full_lifecycle_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-forecast", {"forecasting": {"horizon_limit_days": 90}})

	assert contract["capability"] == "pred"
	assert contract["configuration"]["tenant_id"] == "tenant-forecast"
	assert contract["configuration"]["forecasting"]["horizon_limit_days"] == 90
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"forecasting",
		"scoring",
		"feature_sets",
		"models",
		"scenarios",
		"drift",
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert len(contract["rule_engine"]["rules"]) >= 39
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"forecasts",
		"scores",
		"features",
		"scenarios",
		"models",
		"drift",
		"batch",
		"explainability",
		"agents",
		"lifecycle",
		"governance",
		"audit",
		"settings",
	}
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "service.PredService"
	assert contract["agents"]["first_class"] is True
	assert {"codex", "claude_code", "opencode", "pi"} <= set(contract["agents"]["supported_runtimes"])
	assert "drift_reviewer" in contract["agents"]["privileged_roles"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert "prediction_agent_batch" in contract["streaming"]["required_operations"]
	assert next(route for route in contract["ui"]["routes"] if route["name"] == "audit")["permission"] == "pred:audit"
	assert contract["ui"]["api_prefix"] == "/pred/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert {"forecast_chart", "score_card", "drift_monitor", "batch_queue", "prediction_agent_roster", "bytewax_lifecycle_panel", "audit_timeline"} <= set(contract["theme"]["components"])


def test_rule_engine_enforces_predictive_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "score",
		"environment": "production",
		"model_approved": False,
		"feature_lineage_present": False,
		"impact": "high",
		"explainability_attached": False,
		"cross_tenant_scoring": True,
	})
	forecast_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "create_forecast",
		"model_present": True,
		"series_name_present": True,
		"history_points": 12,
		"forecast_horizon_days": 7,
		"review_recorded": False,
	})
	batch_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "configure_batch_scoring",
		"event_stream": "legacy_queue",
	})
	state_change_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"state_change_requested": True,
		"audit_event_recorded": False,
	})
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_prediction_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"scope_present": False,
		"owner_present": False,
		"purpose_present": False,
		"contribution_disclosed": False,
		"privileged_role": True,
		"human_approval_required": False,
	})
	lifecycle_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_pred_lifecycle_batch",
		"event_stream": "legacy_queue",
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"production_score_requires_approved_model",
		"scoring_requires_feature_lineage",
		"high_impact_prediction_requires_explainability",
		"cross_tenant_scoring_denied",
	}
	assert forecast_result["decision"] == "deny"
	assert forecast_result["matched_rules"] == ["forecast_requires_history"]
	assert batch_result["matched_rules"] == ["batch_scoring_requires_bytewax"]
	assert batch_result["actions"][0]["reason"] == "bytewax_event_stream_required"
	assert state_change_result["matched_rules"] == ["prediction_state_change_requires_audit"]
	assert state_change_result["actions"][0]["reason"] == "audit_event_required"
	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) >= {
		"prediction_agent_runtime_supported",
		"prediction_agent_role_supported",
		"prediction_agent_requires_scope",
		"prediction_agent_requires_owner",
		"prediction_agent_requires_purpose",
		"prediction_agent_requires_contribution_disclosure",
		"prediction_agent_privileged_role_requires_human_approval",
	}
	assert lifecycle_result["matched_rules"] == ["bytewax_pred_stream_required"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "pred"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "pred_forecast_console"
	assert registration["ui_components"]["forecasts"] == "/pred/forecasts"
	assert registration["ui_components"]["agents"] == "/pred/agents"
	assert registration["ui_components"]["audit"] == "/pred/audit"
	assert "mlcm" in registration["dependencies"]
	assert "pred:audit" in registration["permissions"]
	assert {"feature_registry", "drift_monitoring", "batch_scoring"} <= set(registration["capabilities"])


def test_service_runs_model_forecast_score_scenario_drift_and_audit_lifecycle():
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
		review_recorded=True,
		actor="monitor",
	)
	agent = service.register_prediction_agent(
		agent_id="prediction-agent-1",
		tenant_id=tenant_id,
		name="Prediction Steward",
		runtime="codex",
		role="prediction_steward",
		scope="demand forecasts and drift summaries",
		owner="analytics",
		purpose="review predictive analytics lifecycle changes",
	)
	batch = service.validate_pred_lifecycle_batch(tenant_id, "bytewax", 2, "prediction_agent_batch", "pred-batch-001")

	summary = service.dashboard_summary(tenant_id)
	dashboard = views.dashboard_model(service, tenant_id)
	score_monitor = views.score_monitor_model(service, tenant_id)
	feature_registry = views.feature_registry_model(service, tenant_id)
	drift_monitor = views.drift_monitor_model(service, tenant_id)
	batch_queue = views.batch_scoring_model(service, tenant_id)
	explainability = views.explainability_model(service, tenant_id)
	agent_roster = views.prediction_agent_roster_model(service, tenant_id)
	lifecycle = views.lifecycle_batch_model(service, tenant_id)
	audit_timeline = views.audit_timeline_model(service, tenant_id)

	assert forecast["history_points"] == 24
	assert len(forecast["forecast_values"]) == 7
	assert score["score"] > 0
	assert scenario["delta"] == 3.0
	assert drift["status"] == "review_required"
	assert drift["review_recorded"] is True
	assert agent["runtime"] == "codex"
	assert batch["required_processor"] == "bytewax"
	assert summary["approved_model_count"] == 1
	assert summary["prediction_agent_count"] == 1
	assert summary["lifecycle_batch_count"] == 1
	assert dashboard["summary"]["forecast_count"] == 1
	assert score_monitor["scores"][0]["id"] == "score-order-1"
	assert feature_registry["feature_sets"][0]["source_system"] == "etlp"
	assert drift_monitor["drift_reports"][0]["status"] == "review_required"
	assert batch_queue["streaming"]["required_processor"] == "bytewax"
	assert explainability["models"][0]["explainability_attached"] is True
	assert agent_roster["agents"][0]["id"] == "prediction-agent-1"
	assert lifecycle["batches"][0]["id"] == "pred-batch-001"
	assert audit_timeline["audit_events"]


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

	with pytest.raises(PermissionError, match="model_target_required"):
		service.register_model(
			model_id="missing-target",
			tenant_id=tenant_id,
			name="Missing Target",
			owner="analytics",
			algorithm="linear",
			target="",
		)

	with pytest.raises(PermissionError, match="training_history_review_required"):
		service.register_model(
			model_id="short-training-history",
			tenant_id=tenant_id,
			name="Short Training History",
			owner="analytics",
			algorithm="linear",
			target="risk",
			training_history_points=12,
			feature_names=["risk"],
		)

	with pytest.raises(PermissionError, match="feature_source_system_required"):
		service.register_feature_set(
			feature_set_id="features-no-source",
			tenant_id=tenant_id,
			name="No Source",
			owner="analytics",
			feature_names=["risk"],
			lineage_refs=["etlp://risk/features"],
			source_system="",
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
		feature_names=["risk", "age"],
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
	assert no_lineage["status"] == "review_required"

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

	service.approve_model(unapproved["id"], tenant_id, approver="governance", explainability_ref="explain://model-risk")

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

	with pytest.raises(PermissionError, match="score_entity_required"):
		service.score_entity(
			score_id="missing-entity",
			tenant_id=tenant_id,
			model_id=unapproved["id"],
			feature_set_id=lineage["id"],
			entity_id="",
			feature_values={"risk": 20},
			environment="development",
		)

	with pytest.raises(PermissionError, match="score_features_required"):
		service.score_entity(
			score_id="missing-features",
			tenant_id=tenant_id,
			model_id=unapproved["id"],
			feature_set_id=lineage["id"],
			entity_id="loan-4",
			feature_values={},
			environment="development",
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

	with pytest.raises(PermissionError, match="scenario_adjustments_required"):
		service.simulate_scenario(
			scenario_id="no-adjustments",
			tenant_id=tenant_id,
			model_id=unapproved["id"],
			name="No adjustments",
			baseline_score=50,
			adjustments={},
			assumptions=["portfolio unchanged"],
		)

	with pytest.raises(PermissionError, match="drift_metric_required"):
		service.record_drift(
			report_id="missing-metric",
			tenant_id=tenant_id,
			model_id=unapproved["id"],
			metric_name="",
			drift_score=0.5,
			threshold=0.3,
		)

	with pytest.raises(PermissionError, match="drift_review_required"):
		service.record_drift(
			report_id="high-drift-no-review",
			tenant_id=tenant_id,
			model_id=unapproved["id"],
			metric_name="population_stability_index",
			drift_score=0.5,
			threshold=0.3,
			review_recorded=False,
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


def test_service_enforces_prediction_agent_and_lifecycle_guardrails():
	service = PredService()
	tenant_id = "tenant-agents"

	with pytest.raises(PermissionError, match="unsupported_prediction_agent_runtime"):
		service.register_prediction_agent("agent-bad-runtime", tenant_id, "Bad Runtime", "unknown", "prediction_steward", "doc", "owner", "purpose")
	with pytest.raises(PermissionError, match="prediction_agent_scope_required"):
		service.register_prediction_agent("agent-no-scope", tenant_id, "No Scope", "codex", "prediction_steward", "", "owner", "purpose")
	with pytest.raises(PermissionError, match="prediction_agent_contribution_disclosure_required"):
		service.register_prediction_agent("agent-no-disclosure", tenant_id, "No Disclosure", "codex", "prediction_steward", "doc", "owner", "purpose", contribution_disclosed=False)

	agent = service.register_prediction_agent(
		"agent-review",
		tenant_id,
		"Drift Reviewer",
		"claude-code",
		"drift reviewer",
		"forecast drift review",
		"analytics",
		"review above-threshold drift decisions",
	)
	assert agent["runtime"] == "claude_code"
	assert agent["role"] == "drift_reviewer"
	assert agent["status"] == "pending_review"
	assert service.dashboard_summary(tenant_id)["pending_agent_review_count"] == 1

	with pytest.raises(ValueError, match="pred_lifecycle_batch_empty"):
		service.validate_pred_lifecycle_batch(tenant_id, "bytewax", 0)
	with pytest.raises(ValueError, match="unsupported_pred_lifecycle_operation"):
		service.validate_pred_lifecycle_batch(tenant_id, "bytewax", 1, "unknown_batch")
	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		service.validate_pred_lifecycle_batch(tenant_id, "legacy_queue", 1)

	assert service.list_lifecycle_batches(tenant_id)[0]["status"] == "denied"
	assert service.dashboard_summary(tenant_id)["denied_lifecycle_batch_count"] == 1
