"""Regression coverage for the RECS executable capability contract."""

import pytest

from capabilities.common.recs import register_capability
from capabilities.common.recs.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.recs.service import RecsService
from capabilities.common.recs import views


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


def test_recommendation_lifecycle_records_models_rankings_experiments_and_views():
	service = RecsService()

	item_a = service.register_catalog_item(
		item_id="course-ai",
		tenant_id="tenant-recs",
		name="AI Course",
		item_type="course",
		category="learning",
		features={"ai": 0.9, "finance": 0.1},
		tags=["ai", "upskill"],
	)
	item_b = service.register_catalog_item(
		item_id="course-finance",
		tenant_id="tenant-recs",
		name="Finance Course",
		item_type="course",
		category="learning",
		features={"ai": 0.2, "finance": 0.9},
		tags=["finance"],
	)
	service.register_catalog_item(
		item_id="sensitive-loan",
		tenant_id="tenant-recs",
		name="Loan Offer",
		item_type="offer",
		category="finance",
		features={"ai": 0.1, "finance": 0.95},
		tags=["finance"],
		sensitive_attributes=["credit_score"],
	)
	profile = service.record_profile(
		profile_id="profile-001",
		tenant_id="tenant-recs",
		features={"ai": 0.95, "finance": 0.2},
		segments=["ai", "upskill"],
		consent_recorded=True,
	)
	policy = service.attach_ranking_policy(
		policy_id="policy-safe",
		tenant_id="tenant-recs",
		name="Safe Ranking",
		objective="maximize_relevance_with_diversity",
		minimum_confidence=0.25,
		max_per_category=2,
	)
	model = service.train_model(
		model_id="model-hybrid",
		tenant_id="tenant-recs",
		name="Hybrid Model",
		algorithm="hybrid",
		owner="personalization-team",
		training_event_count=2500,
		feature_names=["ai", "finance"],
		metric_value=0.81,
	)
	recommendations = service.generate_recommendations(
		recommendation_id="recset-001",
		tenant_id="tenant-recs",
		model_id=model["id"],
		profile_id=profile["id"],
		policy_id=policy["id"],
		candidate_item_ids=[item_a["id"], item_b["id"], "sensitive-loan"],
		limit=3,
		impact_level="high",
		explanation_attached=True,
	)
	experiment = service.create_experiment(
		experiment_id="exp-safe",
		tenant_id="tenant-recs",
		name="Recommendation Lift",
		model_id=model["id"],
		policy_id=policy["id"],
		experiment_percent=10,
		holdout_percent=10,
		business_metric="conversion_rate",
		approved=True,
	)
	drifted_model = service.record_drift(model["id"], "tenant-recs", baseline_metric=0.82, current_metric=0.70)

	assert recommendations["recommendations"][0]["item_id"] == "course-ai"
	assert {item["item_id"] for item in recommendations["recommendations"]} == {"course-ai", "course-finance"}
	assert recommendations["impact_level"] == "high"
	assert experiment["business_metric"] == "conversion_rate"
	assert drifted_model["drift_status"] == "watch"

	summary = service.dashboard_summary("tenant-recs")
	assert summary["catalog_item_count"] == 3
	assert summary["model_count"] == 1
	assert summary["recommendation_set_count"] == 1
	assert views.dashboard_model(service, "tenant-recs")["summary"]["recommendation_set_count"] == 1
	assert views.recommendation_console_model(service, "tenant-recs")["recommendation_sets"][0]["id"] == "recset-001"
	assert views.model_registry_model(service, "tenant-recs")["training_runs"]
	assert views.catalog_manager_model(service, "tenant-recs")["catalog_items"][0]["id"] == "course-ai"
	assert views.profile_features_model(service, "tenant-recs")["profiles"][0]["id"] == "profile-001"
	assert views.experiment_studio_model(service, "tenant-recs")["experiments"][0]["id"] == "exp-safe"
	assert views.ranking_policy_model(service, "tenant-recs")["policies"][0]["id"] == "policy-safe"
	assert views.governance_model(service, "tenant-recs")["audit_events"]


def test_recommendation_guardrails_block_unsafe_operations():
	service = RecsService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_catalog_item("item", "", "Item", "course", "learning")

	with pytest.raises(PermissionError, match="insufficient_training_events"):
		service.train_model("model-low", "tenant-recs", "Low", "hybrid", "owner", 25)

	with pytest.raises(PermissionError, match="model_owner_required"):
		service.train_model("model-owner", "tenant-recs", "Owner", "hybrid", "", 1200)

	with pytest.raises(PermissionError, match="drift_monitoring_required"):
		service.train_model("model-drift", "tenant-recs", "Drift", "hybrid", "owner", 1200, drift_monitoring_enabled=False)

	item = service.register_catalog_item("item", "tenant-recs", "Item", "course", "learning", {"topic": 0.9})
	profile = service.record_profile("profile", "tenant-recs", {"topic": 0.9}, ["topic"], consent_recorded=False)
	model = service.train_model("model", "tenant-recs", "Model", "hybrid", "owner", 1200)

	with pytest.raises(PermissionError, match="profile_consent_required"):
		service.generate_recommendations("rec-no-consent", "tenant-recs", model["id"], profile["id"], "", [item["id"]])

	profile = service.record_profile("profile", "tenant-recs", {"topic": 0.9}, ["topic"], consent_recorded=True)

	with pytest.raises(PermissionError, match="ranking_policy_required"):
		service.generate_recommendations("rec-no-policy", "tenant-recs", model["id"], profile["id"], "", [item["id"]])

	policy = service.attach_ranking_policy("policy", "tenant-recs", "Policy", "relevance", minimum_confidence=0.1)

	with pytest.raises(PermissionError, match="explainability_required"):
		service.generate_recommendations("rec-high", "tenant-recs", model["id"], profile["id"], policy["id"], [item["id"]], impact_level="high", explanation_attached=False)

	with pytest.raises(PermissionError, match="experiment_approval_required"):
		service.create_experiment("exp-approval", "tenant-recs", "Experiment", model["id"], policy["id"], 10, 10, "conversion", approved=False)

	with pytest.raises(PermissionError, match="holdout_required"):
		service.create_experiment("exp-holdout", "tenant-recs", "Experiment", model["id"], policy["id"], 10, 0, "conversion", approved=True)

	with pytest.raises(PermissionError, match="business_metric_required"):
		service.create_experiment("exp-metric", "tenant-recs", "Experiment", model["id"], policy["id"], 10, 10, "", approved=True)

	with pytest.raises(PermissionError, match="experiment_review_required"):
		service.create_experiment("exp-large", "tenant-recs", "Experiment", model["id"], policy["id"], 50, 10, "conversion", approved=True, review_recorded=False)


def test_compatibility_record_api_uses_recommendation_model_runtime():
	service = RecsService()

	record = service.create_record(
		record_id="compat-model",
		tenant_id="tenant-recs",
		metadata={"algorithm": "content_based", "owner": "recs", "training_event_count": 1500, "feature_names": ["topic"]},
		status="active",
	)

	assert record["id"] == "compat-model"
	assert record["algorithm"] == "content_based"
	assert record["training_event_count"] == 1500
	assert service.list_records("tenant-recs")[0]["id"] == "compat-model"
