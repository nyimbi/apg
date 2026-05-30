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
	assert contract["configuration_schema"]["required"] == ["tenant_id", "datasets", "models", "ranking", "experiments", "feedback", "recommender_agents", "governance", "observability", "adapters", "deployments", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "recommendations", "datasets", "models", "deployments", "catalogs", "profiles", "feedback", "experiments", "policies", "agents", "audit", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/recs/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "experiment_board" in contract["theme"]["components"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert "codex" in contract["configuration"]["recommender_agents"]["supported_runtimes"]


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
	agent_result = evaluate_capability_rules({
		"recommender_agent_present": True,
		"agent_registered": False,
		"agent_runtime_supported": False,
		"agent_scope_present": False,
		"agent_contribution_disclosed": False,
	})
	stream_result = evaluate_capability_rules({"operation": "batch_recommendation_mutation", "event_stream": "memory"})
	assert set(agent_result["matched_rules"]) == {"recommender_agent_requires_registration", "recommender_agent_runtime_supported", "recommender_agent_requires_scope", "recommender_agent_requires_disclosure"}
	assert stream_result["matched_rules"] == ["batch_recommendation_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "recs"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "recs_recommendation_console"
	assert registration["streaming"]["processor"] == "bytewax"
	assert registration["ui_components"]["recommendations"] == "/recs/recommendations"
	assert registration["ui_components"]["agents"] == "/recs/agents"
	assert "pred" in registration["dependencies"]
	assert "model_deployments" in registration["capabilities"]
	assert "recs:recommend" in registration["permissions"]
	assert "recs:deploy" in registration["permissions"]


def test_recommendation_lifecycle_records_models_rankings_experiments_and_views():
	service = RecsService()

	dataset = service.register_dataset(
		dataset_id="events-learning",
		tenant_id="tenant-recs",
		name="Learning Events",
		owner="personalization-team",
		source_ref="etlp:learning-events",
		schema_fields=["profile_id", "item_id", "event_type", "occurred_at"],
		policy_ref="dataset-policy:learning",
		event_count=2500,
	)
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
	interaction = service.record_interaction(
		event_id="interaction-001",
		tenant_id="tenant-recs",
		dataset_id=dataset["id"],
		profile_id=profile["id"],
		item_id=item_a["id"],
		event_type="click",
		occurred_at="2026-05-30T10:00:00Z",
	)
	policy = service.attach_ranking_policy(
		policy_id="policy-safe",
		tenant_id="tenant-recs",
		name="Safe Ranking",
		objective="maximize_relevance_with_diversity",
		owner="risk-team",
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
	approved_model = service.approve_model(model["id"], "tenant-recs", approval_ref="approval:model-hybrid")
	deployment = service.deploy_model(
		deployment_id="deploy-hybrid",
		tenant_id="tenant-recs",
		model_id=model["id"],
		target_runtime="python",
		target_ref="apg://models/recs/model-hybrid",
		approval_recorded=True,
		approval_ref="approval:deploy-hybrid",
		rollback_plan_ref="rollback:model-hybrid",
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
	feedback = service.record_feedback(
		feedback_id="feedback-001",
		tenant_id="tenant-recs",
		recommendation_set_id=recommendations["id"],
		profile_id=profile["id"],
		item_id=item_a["id"],
		event_type="conversion",
		value=1.0,
	)
	agent = service.register_recommender_agent(
		agent_id="codex-ranking-agent",
		tenant_id="tenant-recs",
		name="Codex Ranking Agent",
		runtime="codex",
		role="ranking_designer",
		scope="ranking,experiments,guardrails",
		contribution_disclosed=True,
		policy_ref="agent-policy:recs",
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
	assert interaction["event_type"] == "click"
	assert approved_model["approved"] is True
	assert deployment["target_runtime"] == "python"
	assert feedback["event_type"] == "conversion"
	assert agent["runtime"] == "codex"
	assert recommendations["impact_level"] == "high"
	assert experiment["business_metric"] == "conversion_rate"
	assert drifted_model["drift_status"] == "watch"

	summary = service.dashboard_summary("tenant-recs")
	assert summary["dataset_count"] == 1
	assert summary["interaction_event_count"] == 1
	assert summary["catalog_item_count"] == 3
	assert summary["model_count"] == 1
	assert summary["deployment_count"] == 1
	assert summary["recommendation_set_count"] == 1
	assert summary["feedback_count"] == 1
	assert summary["agent_count"] == 1
	assert views.dashboard_model(service, "tenant-recs")["summary"]["recommendation_set_count"] == 1
	assert views.recommendation_console_model(service, "tenant-recs")["recommendation_sets"][0]["id"] == "recset-001"
	assert views.dataset_manager_model(service, "tenant-recs")["datasets"][0]["id"] == "events-learning"
	assert views.model_registry_model(service, "tenant-recs")["training_runs"]
	assert views.deployment_center_model(service, "tenant-recs")["deployments"][0]["id"] == "deploy-hybrid"
	assert views.catalog_manager_model(service, "tenant-recs")["catalog_items"][0]["id"] == "course-ai"
	assert views.profile_features_model(service, "tenant-recs")["profiles"][0]["id"] == "profile-001"
	assert views.feedback_console_model(service, "tenant-recs")["feedback"][0]["id"] == "feedback-001"
	assert views.experiment_studio_model(service, "tenant-recs")["experiments"][0]["id"] == "exp-safe"
	assert views.ranking_policy_model(service, "tenant-recs")["policies"][0]["id"] == "policy-safe"
	assert views.recommender_agents_model(service, "tenant-recs")["agents"][0]["id"] == "codex-ranking-agent"
	assert views.audit_trail_model(service, "tenant-recs")["audit_events"]
	assert views.analytics_model(service, "tenant-recs")["summary"]["agent_count"] == 1
	assert views.governance_model(service, "tenant-recs")["audit_events"]


def test_recommendation_guardrails_block_unsafe_operations():
	service = RecsService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_catalog_item("item", "", "Item", "course", "learning")

	with pytest.raises(PermissionError, match="dataset_owner_required"):
		service.register_dataset("dataset", "tenant-recs", "Dataset", "", "etlp:events", ["profile_id"], "policy")

	dataset = service.register_dataset("dataset", "tenant-recs", "Dataset", "owner", "etlp:events", ["profile_id", "item_id"], "policy")
	with pytest.raises(PermissionError, match="interaction_timestamp_required"):
		service.record_interaction("event", "tenant-recs", dataset["id"], "profile", "item", "click", "")

	with pytest.raises(PermissionError, match="insufficient_training_events"):
		service.train_model("model-low", "tenant-recs", "Low", "hybrid", "owner", 25)

	with pytest.raises(PermissionError, match="model_owner_required"):
		service.train_model("model-owner", "tenant-recs", "Owner", "hybrid", "", 1200)

	with pytest.raises(PermissionError, match="drift_monitoring_required"):
		service.train_model("model-drift", "tenant-recs", "Drift", "hybrid", "owner", 1200, drift_monitoring_enabled=False)

	item = service.register_catalog_item("item", "tenant-recs", "Item", "course", "learning", {"topic": 0.9})
	profile = service.record_profile("profile", "tenant-recs", {"topic": 0.9}, ["topic"], consent_recorded=False)
	model = service.train_model("model", "tenant-recs", "Model", "hybrid", "owner", 1200)

	with pytest.raises(PermissionError, match="model_approval_required"):
		service.deploy_model("deploy-unapproved", "tenant-recs", model["id"], "python", "apg://models/model", True, "rollback:model")

	with pytest.raises(PermissionError, match="profile_consent_required"):
		service.generate_recommendations("rec-no-consent", "tenant-recs", model["id"], profile["id"], "", [item["id"]])

	profile = service.record_profile("profile", "tenant-recs", {"topic": 0.9}, ["topic"], consent_recorded=True)

	with pytest.raises(PermissionError, match="ranking_policy_required"):
		service.generate_recommendations("rec-no-policy", "tenant-recs", model["id"], profile["id"], "", [item["id"]])

	policy = service.attach_ranking_policy("policy", "tenant-recs", "Policy", "relevance", minimum_confidence=0.1)

	with pytest.raises(PermissionError, match="recommendation_candidates_required"):
		service.generate_recommendations("rec-no-candidates", "tenant-recs", model["id"], profile["id"], policy["id"], [])

	with pytest.raises(PermissionError, match="explainability_required"):
		service.generate_recommendations("rec-high", "tenant-recs", model["id"], profile["id"], policy["id"], [item["id"]], impact_level="high", explanation_attached=False)

	strict_policy = service.attach_ranking_policy("policy-strict", "tenant-recs", "Strict Policy", "relevance", minimum_confidence=1.0)
	with pytest.raises(PermissionError, match="empty_recommendation_review_required"):
		service.generate_recommendations("rec-empty", "tenant-recs", model["id"], profile["id"], strict_policy["id"], [item["id"]])

	recommendations = service.generate_recommendations("rec-ok", "tenant-recs", model["id"], profile["id"], policy["id"], [item["id"]])
	with pytest.raises(PermissionError, match="feedback_actor_required"):
		service.record_feedback("feedback-bad", "tenant-recs", recommendations["id"], "", item["id"], "click")

	with pytest.raises(PermissionError, match="recommender_agent_disclosure_required"):
		service.register_recommender_agent("agent-bad", "tenant-recs", "Agent", "codex", "ranking_designer", "ranking", False)

	with pytest.raises(PermissionError, match="recommender_agent_runtime_not_supported"):
		service.register_recommender_agent("agent-runtime", "tenant-recs", "Agent", "unknown", "ranking_designer", "ranking", True)

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
