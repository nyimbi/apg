"""API helpers for the Recommender Systems capability."""

from __future__ import annotations

from typing import Any

from .service import RecsService


SERVICE = RecsService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"model_count": summary["model_count"],
		"catalog_item_count": summary["catalog_item_count"],
		"recommendation_set_count": summary["recommendation_set_count"],
		"dataset_count": summary["dataset_count"],
		"deployment_count": summary["deployment_count"],
		"feedback_count": summary["feedback_count"],
		"agent_count": summary["agent_count"],
	}


def register_dataset(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_dataset(
		dataset_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload.get("owner") or ""),
		source_ref=str(payload.get("source_ref") or ""),
		schema_fields=list(payload.get("schema_fields") or ()),
		policy_ref=str(payload.get("policy_ref") or ""),
		event_count=int(payload.get("event_count") or 0),
		actor=str(payload.get("actor") or "recs"),
	)


def record_interaction(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_interaction(
		event_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		dataset_id=str(payload["dataset_id"]),
		profile_id=str(payload.get("profile_id") or ""),
		item_id=str(payload.get("item_id") or ""),
		event_type=str(payload.get("event_type") or "impression"),
		occurred_at=str(payload.get("occurred_at") or ""),
		weight=float(payload.get("weight", 1.0)),
		metadata=dict(payload.get("metadata") or {}),
		actor=str(payload.get("actor") or "recs"),
	)


def register_catalog_item(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_catalog_item(
		item_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		item_type=str(payload.get("item_type") or "item"),
		category=str(payload.get("category") or "default"),
		features=dict(payload.get("features") or {}),
		tags=list(payload.get("tags") or ()),
		sensitive_attributes=list(payload.get("sensitive_attributes") or ()),
		actor=str(payload.get("actor") or "recs"),
	)


def record_profile(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_profile(
		profile_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		features=dict(payload.get("features") or {}),
		segments=list(payload.get("segments") or ()),
		consent_recorded=bool(payload.get("consent_recorded", False)),
		actor=str(payload.get("actor") or "recs"),
	)


def attach_ranking_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.attach_ranking_policy(
		policy_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		objective=str(payload["objective"]),
		owner=str(payload.get("owner") or "recs"),
		minimum_confidence=float(payload.get("minimum_confidence", 0.65)),
		diversity_constraints_enabled=bool(payload.get("diversity_constraints_enabled", True)),
		sensitive_attribute_filtering=bool(payload.get("sensitive_attribute_filtering", True)),
		max_per_category=int(payload.get("max_per_category", 2)),
		actor=str(payload.get("actor") or "recs"),
	)


def train_model(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.train_model(
		model_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		algorithm=str(payload.get("algorithm") or "hybrid"),
		owner=str(payload["owner"]),
		training_event_count=int(payload["training_event_count"]),
		feature_names=list(payload.get("feature_names") or ()),
		drift_monitoring_enabled=bool(payload.get("drift_monitoring_enabled", True)),
		metric_name=str(payload.get("metric_name") or "precision_at_k"),
		metric_value=float(payload.get("metric_value", 0.72)),
		actor=str(payload.get("actor") or "recs"),
	)


def approve_model(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_model(
		model_id=str(payload["model_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		approval_ref=str(payload.get("approval_ref") or ""),
		actor=str(payload.get("actor") or "recs"),
	)


def deploy_model(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.deploy_model(
		deployment_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		model_id=str(payload["model_id"]),
		target_runtime=str(payload.get("target_runtime") or "python"),
		target_ref=str(payload.get("target_ref") or ""),
		approval_recorded=bool(payload.get("approval_recorded")),
		rollback_plan_ref=str(payload.get("rollback_plan_ref") or ""),
		approval_ref=str(payload.get("approval_ref") or ""),
		actor=str(payload.get("actor") or "recs"),
	)


def generate_recommendations(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.generate_recommendations(
		recommendation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		model_id=str(payload["model_id"]),
		profile_id=str(payload["profile_id"]),
		policy_id=str(payload.get("policy_id") or ""),
		candidate_item_ids=list(payload.get("candidate_item_ids") or ()),
		limit=int(payload.get("limit") or 5),
		impact_level=str(payload.get("impact_level") or "low"),
		explanation_attached=bool(payload.get("explanation_attached", False)),
		actor=str(payload.get("actor") or "recs"),
	)


def record_feedback(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_feedback(
		feedback_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		recommendation_set_id=str(payload["recommendation_set_id"]),
		profile_id=str(payload.get("profile_id") or ""),
		item_id=str(payload.get("item_id") or ""),
		event_type=str(payload.get("event_type") or "impression"),
		value=float(payload.get("value", 1.0)),
		metadata=dict(payload.get("metadata") or {}),
		actor=str(payload.get("actor") or "recs"),
	)


def create_experiment(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_experiment(
		experiment_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		model_id=str(payload["model_id"]),
		policy_id=str(payload["policy_id"]),
		experiment_percent=int(payload["experiment_percent"]),
		holdout_percent=int(payload["holdout_percent"]),
		business_metric=str(payload.get("business_metric") or ""),
		approved=bool(payload.get("approved", False)),
		review_recorded=bool(payload.get("review_recorded", False)),
		actor=str(payload.get("actor") or "recs"),
	)


def record_drift(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_drift(
		model_id=str(payload["model_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		baseline_metric=float(payload["baseline_metric"]),
		current_metric=float(payload["current_metric"]),
		actor=str(payload.get("actor") or "recs"),
	)


def register_recommender_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_recommender_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload.get("scope") or ""),
		contribution_disclosed=bool(payload.get("contribution_disclosed")),
		policy_ref=str(payload.get("policy_ref") or ""),
		registered=bool(payload.get("registered", True)),
		actor=str(payload.get("actor") or "recs"),
	)


def change_model_state(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.change_model_state(
		tenant_id=str(payload.get("tenant_id") or "default"),
		model_id=str(payload["model_id"]),
		status=str(payload["status"]),
		reason=str(payload.get("reason") or ""),
		audit_recorded=bool(payload.get("audit_recorded", True)),
		actor=str(payload.get("actor") or "recs"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
