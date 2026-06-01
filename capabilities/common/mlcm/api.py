"""API helpers for the AI Model Lifecycle Management capability."""

from __future__ import annotations

from typing import Any

from .service import MlcmService


SERVICE = MlcmService()


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
		"version_count": summary["version_count"],
		"deployment_count": summary["deployment_count"],
		"unresolved_drift_count": summary["unresolved_drift_count"],
	}


def register_model(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_model(
		model_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner=str(payload.get("owner") or ""),
		problem_type=str(payload.get("problem_type") or "general"),
		risk_level=str(payload.get("risk_level") or "medium"),
		description=str(payload.get("description") or ""),
		tags=list(payload.get("tags") or []),
		metadata=dict(payload.get("metadata") or {}),
	)


def create_version(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_version(
		version_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		model_id=str(payload["model_id"]),
		version=str(payload["version"]),
		artifact_uri=str(payload["artifact_uri"]),
		model_card=dict(payload.get("model_card") or {}),
		training_data_ref=str(payload.get("training_data_ref") or ""),
		baseline_ref=str(payload.get("baseline_ref") or ""),
		stage=str(payload.get("stage") or "dev"),
		metadata=dict(payload.get("metadata") or {}),
	)


def record_evaluation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_evaluation(
		evaluation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		version_id=str(payload["version_id"]),
		score=float(payload["score"]),
		baseline_ref=str(payload.get("baseline_ref") or ""),
		metrics=dict(payload.get("metrics") or {}),
		evidence_refs=list(payload.get("evidence_refs") or []),
		evaluator=str(payload.get("evaluator") or ""),
		fairness_review_recorded=bool(payload.get("fairness_review_recorded", False)),
		explainability_recorded=bool(payload.get("explainability_recorded", False)),
	)


def request_promotion(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_promotion(
		request_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		version_id=str(payload["version_id"]),
		target_stage=str(payload["target_stage"]),
		requested_by=str(payload.get("requested_by") or ""),
		approval_recorded=bool(payload.get("approval_recorded")),
		approval_ref=str(payload.get("approval_ref") or ""),
	)


def create_target(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_target(
		target_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		environment=str(payload["environment"]),
		serving_runtime=str(payload.get("serving_runtime") or "python"),
		owner=str(payload.get("owner") or ""),
		status=str(payload.get("status") or "active"),
		metadata=dict(payload.get("metadata") or {}),
	)


def deploy_model(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.deploy_model(
		deployment_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		version_id=str(payload["version_id"]),
		target_id=str(payload["target_id"]),
		replicas=int(payload.get("replicas") or 1),
		canary_percent=int(payload.get("canary_percent") or 0),
		approved_by=str(payload.get("approved_by") or ""),
		metadata=dict(payload.get("metadata") or {}),
	)


def record_drift(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_drift(
		signal_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		version_id=str(payload["version_id"]),
		metric=str(payload["metric"]),
		score=float(payload["score"]),
		threshold=float(payload["threshold"]),
		metadata=dict(payload.get("metadata") or {}),
	)


def record_drift_review(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_drift_review(
		signal_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		review_ref=str(payload["review_ref"]),
	)


def rollback_deployment(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.rollback_deployment(
		rollback_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		deployment_id=str(payload["deployment_id"]),
		to_version_id=str(payload["to_version_id"]),
		reason=str(payload.get("reason") or "operator_requested"),
		requested_by=str(payload.get("requested_by") or ""),
	)


def retire_model(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.retire_model(
		retirement_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		model_id=str(payload["model_id"]),
		impact_review_ref=str(payload.get("impact_review_ref") or ""),
		retired_by=str(payload.get("retired_by") or ""),
		metadata=dict(payload.get("metadata") or {}),
	)


def register_model_lifecycle_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_model_lifecycle_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		runtime=str(payload.get("runtime") or "codex"),
		role=str(payload.get("role") or "model_steward"),
		scope=str(payload.get("scope") or ""),
		owner=str(payload.get("owner") or ""),
		purpose=str(payload.get("purpose") or ""),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		human_approval_required=bool(payload.get("human_approval_required", False)),
	)


def validate_mlcm_lifecycle_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_mlcm_lifecycle_batch(
		tenant_id=str(payload.get("tenant_id") or "default"),
		event_stream=str(payload.get("event_stream") or "bytewax"),
		mutation_count=int(payload.get("mutation_count") or 0),
		operation=str(payload.get("operation") or "model_lifecycle_agent_batch"),
		batch_id=str(payload["id"]) if payload.get("id") else None,
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


def list_pending_reviews(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_pending_reviews(tenant_id)


def list_model_lifecycle_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_model_lifecycle_agents(tenant_id)


def list_lifecycle_batches(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_lifecycle_batches(tenant_id)
