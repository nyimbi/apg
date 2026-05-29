"""API helpers for the Predictive Analytics capability."""

from __future__ import annotations

from typing import Any

from .service import PredService


SERVICE = PredService()


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
		"forecast_count": summary["forecast_count"],
		"score_count": summary["score_count"],
	}


def register_model(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_model(
		model_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload["owner"]),
		algorithm=str(payload["algorithm"]),
		target=str(payload["target"]),
		environment=str(payload.get("environment") or "development"),
		approved=bool(payload.get("approved", False)),
		explainability_attached=bool(payload.get("explainability_attached", False)),
		training_history_points=int(payload.get("training_history_points") or 0),
		feature_names=list(payload.get("feature_names") or ()),
		metadata=dict(payload.get("metadata") or {}),
	)


def approve_model(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_model(
		model_id=str(payload["model_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		approver=str(payload["approver"]),
		explainability_ref=payload.get("explainability_ref"),
	)


def register_feature_set(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_feature_set(
		feature_set_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload["owner"]),
		feature_names=list(payload.get("feature_names") or ()),
		lineage_refs=list(payload.get("lineage_refs") or ()),
		source_system=str(payload.get("source_system") or "local"),
	)


def create_forecast(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_forecast(
		forecast_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		model_id=str(payload["model_id"]),
		series_name=str(payload["series_name"]),
		history_values=list(payload.get("history_values") or ()),
		horizon_days=int(payload["horizon_days"]),
		review_recorded=bool(payload.get("review_recorded", False)),
		confidence_interval=bool(payload.get("confidence_interval", True)),
		actor=str(payload.get("actor") or "pred"),
	)


def score_entity(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.score_entity(
		score_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		model_id=str(payload["model_id"]),
		feature_set_id=str(payload["feature_set_id"]),
		entity_id=str(payload["entity_id"]),
		feature_values=dict(payload.get("feature_values") or {}),
		environment=str(payload.get("environment") or "production"),
		impact=str(payload.get("impact") or "low"),
		explanation_ref=str(payload.get("explanation_ref") or ""),
		actor=str(payload.get("actor") or "pred"),
	)


def simulate_scenario(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.simulate_scenario(
		scenario_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		model_id=str(payload["model_id"]),
		name=str(payload.get("name") or payload["id"]),
		baseline_score=float(payload["baseline_score"]),
		adjustments=dict(payload.get("adjustments") or {}),
		assumptions=list(payload.get("assumptions") or ()),
		actor=str(payload.get("actor") or "pred"),
	)


def record_drift(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_drift(
		report_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		model_id=str(payload["model_id"]),
		metric_name=str(payload["metric_name"]),
		drift_score=float(payload["drift_score"]),
		threshold=float(payload["threshold"]),
		actor=str(payload.get("actor") or "pred"),
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
