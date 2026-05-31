"""API helpers for APG Anomaly Detection."""

from __future__ import annotations

from typing import Any

from .service import AnomService


SERVICE = AnomService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**SERVICE.signal_summary(tenant_id),
	}


def register_source(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_source(
		source_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or ""),
		kind=str(payload.get("kind") or ""),
		owner=str(payload.get("owner") or ""),
		labels=dict(payload.get("labels") or {}),
	)


def create_baseline(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_baseline(
		baseline_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		source_id=str(payload["source_id"]),
		metric=str(payload.get("metric") or ""),
		values=[float(item) for item in payload.get("values", [])],
		sensitivity=str(payload.get("sensitivity") or ""),
	)


def reset_baseline(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.reset_baseline(
		baseline_id=str(payload["id"]),
		values=[float(item) for item in payload.get("values", [])],
		approval_recorded=bool(payload.get("approval_recorded", False)),
		tenant_id=payload.get("tenant_id"),
	)


def detect(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.detect(
		detection_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		source_id=str(payload["source_id"]),
		baseline_id=str(payload["baseline_id"]),
		metric=str(payload.get("metric") or "value"),
		value=float(payload["value"]),
		timestamp=payload.get("timestamp"),
		context=dict(payload.get("context") or {}),
		owner=payload.get("owner"),
		triage_recorded=bool(payload.get("triage_recorded", False)),
	)


def open_investigation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.open_investigation(
		investigation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		signal_id=str(payload["signal_id"]),
		owner=str(payload["owner"]),
	)


def close_investigation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.close_investigation(
		investigation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		resolution=str(payload["resolution"]),
		closed_by=str(payload["closed_by"]),
		resolution_evidence=[str(item) for item in payload.get("resolution_evidence", [])],
	)


def record_feedback(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_feedback(
		feedback_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		signal_id=str(payload["signal_id"]),
		label=str(payload["label"]),
		reviewer=str(payload.get("reviewer") or ""),
		notes=str(payload.get("notes") or ""),
		tuning_review_recorded=bool(payload.get("tuning_review_recorded", False)),
	)


def register_anomaly_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_anomaly_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or ""),
		runtime=str(payload.get("runtime") or ""),
		role=str(payload.get("role") or ""),
		scope=str(payload.get("scope") or ""),
		owner=str(payload.get("owner") or ""),
		purpose=str(payload.get("purpose") or ""),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		human_approval_required=bool(payload.get("human_approval_required", False)),
	)


def validate_anom_lifecycle_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_anom_lifecycle_batch(
		tenant_id=str(payload.get("tenant_id") or "default"),
		event_stream=str(payload.get("event_stream") or ""),
		mutation_count=int(payload.get("mutation_count") or 0),
		operation=str(payload.get("operation") or "anomaly_agent_batch"),
		batch_id=payload.get("id"),
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


def list_sources(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_sources(tenant_id)


def list_baselines(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_baselines(tenant_id)


def list_signals(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_signals(tenant_id)


def list_investigations(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_investigations(tenant_id)


def list_anomaly_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_anomaly_agents(tenant_id)


def list_lifecycle_batches(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_lifecycle_batches(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)
