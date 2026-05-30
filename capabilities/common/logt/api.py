"""API helpers for APG Logging and Tracing."""

from __future__ import annotations

from typing import Any

from .service import LogtService


SERVICE = LogtService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**SERVICE.dashboard_summary(tenant_id),
	}


def create_retention_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_retention_policy(
		policy_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		log_retention_days=int(payload["log_retention_days"]),
		span_retention_days=int(payload["span_retention_days"]) if payload.get("span_retention_days") is not None else None,
		redaction_required=bool(payload.get("redaction_required", True)),
		export_approval_required=bool(payload.get("export_approval_required", True)),
	)


def create_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_pipeline(
		pipeline_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload["owner"]),
		schema_ref=str(payload["schema_ref"]),
		event_bus_ref=str(payload["event_bus_ref"]),
		sampling_policy=str(payload["sampling_policy"]),
		retention_policy_id=str(payload["retention_policy_id"]),
		status=str(payload.get("status") or "active"),
	)


def ingest_log(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.ingest_log(
		log_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		pipeline_id=str(payload["pipeline_id"]),
		service_name=str(payload["service_name"]),
		severity=str(payload.get("severity") or "info"),
		message=str(payload["message"]),
		attributes=dict(payload.get("attributes") or {}),
		trace_id=str(payload.get("trace_id") or ""),
		span_id=str(payload.get("span_id") or ""),
		sensitive_log_content=bool(payload.get("sensitive_log_content", False)),
		redaction_applied=bool(payload.get("redaction_applied", True)),
	)


def ingest_trace(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.ingest_trace(
		trace_record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		pipeline_id=str(payload["pipeline_id"]),
		trace_id=str(payload["trace_id"]),
		root_service=str(payload["root_service"]),
		operation=str(payload["operation"]),
		trace_context=dict(payload.get("trace_context") or {}),
		sampling_policy=str(payload.get("sampling_policy") or ""),
		status=str(payload.get("status") or "active"),
	)


def record_span(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_span(
		span_record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		trace_id=str(payload["trace_id"]),
		span_id=str(payload["span_id"]),
		service_name=str(payload["service_name"]),
		operation=str(payload["operation"]),
		duration_ms=float(payload["duration_ms"]),
		parent_span_id=str(payload.get("parent_span_id") or ""),
		attributes=dict(payload.get("attributes") or {}),
		error=bool(payload.get("error", False)),
	)


def search_logs(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.search_logs(
		query_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		query_text=str(payload.get("query_text") or ""),
		requested_by=str(payload["requested_by"]),
		query_window_hours=int(payload.get("query_window_hours", 24)),
		query_review_recorded=bool(payload.get("query_review_recorded", False)),
	)


def export_logs(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.export_logs(
		export_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		export_type=str(payload.get("export_type") or "logs"),
		requested_by=str(payload["requested_by"]),
		item_ids=tuple(payload.get("item_ids") or ()),
		approval_recorded=bool(payload.get("approval_recorded", False)),
		approval_ref=str(payload.get("approval_ref") or ""),
	)


def register_logt_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_logt_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload["scope"]),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		agent_id=str(payload["id"]) if payload.get("id") else None,
	)


def validate_batch_diagnostic_mutation(event_stream: str) -> dict[str, Any]:
	return SERVICE.validate_batch_diagnostic_mutation(event_stream)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def list_logt_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_logt_agents(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
