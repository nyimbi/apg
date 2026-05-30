"""Dependency-light AUDL API helpers for generated APG applications."""

from __future__ import annotations

from typing import Any

from .audit_runtime import AudlService


SERVICE = AudlService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**SERVICE.audit_summary(tenant_id),
	}


def append_event(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.append_event(
		event_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		actor=str(payload["actor"]),
		action=str(payload["action"]),
		resource_type=str(payload["resource_type"]),
		resource_id=str(payload["resource_id"]),
		severity=str(payload.get("severity") or "info"),
		contains_pii=_payload_bool(payload, "contains_pii", False),
		immutable=_payload_bool(payload, "immutable", True),
		checksum=payload.get("checksum"),
		details=dict(payload.get("details") or {}),
		escalation_configured=_payload_bool(payload, "escalation_configured", True),
	)


def validate_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_batch(
		tenant_id=str(payload.get("tenant_id") or "default"),
		record_count=int(payload["record_count"]),
		event_stream=str(payload.get("event_stream") or "bytewax"),
		stream_processing_enabled=_payload_bool(payload, "stream_processing_enabled", True),
	)


def register_audit_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_audit_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		purpose=str(payload["purpose"]),
		owner=str(payload["owner"]),
		human_approval_required=_payload_bool(payload, "human_approval_required", True),
		configuration=dict(payload.get("configuration") or {}),
	)


def apply_legal_hold(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.apply_legal_hold(
		hold_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		scope=dict(payload["scope"]),
		reason=str(payload["reason"]),
		approver=str(payload["approver"]),
	)


def release_legal_hold(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.release_legal_hold(
		hold_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		released_by=str(payload["released_by"]),
		release_evidence=str(payload["release_evidence"]),
	)


def request_export(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_export(
		export_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		requested_by=str(payload["requested_by"]),
		query=dict(payload.get("query") or {}),
		contains_pii=_payload_bool(payload, "contains_pii", False),
		masking_enabled=_payload_bool(payload, "masking_enabled", True),
		reason=str(payload["reason"]),
	)


def decide_export(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_export(
		export_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload["notes"]),
	)


def request_purge(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_purge(
		purge_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		requested_by=str(payload["requested_by"]),
		scope=dict(payload["scope"]),
		reason=str(payload["reason"]),
	)


def decide_purge(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_purge(
		purge_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload["notes"]),
	)


def open_investigation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.open_investigation(
		investigation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		event_ids=[str(event_id) for event_id in payload["event_ids"]],
		owner=str(payload["owner"]),
		priority=str(payload.get("priority") or "high"),
	)


def close_investigation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.close_investigation(
		investigation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		closed_by=str(payload["closed_by"]),
		resolution=str(payload["resolution"]),
		evidence=dict(payload["evidence"]),
	)


def list_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_events(tenant_id)


def list_legal_holds(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_legal_holds(tenant_id)


def list_exports(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_exports(tenant_id)


def list_purges(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_purges(tenant_id)


def list_investigations(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_investigations(tenant_id)


def list_audit_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_agents(tenant_id)


def list_governance_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_governance_events(tenant_id)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def _payload_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
	value = payload.get(key, default)
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return bool(value)
