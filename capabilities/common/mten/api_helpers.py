"""Dependency-light API helpers for APG Multi-Tenant Management."""

from __future__ import annotations

from typing import Any

from .mten_runtime import MtenService


SERVICE = MtenService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**SERVICE.portfolio_summary(tenant_id),
	}


def register_tenant(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_tenant(
		target_tenant_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload.get("owner") or ""),
		tier=str(payload.get("tier") or "free"),
		primary_domain=str(payload.get("primary_domain") or ""),
		custom_domain=str(payload.get("custom_domain") or ""),
		dns_validated=_payload_bool(payload, "dns_validated", False),
		projected_compute_units=int(payload.get("projected_compute_units", 0)),
		isolation_boundary_encrypted=_payload_bool(payload, "isolation_boundary_encrypted", True),
		capacity_approval_id=str(payload["capacity_approval_id"]) if payload.get("capacity_approval_id") else None,
		metadata=dict(payload.get("metadata") or {}),
	)


def validate_lifecycle_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_lifecycle_batch(
		tenant_id=str(payload.get("tenant_id") or "default"),
		record_count=int(payload["record_count"]),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def register_tenant_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_tenant_agent(
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


def activate_tenant(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.activate_tenant(
		target_tenant_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		actor=str(payload.get("actor") or "operator"),
		dns_validated=_payload_bool(payload, "dns_validated", False) if "dns_validated" in payload else None,
	)


def request_capacity_approval(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_capacity_approval(
		approval_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		target_tenant_id=str(payload["target_tenant_id"]),
		requested_by=str(payload.get("requested_by") or ""),
		projected_compute_units=int(payload.get("projected_compute_units", 0)),
		justification=str(payload.get("justification") or ""),
	)


def decide_capacity_approval(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_capacity_approval(
		approval_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload.get("reviewer") or ""),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload.get("notes") or ""),
	)


def record_isolation_incident(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_isolation_incident(
		incident_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		target_tenant_id=str(payload["target_tenant_id"]),
		detected_by=str(payload.get("detected_by") or ""),
		breach_summary=str(payload.get("breach_summary") or ""),
		severity=str(payload.get("severity") or "high"),
	)


def reactivate_tenant(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.reactivate_tenant(
		target_tenant_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		actor=str(payload.get("actor") or ""),
		evidence=str(payload.get("evidence") or ""),
	)


def request_live_migration(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_live_migration(
		migration_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		target_tenant_id=str(payload["target_tenant_id"]),
		requested_by=str(payload.get("requested_by") or ""),
		source_provider=str(payload.get("source_provider") or ""),
		target_provider=str(payload.get("target_provider") or ""),
		runbook=str(payload.get("runbook") or ""),
	)


def decide_live_migration(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_live_migration(
		migration_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload.get("reviewer") or ""),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload.get("notes") or ""),
	)


def execute_live_migration(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.execute_live_migration(
		migration_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		actor=str(payload.get("actor") or "operator"),
	)


def list_tenants(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_tenants(tenant_id)


def list_capacity_approvals(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_capacity_approvals(tenant_id)


def list_isolation_incidents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_isolation_incidents(tenant_id)


def list_live_migrations(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_live_migrations(tenant_id)


def list_tenant_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_tenant_agents(tenant_id)


def list_governance_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_governance_events(tenant_id)


def _payload_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
	value = payload.get(key, default)
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return bool(value)
