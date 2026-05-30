"""API helpers for the Tenants Legacy capability."""

from __future__ import annotations

from typing import Any

from .service import TensService


SERVICE = TensService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"legacy_tenant_count": summary["legacy_tenant_count"],
		"mapped_tenant_count": summary["mapped_tenant_count"],
		"migration_count": summary["migration_count"],
		"deprecation_count": summary["deprecation_count"],
		"tens_agent_count": summary["tens_agent_count"],
	}


def register_legacy_tenant(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_legacy_tenant(
		tenant_id=str(payload.get("tenant_id") or "default"),
		legacy_tenant_id=str(payload["legacy_tenant_id"]),
		source_system=str(payload.get("source_system") or ""),
		owner=str(payload.get("owner") or ""),
		compatibility_scope=str(payload.get("compatibility_scope") or ""),
		days_since_activity=int(payload.get("days_since_activity", 0)),
		stale_review_recorded=bool(payload.get("stale_review_recorded", False)),
	)


def map_tenant(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.map_tenant(
		tenant_id=str(payload.get("tenant_id") or "default"),
		legacy_tenant_id=str(payload["legacy_tenant_id"]),
		apg_tenant_id=str(payload["apg_tenant_id"]),
		validated_by=str(payload.get("validated_by") or ""),
		validation_ref=str(payload.get("validation_ref") or ""),
		mapping_validated=bool(payload.get("mapping_validated", True)),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def validate_access_boundary(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_access_boundary(
		tenant_id=str(payload.get("tenant_id") or "default"),
		legacy_tenant_id=str(payload["legacy_tenant_id"]),
		auth_boundary_ref=str(payload.get("auth_boundary_ref") or ""),
		role_mapping_ref=str(payload.get("role_mapping_ref") or ""),
		isolation_validation_ref=str(payload.get("isolation_validation_ref") or ""),
		privileged_review_ref=str(payload.get("privileged_review_ref") or ""),
		actor=str(payload.get("actor") or ""),
	)


def create_migration_plan(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_migration_plan(
		tenant_id=str(payload.get("tenant_id") or "default"),
		legacy_tenant_id=str(payload["legacy_tenant_id"]),
		mapping_id=str(payload["mapping_id"]),
		owner=str(payload.get("owner") or ""),
		approval_ref=str(payload.get("approval_ref") or ""),
		rollback_plan_ref=str(payload.get("rollback_plan_ref") or ""),
		post_migration_validation_ref=str(payload.get("post_migration_validation_ref") or ""),
	)


def complete_migration(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.complete_migration(
		tenant_id=str(payload.get("tenant_id") or "default"),
		migration_id=str(payload["migration_id"]),
		actor=str(payload.get("actor") or ""),
		post_migration_validation_ref=str(payload.get("post_migration_validation_ref") or ""),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def record_deprecation_plan(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_deprecation_plan(
		tenant_id=str(payload.get("tenant_id") or "default"),
		legacy_tenant_id=str(payload["legacy_tenant_id"]),
		owner=str(payload.get("owner") or ""),
		deprecation_ref=str(payload.get("deprecation_ref") or ""),
		target_date=str(payload.get("target_date") or ""),
	)


def register_tens_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_tens_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload.get("runtime") or ""),
		role=str(payload.get("role") or ""),
		scope=str(payload.get("scope") or ""),
		owner=str(payload.get("owner") or "tenant-admin"),
		human_approval_required=bool(payload.get("human_approval_required", True)),
	)


def validate_agent_tenant_action(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_agent_tenant_action(
		tenant_id=str(payload.get("tenant_id") or "default"),
		agent_id=str(payload["agent_id"]),
		privileged_scope=bool(payload.get("privileged_scope", False)),
		human_approval_recorded=bool(payload.get("human_approval_recorded", False)),
	)


def validate_batch_tenant_mapping(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_batch_tenant_mapping(
		tenant_id=str(payload.get("tenant_id") or "default"),
		legacy_tenant_ids=list(payload.get("legacy_tenant_ids") or []),
		event_stream=str(payload.get("event_stream") or "bytewax"),
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


def list_tenant_legacy(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"legacy_tenants": SERVICE.list_legacy_tenants(tenant_id),
		"mappings": SERVICE.list_mappings(tenant_id),
		"boundaries": SERVICE.list_boundaries(tenant_id),
		"migrations": SERVICE.list_migrations(tenant_id),
		"deprecations": SERVICE.list_deprecations(tenant_id),
		"tens_agents": SERVICE.list_tens_agents(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}
