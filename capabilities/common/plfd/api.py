"""Dependency-light API helpers for Platform Foundation."""

from __future__ import annotations

from typing import Any

from .service import PlfdService


SERVICE = PlfdService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"service_count": summary["service_count"],
		"pending_change_count": summary["pending_change_count"],
	}


def register_foundation_service(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_foundation_service(
		service_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload.get("owner") or ""),
		tier=str(payload.get("tier") or "shared"),
		dependencies=list(payload.get("dependencies") or []),
		readiness_score=float(payload.get("readiness_score") or 0),
		configuration_baseline_present=bool(payload.get("configuration_baseline_present", True)),
		health_status=str(payload.get("health_status") or "healthy"),
		monitoring_enabled=bool(payload.get("monitoring_enabled", False)),
		rollback_plan_ref=str(payload.get("rollback_plan_ref") or ""),
		change_window_ref=str(payload.get("change_window_ref") or ""),
		metadata=dict(payload.get("metadata") or {}),
	)


def record_dependency(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_dependency(
		dependency_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		source_service_id=str(payload["source_service_id"]),
		target_service_id=str(payload["target_service_id"]),
		health_status=str(payload.get("health_status") or "healthy"),
		required=bool(payload.get("required", True)),
		evidence_ref=str(payload.get("evidence_ref") or ""),
	)


def attach_baseline(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.attach_baseline(
		baseline_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		service_id=str(payload["service_id"]),
		baseline_type=str(payload["baseline_type"]),
		evidence_ref=str(payload.get("evidence_ref") or ""),
		approved_by=str(payload.get("approved_by") or ""),
		status=str(payload.get("status") or "approved"),
	)


def assess_readiness(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.assess_readiness(
		assessment_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		service_id=str(payload["service_id"]),
	)


def propose_platform_change(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.propose_platform_change(
		change_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		service_id=str(payload["service_id"]),
		title=str(payload.get("title") or payload["id"]),
		owner=str(payload.get("owner") or ""),
		affected_capability_count=int(payload.get("affected_capability_count") or 1),
		dependencies_healthy=payload.get("dependencies_healthy"),
		approval_recorded=bool(payload.get("approval_recorded", False)),
		broad_review_recorded=bool(payload.get("broad_review_recorded", False)),
		security_review_recorded=bool(payload.get("security_review_recorded", False)),
		change_window_ref=str(payload.get("change_window_ref") or ""),
		rollback_plan_ref=str(payload.get("rollback_plan_ref") or ""),
	)


def approve_platform_change(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_platform_change(
		change_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		approver=str(payload.get("approver") or "platform-approver"),
		approval_recorded=bool(payload.get("approval_recorded", True)),
		broad_review_recorded=payload.get("broad_review_recorded"),
		security_review_recorded=payload.get("security_review_recorded"),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def register_plfd_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_plfd_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload["scope"]),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		agent_id=str(payload["id"]) if payload.get("id") else None,
	)


def validate_batch_foundation_mutation(event_stream: str) -> dict[str, Any]:
	return SERVICE.validate_batch_foundation_mutation(event_stream)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def list_plfd_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_plfd_agents(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
