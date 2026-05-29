"""API helpers for the Shutdown and Lifecycle Control capability."""

from __future__ import annotations

from typing import Any

from .service import ShdnService


SERVICE = ShdnService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"target_count": summary["target_count"],
		"active_plan_count": summary["active_plan_count"],
		"shutdown_count": summary["shutdown_count"],
		"recovery_count": summary["recovery_count"],
	}


def register_service(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_service(
		tenant_id=str(payload.get("tenant_id") or "default"),
		target_id=str(payload["target_id"]),
		target_type=str(payload.get("target_type") or "service"),
		owner=str(payload.get("owner") or ""),
		environment=str(payload.get("environment") or "production"),
		dependencies=list(payload.get("dependencies") or []),
		criticality=str(payload.get("criticality") or "normal"),
		drain_timeout_seconds=int(payload.get("drain_timeout_seconds", 300)),
		health_gate_ref=payload.get("health_gate_ref"),
	)


def create_shutdown_plan(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_shutdown_plan(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner=str(payload.get("owner") or ""),
		target_ids=list(payload.get("target_ids") or []),
		reason=str(payload.get("reason") or ""),
		rollback_plan_ref=str(payload.get("rollback_plan_ref") or ""),
		restart_sequence=list(payload.get("restart_sequence") or []),
		approved_by=payload.get("approved_by"),
		scheduled_for=payload.get("scheduled_for"),
		maintenance_window_ref=payload.get("maintenance_window_ref"),
	)


def start_drain(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.start_drain(
		tenant_id=str(payload.get("tenant_id") or "default"),
		plan_id=str(payload["plan_id"]),
		target_id=str(payload["target_id"]),
		active_sessions=int(payload.get("active_sessions", 0)),
		queue_depth=int(payload.get("queue_depth", 0)),
	)


def record_backup_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_backup_snapshot(
		tenant_id=str(payload.get("tenant_id") or "default"),
		plan_id=str(payload["plan_id"]),
		target_id=str(payload["target_id"]),
		evidence_ref=str(payload.get("evidence_ref") or ""),
		restore_test_ref=str(payload.get("restore_test_ref") or ""),
		verified=bool(payload.get("verified", True)),
	)


def execute_shutdown(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.execute_shutdown(
		tenant_id=str(payload.get("tenant_id") or "default"),
		plan_id=str(payload["plan_id"]),
		target_id=str(payload["target_id"]),
		actor=str(payload.get("actor") or ""),
		health_gate_ref=str(payload.get("health_gate_ref") or ""),
		force_shutdown=bool(payload.get("force_shutdown", False)),
		force_review_recorded=bool(payload.get("force_review_recorded", False)),
	)


def record_recovery(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_recovery(
		tenant_id=str(payload.get("tenant_id") or "default"),
		plan_id=str(payload["plan_id"]),
		target_id=str(payload["target_id"]),
		actor=str(payload.get("actor") or ""),
		evidence_ref=str(payload.get("evidence_ref") or ""),
		post_shutdown_health_check_ref=str(payload.get("post_shutdown_health_check_ref") or ""),
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


def list_lifecycle_control(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"targets": SERVICE.list_targets(tenant_id),
		"plans": SERVICE.list_plans(tenant_id),
		"drains": SERVICE.list_drains(tenant_id),
		"snapshots": SERVICE.list_snapshots(tenant_id),
		"executions": SERVICE.list_executions(tenant_id),
		"recoveries": SERVICE.list_recoveries(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}
