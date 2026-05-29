"""API helpers for APG Backup and Restore."""

from __future__ import annotations

from typing import Any

from .service import BkupService


SERVICE = BkupService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**SERVICE.continuity_summary(tenant_id),
	}


def create_backup_plan(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_backup_plan(
		plan_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload.get("owner") or ""),
		schedule=str(payload.get("schedule") or ""),
		sources=[str(item) for item in payload.get("sources", [])],
		retention_days=int(payload.get("retention_days", 30)),
		rpo_minutes=int(payload.get("rpo_minutes", 60)),
		legal_hold=bool(payload.get("legal_hold", False)),
	)


def create_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_snapshot(
		snapshot_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		plan_id=str(payload["plan_id"]),
		source_id=str(payload["source_id"]),
		size_bytes=int(payload.get("size_bytes", 0)),
		encrypted=bool(payload.get("encrypted", True)),
		integrity_check_passed=bool(payload.get("integrity_check_passed", True)),
		lineage=[str(item) for item in payload.get("lineage", [])],
		region=str(payload.get("region") or "primary"),
		data_fingerprint=payload.get("data_fingerprint"),
	)


def restore_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.restore_snapshot(
		restore_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		snapshot_id=str(payload["snapshot_id"]),
		target_environment=str(payload.get("target_environment") or "test"),
		requested_by=str(payload.get("requested_by") or "operator"),
		integrity_check_passed=bool(payload.get("integrity_check_passed", True)),
		approval_recorded=bool(payload.get("approval_recorded", False)),
		point_in_time=payload.get("point_in_time"),
		days_since_restore_test=int(payload.get("days_since_restore_test", 0)),
		restore_test_review_recorded=bool(payload.get("restore_test_review_recorded", True)),
		rto_minutes=int(payload.get("rto_minutes", 0)),
	)


def approve_restore(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_restore(
		restore_id=str(payload["id"]),
		reviewer=str(payload.get("reviewer") or "reviewer"),
	)


def record_restore_test(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_restore_test(
		report_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		plan_id=str(payload["plan_id"]),
		rto_minutes=int(payload.get("rto_minutes", 0)),
		days_since_restore_test=int(payload.get("days_since_restore_test", 0)),
		restore_test_review_recorded=bool(payload.get("restore_test_review_recorded", True)),
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


def list_plans(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_plans(tenant_id)


def list_snapshots(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_snapshots(tenant_id)


def list_restores(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_restores(tenant_id)


def list_reports(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_reports(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)
