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
		legal_hold=_payload_bool(payload, "legal_hold", False),
	)


def create_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_snapshot(
		snapshot_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		plan_id=str(payload["plan_id"]),
		source_id=str(payload["source_id"]),
		size_bytes=int(payload.get("size_bytes", 0)),
		encrypted=_payload_bool(payload, "encrypted", True),
		integrity_check_passed=_payload_bool(payload, "integrity_check_passed", True),
		lineage=[str(item) for item in payload.get("lineage", [])],
		region=str(payload.get("region") or "primary"),
		data_fingerprint=payload.get("data_fingerprint"),
	)


def request_restore_approval(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_restore_approval(
		approval_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		snapshot_id=str(payload["snapshot_id"]),
		target_environment=str(payload.get("target_environment") or "production"),
		requested_by=str(payload["requested_by"]),
		justification=str(payload["justification"]),
		point_in_time=payload.get("point_in_time"),
	)


def decide_restore_approval(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_restore_approval(
		approval_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload["notes"]),
	)


def restore_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.restore_snapshot(
		restore_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		snapshot_id=str(payload["snapshot_id"]),
		target_environment=str(payload.get("target_environment") or "test"),
		requested_by=str(payload.get("requested_by") or "operator"),
		integrity_check_passed=_payload_bool(payload, "integrity_check_passed", True),
		approval_recorded=_payload_bool(payload, "approval_recorded", False),
		point_in_time=payload.get("point_in_time"),
		days_since_restore_test=int(payload.get("days_since_restore_test", 0)),
		restore_test_review_recorded=_payload_bool(payload, "restore_test_review_recorded", True),
		rto_minutes=int(payload.get("rto_minutes", 0)),
		approval_id=str(payload["approval_id"]) if payload.get("approval_id") else None,
	)


def approve_restore(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_restore(
		restore_id=str(payload["id"]),
		reviewer=str(payload.get("reviewer") or "reviewer"),
		tenant_id=str(payload["tenant_id"]) if payload.get("tenant_id") else None,
		notes=str(payload.get("notes") or "Approved restore review."),
	)


def record_restore_test(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_restore_test(
		report_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		plan_id=str(payload["plan_id"]),
		rto_minutes=int(payload.get("rto_minutes", 0)),
		days_since_restore_test=int(payload.get("days_since_restore_test", 0)),
		restore_test_review_recorded=_payload_bool(payload, "restore_test_review_recorded", True),
	)


def request_retention_disposition(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_retention_disposition(
		disposition_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		snapshot_id=str(payload["snapshot_id"]),
		action=str(payload.get("action") or "delete"),
		requested_by=str(payload["requested_by"]),
		reason=str(payload["reason"]),
	)


def decide_retention_disposition(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_retention_disposition(
		disposition_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload["notes"]),
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


def list_restore_approvals(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_restore_approvals(tenant_id)


def list_reports(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_reports(tenant_id)


def list_retention_dispositions(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_retention_dispositions(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)


def _payload_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
	value = payload.get(key, default)
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return bool(value)
