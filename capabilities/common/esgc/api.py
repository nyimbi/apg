"""API helpers for APG ESG/Carbon Tracking."""

from __future__ import annotations

from typing import Any

from .service import EsgcService


SERVICE = EsgcService()


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


def create_inventory(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_inventory(
		inventory_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		organization=str(payload["organization"]),
		owner=str(payload["owner"]),
		reporting_year=int(payload["reporting_year"]),
		boundary_ref=str(payload["boundary_ref"]),
		geospatial_boundary=str(payload["geospatial_boundary"]),
		compliance_framework=str(payload["compliance_framework"]),
		status=str(payload.get("status") or "active"),
	)


def register_factor(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_factor(
		factor_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		scope=str(payload["scope"]),
		unit=str(payload["unit"]),
		co2e_per_unit=float(payload["co2e_per_unit"]),
		source=str(payload["source"]),
		source_evidence=str(payload["source_evidence"]),
		version=str(payload["version"]),
		approved_source=bool(payload.get("approved_source", False)),
		status=str(payload.get("status") or "active"),
	)


def record_activity(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_activity(
		activity_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		inventory_id=str(payload["inventory_id"]),
		factor_id=str(payload["factor_id"]),
		activity_type=str(payload["activity_type"]),
		quantity=float(payload["quantity"]),
		unit=str(payload["unit"]),
		evidence_ref=str(payload["evidence_ref"]),
		expected_max_quantity=payload.get("expected_max_quantity"),
		anomaly_review_recorded=bool(payload.get("anomaly_review_recorded", False)),
	)


def publish_report(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_report(
		report_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		inventory_id=str(payload["inventory_id"]),
		report_type=str(payload["report_type"]),
		period=str(payload["period"]),
		compliance_mapping=str(payload["compliance_mapping"]),
		audit_evidence_ref=str(payload["audit_evidence_ref"]),
		approved_by=str(payload["approved_by"]),
		approval_recorded=bool(payload.get("approval_recorded", False)),
	)


def create_target(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_target(
		target_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		inventory_id=str(payload["inventory_id"]),
		name=str(payload["name"]),
		baseline_year=int(payload["baseline_year"]),
		target_year=int(payload["target_year"]),
		baseline_co2e_tonnes=float(payload["baseline_co2e_tonnes"]),
		target_reduction_percent=float(payload["target_reduction_percent"]),
	)


def list_inventories(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_inventories(tenant_id)


def list_factors(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_factors(tenant_id)


def list_activities(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_activities(tenant_id)


def list_reports(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_reports(tenant_id)


def list_targets(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_targets(tenant_id)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)
