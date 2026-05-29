"""API helpers for APG Accessibility Services."""

from __future__ import annotations

from typing import Any

from .service import AccsService


SERVICE = AccsService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**SERVICE.compliance_summary(tenant_id),
	}


def register_standard(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_standard(
		standard_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or "WCAG"),
		version=str(payload.get("version") or "2.2"),
		level=str(payload.get("level") or "AA"),
		criteria=tuple(payload.get("criteria") or ()),
	)


def register_target(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_target(
		target_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		surface=str(payload.get("surface") or payload["id"]),
		route=str(payload.get("route") or "/"),
		owner=str(payload.get("owner") or "accessibility-owner"),
		published_ui=bool(payload.get("published_ui", False)),
		contrast_ratio=float(payload.get("contrast_ratio", 4.5)),
		semantic_labels_present=bool(payload.get("semantic_labels_present", True)),
		keyboard_navigation_present=bool(payload.get("keyboard_navigation_present", True)),
		media_content_present=bool(payload.get("media_content_present", False)),
		captions_available=bool(payload.get("captions_available", True)),
	)


def run_audit(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.run_audit(
		audit_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		standard_id=str(payload.get("standard_id") or "wcag_2_2_aa"),
		target_ids=tuple(payload.get("target_ids") or ()),
		remediation_owner=payload.get("remediation_owner"),
	)


def record_finding(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_finding(
		finding_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		target_id=str(payload.get("target_id") or "manual"),
		rule=str(payload.get("rule") or "manual_accessibility_review"),
		severity=str(payload.get("severity") or "low"),
		description=str(payload.get("description") or "Manual accessibility review finding."),
		remediation_owner=str(payload.get("remediation_owner") or "accessibility-owner"),
		status=str(payload.get("status") or "open"),
		evidence=dict(payload.get("evidence") or {}),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def update_remediation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.update_remediation(
		finding_id=str(payload["finding_id"]),
		status=str(payload.get("status") or "open"),
		review_recorded=bool(payload.get("review_recorded", False)),
		due_date=payload.get("due_date"),
	)


def validate_publication(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_publication(
		target_id=str(payload["target_id"]),
		tenant_id=payload.get("tenant_id"),
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


def list_targets(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_targets(tenant_id)


def list_findings(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_findings(tenant_id)


def list_remediations(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_remediations(tenant_id)
