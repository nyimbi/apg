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
		"streaming": contract["streaming"],
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
		tenant_id=payload.get("tenant_id"),
	)


def record_review(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_review(
		finding_id=str(payload["finding_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload.get("notes") or ""),
	)


def close_finding(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.close_finding(
		finding_id=str(payload["finding_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		resolution=str(payload.get("resolution") or ""),
	)


def record_accessibility_exception(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_accessibility_exception(
		exception_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		finding_id=str(payload["finding_id"]),
		approver=str(payload["approver"]),
		reason=str(payload.get("reason") or ""),
		expires_on=str(payload.get("expires_on") or ""),
		compensating_controls=tuple(payload.get("compensating_controls") or ()),
		status=str(payload.get("status") or "approved"),
	)


def register_accessibility_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_accessibility_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		runtime=str(payload.get("runtime") or "codex"),
		role=str(payload.get("role") or "audit_reviewer"),
		scope=str(payload.get("scope") or ""),
		registered=bool(payload.get("registered", True)),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		policy_ref=payload.get("policy_ref"),
		status=str(payload.get("status") or "active"),
	)


def validate_batch_accessibility_mutation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_batch_accessibility_mutation(
		tenant_id=str(payload.get("tenant_id") or "default"),
		event_stream=str(payload.get("event_stream") or "bytewax"),
		mutation_count=int(payload.get("mutation_count") or 0),
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


def list_reviews(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_reviews(tenant_id)


def list_accessibility_exceptions(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_accessibility_exceptions(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)


def list_accessibility_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_accessibility_agents(tenant_id)
