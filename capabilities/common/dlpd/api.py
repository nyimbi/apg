"""API helpers for the APG Data Loss Prevention capability."""

from __future__ import annotations

from typing import Any

from .service import DlpdService


SERVICE = DlpdService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"policy_count": summary["policy_count"],
		"classifier_count": summary["classifier_count"],
		"inspection_count": summary["inspection_count"],
		"open_incident_count": summary["open_incident_count"],
	}


def register_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_policy(
		policy_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner=str(payload["owner"]),
		channels=list(payload.get("channels") or []),
		classifiers=list(payload.get("classifiers") or []),
		default_action=str(payload.get("default_action") or "quarantine"),
		egress_policy_attached=bool(payload.get("egress_policy_attached", True)),
		large_export_review_required=bool(payload.get("large_export_review_required", True)),
	)


def register_classifier(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_classifier(
		classifier_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		classifier_type=str(payload.get("classifier_type") or "built_in"),
		sensitivity_label=str(payload.get("sensitivity_label") or "confidential"),
		pattern_keys=list(payload.get("pattern_keys") or []),
		reviewed_by=payload.get("reviewed_by"),
		confidence_threshold=payload.get("confidence_threshold"),
	)


def classify_content(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.classify_content(
		tenant_id=str(payload.get("tenant_id") or "default"),
		content=str(payload.get("content") or ""),
		classifier_ids=list(payload.get("classifier_ids") or []),
	)


def inspect_egress(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.inspect_egress(
		inspection_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		policy_id=str(payload["policy_id"]),
		channel=str(payload["channel"]),
		subject_id=str(payload["subject_id"]),
		destination=str(payload["destination"]),
		content=str(payload.get("content") or ""),
		record_count=int(payload.get("record_count") or 1),
		classification_label=payload.get("classification_label"),
		auto_classify=bool(payload.get("auto_classify", True)),
		review_recorded=bool(payload.get("review_recorded", False)),
		quarantine_encrypted=bool(payload.get("quarantine_encrypted", True)),
	)


def review_export(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.review_export(
		inspection_id=str(payload["inspection_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
	)


def resolve_incident(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.resolve_incident(
		incident_id=str(payload["incident_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		actor=str(payload["actor"]),
		resolution=str(payload["resolution"]),
	)


def dlp_state(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"summary": SERVICE.dashboard_summary(tenant_id),
		"policies": SERVICE.list_policies(tenant_id),
		"classifiers": SERVICE.list_classifiers(tenant_id),
		"inspections": SERVICE.list_inspections(tenant_id),
		"quarantine": SERVICE.list_quarantine(tenant_id),
		"incidents": SERVICE.list_incidents(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
	}
