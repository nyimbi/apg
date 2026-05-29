"""API helpers for the Security Framework capability."""

from __future__ import annotations

from typing import Any

from .service import SecuService


SERVICE = SecuService()


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
		"assessment_count": summary["assessment_count"],
		"active_threat_count": summary["active_threat_count"],
		"compliance_gap_count": summary["compliance_gap_count"],
		"policy_exception_count": summary["policy_exception_count"],
		"open_incident_count": summary["open_incident_count"],
	}


def create_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_policy(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner=str(payload["owner"]),
		security_level=str(payload.get("security_level") or "confidential"),
		required_controls=list(payload.get("required_controls") or []),
		applies_to=list(payload.get("applies_to") or []),
		enabled=bool(payload.get("enabled", True)),
		tags=list(payload.get("tags") or []),
	)


def record_device_posture(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_device_posture(
		tenant_id=str(payload.get("tenant_id") or "default"),
		device_id=str(payload["device_id"]),
		user_id=str(payload["user_id"]),
		trust_state=str(payload.get("trust_state") or "trusted"),
		managed=bool(payload.get("managed", True)),
		risk_score=payload.get("risk_score", 0),
		indicators=list(payload.get("indicators") or []),
	)


def register_threat_indicator(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_threat_indicator(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		indicator_type=str(payload.get("indicator_type") or "indicator"),
		value=str(payload["value"]),
		severity=str(payload.get("severity") or "medium"),
		source=str(payload.get("source") or "manual"),
		ttl_hours=int(payload.get("ttl_hours") or 24),
	)


def assess_access(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.assess_access(
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		subject_type=str(payload.get("subject_type") or "user"),
		risk_score=payload.get("risk_score", 0),
		device_id=payload.get("device_id"),
		is_known_malicious=bool(payload.get("is_known_malicious", False)),
		challenge_completed=bool(payload.get("challenge_completed", False)),
		compliance_violation=bool(payload.get("compliance_violation", False)),
		audit_evidence_attached=bool(payload.get("audit_evidence_attached", True)),
	)


def record_compliance_control(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_compliance_control(
		tenant_id=str(payload.get("tenant_id") or "default"),
		framework=str(payload.get("framework") or "iso_27001"),
		control_id=str(payload["control_id"]),
		owner=str(payload["owner"]),
		compliant=bool(payload.get("compliant", False)),
		evidence_ref=payload.get("evidence_ref"),
		waived=bool(payload.get("waived", False)),
	)


def request_policy_exception(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_policy_exception(
		tenant_id=str(payload.get("tenant_id") or "default"),
		exception_id=str(payload["id"]),
		policy_id=str(payload["policy_id"]),
		requested_by=str(payload.get("requested_by") or ""),
		reason=str(payload.get("reason") or ""),
		expires_at=str(payload.get("expires_at") or ""),
	)


def decide_policy_exception(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_policy_exception(
		tenant_id=str(payload.get("tenant_id") or "default"),
		exception_id=str(payload["id"]),
		reviewer=str(payload.get("reviewer") or ""),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload.get("notes") or ""),
	)


def open_incident(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.open_incident(
		tenant_id=str(payload.get("tenant_id") or "default"),
		incident_id=str(payload["id"]),
		title=str(payload.get("title") or ""),
		severity=str(payload.get("severity") or "medium"),
		opened_by=str(payload.get("opened_by") or ""),
		containment_plan=str(payload.get("containment_plan") or ""),
	)


def contain_incident(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.contain_incident(
		tenant_id=str(payload.get("tenant_id") or "default"),
		incident_id=str(payload["id"]),
		actor=str(payload.get("actor") or ""),
		containment_action=str(payload.get("containment_action") or ""),
		containment_evidence=str(payload.get("containment_evidence") or ""),
	)


def resolve_incident(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.resolve_incident(
		tenant_id=str(payload.get("tenant_id") or "default"),
		incident_id=str(payload["id"]),
		resolved_by=str(payload.get("resolved_by") or ""),
		resolution=str(payload.get("resolution") or ""),
		notes=str(payload.get("notes") or ""),
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


def list_security_posture(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"policies": SERVICE.list_policies(tenant_id),
		"devices": SERVICE.list_devices(tenant_id),
		"threats": SERVICE.list_threats(tenant_id),
		"assessments": SERVICE.list_assessments(tenant_id),
		"controls": SERVICE.list_controls(tenant_id),
		"policy_exceptions": SERVICE.list_policy_exceptions(tenant_id),
		"incidents": SERVICE.list_incidents(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}
