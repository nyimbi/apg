"""API helpers for the Security Operations capability."""

from __future__ import annotations

from typing import Any

from .service import SeopService


SERVICE = SeopService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"detection_count": summary["detection_count"],
		"incident_count": summary["incident_count"],
		"open_incident_count": summary["open_incident_count"],
		"response_count": summary["response_count"],
		"seop_agent_count": summary["seop_agent_count"],
	}


def create_detection(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_detection(
		tenant_id=str(payload.get("tenant_id") or "default"),
		title=str(payload["title"]),
		alert_source=str(payload.get("alert_source") or ""),
		anomaly_confidence=float(payload.get("anomaly_confidence", 0)),
		severity=str(payload.get("severity") or "medium"),
		signal_refs=list(payload.get("signal_refs") or []),
		triage_review_recorded=bool(payload.get("triage_review_recorded", False)),
		owner=payload.get("owner"),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def open_incident(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.open_incident(
		tenant_id=str(payload.get("tenant_id") or "default"),
		title=str(payload["title"]),
		owner=str(payload.get("owner") or ""),
		severity=str(payload.get("severity") or "medium"),
		detection_ids=list(payload.get("detection_ids") or []),
		escalation_recorded=bool(payload.get("escalation_recorded", False)),
		evidence_refs=list(payload.get("evidence_refs") or []),
	)


def approve_playbook(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_playbook(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner=str(payload.get("owner") or ""),
		steps=list(payload.get("steps") or []),
		approved_by=str(payload.get("approved_by") or ""),
	)


def execute_response(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.execute_response(
		tenant_id=str(payload.get("tenant_id") or "default"),
		incident_id=str(payload["incident_id"]),
		playbook_id=str(payload["playbook_id"]),
		action=str(payload["action"]),
		actor=str(payload.get("actor") or ""),
		containment_reviewed=bool(payload.get("containment_reviewed", True)),
	)


def record_posture_control(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_posture_control(
		tenant_id=str(payload.get("tenant_id") or "default"),
		control_id=str(payload["control_id"]),
		domain=str(payload.get("domain") or "security_operations"),
		coverage=float(payload.get("coverage", 0)),
		owner=str(payload.get("owner") or ""),
	)


def close_incident(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.close_incident(
		tenant_id=str(payload.get("tenant_id") or "default"),
		incident_id=str(payload["incident_id"]),
		closure_evidence=str(payload.get("closure_evidence") or ""),
		actor=str(payload.get("actor") or ""),
		post_incident_review=str(payload.get("post_incident_review") or ""),
		compliance_mapping=str(payload.get("compliance_mapping") or ""),
	)


def register_seop_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_seop_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload.get("runtime") or ""),
		role=str(payload.get("role") or ""),
		scope=str(payload.get("scope") or ""),
		owner=str(payload.get("owner") or "secops"),
		human_approval_required=bool(payload.get("human_approval_required", True)),
	)


def validate_agent_response_action(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_agent_response_action(
		tenant_id=str(payload.get("tenant_id") or "default"),
		agent_id=str(payload["agent_id"]),
		incident_severity=str(payload.get("incident_severity") or "medium"),
		human_approval_recorded=bool(payload.get("human_approval_recorded", False)),
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


def list_security_operations(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"detections": SERVICE.list_detections(tenant_id),
		"incidents": SERVICE.list_incidents(tenant_id),
		"playbooks": SERVICE.list_playbooks(tenant_id),
		"responses": SERVICE.list_responses(tenant_id),
		"posture_controls": SERVICE.list_posture_controls(tenant_id),
		"seop_agents": SERVICE.list_seop_agents(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}
