"""Dependency-light API helpers for the APG RCM capability package."""

from __future__ import annotations

from typing import Any

from .service import GrcRcmService


SERVICE = GrcRcmService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"record_count": len(SERVICE.list_records(tenant_id)),
		"summary": summary,
	}


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def register_risk(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_risk(
		risk_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		title=str(payload["title"]),
		category=str(payload.get("category") or "operational"),
		owner_id=str(payload["owner_id"]),
		probability=float(payload["probability"]),
		impact=float(payload["impact"]),
		control_effectiveness=float(payload.get("control_effectiveness", 0.0)),
		tags=list(payload.get("tags") or []),
		metadata=dict(payload.get("metadata") or {}),
		review_recorded=bool(payload.get("review_recorded", True)),
		policy_attached=bool(payload.get("policy_attached", True)),
	)


def register_control(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_control(
		control_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner_id=str(payload["owner_id"]),
		control_type=str(payload.get("control_type") or "preventive"),
		mapped_risk_ids=list(payload.get("mapped_risk_ids") or []),
		effectiveness=float(payload.get("effectiveness", 0.0)),
		test_frequency_days=int(payload.get("test_frequency_days", 90)),
		metadata=dict(payload.get("metadata") or {}),
		policy_attached=bool(payload.get("policy_attached", True)),
	)


def add_compliance_obligation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.add_compliance_obligation(
		obligation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		framework=str(payload["framework"]),
		requirement=str(payload["requirement"]),
		owner_id=str(payload["owner_id"]),
		jurisdiction=str(payload.get("jurisdiction") or "global"),
		due_date=str(payload["due_date"]),
		mapped_control_ids=list(payload.get("mapped_control_ids") or []),
		metadata=dict(payload.get("metadata") or {}),
		policy_attached=bool(payload.get("policy_attached", True)),
	)


def assess_control(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.assess_control(
		assessment_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		control_id=str(payload["control_id"]),
		assessor_id=str(payload["assessor_id"]),
		design_effective=bool(payload["design_effective"]),
		operating_effective=bool(payload["operating_effective"]),
		evidence_refs=list(payload.get("evidence_refs") or []),
		findings=list(payload.get("findings") or []),
		review_recorded=bool(payload.get("review_recorded", True)),
	)


def collect_evidence(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.collect_evidence(
		evidence_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		source=str(payload["source"]),
		linked_control_id=payload.get("linked_control_id"),
		linked_obligation_id=payload.get("linked_obligation_id"),
		encrypted=bool(payload.get("encrypted", True)),
		retention_days=int(payload.get("retention_days", 2555)),
		metadata=dict(payload.get("metadata") or {}),
		policy_attached=bool(payload.get("policy_attached", True)),
	)


def record_governance_decision(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_governance_decision(
		decision_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		title=str(payload["title"]),
		decision_type=str(payload.get("decision_type") or "policy_approval"),
		approver_id=str(payload["approver_id"]),
		related_risk_ids=list(payload.get("related_risk_ids") or []),
		rationale=str(payload.get("rationale") or ""),
		approved=bool(payload.get("approved", True)),
		review_recorded=bool(payload.get("review_recorded", True)),
		policy_attached=bool(payload.get("policy_attached", True)),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def dashboard_summary(tenant_id: str | None = None) -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
