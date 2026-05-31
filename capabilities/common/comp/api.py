"""API helpers for the Compliance Management capability."""

from __future__ import annotations

from typing import Any

from .service import CompService


SERVICE = CompService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"framework_count": summary["framework_count"],
		"control_count": summary["control_count"],
		"open_finding_count": summary["open_finding_count"],
		"compliance_agent_count": summary["compliance_agent_count"],
		"lifecycle_batch_count": summary["lifecycle_batch_count"],
		"coverage": summary["coverage"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
	}


def register_framework(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_framework(
		framework_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner=str(payload["owner"]),
		obligations=[str(item) for item in payload.get("obligations", [])],
		policy_version=str(payload.get("policy_version") or "v1"),
	)


def create_control(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_control(
		control_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		framework_id=str(payload["framework_id"]),
		name=str(payload["name"]),
		owner=str(payload.get("owner") or ""),
		control_type=str(payload.get("control_type") or "preventive"),
		regulated_data_scope=bool(payload.get("regulated_data_scope", False)),
		dlp_policy_linked=bool(payload.get("dlp_policy_linked", False)),
		testing_frequency_days=payload.get("testing_frequency_days"),
	)


def record_evidence(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_evidence(
		evidence_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		control_id=str(payload["control_id"]),
		source=str(payload["source"]),
		collected_by=str(payload["collected_by"]),
		encrypted=bool(payload.get("encrypted", False)),
		immutable_reference=payload.get("immutable_reference"),
		metadata=dict(payload.get("metadata") or {}),
	)


def assess_control(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.assess_control(
		assessment_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		control_id=str(payload["control_id"]),
		evidence_id=str(payload["evidence_id"]),
		tested_by=str(payload["tested_by"]),
	)


def open_finding(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.open_finding(
		finding_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		control_id=str(payload["control_id"]),
		severity=str(payload["severity"]),
		description=str(payload["description"]),
		owner=str(payload["owner"]),
		remediation_plan=str(payload.get("remediation_plan") or ""),
	)


def resolve_finding(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.resolve_finding(
		finding_id=str(payload["finding_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		resolved_by=str(payload["resolved_by"]),
		resolution=str(payload.get("resolution") or ""),
		evidence_id=payload.get("evidence_id"),
	)


def prepare_report(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.prepare_report(
		report_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		framework_id=str(payload["framework_id"]),
		period=str(payload["period"]),
		prepared_by=str(payload["prepared_by"]),
	)


def approve_report(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_report(
		report_id=str(payload["report_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		approved_by=str(payload["approved_by"]),
	)


def attest_report(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.attest_report(
		attestation_id=str(payload["id"]),
		report_id=str(payload["report_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		attested_by=str(payload["attested_by"]),
		statement=str(payload["statement"]),
	)


def publish_report(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_report(
		report_id=str(payload["report_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
	)


def register_compliance_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_compliance_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload["scope"]),
		owner=str(payload["owner"]),
		purpose=str(payload["purpose"]),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		human_approval_required=bool(payload.get("human_approval_required", False)),
	)


def validate_lifecycle_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_comp_lifecycle_batch(
		tenant_id=str(payload.get("tenant_id") or "default"),
		event_stream=str(payload.get("event_stream") or "bytewax"),
		mutation_count=int(payload.get("mutation_count") or 0),
		operation=str(payload.get("operation") or "compliance_agent_batch"),
		batch_id=payload.get("batch_id"),
	)


def compliance_state(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"summary": SERVICE.dashboard_summary(tenant_id),
		"frameworks": SERVICE.list_frameworks(tenant_id),
		"controls": SERVICE.list_controls(tenant_id),
		"evidence": SERVICE.list_evidence(tenant_id),
		"assessments": SERVICE.list_assessments(tenant_id),
		"findings": SERVICE.list_findings(tenant_id),
		"reports": SERVICE.list_reports(tenant_id),
		"attestations": SERVICE.list_attestations(tenant_id),
		"compliance_agents": SERVICE.list_compliance_agents(tenant_id),
		"lifecycle_batches": SERVICE.list_lifecycle_batches(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
	}
