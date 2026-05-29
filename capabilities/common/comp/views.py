"""UI metadata and dashboard helpers for the Compliance Management capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import CompService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: CompService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CompService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"framework_matrix": service.list_frameworks(tenant_id),
		"control_library": service.list_controls(tenant_id),
		"evidence_vault": service.list_evidence(tenant_id),
		"assessment_history": service.list_assessments(tenant_id),
		"remediation_board": service.list_findings(tenant_id),
		"report_builder": service.list_reports(tenant_id),
		"attestation_center": service.list_attestations(tenant_id),
		"audit_timeline": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def control_detail_model(service: CompService, tenant_id: str, control_id: str) -> dict[str, object]:
	controls = [control for control in service.list_controls(tenant_id) if control["id"] == control_id]
	evidence = [item for item in service.list_evidence(tenant_id) if item["control_id"] == control_id]
	assessments = [item for item in service.list_assessments(tenant_id) if item["control_id"] == control_id]
	findings = [item for item in service.list_findings(tenant_id) if item["control_id"] == control_id]
	return {
		"tenant_id": tenant_id,
		"control": controls[0] if controls else None,
		"evidence": evidence,
		"assessments": assessments,
		"findings": findings,
	}
