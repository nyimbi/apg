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
		"compliance_agents": service.list_compliance_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"audit_timeline": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def framework_matrix_model(service: CompService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompService()
	frameworks = service.list_frameworks(tenant_id)
	return {
		"route": "/comp/frameworks",
		"tenant_id": tenant_id,
		"frameworks": frameworks,
		"active": [framework for framework in frameworks if framework["status"] == "active"],
		"actions": ["register_framework", "map_obligations", "version_policy"],
		"theme_component": "framework_matrix",
	}


def control_library_model(service: CompService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompService()
	controls = service.list_controls(tenant_id)
	return {
		"route": "/comp/controls",
		"tenant_id": tenant_id,
		"controls": controls,
		"regulated": [control for control in controls if control["regulated_data_scope"]],
		"dlp_required": [control for control in controls if control["regulated_data_scope"] and not control["dlp_policy_linked"]],
		"actions": ["create_control", "record_evidence", "assess_control"],
		"theme_component": "control_card",
	}


def evidence_vault_model(service: CompService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompService()
	evidence = service.list_evidence(tenant_id)
	return {
		"route": "/comp/evidence",
		"tenant_id": tenant_id,
		"evidence": evidence,
		"encrypted": [item for item in evidence if item["encrypted"]],
		"missing_hash": [item for item in evidence if not item["immutable_reference"]],
		"actions": ["record_evidence", "assess_control"],
		"theme_component": "evidence_vault",
	}


def assessment_workbench_model(service: CompService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompService()
	assessments = service.list_assessments(tenant_id)
	return {
		"route": "/comp/assessments",
		"tenant_id": tenant_id,
		"assessments": assessments,
		"review_required": [assessment for assessment in assessments if assessment["result"] == "review_required"],
		"failed": [assessment for assessment in assessments if assessment["result"] not in {"effective", "review_required"}],
		"actions": ["assess_control", "open_finding"],
		"theme_component": "assessment_workbench",
	}


def finding_board_model(service: CompService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompService()
	findings = service.list_findings(tenant_id)
	return {
		"route": "/comp/findings",
		"tenant_id": tenant_id,
		"findings": findings,
		"open": [finding for finding in findings if finding["status"] == "open"],
		"critical": [finding for finding in findings if finding["severity"] == "critical" and finding["status"] == "open"],
		"escalated": [finding for finding in findings if finding["escalated"]],
		"actions": ["open_finding", "resolve_finding", "escalate_overdue_findings"],
		"theme_component": "finding_board",
	}


def report_builder_model(service: CompService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompService()
	reports = service.list_reports(tenant_id)
	return {
		"route": "/comp/reports",
		"tenant_id": tenant_id,
		"reports": reports,
		"drafts": [report for report in reports if report["status"] == "draft"],
		"approved": [report for report in reports if report["status"] == "approved"],
		"published": [report for report in reports if report["status"] == "published"],
		"actions": ["prepare_report", "approve_report", "attest_report", "publish_report"],
		"theme_component": "report_builder",
	}


def attestation_center_model(service: CompService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompService()
	return {
		"route": "/comp/attestations",
		"tenant_id": tenant_id,
		"attestations": service.list_attestations(tenant_id),
		"actions": ["attest_report", "publish_report"],
		"theme_component": "attestation_center",
	}


def audit_model(service: CompService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompService()
	return {
		"route": "/comp/audit",
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"theme_component": "audit_timeline",
	}


def compliance_agent_roster_model(service: CompService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompService()
	contract = service.describe(tenant_id)
	agents = service.list_compliance_agents(tenant_id)
	return {
		"route": "/comp/agents",
		"tenant_id": tenant_id,
		"agents": agents,
		"active": [agent for agent in agents if agent["status"] == "active"],
		"pending_review": [agent for agent in agents if agent["status"] == "pending_review"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"actions": ["register_compliance_agent", "record_human_compliance_agent_approval"],
		"theme_component": "compliance_agent_roster",
	}


def lifecycle_batch_model(service: CompService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompService()
	contract = service.describe(tenant_id)
	batches = service.list_lifecycle_batches(tenant_id)
	return {
		"route": "/comp/lifecycle",
		"tenant_id": tenant_id,
		"lifecycle_stream": contract["streaming"]["lifecycle_stream"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"batches": batches,
		"accepted": [batch for batch in batches if batch["status"] == "accepted"],
		"denied": [batch for batch in batches if batch["status"] == "denied"],
		"actions": ["validate_lifecycle_batch", "inspect_bytewax_lifecycle"],
		"theme_component": "bytewax_lifecycle_panel",
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/comp/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"theme": contract["theme"],
		"permissions": [route["permission"] for route in contract["ui"]["routes"]],
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
