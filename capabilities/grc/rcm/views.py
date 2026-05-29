"""Dependency-light view models for the APG RCM capability package."""

from __future__ import annotations

from typing import Any

from .service import GrcRcmService


def capability_routes(tenant_id: str = "default") -> list[dict[str, Any]]:
	"""Return route metadata from the executable capability contract."""
	contract = GrcRcmService().describe(tenant_id)
	return list(contract["ui"]["routes"])


def dashboard_model(service: GrcRcmService | None = None, tenant_id: str | None = None) -> dict[str, Any]:
	"""Return a compact executive RCM dashboard view model."""
	svc = service or GrcRcmService()
	contract = svc.describe(tenant_id or "default")
	return {
		"title": "Governance, Risk, and Compliance",
		"theme": contract["theme"],
		"summary": svc.dashboard_summary(tenant_id),
		"routes": contract["ui"]["routes"],
		"panels": [
			{"id": "risk_heatmap", "title": "Risk Heatmap", "count": len(svc.list_risks(tenant_id))},
			{"id": "control_testing", "title": "Control Testing", "count": len(svc.list_assessments(tenant_id))},
			{"id": "compliance_obligations", "title": "Compliance Obligations", "count": len(svc.list_obligations(tenant_id))},
			{"id": "governance_decisions", "title": "Governance Decisions", "count": len(svc.list_decisions(tenant_id))},
		],
	}


def risk_register_model(service: GrcRcmService | None = None, tenant_id: str | None = None) -> dict[str, Any]:
	"""Return risk register rows grouped for a workbench screen."""
	svc = service or GrcRcmService()
	risks = svc.list_risks(tenant_id)
	return {
		"title": "Risk Register",
		"rows": risks,
		"high_priority": [risk for risk in risks if risk["risk_level"] in {"high", "critical"}],
		"columns": ["title", "category", "owner_id", "residual_score", "risk_level", "status"],
	}


def control_testing_model(service: GrcRcmService | None = None, tenant_id: str | None = None) -> dict[str, Any]:
	"""Return control and assessment state for testing queues."""
	svc = service or GrcRcmService()
	return {
		"title": "Control Testing",
		"controls": svc.list_controls(tenant_id),
		"assessments": svc.list_assessments(tenant_id),
		"evidence": svc.list_evidence(tenant_id),
	}


def compliance_workbench_model(service: GrcRcmService | None = None, tenant_id: str | None = None) -> dict[str, Any]:
	"""Return obligations and evidence for compliance workbench screens."""
	svc = service or GrcRcmService()
	return {
		"title": "Compliance Workbench",
		"obligations": svc.list_obligations(tenant_id),
		"evidence": svc.list_evidence(tenant_id),
		"failed_assessments": [
			item for item in svc.list_assessments(tenant_id)
			if item["status"] in {"non_compliant", "partially_compliant"}
		],
	}


def governance_board_model(service: GrcRcmService | None = None, tenant_id: str | None = None) -> dict[str, Any]:
	"""Return governance decisions and policy evidence for board review."""
	svc = service or GrcRcmService()
	return {
		"title": "Governance Board",
		"decisions": svc.list_decisions(tenant_id),
		"audit_events": svc.list_audit_events(tenant_id),
		"summary": svc.dashboard_summary(tenant_id),
	}
