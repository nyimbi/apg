"""View models for APG Pharma Regulatory Compliance screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import RegulatoryComplianceService


def dashboard_model(service: RegulatoryComplianceService, tenant_id: str = "default") -> dict[str, Any]:
	"""Compose the regulatory compliance dashboard view model."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Regulatory Compliance",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
		"overdue_commitments": len(service.check_overdue_commitments(tenant_id)),
	}


def compliance_register_model(service: RegulatoryComplianceService,
								tenant_id: str = "default") -> dict[str, Any]:
	"""Compliance framework register view."""
	frameworks = service.list_frameworks(tenant_id)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Compliance Register",
		"tenant_id": tenant_id,
		"count": len(frameworks),
		"items": [f.model_dump() for f in frameworks],
		"supported_frameworks": contract["configuration"]["compliance_frameworks"]["supported_frameworks"],
	}


def gap_assessment_model(service: RegulatoryComplianceService,
						tenant_id: str = "default") -> dict[str, Any]:
	"""Gap assessment view."""
	assessments = service.list_gap_assessments(tenant_id)
	return {
		"title": "Compliance Gap Assessment",
		"tenant_id": tenant_id,
		"count": len(assessments),
		"total_critical_gaps": sum(a.critical_gaps for a in assessments),
		"items": [a.model_dump() for a in assessments],
	}


def inspection_management_model(service: RegulatoryComplianceService, tenant_id: str = "default",
								status: str | None = None) -> dict[str, Any]:
	"""Inspection management view."""
	inspections = service.list_inspections(tenant_id, status=status)
	return {
		"title": "Inspections",
		"tenant_id": tenant_id,
		"status_filter": status,
		"count": len(inspections),
		"warning_letters": sum(1 for i in inspections if i.outcome == "warning_letter"),
		"items": [i.model_dump() for i in inspections],
	}


def label_management_model(service: RegulatoryComplianceService, tenant_id: str = "default",
							product_id: str | None = None) -> dict[str, Any]:
	"""Label management view."""
	labels = service.list_labels(tenant_id, product_id=product_id)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Label Management",
		"tenant_id": tenant_id,
		"product_id": product_id,
		"count": len(labels),
		"items": [l.model_dump() for l in labels],
		"supported_change_types": contract["configuration"]["labeling"]["supported_change_types"],
	}


def pms_model(service: RegulatoryComplianceService, tenant_id: str = "default") -> dict[str, Any]:
	"""Post-market surveillance view."""
	pms_list = service.list_pms(tenant_id)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Post-Market Surveillance",
		"tenant_id": tenant_id,
		"count": len(pms_list),
		"active_count": sum(1 for p in pms_list if p.status == "active"),
		"items": [p.model_dump() for p in pms_list],
		"supported_types": contract["configuration"]["pms"]["supported_types"],
	}


def regulatory_intelligence_model(service: RegulatoryComplianceService, tenant_id: str = "default",
									region: str | None = None) -> dict[str, Any]:
	"""Regulatory intelligence view."""
	intel = service.list_intel(tenant_id, region=region)
	return {
		"title": "Regulatory Intelligence",
		"tenant_id": tenant_id,
		"region_filter": region,
		"count": len(intel),
		"unassessed_count": sum(1 for i in intel if not i.impact_assessed),
		"items": [i.model_dump() for i in intel],
	}


def commitment_tracker_model(service: RegulatoryComplianceService, tenant_id: str = "default",
							status: str | None = None) -> dict[str, Any]:
	"""Regulatory commitment tracker view."""
	commitments = service.list_commitments(tenant_id, status=status)
	overdue = service.check_overdue_commitments(tenant_id)
	return {
		"title": "Regulatory Commitments",
		"tenant_id": tenant_id,
		"status_filter": status,
		"count": len(commitments),
		"overdue_count": len(overdue),
		"items": [c.model_dump() for c in commitments],
	}
