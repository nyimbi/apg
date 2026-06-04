"""View models for APG Pharma Quality Management System screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import QualityManagementService


def dashboard_model(service: QualityManagementService, tenant_id: str = "default") -> dict[str, Any]:
	"""Compose the QMS dashboard view model."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Quality Management System",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
		"overdue_capas": len(service.check_overdue_capas(tenant_id)),
		"documents_due_review": len(service.check_periodic_review(tenant_id)),
	}


def change_control_queue_model(service: QualityManagementService, tenant_id: str = "default",
								status: str | None = None) -> dict[str, Any]:
	"""Change control queue view."""
	changes = service.list_changes(tenant_id, status=status)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Change Control",
		"tenant_id": tenant_id,
		"status_filter": status,
		"count": len(changes),
		"items": [c.model_dump() for c in changes],
		"supported_types": contract["configuration"]["change_control"]["supported_types"],
	}


def change_detail_model(service: QualityManagementService, change_id: str,
						tenant_id: str = "default") -> dict[str, Any]:
	"""Change control detail view."""
	changes = service.list_changes(tenant_id)
	change = next((c for c in changes if c.id == change_id), None)
	if change is None:
		return {"error": f"Change {change_id} not found"}
	return {
		"title": f"Change: {change.change_number}",
		"tenant_id": tenant_id,
		"change": change.model_dump(),
	}


def capa_management_model(service: QualityManagementService, tenant_id: str = "default",
						status: str | None = None) -> dict[str, Any]:
	"""CAPA management view."""
	capas = service.list_capas(tenant_id, status=status)
	overdue = service.check_overdue_capas(tenant_id)
	return {
		"title": "CAPA Management",
		"tenant_id": tenant_id,
		"status_filter": status,
		"count": len(capas),
		"overdue_count": len(overdue),
		"items": [c.model_dump() for c in capas],
	}


def deviation_queue_model(service: QualityManagementService, tenant_id: str = "default",
						status: str | None = None) -> dict[str, Any]:
	"""Deviation queue view."""
	deviations = service.list_deviations(tenant_id, status=status)
	return {
		"title": "Deviation Management",
		"tenant_id": tenant_id,
		"status_filter": status,
		"count": len(deviations),
		"open_count": sum(1 for d in deviations if d.status == "open"),
		"gmp_impact_count": sum(1 for d in deviations if d.gmp_impact),
		"items": [d.model_dump() for d in deviations],
	}


def document_controller_model(service: QualityManagementService, tenant_id: str = "default",
								document_type: str | None = None,
								status: str | None = None) -> dict[str, Any]:
	"""Document control view."""
	documents = service.list_documents(tenant_id, document_type=document_type, status=status)
	due_review = service.check_periodic_review(tenant_id)
	return {
		"title": "Document Control",
		"tenant_id": tenant_id,
		"type_filter": document_type,
		"status_filter": status,
		"count": len(documents),
		"due_review_count": len(due_review),
		"items": [d.model_dump() for d in documents],
	}


def audit_management_model(service: QualityManagementService, tenant_id: str = "default",
							audit_type: str | None = None) -> dict[str, Any]:
	"""Audit management view."""
	audits = service.list_audits(tenant_id, audit_type=audit_type)
	return {
		"title": "Audit Management",
		"tenant_id": tenant_id,
		"type_filter": audit_type,
		"count": len(audits),
		"open_count": sum(1 for a in audits if a.status != "closed"),
		"items": [a.model_dump() for a in audits],
	}


def validation_registry_model(service: QualityManagementService,
								tenant_id: str = "default") -> dict[str, Any]:
	"""Validation registry view."""
	validations = service.list_validations(tenant_id)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Validation Registry",
		"tenant_id": tenant_id,
		"count": len(validations),
		"items": [v.model_dump() for v in validations],
		"supported_types": contract["configuration"]["validation"]["supported_types"],
	}


def risk_register_model(service: QualityManagementService, tenant_id: str = "default",
						risk_level: str | None = None) -> dict[str, Any]:
	"""Risk register view."""
	risks = service.list_risks(tenant_id, risk_level=risk_level)
	return {
		"title": "Risk Register",
		"tenant_id": tenant_id,
		"risk_level_filter": risk_level,
		"count": len(risks),
		"high_critical_count": sum(1 for r in risks if r.risk_level in ("high", "critical")),
		"items": [r.model_dump() for r in risks],
	}
