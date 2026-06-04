"""View models for APG Pharma Manufacturing screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import PharmaceuticalManufacturingService


def dashboard_model(service: PharmaceuticalManufacturingService, tenant_id: str = "default") -> dict[str, Any]:
	"""Compose the manufacturing dashboard view model."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Pharmaceutical Manufacturing",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def batch_registry_model(service: PharmaceuticalManufacturingService, tenant_id: str = "default",
						status: str | None = None) -> dict[str, Any]:
	"""Batch registry list view."""
	batches = service.list_batches(tenant_id, status=status)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Batch Registry",
		"tenant_id": tenant_id,
		"status_filter": status,
		"count": len(batches),
		"items": [b.model_dump() for b in batches],
		"supported_statuses": contract["configuration"]["batches"]["supported_statuses"],
	}


def batch_detail_model(service: PharmaceuticalManufacturingService, batch_id: str,
					tenant_id: str = "default") -> dict[str, Any]:
	"""Batch detail view with yields and deviations."""
	batch = service.get_batch(batch_id, tenant_id)
	yields = service.list_yields(tenant_id, batch_id=batch_id)
	deviations = service.list_deviations(tenant_id, batch_id=batch_id)
	return {
		"title": f"Batch: {batch.batch_number}",
		"tenant_id": tenant_id,
		"batch": batch.model_dump(),
		"yield_steps": [y.model_dump() for y in yields],
		"deviations": [d.model_dump() for d in deviations],
		"deviation_count": len(deviations),
	}


def production_lines_model(service: PharmaceuticalManufacturingService,
							tenant_id: str = "default") -> dict[str, Any]:
	"""Production lines view."""
	lines = service.list_lines(tenant_id)
	return {
		"title": "Production Lines",
		"tenant_id": tenant_id,
		"count": len(lines),
		"available": sum(1 for l in lines if l.status == "available"),
		"running": sum(1 for l in lines if l.status == "running"),
		"items": [l.model_dump() for l in lines],
	}


def equipment_registry_model(service: PharmaceuticalManufacturingService,
							tenant_id: str = "default", status: str | None = None) -> dict[str, Any]:
	"""Equipment registry view."""
	equipment = service.list_equipment(tenant_id, status=status)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Equipment Registry",
		"tenant_id": tenant_id,
		"status_filter": status,
		"count": len(equipment),
		"qualified_count": sum(1 for e in equipment if e.status == "qualified"),
		"items": [e.model_dump() for e in equipment],
		"supported_qualification_types": contract["configuration"]["equipment"]["supported_qualification_types"],
	}


def deviation_queue_model(service: PharmaceuticalManufacturingService,
						tenant_id: str = "default") -> dict[str, Any]:
	"""Manufacturing deviation queue view."""
	deviations = service.list_deviations(tenant_id)
	return {
		"title": "Deviation Queue",
		"tenant_id": tenant_id,
		"count": len(deviations),
		"open_count": sum(1 for d in deviations if d.status == "open"),
		"critical_count": sum(1 for d in deviations if d.severity == "critical"),
		"items": [d.model_dump() for d in deviations],
	}


def yield_dashboard_model(service: PharmaceuticalManufacturingService,
						tenant_id: str = "default") -> dict[str, Any]:
	"""Yield analysis dashboard view."""
	yields = service.list_yields(tenant_id)
	investigation_required = [y for y in yields if y.investigation_required]
	return {
		"title": "Yield Dashboard",
		"tenant_id": tenant_id,
		"total_yield_records": len(yields),
		"investigation_required_count": len(investigation_required),
		"items": [y.model_dump() for y in yields],
	}


def material_management_model(service: PharmaceuticalManufacturingService,
							tenant_id: str = "default", status: str | None = None) -> dict[str, Any]:
	"""Raw material management view."""
	materials = service.list_materials(tenant_id, status=status)
	return {
		"title": "Material Management",
		"tenant_id": tenant_id,
		"status_filter": status,
		"count": len(materials),
		"quarantine_count": sum(1 for m in materials if m.status == "quarantine"),
		"items": [m.model_dump() for m in materials],
	}
