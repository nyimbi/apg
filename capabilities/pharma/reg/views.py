"""View models for APG Pharma Product Registration screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import ProductRegistrationService


def dashboard_model(service: ProductRegistrationService, tenant_id: str = "default") -> dict[str, Any]:
	"""Compose the registration dashboard view model."""
	contract = get_capability_contract(tenant_id)
	renewal_alerts = service.check_renewal_alerts(tenant_id)
	return {
		"title": "Product Registration",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
		"renewal_alerts": renewal_alerts,
	}


def registration_registry_model(service: ProductRegistrationService, tenant_id: str = "default",
								region: str | None = None, status: str | None = None) -> dict[str, Any]:
	"""Registration registry list view."""
	registrations = service.list_registrations(tenant_id, region=region, status=status)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Registration Registry",
		"tenant_id": tenant_id,
		"region_filter": region,
		"status_filter": status,
		"count": len(registrations),
		"approved_count": sum(1 for r in registrations if r.status == "approved"),
		"items": [r.model_dump() for r in registrations],
		"supported_regions": contract["configuration"]["approvals"]["supported_regions"],
	}


def registration_detail_model(service: ProductRegistrationService, reg_id: str,
								tenant_id: str = "default") -> dict[str, Any]:
	"""Registration detail with dossiers, interactions, and variations."""
	reg = service.get_registration(reg_id, tenant_id)
	dossiers = service.list_dossiers(tenant_id, product_id=reg.product_id)
	interactions = service.list_interactions(tenant_id, registration_id=reg_id)
	variations = service.list_variations(tenant_id, registration_id=reg_id)
	certificates = service.list_certificates(tenant_id, product_id=reg.product_id)
	procedures = service.list_procedures(tenant_id, registration_id=reg_id)
	return {
		"title": f"Registration: {reg.product_name} ({reg.region})",
		"tenant_id": tenant_id,
		"registration": reg.model_dump(),
		"dossiers": [d.model_dump() for d in dossiers],
		"interactions": [i.model_dump() for i in interactions],
		"variations": [v.model_dump() for v in variations],
		"certificates": [c.model_dump() for c in certificates],
		"procedures": [p.model_dump() for p in procedures],
	}


def dossier_workbench_model(service: ProductRegistrationService, tenant_id: str = "default",
							product_id: str | None = None) -> dict[str, Any]:
	"""Dossier workbench view."""
	dossiers = service.list_dossiers(tenant_id, product_id=product_id)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Dossier Workbench",
		"tenant_id": tenant_id,
		"product_id": product_id,
		"count": len(dossiers),
		"items": [d.model_dump() for d in dossiers],
		"supported_formats": contract["configuration"]["dossiers"]["supported_formats"],
		"supported_modules": contract["configuration"]["dossiers"]["supported_modules"],
	}


def approval_tracker_model(service: ProductRegistrationService, tenant_id: str = "default") -> dict[str, Any]:
	"""Approval tracking view with renewal alerts."""
	registrations = service.list_registrations(tenant_id)
	renewal_alerts = service.check_renewal_alerts(tenant_id)
	return {
		"title": "Approval Tracker",
		"tenant_id": tenant_id,
		"total_approved": sum(1 for r in registrations if r.status == "approved"),
		"pending": sum(1 for r in registrations if r.status in ("submitted", "under_review")),
		"renewal_alerts": renewal_alerts,
		"items": [r.model_dump() for r in registrations],
	}


def authority_interactions_model(service: ProductRegistrationService, tenant_id: str = "default",
									registration_id: str | None = None) -> dict[str, Any]:
	"""Authority interactions view."""
	interactions = service.list_interactions(tenant_id, registration_id=registration_id)
	return {
		"title": "Authority Interactions",
		"tenant_id": tenant_id,
		"registration_id": registration_id,
		"count": len(interactions),
		"items": [i.model_dump() for i in interactions],
	}


def renewal_queue_model(service: ProductRegistrationService, tenant_id: str = "default") -> dict[str, Any]:
	"""Renewal queue view."""
	renewal_alerts = service.check_renewal_alerts(tenant_id)
	return {
		"title": "Renewal Queue",
		"tenant_id": tenant_id,
		"count": len(renewal_alerts),
		"alerts": renewal_alerts,
	}


def variation_queue_model(service: ProductRegistrationService, tenant_id: str = "default",
						registration_id: str | None = None) -> dict[str, Any]:
	"""Variation queue view."""
	variations = service.list_variations(tenant_id, registration_id=registration_id)
	return {
		"title": "Variation Queue",
		"tenant_id": tenant_id,
		"registration_id": registration_id,
		"count": len(variations),
		"items": [v.model_dump() for v in variations],
	}
