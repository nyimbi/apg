"""View models for APG Pharma Distribution screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import PharmaceuticalDistributionService


def dashboard_model(service: PharmaceuticalDistributionService, tenant_id: str = "default") -> dict[str, Any]:
	"""Compose the distribution dashboard view model."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Pharmaceutical Distribution",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
		"wda_expiry_alerts": service.check_wda_expiry(tenant_id),
	}


def shipment_tracker_model(service: PharmaceuticalDistributionService, tenant_id: str = "default",
							status: str | None = None) -> dict[str, Any]:
	"""Shipment tracker list view."""
	shipments = service.list_shipments(tenant_id, status=status)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Shipment Tracker",
		"tenant_id": tenant_id,
		"status_filter": status,
		"count": len(shipments),
		"items": [s.model_dump() for s in shipments],
		"supported_statuses": contract["configuration"]["shipments"]["supported_modes"],
	}


def shipment_detail_model(service: PharmaceuticalDistributionService, shipment_id: str,
						tenant_id: str = "default") -> dict[str, Any]:
	"""Detail view for a shipment with cold chain and excursions."""
	shipment = service.get_shipment(shipment_id, tenant_id)
	excursions = service.list_excursions(tenant_id, shipment_id=shipment_id)
	return {
		"title": f"Shipment: {shipment.shipment_number}",
		"tenant_id": tenant_id,
		"shipment": shipment.model_dump(),
		"excursion_count": len(excursions),
		"excursions": [e.model_dump() for e in excursions],
	}


def cold_chain_monitor_model(service: PharmaceuticalDistributionService,
							tenant_id: str = "default") -> dict[str, Any]:
	"""Cold chain monitoring console view."""
	contract = get_capability_contract(tenant_id)
	excursions = service.list_excursions(tenant_id)
	return {
		"title": "Cold Chain Monitor",
		"tenant_id": tenant_id,
		"excursion_count": len(excursions),
		"critical_excursions": [e.model_dump() for e in excursions if e.severity == "critical"],
		"all_excursions": [e.model_dump() for e in excursions],
		"supported_classifications": contract["configuration"]["cold_chain"]["supported_classifications"],
	}


def serialisation_console_model(service: PharmaceuticalDistributionService,
								tenant_id: str = "default") -> dict[str, Any]:
	"""Serialisation console view."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Serialisation Console",
		"tenant_id": tenant_id,
		"supported_standards": contract["configuration"]["serialisation"]["supported_standards"],
		"theme": contract["theme"],
	}


def recall_management_model(service: PharmaceuticalDistributionService,
							tenant_id: str = "default", status: str | None = None) -> dict[str, Any]:
	"""Recall management view."""
	recalls = service.list_recalls(tenant_id, status=status)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Recall Management",
		"tenant_id": tenant_id,
		"status_filter": status,
		"count": len(recalls),
		"active_count": sum(1 for r in recalls if r.status in ("initiated", "in_progress")),
		"items": [r.model_dump() for r in recalls],
		"supported_classes": contract["configuration"]["recalls"]["supported_classes"],
		"reporting_timelines": contract["configuration"]["recalls"]["timeline_hours"],
	}


def gdp_compliance_model(service: PharmaceuticalDistributionService,
						tenant_id: str = "default") -> dict[str, Any]:
	"""GDP compliance console view."""
	deviations = service.list_gdp_deviations(tenant_id)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "GDP Compliance",
		"tenant_id": tenant_id,
		"deviation_count": len(deviations),
		"open_deviations": [d.model_dump() for d in deviations if d.closed_date is None],
		"gdp_frameworks": contract["configuration"]["gdp"]["supported_statuses"],
	}


def wda_registry_model(service: PharmaceuticalDistributionService,
					tenant_id: str = "default") -> dict[str, Any]:
	"""WDA registry view."""
	wdas = service.list_wda(tenant_id)
	expiry_alerts = service.check_wda_expiry(tenant_id)
	return {
		"title": "Wholesale Distribution Authorisations",
		"tenant_id": tenant_id,
		"count": len(wdas),
		"expiry_alerts": expiry_alerts,
		"items": [w.model_dump() for w in wdas],
	}
