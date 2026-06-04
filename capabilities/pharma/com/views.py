"""View models for APG Pharma Commercial Operations screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import CommercialOperationsService


def dashboard_model(service: CommercialOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	"""Compose the dashboard view model."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Commercial Operations",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
		"streaming": contract["streaming"],
	}


def territory_list_model(service: CommercialOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	"""List view for territories."""
	territories = service.list_territories(tenant_id)
	return {
		"title": "Territories",
		"tenant_id": tenant_id,
		"count": len(territories),
		"items": [t.model_dump() for t in territories],
	}


def territory_detail_model(service: CommercialOperationsService, territory_id: str,
							tenant_id: str = "default") -> dict[str, Any]:
	"""Detail view for a single territory with its reps."""
	territory = service.get_territory(territory_id, tenant_id)
	reps = service.list_reps_by_territory(territory_id, tenant_id)
	targets = service.list_targets(tenant_id, territory_id=territory_id)
	return {
		"title": f"Territory: {territory.name}",
		"tenant_id": tenant_id,
		"territory": territory.model_dump(),
		"reps": [r.model_dump() for r in reps],
		"target_count": len(targets),
	}


def rep_list_model(service: CommercialOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	"""List view for sales reps."""
	reps = service.list_reps(tenant_id)
	return {
		"title": "Sales Representatives",
		"tenant_id": tenant_id,
		"count": len(reps),
		"items": [r.model_dump() for r in reps],
	}


def call_log_model(service: CommercialOperationsService, tenant_id: str = "default",
					rep_id: str | None = None) -> dict[str, Any]:
	"""List view for call activity log."""
	calls = service.list_calls(tenant_id, rep_id=rep_id)
	return {
		"title": "Call Activity Log",
		"tenant_id": tenant_id,
		"rep_id": rep_id,
		"count": len(calls),
		"items": [c.model_dump() for c in calls],
	}


def sample_console_model(service: CommercialOperationsService, tenant_id: str = "default",
						rep_id: str | None = None) -> dict[str, Any]:
	"""Sample management console view."""
	samples = service.list_samples(tenant_id, rep_id=rep_id)
	return {
		"title": "Sample Management",
		"tenant_id": tenant_id,
		"rep_id": rep_id,
		"count": len(samples),
		"items": [s.model_dump() for s in samples],
	}


def interaction_ledger_model(service: CommercialOperationsService, tenant_id: str = "default",
							hcp_id: str | None = None) -> dict[str, Any]:
	"""HCP interaction ledger view."""
	interactions = service.list_interactions(tenant_id, hcp_id=hcp_id)
	return {
		"title": "HCP Interactions",
		"tenant_id": tenant_id,
		"hcp_id": hcp_id,
		"count": len(interactions),
		"items": [i.model_dump() for i in interactions],
	}


def plan_workbench_model(service: CommercialOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	"""Commercial plan workbench view."""
	plans = service.list_plans(tenant_id)
	return {
		"title": "Commercial Plans",
		"tenant_id": tenant_id,
		"count": len(plans),
		"items": [p.model_dump() for p in plans],
	}


def target_segmentation_model(service: CommercialOperationsService, tenant_id: str = "default",
								territory_id: str | None = None) -> dict[str, Any]:
	"""Target physician segmentation view."""
	targets = service.list_targets(tenant_id, territory_id=territory_id)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Target Segmentation",
		"tenant_id": tenant_id,
		"territory_id": territory_id,
		"count": len(targets),
		"items": [t.model_dump() for t in targets],
		"supported_tiers": contract["configuration"]["targets"]["supported_tiers"],
	}


def aggregate_spend_model(service: CommercialOperationsService, tenant_id: str = "default",
						hcp_id: str | None = None, fiscal_year: str | None = None) -> dict[str, Any]:
	"""Aggregate spend tracker view."""
	contract = get_capability_contract(tenant_id)
	summary = None
	if hcp_id and fiscal_year:
		summary = service.get_aggregate_spend_summary(tenant_id, hcp_id, fiscal_year)
	return {
		"title": "Aggregate Spend Tracker",
		"tenant_id": tenant_id,
		"hcp_id": hcp_id,
		"fiscal_year": fiscal_year,
		"summary": summary,
		"supported_categories": contract["configuration"]["spend"]["supported_categories"],
		"aggregate_cap": contract["configuration"]["compliance"]["aggregate_spend_cap"],
	}


def _tenant_items(items: list[Any]) -> list[dict[str, Any]]:
	return [item.model_dump() if hasattr(item, "model_dump") else item for item in items]
