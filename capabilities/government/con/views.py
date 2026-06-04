"""View models for generated Government Contracts & Procurement screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import ProcurementService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import ProcurementService  # type: ignore


def dashboard_model(service: ProcurementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Government Contracts and Procurement", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def procurement_console_model(service: ProcurementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"tenders": _tenant_items(service.tenders, tenant_id),
		"evaluations": _tenant_items(service.evaluations, tenant_id),
		"awards": _tenant_items(service.awards, tenant_id),
		"contracts": _tenant_items(service.contracts, tenant_id),
		"variations": _tenant_items(service.variations, tenant_id),
		"debarred": _tenant_items(service.debarred, tenant_id),
	}


def agent_workbench_model(service: ProcurementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id],
	}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
