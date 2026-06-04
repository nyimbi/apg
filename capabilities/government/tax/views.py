"""View models for generated Tax Administration screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import TaxAdministrationService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import TaxAdministrationService  # type: ignore


def dashboard_model(service: TaxAdministrationService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Tax Administration", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def tax_console_model(service: TaxAdministrationService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"registrations": _tenant_items(service.registrations, tenant_id),
		"returns": _tenant_items(service.returns, tenant_id),
		"assessments": _tenant_items(service.assessments, tenant_id),
		"objections": _tenant_items(service.objections, tenant_id),
		"debt_cases": _tenant_items(service.debt_cases, tenant_id),
		"audits": _tenant_items(service.audits, tenant_id),
	}


def agent_workbench_model(service: TaxAdministrationService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id],
	}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
