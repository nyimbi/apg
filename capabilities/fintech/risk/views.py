"""View models for generated FinTech Risk Management screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import RiskManagementService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import RiskManagementService  # type: ignore


def dashboard_model(service: RiskManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "FinTech Risk Management", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def risk_console_model(service: RiskManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "appetites": _tenant_items(service.appetites, tenant_id), "profiles": _tenant_items(service.profiles, tenant_id), "exposures": _tenant_items(service.exposures, tenant_id), "controls": _tenant_items(service.controls, tenant_id), "scenarios": _tenant_items(service.scenarios, tenant_id), "breaches": _tenant_items(service.breaches, tenant_id), "events": _tenant_items(service.events, tenant_id), "reviews": _tenant_items(service.reviews, tenant_id)}


def agent_workbench_model(service: RiskManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": [item.to_dict() for item in service.evidence.values() if item.tenant_id == tenant_id and item.kind == "agent"]}


def _tenant_items(items: dict[str, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda value: value.id) if item.tenant_id == tenant_id]
