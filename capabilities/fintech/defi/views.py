"""View models for generated Decentralized Finance screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import DecentralizedFinanceService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import DecentralizedFinanceService  # type: ignore


def dashboard_model(service: DecentralizedFinanceService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Decentralized Finance", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def defi_console_model(service: DecentralizedFinanceService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "protocols": _tenant_items(service.protocols, tenant_id), "positions": _tenant_items(service.positions, tenant_id), "actions": _tenant_items(service.actions, tenant_id), "yield_strategies": _tenant_items(service.strategies, tenant_id), "rewards": _tenant_items(service.rewards, tenant_id), "governance": _tenant_items(service.governance, tenant_id), "risk_assessments": _tenant_items(service.risk_assessments, tenant_id), "reviews": _tenant_items(service.reviews, tenant_id)}


def agent_workbench_model(service: DecentralizedFinanceService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id]}


def _tenant_items(items: dict[str, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda value: value.id) if item.tenant_id == tenant_id]
