"""View models for generated Data Correlation screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import DataCorrelationService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import DataCorrelationService  # type: ignore


def dashboard_model(service: DataCorrelationService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Data Correlation", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def correlation_console_model(service: DataCorrelationService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "authorities": _tenant_items(service.authorities, tenant_id), "workspaces": _tenant_items(service.workspaces, tenant_id), "sources": _tenant_items(service.sources, tenant_id), "entities": _tenant_items(service.entities, tenant_id), "observations": _tenant_items(service.observations, tenant_id), "rules": _tenant_items(service.rules, tenant_id), "runs": _tenant_items(service.runs, tenant_id), "clusters": _tenant_items(service.clusters, tenant_id), "decisions": _tenant_items(service.decisions, tenant_id), "referrals": _tenant_items(service.referrals, tenant_id), "reviews": _tenant_items(service.reviews, tenant_id)}


def agent_workbench_model(service: DataCorrelationService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id]}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda value: value.id) if item.tenant_id == tenant_id]
