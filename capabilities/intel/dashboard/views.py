"""View models for generated Intelligence Dashboard screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import IntelligenceDashboardService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import IntelligenceDashboardService  # type: ignore


def dashboard_model(service: IntelligenceDashboardService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Intelligence Dashboard", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def dashboard_console_model(service: IntelligenceDashboardService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "authorities": _tenant_items(service.authorities, tenant_id), "workspaces": _tenant_items(service.workspaces, tenant_id), "dashboards": _tenant_items(service.dashboards, tenant_id), "sources": _tenant_items(service.sources, tenant_id), "metrics": _tenant_items(service.metrics, tenant_id), "widgets": _tenant_items(service.widgets, tenant_id), "filters": _tenant_items(service.filters, tenant_id), "views": _tenant_items(service.views, tenant_id), "shares": _tenant_items(service.shares, tenant_id), "reviews": _tenant_items(service.reviews, tenant_id)}


def agent_workbench_model(service: IntelligenceDashboardService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id]}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda value: value.id) if item.tenant_id == tenant_id]

