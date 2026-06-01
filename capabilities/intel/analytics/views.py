"""View models for generated Intelligence Analytics screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import IntelligenceAnalyticsService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import IntelligenceAnalyticsService  # type: ignore


def dashboard_model(service: IntelligenceAnalyticsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Intelligence Analytics", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def analytics_console_model(service: IntelligenceAnalyticsService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "authorities": _tenant_items(service.authorities, tenant_id), "workspaces": _tenant_items(service.workspaces, tenant_id), "datasets": _tenant_items(service.datasets, tenant_id), "feature_sets": _tenant_items(service.feature_sets, tenant_id), "models": _tenant_items(service.models, tenant_id), "runs": _tenant_items(service.runs, tenant_id), "insights": _tenant_items(service.insights, tenant_id), "dashboards": _tenant_items(service.dashboards, tenant_id), "narratives": _tenant_items(service.narratives, tenant_id), "recommendations": _tenant_items(service.recommendations, tenant_id), "reviews": _tenant_items(service.reviews, tenant_id)}


def agent_workbench_model(service: IntelligenceAnalyticsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id]}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda value: value.id) if item.tenant_id == tenant_id]
