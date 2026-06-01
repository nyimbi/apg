"""View models for generated Intelligence Reporting screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import IntelligenceReportingService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import IntelligenceReportingService  # type: ignore


def dashboard_model(service: IntelligenceReportingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Intelligence Reporting", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def reporting_console_model(service: IntelligenceReportingService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "authorities": _tenant_items(service.authorities, tenant_id), "workspaces": _tenant_items(service.workspaces, tenant_id), "templates": _tenant_items(service.templates, tenant_id), "products": _tenant_items(service.products, tenant_id), "sections": _tenant_items(service.sections, tenant_id), "citations": _tenant_items(service.citations, tenant_id), "approvals": _tenant_items(service.approvals, tenant_id), "distributions": _tenant_items(service.distributions, tenant_id), "publications": _tenant_items(service.publications, tenant_id), "reviews": _tenant_items(service.reviews, tenant_id)}


def agent_workbench_model(service: IntelligenceReportingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id]}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda value: value.id) if item.tenant_id == tenant_id]

