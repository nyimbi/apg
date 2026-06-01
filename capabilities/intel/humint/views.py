"""View models for generated Human Intelligence screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import HumanIntelligenceService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import HumanIntelligenceService  # type: ignore


def dashboard_model(service: HumanIntelligenceService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Human Intelligence", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def humint_console_model(service: HumanIntelligenceService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "authorities": _tenant_items(service.authorities, tenant_id), "sources": _tenant_items(service.sources, tenant_id), "contact_plans": _tenant_items(service.contact_plans, tenant_id), "contact_reports": _tenant_items(service.contact_reports, tenant_id), "debriefings": _tenant_items(service.debriefings, tenant_id), "reliability_assessments": _tenant_items(service.reliability_assessments, tenant_id), "leads": _tenant_items(service.leads, tenant_id), "disseminations": _tenant_items(service.disseminations, tenant_id), "reviews": _tenant_items(service.reviews, tenant_id)}


def agent_workbench_model(service: HumanIntelligenceService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id]}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda value: value.id) if item.tenant_id == tenant_id]
