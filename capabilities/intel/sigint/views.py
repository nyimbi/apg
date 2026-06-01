"""View models for generated Signals Intelligence screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import SignalsIntelligenceService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import SignalsIntelligenceService  # type: ignore


def dashboard_model(service: SignalsIntelligenceService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Signals Intelligence", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def sigint_console_model(service: SignalsIntelligenceService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "authorities": _tenant_items(service.authorities, tenant_id), "sources": _tenant_items(service.sources, tenant_id), "collection_tasks": _tenant_items(service.tasks, tenant_id), "observations": _tenant_items(service.observations, tenant_id), "processing_batches": _tenant_items(service.processing_batches, tenant_id), "patterns": _tenant_items(service.patterns, tenant_id), "assessments": _tenant_items(service.assessments, tenant_id), "reviews": _tenant_items(service.reviews, tenant_id)}


def agent_workbench_model(service: SignalsIntelligenceService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id]}


def _tenant_items(items: dict[str, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda value: value.id) if item.tenant_id == tenant_id]
