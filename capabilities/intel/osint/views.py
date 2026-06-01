"""View models for generated Open Source Intelligence screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import OpenSourceIntelligenceService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import OpenSourceIntelligenceService  # type: ignore


def dashboard_model(service: OpenSourceIntelligenceService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Open Source Intelligence", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def osint_console_model(service: OpenSourceIntelligenceService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "requirements": _tenant_items(service.requirements, tenant_id), "sources": _tenant_items(service.sources, tenant_id), "collection_plans": _tenant_items(service.plans, tenant_id), "evidence": _tenant_items(service.evidence, tenant_id), "triage": _tenant_items(service.triage, tenant_id), "assessments": _tenant_items(service.assessments, tenant_id), "dissemination": _tenant_items(service.dissemination, tenant_id), "reviews": _tenant_items(service.reviews, tenant_id)}


def agent_workbench_model(service: OpenSourceIntelligenceService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id]}


def _tenant_items(items: dict[str, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda value: value.id) if item.tenant_id == tenant_id]
