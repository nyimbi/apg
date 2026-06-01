"""View models for generated Regulatory Technology screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import RegTechService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import RegTechService  # type: ignore


def dashboard_model(service: RegTechService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Regulatory Technology", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def regtech_console_model(service: RegTechService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "sources": _tenant_items(service.sources, tenant_id), "changes": _tenant_items(service.changes, tenant_id), "obligations": _tenant_items(service.obligations, tenant_id), "impacts": _tenant_items(service.impacts, tenant_id), "filings": _tenant_items(service.filings, tenant_id), "submissions": _tenant_items(service.submissions, tenant_id), "inquiries": _tenant_items(service.inquiries, tenant_id), "responses": _tenant_items(service.responses, tenant_id), "reviews": _tenant_items(service.reviews, tenant_id)}


def agent_workbench_model(service: RegTechService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id]}


def _tenant_items(items: dict[str, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda value: value.id) if item.tenant_id == tenant_id]
