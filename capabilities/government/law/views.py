"""View models for generated Law Enforcement & Justice screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import LawEnforcementService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import LawEnforcementService  # type: ignore


def dashboard_model(service: LawEnforcementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Law Enforcement and Justice", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def law_enforcement_console_model(service: LawEnforcementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"incidents": _tenant_items(service.incidents, tenant_id),
		"dockets": _tenant_items(service.dockets, tenant_id),
		"evidence": _tenant_items(service.evidence, tenant_id),
		"custody_actions": _tenant_items(service.custody_actions, tenant_id),
		"court_hearings": _tenant_items(service.court_hearings, tenant_id),
		"prosecutions": _tenant_items(service.prosecutions, tenant_id),
	}


def agent_workbench_model(service: LawEnforcementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id],
	}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
