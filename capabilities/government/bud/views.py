"""View models for generated Budget Management screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import BudgetManagementService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import BudgetManagementService  # type: ignore


def dashboard_model(service: BudgetManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Budget Management", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def budget_console_model(service: BudgetManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"budgets": _tenant_items(service.budgets, tenant_id),
		"votes": _tenant_items(service.votes, tenant_id),
		"revisions": _tenant_items(service.revisions, tenant_id),
		"commitments": _tenant_items(service.commitments, tenant_id),
		"expenditures": _tenant_items(service.expenditures, tenant_id),
		"reports": _tenant_items(service.reports, tenant_id),
	}


def agent_workbench_model(service: BudgetManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id],
	}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
