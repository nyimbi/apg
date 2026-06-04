"""View models for generated Project Accounting screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import ProjectAccountingService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import ProjectAccountingService  # type: ignore


def dashboard_model(service: ProjectAccountingService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the accounting dashboard screen."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Project Accounting",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def account_list_model(service: ProjectAccountingService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the project account list screen."""
	return {
		"tenant_id": tenant_id,
		"accounts": _tenant_items(service.accounts, tenant_id),
	}


def cost_ledger_model(service: ProjectAccountingService, tenant_id: str = "default", account_id: str | None = None) -> dict[str, Any]:
	"""View model for the cost transaction ledger."""
	return {
		"tenant_id": tenant_id,
		"account_id": account_id,
		"cost_transactions": [
			v.to_dict() for v in sorted(service.cost_transactions.values(), key=lambda x: x.id)
			if v.tenant_id == tenant_id and (account_id is None or v.account_id == account_id)
		],
	}


def revenue_console_model(service: ProjectAccountingService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the revenue recognition console."""
	return {
		"tenant_id": tenant_id,
		"revenue_recognitions": _tenant_items(service.revenue_recognitions, tenant_id),
		"wip_adjustments": _tenant_items(service.wip_adjustments, tenant_id),
	}


def billing_console_model(service: ProjectAccountingService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the milestone billing console."""
	return {
		"tenant_id": tenant_id,
		"invoices": _tenant_items(service.invoices, tenant_id),
		"budget_overrides": _tenant_items(service.budget_overrides, tenant_id),
	}


def approval_queue_model(service: ProjectAccountingService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the accounting approval queue."""
	return {
		"tenant_id": tenant_id,
		"approvals": _tenant_items(service.approvals, tenant_id),
	}


def agent_workbench_model(service: ProjectAccountingService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the accounting agent workbench."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [v.to_dict() for v in service.agents.values() if v.tenant_id == tenant_id],
	}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [v.to_dict() for v in sorted(items.values(), key=lambda x: x.id) if v.tenant_id == tenant_id]
