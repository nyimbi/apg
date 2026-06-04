"""View models for APG Telecom Billing screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import TelecomBilService


def dashboard_model(service: TelecomBilService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Telecom Billing", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def mediation_console_model(service: TelecomBilService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "cdrs": _items(service.cdrs, tenant_id), "charges": _items(service.charges, tenant_id)}


def invoice_console_model(service: TelecomBilService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "cycles": _items(service.cycles, tenant_id), "invoices": _items(service.invoices, tenant_id), "dunning_steps": _items(service.dunning_steps, tenant_id)}


def payment_ledger_model(service: TelecomBilService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "payments": _items(service.payments, tenant_id), "discounts": _items(service.discounts, tenant_id)}


def convergent_console_model(service: TelecomBilService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "convergent_accounts": _items(service.convergent_accounts, tenant_id), "supported_modes": contract["configuration"]["convergent"]["supported_modes"]}


def agent_workbench_model(service: TelecomBilService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _items(service.agents, tenant_id)}


def _items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(store.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
