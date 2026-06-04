"""View models for APG Customer Management screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import TelecomCusService


def dashboard_model(service: TelecomCusService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Customer Management", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def customer_console_model(service: TelecomCusService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "customers": _items(service.customers, tenant_id)}


def customer_360_model(service: TelecomCusService, tenant_id: str, customer_id: str) -> dict[str, Any]:
	customer = service.customers.get((tenant_id, customer_id))
	if customer is None:
		return {"error": "customer_not_found"}
	kyc_docs = [d.to_dict() for d in service.kyc_documents.values() if d.tenant_id == tenant_id and d.customer_id == customer_id]
	plans = [p.to_dict() for p in service.plans.values() if p.tenant_id == tenant_id and p.customer_id == customer_id]
	sims = [s.to_dict() for s in service.sims.values() if s.tenant_id == tenant_id and s.customer_id == customer_id]
	devices = [d.to_dict() for d in service.devices.values() if d.tenant_id == tenant_id and d.customer_id == customer_id]
	cases = [c.to_dict() for c in service.cases.values() if c.tenant_id == tenant_id and c.customer_id == customer_id]
	events = [e.to_dict() for e in service.lifecycle_events.values() if e.tenant_id == tenant_id and e.customer_id == customer_id]
	return {"tenant_id": tenant_id, "customer": customer.to_dict(), "kyc_documents": kyc_docs, "plans": plans, "sims": sims, "devices": devices, "cases": cases, "lifecycle_events": events}


def kyc_console_model(service: TelecomCusService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "kyc_documents": _items(service.kyc_documents, tenant_id)}


def case_queue_model(service: TelecomCusService, tenant_id: str = "default") -> dict[str, Any]:
	open_cases = [c.to_dict() for c in service.cases.values() if c.tenant_id == tenant_id and c.status == "open"]
	return {"tenant_id": tenant_id, "open_cases": open_cases, "all_cases": _items(service.cases, tenant_id)}


def agent_workbench_model(service: TelecomCusService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _items(service.agents, tenant_id)}


def _items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(store.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
