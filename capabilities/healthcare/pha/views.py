"""View model builders for APG Pharmacy Management screens."""

from __future__ import annotations

import asyncio
from typing import Any

from .capability_contract import get_capability_contract
from .service import PharmacyManagementService


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def dashboard_view_model(service: PharmacyManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Pharmacy Management", "tenant_id": tenant_id, "summary": _run(service.dashboard_summary(tenant_id)), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def formulary_view_model(service: PharmacyManagementService, tenant_id: str, formulary_status: str | None = None) -> dict[str, Any]:
	drugs = _run(service.list_drugs(tenant_id, formulary_status=formulary_status))
	lasa_drugs = [d for d in drugs if d.is_lasa]
	return {"title": "Drug Formulary", "tenant_id": tenant_id, "drugs": [d.model_dump() for d in drugs], "lasa_count": len(lasa_drugs), "filter": {"formulary_status": formulary_status}}


def dispense_queue_view_model(service: PharmacyManagementService, tenant_id: str, status: str | None = None) -> dict[str, Any]:
	orders = _run(service.list_dispense_orders(tenant_id, status=status))
	pending = [o for o in orders if o.status == "pending"]
	return {"title": "Dispense Queue", "tenant_id": tenant_id, "orders": [o.model_dump() for o in orders], "pending_count": len(pending), "filter": {"status": status}}


def interaction_view_model(service: PharmacyManagementService, tenant_id: str, severity: str | None = None) -> dict[str, Any]:
	interactions = _run(service.list_interactions(tenant_id, severity=severity))
	contraindicated = [i for i in interactions if i.severity == "contraindicated"]
	return {"title": "Drug Interactions", "tenant_id": tenant_id, "interactions": [i.model_dump() for i in interactions], "contraindicated_count": len(contraindicated), "filter": {"severity": severity}}


def controlled_substance_view_model(service: PharmacyManagementService, tenant_id: str, action: str | None = None) -> dict[str, Any]:
	logs = _run(service.list_controlled_logs(tenant_id, action=action))
	waste_events = [l for l in logs if l.action == "waste"]
	return {"title": "Controlled Substances", "tenant_id": tenant_id, "logs": [l.model_dump() for l in logs], "waste_count": len(waste_events), "filter": {"action": action}}


def inventory_view_model(service: PharmacyManagementService, tenant_id: str) -> dict[str, Any]:
	all_items = _run(service.list_inventory(tenant_id))
	low = [i for i in all_items if i.status == "low_stock"]
	recalled = [i for i in all_items if i.status == "recalled"]
	expired = [i for i in all_items if i.status == "expired"]
	return {"title": "Pharmacy Inventory", "tenant_id": tenant_id, "items": [i.model_dump() for i in all_items], "low_stock_count": len(low), "recalled_count": len(recalled), "expired_count": len(expired)}


def prior_auth_view_model(service: PharmacyManagementService, tenant_id: str, status: str | None = None) -> dict[str, Any]:
	pas = _run(service.list_prior_auths(tenant_id, status=status))
	pending = [p for p in pas if p.status == "pending"]
	return {"title": "Prior Authorizations", "tenant_id": tenant_id, "prior_auths": [p.model_dump() for p in pas], "pending_count": len(pending), "filter": {"status": status}}
