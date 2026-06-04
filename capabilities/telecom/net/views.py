"""View models for APG Network Management screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import TelecomNetService


def dashboard_model(service: TelecomNetService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Network Management", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def alarm_console_model(service: TelecomNetService, tenant_id: str = "default") -> dict[str, Any]:
	active = [a.to_dict() for a in service.alarms.values() if a.tenant_id == tenant_id and a.status == "raised"]
	return {"tenant_id": tenant_id, "active_alarms": active, "all_alarms": _items(service.alarms, tenant_id)}


def fault_ticket_queue_model(service: TelecomNetService, tenant_id: str = "default") -> dict[str, Any]:
	open_tickets = [t.to_dict() for t in service.fault_tickets.values() if t.tenant_id == tenant_id and t.status == "open"]
	return {"tenant_id": tenant_id, "open_tickets": open_tickets, "all_tickets": _items(service.fault_tickets, tenant_id)}


def performance_console_model(service: TelecomNetService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "performance_records": _items(service.performance_records, tenant_id)}


def change_console_model(service: TelecomNetService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "config_changes": _items(service.config_changes, tenant_id)}


def sla_console_model(service: TelecomNetService, tenant_id: str = "default") -> dict[str, Any]:
	breached = [s.to_dict() for s in service.sla_records.values() if s.tenant_id == tenant_id and s.status == "breached"]
	return {"tenant_id": tenant_id, "breached_slas": breached, "all_sla_records": _items(service.sla_records, tenant_id)}


def noc_view_model(service: TelecomNetService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "noc_handovers": _items(service.noc_handovers, tenant_id), "active_alarms_count": sum(1 for a in service.alarms.values() if a.tenant_id == tenant_id and a.status == "raised"), "open_tickets_count": sum(1 for t in service.fault_tickets.values() if t.tenant_id == tenant_id and t.status == "open")}


def agent_workbench_model(service: TelecomNetService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _items(service.agents, tenant_id)}


def _items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(store.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
