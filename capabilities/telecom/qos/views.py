"""View models for APG Quality of Service screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import TelecomQosService


def dashboard_model(service: TelecomQosService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Quality of Service", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def policy_console_model(service: TelecomQosService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "policies": _items(service.policies, tenant_id), "enforcement_records": _items(service.enforcement_records, tenant_id)}


def traffic_console_model(service: TelecomQosService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "traffic_classifications": _items(service.traffic_classifications, tenant_id)}


def sla_console_model(service: TelecomQosService, tenant_id: str = "default") -> dict[str, Any]:
	breaches = [m.to_dict() for m in service.sla_measurements.values() if m.tenant_id == tenant_id and m.is_breach]
	return {"tenant_id": tenant_id, "sla_breaches": breaches, "all_measurements": _items(service.sla_measurements, tenant_id)}


def degradation_console_model(service: TelecomQosService, tenant_id: str = "default") -> dict[str, Any]:
	open_degradations = [d.to_dict() for d in service.degradations.values() if d.tenant_id == tenant_id and d.status == "open"]
	return {"tenant_id": tenant_id, "open_degradations": open_degradations, "root_causes": _items(service.root_causes, tenant_id), "remediations": _items(service.remediations, tenant_id)}


def agent_workbench_model(service: TelecomQosService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _items(service.agents, tenant_id)}


def _items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(store.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
