"""View models for APG Telecom Security screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import TelecomSecService


def dashboard_model(service: TelecomSecService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Telecom Security", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def fraud_console_model(service: TelecomSecService, tenant_id: str = "default") -> dict[str, Any]:
	open_cases = [c.to_dict() for c in service.fraud_cases.values() if c.tenant_id == tenant_id and c.status == "open"]
	return {"tenant_id": tenant_id, "open_fraud_cases": open_cases, "all_fraud_cases": _items(service.fraud_cases, tenant_id)}


def signalling_security_model(service: TelecomSecService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "ss7_attacks": _items(service.ss7_attacks, tenant_id), "diameter_attacks": _items(service.diameter_attacks, tenant_id)}


def intercept_console_model(service: TelecomSecService, tenant_id: str = "default") -> dict[str, Any]:
	active_intercepts = [i.to_dict() for i in service.intercepts.values() if i.tenant_id == tenant_id and i.status == "active"]
	return {"tenant_id": tenant_id, "active_intercepts": active_intercepts, "all_intercepts": _items(service.intercepts, tenant_id)}


def incident_queue_model(service: TelecomSecService, tenant_id: str = "default") -> dict[str, Any]:
	open_incidents = [i.to_dict() for i in service.incidents.values() if i.tenant_id == tenant_id and i.status in ("new", "under_investigation")]
	return {"tenant_id": tenant_id, "open_incidents": open_incidents, "all_incidents": _items(service.incidents, tenant_id)}


def threat_intel_console_model(service: TelecomSecService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "threat_intel": _items(service.threat_intel, tenant_id)}


def agent_workbench_model(service: TelecomSecService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _items(service.agents, tenant_id)}


def _items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(store.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
