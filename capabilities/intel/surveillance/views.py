"""View models for generated Digital Surveillance screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import DigitalSurveillanceService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import DigitalSurveillanceService  # type: ignore


def dashboard_model(service: DigitalSurveillanceService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Digital Surveillance", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def surveillance_console_model(service: DigitalSurveillanceService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "authorities": _tenant_items(service.authorities, tenant_id), "programs": _tenant_items(service.programs, tenant_id), "assets": _tenant_items(service.assets, tenant_id), "sensors": _tenant_items(service.sensors, tenant_id), "observations": _tenant_items(service.observations, tenant_id), "alerts": _tenant_items(service.alerts, tenant_id), "risks": _tenant_items(service.risks, tenant_id), "referrals": _tenant_items(service.referrals, tenant_id), "disseminations": _tenant_items(service.disseminations, tenant_id), "reviews": _tenant_items(service.reviews, tenant_id)}


def agent_workbench_model(service: DigitalSurveillanceService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id]}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda value: value.id) if item.tenant_id == tenant_id]
