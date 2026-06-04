"""View models for generated Delivery Management screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import DeliveryManagementService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import DeliveryManagementService  # type: ignore


def dashboard_model(service: DeliveryManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Delivery Management", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def delivery_console_model(service: DeliveryManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "deliveries": _tenant_items(service.deliveries, tenant_id), "supported_types": contract["configuration"]["deliveries"]["supported_types"]}


def pod_console_model(service: DeliveryManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "pods": _tenant_items(service.pods, tenant_id)}


def failed_delivery_console_model(service: DeliveryManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "failed_deliveries": _tenant_items(service.failed_deliveries, tenant_id), "failure_reasons": contract["configuration"]["failed_deliveries"]["failure_reasons"]}


def sla_console_model(service: DeliveryManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "sla_records": _tenant_items(service.sla_records, tenant_id)}


def return_console_model(service: DeliveryManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "returns": _tenant_items(service.returns, tenant_id)}


def agent_workbench_model(service: DeliveryManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _tenant_items(service.agents, tenant_id)}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
