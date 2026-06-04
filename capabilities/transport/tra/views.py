"""View models for generated Asset Tracking screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import AssetTrackingService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import AssetTrackingService  # type: ignore


def dashboard_model(service: AssetTrackingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Asset Tracking", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def live_map_model(service: AssetTrackingService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "assets": _tenant_items(service.assets, tenant_id), "recent_locations": _tenant_items(service.location_updates, tenant_id)}


def asset_console_model(service: AssetTrackingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "assets": _tenant_items(service.assets, tenant_id), "asset_types": contract["configuration"]["assets"]["supported_types"]}


def geofence_console_model(service: AssetTrackingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "geofences": _tenant_items(service.geofences, tenant_id), "geofence_types": contract["configuration"]["geofencing"]["types"]}


def alert_console_model(service: AssetTrackingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "alerts": _tenant_items(service.alerts, tenant_id), "active_alerts": service.list_active_alerts(tenant_id), "alert_types": contract["configuration"]["alerts"]["types"]}


def cold_chain_console_model(service: AssetTrackingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "cold_chain_records": _tenant_items(service.cold_chain_records, tenant_id), "standards": contract["configuration"]["cold_chain"]["standards"]}


def container_console_model(service: AssetTrackingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "containers": _tenant_items(service.containers, tenant_id), "statuses": contract["configuration"]["containers"]["statuses"]}


def agent_workbench_model(service: AssetTrackingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _tenant_items(service.agents, tenant_id)}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
