"""View models for APG enterprise asset management screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_EAM_AGENT_ROLES, SUPPORTED_EAM_AGENT_RUNTIMES, get_capability_contract
	from .service import EnterpriseAssetManagementService
except ImportError:
	from capability_contract import SUPPORTED_EAM_AGENT_ROLES, SUPPORTED_EAM_AGENT_RUNTIMES, get_capability_contract
	from service import EnterpriseAssetManagementService


def navigation_model(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"capability": contract["capability"],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
		"api_prefix": contract["ui"]["api_prefix"],
	}


def dashboard_model(service: EnterpriseAssetManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "dashboard",
		"title": "Enterprise Asset Management",
		"summary": service.dashboard_summary(tenant_id),
		"sections": ["asset_health", "work_orders", "condition_readings", "inventory_reservations"],
	}


def location_model(service: EnterpriseAssetManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "locations",
		"records": service.list_locations(tenant_id),
		"columns": ["location_id", "name", "location_type", "parent_location_id", "status"],
		"actions": ["register_location", "link_parent", "view_assets"],
	}


def asset_model(service: EnterpriseAssetManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "assets",
		"records": service.list_assets(tenant_id),
		"columns": ["asset_id", "name", "owner", "category", "location_id", "criticality", "health_score", "status"],
		"actions": ["register_asset", "create_plan", "open_work_order", "record_condition"],
	}


def maintenance_plan_model(service: EnterpriseAssetManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "maintenance_plans",
		"records": service.list_maintenance_plans(tenant_id),
		"columns": ["plan_id", "asset_id", "strategy", "interval_days", "condition_source", "status"],
		"actions": ["create_plan", "review_strategy", "open_work_order"],
	}


def maintenance_model(service: EnterpriseAssetManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return maintenance_plan_model(service, tenant_id)


def work_order_model(service: EnterpriseAssetManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "work_orders",
		"records": service.list_work_orders(tenant_id),
		"columns": ["work_order_id", "asset_id", "title", "priority", "approved_by", "status"],
		"actions": ["open_work_order", "reserve_inventory", "complete_work_order"],
	}


def inspection_model(service: EnterpriseAssetManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "inspections",
		"records": service.list_inspections(tenant_id),
		"columns": ["inspection_id", "asset_id", "result", "inspector", "condition_score", "status"],
		"actions": ["record_inspection", "record_condition_reading"],
	}


def inventory_model(service: EnterpriseAssetManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "inventory",
		"records": service.list_inventory_reservations(tenant_id),
		"columns": ["reservation_id", "part_id", "quantity", "work_order_record_id", "status"],
		"actions": ["reserve_inventory", "release_reservation", "attach_to_work_order"],
	}


def analytics_model(service: EnterpriseAssetManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "analytics",
		"summary": service.reliability_summary(tenant_id),
		"condition_readings": service.list_condition_readings(tenant_id),
		"actions": ["review_degraded_assets", "schedule_maintenance", "export_reliability_summary"],
	}


def condition_model(service: EnterpriseAssetManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "condition_readings",
		"records": service.list_condition_readings(tenant_id),
		"columns": ["reading_id", "asset_id", "metric", "value", "unit", "status"],
		"actions": ["record_condition_reading", "review_alert"],
	}


def agent_workbench_model(service: EnterpriseAssetManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "agents",
		"records": service.list_eam_agents(tenant_id),
		"supported_runtimes": SUPPORTED_EAM_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_EAM_AGENT_ROLES,
		"actions": ["register_agent", "validate_action", "record_human_approval"],
	}
