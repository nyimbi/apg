"""Dependency-light API helpers for APG enterprise asset management."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import EnterpriseAssetManagementService
except ImportError:
	from capability_contract import get_capability_contract
	from service import EnterpriseAssetManagementService


_SERVICE = EnterpriseAssetManagementService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"ok": True,
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"provides": contract["provides"],
		"requires": contract["requires"],
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"streaming": contract["streaming"],
		"summary": _SERVICE.dashboard_summary(tenant_id),
	}


def register_location(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_location(
		payload["location_id"],
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["location_type"],
		payload.get("parent_location_id"),
	)


def register_asset(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_asset(
		payload["asset_id"],
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["owner"],
		payload["category"],
		payload["location_id"],
		payload["criticality"],
		payload.get("health_score", 100),
		payload.get("capitalized", False),
		payload.get("fixed_asset_ref"),
	)


def create_maintenance_plan(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_maintenance_plan(
		payload["plan_id"],
		payload.get("tenant_id", "default"),
		payload["asset_record_id"],
		payload["strategy"],
		payload["interval_days"],
		payload.get("condition_source"),
	)


def open_work_order(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.open_work_order(
		payload["work_order_id"],
		payload.get("tenant_id", "default"),
		payload["asset_record_id"],
		payload["title"],
		payload["priority"],
		payload["safety_plan"],
		payload.get("approved_by"),
	)


def complete_work_order(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.complete_work_order(
		payload.get("tenant_id", "default"),
		payload["work_order_record_id"],
		payload["outcome"],
		payload["completed_by"],
	)


def record_inspection(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_inspection(
		payload["inspection_id"],
		payload.get("tenant_id", "default"),
		payload["asset_record_id"],
		payload["result"],
		payload["inspector"],
		payload.get("condition_score"),
	)


def record_condition_reading(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_condition_reading(
		payload["reading_id"],
		payload.get("tenant_id", "default"),
		payload["asset_record_id"],
		payload["metric"],
		payload["value"],
		payload["unit"],
		payload.get("review_recorded", False),
		payload.get("alert_threshold"),
	)


def reserve_inventory(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.reserve_inventory(
		payload["reservation_id"],
		payload.get("tenant_id", "default"),
		payload["part_id"],
		payload["quantity"],
		payload.get("work_order_record_id"),
	)


def register_eam_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_eam_agent(
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("instructions", ""),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_record(payload)


def list_records(tenant_id: str = "default") -> list[dict[str, Any]]:
	return _SERVICE.list_records(tenant_id)


def service() -> EnterpriseAssetManagementService:
	return _SERVICE
