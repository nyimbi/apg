"""Dependency-light API helpers for APG advanced CRM analytics."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import AdvancedCRMService
except ImportError:
	from capability_contract import get_capability_contract
	from service import AdvancedCRMService


_SERVICE = AdvancedCRMService()


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


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_record(payload)


def list_records(tenant_id: str = "default") -> list[dict[str, Any]]:
	return _SERVICE.list_records(tenant_id)


def create_account(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_account(payload["account_id"], payload.get("tenant_id", "default"), payload["name"], payload["owner"], payload["segment"], payload.get("territory"))


def create_lead(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_lead(payload["lead_id"], payload.get("tenant_id", "default"), payload["name"], payload["source"], payload.get("score"))


def create_opportunity(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_opportunity(payload["opportunity_id"], payload.get("tenant_id", "default"), payload["account_id"], payload["name"], payload["stage"], payload["amount"], payload["close_date"])


def register_crm_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_crm_agent(payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("instructions", ""))


def service() -> AdvancedCRMService:
	return _SERVICE
