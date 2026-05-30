"""Dependency-light API helpers for APG capability registry."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import CompositionRegistryService
except ImportError:
	from capability_contract import get_capability_contract
	from service import CompositionRegistryService


_SERVICE = CompositionRegistryService()


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


def register_capability(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_capability(
		payload["capability_id"],
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["owner"],
		payload["category"],
		payload.get("version", "1.0.0"),
		payload["provides"],
		payload["contract_ref"],
		payload.get("requires"),
	)


def create_composition(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_composition(
		payload["composition_id"],
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["owner"],
		payload["capability_ids"],
	)


def register_registry_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_registry_agent(
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("instructions", ""),
	)


def service() -> CompositionRegistryService:
	return _SERVICE
