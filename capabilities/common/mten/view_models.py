"""Dependency-light MTEN view models for generated APG applications."""

from __future__ import annotations

from . import api_helpers
from .capability_contract import get_capability_contract
from .mten_runtime import MtenService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: MtenService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api_helpers.SERVICE
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.portfolio_summary(tenant_id),
		"tenants": service.list_tenants(tenant_id),
		"capacity_approvals": service.list_capacity_approvals(tenant_id),
		"isolation_incidents": service.list_isolation_incidents(tenant_id),
		"live_migrations": service.list_live_migrations(tenant_id),
		"governance_events": service.list_governance_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def provisioning_model(service: MtenService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api_helpers.SERVICE
	return {
		"tenant_id": tenant_id,
		"provisioning": [
			item for item in service.list_tenants(tenant_id)
			if item["status"] == "provisioning"
		],
		"active": [
			item for item in service.list_tenants(tenant_id)
			if item["status"] == "active"
		],
	}


def capacity_approval_model(service: MtenService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api_helpers.SERVICE
	return {
		"tenant_id": tenant_id,
		"approvals": service.list_capacity_approvals(tenant_id),
		"pending": [
			item for item in service.list_capacity_approvals(tenant_id)
			if item["status"] == "pending"
		],
	}


def isolation_model(service: MtenService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api_helpers.SERVICE
	return {
		"tenant_id": tenant_id,
		"incidents": service.list_isolation_incidents(tenant_id),
		"suspended_tenants": [
			item for item in service.list_tenants(tenant_id)
			if item["status"] == "suspended"
		],
	}


def migration_model(service: MtenService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api_helpers.SERVICE
	return {
		"tenant_id": tenant_id,
		"migrations": service.list_live_migrations(tenant_id),
	}


def governance_model(service: MtenService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api_helpers.SERVICE
	return {
		"tenant_id": tenant_id,
		"events": service.list_governance_events(tenant_id),
	}
