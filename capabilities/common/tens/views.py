"""UI metadata helpers for the Tenants Legacy capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import TensService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: TensService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or TensService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def legacy_tenant_registry_model(
	service: TensService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or TensService()
	return {
		"route": "/tens/tenants",
		"tenant_id": tenant_id,
		"legacy_tenants": service.list_legacy_tenants(tenant_id),
		"states": ["active", "stale", "mapped", "migration_ready", "migrated", "deprecated", "blocked"],
	}


def mapping_workbench_model(
	service: TensService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or TensService()
	return {
		"route": "/tens/mappings",
		"tenant_id": tenant_id,
		"mappings": service.list_mappings(tenant_id),
		"validation_required": True,
	}


def migration_queue_model(
	service: TensService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or TensService()
	return {
		"route": "/tens/migrations",
		"tenant_id": tenant_id,
		"migrations": service.list_migrations(tenant_id),
		"states": ["planned", "approved", "executing", "completed", "blocked"],
	}


def boundary_review_model(
	service: TensService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or TensService()
	return {
		"route": "/tens/boundaries",
		"tenant_id": tenant_id,
		"boundaries": service.list_boundaries(tenant_id),
		"required_evidence": ["auth_boundary", "role_mapping", "tenant_isolation", "privileged_access_review"],
	}


def deprecation_model(
	service: TensService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or TensService()
	return {
		"route": "/tens/deprecation",
		"tenant_id": tenant_id,
		"deprecations": service.list_deprecations(tenant_id),
		"plan_required": True,
	}


def audit_model(
	service: TensService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or TensService()
	return {
		"route": "/tens/audit",
		"tenant_id": tenant_id,
		"events": service.list_audit_events(tenant_id),
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/tens/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"theme": contract["theme"],
	}
