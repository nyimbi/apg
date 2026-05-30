"""UI metadata helpers for the Geo-Spatial Services capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import GeosService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: GeosService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or GeosService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"event_sources": service.list_event_sources(tenant_id),
		"geofences": service.list_geofences(tenant_id),
		"location_events": service.list_location_events(tenant_id),
		"territories": service.list_territories(tenant_id),
		"analytics": service.list_analytics(tenant_id),
		"location_agents": service.list_location_agents(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


def map_console_model(service: GeosService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GeosService()
	return {
		"tenant_id": tenant_id,
		"geofences": service.list_geofences(tenant_id),
		"territories": service.list_territories(tenant_id),
		"location_events": service.list_location_events(tenant_id),
		"layers": ["geofences", "territories", "events", "analytics"],
	}


def event_monitor_model(service: GeosService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GeosService()
	return {
		"tenant_id": tenant_id,
		"event_sources": service.list_event_sources(tenant_id),
		"location_events": service.list_location_events(tenant_id),
		"states": ["registered", "processed", "blocked"],
	}


def spatial_analytics_model(service: GeosService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GeosService()
	return {
		"tenant_id": tenant_id,
		"analytics": service.list_analytics(tenant_id),
		"required_controls": ["spatial_index_available", "aggregation_privacy_applied"],
	}


def geofence_editor_model(service: GeosService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GeosService()
	return {
		"tenant_id": tenant_id,
		"geofences": service.list_geofences(tenant_id),
		"required_fields": ["id", "name", "owner", "boundary", "trigger_events"],
	}


def territory_manager_model(service: GeosService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GeosService()
	return {
		"tenant_id": tenant_id,
		"territories": service.list_territories(tenant_id),
		"required_controls": ["owner", "boundary", "overlap_review_recorded"],
	}


def location_agents_model(service: GeosService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GeosService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": service.list_location_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["location_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["location_agents"]["allowed_roles"],
		"theme": contract["theme"]["components"]["agent_panel"],
	}


def audit_trail_model(service: GeosService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GeosService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"streaming": contract["streaming"],
		"theme": contract["theme"]["components"]["audit_timeline"],
	}


def settings_model(service: GeosService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GeosService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"permissions": sorted({route["permission"] for route in contract["ui"]["routes"]}),
	}
