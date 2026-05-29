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
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
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
