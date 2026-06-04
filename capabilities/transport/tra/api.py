"""Process-local API helpers for APG Asset Tracking."""

from __future__ import annotations

try:
	from .service import AssetTrackingService
except ImportError:
	from service import AssetTrackingService  # type: ignore

_SERVICE = AssetTrackingService()


def service() -> AssetTrackingService:
	return _SERVICE


def register_asset(payload: dict):
	return _SERVICE.register_asset(payload["asset_id"], payload.get("tenant_id", "default"), payload["asset_type"], payload["unique_id"], payload["owner_id"], payload.get("registration", ""), payload.get("tracking_technology", "gps"), payload.get("policy_attached", True))


def update_asset_location(payload: dict):
	return _SERVICE.update_asset_location(payload["update_id"], payload.get("tenant_id", "default"), payload["asset_id"], payload["latitude"], payload["longitude"], payload.get("speed_kmh", 0.0), payload.get("heading_degrees", 0.0), payload["timestamp"], payload.get("source", "gps"), payload.get("tamper_detected", False))


def create_geofence(payload: dict):
	return _SERVICE.create_geofence(payload["geofence_id"], payload.get("tenant_id", "default"), payload["geofence_type"], payload["name"], payload["boundary_definition"], payload.get("alert_on_entry", True), payload.get("alert_on_exit", True))


def raise_alert(payload: dict):
	return _SERVICE.raise_alert(payload["alert_id"], payload.get("tenant_id", "default"), payload["asset_id"], payload["alert_type"], payload.get("severity", "high"), payload["raised_at"], payload.get("details", ""))


def record_cold_chain(payload: dict):
	return _SERVICE.record_cold_chain(payload["record_id"], payload.get("tenant_id", "default"), payload["asset_id"], payload["standard"], payload["min_temp_c"], payload["max_temp_c"], payload["recorded_temp_c"], payload["timestamp"])


def register_container(payload: dict):
	return _SERVICE.register_container(payload["container_id"], payload.get("tenant_id", "default"), payload["iso_number"], payload.get("seal_number", ""), payload["owner_id"], payload.get("current_location", ""), payload["last_updated"])


def update_container_status(payload: dict):
	return _SERVICE.update_container_status(payload["container_id"], payload.get("tenant_id", "default"), payload["status"])


def record_utilisation(payload: dict):
	return _SERVICE.record_utilisation(payload["record_id"], payload.get("tenant_id", "default"), payload["asset_id"], payload["period"], payload["period_start"], payload["period_end"], payload["idle_time_minutes"], payload["active_time_minutes"], payload.get("distance_km", 0.0))


def register_tracking_agent(payload: dict):
	return _SERVICE.register_tracking_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "asset tracking operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
