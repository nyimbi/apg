"""Process-local API helpers for APG Route Optimisation."""

from __future__ import annotations

try:
	from .service import RouteOptimisationService
except ImportError:
	from service import RouteOptimisationService  # type: ignore

_SERVICE = RouteOptimisationService()


def service() -> RouteOptimisationService:
	return _SERVICE


def plan_route(payload: dict):
	return _SERVICE.plan_route(payload["route_id"], payload.get("tenant_id", "default"), payload["route_type"], payload["origin"], payload["destination"], payload["vehicle_id"], payload.get("transport_mode", "road"), payload.get("stop_count", 1), payload.get("total_distance_km", 0.0), payload.get("estimated_duration_minutes", 0), payload.get("optimisation_objective", "minimize_cost"), payload.get("address_validated", True), payload.get("capacity_constraint_violated", False), payload.get("stops_exceed_maximum", False), payload.get("policy_attached", True))


def add_route_stop(payload: dict):
	return _SERVICE.add_route_stop(payload["stop_id"], payload.get("tenant_id", "default"), payload["route_id"], payload["sequence"], payload["location"], payload["address"], payload["time_window_start"], payload["time_window_end"], payload.get("service_time_minutes", 15))


def add_constraint(payload: dict):
	return _SERVICE.add_constraint(payload["constraint_id"], payload.get("tenant_id", "default"), payload["route_id"], payload["constraint_type"], payload.get("parameters", "{}"))


def record_traffic_event(payload: dict):
	return _SERVICE.record_traffic_event(payload["event_id"], payload.get("tenant_id", "default"), payload["provider"], payload["route_id"], payload.get("delay_minutes", 0), payload["recorded_at"], payload.get("incident_type"))


def trigger_reroute(payload: dict):
	return _SERVICE.trigger_reroute(payload["reroute_id"], payload.get("tenant_id", "default"), payload["original_route_id"], payload["new_route_id"], payload["trigger"], payload["triggered_at"], payload.get("distance_delta_km", 0.0))


def plan_multimodal_segment(payload: dict):
	return _SERVICE.plan_multimodal_segment(payload["segment_id"], payload.get("tenant_id", "default"), payload["route_id"], payload["transport_mode"], payload["segment_origin"], payload["segment_destination"], payload.get("carrier_ref", ""), payload.get("estimated_duration_minutes", 0))


def register_route_agent(payload: dict):
	return _SERVICE.register_route_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "route optimisation operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
