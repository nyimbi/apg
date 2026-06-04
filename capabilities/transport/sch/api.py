"""Process-local API helpers for APG Transport Scheduling."""

from __future__ import annotations

try:
	from .service import TransportSchedulingService
except ImportError:
	from service import TransportSchedulingService  # type: ignore

_SERVICE = TransportSchedulingService()


def service() -> TransportSchedulingService:
	return _SERVICE


def create_schedule(payload: dict):
	return _SERVICE.create_schedule(payload["schedule_id"], payload.get("tenant_id", "default"), payload["schedule_type"], payload["start_date"], payload["end_date"], payload.get("optimisation_mode", "balanced"), payload.get("created_by", "system"), payload.get("policy_attached", True))


def publish_schedule(payload: dict):
	return _SERVICE.publish_schedule(payload["schedule_id"], payload.get("tenant_id", "default"))


def create_shift(payload: dict):
	return _SERVICE.create_shift(payload["shift_id"], payload.get("tenant_id", "default"), payload["schedule_id"], payload["driver_id"], payload["shift_type"], payload["start_time"], payload["end_time"], payload.get("hours", 8.0), payload.get("driver_hours_compliant", True), payload.get("tacho_compliant", True))


def assign_vehicle(payload: dict):
	return _SERVICE.assign_vehicle(payload["assignment_id"], payload.get("tenant_id", "default"), payload["schedule_id"], payload["vehicle_id"], payload.get("route_id", ""), payload["assigned_from"], payload["assigned_until"], payload.get("double_booking_detected", False))


def create_charter(payload: dict):
	return _SERVICE.create_charter(payload["charter_id"], payload.get("tenant_id", "default"), payload["schedule_id"], payload["charter_type"], payload["customer_id"], payload["vehicle_id"], payload["driver_id"], payload["pickup_location"], payload["destination"], payload["charter_date"], payload.get("customer_confirmed", False))


def record_conflict(payload: dict):
	return _SERVICE.record_conflict(payload["conflict_id"], payload.get("tenant_id", "default"), payload["schedule_id"], payload["conflict_type"], payload["resource_id"], payload["detected_at"])


def resolve_conflict(payload: dict):
	return _SERVICE.resolve_conflict(payload["conflict_id"], payload.get("tenant_id", "default"), payload["resolved_at"], payload.get("resolution_notes", ""))


def register_scheduling_agent(payload: dict):
	return _SERVICE.register_scheduling_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "transport scheduling operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
