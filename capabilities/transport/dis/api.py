"""Process-local API helpers for APG Dispatch Operations."""

from __future__ import annotations

try:
	from .service import DispatchOperationsService
except ImportError:
	from service import DispatchOperationsService  # type: ignore

_SERVICE = DispatchOperationsService()


def service() -> DispatchOperationsService:
	return _SERVICE


def plan_load(payload: dict):
	return _SERVICE.plan_load(payload["load_id"], payload.get("tenant_id", "default"), payload["load_type"], payload["vehicle_id"], payload["total_weight_kg"], payload.get("total_volume_cbm", 0.0), payload.get("stop_count", 1), payload.get("optimisation_mode", "balanced"), payload.get("policy_attached", True))


def assign_driver(payload: dict):
	return _SERVICE.assign_driver(payload["assignment_id"], payload.get("tenant_id", "default"), payload["dispatch_id"], payload["driver_id"], payload["assignment_type"], payload["assigned_at"], payload.get("hours_available", 10.0))


def create_dispatch(payload: dict):
	return _SERVICE.create_dispatch(payload["dispatch_id"], payload.get("tenant_id", "default"), payload["load_plan_id"], payload["vehicle_id"], payload["driver_id"], payload["route_id"])


def update_dispatch_status(payload: dict):
	return _SERVICE.update_dispatch_status(payload["dispatch_id"], payload.get("tenant_id", "default"), payload["status"], payload.get("timestamp"))


def update_tracking(payload: dict):
	return _SERVICE.update_tracking(payload["update_id"], payload.get("tenant_id", "default"), payload["dispatch_id"], payload["update_type"], payload["location"], payload["timestamp"], payload.get("eta_minutes"))


def raise_exception(payload: dict):
	return _SERVICE.raise_exception(payload["exception_id"], payload.get("tenant_id", "default"), payload["dispatch_id"], payload["exception_type"], payload["raised_at"])


def resolve_exception(payload: dict):
	return _SERVICE.resolve_exception(payload["exception_id"], payload.get("tenant_id", "default"), payload["resolved_at"], payload.get("resolution_notes", ""))


def register_dispatch_agent(payload: dict):
	return _SERVICE.register_dispatch_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "dispatch operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
