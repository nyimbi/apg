"""Process-local API helpers for APG Vehicle Maintenance."""

from __future__ import annotations

try:
	from .service import VehicleMaintenanceService
except ImportError:
	from service import VehicleMaintenanceService  # type: ignore

_SERVICE = VehicleMaintenanceService()


def service() -> VehicleMaintenanceService:
	return _SERVICE


def create_job(payload: dict):
	return _SERVICE.create_job(payload["job_id"], payload.get("tenant_id", "default"), payload["vehicle_id"], payload["maintenance_type"], payload.get("priority", "medium"), payload["technician_id"], payload.get("workshop_type", "in_house"), payload.get("estimated_hours", 1.0), payload.get("job_card_ref", ""), payload.get("policy_attached", True))


def update_job_status(payload: dict):
	return _SERVICE.update_job_status(payload["job_id"], payload.get("tenant_id", "default"), payload["status"], payload.get("actual_hours"))


def allocate_workshop(payload: dict):
	return _SERVICE.allocate_workshop(payload["allocation_id"], payload.get("tenant_id", "default"), payload["workshop_type"], payload["location"], payload["bay_number"], payload["job_id"], payload["allocated_at"])


def order_parts(payload: dict):
	return _SERVICE.order_parts(payload["order_id"], payload.get("tenant_id", "default"), payload["job_id"], payload["parts_category"], payload["part_number"], payload["description"], payload["quantity"], payload["supplier_id"], payload["ordered_at"])


def record_warranty(payload: dict):
	return _SERVICE.record_warranty(payload["warranty_id"], payload.get("tenant_id", "default"), payload["vehicle_id"], payload["warranty_type"], payload["provider"], payload["start_date"], payload["expiry_date"], payload.get("claim_ref"))


def conduct_inspection(payload: dict):
	return _SERVICE.conduct_inspection(payload["inspection_id"], payload.get("tenant_id", "default"), payload["vehicle_id"], payload["inspection_type"], payload["inspector_id"], payload["conducted_at"], payload.get("defects_found", False), payload["digital_signature"], payload.get("passed", True))


def issue_roadworthiness(payload: dict):
	return _SERVICE.issue_roadworthiness(payload["record_id"], payload.get("tenant_id", "default"), payload["vehicle_id"], payload["standard"], payload["certificate_number"], payload["issued_at"], payload["expires_at"], payload["issuing_authority"])


def create_schedule(payload: dict):
	return _SERVICE.create_schedule(payload["schedule_id"], payload.get("tenant_id", "default"), payload["vehicle_id"], payload["maintenance_type"], payload["scheduled_at"], payload.get("interval_km"), payload.get("interval_days"))


def register_maintenance_agent(payload: dict):
	return _SERVICE.register_maintenance_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "vehicle maintenance operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
