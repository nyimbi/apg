"""Process-local API helpers for APG Cargo Management."""

from __future__ import annotations

try:
	from .service import CargoManagementService
except ImportError:
	from service import CargoManagementService  # type: ignore

_SERVICE = CargoManagementService()


def service() -> CargoManagementService:
	return _SERVICE


def create_booking(payload: dict):
	return _SERVICE.create_booking(payload["booking_id"], payload.get("tenant_id", "default"), payload["cargo_type"], payload["shipper_id"], payload["consignee_id"], payload["origin"], payload["destination"], payload["weight_kg"], payload.get("volume_cbm", 0.0), payload.get("incoterm", "fob"), payload.get("packaging_type", "pallet"), payload.get("policy_attached", True))


def create_manifest(payload: dict):
	return _SERVICE.create_manifest(payload["manifest_id"], payload.get("tenant_id", "default"), payload["booking_id"], payload["customs_declaration_ref"], payload.get("submitted_at"))


def declare_dangerous_goods(payload: dict):
	return _SERVICE.declare_dangerous_goods(payload["dg_id"], payload.get("tenant_id", "default"), payload["booking_id"], payload["dg_class"], payload["un_number"], payload["packing_group"], payload["emergency_contact"], payload.get("compliance_standard", "iata"))


def update_tracking(payload: dict):
	return _SERVICE.update_tracking(payload["event_id"], payload.get("tenant_id", "default"), payload["booking_id"], payload["event_type"], payload["location"], payload["timestamp"], payload.get("notes", ""))


def record_revenue(payload: dict):
	return _SERVICE.record_revenue(payload["record_id"], payload.get("tenant_id", "default"), payload["booking_id"], payload["revenue_type"], payload["amount"], payload["currency"], payload.get("reference", ""))


def record_compliance(payload: dict):
	return _SERVICE.record_compliance(payload["record_id"], payload.get("tenant_id", "default"), payload["booking_id"], payload["standard"], payload["certificate_ref"], payload["checked_at"], payload.get("passed", True))


def register_cargo_agent(payload: dict):
	return _SERVICE.register_cargo_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "cargo management operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
