"""Process-local API helpers for APG Fuel Management."""

from __future__ import annotations

try:
	from .service import FuelManagementService
except ImportError:
	from service import FuelManagementService  # type: ignore

_SERVICE = FuelManagementService()


def service() -> FuelManagementService:
	return _SERVICE


def create_procurement(payload: dict):
	return _SERVICE.create_procurement(payload["procurement_id"], payload.get("tenant_id", "default"), payload["procurement_type"], payload["supplier_id"], payload["fuel_type"], payload["quantity_litres"], payload["unit_price"], payload["currency"], payload.get("purchase_order_ref", ""), payload.get("policy_attached", True))


def record_transaction(payload: dict):
	return _SERVICE.record_transaction(payload["transaction_id"], payload.get("tenant_id", "default"), payload["transaction_type"], payload["vehicle_id"], payload["driver_id"], payload["fuel_type"], payload["quantity_litres"], payload["odometer_km"], payload["unit_price"], payload["currency"], payload["transaction_at"], payload.get("card_id"), payload.get("phantom_fill_detected", False), payload.get("theft_pattern_detected", False))


def register_fuel_card(payload: dict):
	return _SERVICE.register_fuel_card(payload["card_id"], payload.get("tenant_id", "default"), payload["provider"], payload["card_number_masked"], payload.get("vehicle_id"), payload.get("driver_id"))


def reconcile_fuel_card(payload: dict):
	return _SERVICE.reconcile_fuel_card(payload["reconciliation_id"], payload.get("tenant_id", "default"), payload["card_id"], payload["period_start"], payload["period_end"], payload["expected_total"], payload["actual_total"], payload["currency"])


def record_carbon_emission(payload: dict):
	return _SERVICE.record_carbon_emission(payload["record_id"], payload.get("tenant_id", "default"), payload["vehicle_id"], payload["standard"], payload["fuel_type"], payload["quantity_litres"], payload["co2_kg"], payload["period_start"], payload["period_end"])


def register_storage_tank(payload: dict):
	return _SERVICE.register_storage_tank(payload["tank_id"], payload.get("tenant_id", "default"), payload["storage_type"], payload["location"], payload["capacity_litres"], payload["fuel_type"], payload["last_calibrated"])


def register_fuel_agent(payload: dict):
	return _SERVICE.register_fuel_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "fuel management operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
