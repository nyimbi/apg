"""Process-local API helpers for APG Network Inventory."""

from __future__ import annotations

from .service import TelecomInvService

_SERVICE = TelecomInvService()


def service() -> TelecomInvService:
	return _SERVICE


def commission_asset(payload: dict) -> dict:
	return _SERVICE.commission_asset(payload["asset_id"], payload.get("tenant_id", "default"), payload["asset_type"], payload["serial_number"], payload.get("vendor", "other"), payload.get("model", ""), payload["location"], payload.get("commissioned_at", ""), payload.get("policy_attached", True))


def decommission_asset(payload: dict) -> dict:
	return _SERVICE.decommission_asset(payload["asset_id"], payload.get("tenant_id", "default"), payload["approval_reference"])


def provision_circuit(payload: dict) -> dict:
	return _SERVICE.provision_circuit(payload["circuit_id"], payload.get("tenant_id", "default"), payload["circuit_type"], payload["a_end"], payload["z_end"], payload["capacity"], payload.get("provisioned_at", ""))


def allocate_ip_block(payload: dict) -> dict:
	return _SERVICE.allocate_ip_block(payload["block_id"], payload.get("tenant_id", "default"), payload["ip_version"], payload["prefix"], payload["prefix_length"], payload["block_type"], payload["vrf"], payload.get("allocated_to"), payload.get("allocated_at", ""))


def release_ip_block(payload: dict) -> dict:
	return _SERVICE.release_ip_block(payload["block_id"], payload.get("tenant_id", "default"))


def record_topology(payload: dict) -> dict:
	return _SERVICE.record_topology(payload["topology_id"], payload.get("tenant_id", "default"), payload["topology_type"], payload["domain"], payload["name"], payload.get("description", ""), payload.get("nodes", "[]"), payload.get("edges", "[]"), payload.get("recorded_at", ""))


def register_site(payload: dict) -> dict:
	return _SERVICE.register_site(payload["site_id"], payload.get("tenant_id", "default"), payload["site_name"], payload.get("site_type", "tower"), payload.get("latitude", 0.0), payload.get("longitude", 0.0), payload.get("address", ""), payload.get("region", ""))


def record_discrepancy(payload: dict) -> dict:
	return _SERVICE.record_discrepancy(payload["reconcile_id"], payload.get("tenant_id", "default"), payload["asset_id"], payload["discrepancy_description"])


def approve_reconciliation(payload: dict) -> dict:
	return _SERVICE.approve_reconciliation(payload["reconcile_id"], payload.get("tenant_id", "default"), payload["approval_reference"], payload["resolved_by"], payload.get("resolved_at", ""))


def register_agent(payload: dict) -> dict:
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "inventory operations"))


def validate_batch(payload: dict) -> dict:
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict) -> dict:
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
