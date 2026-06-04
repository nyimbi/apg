"""Process-local API helpers for APG Service Provisioning."""

from __future__ import annotations

from .service import TelecomProService

_SERVICE = TelecomProService()


def service() -> TelecomProService:
	return _SERVICE


def start_workflow(payload: dict) -> dict:
	return _SERVICE.start_workflow(payload["workflow_id"], payload.get("tenant_id", "default"), payload["workflow_type"], payload["order_reference"], payload.get("started_at", ""), payload.get("policy_attached", True))


def update_workflow_status(payload: dict) -> dict:
	return _SERVICE.update_workflow_status(payload["workflow_id"], payload.get("tenant_id", "default"), payload["new_status"], payload.get("completed_at"))


def reserve_resource(payload: dict) -> dict:
	return _SERVICE.reserve_resource(payload["reservation_id"], payload.get("tenant_id", "default"), payload["workflow_id"], payload["resource_type"], payload["resource_value"], payload.get("reserved_at", ""), payload.get("expires_at", ""))


def release_resource(payload: dict) -> dict:
	return _SERVICE.release_resource(payload["reservation_id"], payload.get("tenant_id", "default"))


def push_config(payload: dict) -> dict:
	return _SERVICE.push_config(payload["push_id"], payload.get("tenant_id", "default"), payload["workflow_id"], payload["ne_reference"], payload["push_method"], payload.get("template_reference", ""), payload.get("pushed_at", ""))


def confirm_activation(payload: dict) -> dict:
	return _SERVICE.confirm_activation(payload["activation_id"], payload.get("tenant_id", "default"), payload["workflow_id"], payload["service_reference"], payload.get("activated_at", ""), payload.get("confirmed_by", ""))


def trigger_rollback(payload: dict) -> dict:
	return _SERVICE.trigger_rollback(payload["rollback_id"], payload.get("tenant_id", "default"), payload["workflow_id"], payload["trigger"], payload.get("description", ""), payload.get("triggered_at", ""))


def complete_rollback(payload: dict) -> dict:
	return _SERVICE.complete_rollback(payload["rollback_id"], payload.get("tenant_id", "default"), payload.get("completed_at", ""))


def start_bulk_provisioning(payload: dict) -> dict:
	return _SERVICE.start_bulk_provisioning(payload["bulk_id"], payload.get("tenant_id", "default"), payload["workflow_type"], payload["item_count"], payload["approval_reference"], payload.get("submitted_by", ""), payload.get("submitted_at", ""))


def register_agent(payload: dict) -> dict:
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "provisioning operations"))


def validate_batch(payload: dict) -> dict:
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict) -> dict:
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
