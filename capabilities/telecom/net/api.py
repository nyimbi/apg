"""Process-local API helpers for APG Network Management."""

from __future__ import annotations

from .service import TelecomNetService

_SERVICE = TelecomNetService()


def service() -> TelecomNetService:
	return _SERVICE


def raise_alarm(payload: dict) -> dict:
	return _SERVICE.raise_alarm(payload["alarm_id"], payload.get("tenant_id", "default"), payload["ne_reference"], payload["severity"], payload["category"], payload.get("description", ""), payload.get("raised_at", ""), payload.get("policy_attached", True))


def update_alarm_status(payload: dict) -> dict:
	return _SERVICE.update_alarm_status(payload["alarm_id"], payload.get("tenant_id", "default"), payload["new_status"], payload.get("cleared_at"))


def suppress_alarm(payload: dict) -> dict:
	return _SERVICE.suppress_alarm(payload["alarm_id"], payload.get("tenant_id", "default"), payload["approval_reference"])


def open_fault_ticket(payload: dict) -> dict:
	return _SERVICE.open_fault_ticket(payload["ticket_id"], payload.get("tenant_id", "default"), payload["alarm_id"], payload["title"], payload["severity"], payload.get("escalation_level", "tier1"))


def resolve_fault_ticket(payload: dict) -> dict:
	return _SERVICE.resolve_fault_ticket(payload["ticket_id"], payload.get("tenant_id", "default"), payload.get("resolved_at", ""))


def record_performance(payload: dict) -> dict:
	return _SERVICE.record_performance(payload["record_id"], payload.get("tenant_id", "default"), payload["ne_reference"], payload["metric_type"], payload["value"], payload.get("threshold", 0.0), payload.get("domain", "core"), payload.get("recorded_at", ""))


def submit_config_change(payload: dict) -> dict:
	return _SERVICE.submit_config_change(payload["change_id"], payload.get("tenant_id", "default"), payload["ne_reference"], payload["change_type"], payload["description"], payload["approval_reference"], payload.get("submitted_by", ""), payload.get("submitted_at", ""), payload.get("in_freeze_period", False))


def complete_config_change(payload: dict) -> dict:
	return _SERVICE.complete_config_change(payload["change_id"], payload.get("tenant_id", "default"))


def record_sla(payload: dict) -> dict:
	return _SERVICE.record_sla(payload["sla_id"], payload.get("tenant_id", "default"), payload["sla_type"], payload.get("customer_id"), payload["target_value"], payload["actual_value"], payload.get("period", ""))


def record_noc_handover(payload: dict) -> dict:
	return _SERVICE.record_noc_handover(payload["handover_id"], payload.get("tenant_id", "default"), payload["shift"], payload["handing_over_operator"], payload["taking_over_operator"], payload["notes"], payload.get("open_alarms_count", 0), payload.get("handover_at", ""))


def register_agent(payload: dict) -> dict:
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "network management operations"))


def validate_batch(payload: dict) -> dict:
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict) -> dict:
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
