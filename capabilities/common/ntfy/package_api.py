"""Dependency-light API helpers for package-backed NTFY composition."""

from __future__ import annotations

from typing import Any

from .notification_runtime import NotificationRuntime


RUNTIME = NotificationRuntime()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = RUNTIME.describe(tenant_id)
	summary = RUNTIME.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**summary,
	}


def register_channel(payload: dict[str, Any]) -> dict[str, Any]:
	return RUNTIME.register_channel(
		tenant_id=str(payload.get("tenant_id") or "default"),
		channel=str(payload["channel"]),
		provider=str(payload["provider"]),
		owner=str(payload["owner"]),
		healthy=bool(payload.get("healthy", True)),
		fallback_channel=payload.get("fallback_channel"),
	)


def register_preference(payload: dict[str, Any]) -> dict[str, Any]:
	return RUNTIME.register_preference(
		tenant_id=str(payload.get("tenant_id") or "default"),
		recipient_id=str(payload["recipient_id"]),
		addresses=dict(payload.get("addresses") or {}),
		preferred_channels=[str(item) for item in payload.get("preferred_channels", [])],
		opted_in=bool(payload.get("opted_in", False)),
		unsubscribed=bool(payload.get("unsubscribed", False)),
		quiet_hours=dict(payload.get("quiet_hours") or {}),
	)


def register_template(payload: dict[str, Any]) -> dict[str, Any]:
	return RUNTIME.register_template(
		tenant_id=str(payload.get("tenant_id") or "default"),
		template_id=str(payload["template_id"]),
		name=str(payload["name"]),
		owner=str(payload["owner"]),
		locale=str(payload.get("locale") or "en"),
		channels=[str(item) for item in payload.get("channels", [])],
		content=dict(payload.get("content") or {}),
		approved=bool(payload.get("approved", False)),
		version=str(payload.get("version") or "v1"),
	)


def approve_template(payload: dict[str, Any]) -> dict[str, Any]:
	return RUNTIME.approve_template(
		tenant_id=str(payload.get("tenant_id") or "default"),
		template_id=str(payload["template_id"]),
		approved_by=str(payload["approved_by"]),
	)


def send_message(payload: dict[str, Any]) -> dict[str, Any]:
	return RUNTIME.send_message(
		tenant_id=str(payload.get("tenant_id") or "default"),
		template_id=str(payload["template_id"]),
		recipient_id=str(payload["recipient_id"]),
		channel=str(payload["channel"]),
		message_class=str(payload.get("message_class") or "transactional"),
		priority=str(payload.get("priority") or "normal"),
		sensitive_payload=bool(payload.get("sensitive_payload", False)),
		payload_encrypted=bool(payload.get("payload_encrypted", False)),
		idempotency_key=payload.get("idempotency_key"),
		webhook_signature_present=bool(payload.get("webhook_signature_present", True)),
		event_bus_present=bool(payload.get("event_bus_present", True)),
		quiet_hours_active=bool(payload.get("quiet_hours_active", False)),
	)


def create_campaign(payload: dict[str, Any]) -> dict[str, Any]:
	return RUNTIME.create_campaign(
		tenant_id=str(payload.get("tenant_id") or "default"),
		campaign_id=str(payload["campaign_id"]),
		name=str(payload["name"]),
		owner=str(payload["owner"]),
		template_id=str(payload["template_id"]),
		audience=[str(item) for item in payload.get("audience", [])],
		channels=[str(item) for item in payload.get("channels", [])],
		message_class=str(payload.get("message_class") or "marketing"),
	)


def approve_campaign(payload: dict[str, Any]) -> dict[str, Any]:
	return RUNTIME.approve_campaign(
		tenant_id=str(payload.get("tenant_id") or "default"),
		campaign_id=str(payload["campaign_id"]),
		approved_by=str(payload["approved_by"]),
	)


def send_campaign(payload: dict[str, Any]) -> dict[str, Any]:
	return RUNTIME.send_campaign(
		tenant_id=str(payload.get("tenant_id") or "default"),
		campaign_id=str(payload["campaign_id"]),
		batch_review_recorded=bool(payload.get("batch_review_recorded", False)),
	)


def notification_state(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"summary": RUNTIME.dashboard_summary(tenant_id),
		"channels": RUNTIME.list_channels(tenant_id),
		"preferences": RUNTIME.list_preferences(tenant_id),
		"templates": RUNTIME.list_templates(tenant_id),
		"deliveries": RUNTIME.list_deliveries(tenant_id),
		"campaigns": RUNTIME.list_campaigns(tenant_id),
		"audit_events": RUNTIME.list_audit_events(tenant_id),
	}
