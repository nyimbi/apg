"""API helpers for APG Multi-Channel Output."""

from __future__ import annotations

from typing import Any

from .service import MchnService


SERVICE = MchnService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"contract_route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**SERVICE.dashboard_summary(tenant_id),
	}


def create_channel(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_channel(
		channel_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		channel_type=str(payload["channel_type"]),
		owner=str(payload["owner"]),
		provider_ref=str(payload["provider_ref"]),
		health=str(payload.get("health") or "healthy"),
		fallback_channel_id=str(payload.get("fallback_channel_id") or ""),
		status=str(payload.get("status") or "active"),
	)


def publish_template(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_template(
		template_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		channel_types=tuple(payload.get("channel_types") or (payload.get("channel_type") or "email",)),
		subject_template=str(payload.get("subject_template") or ""),
		body_template=str(payload.get("body_template") or ""),
		locale=str(payload.get("locale") or "en"),
		theme_ref=str(payload.get("theme_ref") or "mchn_omnichannel_output"),
		approved=bool(payload.get("approved", False)),
		approved_by=str(payload.get("approved_by") or ""),
		status=str(payload.get("status") or "published"),
	)


def create_delivery_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_delivery_policy(
		policy_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		max_recipients=int(payload.get("max_recipients", 10000)),
		throttle_per_minute=int(payload.get("throttle_per_minute", 1000)),
		requires_encryption_for_sensitive=bool(payload.get("requires_encryption_for_sensitive", True)),
		compliance_ref=str(payload.get("compliance_ref") or ""),
		status=str(payload.get("status") or "active"),
	)


def create_route(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_route(
		route_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		template_id=str(payload["template_id"]),
		primary_channel_id=str(payload["primary_channel_id"]),
		fallback_channel_ids=tuple(payload.get("fallback_channel_ids") or ()),
		policy_id=str(payload["policy_id"]),
		status=str(payload.get("status") or "active"),
	)


def render_output(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.render_output(
		output_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		route_id=str(payload["route_id"]),
		recipient_ref=str(payload["recipient_ref"]),
		variables=dict(payload.get("variables") or {}),
		output_format=str(payload.get("output_format") or "text"),
		sensitive_output=bool(payload.get("sensitive_output", False)),
		output_encrypted=bool(payload.get("output_encrypted", True)),
	)


def deliver_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.deliver_batch(
		batch_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		route_id=str(payload["route_id"]),
		requested_by=str(payload["requested_by"]),
		rendered_output_ids=tuple(payload.get("rendered_output_ids") or ()),
		recipient_count=int(payload.get("recipient_count", 0)),
		delivery_review_recorded=bool(payload.get("delivery_review_recorded", False)),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def record_receipt(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_receipt(
		receipt_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		batch_id=str(payload["batch_id"]),
		rendered_output_id=str(payload["rendered_output_id"]),
		delivery_state=str(payload["delivery_state"]),
		provider_message_id=str(payload.get("provider_message_id") or ""),
	)


def register_mchn_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_mchn_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload["scope"]),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		agent_id=str(payload["id"]) if payload.get("id") else None,
	)


def validate_batch_output_mutation(event_stream: str) -> dict[str, Any]:
	return SERVICE.validate_batch_output_mutation(event_stream)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def list_mchn_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_mchn_agents(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
