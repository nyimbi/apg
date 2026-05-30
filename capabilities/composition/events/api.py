"""API helpers for the Event Streaming Bus capability."""

from __future__ import annotations

import asyncio
from typing import Any

from .service import CompositionEventsService


SERVICE = CompositionEventsService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"record_count": len(SERVICE.list_records(tenant_id)),
		"stream_count": summary["stream_count"],
		"schema_count": summary["schema_count"],
		"subscription_count": summary["subscription_count"],
		"processor_count": summary["processor_count"],
		"event_agent_count": summary["event_agent_count"],
		"audit_event_count": summary["audit_event_count"],
		"streaming": summary["streaming"],
	}


def create_stream(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_stream(
		stream_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner_id=str(payload["owner_id"]),
		source_capability=str(payload.get("source_capability") or "composition_events"),
		retention_policy=str(payload.get("retention_policy") or "7d"),
		partition_key=str(payload.get("partition_key") or "tenant_id"),
		pii_stream=bool(payload.get("pii_stream", False)),
		schema_id=payload.get("schema_id"),
		event_stream=str(payload.get("event_stream") or "bytewax"),
		metadata=dict(payload.get("metadata") or {}),
	)


def register_schema(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_schema(
		schema_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		version=str(payload.get("version") or "1.0.0"),
		definition=dict(payload.get("definition") or {}),
		breaking_change=bool(payload.get("breaking_change", False)),
		reviewed_by=payload.get("reviewed_by"),
	)


def publish_event(payload: dict[str, Any]) -> dict[str, Any]:
	return asyncio.run(
		SERVICE.publish_event(
			stream_id=str(payload["stream_id"]),
			tenant_id=str(payload.get("tenant_id") or "default"),
			event_type=str(payload["event_type"]),
			payload=dict(payload.get("payload") or {}),
			source_capability=str(payload.get("source_capability") or ""),
			correlation_id=str(payload.get("correlation_id") or ""),
			partition_key=str(payload.get("partition_key") or "tenant_id"),
			event_stream=str(payload.get("event_stream") or "bytewax"),
		)
	)


def validate_batch_publish(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_batch_publish(
		tenant_id=str(payload.get("tenant_id") or "default"),
		batch_size=int(payload.get("batch_size") or 0),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def create_subscription(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_subscription(
		subscription_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		stream_id=str(payload["stream_id"]),
		consumer_owner_id=str(payload["consumer_owner_id"]),
		delivery_mode=str(payload.get("delivery_mode") or "at_least_once"),
		retry_enabled=bool(payload.get("retry_enabled", False)),
		dead_letter_stream_id=payload.get("dead_letter_stream_id"),
	)


def register_processor(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_processor(
		processor_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		stream_id=str(payload["stream_id"]),
		stateful=bool(payload.get("stateful", False)),
		checkpoint_configured=bool(payload.get("checkpoint_configured", True)),
		reviewed_by=payload.get("reviewed_by"),
		processor_runtime=str(payload.get("processor_runtime") or "bytewax"),
	)


def register_event_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_event_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		instructions=str(payload.get("instructions") or ""),
	)


def validate_agent_event_action(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_agent_event_action(
		tenant_id=str(payload.get("tenant_id") or "default"),
		agent_id=str(payload["agent_id"]),
		action=str(payload.get("action") or "review"),
		privileged_scope=bool(payload.get("privileged_scope", False)),
		human_approval_recorded=bool(payload.get("human_approval_recorded", False)),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def capability_listing(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"streams": SERVICE.list_streams(tenant_id),
		"schemas": SERVICE.list_schemas(tenant_id),
		"subscriptions": SERVICE.list_subscriptions(tenant_id),
		"processors": SERVICE.list_processors(tenant_id),
		"agents": SERVICE.list_event_agents(tenant_id),
		"audit_events": SERVICE.audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}
