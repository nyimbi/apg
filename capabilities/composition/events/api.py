"""API helpers for the Event Streaming Bus capability."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.requests import Request

from .service import CompositionEventsService

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


def _clean_text(value: Any) -> Optional[str]:
	"""Return a non-empty stripped string or None."""
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _decode_jwt_claims(token: str) -> Optional[Dict[str, Any]]:
	"""Decode JWT payload without signature verification."""
	try:
		parts = token.split(".")
		if len(parts) < 2:
			return None
		payload = parts[1]
		padding = "=" * (-len(payload) % 4)
		data = base64.urlsafe_b64decode(f"{payload}{padding}".encode("ascii"))
		return json.loads(data.decode("utf-8"))
	except (binascii.Error, json.JSONDecodeError, Exception):
		return None


async def get_current_user(
	request: Request,
	credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
	"""Resolve current user from JWT claims, headers, or environment."""
	if credentials and credentials.credentials:
		claims = _decode_jwt_claims(credentials.credentials)
		if claims:
			user_id = _clean_text(claims.get("sub") or claims.get("user_id"))
			tenant_id = _clean_text(claims.get("tenant_id") or claims.get("org_id"))
			permissions = claims.get("permissions") or []
			if user_id:
				return {
					"user_id": user_id,
					"tenant_id": tenant_id or os.getenv("APG_DEFAULT_TENANT_ID", os.getenv("APG_TENANT_ID", "default")),
					"permissions": permissions,
				}

	headers = getattr(request, "headers", {})
	query_params = getattr(request, "query_params", {})

	def _hget(*keys: str) -> Optional[str]:
		for k in keys:
			v = _clean_text(headers.get(k))
			if v:
				return v
		return None

	def _qget(*keys: str) -> Optional[str]:
		for k in keys:
			v = _clean_text(query_params.get(k))
			if v:
				return v
		return None

	return {
		"user_id": (
			_hget("X-APG-User-ID", "X-User-ID")
			or _qget("user_id", "user")
			or os.getenv("APG_DEFAULT_USER_ID", os.getenv("APG_USER_ID", "system"))
		),
		"tenant_id": (
			_hget("X-APG-Tenant-ID", "X-Tenant-ID")
			or _qget("tenant_id", "tenant")
			or os.getenv("APG_DEFAULT_TENANT_ID", os.getenv("APG_TENANT_ID", "default"))
		),
		"permissions": [],
	}


async def get_event_streaming_service(request: Request) -> "CompositionEventsService":
	"""FastAPI dependency: resolve the event streaming service from app state or default."""
	state = getattr(getattr(request, "app", None), "state", None)
	service = getattr(state, "event_streaming_service", None)
	if service is not None:
		return service
	return SERVICE


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
