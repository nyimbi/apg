"""API helpers for the API Service Mesh capability."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import HTTPException
from starlette.requests import Request

from .context import get_current_user_id_from_request, get_tenant_id_from_request
from .service import CompositionGatewayService

# Type aliases for dependency injection signatures
try:
	from sqlalchemy.ext.asyncio import AsyncSession
except ImportError:
	AsyncSession = Any  # type: ignore[misc,assignment]

# ASMService forward reference — resolved at runtime from app.state
ASMService = Any  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process runtime state store (keyed by tenant_id -> entity_type -> id)
# ---------------------------------------------------------------------------

gateway_runtime_state: Dict[str, Any] = {}


def _coerce_value(value: Any) -> Any:
	"""Recursively coerce enum values to their .value string."""
	if hasattr(value, "value"):
		return value.value
	if isinstance(value, dict):
		return {k: _coerce_value(v) for k, v in value.items()}
	if isinstance(value, list):
		return [_coerce_value(v) for v in value]
	return value


def _store_runtime_item(
	tenant_id: str,
	entity_type: str,
	item_id: str,
	data: Dict[str, Any],
	created_by: str,
) -> Dict[str, Any]:
	"""Upsert an item in the in-process runtime state and return the stored record."""
	tenant_bucket = gateway_runtime_state.setdefault(tenant_id, {})
	entity_bucket = tenant_bucket.setdefault(entity_type, {})

	now = datetime.now(timezone.utc).isoformat()
	existing = entity_bucket.get(item_id, {})

	record: Dict[str, Any] = {
		**{k: _coerce_value(v) for k, v in data.items()},
		"created_by": created_by,
		"created_at": existing.get("created_at", now),
		"updated_at": now,
	}
	entity_bucket[item_id] = record
	return record


def _list_runtime_items(
	tenant_id: str,
	entity_type: str,
	**filters: Any,
) -> Dict[str, Any]:
	"""Return paginated items from runtime state, optionally filtered by field equality."""
	bucket = (
		gateway_runtime_state
		.get(tenant_id, {})
		.get(entity_type, {})
	)
	items = list(bucket.values())
	for field_name, field_value in filters.items():
		items = [item for item in items if item.get(field_name) == field_value]
	return {"total": len(items), "items": items}


def _record_runtime_health_check(
	tenant_id: str,
	service_id: str,
	status: str,
	metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""Record a health-check event for a service in the runtime state."""
	from uuid import uuid4
	health_check_id = str(uuid4())
	record: Dict[str, Any] = {
		"health_check_id": health_check_id,
		"service_id": service_id,
		"status": status,
		"metadata": metadata or {},
		"checked_at": datetime.now(timezone.utc).isoformat(),
	}
	tenant_bucket = gateway_runtime_state.setdefault(tenant_id, {})
	health_bucket = tenant_bucket.setdefault("health_checks", {})
	health_bucket[health_check_id] = record
	return record


@asynccontextmanager
async def lifespan(app: Any) -> AsyncGenerator[None, None]:
	"""Application lifespan context manager."""
	yield


# ---------------------------------------------------------------------------
# FastAPI dependency helpers
# ---------------------------------------------------------------------------

async def get_db_session(request: Request) -> AsyncSession:
	"""Yield an async DB session from app.state.db_session."""
	session = await _resolve_app_state_dependency(request, ("db_session",), ())
	if session is None:
		raise HTTPException(status_code=503, detail="Database session is not configured")
	return session


async def get_asm_service(request: Request) -> ASMService:
	"""Resolve the ASM service instance from app.state."""
	service = await _resolve_app_state_dependency(
		request,
		("asm_service",),
		("asm_service_factory",),
	)
	if service is None:
		raise HTTPException(status_code=503, detail="ASM service provider is not configured")
	return service


async def _resolve_app_state_dependency(
	request: Any,
	attr_names: tuple[str, ...],
	factory_attr_names: tuple[str, ...],
) -> Any:
	"""Resolve a dependency from app.state, trying attrs then factory callables."""
	state = getattr(getattr(request, "app", None), "state", None)
	for attr in attr_names:
		value = getattr(state, attr, None)
		if value is not None:
			return value
	for factory_attr in factory_attr_names:
		factory = getattr(state, factory_attr, None)
		if factory is not None:
			return factory()
	return None


async def get_tenant_id(request: Request) -> str:
	"""FastAPI dependency: resolve tenant ID from the current request."""
	return get_tenant_id_from_request(request)


async def get_user_id(request: Request) -> str:
	"""FastAPI dependency: resolve user ID from the current request."""
	return get_current_user_id_from_request(request)


# =============================================================================
# Service Management Endpoints
# =============================================================================

SERVICE = CompositionGatewayService()


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
		"service_count": summary["service_count"],
		"mesh_route_count": summary["route_count"],
		"policy_count": summary["policy_count"],
		"gateway_agent_count": summary["gateway_agent_count"],
		"audit_event_count": summary["audit_event_count"],
		"streaming": summary["streaming"],
	}


def register_service(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_service(
		service_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner_id=str(payload["owner_id"]),
		endpoints=list(payload.get("endpoints") or []),
		health_check_path=str(payload.get("health_check_path") or ""),
		capability_id=str(payload.get("capability_id") or "composition_gateway"),
		public_service=bool(payload.get("public_service", False)),
		metadata=dict(payload.get("metadata") or {}),
	)


def create_route(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_route(
		route_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		service_id=str(payload["service_id"]),
		path=str(payload["path"]),
		methods=list(payload.get("methods") or ["GET"]),
		public_route=bool(payload.get("public_route", False)),
		policy_id=payload.get("policy_id"),
		approved_by=payload.get("approved_by"),
		tls_enabled=bool(payload.get("tls_enabled", False)),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def attach_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.attach_policy(
		policy_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		service_id=str(payload["service_id"]),
		rate_limit_configured=bool(payload.get("rate_limit_configured", False)),
		circuit_breaker_configured=bool(payload.get("circuit_breaker_configured", False)),
		owner_id=str(payload["owner_id"]),
	)


def shift_traffic(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.shift_traffic(
		shift_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		route_id=str(payload["route_id"]),
		weights=dict(payload.get("weights") or {}),
		actor_id=str(payload["actor_id"]),
		canary_shift=bool(payload.get("canary_shift", False)),
		canary_evidence=payload.get("canary_evidence"),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def register_certificate(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_certificate(
		certificate_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		domain=str(payload["domain"]),
		owner_id=str(payload["owner_id"]),
		secret_reference=str(payload.get("secret_reference") or ""),
		expires_at=str(payload["expires_at"]),
	)


def register_gateway_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_gateway_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		instructions=str(payload.get("instructions") or ""),
	)


def validate_agent_gateway_action(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_agent_gateway_action(
		tenant_id=str(payload.get("tenant_id") or "default"),
		agent_id=str(payload["agent_id"]),
		action=str(payload.get("action") or "review"),
		privileged_scope=bool(payload.get("privileged_scope", False)),
		human_approval_recorded=bool(payload.get("human_approval_recorded", False)),
	)


def validate_batch_route_change(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_batch_route_change(
		tenant_id=str(payload.get("tenant_id") or "default"),
		route_count=int(payload.get("route_count") or 0),
		event_stream=str(payload.get("event_stream") or "bytewax"),
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
		"services": SERVICE.list_services(tenant_id),
		"routes": SERVICE.list_routes(tenant_id),
		"policies": SERVICE.list_policies(tenant_id),
		"certificates": SERVICE.list_certificates(tenant_id),
		"traffic_shifts": SERVICE.list_traffic_shifts(tenant_id),
		"agents": SERVICE.list_gateway_agents(tenant_id),
		"audit_events": SERVICE.audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}



async def api_create_route(
	payload: Dict[str, Any],
	tenant_id: str,
	user_id: str,
) -> Dict[str, Any]:
	"""Create a mesh route and persist it in runtime state."""
	item_id = str(payload.get("route_id") or payload.get("id", ""))
	return _store_runtime_item(
		tenant_id,
		"routes",
		item_id,
		payload,
		created_by=user_id,
	)


async def api_create_policy(
	payload: Dict[str, Any],
	tenant_id: str,
	user_id: str,
) -> Dict[str, Any]:
	"""Create a gateway policy and persist it in runtime state."""
	item_id = str(payload.get("policy_id") or payload.get("id", ""))
	record = _store_runtime_item(
		tenant_id,
		"policies",
		item_id,
		payload,
		created_by=user_id,
	)
	record["updated_by"] = user_id
	return record


async def api_create_load_balancer(
	payload: Dict[str, Any],
	tenant_id: str,
	user_id: str,
) -> Dict[str, Any]:
	"""Create a load balancer entry in runtime state."""
	item_id = str(payload.get("load_balancer_id") or payload.get("id", ""))
	record = _store_runtime_item(
		tenant_id,
		"load_balancers",
		item_id,
		payload,
		created_by=user_id,
	)
	record["updated_by"] = user_id
	return record


async def api_trigger_health_check(
	tenant_id: str,
	service_id: str,
	user_id: str,
	metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""Trigger a health check and record it in runtime state."""
	_ = user_id  # audit context; updated_by=user_id captured at call site
	return _record_runtime_health_check(tenant_id, service_id, "queued", metadata)


async def api_list_routes(tenant_id: str) -> Dict[str, Any]:
	"""List all routes for a tenant from runtime state."""
	return _list_runtime_items(tenant_id, "routes")


async def api_list_policies(
	tenant_id: str,
	policy_type: Optional[str] = None,
) -> Dict[str, Any]:
	"""List policies for a tenant, optionally filtered by type."""
	if policy_type is not None:
		return _list_runtime_items(tenant_id, "policies", policy_type=policy_type)
	return _list_runtime_items(tenant_id, "policies")
