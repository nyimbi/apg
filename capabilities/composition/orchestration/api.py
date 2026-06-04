"""Dependency-light API helpers for APG workflow orchestration."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.requests import Request

try:
	from .capability_contract import get_capability_contract
	from .service import WorkflowOrchestrationService
except ImportError:
	from capability_contract import get_capability_contract
	from service import WorkflowOrchestrationService

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
	import base64 as _base64
	try:
		parts = token.split(".")
		if len(parts) < 2:
			return None
		payload = parts[1]
		padding = "=" * (-len(payload) % 4)
		data = _base64.urlsafe_b64decode(f"{payload}{padding}".encode("ascii"))
		return json.loads(data.decode("utf-8"))
	except Exception:
		return None


async def get_current_user(
	request: Request,
	credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
	"""Resolve current user from JWT claims, headers, query params, or environment."""
	# 1. Try JWT bearer claims
	if credentials and credentials.credentials:
		claims = _decode_jwt_claims(credentials.credentials)
		if claims:
			user_id = _clean_text(claims.get("sub") or claims.get("user_id"))
			tenant_id = _clean_text(claims.get("tenant_id") or claims.get("org_id"))
			roles = claims.get("roles") or []
			permissions = claims.get("permissions") or []
			if user_id:
				return {
					"user_id": user_id,
					"tenant_id": tenant_id or os.getenv("APG_DEFAULT_TENANT_ID", "default"),
					"roles": roles,
					"permissions": permissions,
				}

	# 2. Fallback: headers, query params, environment
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

	raw_permissions = _hget("X-APG-Permissions", "X-Permissions")
	permissions_list: List[str] = raw_permissions.split() if raw_permissions else []

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
		"roles": [],
		"permissions": permissions_list,
	}


async def get_tenant_id(
	request: Request,
	current_user: Dict[str, Any] = Depends(get_current_user),
) -> str:
	"""FastAPI dependency: resolve tenant ID from the current request context."""
	_ = request
	return str(current_user.get("tenant_id") or "default")


_SERVICE = WorkflowOrchestrationService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	"""Return service status and contract metadata for generated applications."""
	contract = get_capability_contract(tenant_id)
	return {
		"ok": True,
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"provides": contract["provides"],
		"requires": contract["requires"],
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"streaming": contract["streaming"],
		"summary": _SERVICE.dashboard_summary(tenant_id),
	}


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	"""Create a workflow definition using the package service."""
	return _SERVICE.create_record(payload)


def list_records(tenant_id: str = "default") -> list[dict[str, Any]]:
	"""List workflow definitions for a tenant."""
	return _SERVICE.list_records(tenant_id)


def define_workflow(payload: dict[str, Any]) -> dict[str, Any]:
	"""Define and validate a workflow graph."""
	return _SERVICE.define_workflow(
		payload["workflow_id"],
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["owner"],
		payload.get("version", "1.0.0"),
		payload["tasks"],
		payload.get("start_event", "manual"),
		payload.get("terminal_state", "completed"),
		transactional=payload.get("transactional", False),
		compensation_steps=payload.get("compensation_steps"),
	)


def release_workflow(payload: dict[str, Any]) -> dict[str, Any]:
	"""Release a validated workflow definition."""
	return _SERVICE.release_workflow(
		payload["release_id"],
		payload.get("tenant_id", "default"),
		payload["workflow_definition_id"],
		payload["validation_evidence"],
		payload["rollback_plan"],
		dry_run_passed=payload.get("dry_run_passed", False),
		approved_by=payload.get("approved_by"),
	)


def start_execution(payload: dict[str, Any]) -> dict[str, Any]:
	"""Start a workflow execution with an idempotency key."""
	return _SERVICE.start_execution(
		payload["execution_id"],
		payload.get("tenant_id", "default"),
		payload["workflow_definition_id"],
		payload["idempotency_key"],
		payload.get("inputs"),
		risk_level=payload.get("risk_level", "normal"),
		reviewed_by=payload.get("reviewed_by"),
	)


def complete_task(payload: dict[str, Any]) -> dict[str, Any]:
	"""Complete an active task and advance the execution graph."""
	return _SERVICE.complete_task(
		payload.get("tenant_id", "default"),
		payload["execution_record_id"],
		payload["task_id"],
		payload.get("result"),
	)


def register_workflow_agent(payload: dict[str, Any]) -> dict[str, Any]:
	"""Register an AI agent runtime for orchestration review work."""
	return _SERVICE.register_workflow_agent(
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("instructions", ""),
	)


def service() -> WorkflowOrchestrationService:
	"""Return the in-process service used by generated application adapters."""
	return _SERVICE
