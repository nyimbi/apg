"""Dependency-light API helpers for Integration API Management."""

from __future__ import annotations

from typing import Any

try:
	from .service import IntApiService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from service import IntApiService  # type: ignore


SERVICE = IntApiService()


def service() -> IntApiService:
	"""Return the process-local API management service."""
	return SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"ok": True,
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}


def register_api(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_api(
		payload.get("api_id", payload.get("id", "api")),
		payload["tenant_id"],
		payload["name"],
		payload["title"],
		payload["base_path"],
		payload["upstream_url"],
		payload["owner_id"],
		payload.get("version", "1.0.0"),
		payload.get("protocol", "rest"),
		payload.get("auth_type", "api_key"),
		int(payload.get("rate_limit_per_minute", 1000)),
		payload.get("reviewed_by"),
		payload.get("metadata", {}),
	)


def register_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_endpoint(
		payload.get("endpoint_id", payload.get("id", "endpoint")),
		payload["tenant_id"],
		payload["api_id"],
		payload["path"],
		payload["method"],
		bool(payload.get("auth_required", True)),
		payload.get("rate_limit_override"),
	)


def attach_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.attach_policy(
		payload.get("policy_id", payload.get("id", "policy")),
		payload["tenant_id"],
		payload["api_id"],
		payload["policy_type"],
		payload["name"],
		dict(payload.get("config") or {}),
		int(payload.get("execution_order", 100)),
	)


def register_consumer(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_consumer(
		payload.get("consumer_id", payload.get("id", "consumer")),
		payload["tenant_id"],
		payload["name"],
		payload["contact_email"],
		payload["owner_id"],
		bool(payload.get("external", False)),
		payload.get("reviewed_by"),
	)


def issue_api_key(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.issue_api_key(
		payload.get("key_id", payload.get("id", "key")),
		payload["tenant_id"],
		payload["consumer_id"],
		payload["name"],
		list(payload.get("scopes") or []),
		payload["expires_on"],
	)


def create_subscription(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_subscription(
		payload.get("subscription_id", payload.get("id", "subscription")),
		payload["tenant_id"],
		payload["consumer_id"],
		payload["api_id"],
		payload.get("plan", "standard"),
		payload["approved_by"],
	)


def approve_api(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_api(payload["api_id"], payload["tenant_id"], payload["approved_by"])


def deploy_api(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.deploy_api(
		payload.get("deployment_id", payload.get("id", "deployment")),
		payload["tenant_id"],
		payload["api_id"],
		payload.get("environment", "stage"),
		payload["gateway_route"],
		payload["deployed_by"],
		payload.get("approved_by"),
	)


def record_usage(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_usage(
		payload.get("usage_id", payload.get("id", "usage")),
		payload["tenant_id"],
		payload["api_id"],
		payload.get("consumer_id"),
		payload.get("endpoint_id"),
		int(payload["status_code"]),
		int(payload["latency_ms"]),
		payload.get("reviewed_by"),
	)


def register_api_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_api_agent(
		payload["tenant_id"],
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("scope", "review API management operations"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	"""Generic composition helper used by APG package probes."""
	return SERVICE.create_record(
		str(payload.get("id", "api-record")),
		str(payload.get("tenant_id") or "default"),
		{
			"name": payload.get("name", "api-record"),
			"title": payload.get("title", "API Record"),
			"base_path": payload.get("base_path", "/api-record"),
			"upstream_url": payload.get("upstream_url", "internal://service"),
			"owner_id": payload.get("owner_id", "api-owner"),
			"protocol": payload.get("protocol", "rest"),
			"auth_type": payload.get("auth_type", "api_key"),
			"rate_limit_per_minute": payload.get("rate_limit_per_minute", 1000),
			"reviewed_by": payload.get("reviewed_by"),
		},
		str(payload.get("status") or "draft"),
	)


def list_records(collection: str | None = None, tenant_id: str = "default") -> list[dict[str, Any]]:
	return SERVICE.list_records(collection, tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)


class APIManagementApi:
	"""Compatibility shim for Flask-AppBuilder endpoint registration."""


class ConsumerManagementApi:
	"""Compatibility shim for Flask-AppBuilder endpoint registration."""


class AnalyticsApi:
	"""Compatibility shim for Flask-AppBuilder endpoint registration."""


class GatewayApi:
	"""Compatibility shim for gateway endpoint registration."""


def register_api_endpoints(*_: Any, **__: Any) -> None:
	"""Compatibility hook for older Flask-AppBuilder setup code."""
	return None
