"""Screen-model helpers for Integration API Management."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import IntApiService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import get_capability_contract  # type: ignore
	from service import IntApiService  # type: ignore


NAVIGATION = [
	{"name": "Dashboard", "route": "/int-api/dashboard", "icon": "layout-dashboard"},
	{"name": "APIs", "route": "/int-api/apis", "icon": "network"},
	{"name": "Endpoints", "route": "/int-api/endpoints", "icon": "route"},
	{"name": "Policies", "route": "/int-api/policies", "icon": "shield-check"},
	{"name": "Consumers", "route": "/int-api/consumers", "icon": "users"},
	{"name": "Keys", "route": "/int-api/keys", "icon": "key-round"},
	{"name": "Subscriptions", "route": "/int-api/subscriptions", "icon": "badge-check"},
	{"name": "Deployments", "route": "/int-api/deployments", "icon": "rocket"},
	{"name": "Analytics", "route": "/int-api/analytics", "icon": "chart-line"},
	{"name": "Agents", "route": "/int-api/agents", "icon": "bot"},
	{"name": "Settings", "route": "/int-api/settings", "icon": "settings"},
]


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def _base(screen: str, tenant_id: str) -> dict[str, Any]:
	return {"screen": screen, "tenant_id": tenant_id, "navigation": NAVIGATION}


def dashboard_model(service: IntApiService, tenant_id: str) -> dict[str, Any]:
	model = _base("dashboard", tenant_id)
	model["summary"] = service.dashboard_summary(tenant_id)
	model["work_queue"] = {
		"draft_apis": len([record for record in service.apis.values() if record["tenant_id"] == tenant_id and record["status"] == "draft"]),
		"deployed_apis": len([record for record in service.apis.values() if record["tenant_id"] == tenant_id and record["status"] == "deployed"]),
		"slow_requests": len([record for record in service.usage_records.values() if record["tenant_id"] == tenant_id and record["latency_ms"] >= 2000]),
		"production_deployments": len([record for record in service.deployments.values() if record["tenant_id"] == tenant_id and record["environment"] == "prod"]),
	}
	return model


def api_registry_model(service: IntApiService, tenant_id: str) -> dict[str, Any]:
	model = _base("apis", tenant_id)
	model["records"] = service.list_records("apis", tenant_id)
	model["columns"] = ["name", "title", "base_path", "protocol", "auth_type", "rate_limit_per_minute", "status"]
	return model


def endpoint_model(service: IntApiService, tenant_id: str) -> dict[str, Any]:
	model = _base("endpoints", tenant_id)
	model["records"] = service.list_records("endpoints", tenant_id)
	model["columns"] = ["api_id", "method", "path", "auth_required", "rate_limit_override", "status"]
	return model


def policy_model(service: IntApiService, tenant_id: str) -> dict[str, Any]:
	model = _base("policies", tenant_id)
	model["records"] = service.list_records("policies", tenant_id)
	model["columns"] = ["api_id", "policy_type", "name", "execution_order", "status"]
	return model


def consumer_model(service: IntApiService, tenant_id: str) -> dict[str, Any]:
	model = _base("consumers", tenant_id)
	model["records"] = service.list_records("consumers", tenant_id)
	model["columns"] = ["name", "contact_email", "owner_id", "external", "status"]
	return model


def key_model(service: IntApiService, tenant_id: str) -> dict[str, Any]:
	model = _base("keys", tenant_id)
	model["records"] = service.list_records("api_keys", tenant_id)
	model["columns"] = ["consumer_id", "name", "key_prefix", "scopes", "expires_on", "status"]
	return model


def subscription_model(service: IntApiService, tenant_id: str) -> dict[str, Any]:
	model = _base("subscriptions", tenant_id)
	model["records"] = service.list_records("subscriptions", tenant_id)
	model["columns"] = ["consumer_id", "api_id", "plan", "approved_by", "status"]
	return model


def deployment_model(service: IntApiService, tenant_id: str) -> dict[str, Any]:
	model = _base("deployments", tenant_id)
	model["records"] = service.list_records("deployments", tenant_id)
	model["columns"] = ["api_id", "environment", "gateway_route", "deployed_by", "approved_by", "status"]
	return model


def analytics_model(service: IntApiService, tenant_id: str) -> dict[str, Any]:
	model = _base("analytics", tenant_id)
	model["records"] = service.list_records("usage_records", tenant_id)
	model["columns"] = ["api_id", "consumer_id", "endpoint_id", "status_code", "latency_ms", "status"]
	return model


def agent_workbench_model(service: IntApiService, tenant_id: str) -> dict[str, Any]:
	model = _base("agents", tenant_id)
	model["records"] = service.list_records("agents", tenant_id)
	model["actions"] = ["review_api", "review_policy", "review_consumer", "review_deployment", "review_analytics"]
	return model


class APIConfigForm:
	"""Compatibility shim for older Flask-AppBuilder form imports."""


class PolicyConfigForm:
	"""Compatibility shim for older Flask-AppBuilder form imports."""


class ConsumerRegistrationForm:
	"""Compatibility shim for older Flask-AppBuilder form imports."""


class APIManagementView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class EndpointManagementView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class PolicyManagementView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class ConsumerManagementView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class APIKeyManagementView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class AnalyticsDashboardView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class UsageRecordsView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class DeveloperPortalView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class DeploymentManagementView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""
