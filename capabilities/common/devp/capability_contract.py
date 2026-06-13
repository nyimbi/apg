"""Executable capability contract for APG Developer Portal."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "common_devp"
CAPABILITY_NAME = "Developer Portal"
CAPABILITY_VERSION = "1.0.0"
DEVP_EVENT_STREAM = "apg.common.devp.lifecycle"

SUPPORTED_KEY_STATUSES = ["active", "revoked", "suspended", "expired"]
SUPPORTED_SUBSCRIPTION_STATUSES = ["pending", "active", "suspended", "cancelled"]
SUPPORTED_PLAN_TYPES = ["free", "starter", "professional", "enterprise", "custom"]
SUPPORTED_WEBHOOK_EVENTS = [
	"api_key.created",
	"api_key.revoked",
	"api_key.suspended",
	"subscription.activated",
	"subscription.cancelled",
	"usage.quota_warning",
	"usage.quota_exceeded",
]
SUPPORTED_RATE_LIMIT_WINDOWS = ["second", "minute", "hour", "day", "month"]
PUBLISHES = ["api_key.created", "api_key.revoked", "subscription.activated"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"api_keys": {
		"supported_statuses": SUPPORTED_KEY_STATUSES,
		"key_length_bytes": 32,
		"hash_algorithm": "sha256",
		"default_rate_limit": {"requests_per_minute": 60, "requests_per_day": 10000},
		"require_app_binding": True,
	},
	"api_products": {
		"supported_plan_types": SUPPORTED_PLAN_TYPES,
		"require_capability_list": True,
		"require_endpoint_list": True,
	},
	"developer_apps": {
		"require_owner": True,
		"max_keys_per_app": 5,
		"max_apps_per_tenant": 100,
	},
	"subscriptions": {
		"supported_statuses": SUPPORTED_SUBSCRIPTION_STATUSES,
		"require_product": True,
		"require_developer_app": True,
	},
	"webhooks": {
		"supported_events": SUPPORTED_WEBHOOK_EVENTS,
		"require_https": True,
		"require_secret": True,
		"max_endpoints_per_app": 10,
	},
	"usage_stats": {
		"retention_days": 90,
		"aggregation_intervals": ["1h", "1d", "7d", "30d"],
		"track_per_endpoint": True,
		"track_errors": True,
		"track_latency": True,
	},
	"openapi_browser": {
		"enabled": True,
		"proxy_to_capability": True,
		"cache_ttl_seconds": 300,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_events": True,
		"cross_tenant_key_denied": True,
		"key_exposure_denied": True,
		"policy_attached_for_writes": True,
	},
	"observability": {
		"event_stream": DEVP_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"api_gateway": "common_apig",
		"billing": "common_sbl",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_api_products": True,
		"enable_api_keys": True,
		"enable_developer_apps": True,
		"enable_subscriptions": True,
		"enable_usage_analytics": True,
		"enable_openapi_browser": True,
		"enable_webhooks": True,
	},
	"theme": {
		"default_theme": "devp_portal",
		"allow_tenant_overrides": True,
	},
}

PROVIDES = [
	"api_key_management",
	"developer_onboarding",
	"usage_analytics",
	"openapi_browser",
	"webhook_management",
]
REQUIRES = ["auth", "audl", "ntfy", "common_apig", "common_sbl"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/devp/dashboard", "component": "DevpDashboard", "permission": "devp:view", "nav_group": "Overview"},
	{"name": "api_products", "path": "/devp/products", "component": "DevpApiProductCatalog", "permission": "devp:products", "nav_group": "Catalog"},
	{"name": "openapi_browser", "path": "/devp/openapi", "component": "DevpOpenApiBrowser", "permission": "devp:products", "nav_group": "Catalog"},
	{"name": "developer_apps", "path": "/devp/apps", "component": "DevpAppConsole", "permission": "devp:apps", "nav_group": "My Apps"},
	{"name": "api_keys", "path": "/devp/keys", "component": "DevpApiKeyManager", "permission": "devp:keys", "nav_group": "My Apps"},
	{"name": "subscriptions", "path": "/devp/subscriptions", "component": "DevpSubscriptionConsole", "permission": "devp:subscriptions", "nav_group": "My Apps"},
	{"name": "usage_analytics", "path": "/devp/usage", "component": "DevpUsageAnalytics", "permission": "devp:usage", "nav_group": "Analytics"},
	{"name": "webhooks", "path": "/devp/webhooks", "component": "DevpWebhookConsole", "permission": "devp:webhooks", "nav_group": "My Apps"},
	{"name": "settings", "path": "/devp/settings", "component": "DevpSettings", "permission": "devp:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "devp_portal",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#0891B2",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F8FAFC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"api_keys": {"icon": "key", "status_indicator": "key-status-chip"},
		"api_products": {"icon": "package", "status_indicator": "plan-chip"},
		"developer_apps": {"icon": "code-2", "status_indicator": "app-status-chip"},
		"subscriptions": {"icon": "layers", "status_indicator": "subscription-chip"},
		"usage": {"icon": "bar-chart-2", "status_indicator": "usage-gauge"},
		"webhooks": {"icon": "webhook", "status_indicator": "webhook-chip"},
		"openapi": {"icon": "file-code", "status_indicator": "spec-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": DEVP_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"api_key_created",
		"api_key_revoked",
		"api_key_suspended",
		"developer_app_created",
		"developer_app_updated",
		"subscription_activated",
		"subscription_cancelled",
		"webhook_registered",
		"webhook_deleted",
		"usage_quota_warning",
		"usage_quota_exceeded",
	],
	"guardrails": [
		"cross_tenant_key_access_denied",
		"key_plaintext_exposure_denied",
		"write_requires_policy",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "devp_policy_required", "required_action": "attach_devp_policy"}},
	{"name": "cross_tenant_key_denied", "condition": {"operation": "validate_api_key", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_key_access_denied", "required_action": "use_tenant_scoped_key"}},
	{"name": "key_plaintext_denied", "condition": {"operation": "get_api_key", "return_plaintext": True}, "effect": {"decision": "deny", "reason": "key_plaintext_exposure_denied", "required_action": "use_key_hash_only"}},
	{"name": "app_required_for_key", "condition": {"operation": "create_api_key", "app_present": False}, "effect": {"decision": "deny", "reason": "developer_app_required", "required_action": "create_or_select_developer_app"}},
	{"name": "scopes_required_for_key", "condition": {"operation": "create_api_key", "scopes_present": False}, "effect": {"decision": "deny", "reason": "key_scopes_required", "required_action": "specify_key_scopes"}},
	{"name": "product_required_for_subscription", "condition": {"operation": "subscribe_to_product", "product_present": False}, "effect": {"decision": "deny", "reason": "api_product_required", "required_action": "select_api_product"}},
	{"name": "webhook_https_required", "condition": {"operation": "register_webhook", "url_scheme_ne": "https"}, "effect": {"decision": "deny", "reason": "webhook_https_required", "required_action": "use_https_webhook_url"}},
	{"name": "webhook_secret_required", "condition": {"operation": "register_webhook", "secret_present": False}, "effect": {"decision": "deny", "reason": "webhook_secret_required", "required_action": "provide_webhook_secret"}},
	{"name": "webhook_events_required", "condition": {"operation": "register_webhook", "events_present": False}, "effect": {"decision": "deny", "reason": "webhook_event_list_required", "required_action": "specify_webhook_events"}},
	{"name": "revoke_requires_ownership", "condition": {"operation": "revoke_api_key", "owner_match": False}, "effect": {"decision": "deny", "reason": "key_ownership_required_for_revocation", "required_action": "use_owned_key_or_admin_role"}},
	{"name": "stats_require_key_ownership", "condition": {"operation": "get_usage_stats", "owner_match": False}, "effect": {"decision": "deny", "reason": "key_ownership_required_for_stats", "required_action": "use_owned_key_or_admin_role"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"name": CAPABILITY_NAME,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"publishes": list(PUBLISHES),
		"configuration": configuration,
		"configuration_schema": {
			"type": "object",
			"required": list(configuration),
			"properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}},
		},
		"rule_engine": {
			"type": "deterministic",
			"default_decision": "allow",
			"rules": deepcopy(RULES),
		},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/devp/api/v1",
			"requires_theme": True,
			"view_module": "views.py",
			"template_roots": ["templates/", "static/"],
			"routes": deepcopy(UI_ROUTES),
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions: list[dict[str, Any]] = []
	for rule in RULES:
		if _matches(rule["condition"], context):
			actions.append(rule["effect"] | {"rule": rule["name"]})
	if not actions:
		return {"decision": "allow", "actions": [], "context": dict(context)}
	return {"decision": "deny", "actions": actions, "context": dict(context)}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True
