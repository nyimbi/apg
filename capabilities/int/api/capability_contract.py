"""Executable capability contract for Integration API Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "int_api"
CAPABILITY_NAME = "Integration API Management"
CAPABILITY_VERSION = "2.1.0"
API_EVENT_STREAM = "apg.int.api.lifecycle"

SUPPORTED_PROTOCOLS = ["rest", "graphql", "grpc", "webhook"]
SUPPORTED_AUTH_TYPES = ["api_key", "oauth2", "jwt", "mtls", "none"]
SUPPORTED_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
SUPPORTED_POLICY_TYPES = ["rate_limit", "quota", "auth", "transform", "cors", "ip_filter", "circuit_breaker"]
SUPPORTED_PLANS = ["sandbox", "standard", "premium", "internal"]
SUPPORTED_ENVIRONMENTS = ["dev", "test", "stage", "prod"]
SUPPORTED_API_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_API_AGENT_ROLES = [
	"api_designer",
	"policy_reviewer",
	"security_reviewer",
	"consumer_reviewer",
	"deployment_reviewer",
	"analytics_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"apis": {
		"name_required": True,
		"title_required": True,
		"base_path_required": True,
		"upstream_required": True,
		"owner_required": True,
		"supported_protocols": SUPPORTED_PROTOCOLS,
		"supported_auth_types": SUPPORTED_AUTH_TYPES,
		"default_rate_limit_minimum": 1,
		"external_upstream_review_required": True,
	},
	"endpoints": {
		"api_required": True,
		"path_required": True,
		"supported_methods": SUPPORTED_METHODS,
		"auth_required_by_default": True,
	},
	"policies": {
		"supported_policy_types": SUPPORTED_POLICY_TYPES,
		"name_required": True,
		"config_required": True,
		"execution_order_required": True,
	},
	"consumers": {
		"name_required": True,
		"contact_email_required": True,
		"owner_required": True,
		"external_consumer_review_required": True,
	},
	"api_keys": {
		"consumer_required": True,
		"name_required": True,
		"scope_required": True,
		"expiration_required": True,
	},
	"subscriptions": {
		"consumer_required": True,
		"api_required": True,
		"supported_plans": SUPPORTED_PLANS,
		"approval_required": True,
	},
	"deployments": {
		"api_required": True,
		"supported_environments": SUPPORTED_ENVIRONMENTS,
		"route_required": True,
		"deployer_required": True,
		"production_approval_required": True,
	},
	"analytics": {
		"api_required": True,
		"status_code_required": True,
		"latency_required": True,
		"latency_review_threshold_ms": 2000,
	},
	"api_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_API_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_API_AGENT_ROLES,
		"max_autonomous_scope": "review_prepare_and_recommend",
		"human_approval_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_state_changes": True,
		"segregation_of_duties": True,
	},
	"observability": {
		"event_stream": API_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_api_events": True,
		"emit_endpoint_events": True,
		"emit_policy_events": True,
		"emit_consumer_events": True,
		"emit_key_events": True,
		"emit_subscription_events": True,
		"emit_deployment_events": True,
		"emit_analytics_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"notification": "adapter",
		"gateway": "adapter",
		"developer_portal": "adapter",
		"regy": "adapter",
		"policy_management": "adapter",
		"analytics_sink": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_apis": True,
		"enable_endpoints": True,
		"enable_policies": True,
		"enable_consumers": True,
		"enable_keys": True,
		"enable_subscriptions": True,
		"enable_deployments": True,
		"enable_analytics": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {
		"default_theme": "int_api_control",
		"allow_tenant_overrides": True,
	},
}


PROVIDES = [
	"api_registry_lifecycle",
	"api_endpoint_lifecycle",
	"api_policy_lifecycle",
	"api_consumer_lifecycle",
	"api_key_lifecycle",
	"api_subscription_lifecycle",
	"api_deployment_workflow",
	"api_gateway_route_catalog",
	"api_analytics_workflow",
	"api_dashboard_service",
	"api_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"composition_events",
	"composition_config",
	"grc_pol",
	"regy",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/int-api/dashboard", "component": "ApiManagementDashboard", "permission": "int_api:view", "nav_group": "Overview"},
	{"name": "apis", "path": "/int-api/apis", "component": "ApiRegistryWorkbench", "permission": "int_api:manage_apis", "nav_group": "APIs"},
	{"name": "endpoints", "path": "/int-api/endpoints", "component": "EndpointWorkbench", "permission": "int_api:manage_endpoints", "nav_group": "APIs"},
	{"name": "policies", "path": "/int-api/policies", "component": "PolicyWorkbench", "permission": "int_api:manage_policies", "nav_group": "Governance"},
	{"name": "consumers", "path": "/int-api/consumers", "component": "ConsumerWorkbench", "permission": "int_api:manage_consumers", "nav_group": "Consumers"},
	{"name": "keys", "path": "/int-api/keys", "component": "ApiKeyWorkbench", "permission": "int_api:manage_keys", "nav_group": "Consumers"},
	{"name": "subscriptions", "path": "/int-api/subscriptions", "component": "SubscriptionWorkbench", "permission": "int_api:manage_subscriptions", "nav_group": "Consumers"},
	{"name": "deployments", "path": "/int-api/deployments", "component": "DeploymentWorkbench", "permission": "int_api:deploy", "nav_group": "Gateway"},
	{"name": "analytics", "path": "/int-api/analytics", "component": "ApiAnalyticsDashboard", "permission": "int_api:view_analytics", "nav_group": "Analytics"},
	{"name": "agents", "path": "/int-api/agents", "component": "ApiAgentWorkbench", "permission": "int_api:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/int-api/settings", "component": "ApiManagementSettings", "permission": "int_api:admin", "nav_group": "Administration"},
]


THEME = {
	"name": "int_api_control",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#6B5B95",
		"color.success": "#237A57",
		"color.warning": "#B7791F",
		"color.danger": "#B42318",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"apis": {"icon": "network", "status_indicator": "api-pill", "visual": "registry-list"},
		"endpoints": {"visual": "route-table", "status_style": "method-chip"},
		"policies": {"visual": "policy-stack", "status_style": "policy-chip"},
		"consumers": {"visual": "consumer-list", "status_style": "consumer-chip"},
		"keys": {"visual": "key-ledger", "status_style": "key-chip"},
		"subscriptions": {"visual": "subscription-board", "status_style": "plan-chip"},
		"deployments": {"visual": "deployment-lane", "status_style": "environment-chip"},
		"analytics": {"visual": "metrics-grid", "status_style": "sla-chip"},
		"agents": {"visual": "review-lane", "status_style": "agent-chip"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"stream": API_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"api_registered",
		"endpoint_registered",
		"policy_attached",
		"consumer_registered",
		"api_key_issued",
		"subscription_created",
		"api_approved",
		"api_deployed",
		"usage_recorded",
		"api_agent_registered",
	],
	"states": ["draft", "active", "approved", "deployed", "suspended", "revoked", "queued", "blocked"],
	"guardrails": [
		"api_batch_requires_bytewax",
		"api_event_requires_bytewax",
		"privileged_api_agent_action_requires_human_approval",
	],
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "API management operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "api_write_requires_policy", "description": "API management writes require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "api_requires_name", "description": "APIs require name.", "condition": {"operation": "register_api", "name_present": False}, "effect": {"decision": "deny", "reason": "api_name_required", "required_action": "set_api_name"}},
	{"name": "api_requires_title", "description": "APIs require title.", "condition": {"operation": "register_api", "title_present": False}, "effect": {"decision": "deny", "reason": "api_title_required", "required_action": "set_api_title"}},
	{"name": "api_requires_base_path", "description": "APIs require base path.", "condition": {"operation": "register_api", "base_path_present": False}, "effect": {"decision": "deny", "reason": "api_base_path_required", "required_action": "set_base_path"}},
	{"name": "api_base_path_format", "description": "API base path must start with slash.", "condition": {"operation": "register_api", "base_path_valid": False}, "effect": {"decision": "deny", "reason": "api_base_path_invalid", "required_action": "set_valid_base_path"}},
	{"name": "api_requires_upstream", "description": "APIs require upstream URL.", "condition": {"operation": "register_api", "upstream_present": False}, "effect": {"decision": "deny", "reason": "api_upstream_required", "required_action": "set_upstream_url"}},
	{"name": "api_requires_owner", "description": "APIs require owner.", "condition": {"operation": "register_api", "owner_present": False}, "effect": {"decision": "deny", "reason": "api_owner_required", "required_action": "assign_api_owner"}},
	{"name": "api_protocol_supported", "description": "API protocol must be supported.", "condition": {"operation": "register_api", "protocol_supported": False}, "effect": {"decision": "deny", "reason": "api_protocol_not_supported", "required_action": "select_supported_protocol"}},
	{"name": "api_auth_supported", "description": "API authentication type must be supported.", "condition": {"operation": "register_api", "auth_type_supported": False}, "effect": {"decision": "deny", "reason": "api_auth_type_not_supported", "required_action": "select_supported_auth_type"}},
	{"name": "api_rate_limit_positive", "description": "API rate limit must be positive.", "condition": {"operation": "register_api", "rate_limit_lte": 0}, "effect": {"decision": "deny", "reason": "api_rate_limit_required", "required_action": "set_rate_limit"}},
	{"name": "external_upstream_requires_review", "description": "External upstream APIs require review.", "condition": {"operation": "register_api", "external_upstream": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "external_upstream_review_required", "required_action": "record_upstream_review"}},
	{"name": "endpoint_requires_api", "description": "Endpoints require API.", "condition": {"operation": "register_endpoint", "api_present": False}, "effect": {"decision": "deny", "reason": "api_required", "required_action": "select_api"}},
	{"name": "endpoint_requires_path", "description": "Endpoints require path.", "condition": {"operation": "register_endpoint", "path_present": False}, "effect": {"decision": "deny", "reason": "endpoint_path_required", "required_action": "set_endpoint_path"}},
	{"name": "endpoint_path_format", "description": "Endpoint path must start with slash.", "condition": {"operation": "register_endpoint", "path_valid": False}, "effect": {"decision": "deny", "reason": "endpoint_path_invalid", "required_action": "set_valid_endpoint_path"}},
	{"name": "endpoint_method_supported", "description": "Endpoint method must be supported.", "condition": {"operation": "register_endpoint", "method_supported": False}, "effect": {"decision": "deny", "reason": "endpoint_method_not_supported", "required_action": "select_supported_method"}},
	{"name": "policy_requires_api", "description": "Policies require API.", "condition": {"operation": "attach_policy", "api_present": False}, "effect": {"decision": "deny", "reason": "api_required", "required_action": "select_api"}},
	{"name": "policy_requires_name", "description": "Policies require name.", "condition": {"operation": "attach_policy", "name_present": False}, "effect": {"decision": "deny", "reason": "policy_name_required", "required_action": "set_policy_name"}},
	{"name": "policy_type_supported", "description": "Policy type must be supported.", "condition": {"operation": "attach_policy", "policy_type_supported": False}, "effect": {"decision": "deny", "reason": "policy_type_not_supported", "required_action": "select_supported_policy_type"}},
	{"name": "policy_requires_config", "description": "Policies require configuration.", "condition": {"operation": "attach_policy", "config_present": False}, "effect": {"decision": "deny", "reason": "policy_config_required", "required_action": "set_policy_config"}},
	{"name": "policy_execution_order_nonnegative", "description": "Policy execution order must be nonnegative.", "condition": {"operation": "attach_policy", "execution_order_lt": 0}, "effect": {"decision": "deny", "reason": "policy_execution_order_invalid", "required_action": "set_execution_order"}},
	{"name": "consumer_requires_name", "description": "Consumers require name.", "condition": {"operation": "register_consumer", "name_present": False}, "effect": {"decision": "deny", "reason": "consumer_name_required", "required_action": "set_consumer_name"}},
	{"name": "consumer_requires_email", "description": "Consumers require contact email.", "condition": {"operation": "register_consumer", "email_present": False}, "effect": {"decision": "deny", "reason": "consumer_email_required", "required_action": "set_contact_email"}},
	{"name": "consumer_email_format", "description": "Consumer email must be valid.", "condition": {"operation": "register_consumer", "email_valid": False}, "effect": {"decision": "deny", "reason": "consumer_email_invalid", "required_action": "set_valid_email"}},
	{"name": "consumer_requires_owner", "description": "Consumers require owner.", "condition": {"operation": "register_consumer", "owner_present": False}, "effect": {"decision": "deny", "reason": "consumer_owner_required", "required_action": "assign_consumer_owner"}},
	{"name": "external_consumer_requires_review", "description": "External consumers require review.", "condition": {"operation": "register_consumer", "external_consumer": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "external_consumer_review_required", "required_action": "record_consumer_review"}},
	{"name": "api_key_requires_consumer", "description": "API keys require consumer.", "condition": {"operation": "issue_api_key", "consumer_present": False}, "effect": {"decision": "deny", "reason": "consumer_required", "required_action": "select_consumer"}},
	{"name": "api_key_requires_name", "description": "API keys require name.", "condition": {"operation": "issue_api_key", "name_present": False}, "effect": {"decision": "deny", "reason": "api_key_name_required", "required_action": "set_key_name"}},
	{"name": "api_key_requires_scope", "description": "API keys require scopes.", "condition": {"operation": "issue_api_key", "scope_present": False}, "effect": {"decision": "deny", "reason": "api_key_scope_required", "required_action": "set_key_scope"}},
	{"name": "api_key_requires_expiration", "description": "API keys require expiration.", "condition": {"operation": "issue_api_key", "expiration_present": False}, "effect": {"decision": "deny", "reason": "api_key_expiration_required", "required_action": "set_key_expiration"}},
	{"name": "subscription_requires_consumer", "description": "Subscriptions require consumer.", "condition": {"operation": "create_subscription", "consumer_present": False}, "effect": {"decision": "deny", "reason": "consumer_required", "required_action": "select_consumer"}},
	{"name": "subscription_requires_api", "description": "Subscriptions require API.", "condition": {"operation": "create_subscription", "api_present": False}, "effect": {"decision": "deny", "reason": "api_required", "required_action": "select_api"}},
	{"name": "subscription_plan_supported", "description": "Subscription plan must be supported.", "condition": {"operation": "create_subscription", "plan_supported": False}, "effect": {"decision": "deny", "reason": "subscription_plan_not_supported", "required_action": "select_supported_plan"}},
	{"name": "subscription_requires_approval", "description": "Subscriptions require approval.", "condition": {"operation": "create_subscription", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "subscription_approval_required", "required_action": "record_subscription_approval"}},
	{"name": "api_approval_requires_approver", "description": "API approval requires approver.", "condition": {"operation": "approve_api", "approver_present": False}, "effect": {"decision": "deny", "reason": "api_approver_required", "required_action": "assign_api_approver"}},
	{"name": "deployment_requires_api", "description": "Deployments require API.", "condition": {"operation": "deploy_api", "api_present": False}, "effect": {"decision": "deny", "reason": "api_required", "required_action": "select_api"}},
	{"name": "deployment_environment_supported", "description": "Deployment environment must be supported.", "condition": {"operation": "deploy_api", "environment_supported": False}, "effect": {"decision": "deny", "reason": "environment_not_supported", "required_action": "select_supported_environment"}},
	{"name": "deployment_requires_route", "description": "Deployments require gateway route.", "condition": {"operation": "deploy_api", "route_present": False}, "effect": {"decision": "deny", "reason": "gateway_route_required", "required_action": "set_gateway_route"}},
	{"name": "deployment_requires_deployer", "description": "Deployments require deployer.", "condition": {"operation": "deploy_api", "deployer_present": False}, "effect": {"decision": "deny", "reason": "deployer_required", "required_action": "assign_deployer"}},
	{"name": "production_deployment_requires_approval", "description": "Production deployments require approval.", "condition": {"operation": "deploy_api", "production_environment": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "production_deployment_approval_required", "required_action": "record_deployment_approval"}},
	{"name": "usage_requires_api", "description": "Usage records require API.", "condition": {"operation": "record_usage", "api_present": False}, "effect": {"decision": "deny", "reason": "api_required", "required_action": "select_api"}},
	{"name": "usage_status_code_required", "description": "Usage records require status code.", "condition": {"operation": "record_usage", "status_code_present": False}, "effect": {"decision": "deny", "reason": "status_code_required", "required_action": "set_status_code"}},
	{"name": "usage_latency_nonnegative", "description": "Usage latency must be nonnegative.", "condition": {"operation": "record_usage", "latency_ms_lt": 0}, "effect": {"decision": "deny", "reason": "latency_invalid", "required_action": "set_valid_latency"}},
	{"name": "slow_usage_requires_review", "description": "Slow API usage requires review.", "condition": {"operation": "record_usage", "slow_request": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "slow_request_review_required", "required_action": "record_latency_review"}},
	{"name": "api_batch_requires_bytewax", "description": "API batches require Bytewax coordination.", "condition": {"operation": "api_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_api_batch_to_bytewax"}},
	{"name": "api_event_requires_bytewax", "description": "API events require Bytewax.", "condition": {"operation": "api_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_api_event_to_bytewax"}},
	{"name": "api_agent_runtime_supported", "description": "API agents must use an approved runtime.", "condition": {"operation": "register_api_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "api_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "api_agent_role_supported", "description": "API agents must use an approved role.", "condition": {"operation": "register_api_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "api_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_api_agent_action_requires_human_approval", "description": "Privileged API actions proposed by agents require human approval.", "condition": {"operation": "api_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def _configuration_schema() -> dict[str, Any]:
	return {
		"type": "object",
		"required": list(DEFAULT_CONFIGURATION),
		"properties": {
			key: {"type": "object"} for key in DEFAULT_CONFIGURATION if key != "tenant_id"
		} | {"tenant_id": {"type": "string", "minLength": 1}},
	}


def _matches_condition(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			if context.get(key[:-4]) is None or context[key[:-4]] > expected:
				return False
			continue
		if key.endswith("_lt"):
			if context.get(key[:-3]) is None or context[key[:-3]] >= expected:
				return False
			continue
		if key.endswith("_gte"):
			if context.get(key[:-4]) is None or context[key[:-4]] < expected:
				return False
			continue
		if key.endswith("_gt"):
			if context.get(key[:-3]) is None or context[key[:-3]] <= expected:
				return False
			continue
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	if overrides:
		for key, value in overrides.items():
			if isinstance(value, dict) and isinstance(configuration.get(key), dict):
				configuration[key].update(value)
			else:
				configuration[key] = value

	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": configuration,
		"configuration_schema": _configuration_schema(),
		"provides": PROVIDES,
		"requires": REQUIRES,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/int-api/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"requires_theme": True,
			"view_module": "views.py",
			"template_roots": ["templates/", "static/"],
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	contract = get_capability_contract(context.get("tenant_id", "default"))
	matched = [
		rule for rule in contract["rule_engine"]["rules"]
		if _matches_condition(rule["condition"], context)
	]
	decision = "allow"
	for rule in matched:
		rule_decision = rule["effect"]["decision"]
		if rule_decision == "deny":
			decision = "deny"
			break
		if rule_decision == "require_review" and decision == "allow":
			decision = "require_review"
	return {
		"decision": decision,
		"matched_rules": [rule["name"] for rule in matched],
		"effects": [rule["effect"] for rule in matched],
	}
