"""Executable capability contract for APG Banking APIs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_apis"
CAPABILITY_NAME = "Banking APIs"
CAPABILITY_VERSION = "1.1.0"
APIS_EVENT_STREAM = "apg.fintech.apis.lifecycle"

SUPPORTED_API_PRODUCTS = ["accounts", "balances", "transactions", "payments", "cards", "wallets", "loans", "bnpl", "agency", "customer_identity", "statements", "webhooks"]
SUPPORTED_ENVIRONMENTS = ["sandbox", "pilot", "production"]
SUPPORTED_AUTH_FLOWS = ["oauth2_auth_code", "client_credentials", "mtls", "signed_request", "device_code"]
SUPPORTED_REGIONS = ["KE", "UG", "TZ", "RW", "GH", "NG", "ZA", "GB", "US", "AE", "EU"]
SUPPORTED_WEBHOOK_EVENTS = ["account_updated", "transaction_posted", "payment_status", "card_event", "wallet_event", "loan_event", "bnpl_event", "agency_event", "fraud_alert", "consent_revoked"]
SUPPORTED_INCIDENT_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["api_ops_reviewer", "consent_reviewer", "developer_risk_reviewer", "rate_limit_reviewer", "webhook_reviewer", "incident_reviewer", "api_compliance_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"products": {"owner_required": True, "supported_products": SUPPORTED_API_PRODUCTS, "supported_environments": SUPPORTED_ENVIRONMENTS, "scope_required": True},
	"developers": {"kyb_required": True, "security_review_required": True, "risk_clearance_required": True},
	"applications": {"supported_environments": SUPPORTED_ENVIRONMENTS, "redirect_uri_required": True, "terms_required": True},
	"consents": {"scope_required": True, "customer_reference_required": True, "expiry_required": True},
	"clients": {"supported_auth_flows": SUPPORTED_AUTH_FLOWS, "key_reference_required": True, "scope_required": True},
	"endpoints": {"route_required": True, "scope_required": True, "throttle_policy_required": True, "risk_policy_required": True},
	"webhooks": {"supported_events": SUPPORTED_WEBHOOK_EVENTS, "endpoint_required": True, "signing_secret_required": True},
	"calls": {"risk_reference_required": True, "rate_limit_enforced": True, "high_volume_threshold": 10000},
	"rate_limits": {"default_limit": 1000, "burst_limit": 5000, "window_seconds": 60},
	"incidents": {"supported_severities": SUPPORTED_INCIDENT_SEVERITIES, "owner_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_api_events": True},
	"observability": {"event_stream": APIS_EVENT_STREAM, "stream_processor": "bytewax", "emit_product_events": True, "emit_developer_events": True, "emit_application_events": True, "emit_consent_events": True, "emit_client_events": True, "emit_endpoint_events": True, "emit_webhook_events": True, "emit_call_events": True, "emit_rate_limit_events": True, "emit_incident_events": True, "emit_agent_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "payments": "fintech_payments", "wallets": "fintech_wallets", "cards": "fintech_cards", "kyc": "fintech_kyc", "aml": "fintech_aml", "fraud": "fintech_fraud", "neobanking": "fintech_neobanking", "lending": "fintech_lending", "bnpl": "fintech_bnpl", "agency": "fintech_agency", "mobile": "fintech_mobile", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_products": True, "enable_developers": True, "enable_applications": True, "enable_consents": True, "enable_clients": True, "enable_endpoints": True, "enable_webhooks": True, "enable_calls": True, "enable_rate_limits": True, "enable_incidents": True, "enable_agents": True},
	"theme": {"default_theme": "fintech_apis_control", "allow_tenant_overrides": True},
}

PROVIDES = ["banking_api_product_governance", "developer_onboarding_workflow", "developer_application_workflow", "banking_consent_workflow", "api_client_credential_workflow", "api_endpoint_policy_workflow", "webhook_subscription_workflow", "api_call_audit_workflow", "api_rate_limit_workflow", "api_sla_incident_workflow", "banking_api_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_payments", "fintech_wallets", "fintech_cards", "fintech_kyc", "fintech_aml", "fintech_fraud", "fintech_neobanking", "fintech_lending", "fintech_bnpl", "fintech_agency", "fintech_mobile"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-apis/dashboard", "component": "BankingAPIDashboard", "permission": "fintech_apis:view", "nav_group": "Overview"},
	{"name": "products", "path": "/fintech-apis/products", "component": "APIProductConsole", "permission": "fintech_apis:products", "nav_group": "Products"},
	{"name": "developers", "path": "/fintech-apis/developers", "component": "DeveloperConsole", "permission": "fintech_apis:developers", "nav_group": "Developers"},
	{"name": "applications", "path": "/fintech-apis/applications", "component": "ApplicationConsole", "permission": "fintech_apis:applications", "nav_group": "Developers"},
	{"name": "consents", "path": "/fintech-apis/consents", "component": "ConsentWorkbench", "permission": "fintech_apis:consents", "nav_group": "Consent"},
	{"name": "clients", "path": "/fintech-apis/clients", "component": "APIClientConsole", "permission": "fintech_apis:clients", "nav_group": "Security"},
	{"name": "endpoints", "path": "/fintech-apis/endpoints", "component": "EndpointPolicyConsole", "permission": "fintech_apis:endpoints", "nav_group": "Gateway"},
	{"name": "webhooks", "path": "/fintech-apis/webhooks", "component": "WebhookConsole", "permission": "fintech_apis:webhooks", "nav_group": "Gateway"},
	{"name": "calls", "path": "/fintech-apis/calls", "component": "APICallAuditConsole", "permission": "fintech_apis:calls", "nav_group": "Operations"},
	{"name": "rate_limits", "path": "/fintech-apis/rate-limits", "component": "RateLimitConsole", "permission": "fintech_apis:rate_limits", "nav_group": "Operations"},
	{"name": "incidents", "path": "/fintech-apis/incidents", "component": "SLAIncidentWorkbench", "permission": "fintech_apis:incidents", "nav_group": "Operations"},
	{"name": "agents", "path": "/fintech-apis/agents", "component": "BankingAPIAgentWorkbench", "permission": "fintech_apis:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-apis/settings", "component": "BankingAPISettings", "permission": "fintech_apis:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_apis_control",
	"tokens": {"color.primary": "#1D4ED8", "color.accent": "#0F766E", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"products": {"icon": "box", "status_indicator": "product-chip"}, "developers": {"icon": "building-2", "status_indicator": "developer-chip"}, "applications": {"icon": "app-window", "status_indicator": "app-chip"}, "consents": {"icon": "file-check", "status_indicator": "consent-chip"}, "clients": {"icon": "key-round", "status_indicator": "client-chip"}, "endpoints": {"icon": "route", "status_indicator": "endpoint-chip"}, "webhooks": {"icon": "webhook", "status_indicator": "webhook-chip"}, "calls": {"icon": "activity", "status_indicator": "call-chip"}, "rate_limits": {"icon": "gauge", "status_indicator": "limit-chip"}, "incidents": {"icon": "sirens", "status_indicator": "incident-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": APIS_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["api_product_registered", "developer_onboarded", "developer_application_registered", "consent_grant_created", "api_client_issued", "endpoint_policy_published", "webhook_subscribed", "api_call_recorded", "rate_limit_updated", "sla_incident_opened", "api_agent_registered"],
	"guardrails": ["apis_batch_requires_bytewax", "privileged_api_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Banking API operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "apis_write_requires_policy", "description": "Banking API writes require policy evidence.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_evidence_required", "required_action": "attach_policy_evidence"}},
	{"name": "product_owner_required", "description": "API products require owner.", "condition": {"operation": "register_api_product", "owner_present": False}, "effect": {"decision": "deny", "reason": "product_owner_required", "required_action": "assign_product_owner"}},
	{"name": "product_type_supported", "description": "API product type must be supported.", "condition": {"operation": "register_api_product", "product_type_supported": False}, "effect": {"decision": "deny", "reason": "api_product_type_not_supported", "required_action": "select_supported_product"}},
	{"name": "product_environment_supported", "description": "API product environment must be supported.", "condition": {"operation": "register_api_product", "environment_supported": False}, "effect": {"decision": "deny", "reason": "api_environment_not_supported", "required_action": "select_supported_environment"}},
	{"name": "product_scopes_required", "description": "API products require scopes.", "condition": {"operation": "register_api_product", "scopes_present": False}, "effect": {"decision": "deny", "reason": "product_scopes_required", "required_action": "attach_scopes"}},
	{"name": "developer_kyb_required", "description": "Developers require KYB evidence.", "condition": {"operation": "onboard_developer", "kyb_present": False}, "effect": {"decision": "deny", "reason": "developer_kyb_required", "required_action": "attach_kyb_evidence"}},
	{"name": "developer_security_required", "description": "Developers require security review.", "condition": {"operation": "onboard_developer", "security_review_present": False}, "effect": {"decision": "deny", "reason": "developer_security_review_required", "required_action": "attach_security_review"}},
	{"name": "developer_risk_required", "description": "Developers require risk clearance.", "condition": {"operation": "onboard_developer", "risk_clearance_present": False}, "effect": {"decision": "deny", "reason": "developer_risk_clearance_required", "required_action": "attach_risk_clearance"}},
	{"name": "application_developer_required", "description": "Applications require developer.", "condition": {"operation": "register_application", "developer_present": False}, "effect": {"decision": "deny", "reason": "application_developer_required", "required_action": "select_developer"}},
	{"name": "application_environment_supported", "description": "Application environment must be supported.", "condition": {"operation": "register_application", "environment_supported": False}, "effect": {"decision": "deny", "reason": "application_environment_not_supported", "required_action": "select_supported_environment"}},
	{"name": "application_redirect_required", "description": "Applications require redirect URI.", "condition": {"operation": "register_application", "redirect_uri_present": False}, "effect": {"decision": "deny", "reason": "redirect_uri_required", "required_action": "attach_redirect_uri"}},
	{"name": "application_terms_required", "description": "Applications require terms evidence.", "condition": {"operation": "register_application", "terms_present": False}, "effect": {"decision": "deny", "reason": "application_terms_required", "required_action": "attach_terms"}},
	{"name": "consent_application_required", "description": "Consents require application.", "condition": {"operation": "create_consent_grant", "application_present": False}, "effect": {"decision": "deny", "reason": "consent_application_required", "required_action": "select_application"}},
	{"name": "consent_customer_required", "description": "Consents require customer.", "condition": {"operation": "create_consent_grant", "customer_present": False}, "effect": {"decision": "deny", "reason": "consent_customer_required", "required_action": "attach_customer_reference"}},
	{"name": "consent_scopes_required", "description": "Consents require scopes.", "condition": {"operation": "create_consent_grant", "scopes_present": False}, "effect": {"decision": "deny", "reason": "consent_scopes_required", "required_action": "attach_scopes"}},
	{"name": "consent_expiry_required", "description": "Consents require expiry.", "condition": {"operation": "create_consent_grant", "expiry_present": False}, "effect": {"decision": "deny", "reason": "consent_expiry_required", "required_action": "attach_expiry"}},
	{"name": "client_application_required", "description": "API clients require application.", "condition": {"operation": "issue_api_client", "application_present": False}, "effect": {"decision": "deny", "reason": "client_application_required", "required_action": "select_application"}},
	{"name": "client_auth_flow_supported", "description": "API client auth flow must be supported.", "condition": {"operation": "issue_api_client", "auth_flow_supported": False}, "effect": {"decision": "deny", "reason": "client_auth_flow_not_supported", "required_action": "select_supported_auth_flow"}},
	{"name": "client_key_required", "description": "API clients require key reference.", "condition": {"operation": "issue_api_client", "key_reference_present": False}, "effect": {"decision": "deny", "reason": "client_key_reference_required", "required_action": "attach_key_reference"}},
	{"name": "client_scopes_required", "description": "API clients require scopes.", "condition": {"operation": "issue_api_client", "scopes_present": False}, "effect": {"decision": "deny", "reason": "client_scopes_required", "required_action": "attach_scopes"}},
	{"name": "client_scopes_allowed_by_consent", "description": "API client scopes must be covered by active consent.", "condition": {"operation": "issue_api_client", "scopes_allowed_by_consent": False}, "effect": {"decision": "deny", "reason": "client_scopes_not_consented", "required_action": "capture_matching_consent"}},
	{"name": "endpoint_product_required", "description": "Endpoint policies require product.", "condition": {"operation": "publish_endpoint_policy", "product_present": False}, "effect": {"decision": "deny", "reason": "endpoint_product_required", "required_action": "select_product"}},
	{"name": "endpoint_route_required", "description": "Endpoint policies require route.", "condition": {"operation": "publish_endpoint_policy", "route_present": False}, "effect": {"decision": "deny", "reason": "endpoint_route_required", "required_action": "attach_route"}},
	{"name": "endpoint_scope_required", "description": "Endpoint policies require scope.", "condition": {"operation": "publish_endpoint_policy", "scope_present": False}, "effect": {"decision": "deny", "reason": "endpoint_scope_required", "required_action": "attach_scope"}},
	{"name": "endpoint_throttle_required", "description": "Endpoint policies require throttle policy.", "condition": {"operation": "publish_endpoint_policy", "throttle_policy_present": False}, "effect": {"decision": "deny", "reason": "endpoint_throttle_required", "required_action": "attach_throttle_policy"}},
	{"name": "endpoint_risk_required", "description": "Endpoint policies require risk policy.", "condition": {"operation": "publish_endpoint_policy", "risk_policy_present": False}, "effect": {"decision": "deny", "reason": "endpoint_risk_policy_required", "required_action": "attach_risk_policy"}},
	{"name": "webhook_application_required", "description": "Webhooks require application.", "condition": {"operation": "subscribe_webhook", "application_present": False}, "effect": {"decision": "deny", "reason": "webhook_application_required", "required_action": "select_application"}},
	{"name": "webhook_event_supported", "description": "Webhook event must be supported.", "condition": {"operation": "subscribe_webhook", "event_supported": False}, "effect": {"decision": "deny", "reason": "webhook_event_not_supported", "required_action": "select_supported_event"}},
	{"name": "webhook_endpoint_required", "description": "Webhooks require endpoint.", "condition": {"operation": "subscribe_webhook", "endpoint_present": False}, "effect": {"decision": "deny", "reason": "webhook_endpoint_required", "required_action": "attach_endpoint"}},
	{"name": "webhook_signing_secret_required", "description": "Webhooks require signing secret.", "condition": {"operation": "subscribe_webhook", "signing_secret_present": False}, "effect": {"decision": "deny", "reason": "webhook_signing_secret_required", "required_action": "attach_signing_secret"}},
	{"name": "api_call_client_required", "description": "API calls require client.", "condition": {"operation": "record_api_call", "client_present": False}, "effect": {"decision": "deny", "reason": "api_call_client_required", "required_action": "select_client"}},
	{"name": "api_call_product_required", "description": "API calls require product.", "condition": {"operation": "record_api_call", "product_present": False}, "effect": {"decision": "deny", "reason": "api_call_product_required", "required_action": "select_product"}},
	{"name": "api_call_endpoint_required", "description": "API calls require endpoint policy.", "condition": {"operation": "record_api_call", "endpoint_present": False}, "effect": {"decision": "deny", "reason": "api_call_endpoint_required", "required_action": "select_endpoint"}},
	{"name": "api_call_endpoint_matches_product", "description": "API call endpoint must belong to the selected product.", "condition": {"operation": "record_api_call", "endpoint_matches_product": False}, "effect": {"decision": "deny", "reason": "api_call_endpoint_product_mismatch", "required_action": "select_product_endpoint"}},
	{"name": "api_call_rate_limit_allowed", "description": "API calls must be within rate limit.", "condition": {"operation": "record_api_call", "rate_limit_allowed": False}, "effect": {"decision": "deny", "reason": "api_rate_limit_exceeded", "required_action": "throttle_client"}},
	{"name": "api_call_risk_reference_required", "description": "API calls require risk reference.", "condition": {"operation": "record_api_call", "risk_reference_present": False}, "effect": {"decision": "deny", "reason": "api_call_risk_reference_required", "required_action": "attach_risk_reference"}},
	{"name": "high_volume_api_call_requires_review", "description": "High-volume call batches require review.", "condition": {"operation": "record_api_call", "high_volume": True, "human_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "api_call_review_required", "required_action": "record_call_review"}},
	{"name": "rate_limit_client_required", "description": "Rate limit buckets require client.", "condition": {"operation": "update_rate_limit", "client_present": False}, "effect": {"decision": "deny", "reason": "rate_limit_client_required", "required_action": "select_client"}},
	{"name": "rate_limit_positive", "description": "Rate limits must be positive.", "condition": {"operation": "update_rate_limit", "positive_limit": False}, "effect": {"decision": "deny", "reason": "positive_rate_limit_required", "required_action": "set_positive_limit"}},
	{"name": "incident_severity_supported", "description": "Incident severity must be supported.", "condition": {"operation": "open_sla_incident", "severity_supported": False}, "effect": {"decision": "deny", "reason": "incident_severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "incident_owner_required", "description": "Incidents require owner.", "condition": {"operation": "open_sla_incident", "owner_present": False}, "effect": {"decision": "deny", "reason": "incident_owner_required", "required_action": "assign_owner"}},
	{"name": "incident_evidence_required", "description": "Incidents require evidence.", "condition": {"operation": "open_sla_incident", "evidence_present": False}, "effect": {"decision": "deny", "reason": "incident_evidence_required", "required_action": "attach_incident_evidence"}},
	{"name": "critical_incident_requires_approval", "description": "Critical incidents require approval.", "condition": {"operation": "open_sla_incident", "critical_severity": True, "human_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "incident_approval_required", "required_action": "record_incident_approval"}},
	{"name": "apis_batch_requires_bytewax", "description": "Banking API batches require Bytewax.", "condition": {"operation": "apis_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_api_batch_to_bytewax"}},
	{"name": "api_agent_runtime_supported", "description": "Banking API agents must use a supported runtime.", "condition": {"operation": "register_api_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "api_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "api_agent_role_supported", "description": "Banking API agents must use a supported role.", "condition": {"operation": "register_api_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "api_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_api_agent_action_requires_human_approval", "description": "Privileged API-agent actions require human approval.", "condition": {"operation": "api_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def _configuration_schema() -> dict[str, Any]:
	return {"type": "object", "required": list(DEFAULT_CONFIGURATION), "properties": {key: {"type": "object"} for key in DEFAULT_CONFIGURATION if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}


def _matches_condition(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
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
	return {"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "configuration": configuration, "configuration_schema": _configuration_schema(), "provides": PROVIDES, "requires": REQUIRES, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-apis/api/v1", "routes": deepcopy(UI_ROUTES), "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"]}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	contract = get_capability_contract(str(context.get("tenant_id") or "default"))
	matched = [rule for rule in contract["rule_engine"]["rules"] if _matches_condition(rule["condition"], context)]
	decision = "allow"
	for rule in matched:
		effect = rule["effect"]["decision"]
		if effect == "deny":
			decision = "deny"
			break
		if effect == "require_review" and decision == "allow":
			decision = "require_review"
	return {"decision": decision, "matched_rules": [rule["name"] for rule in matched], "actions": [rule["effect"] for rule in matched], "effects": [rule["effect"] for rule in matched]}
