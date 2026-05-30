"""Executable capability contract for APG Consent and Privacy Management."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any

SUPPORTED_PRIVACY_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_PRIVACY_AGENT_ROLES = ["notice_reviewer", "consent_operator", "privacy_request_reviewer", "dlp_reviewer", "compliance_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"purposes": {"legal_basis_required": True, "purpose_owner_required": True, "retention_policy_required": True, "notice_link_required": True},
	"consents": {"notice_required": True, "active_consent_required": True, "withdrawal_supported": True, "stale_review_days": 365},
	"privacy_requests": {"identity_verification_required": True, "sla_tracking_enabled": True, "request_evidence_required": True, "dlpd_integration_required": True},
	"privacy_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_PRIVACY_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_PRIVACY_AGENT_ROLES,
	},
	"governance": {"require_tenant_context": True, "audit_consent_changes": True, "compliance_mapping_required": True, "restricted_processing_controls": True, "tenant_isolation_required": True, "state_change_reason_required": True, "batch_event_stream": "bytewax"},
	"observability": {"audit_required": True, "privacy_metrics_required": True, "agent_activity_required": True, "sla_metrics_required": True, "event_stream": "bytewax"},
	"adapters": {"generated_app_runtime": "service.ConsService", "api_helpers": "api.py", "view_models": "views.py", "event_stream": "bytewax", "compliance": "comp", "auth": "auth", "dlp": "dlpd", "audit_sink": "audl", "notifications": "ntfy", "workflow": "wflo", "i18n": "i18n"},
	"ui": {"enable_privacy_dashboard": True, "enable_purpose_registry": True, "enable_consent_ledger": True, "enable_request_queue": True, "enable_agent_panel": True, "enable_analytics": True},
	"theme": {"default_theme": "cons_privacy_center", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "purposes", "consents", "privacy_requests", "privacy_agents", "governance", "observability", "adapters", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["purposes", "consents", "privacy_requests", "privacy_agents", "governance", "observability", "adapters", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All consent and privacy operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "purpose_requires_legal_basis", "description": "Privacy purposes require a legal basis.", "condition": {"operation": "create_purpose", "legal_basis_present": False}, "effect": {"decision": "deny", "reason": "legal_basis_required", "required_action": "attach_legal_basis"}},
	{"name": "consent_capture_requires_notice", "description": "Consent capture requires a notice reference.", "condition": {"operation": "capture_consent", "notice_present": False}, "effect": {"decision": "deny", "reason": "notice_required", "required_action": "attach_privacy_notice"}},
	{"name": "processing_requires_active_consent", "description": "Consent-gated processing requires active consent.", "condition": {"operation": "process_consent_gated_data", "active_consent_present": False}, "effect": {"decision": "deny", "reason": "active_consent_required", "required_action": "collect_active_consent"}},
	{"name": "privacy_request_requires_identity_verification", "description": "Privacy requests require identity verification.", "condition": {"operation": "process_privacy_request", "identity_verified": False}, "effect": {"decision": "deny", "reason": "identity_verification_required", "required_action": "verify_request_identity"}},
	{"name": "stale_consent_requires_review", "description": "Stale consents require review.", "condition": {"consent_age_days_gt": 365, "stale_consent_reviewed": False}, "effect": {"decision": "require_review", "reason": "stale_consent_review_required", "required_action": "review_stale_consent"}},
	{"name": "purpose_requires_owner", "description": "Privacy purposes require an accountable owner.", "condition": {"operation": "create_purpose", "purpose_owner_assigned": False}, "effect": {"decision": "deny", "reason": "purpose_owner_required", "required_action": "assign_purpose_owner"}},
	{"name": "purpose_requires_retention_policy", "description": "Privacy purposes require retention policy.", "condition": {"operation": "create_purpose", "retention_policy_present": False}, "effect": {"decision": "deny", "reason": "retention_policy_required", "required_action": "attach_retention_policy"}},
	{"name": "purpose_requires_notice_link", "description": "Privacy purposes require notice linkage.", "condition": {"operation": "create_purpose", "notice_link_present": False}, "effect": {"decision": "deny", "reason": "notice_link_required", "required_action": "link_privacy_notice"}},
	{"name": "privacy_request_requires_evidence", "description": "Privacy requests require request evidence.", "condition": {"operation": "process_privacy_request", "request_evidence_present": False}, "effect": {"decision": "deny", "reason": "request_evidence_required", "required_action": "attach_request_evidence"}},
	{"name": "privacy_agent_requires_registration", "description": "AI privacy agents must be registered.", "condition": {"privacy_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "privacy_agent_registration_required", "required_action": "register_privacy_agent"}},
	{"name": "privacy_agent_runtime_supported", "description": "AI privacy agents must use a supported runtime.", "condition": {"privacy_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "privacy_agent_runtime_not_supported", "required_action": "choose_supported_privacy_agent_runtime"}},
	{"name": "privacy_agent_role_supported", "description": "AI privacy agents must use a supported role.", "condition": {"privacy_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "privacy_agent_role_not_supported", "required_action": "choose_supported_privacy_agent_role"}},
	{"name": "privacy_agent_requires_scope", "description": "AI privacy agents require explicit scope.", "condition": {"privacy_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "privacy_agent_scope_required", "required_action": "set_privacy_agent_scope"}},
	{"name": "privacy_agent_requires_disclosure", "description": "AI privacy-agent contributions require disclosure.", "condition": {"privacy_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "privacy_agent_disclosure_required", "required_action": "disclose_privacy_agent"}},
	{"name": "cons_state_change_requires_reason", "description": "Privacy lifecycle state changes require a reason.", "condition": {"state_change_requested": True, "state_change_reason_present": False}, "effect": {"decision": "deny", "reason": "cons_state_change_reason_required", "required_action": "record_state_change_reason"}},
	{"name": "cons_state_change_requires_audit", "description": "Privacy lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "cons_audit_event_required", "required_action": "record_cons_audit_event"}},
	{"name": "cross_tenant_privacy_access_denied", "description": "Privacy records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_privacy_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_privacy_mutation_requires_bytewax", "description": "Batch privacy mutations must use Bytewax event streams.", "condition": {"operation": "batch_privacy_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/cons/dashboard", "component": "CONSDashboard", "permission": "cons:view", "nav_group": "Overview"},
	{"name": "purposes", "path": "/cons/purposes", "component": "PurposeRegistry", "permission": "cons:manage_purposes", "nav_group": "Policy"},
	{"name": "notices", "path": "/cons/notices", "component": "PrivacyNotices", "permission": "cons:manage_purposes", "nav_group": "Policy"},
	{"name": "consents", "path": "/cons/consents", "component": "ConsentLedger", "permission": "cons:view", "nav_group": "Consent"},
	{"name": "requests", "path": "/cons/requests", "component": "PrivacyRequestQueue", "permission": "cons:process_requests", "nav_group": "Requests"},
	{"name": "preferences", "path": "/cons/preferences", "component": "PreferenceCenter", "permission": "cons:capture", "nav_group": "Consent"},
	{"name": "agents", "path": "/cons/agents", "component": "PrivacyAgentPanel", "permission": "cons:process_requests", "nav_group": "Agents"},
	{"name": "analytics", "path": "/cons/analytics", "component": "PrivacyAnalytics", "permission": "cons:view", "nav_group": "Operations"},
	{"name": "audit", "path": "/cons/audit", "component": "PrivacyAudit", "permission": "cons:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/cons/settings", "component": "CONSSettings", "permission": "cons:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "cons_privacy_center",
	"tokens": {"color.primary": "#234E52", "color.accent": "#805AD5", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"purpose_card": {"icon": "shield-check", "status_indicator": "basis-pill", "risk_style": "privacy-band"}, "consent_ledger": {"visual": "event-timeline", "highlight": "withdrawal-chip"}, "request_queue": {"visual": "sla-board", "status_style": "verification-chip"}, "preference_center": {"visual": "consent-toggle-list", "status_style": "active-chip"}, "agent_panel": {"icon": "bot", "status_style": "scope-chip"}, "audit_timeline": {"icon": "list-checks", "status_style": "privacy-chip"}}
}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.cons.lifecycle",
	"state": ["purposes", "notices", "consents", "preferences", "privacy_requests", "processing_decisions", "privacy_agents", "audit_events"],
	"events": ["notice_published", "purpose_created", "purpose_state_changed", "consent_captured", "consent_withdrawn", "preferences_updated", "privacy_request_submitted", "privacy_request_completed", "privacy_agent_registered"],
	"batch_mutation_guardrail": "batch_privacy_mutation_requires_bytewax",
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "cons", "display_name": "Consent and Privacy Management", "provides": ["purpose_registry", "consent_capture", "privacy_requests", "preference_center", "privacy_audit", "privacy_agents"], "requires": ["comp", "auth", "dlpd"], "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": config["adapters"]["view_models"], "api_prefix": "/cons/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
