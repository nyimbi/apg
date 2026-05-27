"""Executable capability contract for APG Consent and Privacy Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"purposes": {"legal_basis_required": True, "purpose_owner_required": True, "retention_policy_required": True, "notice_link_required": True},
	"consents": {"notice_required": True, "active_consent_required": True, "withdrawal_supported": True, "stale_review_days": 365},
	"privacy_requests": {"identity_verification_required": True, "sla_tracking_enabled": True, "request_evidence_required": True, "dlpd_integration_required": True},
	"governance": {"require_tenant_context": True, "audit_consent_changes": True, "compliance_mapping_required": True, "restricted_processing_controls": True},
	"ui": {"enable_privacy_dashboard": True, "enable_purpose_registry": True, "enable_consent_ledger": True, "enable_request_queue": True},
	"theme": {"default_theme": "cons_privacy_center", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "purposes", "consents", "privacy_requests", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["purposes", "consents", "privacy_requests", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All consent and privacy operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "purpose_requires_legal_basis", "description": "Privacy purposes require a legal basis.", "condition": {"operation": "create_purpose", "legal_basis_present": False}, "effect": {"decision": "deny", "reason": "legal_basis_required", "required_action": "attach_legal_basis"}},
	{"name": "consent_capture_requires_notice", "description": "Consent capture requires a notice reference.", "condition": {"operation": "capture_consent", "notice_present": False}, "effect": {"decision": "deny", "reason": "notice_required", "required_action": "attach_privacy_notice"}},
	{"name": "processing_requires_active_consent", "description": "Consent-gated processing requires active consent.", "condition": {"operation": "process_consent_gated_data", "active_consent_present": False}, "effect": {"decision": "deny", "reason": "active_consent_required", "required_action": "collect_active_consent"}},
	{"name": "privacy_request_requires_identity_verification", "description": "Privacy requests require identity verification.", "condition": {"operation": "process_privacy_request", "identity_verified": False}, "effect": {"decision": "deny", "reason": "identity_verification_required", "required_action": "verify_request_identity"}},
	{"name": "stale_consent_requires_review", "description": "Stale consents require review.", "condition": {"consent_age_days_gt": 365, "stale_consent_reviewed": False}, "effect": {"decision": "require_review", "reason": "stale_consent_review_required", "required_action": "review_stale_consent"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/cons/dashboard", "component": "CONSDashboard", "permission": "cons:view", "nav_group": "Overview"},
	{"name": "purposes", "path": "/cons/purposes", "component": "PurposeRegistry", "permission": "cons:manage_purposes", "nav_group": "Policy"},
	{"name": "notices", "path": "/cons/notices", "component": "PrivacyNotices", "permission": "cons:manage_purposes", "nav_group": "Policy"},
	{"name": "consents", "path": "/cons/consents", "component": "ConsentLedger", "permission": "cons:view", "nav_group": "Consent"},
	{"name": "requests", "path": "/cons/requests", "component": "PrivacyRequestQueue", "permission": "cons:process_requests", "nav_group": "Requests"},
	{"name": "preferences", "path": "/cons/preferences", "component": "PreferenceCenter", "permission": "cons:capture", "nav_group": "Consent"},
	{"name": "audit", "path": "/cons/audit", "component": "PrivacyAudit", "permission": "cons:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/cons/settings", "component": "CONSSettings", "permission": "cons:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "cons_privacy_center",
	"tokens": {"color.primary": "#234E52", "color.accent": "#805AD5", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"purpose_card": {"icon": "shield-check", "status_indicator": "basis-pill", "risk_style": "privacy-band"}, "consent_ledger": {"visual": "event-timeline", "highlight": "withdrawal-chip"}, "request_queue": {"visual": "sla-board", "status_style": "verification-chip"}, "preference_center": {"visual": "consent-toggle-list", "status_style": "active-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "cons", "display_name": "Consent and Privacy Management", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/cons/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
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
