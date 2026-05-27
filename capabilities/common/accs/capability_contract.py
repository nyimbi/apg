"""Executable capability contract for APG Accessibility Services."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"standards": {"default_standard": "WCAG-2.2-AA", "standard_required": True, "localized_guidance_enabled": True, "policy_versioning_enabled": True},
	"audits": {"audit_standard_required": True, "published_ui_contrast_required": True, "critical_issue_review_required": True, "automated_checks_enabled": True},
	"assistive": {"semantic_labels_required": True, "keyboard_navigation_required": True, "media_captions_required": True, "screen_reader_preview_enabled": True},
	"governance": {"require_tenant_context": True, "remediation_owner_required": True, "audit_change_logging": True, "compliance_exports_enabled": True},
	"ui": {"enable_audit_console": True, "enable_findings_board": True, "enable_remediation_queue": True, "enable_assistive_preview": True},
	"theme": {"default_theme": "accs_accessibility_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "standards", "audits", "assistive", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["standards", "audits", "assistive", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All accessibility operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "audit_requires_standard", "description": "Accessibility audits require a selected standard.", "condition": {"operation": "start_audit", "standard_selected": False}, "effect": {"decision": "deny", "reason": "audit_standard_required", "required_action": "select_accessibility_standard"}},
	{"name": "violation_requires_remediation_owner", "description": "Accessibility findings require a remediation owner.", "condition": {"violation_detected": True, "remediation_owner_assigned": False}, "effect": {"decision": "deny", "reason": "remediation_owner_required", "required_action": "assign_remediation_owner"}},
	{"name": "published_ui_requires_contrast", "description": "Published UI requires contrast validation.", "condition": {"published_ui": True, "contrast_passed": False}, "effect": {"decision": "deny", "reason": "contrast_validation_required", "required_action": "fix_contrast"}},
	{"name": "media_requires_captions", "description": "Media content requires captions or transcripts.", "condition": {"media_content_present": True, "captions_available": False}, "effect": {"decision": "deny", "reason": "captions_required", "required_action": "add_captions_or_transcript"}},
	{"name": "critical_issue_requires_review", "description": "Critical accessibility issues require formal review.", "condition": {"issue_severity": "critical", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "critical_accessibility_review_required", "required_action": "review_critical_issue"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/accs/dashboard", "component": "ACCSDashboard", "permission": "accs:view", "nav_group": "Overview"},
	{"name": "audits", "path": "/accs/audits", "component": "AuditConsole", "permission": "accs:audit", "nav_group": "Audits"},
	{"name": "findings", "path": "/accs/findings", "component": "FindingsBoard", "permission": "accs:view", "nav_group": "Audits"},
	{"name": "remediation", "path": "/accs/remediation", "component": "RemediationQueue", "permission": "accs:remediate", "nav_group": "Remediation"},
	{"name": "assistive", "path": "/accs/assistive", "component": "AssistivePreview", "permission": "accs:audit", "nav_group": "Assistive"},
	{"name": "media", "path": "/accs/media", "component": "MediaAccessibility", "permission": "accs:remediate", "nav_group": "Content"},
	{"name": "compliance", "path": "/accs/compliance", "component": "AccessibilityCompliance", "permission": "accs:manage_standards", "nav_group": "Governance"},
	{"name": "settings", "path": "/accs/settings", "component": "ACCSSettings", "permission": "accs:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "accs_accessibility_ops",
	"tokens": {"color.primary": "#22543D", "color.accent": "#3182CE", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"audit_score": {"icon": "badge-check", "status_indicator": "score-pill", "risk_style": "severity-band"}, "finding_board": {"visual": "kanban-list", "highlight": "blocked-chip"}, "assistive_preview": {"visual": "semantic-tree", "status_style": "label-chip"}, "compliance_panel": {"visual": "standard-matrix", "status_style": "evidence-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "accs", "display_name": "Accessibility Services", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/accs/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
