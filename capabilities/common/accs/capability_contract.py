"""Executable capability contract for APG Accessibility Services."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any

SUPPORTED_ACCESSIBILITY_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_ACCESSIBILITY_AGENT_ROLES = ["audit_reviewer", "remediation_planner", "caption_reviewer", "standards_advisor", "release_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"standards": {"default_standard": "WCAG-2.2-AA", "standard_required": True, "localized_guidance_enabled": True, "policy_versioning_enabled": True},
	"audits": {"audit_standard_required": True, "published_ui_contrast_required": True, "critical_issue_review_required": True, "automated_checks_enabled": True},
	"assistive": {"semantic_labels_required": True, "keyboard_navigation_required": True, "media_captions_required": True, "screen_reader_preview_enabled": True},
	"accessibility_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_ACCESSIBILITY_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_ACCESSIBILITY_AGENT_ROLES,
	},
	"governance": {"require_tenant_context": True, "remediation_owner_required": True, "audit_change_logging": True, "compliance_exports_enabled": True, "tenant_isolation_required": True, "state_change_reason_required": True, "batch_event_stream": "bytewax"},
	"observability": {"audit_required": True, "trace_required": True, "accessibility_metrics_required": True, "agent_activity_required": True, "event_stream": "bytewax"},
	"adapters": {"generated_app_runtime": "service.AccsService", "api_helpers": "api.py", "view_models": "views.py", "event_stream": "bytewax", "theme": "them", "i18n": "i18n", "nlp_guidance": "nlpc", "audit_sink": "audl", "compliance": "comp"},
	"ui": {"enable_audit_console": True, "enable_findings_board": True, "enable_remediation_queue": True, "enable_assistive_preview": True, "enable_agent_panel": True, "enable_audit_events": True, "enable_analytics": True},
	"theme": {"default_theme": "accs_accessibility_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "standards", "audits", "assistive", "accessibility_agents", "governance", "observability", "adapters", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["standards", "audits", "assistive", "accessibility_agents", "governance", "observability", "adapters", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All accessibility operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "audit_requires_standard", "description": "Accessibility audits require a selected standard.", "condition": {"operation": "start_audit", "standard_selected": False}, "effect": {"decision": "deny", "reason": "audit_standard_required", "required_action": "select_accessibility_standard"}},
	{"name": "violation_requires_remediation_owner", "description": "Accessibility findings require a remediation owner.", "condition": {"violation_detected": True, "remediation_owner_assigned": False}, "effect": {"decision": "deny", "reason": "remediation_owner_required", "required_action": "assign_remediation_owner"}},
	{"name": "published_ui_requires_contrast", "description": "Published UI requires contrast validation.", "condition": {"published_ui": True, "contrast_passed": False}, "effect": {"decision": "deny", "reason": "contrast_validation_required", "required_action": "fix_contrast"}},
	{"name": "media_requires_captions", "description": "Media content requires captions or transcripts.", "condition": {"media_content_present": True, "captions_available": False}, "effect": {"decision": "deny", "reason": "captions_required", "required_action": "add_captions_or_transcript"}},
	{"name": "critical_issue_requires_review", "description": "Critical accessibility issues require formal review.", "condition": {"issue_severity": "critical", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "critical_accessibility_review_required", "required_action": "review_critical_issue"}},
	{"name": "finding_closure_requires_resolution", "description": "Finding closure requires resolution evidence.", "condition": {"operation": "close_finding", "resolution_evidence_present": False}, "effect": {"decision": "deny", "reason": "resolution_evidence_required", "required_action": "attach_resolution_evidence"}},
	{"name": "accessibility_exception_requires_expiry", "description": "Accessibility exceptions require an active expiry date.", "condition": {"operation": "record_accessibility_exception", "exception_expiry_present": False}, "effect": {"decision": "deny", "reason": "accessibility_exception_expiry_required", "required_action": "set_accessibility_exception_expiry"}},
	{"name": "accessibility_exception_requires_compensating_controls", "description": "Accessibility exceptions require compensating controls.", "condition": {"operation": "record_accessibility_exception", "compensating_controls_present": False}, "effect": {"decision": "deny", "reason": "accessibility_exception_compensating_controls_required", "required_action": "attach_compensating_controls"}},
	{"name": "accessibility_agent_requires_registration", "description": "AI accessibility agents must be registered.", "condition": {"accessibility_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "accessibility_agent_registration_required", "required_action": "register_accessibility_agent"}},
	{"name": "accessibility_agent_runtime_supported", "description": "AI accessibility agents must use a supported runtime.", "condition": {"accessibility_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "accessibility_agent_runtime_not_supported", "required_action": "choose_supported_accessibility_agent_runtime"}},
	{"name": "accessibility_agent_role_supported", "description": "AI accessibility agents must use a supported role.", "condition": {"accessibility_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "accessibility_agent_role_not_supported", "required_action": "choose_supported_accessibility_agent_role"}},
	{"name": "accessibility_agent_requires_scope", "description": "AI accessibility agents require explicit scope.", "condition": {"accessibility_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "accessibility_agent_scope_required", "required_action": "set_accessibility_agent_scope"}},
	{"name": "accessibility_agent_requires_disclosure", "description": "AI accessibility-agent contributions require disclosure.", "condition": {"accessibility_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "accessibility_agent_disclosure_required", "required_action": "disclose_accessibility_agent"}},
	{"name": "accs_state_change_requires_audit", "description": "Accessibility lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "accs_audit_event_required", "required_action": "record_accessibility_audit_event"}},
	{"name": "cross_tenant_accessibility_access_denied", "description": "Accessibility records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_accessibility_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_accessibility_mutation_requires_bytewax", "description": "Batch accessibility mutations must use Bytewax event streams.", "condition": {"operation": "batch_accessibility_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "write_requires_policy", "description": "Accessibility write operations require an explicit authorization policy.", "condition": {"operation_type": "write", "write_policy_present": False}, "effect": {"decision": "deny", "reason": "accs_write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "privilege_escalation_denied", "description": "Accessibility operators cannot self-grant elevated permissions.", "condition": {"operation": "assign_accs_permission", "target_tier_exceeds_actor_tier": True}, "effect": {"decision": "deny", "reason": "privilege_escalation_prevented", "required_action": "route_to_higher_authority_approver"}},
	{"name": "accessibility_finding_delete_requires_approval", "description": "Deleting an accessibility finding requires explicit approval.", "condition": {"operation": "delete_finding", "delete_approved": False}, "effect": {"decision": "deny", "reason": "finding_delete_approval_required", "required_action": "record_finding_delete_approval"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/accs/dashboard", "component": "ACCSDashboard", "permission": "accs:view", "nav_group": "Overview"},
	{"name": "audits", "path": "/accs/audits", "component": "AuditConsole", "permission": "accs:audit", "nav_group": "Audits"},
	{"name": "findings", "path": "/accs/findings", "component": "FindingsBoard", "permission": "accs:view", "nav_group": "Audits"},
	{"name": "remediation", "path": "/accs/remediation", "component": "RemediationQueue", "permission": "accs:remediate", "nav_group": "Remediation"},
	{"name": "exceptions", "path": "/accs/exceptions", "component": "AccessibilityExceptionBoard", "permission": "accs:review", "nav_group": "Governance"},
	{"name": "assistive", "path": "/accs/assistive", "component": "AssistivePreview", "permission": "accs:audit", "nav_group": "Assistive"},
	{"name": "media", "path": "/accs/media", "component": "MediaAccessibility", "permission": "accs:remediate", "nav_group": "Content"},
	{"name": "compliance", "path": "/accs/compliance", "component": "AccessibilityCompliance", "permission": "accs:manage_standards", "nav_group": "Governance"},
	{"name": "agents", "path": "/accs/agents", "component": "AccessibilityAgentPanel", "permission": "accs:audit", "nav_group": "Agents"},
	{"name": "audit", "path": "/accs/audit", "component": "AccessibilityAuditTrail", "permission": "accs:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/accs/analytics", "component": "AccessibilityAnalytics", "permission": "accs:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/accs/settings", "component": "ACCSSettings", "permission": "accs:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "accs_accessibility_ops",
	"tokens": {"color.primary": "#22543D", "color.accent": "#3182CE", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"audit_score": {"icon": "badge-check", "status_indicator": "score-pill", "risk_style": "severity-band"}, "finding_board": {"visual": "kanban-list", "highlight": "blocked-chip"}, "exception_board": {"visual": "expiry-list", "status_style": "risk-acceptance-chip"}, "assistive_preview": {"visual": "semantic-tree", "status_style": "label-chip"}, "compliance_panel": {"visual": "standard-matrix", "status_style": "evidence-chip"}, "agent_panel": {"icon": "bot", "status_style": "scope-chip"}, "audit_timeline": {"icon": "list-checks", "status_style": "governance-chip"}}
}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.accs.lifecycle",
	"state": ["standards", "targets", "audits", "findings", "remediations", "reviews", "accessibility_exceptions", "accessibility_agents", "audit_events"],
	"events": ["standard_registered", "target_registered", "audit_completed", "finding_recorded", "remediation_updated", "finding_review_recorded", "finding_closed", "accessibility_exception_recorded", "accessibility_agent_registered"],
	"batch_mutation_guardrail": "batch_accessibility_mutation_requires_bytewax",
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "accs", "display_name": "Accessibility Services", "provides": ["accessibility_audits", "remediation_workflows", "accessibility_exceptions", "assistive_metadata", "media_accessibility", "standards_governance", "accessibility_agents"], "requires": ["them", "i18n", "nlpc"], "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": config["adapters"]["view_models"], "api_prefix": "/accs/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
