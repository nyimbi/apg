"""Executable capability contract for APG Anti Money Laundering."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_aml"
CAPABILITY_NAME = "Anti Money Laundering"
CAPABILITY_VERSION = "1.1.0"
AML_EVENT_STREAM = "apg.fintech.aml.lifecycle"

SUPPORTED_ALERT_TYPES = ["large_transaction", "velocity", "structuring", "sanctions", "pep", "high_risk_kyc", "mule_account", "agent_review"]
SUPPORTED_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_CASE_TYPES = ["transaction_monitoring", "sanctions_alert", "structuring_alert", "mule_account", "high_risk_customer", "suspicious_activity_report"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["aml_ops_reviewer", "transaction_monitoring_analyst", "sanctions_reviewer", "case_investigator", "sar_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"monitoring": {"large_transaction_threshold": 10000.0, "velocity_window_minutes": 60, "velocity_count_threshold": 5, "velocity_amount_threshold": 25000.0, "structuring_threshold": 9500.0, "structuring_count_threshold": 3, "high_risk_score_threshold": 75},
	"alerts": {"supported_types": SUPPORTED_ALERT_TYPES, "supported_severities": SUPPORTED_SEVERITIES, "evidence_required": True, "disposition_required_to_close": True, "auto_close_allowed": False},
	"cases": {"supported_types": SUPPORTED_CASE_TYPES, "investigator_required_for_escalation": True, "evidence_required": True, "sar_allowed_statuses": ["escalated", "under_investigation", "confirmed_suspicious"]},
	"sar": {"human_approval_required": True, "subject_required": True, "jurisdiction_required": True, "narrative_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_aml_events": True, "kyc_link_required": True, "human_approval_required_for_sar": True},
	"observability": {"event_stream": AML_EVENT_STREAM, "stream_processor": "bytewax", "emit_transaction_events": True, "emit_alert_events": True, "emit_case_events": True, "emit_sar_events": True, "emit_agent_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "payments": "fintech_payments", "wallets": "fintech_wallets", "kyc": "fintech_kyc", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_alerts": True, "enable_monitoring": True, "enable_cases": True, "enable_sar": True, "enable_typologies": True, "enable_agents": True},
	"theme": {"default_theme": "fintech_aml_control", "allow_tenant_overrides": True},
}

PROVIDES = ["transaction_monitoring", "aml_alert_triage", "sanctions_pep_escalation", "suspicious_activity_case_management", "sar_workflow", "typology_rule_engine", "aml_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_payments", "fintech_wallets", "fintech_kyc"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-aml/dashboard", "component": "AmlDashboard", "permission": "fintech_aml:view", "nav_group": "Overview"},
	{"name": "alerts", "path": "/fintech-aml/alerts", "component": "AmlAlertQueue", "permission": "fintech_aml:triage", "nav_group": "Alerts"},
	{"name": "monitoring", "path": "/fintech-aml/monitoring", "component": "AmlMonitoringConsole", "permission": "fintech_aml:monitor", "nav_group": "Monitoring"},
	{"name": "cases", "path": "/fintech-aml/cases", "component": "AmlCaseWorkbench", "permission": "fintech_aml:investigate", "nav_group": "Cases"},
	{"name": "sar", "path": "/fintech-aml/sar", "component": "AmlSarWorkflow", "permission": "fintech_aml:file_sar", "nav_group": "Regulatory"},
	{"name": "typologies", "path": "/fintech-aml/typologies", "component": "AmlTypologyRules", "permission": "fintech_aml:admin", "nav_group": "Rules"},
	{"name": "agents", "path": "/fintech-aml/agents", "component": "AmlAgentWorkbench", "permission": "fintech_aml:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-aml/settings", "component": "AmlSettings", "permission": "fintech_aml:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_aml_control",
	"tokens": {"color.primary": "#27374D", "color.accent": "#0F766E", "color.success": "#166534", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"alerts": {"icon": "shield-alert", "status_indicator": "aml-severity-chip"}, "monitoring": {"icon": "activity", "visual": "transaction-lane"}, "cases": {"icon": "folder-search", "status_indicator": "case-status-chip"}, "sar": {"icon": "file-warning", "status_indicator": "sar-approval-chip"}, "typologies": {"icon": "list-checks", "visual": "rule-matrix"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": AML_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["aml_transaction_monitored", "aml_alert_created", "aml_alert_triaged", "aml_case_opened", "aml_sar_drafted", "aml_agent_registered"],
	"guardrails": ["aml_batch_requires_bytewax", "aml_event_requires_bytewax", "privileged_aml_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "AML operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "aml_write_requires_policy", "description": "AML writes require policy evidence.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "aml_policy_required", "required_action": "attach_aml_policy"}},
	{"name": "transaction_subject_required", "description": "Monitored transactions require a subject reference.", "condition": {"operation": "monitor_transaction", "subject_present": False}, "effect": {"decision": "deny", "reason": "transaction_subject_required", "required_action": "attach_subject_reference"}},
	{"name": "transaction_amount_required", "description": "Monitored transactions require a positive amount.", "condition": {"operation": "monitor_transaction", "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_amount_required", "required_action": "set_positive_amount"}},
	{"name": "transaction_currency_required", "description": "Monitored transactions require currency.", "condition": {"operation": "monitor_transaction", "currency_present": False}, "effect": {"decision": "deny", "reason": "currency_required", "required_action": "set_currency"}},
	{"name": "transaction_source_required", "description": "Monitored transactions require source capability reference.", "condition": {"operation": "monitor_transaction", "source_reference_present": False}, "effect": {"decision": "deny", "reason": "source_reference_required", "required_action": "attach_source_reference"}},
	{"name": "transaction_requires_kyc_link", "description": "AML monitoring requires linked KYC profile evidence.", "condition": {"operation": "monitor_transaction", "kyc_link_present": False}, "effect": {"decision": "deny", "reason": "kyc_link_required", "required_action": "attach_kyc_profile"}},
	{"name": "large_transaction_requires_review", "description": "Large transactions require AML review.", "condition": {"operation": "monitor_transaction", "large_transaction": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_transaction_review_required", "required_action": "review_large_transaction"}},
	{"name": "velocity_requires_review", "description": "Velocity indicators require AML review.", "condition": {"operation": "monitor_transaction", "velocity_indicator": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "velocity_review_required", "required_action": "review_velocity_pattern"}},
	{"name": "structuring_requires_review", "description": "Structuring indicators require AML review.", "condition": {"operation": "monitor_transaction", "structuring_indicator": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "structuring_review_required", "required_action": "review_structuring_pattern"}},
	{"name": "sanctions_requires_escalation", "description": "Sanctions indicators require escalation.", "condition": {"operation": "monitor_transaction", "sanctions_hit": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "sanctions_escalation_required", "required_action": "escalate_sanctions_hit"}},
	{"name": "high_risk_kyc_requires_review", "description": "High-risk KYC subjects require AML review.", "condition": {"operation": "monitor_transaction", "high_risk_kyc": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "high_risk_kyc_review_required", "required_action": "review_high_risk_subject"}},
	{"name": "alert_type_supported", "description": "AML alert type must be supported.", "condition": {"operation": "create_alert", "alert_type_supported": False}, "effect": {"decision": "deny", "reason": "aml_alert_type_not_supported", "required_action": "select_supported_alert_type"}},
	{"name": "alert_severity_supported", "description": "AML alert severity must be supported.", "condition": {"operation": "create_alert", "severity_supported": False}, "effect": {"decision": "deny", "reason": "aml_severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "alert_evidence_required", "description": "AML alerts require evidence references.", "condition": {"operation": "create_alert", "evidence_present": False}, "effect": {"decision": "deny", "reason": "aml_alert_evidence_required", "required_action": "attach_alert_evidence"}},
	{"name": "alert_close_requires_disposition", "description": "Closing AML alerts requires disposition.", "condition": {"operation": "triage_alert", "closing_alert": True, "disposition_present": False}, "effect": {"decision": "deny", "reason": "alert_disposition_required", "required_action": "record_alert_disposition"}},
	{"name": "alert_escalation_requires_reviewer", "description": "Escalated alerts require reviewer evidence.", "condition": {"operation": "triage_alert", "escalating_alert": True, "reviewer_present": False}, "effect": {"decision": "deny", "reason": "alert_reviewer_required", "required_action": "assign_aml_reviewer"}},
	{"name": "case_alert_required", "description": "AML cases require an alert reference.", "condition": {"operation": "open_case", "alert_present": False}, "effect": {"decision": "deny", "reason": "aml_alert_required", "required_action": "select_alert"}},
	{"name": "case_type_supported", "description": "AML case type must be supported.", "condition": {"operation": "open_case", "case_type_supported": False}, "effect": {"decision": "deny", "reason": "aml_case_type_not_supported", "required_action": "select_supported_case_type"}},
	{"name": "case_investigator_required", "description": "AML cases require an investigator.", "condition": {"operation": "open_case", "investigator_present": False}, "effect": {"decision": "deny", "reason": "case_investigator_required", "required_action": "assign_investigator"}},
	{"name": "sar_case_required", "description": "SAR drafts require a case.", "condition": {"operation": "draft_sar", "case_present": False}, "effect": {"decision": "deny", "reason": "sar_case_required", "required_action": "select_case"}},
	{"name": "sar_subject_required", "description": "SAR drafts require subject reference.", "condition": {"operation": "draft_sar", "subject_present": False}, "effect": {"decision": "deny", "reason": "sar_subject_required", "required_action": "attach_subject_reference"}},
	{"name": "sar_jurisdiction_required", "description": "SAR drafts require jurisdiction.", "condition": {"operation": "draft_sar", "jurisdiction_present": False}, "effect": {"decision": "deny", "reason": "sar_jurisdiction_required", "required_action": "set_jurisdiction"}},
	{"name": "sar_narrative_required", "description": "SAR drafts require a narrative.", "condition": {"operation": "draft_sar", "narrative_present": False}, "effect": {"decision": "deny", "reason": "sar_narrative_required", "required_action": "write_sar_narrative"}},
	{"name": "sar_evidence_required", "description": "SAR drafts require evidence references.", "condition": {"operation": "draft_sar", "evidence_present": False}, "effect": {"decision": "deny", "reason": "sar_evidence_required", "required_action": "attach_sar_evidence"}},
	{"name": "sar_human_approval_required", "description": "SAR drafts require human approval before filing.", "condition": {"operation": "draft_sar", "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "sar_human_approval_required", "required_action": "record_human_approval"}},
	{"name": "aml_batch_requires_bytewax", "description": "AML batches require Bytewax.", "condition": {"operation": "aml_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_aml_batch_to_bytewax"}},
	{"name": "aml_event_requires_bytewax", "description": "AML events require Bytewax.", "condition": {"operation": "aml_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_aml_event_to_bytewax"}},
	{"name": "aml_agent_runtime_supported", "description": "AML agents must use a supported runtime.", "condition": {"operation": "register_aml_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "aml_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "aml_agent_role_supported", "description": "AML agents must use a supported role.", "condition": {"operation": "register_aml_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "aml_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_aml_agent_action_requires_human_approval", "description": "Privileged AML-agent actions require human approval.", "condition": {"operation": "aml_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
	return {"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "configuration": configuration, "configuration_schema": _configuration_schema(), "provides": PROVIDES, "requires": REQUIRES, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-aml/api/v1", "routes": deepcopy(UI_ROUTES), "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"]}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
