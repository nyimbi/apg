"""Executable capability contract for APG Regulatory Technology."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_regtech"
CAPABILITY_NAME = "Regulatory Technology"
CAPABILITY_VERSION = "1.1.0"
REGTECH_EVENT_STREAM = "apg.fintech.regtech.lifecycle"

SUPPORTED_REGULATORY_FRAMEWORKS = ["pci_dss", "psd2", "open_banking", "gdpr", "sox", "basel_iii", "mifid_ii", "aml", "kyc", "data_privacy"]
SUPPORTED_REGULATORS = ["central_bank", "securities_regulator", "data_protection_authority", "financial_conduct_authority", "payments_regulator", "tax_authority"]
SUPPORTED_JURISDICTIONS = ["KE", "US", "GB", "EU", "NG", "GH", "ZA", "SG", "GLOBAL"]
SUPPORTED_CHANGE_TYPES = ["new_rule", "rule_update", "guidance", "enforcement_action", "consultation", "deadline_change"]
SUPPORTED_FILING_TYPES = ["regulatory_return", "incident_notice", "license_update", "audit_response", "prudential_report", "conduct_report"]
SUPPORTED_SUBMISSION_CHANNELS = ["portal", "api", "email", "sftp", "manual"]
SUPPORTED_RISK_RATINGS = ["low", "medium", "high", "critical"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["regulatory_change_reviewer", "filing_preparer", "submission_reviewer", "inquiry_response_agent", "policy_mapping_agent", "regulatory_horizon_agent"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"sources": {"supported_regulators": SUPPORTED_REGULATORS, "supported_jurisdictions": SUPPORTED_JURISDICTIONS, "source_reference_required": True, "owner_required": True, "evidence_required": True},
	"changes": {"supported_frameworks": SUPPORTED_REGULATORY_FRAMEWORKS, "supported_types": SUPPORTED_CHANGE_TYPES, "source_required": True, "effective_date_required": True, "severity_required": True, "evidence_required": True},
	"obligations": {"change_required": True, "obligation_reference_required": True, "policy_reference_required": True, "owner_required": True, "due_date_required": True},
	"impact_assessments": {"change_required": True, "supported_risk_ratings": SUPPORTED_RISK_RATINGS, "impacted_capability_required": True, "reviewer_required": True, "evidence_required": True},
	"filings": {"supported_frameworks": SUPPORTED_REGULATORY_FRAMEWORKS, "supported_types": SUPPORTED_FILING_TYPES, "period_required": True, "owner_required": True, "evidence_required": True},
	"submissions": {"filing_required": True, "supported_channels": SUPPORTED_SUBMISSION_CHANNELS, "submitted_by_required": True, "submitted_at_required": True, "acknowledgment_required": True},
	"inquiries": {"supported_regulators": SUPPORTED_REGULATORS, "supported_severities": SUPPORTED_RISK_RATINGS, "reference_required": True, "due_date_required": True, "evidence_required": True},
	"responses": {"inquiry_required": True, "responder_required": True, "response_reference_required": True, "approval_reference_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True},
	"observability": {"event_stream": REGTECH_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "compliance": "fintech_compliance", "risk": "fintech_risk", "aml": "fintech_aml", "kyc": "fintech_kyc", "reporting": "fin_rpt", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_sources": True, "enable_changes": True, "enable_obligations": True, "enable_impact": True, "enable_filings": True, "enable_submissions": True, "enable_inquiries": True, "enable_responses": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "regtech_control", "allow_tenant_overrides": True},
}

PROVIDES = ["regulatory_source_workflow", "regulatory_change_workflow", "regulatory_obligation_mapping_workflow", "regulatory_policy_mapping_workflow", "regulatory_impact_workflow", "regulatory_filing_workflow", "regulatory_submission_workflow", "regulatory_inquiry_workflow", "regulatory_response_workflow", "regulatory_review_workflow", "regulatory_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_compliance", "fintech_risk", "fintech_aml", "fintech_kyc", "fin_rpt"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-regtech/dashboard", "component": "RegTechDashboard", "permission": "fintech_regtech:view", "nav_group": "Overview"},
	{"name": "sources", "path": "/fintech-regtech/sources", "component": "RegulatorySourceConsole", "permission": "fintech_regtech:sources", "nav_group": "Sources"},
	{"name": "changes", "path": "/fintech-regtech/changes", "component": "RegulatoryChangeQueue", "permission": "fintech_regtech:changes", "nav_group": "Horizon"},
	{"name": "obligations", "path": "/fintech-regtech/obligations", "component": "ObligationMappingConsole", "permission": "fintech_regtech:obligations", "nav_group": "Obligations"},
	{"name": "impact", "path": "/fintech-regtech/impact", "component": "ImpactAssessmentWorkbench", "permission": "fintech_regtech:impact", "nav_group": "Impact"},
	{"name": "filings", "path": "/fintech-regtech/filings", "component": "FilingConsole", "permission": "fintech_regtech:filings", "nav_group": "Filings"},
	{"name": "submissions", "path": "/fintech-regtech/submissions", "component": "SubmissionMonitor", "permission": "fintech_regtech:submissions", "nav_group": "Filings"},
	{"name": "inquiries", "path": "/fintech-regtech/inquiries", "component": "RegulatoryInquiryWorkbench", "permission": "fintech_regtech:inquiries", "nav_group": "Inquiries"},
	{"name": "responses", "path": "/fintech-regtech/responses", "component": "RegulatoryResponseConsole", "permission": "fintech_regtech:responses", "nav_group": "Inquiries"},
	{"name": "reviews", "path": "/fintech-regtech/reviews", "component": "RegTechReviewConsole", "permission": "fintech_regtech:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/fintech-regtech/agents", "component": "RegTechAgentWorkbench", "permission": "fintech_regtech:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-regtech/settings", "component": "RegTechSettings", "permission": "fintech_regtech:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "regtech_control",
	"tokens": {"color.primary": "#075985", "color.accent": "#4F46E5", "color.success": "#15803D", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"sources": {"icon": "satellite-dish", "status_indicator": "source-chip"}, "changes": {"icon": "radar", "status_indicator": "change-chip"}, "obligations": {"icon": "scale", "status_indicator": "obligation-chip"}, "impact": {"icon": "git-branch", "status_indicator": "impact-chip"}, "filings": {"icon": "file-check", "status_indicator": "filing-chip"}, "submissions": {"icon": "send", "status_indicator": "submission-chip"}, "inquiries": {"icon": "message-square-warning", "status_indicator": "inquiry-chip"}, "responses": {"icon": "reply", "status_indicator": "response-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": REGTECH_EVENT_STREAM, "key": "tenant_id", "events": ["regulatory_source_registered", "regulatory_change_recorded", "regulatory_obligation_mapped", "regulatory_impact_assessed", "regulatory_filing_prepared", "regulatory_submission_recorded", "regulatory_inquiry_opened", "regulatory_response_recorded", "regulatory_review_recorded", "regulatory_agent_registered"], "guardrails": ["regtech_batch_requires_bytewax", "privileged_regtech_agent_action_requires_human_approval"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "regtech_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "regtech_policy_required", "required_action": "attach_regtech_policy"}},
	{"name": "source_regulator_supported", "condition": {"operation": "register_source", "regulator_supported": False}, "effect": {"decision": "deny", "reason": "regulator_not_supported", "required_action": "select_supported_regulator"}},
	{"name": "source_jurisdiction_supported", "condition": {"operation": "register_source", "jurisdiction_supported": False}, "effect": {"decision": "deny", "reason": "jurisdiction_not_supported", "required_action": "select_supported_jurisdiction"}},
	{"name": "source_reference_required", "condition": {"operation": "register_source", "source_present": False}, "effect": {"decision": "deny", "reason": "source_reference_required", "required_action": "attach_source_reference"}},
	{"name": "source_owner_required", "condition": {"operation": "register_source", "owner_present": False}, "effect": {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}},
	{"name": "source_evidence_required", "condition": {"operation": "register_source", "evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "change_source_required", "condition": {"operation": "record_change", "source_present": False}, "effect": {"decision": "deny", "reason": "regulatory_source_required", "required_action": "select_source"}},
	{"name": "change_framework_supported", "condition": {"operation": "record_change", "framework_supported": False}, "effect": {"decision": "deny", "reason": "framework_not_supported", "required_action": "select_supported_framework"}},
	{"name": "change_type_supported", "condition": {"operation": "record_change", "change_type_supported": False}, "effect": {"decision": "deny", "reason": "change_type_not_supported", "required_action": "select_supported_change_type"}},
	{"name": "change_effective_date_required", "condition": {"operation": "record_change", "effective_date_present": False}, "effect": {"decision": "deny", "reason": "effective_date_required", "required_action": "set_effective_date"}},
	{"name": "change_severity_supported", "condition": {"operation": "record_change", "severity_supported": False}, "effect": {"decision": "deny", "reason": "severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "change_evidence_required", "condition": {"operation": "record_change", "evidence_present": False}, "effect": {"decision": "deny", "reason": "change_evidence_required", "required_action": "attach_change_evidence"}},
	{"name": "obligation_change_required", "condition": {"operation": "map_obligation", "change_present": False}, "effect": {"decision": "deny", "reason": "regulatory_change_required", "required_action": "select_change"}},
	{"name": "obligation_reference_required", "condition": {"operation": "map_obligation", "obligation_present": False}, "effect": {"decision": "deny", "reason": "obligation_reference_required", "required_action": "attach_obligation_reference"}},
	{"name": "policy_reference_required", "condition": {"operation": "map_obligation", "policy_present": False}, "effect": {"decision": "deny", "reason": "policy_reference_required", "required_action": "attach_policy_reference"}},
	{"name": "obligation_owner_required", "condition": {"operation": "map_obligation", "owner_present": False}, "effect": {"decision": "deny", "reason": "obligation_owner_required", "required_action": "assign_obligation_owner"}},
	{"name": "obligation_due_date_required", "condition": {"operation": "map_obligation", "due_date_present": False}, "effect": {"decision": "deny", "reason": "obligation_due_date_required", "required_action": "set_due_date"}},
	{"name": "impact_change_required", "condition": {"operation": "assess_impact", "change_present": False}, "effect": {"decision": "deny", "reason": "regulatory_change_required", "required_action": "select_change"}},
	{"name": "impact_capability_required", "condition": {"operation": "assess_impact", "impacted_capability_present": False}, "effect": {"decision": "deny", "reason": "impacted_capability_required", "required_action": "attach_impacted_capability"}},
	{"name": "impact_risk_rating_supported", "condition": {"operation": "assess_impact", "risk_rating_supported": False}, "effect": {"decision": "deny", "reason": "risk_rating_not_supported", "required_action": "select_supported_risk_rating"}},
	{"name": "impact_evidence_required", "condition": {"operation": "assess_impact", "evidence_present": False}, "effect": {"decision": "deny", "reason": "impact_evidence_required", "required_action": "attach_impact_evidence"}},
	{"name": "impact_reviewer_required", "condition": {"operation": "assess_impact", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "impact_reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "filing_framework_supported", "condition": {"operation": "prepare_filing", "framework_supported": False}, "effect": {"decision": "deny", "reason": "framework_not_supported", "required_action": "select_supported_framework"}},
	{"name": "filing_type_supported", "condition": {"operation": "prepare_filing", "filing_type_supported": False}, "effect": {"decision": "deny", "reason": "filing_type_not_supported", "required_action": "select_supported_filing_type"}},
	{"name": "filing_period_required", "condition": {"operation": "prepare_filing", "period_present": False}, "effect": {"decision": "deny", "reason": "filing_period_required", "required_action": "set_period"}},
	{"name": "filing_evidence_required", "condition": {"operation": "prepare_filing", "evidence_present": False}, "effect": {"decision": "deny", "reason": "filing_evidence_required", "required_action": "attach_filing_evidence"}},
	{"name": "filing_owner_required", "condition": {"operation": "prepare_filing", "owner_present": False}, "effect": {"decision": "deny", "reason": "filing_owner_required", "required_action": "assign_filing_owner"}},
	{"name": "submission_filing_required", "condition": {"operation": "record_submission", "filing_present": False}, "effect": {"decision": "deny", "reason": "filing_required", "required_action": "select_filing"}},
	{"name": "submission_channel_supported", "condition": {"operation": "record_submission", "channel_supported": False}, "effect": {"decision": "deny", "reason": "submission_channel_not_supported", "required_action": "select_supported_submission_channel"}},
	{"name": "submission_submitter_required", "condition": {"operation": "record_submission", "submitted_by_present": False}, "effect": {"decision": "deny", "reason": "submitted_by_required", "required_action": "record_submitter"}},
	{"name": "submission_timestamp_required", "condition": {"operation": "record_submission", "submitted_at_present": False}, "effect": {"decision": "deny", "reason": "submitted_at_required", "required_action": "record_submission_time"}},
	{"name": "submission_acknowledgment_required", "condition": {"operation": "record_submission", "acknowledgment_present": False}, "effect": {"decision": "deny", "reason": "acknowledgment_required", "required_action": "attach_acknowledgment"}},
	{"name": "inquiry_regulator_supported", "condition": {"operation": "open_inquiry", "regulator_supported": False}, "effect": {"decision": "deny", "reason": "regulator_not_supported", "required_action": "select_supported_regulator"}},
	{"name": "inquiry_reference_required", "condition": {"operation": "open_inquiry", "reference_present": False}, "effect": {"decision": "deny", "reason": "inquiry_reference_required", "required_action": "attach_inquiry_reference"}},
	{"name": "inquiry_severity_supported", "condition": {"operation": "open_inquiry", "severity_supported": False}, "effect": {"decision": "deny", "reason": "severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "inquiry_due_date_required", "condition": {"operation": "open_inquiry", "due_date_present": False}, "effect": {"decision": "deny", "reason": "inquiry_due_date_required", "required_action": "set_due_date"}},
	{"name": "inquiry_evidence_required", "condition": {"operation": "open_inquiry", "evidence_present": False}, "effect": {"decision": "deny", "reason": "inquiry_evidence_required", "required_action": "attach_inquiry_evidence"}},
	{"name": "response_inquiry_required", "condition": {"operation": "record_response", "inquiry_present": False}, "effect": {"decision": "deny", "reason": "inquiry_required", "required_action": "select_inquiry"}},
	{"name": "response_responder_required", "condition": {"operation": "record_response", "responder_present": False}, "effect": {"decision": "deny", "reason": "responder_required", "required_action": "assign_responder"}},
	{"name": "response_reference_required", "condition": {"operation": "record_response", "response_present": False}, "effect": {"decision": "deny", "reason": "response_reference_required", "required_action": "attach_response"}},
	{"name": "response_approval_required", "condition": {"operation": "record_response", "approval_present": False}, "effect": {"decision": "deny", "reason": "response_approval_required", "required_action": "attach_response_approval"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "regtech_batch_requires_bytewax", "condition": {"operation": "regtech_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_regtech_batch_to_bytewax"}},
	{"name": "regtech_agent_runtime_supported", "condition": {"operation": "register_regtech_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "regtech_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "regtech_agent_role_supported", "condition": {"operation": "register_regtech_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "regtech_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_regtech_agent_action_requires_human_approval", "condition": {"operation": "regtech_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-regtech/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
