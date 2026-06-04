"""Executable capability contract for APG Government Contracts & Procurement."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "government_con"
CAPABILITY_NAME = "Government Contracts and Procurement"
CAPABILITY_VERSION = "1.0.0"
CON_EVENT_STREAM = "apg.government.con.lifecycle"

SUPPORTED_PROCUREMENT_METHODS = ["open_tender", "restricted_tender", "direct_procurement", "request_for_quotation", "framework_agreement", "emergency_procurement", "design_competition"]
SUPPORTED_TENDER_STATUSES = ["draft", "published", "clarification", "evaluation", "awarded", "cancelled", "suspended"]
SUPPORTED_EVALUATION_CRITERIA = ["price", "technical_quality", "past_performance", "financial_capacity", "local_content", "combined_score"]
SUPPORTED_CONTRACT_TYPES = ["supply", "works", "consultancy", "non_consultancy_services", "framework", "concession", "ppp"]
SUPPORTED_CONTRACT_STATUSES = ["draft", "signed", "active", "suspended", "varied", "completed", "terminated", "disputed"]
SUPPORTED_VARIATION_TYPES = ["scope_change", "time_extension", "price_adjustment", "specification_change", "force_majeure"]
SUPPORTED_PERFORMANCE_STATUSES = ["on_track", "at_risk", "delayed", "in_dispute", "completed_satisfactorily", "completed_unsatisfactorily"]
SUPPORTED_PPDA_THRESHOLDS = ["micro", "small", "medium", "large", "strategic"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["tender_analyst", "evaluation_reviewer", "contract_monitor", "compliance_checker", "award_recorder"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"tenders": {
		"supported_procurement_methods": SUPPORTED_PROCUREMENT_METHODS,
		"supported_tender_statuses": SUPPORTED_TENDER_STATUSES,
		"ppda_threshold_required": True,
		"approver_required": True,
		"evidence_required": True,
	},
	"evaluations": {
		"supported_evaluation_criteria": SUPPORTED_EVALUATION_CRITERIA,
		"tender_required": True,
		"evaluator_required": True,
		"score_required": True,
		"evidence_required": True,
	},
	"awards": {
		"tender_required": True,
		"approved_evaluation_required": True,
		"ppda_notification_required": True,
		"evidence_required": True,
	},
	"contracts": {
		"supported_contract_types": SUPPORTED_CONTRACT_TYPES,
		"supported_contract_statuses": SUPPORTED_CONTRACT_STATUSES,
		"award_required": True,
		"signed_by_required": True,
		"evidence_required": True,
	},
	"variations": {
		"supported_variation_types": SUPPORTED_VARIATION_TYPES,
		"contract_required": True,
		"approval_required": True,
		"ppda_notification_required": True,
		"evidence_required": True,
	},
	"performance": {
		"supported_performance_statuses": SUPPORTED_PERFORMANCE_STATUSES,
		"contract_required": True,
		"reviewer_required": True,
		"evidence_required": True,
	},
	"ppda_compliance": {
		"supported_thresholds": SUPPORTED_PPDA_THRESHOLDS,
		"annual_report_required": True,
		"debarment_register_enabled": True,
	},
	"reviews": {
		"supported_statuses": SUPPORTED_REVIEW_STATUSES,
		"reviewer_required": True,
		"evidence_required": True,
	},
	"agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AGENT_ROLES,
		"name_required": True,
		"scope_required": True,
		"human_approval_required_for_privileged_actions": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"award_without_evaluation_denied": True,
		"single_source_requires_justification": True,
		"contract_variation_limit_enforced": True,
		"debarred_bidder_denied": True,
		"ppda_threshold_compliance_required": True,
		"evidence_fabrication_denied": True,
	},
	"observability": {"event_stream": CON_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"workflow": "wflo",
		"compliance": "comp",
		"monitoring": "moni",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_tenders": True,
		"enable_evaluations": True,
		"enable_awards": True,
		"enable_contracts": True,
		"enable_variations": True,
		"enable_performance": True,
		"enable_ppda_compliance": True,
		"enable_reviews": True,
		"enable_agents": True,
	},
	"theme": {"default_theme": "government_con_control", "allow_tenant_overrides": True},
}

PROVIDES = [
	"tender_management_workflow",
	"evaluation_workflow",
	"contract_award_workflow",
	"contract_lifecycle_workflow",
	"contract_variation_workflow",
	"contract_performance_workflow",
	"ppda_compliance_workflow",
	"procurement_review_workflow",
	"procurement_agent_workflow",
	"debarment_register_workflow",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/government-con/dashboard", "component": "ProcurementDashboard", "permission": "government_con:view", "nav_group": "Overview"},
	{"name": "tenders", "path": "/government-con/tenders", "component": "TenderManagementConsole", "permission": "government_con:tenders", "nav_group": "Procurement"},
	{"name": "evaluations", "path": "/government-con/evaluations", "component": "EvaluationWorkbench", "permission": "government_con:evaluate", "nav_group": "Procurement"},
	{"name": "awards", "path": "/government-con/awards", "component": "AwardConsole", "permission": "government_con:award", "nav_group": "Procurement"},
	{"name": "contracts", "path": "/government-con/contracts", "component": "ContractLedger", "permission": "government_con:contracts", "nav_group": "Contracts"},
	{"name": "variations", "path": "/government-con/variations", "component": "ContractVariationConsole", "permission": "government_con:vary", "nav_group": "Contracts"},
	{"name": "performance", "path": "/government-con/performance", "component": "ContractPerformanceConsole", "permission": "government_con:performance", "nav_group": "Monitoring"},
	{"name": "ppda", "path": "/government-con/ppda", "component": "PpdaComplianceConsole", "permission": "government_con:ppda", "nav_group": "Compliance"},
	{"name": "debarment", "path": "/government-con/debarment", "component": "DebarmentRegister", "permission": "government_con:debarment", "nav_group": "Compliance"},
	{"name": "reviews", "path": "/government-con/reviews", "component": "ProcurementReviewConsole", "permission": "government_con:review", "nav_group": "Governance"},
	{"name": "agents", "path": "/government-con/agents", "component": "ProcurementAgentWorkbench", "permission": "government_con:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/government-con/settings", "component": "ProcurementSettings", "permission": "government_con:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "government_con_control",
	"tokens": {
		"color.primary": "#4338CA",
		"color.accent": "#0F766E",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#991B1B",
		"surface.canvas": "#F5F3FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E1B4B",
		"text.secondary": "#475569",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"tenders": {"icon": "file-search", "status_indicator": "tender-status-chip"},
		"evaluations": {"icon": "scale", "status_indicator": "evaluation-score-chip"},
		"awards": {"icon": "award", "status_indicator": "award-status-chip"},
		"contracts": {"icon": "file-signature", "status_indicator": "contract-status-chip"},
		"variations": {"icon": "git-merge", "status_indicator": "variation-type-chip"},
		"performance": {"icon": "activity", "status_indicator": "performance-status-chip"},
		"ppda": {"icon": "shield", "status_indicator": "ppda-threshold-chip"},
		"reviews": {"icon": "clipboard-check", "status_indicator": "review-status-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": CON_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"tender_published",
		"tender_awarded",
		"contract_signed",
		"contract_varied",
		"contract_performance_recorded",
		"ppda_submission_recorded",
		"bidder_debarred",
		"procurement_review_recorded",
		"procurement_agent_registered",
		"tender_cancelled",
	],
	"guardrails": [
		"con_batch_requires_bytewax",
		"award_without_evaluation_denied",
		"single_source_requires_justification",
		"debarred_bidder_denied",
		"ppda_threshold_compliance_required",
		"evidence_fabrication_denied",
		"privileged_procurement_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "con_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "procurement_policy_required", "required_action": "attach_procurement_policy"}},
	{"name": "procurement_method_supported", "condition": {"operation": "publish_tender", "procurement_method_supported": False}, "effect": {"decision": "deny", "reason": "procurement_method_not_supported", "required_action": "select_supported_procurement_method"}},
	{"name": "tender_ppda_threshold_required", "condition": {"operation": "publish_tender", "ppda_threshold_present": False}, "effect": {"decision": "deny", "reason": "ppda_threshold_required", "required_action": "set_ppda_threshold"}},
	{"name": "tender_approver_required", "condition": {"operation": "publish_tender", "approver_present": False}, "effect": {"decision": "deny", "reason": "approver_required", "required_action": "assign_approver"}},
	{"name": "tender_evidence_required", "condition": {"operation": "publish_tender", "evidence_present": False}, "effect": {"decision": "deny", "reason": "tender_evidence_required", "required_action": "attach_tender_evidence"}},
	{"name": "single_source_requires_justification", "condition": {"operation": "publish_tender", "procurement_method": "direct_procurement", "justification_present": False}, "effect": {"decision": "deny", "reason": "single_source_justification_required", "required_action": "attach_justification"}},
	{"name": "evaluation_tender_required", "condition": {"operation": "record_evaluation", "tender_present": False}, "effect": {"decision": "deny", "reason": "tender_required", "required_action": "select_tender"}},
	{"name": "evaluation_criteria_supported", "condition": {"operation": "record_evaluation", "criteria_supported": False}, "effect": {"decision": "deny", "reason": "evaluation_criteria_not_supported", "required_action": "select_supported_criteria"}},
	{"name": "evaluation_score_required", "condition": {"operation": "record_evaluation", "score_present": False}, "effect": {"decision": "deny", "reason": "evaluation_score_required", "required_action": "provide_evaluation_score"}},
	{"name": "evaluation_evaluator_required", "condition": {"operation": "record_evaluation", "evaluator_present": False}, "effect": {"decision": "deny", "reason": "evaluator_required", "required_action": "assign_evaluator"}},
	{"name": "evaluation_evidence_required", "condition": {"operation": "record_evaluation", "evidence_present": False}, "effect": {"decision": "deny", "reason": "evaluation_evidence_required", "required_action": "attach_evaluation_evidence"}},
	{"name": "award_evaluation_required", "condition": {"operation": "record_award", "approved_evaluation_present": False}, "effect": {"decision": "deny", "reason": "approved_evaluation_required", "required_action": "complete_evaluation"}},
	{"name": "award_ppda_notification_required", "condition": {"operation": "record_award", "ppda_notification_present": False}, "effect": {"decision": "deny", "reason": "ppda_notification_required", "required_action": "notify_ppda"}},
	{"name": "award_evidence_required", "condition": {"operation": "record_award", "evidence_present": False}, "effect": {"decision": "deny", "reason": "award_evidence_required", "required_action": "attach_award_evidence"}},
	{"name": "contract_type_supported", "condition": {"operation": "record_contract", "contract_type_supported": False}, "effect": {"decision": "deny", "reason": "contract_type_not_supported", "required_action": "select_supported_contract_type"}},
	{"name": "contract_award_required", "condition": {"operation": "record_contract", "award_present": False}, "effect": {"decision": "deny", "reason": "award_required", "required_action": "select_award"}},
	{"name": "contract_signed_by_required", "condition": {"operation": "record_contract", "signed_by_present": False}, "effect": {"decision": "deny", "reason": "signed_by_required", "required_action": "record_signatories"}},
	{"name": "contract_evidence_required", "condition": {"operation": "record_contract", "evidence_present": False}, "effect": {"decision": "deny", "reason": "contract_evidence_required", "required_action": "attach_contract_evidence"}},
	{"name": "debarred_bidder_denied", "condition": {"operation": "record_evaluation", "bidder_debarred": True}, "effect": {"decision": "deny", "reason": "debarred_bidder_denied", "required_action": "exclude_debarred_bidder"}},
	{"name": "variation_type_supported", "condition": {"operation": "record_variation", "variation_type_supported": False}, "effect": {"decision": "deny", "reason": "variation_type_not_supported", "required_action": "select_supported_variation_type"}},
	{"name": "variation_approval_required", "condition": {"operation": "record_variation", "approval_present": False}, "effect": {"decision": "deny", "reason": "variation_approval_required", "required_action": "obtain_variation_approval"}},
	{"name": "variation_ppda_notification_required", "condition": {"operation": "record_variation", "ppda_notification_present": False}, "effect": {"decision": "deny", "reason": "ppda_variation_notification_required", "required_action": "notify_ppda_of_variation"}},
	{"name": "performance_status_supported", "condition": {"operation": "record_performance", "performance_status_supported": False}, "effect": {"decision": "deny", "reason": "performance_status_not_supported", "required_action": "select_supported_performance_status"}},
	{"name": "con_batch_requires_bytewax", "condition": {"operation": "con_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_con_batch_to_bytewax"}},
	{"name": "con_agent_runtime_supported", "condition": {"operation": "register_con_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "con_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "con_agent_role_supported", "condition": {"operation": "register_con_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "con_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "con_agent_name_required", "condition": {"operation": "register_con_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "con_agent_name_required", "required_action": "name_con_agent"}},
	{"name": "con_agent_scope_required", "condition": {"operation": "register_con_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "con_agent_scope_required", "required_action": "bound_con_agent_scope"}},
	{"name": "privileged_con_agent_action_requires_human_approval", "condition": {"operation": "con_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "evidence_fabrication_denied", "condition": {"operation": "con_agent_action", "evidence_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "evidence_fabrication_denied", "required_action": "remove_evidence_fabrication_scope"}},
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
		"configuration": configuration,
		"configuration_schema": {
			"type": "object",
			"required": list(configuration),
			"properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/government-con/api/v1",
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
