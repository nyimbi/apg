"""Executable capability contract for Risk and Compliance Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "grc_rcm"
CAPABILITY_NAME = "Risk and Compliance Management"
CAPABILITY_VERSION = "2.1.0"
RCM_EVENT_STREAM = "apg.grc.rcm.lifecycle"

SUPPORTED_RISK_CATEGORIES = ["operational", "financial", "technology", "regulatory", "third_party", "strategic", "security", "privacy"]
SUPPORTED_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_CONTROL_TYPES = ["preventive", "detective", "corrective", "directive"]
SUPPORTED_ASSESSMENT_RESULTS = ["effective", "partially_effective", "ineffective"]
SUPPORTED_ISSUE_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_EXCEPTION_TYPES = ["risk_acceptance", "policy_exception", "control_waiver", "deadline_extension"]
SUPPORTED_RCM_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_RCM_AGENT_ROLES = [
	"risk_reviewer",
	"control_reviewer",
	"compliance_reviewer",
	"evidence_reviewer",
	"issue_reviewer",
	"governance_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"risks": {
		"title_required": True,
		"owner_required": True,
		"category_required": True,
		"supported_categories": SUPPORTED_RISK_CATEGORIES,
		"likelihood_range": [0, 1],
		"impact_range": [0, 1],
		"high_risk_review_required": True,
	},
	"controls": {
		"name_required": True,
		"owner_required": True,
		"control_type_required": True,
		"supported_control_types": SUPPORTED_CONTROL_TYPES,
		"mapped_risk_required": True,
		"test_frequency_required": True,
	},
	"obligations": {
		"framework_required": True,
		"requirement_required": True,
		"owner_required": True,
		"jurisdiction_required": True,
		"due_date_required": True,
		"mapped_control_required": True,
	},
	"assessments": {
		"control_required": True,
		"assessor_required": True,
		"supported_results": SUPPORTED_ASSESSMENT_RESULTS,
		"ineffective_control_requires_evidence": True,
		"ineffective_control_opens_issue_workflow": True,
	},
	"evidence": {
		"source_required": True,
		"linked_record_required": True,
		"encryption_required": True,
		"minimum_retention_days": 365,
	},
	"issues": {
		"title_required": True,
		"severity_required": True,
		"supported_severities": SUPPORTED_ISSUE_SEVERITIES,
		"owner_required": True,
		"remediation_plan_required": True,
		"high_severity_review_required": True,
	},
	"governance_decisions": {
		"title_required": True,
		"approver_required": True,
		"rationale_required": True,
		"linked_risk_review_required": True,
	},
	"exceptions": {
		"exception_type_required": True,
		"supported_exception_types": SUPPORTED_EXCEPTION_TYPES,
		"expiration_required": True,
		"approval_required": True,
	},
	"rcm_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_RCM_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_RCM_AGENT_ROLES,
		"max_autonomous_scope": "recommend_validate_and_prepare",
		"human_approval_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_state_changes": True,
		"segregation_of_duties": True,
	},
	"observability": {
		"event_stream": RCM_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_risk_events": True,
		"emit_control_events": True,
		"emit_obligation_events": True,
		"emit_assessment_events": True,
		"emit_evidence_events": True,
		"emit_issue_events": True,
		"emit_governance_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"notification": "adapter",
		"document_management": "adapter",
		"business_intelligence": "adapter",
		"policy_management": "adapter",
		"workflow_orchestration": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_risks": True,
		"enable_controls": True,
		"enable_obligations": True,
		"enable_assessments": True,
		"enable_evidence": True,
		"enable_issues": True,
		"enable_governance": True,
		"enable_exceptions": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {
		"default_theme": "grc_rcm_control",
		"allow_tenant_overrides": True,
	},
}


PROVIDES = [
	"risk_register_lifecycle",
	"control_library_lifecycle",
	"compliance_obligation_lifecycle",
	"control_assessment_workflow",
	"evidence_management_workflow",
	"issue_remediation_workflow",
	"governance_decision_workflow",
	"exception_management_workflow",
	"rcm_dashboard_service",
	"rcm_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"composition_events",
	"composition_config",
	"document_management",
	"business_intelligence",
	"workflow_orchestration",
	"policy_management",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/grc-rcm/dashboard", "component": "RcmDashboard", "permission": "grc_rcm:view", "nav_group": "Overview"},
	{"name": "risks", "path": "/grc-rcm/risks", "component": "RiskRegisterWorkbench", "permission": "grc_rcm:manage_risks", "nav_group": "Risk"},
	{"name": "controls", "path": "/grc-rcm/controls", "component": "ControlLibraryWorkbench", "permission": "grc_rcm:manage_controls", "nav_group": "Controls"},
	{"name": "obligations", "path": "/grc-rcm/obligations", "component": "ComplianceObligationWorkbench", "permission": "grc_rcm:manage_obligations", "nav_group": "Compliance"},
	{"name": "assessments", "path": "/grc-rcm/assessments", "component": "ControlAssessmentWorkbench", "permission": "grc_rcm:assess_controls", "nav_group": "Controls"},
	{"name": "evidence", "path": "/grc-rcm/evidence", "component": "EvidenceVault", "permission": "grc_rcm:manage_evidence", "nav_group": "Compliance"},
	{"name": "issues", "path": "/grc-rcm/issues", "component": "IssueRemediationBoard", "permission": "grc_rcm:manage_issues", "nav_group": "Remediation"},
	{"name": "governance", "path": "/grc-rcm/governance", "component": "GovernanceDecisionBoard", "permission": "grc_rcm:govern", "nav_group": "Governance"},
	{"name": "exceptions", "path": "/grc-rcm/exceptions", "component": "ExceptionWorkbench", "permission": "grc_rcm:manage_exceptions", "nav_group": "Governance"},
	{"name": "agents", "path": "/grc-rcm/agents", "component": "RcmAgentWorkbench", "permission": "grc_rcm:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/grc-rcm/settings", "component": "RcmSettings", "permission": "grc_rcm:admin", "nav_group": "Administration"},
]


THEME = {
	"name": "grc_rcm_control",
	"tokens": {
		"color.primary": "#2F4A60",
		"color.accent": "#B7791F",
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
		"risks": {"icon": "shield-alert", "status_indicator": "risk-pill", "risk_style": "heat-band"},
		"controls": {"visual": "control-grid", "status_style": "control-chip"},
		"obligations": {"visual": "obligation-list", "status_style": "deadline-chip"},
		"assessments": {"visual": "assessment-queue", "status_style": "test-chip"},
		"evidence": {"visual": "evidence-vault", "status_style": "retention-chip"},
		"issues": {"visual": "remediation-board", "status_style": "severity-chip"},
		"governance": {"visual": "decision-ledger", "status_style": "approval-chip"},
		"exceptions": {"visual": "exception-register", "status_style": "expiry-chip"},
		"agents": {"visual": "review-lane", "status_style": "agent-chip"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"stream": RCM_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"risk_registered",
		"control_registered",
		"obligation_registered",
		"control_assessed",
		"evidence_collected",
		"issue_opened",
		"issue_remediated",
		"governance_decision_recorded",
		"exception_registered",
		"rcm_agent_registered",
	],
	"states": ["draft", "active", "review", "effective", "ineffective", "open", "remediated", "approved", "expired", "blocked"],
	"guardrails": [
		"rcm_batch_requires_bytewax",
		"rcm_event_requires_bytewax",
		"privileged_rcm_agent_action_requires_human_approval",
	],
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "RCM operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "rcm_write_requires_policy", "description": "RCM writes require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "risk_requires_title", "description": "Risks require title.", "condition": {"operation": "register_risk", "title_present": False}, "effect": {"decision": "deny", "reason": "risk_title_required", "required_action": "set_risk_title"}},
	{"name": "risk_requires_owner", "description": "Risks require owner.", "condition": {"operation": "register_risk", "owner_present": False}, "effect": {"decision": "deny", "reason": "risk_owner_required", "required_action": "assign_risk_owner"}},
	{"name": "risk_category_supported", "description": "Risk category must be supported.", "condition": {"operation": "register_risk", "risk_category_supported": False}, "effect": {"decision": "deny", "reason": "risk_category_not_supported", "required_action": "select_supported_risk_category"}},
	{"name": "risk_likelihood_in_range", "description": "Risk likelihood must be 0..1.", "condition": {"operation": "register_risk", "likelihood_in_range": False}, "effect": {"decision": "deny", "reason": "risk_likelihood_out_of_range", "required_action": "set_likelihood_range"}},
	{"name": "risk_impact_in_range", "description": "Risk impact must be 0..1.", "condition": {"operation": "register_risk", "impact_in_range": False}, "effect": {"decision": "deny", "reason": "risk_impact_out_of_range", "required_action": "set_impact_range"}},
	{"name": "high_risk_requires_review", "description": "High or critical risk requires review.", "condition": {"operation": "register_risk", "high_risk": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "high_risk_review_required", "required_action": "record_risk_review"}},
	{"name": "control_requires_name", "description": "Controls require name.", "condition": {"operation": "register_control", "name_present": False}, "effect": {"decision": "deny", "reason": "control_name_required", "required_action": "set_control_name"}},
	{"name": "control_requires_owner", "description": "Controls require owner.", "condition": {"operation": "register_control", "owner_present": False}, "effect": {"decision": "deny", "reason": "control_owner_required", "required_action": "assign_control_owner"}},
	{"name": "control_type_supported", "description": "Control type must be supported.", "condition": {"operation": "register_control", "control_type_supported": False}, "effect": {"decision": "deny", "reason": "control_type_not_supported", "required_action": "select_supported_control_type"}},
	{"name": "control_requires_mapped_risk", "description": "Controls require mapped risk.", "condition": {"operation": "register_control", "mapped_risk_present": False}, "effect": {"decision": "deny", "reason": "mapped_risk_required", "required_action": "map_control_to_risk"}},
	{"name": "control_frequency_positive", "description": "Control test frequency must be positive.", "condition": {"operation": "register_control", "test_frequency_days_lte": 0}, "effect": {"decision": "deny", "reason": "control_test_frequency_required", "required_action": "set_test_frequency"}},
	{"name": "obligation_requires_framework", "description": "Obligations require framework.", "condition": {"operation": "register_obligation", "framework_present": False}, "effect": {"decision": "deny", "reason": "framework_required", "required_action": "set_framework"}},
	{"name": "obligation_requires_requirement", "description": "Obligations require requirement.", "condition": {"operation": "register_obligation", "requirement_present": False}, "effect": {"decision": "deny", "reason": "requirement_required", "required_action": "set_requirement"}},
	{"name": "obligation_requires_owner", "description": "Obligations require owner.", "condition": {"operation": "register_obligation", "owner_present": False}, "effect": {"decision": "deny", "reason": "obligation_owner_required", "required_action": "assign_obligation_owner"}},
	{"name": "obligation_requires_jurisdiction", "description": "Obligations require jurisdiction.", "condition": {"operation": "register_obligation", "jurisdiction_present": False}, "effect": {"decision": "deny", "reason": "obligation_jurisdiction_required", "required_action": "set_jurisdiction"}},
	{"name": "obligation_requires_due_date", "description": "Obligations require due date.", "condition": {"operation": "register_obligation", "due_date_present": False}, "effect": {"decision": "deny", "reason": "obligation_due_date_required", "required_action": "set_due_date"}},
	{"name": "obligation_requires_control", "description": "Obligations require mapped control.", "condition": {"operation": "register_obligation", "mapped_control_present": False}, "effect": {"decision": "deny", "reason": "mapped_control_required", "required_action": "map_obligation_to_control"}},
	{"name": "assessment_requires_control", "description": "Assessments require control.", "condition": {"operation": "assess_control", "control_present": False}, "effect": {"decision": "deny", "reason": "control_required", "required_action": "select_control"}},
	{"name": "assessment_requires_assessor", "description": "Assessments require assessor.", "condition": {"operation": "assess_control", "assessor_present": False}, "effect": {"decision": "deny", "reason": "assessor_required", "required_action": "assign_assessor"}},
	{"name": "assessment_result_supported", "description": "Assessment result must be supported.", "condition": {"operation": "assess_control", "assessment_result_supported": False}, "effect": {"decision": "deny", "reason": "assessment_result_not_supported", "required_action": "select_supported_assessment_result"}},
	{"name": "failed_assessment_requires_evidence", "description": "Ineffective assessments require evidence.", "condition": {"operation": "assess_control", "failed_assessment": True, "evidence_present": False}, "effect": {"decision": "deny", "reason": "failed_assessment_evidence_required", "required_action": "attach_evidence"}},
	{"name": "evidence_requires_source", "description": "Evidence requires source.", "condition": {"operation": "collect_evidence", "source_present": False}, "effect": {"decision": "deny", "reason": "evidence_source_required", "required_action": "set_evidence_source"}},
	{"name": "evidence_requires_link", "description": "Evidence requires linked record.", "condition": {"operation": "collect_evidence", "linked_record_present": False}, "effect": {"decision": "deny", "reason": "evidence_link_required", "required_action": "link_evidence"}},
	{"name": "evidence_requires_encryption", "description": "Evidence must be encrypted.", "condition": {"operation": "collect_evidence", "encrypted": False}, "effect": {"decision": "deny", "reason": "evidence_encryption_required", "required_action": "encrypt_evidence"}},
	{"name": "evidence_retention_minimum", "description": "Evidence retention must meet minimum.", "condition": {"operation": "collect_evidence", "retention_days_lt": 365}, "effect": {"decision": "deny", "reason": "evidence_retention_too_short", "required_action": "increase_retention"}},
	{"name": "issue_requires_title", "description": "Issues require title.", "condition": {"operation": "open_issue", "title_present": False}, "effect": {"decision": "deny", "reason": "issue_title_required", "required_action": "set_issue_title"}},
	{"name": "issue_severity_supported", "description": "Issue severity must be supported.", "condition": {"operation": "open_issue", "issue_severity_supported": False}, "effect": {"decision": "deny", "reason": "issue_severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "issue_requires_owner", "description": "Issues require owner.", "condition": {"operation": "open_issue", "owner_present": False}, "effect": {"decision": "deny", "reason": "issue_owner_required", "required_action": "assign_issue_owner"}},
	{"name": "issue_requires_remediation_plan", "description": "Issues require remediation plan.", "condition": {"operation": "open_issue", "remediation_plan_present": False}, "effect": {"decision": "deny", "reason": "remediation_plan_required", "required_action": "add_remediation_plan"}},
	{"name": "high_severity_issue_requires_review", "description": "High severity issue requires review.", "condition": {"operation": "open_issue", "high_severity": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "issue_review_required", "required_action": "record_issue_review"}},
	{"name": "issue_remediation_requires_issue", "description": "Issue remediation requires issue.", "condition": {"operation": "remediate_issue", "issue_present": False}, "effect": {"decision": "deny", "reason": "issue_required", "required_action": "select_issue"}},
	{"name": "issue_remediation_requires_evidence", "description": "Issue remediation requires evidence.", "condition": {"operation": "remediate_issue", "remediation_evidence_present": False}, "effect": {"decision": "deny", "reason": "remediation_evidence_required", "required_action": "attach_remediation_evidence"}},
	{"name": "governance_requires_title", "description": "Governance decisions require title.", "condition": {"operation": "record_governance_decision", "title_present": False}, "effect": {"decision": "deny", "reason": "governance_title_required", "required_action": "set_governance_title"}},
	{"name": "governance_requires_approver", "description": "Governance decisions require approver.", "condition": {"operation": "record_governance_decision", "approver_present": False}, "effect": {"decision": "deny", "reason": "approver_required", "required_action": "assign_approver"}},
	{"name": "governance_requires_rationale", "description": "Governance decisions require rationale.", "condition": {"operation": "record_governance_decision", "rationale_present": False}, "effect": {"decision": "deny", "reason": "rationale_required", "required_action": "record_rationale"}},
	{"name": "high_risk_governance_requires_review", "description": "High-risk governance decisions require review.", "condition": {"operation": "record_governance_decision", "high_risk": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "governance_review_required", "required_action": "record_governance_review"}},
	{"name": "exception_type_supported", "description": "Exception type must be supported.", "condition": {"operation": "register_exception", "exception_type_supported": False}, "effect": {"decision": "deny", "reason": "exception_type_not_supported", "required_action": "select_supported_exception_type"}},
	{"name": "exception_requires_expiration", "description": "Exceptions require expiration.", "condition": {"operation": "register_exception", "expiration_present": False}, "effect": {"decision": "deny", "reason": "exception_expiration_required", "required_action": "set_exception_expiration"}},
	{"name": "exception_requires_approval", "description": "Exceptions require approval.", "condition": {"operation": "register_exception", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "exception_approval_required", "required_action": "record_exception_approval"}},
	{"name": "rcm_batch_requires_bytewax", "description": "RCM batches require Bytewax coordination.", "condition": {"operation": "rcm_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_rcm_batch_to_bytewax"}},
	{"name": "rcm_event_requires_bytewax", "description": "RCM events require Bytewax.", "condition": {"operation": "rcm_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_rcm_event_to_bytewax"}},
	{"name": "rcm_agent_runtime_supported", "description": "RCM agents must use an approved runtime.", "condition": {"operation": "register_rcm_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "rcm_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "rcm_agent_role_supported", "description": "RCM agents must use an approved role.", "condition": {"operation": "register_rcm_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "rcm_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_rcm_agent_action_requires_human_approval", "description": "Privileged RCM actions proposed by agents require human approval.", "condition": {"operation": "rcm_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
		"name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": configuration,
		"configuration_schema": _configuration_schema(),
		"provides": PROVIDES,
		"requires": REQUIRES,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/grc-rcm/api/v1",
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
