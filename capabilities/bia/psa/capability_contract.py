"""Executable capability contract for APG Prescriptive Analytics (bia_psa)."""

from __future__ import annotations
from copy import deepcopy
from typing import Any

CAPABILITY_ID = "bia_psa"
CAPABILITY_NAME = "Prescriptive Analytics"
CAPABILITY_VERSION = "1.0.0"
PSA_EVENT_STREAM = "apg.bia.psa.lifecycle"

SUPPORTED_OPTIMISATION_TYPES = ["linear_programming", "integer_programming", "constraint_satisfaction", "genetic_algorithm", "simulated_annealing", "reinforcement_learning", "multi_objective"]
SUPPORTED_DECISION_TYPES = ["binary", "multi_class", "ranking", "allocation", "scheduling", "routing"]
SUPPORTED_RECOMMENDATION_TYPES = ["action", "allocation", "configuration", "process_change", "investment", "risk_mitigation"]
SUPPORTED_WHATIF_PARAMETER_TYPES = ["continuous", "discrete", "boolean", "categorical", "range"]
SUPPORTED_CONSTRAINT_TYPES = ["hard", "soft", "preference"]
SUPPORTED_OBJECTIVE_TYPES = ["minimise", "maximise", "satisfice", "balance"]
SUPPORTED_ANALYSIS_STATES = ["draft", "running", "completed", "failed", "archived"]
SUPPORTED_APPROVAL_STATES = ["pending", "approved", "rejected", "auto_approved"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["optimisation_designer", "decision_analyst", "recommendation_reviewer", "whatif_modeller", "constraint_steward"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"optimisation": {"supported_types": SUPPORTED_OPTIMISATION_TYPES, "max_variables": 10000, "max_constraints": 50000, "require_owner": True},
	"decisions": {"supported_types": SUPPORTED_DECISION_TYPES, "require_explainability": True, "audit_all_decisions": True},
	"recommendations": {"supported_types": SUPPORTED_RECOMMENDATION_TYPES, "require_approval": True, "max_recommendations_per_run": 50},
	"whatif": {"supported_parameter_types": SUPPORTED_WHATIF_PARAMETER_TYPES, "max_scenarios": 20, "require_baseline": True},
	"constraints": {"supported_types": SUPPORTED_CONSTRAINT_TYPES},
	"objectives": {"supported_types": SUPPORTED_OBJECTIVE_TYPES},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_analysis_denied": True, "unapproved_recommendation_action_denied": True},
	"observability": {"event_stream": PSA_EVENT_STREAM, "stream_processor": "bytewax"},
	"theme": {"default_theme": "bia_psa_prescriptive", "allow_tenant_overrides": True},
}

PROVIDES = ["optimisation_engine", "decision_support_system", "recommendation_actions", "whatif_analysis", "constraint_management", "multi_objective_analysis", "allocation_optimisation", "process_improvement_recommendations"]

REQUIRES = ["auth", "audl", "mten", "conf", "mqeb", "moni", "wflo", "bia_pda"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/bia/psa/dashboard", "component": "PrescriptiveDashboard", "permission": "bia_psa:view", "nav_group": "Overview"},
	{"name": "optimisations", "path": "/bia/psa/optimisations", "component": "OptimisationManager", "permission": "bia_psa:optimise", "nav_group": "Optimisation"},
	{"name": "optimisation_detail", "path": "/bia/psa/optimisations/<id>", "component": "OptimisationDetail", "permission": "bia_psa:optimise", "nav_group": "Optimisation"},
	{"name": "decisions", "path": "/bia/psa/decisions", "component": "DecisionLog", "permission": "bia_psa:decisions", "nav_group": "Decisions"},
	{"name": "decision_detail", "path": "/bia/psa/decisions/<id>", "component": "DecisionDetail", "permission": "bia_psa:decisions", "nav_group": "Decisions"},
	{"name": "recommendations", "path": "/bia/psa/recommendations", "component": "RecommendationQueue", "permission": "bia_psa:recommendations", "nav_group": "Recommendations"},
	{"name": "recommendation_detail", "path": "/bia/psa/recommendations/<id>", "component": "RecommendationDetail", "permission": "bia_psa:recommendations", "nav_group": "Recommendations"},
	{"name": "whatif", "path": "/bia/psa/whatif", "component": "WhatIfBuilder", "permission": "bia_psa:whatif", "nav_group": "Simulation"},
	{"name": "whatif_detail", "path": "/bia/psa/whatif/<id>", "component": "WhatIfDetail", "permission": "bia_psa:whatif", "nav_group": "Simulation"},
	{"name": "constraints", "path": "/bia/psa/constraints", "component": "ConstraintManager", "permission": "bia_psa:admin", "nav_group": "Configuration"},
	{"name": "approvals", "path": "/bia/psa/approvals", "component": "RecommendationApprovalQueue", "permission": "bia_psa:approve", "nav_group": "Governance"},
	{"name": "audit_log", "path": "/bia/psa/audit", "component": "PrescriptiveAuditLog", "permission": "bia_psa:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/bia/psa/settings", "component": "PrescriptiveSettings", "permission": "bia_psa:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "bia_psa_prescriptive",
	"tokens": {"color.primary": "#1565C0", "color.accent": "#00838F", "color.success": "#2E7D32", "color.warning": "#EF6C00", "color.danger": "#C62828", "surface.canvas": "#F0F4FF", "surface.panel": "#FFFFFF", "text.primary": "#0D1B2A", "text.secondary": "#37474F", "border.radius": "6px", "density": "comfortable"},
	"components": {
		"optimisation": {"icon": "settings-2", "status_indicator": "analysis-state-chip"},
		"decision": {"icon": "git-merge", "status_indicator": "decision-type-chip"},
		"recommendation": {"icon": "lightbulb", "status_indicator": "approval-state-chip"},
		"whatif": {"icon": "sliders", "status_indicator": "analysis-state-chip"},
		"constraint": {"icon": "lock", "status_indicator": "constraint-type-chip"},
	},
}

STREAMING = {
	"processor": "bytewax", "stream": PSA_EVENT_STREAM, "key": "tenant_id",
	"events": ["optimisation_started", "optimisation_completed", "decision_recorded", "recommendation_generated", "recommendation_approved", "recommendation_rejected", "whatif_simulated", "constraint_added", "constraint_violated", "allocation_optimised"],
	"guardrails": ["cross_tenant_analysis_denied", "unapproved_recommendation_action_denied", "decision_explainability_required", "audit_all_decisions"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_policy"}},
	{"name": "cross_tenant_analysis_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_analysis_not_permitted", "required_action": "restrict_to_tenant"}},
	{"name": "optimisation_type_supported", "condition": {"operation": "create_optimisation", "optimisation_type_supported": False}, "effect": {"decision": "deny", "reason": "optimisation_type_not_supported", "required_action": "select_supported_optimisation_type"}},
	{"name": "optimisation_owner_required", "condition": {"operation": "create_optimisation", "owner_present": False}, "effect": {"decision": "deny", "reason": "optimisation_owner_required", "required_action": "attach_owner"}},
	{"name": "decision_type_supported", "condition": {"operation": "record_decision", "decision_type_supported": False}, "effect": {"decision": "deny", "reason": "decision_type_not_supported", "required_action": "select_supported_decision_type"}},
	{"name": "recommendation_type_supported", "condition": {"operation": "generate_recommendation", "recommendation_type_supported": False}, "effect": {"decision": "deny", "reason": "recommendation_type_not_supported", "required_action": "select_supported_recommendation_type"}},
	{"name": "unapproved_recommendation_action_denied", "condition": {"operation": "act_on_recommendation", "approval_state": "pending"}, "effect": {"decision": "deny", "reason": "recommendation_must_be_approved_before_action", "required_action": "submit_for_approval"}},
	{"name": "rejected_recommendation_cannot_be_acted", "condition": {"operation": "act_on_recommendation", "approval_state": "rejected"}, "effect": {"decision": "deny", "reason": "rejected_recommendation_cannot_be_actioned", "required_action": "create_new_recommendation"}},
	{"name": "whatif_requires_baseline", "condition": {"operation": "create_whatif", "baseline_present": False}, "effect": {"decision": "deny", "reason": "whatif_analysis_requires_baseline", "required_action": "define_baseline_scenario"}},
	{"name": "whatif_parameter_type_supported", "condition": {"operation": "add_whatif_parameter", "parameter_type_supported": False}, "effect": {"decision": "deny", "reason": "whatif_parameter_type_not_supported", "required_action": "select_supported_parameter_type"}},
	{"name": "whatif_scenario_limit_enforced", "condition": {"operation": "create_whatif", "scenario_limit_exceeded": True}, "effect": {"decision": "deny", "reason": "whatif_scenario_limit_exceeded", "required_action": "delete_old_whatif_scenario"}},
	{"name": "constraint_type_supported", "condition": {"operation": "add_constraint", "constraint_type_supported": False}, "effect": {"decision": "deny", "reason": "constraint_type_not_supported", "required_action": "select_supported_constraint_type"}},
	{"name": "objective_type_supported", "condition": {"operation": "set_objective", "objective_type_supported": False}, "effect": {"decision": "deny", "reason": "objective_type_not_supported", "required_action": "select_supported_objective_type"}},
	{"name": "hard_constraint_violation_denied", "condition": {"operation": "run_optimisation", "hard_constraint_violated": True}, "effect": {"decision": "deny", "reason": "hard_constraint_violated", "required_action": "revise_constraints_or_objective"}},
	{"name": "max_recommendations_enforced", "condition": {"operation": "generate_recommendation", "recommendation_limit_exceeded": True}, "effect": {"decision": "deny", "reason": "recommendation_limit_exceeded_per_run", "required_action": "reduce_recommendation_scope"}},
	{"name": "decision_explainability_required", "condition": {"operation": "record_decision", "explainability_present": False}, "effect": {"decision": "deny", "reason": "decision_explainability_required", "required_action": "attach_decision_rationale"}},
	{"name": "failed_optimisation_cannot_generate_recommendations", "condition": {"operation": "generate_recommendation", "analysis_state": "failed"}, "effect": {"decision": "deny", "reason": "failed_analysis_cannot_generate_recommendations", "required_action": "re_run_optimisation"}},
	{"name": "audit_all_decisions", "condition": {"operation": "record_decision", "audit_enabled": True}, "effect": {"decision": "allow", "reason": "decision_audited", "required_action": "emit_decision_recorded_event"}},
	{"name": "archived_analysis_read_only", "condition": {"operation": "update_optimisation", "analysis_state": "archived"}, "effect": {"decision": "deny", "reason": "archived_analysis_is_read_only", "required_action": "create_new_analysis"}},
	{"name": "recommendation_max_per_run_enforced", "condition": {"operation": "generate_recommendation", "per_run_limit_exceeded": True}, "effect": {"decision": "deny", "reason": "max_50_recommendations_per_run", "required_action": "narrow_recommendation_criteria"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {"required": ["tenant_id", "ui", "theme"], "properties": {"tenant_id": {"type": "string"}, "ui": {"type": "object"}, "theme": {"type": "object"}}},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["bia/psa/templates"], "routes": UI_ROUTES},
		"theme": THEME, "streaming": STREAMING, "provides": PROVIDES, "requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	for rule in RULES:
		if all(context.get(k) == v for k, v in rule["condition"].items()):
			return {"matched_rule": rule["name"], "decision": rule["effect"]["decision"], "reason": rule["effect"]["reason"], "required_action": rule["effect"]["required_action"]}
	return {"matched_rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
