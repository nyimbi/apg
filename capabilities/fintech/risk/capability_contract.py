"""Executable capability contract for APG FinTech Risk Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_risk"
CAPABILITY_NAME = "FinTech Risk Management"
CAPABILITY_VERSION = "1.1.0"
RISK_EVENT_STREAM = "apg.fintech.risk.lifecycle"

SUPPORTED_RISK_DOMAINS = ["credit", "market", "liquidity", "operational", "fraud", "compliance", "model", "third_party"]
SUPPORTED_SUBJECT_TYPES = ["customer", "merchant", "wallet", "account", "portfolio", "loan", "agent", "counterparty"]
SUPPORTED_EXPOSURE_TYPES = ["credit_limit", "settlement", "liquidity", "fx", "market_value", "operational_loss", "fraud_loss"]
SUPPORTED_CONTROL_TYPES = ["preventive", "detective", "corrective", "compensating", "automated", "manual"]
SUPPORTED_SCENARIO_TYPES = ["macro_shock", "liquidity_run", "fraud_spike", "counterparty_default", "market_drawdown", "outage", "regulatory_change"]
SUPPORTED_EVENT_TYPES = ["limit_breach", "control_failure", "loss_event", "model_drift", "policy_exception", "third_party_issue"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_CURRENCIES = ["USD", "KES", "EUR", "GBP", "NGN", "GHS", "ZAR"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["risk_appetite_reviewer", "exposure_monitor", "stress_testing_reviewer", "control_assurance_agent", "risk_event_reviewer", "model_risk_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"appetite": {"supported_domains": SUPPORTED_RISK_DOMAINS, "threshold_required": True, "owner_required": True, "evidence_required": True},
	"profiles": {"supported_subject_types": SUPPORTED_SUBJECT_TYPES, "kyc_required": True, "score_required": True, "source_required": True, "supported_currencies": SUPPORTED_CURRENCIES},
	"exposures": {"supported_types": SUPPORTED_EXPOSURE_TYPES, "profile_required": True, "positive_amount_required": True, "limit_required": True, "source_required": True},
	"controls": {"supported_types": SUPPORTED_CONTROL_TYPES, "owner_required": True, "evidence_required": True, "effectiveness_score_required": True},
	"stress_testing": {"supported_scenarios": SUPPORTED_SCENARIO_TYPES, "profile_required": True, "impact_required": True, "probability_required": True, "mitigation_required": True},
	"breaches": {"supported_severities": SUPPORTED_SEVERITIES, "exposure_required": True, "evidence_required": True, "remediation_owner_required": True},
	"events": {"supported_types": SUPPORTED_EVENT_TYPES, "supported_severities": SUPPORTED_SEVERITIES, "profile_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "human_approval_required_for_limit_override": True},
	"observability": {"event_stream": RISK_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "payments": "fintech_payments", "wallets": "fintech_wallets", "kyc": "fintech_kyc", "aml": "fintech_aml", "fraud": "fintech_fraud", "analytics": "bia", "reporting": "fin_rpt", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_appetite": True, "enable_profiles": True, "enable_exposures": True, "enable_controls": True, "enable_stress_tests": True, "enable_breaches": True, "enable_events": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "fintech_risk_control", "allow_tenant_overrides": True},
}

PROVIDES = ["risk_appetite_workflow", "risk_profile_workflow", "risk_exposure_workflow", "risk_control_workflow", "risk_stress_testing_workflow", "risk_limit_breach_workflow", "risk_event_workflow", "risk_review_workflow", "risk_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_payments", "fintech_wallets", "fintech_kyc", "fintech_aml", "fintech_fraud", "bia", "fin_rpt"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-risk/dashboard", "component": "RiskDashboard", "permission": "fintech_risk:view", "nav_group": "Overview"},
	{"name": "appetite", "path": "/fintech-risk/appetite", "component": "RiskAppetiteConsole", "permission": "fintech_risk:appetite", "nav_group": "Governance"},
	{"name": "profiles", "path": "/fintech-risk/profiles", "component": "RiskProfileConsole", "permission": "fintech_risk:profiles", "nav_group": "Risk"},
	{"name": "exposures", "path": "/fintech-risk/exposures", "component": "ExposureMonitor", "permission": "fintech_risk:exposures", "nav_group": "Risk"},
	{"name": "controls", "path": "/fintech-risk/controls", "component": "ControlAssuranceConsole", "permission": "fintech_risk:controls", "nav_group": "Controls"},
	{"name": "stress_tests", "path": "/fintech-risk/stress-tests", "component": "StressTestingWorkbench", "permission": "fintech_risk:stress", "nav_group": "Analytics"},
	{"name": "breaches", "path": "/fintech-risk/breaches", "component": "LimitBreachQueue", "permission": "fintech_risk:breaches", "nav_group": "Issues"},
	{"name": "events", "path": "/fintech-risk/events", "component": "RiskEventWorkbench", "permission": "fintech_risk:events", "nav_group": "Issues"},
	{"name": "reviews", "path": "/fintech-risk/reviews", "component": "RiskReviewConsole", "permission": "fintech_risk:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/fintech-risk/agents", "component": "RiskAgentWorkbench", "permission": "fintech_risk:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-risk/settings", "component": "RiskSettings", "permission": "fintech_risk:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_risk_control",
	"tokens": {"color.primary": "#0F766E", "color.accent": "#2563EB", "color.success": "#15803D", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"appetite": {"icon": "gauge", "status_indicator": "appetite-chip"}, "profiles": {"icon": "user-round-check", "status_indicator": "risk-score-chip"}, "exposures": {"icon": "chart-no-axes-combined", "status_indicator": "exposure-chip"}, "controls": {"icon": "shield-check", "status_indicator": "control-chip"}, "stress_tests": {"icon": "activity", "status_indicator": "scenario-chip"}, "breaches": {"icon": "triangle-alert", "status_indicator": "breach-chip"}, "events": {"icon": "sirens", "status_indicator": "event-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": RISK_EVENT_STREAM, "key": "tenant_id", "events": ["risk_appetite_registered", "risk_profile_created", "risk_exposure_recorded", "risk_control_evaluated", "risk_stress_scenario_recorded", "risk_limit_breach_recorded", "risk_event_opened", "risk_review_recorded", "risk_agent_registered"], "guardrails": ["risk_batch_requires_bytewax", "privileged_risk_agent_action_requires_human_approval"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "risk_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "risk_policy_required", "required_action": "attach_risk_policy"}},
	{"name": "appetite_domain_supported", "condition": {"operation": "register_appetite", "domain_supported": False}, "effect": {"decision": "deny", "reason": "risk_domain_not_supported", "required_action": "select_supported_domain"}},
	{"name": "appetite_threshold_required", "condition": {"operation": "register_appetite", "positive_threshold": False}, "effect": {"decision": "deny", "reason": "risk_threshold_required", "required_action": "set_positive_threshold"}},
	{"name": "appetite_owner_required", "condition": {"operation": "register_appetite", "owner_present": False}, "effect": {"decision": "deny", "reason": "risk_owner_required", "required_action": "assign_owner"}},
	{"name": "appetite_evidence_required", "condition": {"operation": "register_appetite", "evidence_present": False}, "effect": {"decision": "deny", "reason": "risk_evidence_required", "required_action": "attach_evidence"}},
	{"name": "profile_subject_required", "condition": {"operation": "create_profile", "subject_present": False}, "effect": {"decision": "deny", "reason": "risk_subject_required", "required_action": "attach_subject"}},
	{"name": "profile_subject_type_supported", "condition": {"operation": "create_profile", "subject_type_supported": False}, "effect": {"decision": "deny", "reason": "risk_subject_type_not_supported", "required_action": "select_supported_subject_type"}},
	{"name": "profile_kyc_required", "condition": {"operation": "create_profile", "kyc_present": False}, "effect": {"decision": "deny", "reason": "risk_kyc_required", "required_action": "attach_kyc"}},
	{"name": "profile_score_range", "condition": {"operation": "create_profile", "score_valid": False}, "effect": {"decision": "deny", "reason": "risk_score_out_of_range", "required_action": "set_valid_score"}},
	{"name": "profile_currency_supported", "condition": {"operation": "create_profile", "currency_supported": False}, "effect": {"decision": "deny", "reason": "risk_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "profile_source_required", "condition": {"operation": "create_profile", "source_present": False}, "effect": {"decision": "deny", "reason": "risk_source_required", "required_action": "attach_source"}},
	{"name": "exposure_profile_required", "condition": {"operation": "record_exposure", "profile_present": False}, "effect": {"decision": "deny", "reason": "risk_profile_required", "required_action": "select_profile"}},
	{"name": "exposure_type_supported", "condition": {"operation": "record_exposure", "exposure_type_supported": False}, "effect": {"decision": "deny", "reason": "risk_exposure_type_not_supported", "required_action": "select_supported_exposure_type"}},
	{"name": "exposure_amount_positive", "condition": {"operation": "record_exposure", "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_exposure_amount_required", "required_action": "set_positive_amount"}},
	{"name": "exposure_currency_supported", "condition": {"operation": "record_exposure", "currency_supported": False}, "effect": {"decision": "deny", "reason": "risk_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "exposure_limit_required", "condition": {"operation": "record_exposure", "positive_limit": False}, "effect": {"decision": "deny", "reason": "positive_limit_required", "required_action": "set_positive_limit"}},
	{"name": "exposure_source_required", "condition": {"operation": "record_exposure", "source_present": False}, "effect": {"decision": "deny", "reason": "risk_source_required", "required_action": "attach_source"}},
	{"name": "limit_override_requires_human_approval", "condition": {"operation": "record_exposure", "over_limit": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_limit_override_approval"}},
	{"name": "control_profile_required", "condition": {"operation": "evaluate_control", "profile_present": False}, "effect": {"decision": "deny", "reason": "risk_profile_required", "required_action": "select_profile"}},
	{"name": "control_type_supported", "condition": {"operation": "evaluate_control", "control_type_supported": False}, "effect": {"decision": "deny", "reason": "risk_control_type_not_supported", "required_action": "select_supported_control_type"}},
	{"name": "control_owner_required", "condition": {"operation": "evaluate_control", "owner_present": False}, "effect": {"decision": "deny", "reason": "control_owner_required", "required_action": "assign_control_owner"}},
	{"name": "control_evidence_required", "condition": {"operation": "evaluate_control", "evidence_present": False}, "effect": {"decision": "deny", "reason": "control_evidence_required", "required_action": "attach_control_evidence"}},
	{"name": "control_effectiveness_required", "condition": {"operation": "evaluate_control", "effectiveness_score_valid": False}, "effect": {"decision": "deny", "reason": "control_effectiveness_out_of_range", "required_action": "set_valid_effectiveness_score"}},
	{"name": "scenario_profile_required", "condition": {"operation": "run_stress_scenario", "profile_present": False}, "effect": {"decision": "deny", "reason": "risk_profile_required", "required_action": "select_profile"}},
	{"name": "scenario_type_supported", "condition": {"operation": "run_stress_scenario", "scenario_type_supported": False}, "effect": {"decision": "deny", "reason": "scenario_type_not_supported", "required_action": "select_supported_scenario"}},
	{"name": "scenario_impact_positive", "condition": {"operation": "run_stress_scenario", "positive_impact": False}, "effect": {"decision": "deny", "reason": "scenario_impact_required", "required_action": "set_positive_impact"}},
	{"name": "scenario_probability_valid", "condition": {"operation": "run_stress_scenario", "probability_valid": False}, "effect": {"decision": "deny", "reason": "scenario_probability_out_of_range", "required_action": "set_probability_bps"}},
	{"name": "scenario_mitigation_required", "condition": {"operation": "run_stress_scenario", "mitigation_present": False}, "effect": {"decision": "deny", "reason": "scenario_mitigation_required", "required_action": "attach_mitigation"}},
	{"name": "breach_exposure_required", "condition": {"operation": "record_limit_breach", "exposure_present": False}, "effect": {"decision": "deny", "reason": "risk_exposure_required", "required_action": "select_exposure"}},
	{"name": "breach_severity_supported", "condition": {"operation": "record_limit_breach", "severity_supported": False}, "effect": {"decision": "deny", "reason": "risk_severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "breach_evidence_required", "condition": {"operation": "record_limit_breach", "evidence_present": False}, "effect": {"decision": "deny", "reason": "breach_evidence_required", "required_action": "attach_breach_evidence"}},
	{"name": "breach_owner_required", "condition": {"operation": "record_limit_breach", "owner_present": False}, "effect": {"decision": "deny", "reason": "remediation_owner_required", "required_action": "assign_remediation_owner"}},
	{"name": "event_profile_required", "condition": {"operation": "open_risk_event", "profile_present": False}, "effect": {"decision": "deny", "reason": "risk_profile_required", "required_action": "select_profile"}},
	{"name": "event_type_supported", "condition": {"operation": "open_risk_event", "event_type_supported": False}, "effect": {"decision": "deny", "reason": "risk_event_type_not_supported", "required_action": "select_supported_event_type"}},
	{"name": "event_severity_supported", "condition": {"operation": "open_risk_event", "severity_supported": False}, "effect": {"decision": "deny", "reason": "risk_severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "event_evidence_required", "condition": {"operation": "open_risk_event", "evidence_present": False}, "effect": {"decision": "deny", "reason": "risk_event_evidence_required", "required_action": "attach_event_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "risk_batch_requires_bytewax", "condition": {"operation": "risk_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_risk_batch_to_bytewax"}},
	{"name": "risk_agent_runtime_supported", "condition": {"operation": "register_risk_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "risk_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "risk_agent_role_supported", "condition": {"operation": "register_risk_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "risk_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_risk_agent_action_requires_human_approval", "condition": {"operation": "risk_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-risk/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
