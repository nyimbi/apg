"""Executable capability contract for APG Fraud Detection."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_fraud"
CAPABILITY_NAME = "Fraud Detection"
CAPABILITY_VERSION = "1.1.0"
FRAUD_EVENT_STREAM = "apg.fintech.fraud.lifecycle"

SUPPORTED_SIGNAL_TYPES = ["payment", "wallet_transfer", "card_not_present", "account_login", "device_change", "refund", "chargeback", "agent_review"]
SUPPORTED_CHANNELS = ["api", "mobile", "web", "pos", "atm", "agent", "batch"]
SUPPORTED_DECISIONS = ["approve", "step_up", "hold", "block", "review"]
SUPPORTED_CASE_TYPES = ["transaction_fraud", "account_takeover", "synthetic_identity", "mule_account", "chargeback_abuse", "device_fraud"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["fraud_ops_reviewer", "transaction_risk_analyst", "chargeback_reviewer", "device_risk_reviewer", "case_investigator"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"scoring": {"review_threshold": 45, "step_up_threshold": 60, "hold_threshold": 75, "block_threshold": 90, "max_score": 100, "min_score": 0},
	"signals": {"supported_types": SUPPORTED_SIGNAL_TYPES, "supported_channels": SUPPORTED_CHANNELS, "money_required_for_transaction_signals": True, "kyc_link_required": True, "source_reference_required": True},
	"indicators": {"velocity_requires_review": True, "device_anomaly_requires_review": True, "geo_anomaly_requires_review": True, "aml_alert_requires_review": True, "chargeback_requires_evidence": True, "account_takeover_requires_review": True},
	"decisions": {"supported_decisions": SUPPORTED_DECISIONS, "challenge_required_for_step_up": True, "reason_required_for_hold_or_block": True, "human_approval_required_for_hold_or_block": True},
	"cases": {"supported_types": SUPPORTED_CASE_TYPES, "investigator_required": True, "evidence_required": True, "resolution_requires_disposition": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_fraud_events": True, "kyc_link_required": True, "human_approval_required_for_high_impact_actions": True},
	"observability": {"event_stream": FRAUD_EVENT_STREAM, "stream_processor": "bytewax", "emit_signal_events": True, "emit_decision_events": True, "emit_case_events": True, "emit_agent_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "payments": "fintech_payments", "wallets": "fintech_wallets", "kyc": "fintech_kyc", "aml": "fintech_aml", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_signals": True, "enable_decisions": True, "enable_cases": True, "enable_chargebacks": True, "enable_devices": True, "enable_agents": True},
	"theme": {"default_theme": "fintech_fraud_control", "allow_tenant_overrides": True},
}

PROVIDES = ["fraud_signal_scoring", "transaction_risk_decisioning", "account_takeover_detection", "device_risk_detection", "chargeback_evidence_workflow", "fraud_case_management", "fraud_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_payments", "fintech_wallets", "fintech_kyc", "fintech_aml"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-fraud/dashboard", "component": "FraudDashboard", "permission": "fintech_fraud:view", "nav_group": "Overview"},
	{"name": "signals", "path": "/fintech-fraud/signals", "component": "FraudSignalQueue", "permission": "fintech_fraud:score", "nav_group": "Signals"},
	{"name": "decisions", "path": "/fintech-fraud/decisions", "component": "FraudDecisionConsole", "permission": "fintech_fraud:decide", "nav_group": "Decisions"},
	{"name": "cases", "path": "/fintech-fraud/cases", "component": "FraudCaseWorkbench", "permission": "fintech_fraud:investigate", "nav_group": "Cases"},
	{"name": "chargebacks", "path": "/fintech-fraud/chargebacks", "component": "FraudChargebackEvidence", "permission": "fintech_fraud:chargebacks", "nav_group": "Evidence"},
	{"name": "devices", "path": "/fintech-fraud/devices", "component": "FraudDeviceConsole", "permission": "fintech_fraud:devices", "nav_group": "Signals"},
	{"name": "agents", "path": "/fintech-fraud/agents", "component": "FraudAgentWorkbench", "permission": "fintech_fraud:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-fraud/settings", "component": "FraudSettings", "permission": "fintech_fraud:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_fraud_control",
	"tokens": {"color.primary": "#1F3A5F", "color.accent": "#7C3AED", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"signals": {"icon": "radar", "status_indicator": "fraud-score-chip"}, "decisions": {"icon": "shield-check", "status_indicator": "decision-chip"}, "cases": {"icon": "folder-search", "status_indicator": "case-status-chip"}, "chargebacks": {"icon": "receipt-text", "status_indicator": "evidence-chip"}, "devices": {"icon": "smartphone", "status_indicator": "device-risk-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": FRAUD_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["fraud_signal_scored", "fraud_decision_recorded", "fraud_case_opened", "fraud_case_resolved", "fraud_agent_registered"],
	"guardrails": ["fraud_batch_requires_bytewax", "fraud_event_requires_bytewax", "privileged_fraud_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Fraud operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "fraud_write_requires_policy", "description": "Fraud writes require policy evidence.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "fraud_policy_required", "required_action": "attach_fraud_policy"}},
	{"name": "signal_subject_required", "description": "Fraud signals require subject reference.", "condition": {"operation": "score_signal", "subject_present": False}, "effect": {"decision": "deny", "reason": "fraud_subject_required", "required_action": "attach_subject_reference"}},
	{"name": "signal_type_supported", "description": "Fraud signal type must be supported.", "condition": {"operation": "score_signal", "signal_type_supported": False}, "effect": {"decision": "deny", "reason": "fraud_signal_type_not_supported", "required_action": "select_supported_signal_type"}},
	{"name": "signal_channel_supported", "description": "Fraud signal channel must be supported.", "condition": {"operation": "score_signal", "channel_supported": False}, "effect": {"decision": "deny", "reason": "fraud_channel_not_supported", "required_action": "select_supported_channel"}},
	{"name": "signal_source_required", "description": "Fraud signals require source reference.", "condition": {"operation": "score_signal", "source_reference_present": False}, "effect": {"decision": "deny", "reason": "source_reference_required", "required_action": "attach_source_reference"}},
	{"name": "signal_requires_kyc_link", "description": "Fraud signals require linked KYC profile evidence.", "condition": {"operation": "score_signal", "kyc_link_present": False}, "effect": {"decision": "deny", "reason": "kyc_link_required", "required_action": "attach_kyc_profile"}},
	{"name": "money_amount_positive", "description": "Money-bearing fraud signals require a positive amount.", "condition": {"operation": "score_signal", "money_signal": True, "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_amount_required", "required_action": "set_positive_amount"}},
	{"name": "money_currency_required", "description": "Money-bearing fraud signals require currency.", "condition": {"operation": "score_signal", "money_signal": True, "currency_present": False}, "effect": {"decision": "deny", "reason": "currency_required", "required_action": "set_currency"}},
	{"name": "risk_score_range", "description": "Fraud risk score must be between 0 and 100.", "condition": {"operation": "score_signal", "risk_score_out_of_range": True}, "effect": {"decision": "deny", "reason": "risk_score_out_of_range", "required_action": "set_valid_risk_score"}},
	{"name": "high_risk_score_requires_review", "description": "High fraud risk scores require review evidence.", "condition": {"operation": "score_signal", "high_risk_score": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "high_fraud_risk_review_required", "required_action": "review_high_risk_signal"}},
	{"name": "velocity_requires_review", "description": "Velocity indicators require fraud review.", "condition": {"operation": "score_signal", "velocity_indicator": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "velocity_review_required", "required_action": "review_velocity_pattern"}},
	{"name": "device_anomaly_requires_review", "description": "Device anomalies require review.", "condition": {"operation": "score_signal", "device_anomaly": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "device_review_required", "required_action": "review_device_signal"}},
	{"name": "geo_anomaly_requires_review", "description": "Geography anomalies require review.", "condition": {"operation": "score_signal", "geo_anomaly": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "geo_review_required", "required_action": "review_geo_signal"}},
	{"name": "aml_alert_requires_review", "description": "Fraud signals linked to AML alerts require review.", "condition": {"operation": "score_signal", "aml_alert_present": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "aml_alert_review_required", "required_action": "review_aml_link"}},
	{"name": "account_takeover_requires_review", "description": "Account takeover indicators require review.", "condition": {"operation": "score_signal", "account_takeover_indicator": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "account_takeover_review_required", "required_action": "review_account_takeover_signal"}},
	{"name": "chargeback_requires_evidence", "description": "Chargeback fraud signals require evidence references.", "condition": {"operation": "score_signal", "chargeback_signal": True, "evidence_present": False}, "effect": {"decision": "deny", "reason": "chargeback_evidence_required", "required_action": "attach_chargeback_evidence"}},
	{"name": "decision_signal_required", "description": "Fraud decisions require an existing signal.", "condition": {"operation": "record_decision", "signal_present": False}, "effect": {"decision": "deny", "reason": "fraud_signal_required", "required_action": "select_signal"}},
	{"name": "decision_supported", "description": "Fraud decision must be supported.", "condition": {"operation": "record_decision", "decision_supported": False}, "effect": {"decision": "deny", "reason": "fraud_decision_not_supported", "required_action": "select_supported_decision"}},
	{"name": "step_up_requires_challenge", "description": "Step-up decisions require auth challenge evidence.", "condition": {"operation": "record_decision", "step_up_decision": True, "challenge_present": False}, "effect": {"decision": "deny", "reason": "challenge_reference_required", "required_action": "attach_challenge_reference"}},
	{"name": "hold_or_block_requires_reason", "description": "Hold or block decisions require reason.", "condition": {"operation": "record_decision", "hold_or_block": True, "reason_present": False}, "effect": {"decision": "deny", "reason": "decision_reason_required", "required_action": "record_decision_reason"}},
	{"name": "hold_or_block_requires_human_approval", "description": "Hold or block decisions require human approval.", "condition": {"operation": "record_decision", "hold_or_block": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "case_signal_required", "description": "Fraud cases require a signal reference.", "condition": {"operation": "open_case", "signal_present": False}, "effect": {"decision": "deny", "reason": "fraud_signal_required", "required_action": "select_signal"}},
	{"name": "case_type_supported", "description": "Fraud case type must be supported.", "condition": {"operation": "open_case", "case_type_supported": False}, "effect": {"decision": "deny", "reason": "fraud_case_type_not_supported", "required_action": "select_supported_case_type"}},
	{"name": "case_investigator_required", "description": "Fraud cases require an investigator.", "condition": {"operation": "open_case", "investigator_present": False}, "effect": {"decision": "deny", "reason": "case_investigator_required", "required_action": "assign_investigator"}},
	{"name": "case_evidence_required", "description": "Fraud cases require evidence references.", "condition": {"operation": "open_case", "evidence_present": False}, "effect": {"decision": "deny", "reason": "case_evidence_required", "required_action": "attach_case_evidence"}},
	{"name": "case_resolution_requires_disposition", "description": "Resolving fraud cases requires disposition.", "condition": {"operation": "resolve_case", "disposition_present": False}, "effect": {"decision": "deny", "reason": "case_disposition_required", "required_action": "record_case_disposition"}},
	{"name": "case_resolution_requires_reviewer", "description": "Resolving fraud cases requires reviewer.", "condition": {"operation": "resolve_case", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "case_reviewer_required", "required_action": "assign_case_reviewer"}},
	{"name": "fraud_batch_requires_bytewax", "description": "Fraud batches require Bytewax.", "condition": {"operation": "fraud_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_fraud_batch_to_bytewax"}},
	{"name": "fraud_event_requires_bytewax", "description": "Fraud events require Bytewax.", "condition": {"operation": "fraud_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_fraud_event_to_bytewax"}},
	{"name": "fraud_agent_runtime_supported", "description": "Fraud agents must use a supported runtime.", "condition": {"operation": "register_fraud_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "fraud_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "fraud_agent_role_supported", "description": "Fraud agents must use a supported role.", "condition": {"operation": "register_fraud_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "fraud_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_fraud_agent_action_requires_human_approval", "description": "Privileged fraud-agent actions require human approval.", "condition": {"operation": "fraud_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},

	# Cross-tenant and privilege escalation guards
	{"name": "cross_tenant_fraud_access_denied", "description": "Fraud resources cannot be accessed across tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_credentials"}},
	{"name": "privilege_escalation_denied", "description": "Fraud privilege escalation without approval is denied.", "condition": {"privilege_escalation_attempt": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "obtain_escalation_approval"}},

	# Africa-specific fraud rules
	{"name": "mpesa_sim_swap_fraud_check", "description": "M-Pesa transactions must check for recent SIM swap activity.", "condition": {"operation": "mpesa_transaction", "sim_swap_detected": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "mpesa_sim_swap_fraud_detected", "required_action": "review_sim_swap_activity"}},
	{"name": "mobile_money_account_takeover_detection", "description": "Mobile money account takeover indicators trigger fraud review.", "condition": {"operation": "mobile_money_transaction", "account_takeover_indicator": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "mobile_money_account_takeover_detected", "required_action": "review_account_takeover"}},
	{"name": "mpesa_agent_fraud_velocity_check", "description": "M-Pesa agent transactions with velocity fraud patterns require review.", "condition": {"operation": "mpesa_agent_transaction", "velocity_fraud_indicator": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "mpesa_agent_velocity_fraud_detected", "required_action": "review_agent_velocity_pattern"}},
	{"name": "ke_cbk_fraud_reporting_required", "description": "Kenya CBK requires fraud incident reporting within 24 hours.", "condition": {"operation": "confirm_fraud_case", "jurisdiction": "KE", "cbk_fraud_report_filed": False}, "effect": {"decision": "require_review", "reason": "ke_cbk_fraud_reporting_required", "required_action": "file_cbk_fraud_report"}},
	{"name": "mobile_money_phishing_block", "description": "Mobile money transactions from known phishing patterns are blocked.", "condition": {"operation": "mobile_money_transaction", "phishing_indicator": True}, "effect": {"decision": "deny", "reason": "mobile_money_phishing_blocked", "required_action": "block_and_notify_customer"}},
	{"name": "ussd_fraud_session_timeout", "description": "USSD fraud detection triggers immediate session termination.", "condition": {"operation": "ussd_transaction", "fraud_indicator": True}, "effect": {"decision": "deny", "reason": "ussd_fraud_detected", "required_action": "terminate_ussd_session_and_review"}},
	{"name": "ke_ke_telecoms_fraud_coordination", "description": "Kenya fraud cases involving telecoms require coordinated investigation with Safaricom/Airtel.", "condition": {"operation": "confirm_fraud_case", "jurisdiction": "KE", "telco_fraud_involved": True, "telco_coordination_initiated": False}, "effect": {"decision": "require_review", "reason": "ke_telco_fraud_coordination_required", "required_action": "initiate_telco_fraud_coordination"}},
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
	return {"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "configuration": configuration, "configuration_schema": _configuration_schema(), "provides": PROVIDES, "requires": REQUIRES, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-fraud/api/v1", "routes": deepcopy(UI_ROUTES), "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"]}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
