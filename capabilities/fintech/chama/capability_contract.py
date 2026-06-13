"""Executable capability contract for APG Chama & ROSCA Engine."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_chama"
CAPABILITY_NAME = "Chama & ROSCA Engine"
CAPABILITY_VERSION = "1.0.0"
CHAMA_EVENT_STREAM = "apg.fintech.chama.lifecycle"

SUPPORTED_GROUP_TYPES = ["chama", "rosca", "table_banking"]
SUPPORTED_CONTRIBUTION_STATUSES = ["pending", "paid", "overdue", "waived", "partial"]
SUPPORTED_PAYOUT_STATUSES = ["pending", "processing", "disbursed", "failed", "reversed"]
SUPPORTED_LOAN_STATUSES = ["pending_approval", "approved", "active", "fully_repaid", "defaulted", "written_off"]
SUPPORTED_PAYMENT_METHODS = ["mpesa", "airtel_money", "bank_transfer", "cash", "equity_eazzy", "kcb_mobi"]
SUPPORTED_CYCLE_STATUSES = ["active", "completed", "skipped", "suspended"]
SUPPORTED_MEETING_TYPES = ["regular", "special", "agm", "emergency"]
SUPPORTED_FREQUENCIES = ["weekly", "biweekly", "monthly", "quarterly"]
SUPPORTED_REMINDER_CHANNELS = ["sms", "whatsapp", "push_notification", "voice_call"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"groups": {
		"supported_group_types": SUPPORTED_GROUP_TYPES,
		"supported_frequencies": SUPPORTED_FREQUENCIES,
		"min_members": 3,
		"max_members": 100,
		"name_required": True,
		"contribution_amount_required": True,
		"kyc_required": True,
	},
	"contributions": {
		"supported_statuses": SUPPORTED_CONTRIBUTION_STATUSES,
		"supported_payment_methods": SUPPORTED_PAYMENT_METHODS,
		"member_required": True,
		"amount_required": True,
		"cycle_required": True,
	},
	"payouts": {
		"supported_statuses": SUPPORTED_PAYOUT_STATUSES,
		"supported_payment_methods": SUPPORTED_PAYMENT_METHODS,
		"mpesa_default": True,
		"approval_required": True,
		"cycle_complete_before_payout": True,
	},
	"loans": {
		"supported_statuses": SUPPORTED_LOAN_STATUSES,
		"max_multiple_of_savings": 3,
		"min_guarantors": 2,
		"max_interest_rate_monthly_pct": 10.0,
		"approval_required": True,
		"guarantors_required": True,
	},
	"treasury": {
		"real_time_balance": True,
		"track_interest_income": True,
		"track_penalty_income": True,
		"reserve_ratio_required": False,
	},
	"cycles": {
		"supported_statuses": SUPPORTED_CYCLE_STATUSES,
		"auto_advance": True,
		"completion_requires_all_contributions": False,
	},
	"meetings": {
		"supported_types": SUPPORTED_MEETING_TYPES,
		"minutes_required": False,
		"quorum_tracking": True,
	},
	"reminders": {
		"supported_channels": SUPPORTED_REMINDER_CHANNELS,
		"days_before_due": [3, 1],
		"overdue_escalation_days": 3,
		"enabled": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_events": True,
		"cross_tenant_group_denied": True,
		"payout_without_cycle_denied": True,
		"loan_exceeds_treasury_denied": True,
		"overpayment_flagged": True,
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"payments": "fintech_payments",
		"kyc": "fintech_kyc",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_groups": True,
		"enable_members": True,
		"enable_contributions": True,
		"enable_payouts": True,
		"enable_loans": True,
		"enable_treasury": True,
		"enable_cycles": True,
		"enable_meetings": True,
		"enable_statements": True,
	},
	"theme": {
		"default_theme": "chama_community",
		"allow_tenant_overrides": True,
	},
}

PROVIDES = [
	"chama_management",
	"rosca_rotation",
	"group_lending",
	"treasury_management",
	"mobile_disbursement",
]
REQUIRES = ["auth", "audl", "ntfy", "fintech_payments", "fintech_kyc"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/chama/dashboard", "component": "ChamaDashboard", "permission": "chama:view", "nav_group": "Overview"},
	{"name": "groups", "path": "/chama/groups", "component": "ChamaGroupConsole", "permission": "chama:groups", "nav_group": "Groups"},
	{"name": "members", "path": "/chama/members", "component": "ChamaMemberConsole", "permission": "chama:members", "nav_group": "Groups"},
	{"name": "contributions", "path": "/chama/contributions", "component": "ChamaContributionLedger", "permission": "chama:contributions", "nav_group": "Transactions"},
	{"name": "payouts", "path": "/chama/payouts", "component": "ChamaPayoutConsole", "permission": "chama:payouts", "nav_group": "Transactions"},
	{"name": "loans", "path": "/chama/loans", "component": "ChamaLoanWorkbench", "permission": "chama:loans", "nav_group": "Lending"},
	{"name": "treasury", "path": "/chama/treasury", "component": "ChamaTreasuryView", "permission": "chama:treasury", "nav_group": "Finance"},
	{"name": "cycles", "path": "/chama/cycles", "component": "ChamaCycleConsole", "permission": "chama:cycles", "nav_group": "Operations"},
	{"name": "meetings", "path": "/chama/meetings", "component": "ChamaMeetingConsole", "permission": "chama:meetings", "nav_group": "Operations"},
	{"name": "statements", "path": "/chama/statements", "component": "ChamaStatementView", "permission": "chama:statements", "nav_group": "Reports"},
	{"name": "settings", "path": "/chama/settings", "component": "ChamaSettings", "permission": "chama:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "chama_community",
	"tokens": {
		"color.primary": "#8B5E3C",
		"color.accent": "#C49A6C",
		"color.success": "#2D6A4F",
		"color.warning": "#E9A825",
		"color.danger": "#9B2335",
		"surface.canvas": "#FDF6EE",
		"surface.panel": "#FFFFFF",
		"text.primary": "#2C1810",
		"text.secondary": "#6B4C35",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"groups": {"icon": "users", "status_indicator": "group-type-chip"},
		"contributions": {"icon": "coins", "status_indicator": "contribution-status-chip"},
		"payouts": {"icon": "banknote", "status_indicator": "payout-status-chip"},
		"loans": {"icon": "hand-coins", "status_indicator": "loan-status-chip"},
		"treasury": {"icon": "vault", "status_indicator": "balance-chip"},
		"cycles": {"icon": "refresh-cw", "status_indicator": "cycle-status-chip"},
		"meetings": {"icon": "calendar-check", "status_indicator": "meeting-type-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": CHAMA_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"group.created",
		"contribution.received",
		"payout.disbursed",
		"loan.approved",
		"loan.repayment.received",
		"cycle.completed",
		"cycle.started",
		"meeting.recorded",
		"treasury.updated",
		"reminder.sent",
	],
	"guardrails": [
		"payout_without_completed_cycle_denied",
		"loan_exceeds_treasury_denied",
		"cross_tenant_group_denied",
		"overpayment_flagged",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "chama_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "chama_policy_required", "required_action": "attach_chama_policy"}},
	{"name": "group_type_supported", "condition": {"operation": "create_group", "group_type_supported": False}, "effect": {"decision": "deny", "reason": "group_type_not_supported", "required_action": "select_supported_group_type"}},
	{"name": "group_name_required", "condition": {"operation": "create_group", "group_name_present": False}, "effect": {"decision": "deny", "reason": "group_name_required", "required_action": "provide_group_name"}},
	{"name": "group_contribution_amount_required", "condition": {"operation": "create_group", "contribution_amount_present": False}, "effect": {"decision": "deny", "reason": "contribution_amount_required", "required_action": "set_contribution_amount"}},
	{"name": "group_frequency_supported", "condition": {"operation": "create_group", "frequency_supported": False}, "effect": {"decision": "deny", "reason": "frequency_not_supported", "required_action": "select_supported_frequency"}},
	{"name": "contribution_member_required", "condition": {"operation": "record_contribution", "member_present": False}, "effect": {"decision": "deny", "reason": "member_required", "required_action": "identify_member"}},
	{"name": "contribution_payment_method_supported", "condition": {"operation": "record_contribution", "payment_method_supported": False}, "effect": {"decision": "deny", "reason": "payment_method_not_supported", "required_action": "select_supported_payment_method"}},
	{"name": "contribution_amount_positive", "condition": {"operation": "record_contribution", "amount_positive": False}, "effect": {"decision": "deny", "reason": "contribution_amount_must_be_positive", "required_action": "set_positive_amount"}},
	{"name": "payout_cycle_required", "condition": {"operation": "disburse_payout", "cycle_present": False}, "effect": {"decision": "deny", "reason": "cycle_required_for_payout", "required_action": "specify_cycle"}},
	{"name": "payout_recipient_required", "condition": {"operation": "disburse_payout", "recipient_present": False}, "effect": {"decision": "deny", "reason": "recipient_required", "required_action": "specify_recipient"}},
	{"name": "loan_amount_positive", "condition": {"operation": "create_loan", "amount_positive": False}, "effect": {"decision": "deny", "reason": "loan_amount_must_be_positive", "required_action": "set_positive_amount"}},
	{"name": "loan_guarantors_required", "condition": {"operation": "create_loan", "guarantors_present": False}, "effect": {"decision": "deny", "reason": "guarantors_required", "required_action": "identify_guarantors"}},
	{"name": "loan_exceeds_treasury", "condition": {"operation": "create_loan", "loan_exceeds_treasury": True}, "effect": {"decision": "deny", "reason": "loan_exceeds_available_treasury", "required_action": "reduce_loan_amount"}},
	{"name": "repayment_amount_positive", "condition": {"operation": "record_repayment", "amount_positive": False}, "effect": {"decision": "deny", "reason": "repayment_amount_must_be_positive", "required_action": "set_positive_amount"}},
	{"name": "cross_tenant_group_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_group_access_denied", "required_action": "use_own_tenant_id"}},
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
		"rule_engine": {
			"type": "deterministic",
			"default_decision": "allow",
			"rules": deepcopy(RULES),
		},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/chama/api/v1",
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
