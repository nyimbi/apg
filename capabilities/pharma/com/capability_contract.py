"""Executable capability contract for APG Pharma Commercial Operations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "pharma_com"
CAPABILITY_NAME = "Commercial Operations"
CAPABILITY_VERSION = "1.0.0"
COM_EVENT_STREAM = "apg.pharma.com.lifecycle"

SUPPORTED_TERRITORY_TYPES = ["national", "regional", "district", "specialty", "key_account", "hospital", "retail_chain", "export"]
SUPPORTED_REP_TYPES = ["primary_care", "specialty", "hospital", "key_account_manager", "medical_science_liaison", "oncology", "vaccines", "rare_disease"]
SUPPORTED_CALL_TYPES = ["detailing", "sample_drop", "education", "follow_up", "formulary", "congress", "virtual", "pharmacy_call"]
SUPPORTED_SAMPLE_TYPES = ["promotional_sample", "starter_pack", "patient_assistance", "clinical_sample", "device_sample"]
SUPPORTED_INTERACTION_TYPES = ["office_visit", "hospital_call", "pharmacy_visit", "congress_meeting", "webinar", "advisory_board", "speaker_program"]
SUPPORTED_PLAN_STATUSES = ["draft", "approved", "active", "under_review", "suspended", "archived"]
SUPPORTED_COMPLIANCE_FRAMEWORKS = ["pdma", "efpia", "abpi", "pharmag", "local_code", "sunshine_act", "aggregate_spend"]
SUPPORTED_APPROVAL_STATUSES = ["pending", "approved", "rejected", "escalated", "withdrawn"]
SUPPORTED_CHANNEL_TYPES = ["field_force", "inside_sales", "digital", "pharmacy", "hospital", "distributor", "direct"]
SUPPORTED_PRODUCT_STATUSES = ["pipeline", "pre_launch", "launched", "mature", "loss_of_exclusivity", "discontinued"]
SUPPORTED_TARGET_TIERS = ["tier_1", "tier_2", "tier_3", "non_target", "key_account", "kol"]
SUPPORTED_SPEND_CATEGORIES = ["meals", "entertainment", "travel", "education", "research_grant", "consulting_fee", "speaker_fee", "sample_value"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["territory_optimizer", "call_planner", "compliance_reviewer", "spend_auditor", "target_segmenter"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"territories": {"supported_territory_types": SUPPORTED_TERRITORY_TYPES, "owner_required": True, "product_alignment_required": True, "approval_required": True},
	"reps": {"supported_rep_types": SUPPORTED_REP_TYPES, "territory_required": True, "quota_required": True, "certification_required": True},
	"calls": {"supported_call_types": SUPPORTED_CALL_TYPES, "physician_id_required": True, "product_discussed_required": True, "outcome_required": True},
	"samples": {"supported_sample_types": SUPPORTED_SAMPLE_TYPES, "pdma_compliance_required": True, "signature_required": True, "lot_number_required": True, "expiry_required": True},
	"interactions": {"supported_interaction_types": SUPPORTED_INTERACTION_TYPES, "hcp_id_required": True, "value_threshold_reporting": 10.0, "aggregate_spend_tracking": True},
	"plans": {"supported_statuses": SUPPORTED_PLAN_STATUSES, "approval_required": True, "territory_alignment_required": True, "quota_setting_required": True},
	"compliance": {"supported_frameworks": SUPPORTED_COMPLIANCE_FRAMEWORKS, "aggregate_spend_cap": 500.0, "sunshine_act_reporting": True, "pdma_audit_required": True},
	"targets": {"supported_tiers": SUPPORTED_TARGET_TIERS, "segmentation_required": True, "call_frequency_required": True, "review_cycle_months": 6},
	"spend": {"supported_categories": SUPPORTED_SPEND_CATEGORIES, "receipt_required_above": 25.0, "pre_approval_required_above": 100.0, "hcp_consent_required": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "pdma_compliance_required": True, "cross_tenant_denied": True, "unapproved_sample_denied": True, "aggregate_spend_cap_enforced": True, "unlicensed_rep_denied": True},
	"observability": {"event_stream": COM_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "compliance": "comp", "scheduler": "schd", "event_stream": "mqeb"},
	"ui": {"enable_dashboard": True, "enable_territories": True, "enable_reps": True, "enable_calls": True, "enable_samples": True, "enable_interactions": True, "enable_plans": True, "enable_targets": True, "enable_spend": True, "enable_compliance": True},
	"theme": {"default_theme": "pharma_com_field", "allow_tenant_overrides": True},
}

PROVIDES = [
	"territory_management_workflow",
	"sales_rep_management_workflow",
	"call_activity_workflow",
	"sample_management_workflow",
	"hcp_interaction_workflow",
	"commercial_plan_workflow",
	"target_segmentation_workflow",
	"aggregate_spend_workflow",
	"pdma_compliance_workflow",
	"commercial_dashboard_workflow",
]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "schd", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/pharma-com/dashboard", "component": "ComDashboard", "permission": "pharma_com:view", "nav_group": "Overview"},
	{"name": "territories", "path": "/pharma-com/territories", "component": "TerritoryConsole", "permission": "pharma_com:territories", "nav_group": "Territory"},
	{"name": "territory_detail", "path": "/pharma-com/territories/<id>", "component": "TerritoryDetail", "permission": "pharma_com:territories", "nav_group": "Territory"},
	{"name": "reps", "path": "/pharma-com/reps", "component": "SalesRepRoster", "permission": "pharma_com:reps", "nav_group": "Field Force"},
	{"name": "calls", "path": "/pharma-com/calls", "component": "CallActivityLog", "permission": "pharma_com:calls", "nav_group": "Field Force"},
	{"name": "samples", "path": "/pharma-com/samples", "component": "SampleManagementConsole", "permission": "pharma_com:samples", "nav_group": "Samples"},
	{"name": "sample_reconciliation", "path": "/pharma-com/samples/reconcile", "component": "SampleReconciliation", "permission": "pharma_com:samples_admin", "nav_group": "Samples"},
	{"name": "interactions", "path": "/pharma-com/interactions", "component": "HcpInteractionLedger", "permission": "pharma_com:interactions", "nav_group": "HCP Engagement"},
	{"name": "targets", "path": "/pharma-com/targets", "component": "TargetSegmentation", "permission": "pharma_com:targets", "nav_group": "Planning"},
	{"name": "plans", "path": "/pharma-com/plans", "component": "CommercialPlanWorkbench", "permission": "pharma_com:plans", "nav_group": "Planning"},
	{"name": "spend", "path": "/pharma-com/spend", "component": "AggregateSpendTracker", "permission": "pharma_com:spend", "nav_group": "Compliance"},
	{"name": "compliance", "path": "/pharma-com/compliance", "component": "PdmaComplianceConsole", "permission": "pharma_com:compliance", "nav_group": "Compliance"},
	{"name": "reports", "path": "/pharma-com/reports", "component": "CommercialReports", "permission": "pharma_com:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/pharma-com/settings", "component": "ComSettings", "permission": "pharma_com:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "pharma_com_field",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#059669",
		"color.success": "#166534",
		"color.warning": "#B45309",
		"color.danger": "#DC2626",
		"surface.canvas": "#F0F4FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E1B4B",
		"text.secondary": "#4B5563",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"territories": {"icon": "map-pin", "status_indicator": "territory-type-chip"},
		"reps": {"icon": "user-tie", "status_indicator": "rep-type-chip"},
		"calls": {"icon": "phone-call", "status_indicator": "call-outcome-chip"},
		"samples": {"icon": "package", "status_indicator": "sample-status-chip"},
		"interactions": {"icon": "handshake", "status_indicator": "interaction-type-chip"},
		"plans": {"icon": "clipboard-list", "status_indicator": "plan-status-chip"},
		"targets": {"icon": "target", "status_indicator": "tier-chip"},
		"spend": {"icon": "dollar-sign", "status_indicator": "spend-category-chip"},
		"compliance": {"icon": "shield-check", "status_indicator": "compliance-status-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": COM_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"territory_created", "territory_updated", "rep_assigned", "call_recorded",
		"sample_dispensed", "sample_reconciled", "interaction_recorded",
		"spend_recorded", "plan_approved", "compliance_flag_raised",
		"pdma_violation_detected", "aggregate_spend_cap_exceeded",
	],
	"guardrails": [
		"pdma_compliance_required_for_sampling",
		"aggregate_spend_cap_enforced",
		"unlicensed_rep_action_denied",
		"unapproved_sample_denied",
		"cross_tenant_interaction_denied",
		"sunshine_act_reporting_required",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required", "required_action": "attach_policy"}},
	{"name": "territory_type_supported", "condition": {"operation": "create_territory", "territory_type_supported": False}, "effect": {"decision": "deny", "reason": "territory_type_not_supported", "required_action": "select_supported_territory_type"}},
	{"name": "territory_owner_required", "condition": {"operation": "create_territory", "owner_present": False}, "effect": {"decision": "deny", "reason": "territory_owner_required", "required_action": "assign_territory_owner"}},
	{"name": "territory_approval_required", "condition": {"operation": "create_territory", "approval_present": False}, "effect": {"decision": "deny", "reason": "territory_approval_required", "required_action": "obtain_territory_approval"}},
	{"name": "rep_type_supported", "condition": {"operation": "assign_rep", "rep_type_supported": False}, "effect": {"decision": "deny", "reason": "rep_type_not_supported", "required_action": "select_supported_rep_type"}},
	{"name": "rep_territory_required", "condition": {"operation": "assign_rep", "territory_present": False}, "effect": {"decision": "deny", "reason": "territory_required", "required_action": "assign_to_territory"}},
	{"name": "rep_certification_required", "condition": {"operation": "assign_rep", "certification_present": False}, "effect": {"decision": "deny", "reason": "rep_certification_required", "required_action": "complete_certification"}},
	{"name": "call_physician_required", "condition": {"operation": "record_call", "physician_id_present": False}, "effect": {"decision": "deny", "reason": "physician_id_required", "required_action": "attach_physician_id"}},
	{"name": "call_type_supported", "condition": {"operation": "record_call", "call_type_supported": False}, "effect": {"decision": "deny", "reason": "call_type_not_supported", "required_action": "select_supported_call_type"}},
	{"name": "call_product_required", "condition": {"operation": "record_call", "product_present": False}, "effect": {"decision": "deny", "reason": "product_discussed_required", "required_action": "record_product_discussed"}},
	{"name": "sample_pdma_compliance_required", "condition": {"operation": "dispense_sample", "pdma_compliant": False}, "effect": {"decision": "deny", "reason": "pdma_compliance_required", "required_action": "complete_pdma_workflow"}},
	{"name": "sample_signature_required", "condition": {"operation": "dispense_sample", "signature_present": False}, "effect": {"decision": "deny", "reason": "hcp_signature_required", "required_action": "capture_hcp_signature"}},
	{"name": "sample_lot_required", "condition": {"operation": "dispense_sample", "lot_number_present": False}, "effect": {"decision": "deny", "reason": "lot_number_required", "required_action": "record_lot_number"}},
	{"name": "sample_expiry_required", "condition": {"operation": "dispense_sample", "expiry_present": False}, "effect": {"decision": "deny", "reason": "expiry_date_required", "required_action": "record_expiry_date"}},
	{"name": "sample_type_supported", "condition": {"operation": "dispense_sample", "sample_type_supported": False}, "effect": {"decision": "deny", "reason": "sample_type_not_supported", "required_action": "select_supported_sample_type"}},
	{"name": "interaction_hcp_required", "condition": {"operation": "record_interaction", "hcp_id_present": False}, "effect": {"decision": "deny", "reason": "hcp_id_required", "required_action": "identify_hcp"}},
	{"name": "interaction_type_supported", "condition": {"operation": "record_interaction", "interaction_type_supported": False}, "effect": {"decision": "deny", "reason": "interaction_type_not_supported", "required_action": "select_supported_interaction_type"}},
	{"name": "spend_receipt_required", "condition": {"operation": "record_spend", "amount_above_threshold": True, "receipt_present": False}, "effect": {"decision": "deny", "reason": "receipt_required_above_threshold", "required_action": "attach_receipt"}},
	{"name": "spend_pre_approval_required", "condition": {"operation": "record_spend", "amount_above_approval_threshold": True, "pre_approval_present": False}, "effect": {"decision": "deny", "reason": "pre_approval_required", "required_action": "obtain_pre_approval"}},
	{"name": "aggregate_spend_cap_enforced", "condition": {"operation": "record_spend", "aggregate_cap_exceeded": True}, "effect": {"decision": "deny", "reason": "aggregate_spend_cap_exceeded", "required_action": "escalate_to_compliance"}},
	{"name": "plan_approval_required", "condition": {"operation": "approve_plan", "approval_present": False}, "effect": {"decision": "deny", "reason": "plan_approval_required", "required_action": "route_for_approval"}},
	{"name": "target_tier_supported", "condition": {"operation": "set_target_tier", "tier_supported": False}, "effect": {"decision": "deny", "reason": "tier_not_supported", "required_action": "select_supported_tier"}},
	{"name": "cross_tenant_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_operation_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "compliance_framework_supported", "condition": {"operation": "record_compliance", "framework_supported": False}, "effect": {"decision": "deny", "reason": "framework_not_supported", "required_action": "select_supported_framework"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"configuration": configuration,
		"configuration_schema": {
			"type": "object",
			"required": ["tenant_id", "ui", "theme"],
			"properties": {
				"tenant_id": {"type": "string", "minLength": 1},
				"ui": {"type": "object"},
				"theme": {"type": "object"},
			},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/pharma-com/api/v1",
			"requires_theme": True,
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
