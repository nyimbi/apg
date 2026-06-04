"""Executable capability contract for APG Pharmacy Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "healthcare_pha"
CAPABILITY_NAME = "Pharmacy Management"
CAPABILITY_VERSION = "1.0.0"
PHA_EVENT_STREAM = "apg.healthcare.pha.lifecycle"

SUPPORTED_DRUG_TYPES = [
	"brand", "generic", "biosimilar", "otc", "compounded",
	"investigational", "vaccine", "blood_product",
]
SUPPORTED_DRUG_SCHEDULES = ["schedule_i", "schedule_ii", "schedule_iii", "schedule_iv", "schedule_v", "non_controlled"]
SUPPORTED_DOSAGE_FORMS = [
	"tablet", "capsule", "liquid", "injection", "patch", "inhaler",
	"suppository", "cream", "ointment", "drops", "infusion",
]
SUPPORTED_DISPENSE_STATUSES = ["pending", "verified", "dispensed", "picked_up", "returned", "cancelled"]
SUPPORTED_INTERACTION_SEVERITIES = ["contraindicated", "major", "moderate", "minor", "informational"]
SUPPORTED_FORMULARY_STATUSES = ["preferred", "non_preferred", "non_formulary", "prior_auth_required", "step_therapy"]
SUPPORTED_CONTROLLED_SUBSTANCE_ACTIONS = ["dispense", "waste", "destroy", "count", "transfer", "receive"]
SUPPORTED_INVENTORY_STATUSES = ["in_stock", "low_stock", "out_of_stock", "on_order", "recalled", "expired"]
SUPPORTED_LASA_ALERT_TYPES = ["look_alike", "sound_alike", "look_and_sound_alike"]
SUPPORTED_RETURN_REASONS = ["adverse_reaction", "patient_refused", "wrong_medication", "expired", "dispensing_error"]
SUPPORTED_PRIOR_AUTH_STATUSES = ["pending", "approved", "denied", "expired"]
SUPPORTED_AGENT_ROLES = ["pharmacy_steward", "dispense_reviewer", "formulary_reviewer", "controlled_substance_reviewer"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"formulary": {"supported_statuses": SUPPORTED_FORMULARY_STATUSES, "prior_auth_workflow_enabled": True},
	"dispensing": {"supported_statuses": SUPPORTED_DISPENSE_STATUSES, "pharmacist_verification_required": True, "barcode_scanning_required": True},
	"interactions": {"supported_severities": SUPPORTED_INTERACTION_SEVERITIES, "contraindicated_blocks_dispense": True},
	"controlled_substances": {"supported_schedules": SUPPORTED_DRUG_SCHEDULES, "supported_actions": SUPPORTED_CONTROLLED_SUBSTANCE_ACTIONS, "dual_witness_required_for_waste": True, "count_frequency_hours": 8},
	"inventory": {"supported_statuses": SUPPORTED_INVENTORY_STATUSES, "low_stock_threshold_days": 7, "expiry_warning_days": 30},
	"lasa": {"supported_alert_types": SUPPORTED_LASA_ALERT_TYPES, "tall_man_lettering_enabled": True},
	"governance": {
		"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True,
		"hipaa_phi_protection": True, "cross_tenant_dispense_denied": True,
		"contraindicated_dispense_denied": True, "pharmacist_verification_required": True,
		"controlled_substance_dual_witness_required": True,
		"recalled_drug_dispense_denied": True,
	},
	"observability": {"event_stream": PHA_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "compliance": "comp", "monitoring": "moni", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_formulary": True, "enable_dispensing": True, "enable_interactions": True, "enable_inventory": True, "enable_controlled": True, "enable_lasa": True},
	"theme": {"default_theme": "healthcare_pha_clinical", "allow_tenant_overrides": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
}

PROVIDES = [
	"drug_formulary_management", "prescription_dispensing",
	"lasa_alert_management", "controlled_substance_tracking",
	"drug_interaction_checking", "pharmacy_inventory_management",
	"prior_authorization_workflow", "medication_adherence_tracking",
	"pharmacist_verification_workflow",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/healthcare-pha/dashboard", "component": "PhaDashboard", "permission": "healthcare_pha:view", "nav_group": "Overview"},
	{"name": "formulary", "path": "/healthcare-pha/formulary", "component": "PhaFormularyManager", "permission": "healthcare_pha:formulary", "nav_group": "Formulary"},
	{"name": "drug_detail", "path": "/healthcare-pha/formulary/<id>", "component": "PhaDrugDetail", "permission": "healthcare_pha:formulary", "nav_group": "Formulary"},
	{"name": "dispense_queue", "path": "/healthcare-pha/dispense", "component": "PhaDispenseQueue", "permission": "healthcare_pha:dispense", "nav_group": "Dispensing"},
	{"name": "dispense_verify", "path": "/healthcare-pha/dispense/<id>/verify", "component": "PhaDispenseVerify", "permission": "healthcare_pha:dispense_verify", "nav_group": "Dispensing"},
	{"name": "interactions", "path": "/healthcare-pha/interactions", "component": "PhaInteractionChecker", "permission": "healthcare_pha:interactions", "nav_group": "Safety"},
	{"name": "lasa_alerts", "path": "/healthcare-pha/lasa", "component": "PhaLasaAlerts", "permission": "healthcare_pha:lasa", "nav_group": "Safety"},
	{"name": "controlled", "path": "/healthcare-pha/controlled", "component": "PhaControlledSubstances", "permission": "healthcare_pha:controlled", "nav_group": "Controlled"},
	{"name": "controlled_log", "path": "/healthcare-pha/controlled/log", "component": "PhaControlledLog", "permission": "healthcare_pha:controlled", "nav_group": "Controlled"},
	{"name": "inventory", "path": "/healthcare-pha/inventory", "component": "PhaInventoryManager", "permission": "healthcare_pha:inventory", "nav_group": "Inventory"},
	{"name": "prior_auth", "path": "/healthcare-pha/prior-auth", "component": "PhaPriorAuthQueue", "permission": "healthcare_pha:prior_auth", "nav_group": "Authorization"},
	{"name": "agents", "path": "/healthcare-pha/agents", "component": "PhaAgentWorkbench", "permission": "healthcare_pha:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/healthcare-pha/settings", "component": "PhaSettings", "permission": "healthcare_pha:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "healthcare_pha_clinical",
	"tokens": {
		"color.primary": "#065F46", "color.accent": "#0369A1", "color.success": "#166534",
		"color.warning": "#A16207", "color.danger": "#B91C1C",
		"surface.canvas": "#ECFDF5", "surface.panel": "#FFFFFF",
		"text.primary": "#022C22", "text.secondary": "#065F46",
		"border.radius": "6px", "density": "comfortable",
	},
	"components": {
		"formulary": {"icon": "book-open", "status_indicator": "formulary-status-chip"},
		"dispense": {"icon": "package", "status_indicator": "dispense-status-chip"},
		"interactions": {"icon": "zap", "status_indicator": "interaction-severity-chip"},
		"lasa": {"icon": "eye", "status_indicator": "lasa-alert-chip"},
		"controlled": {"icon": "lock", "status_indicator": "schedule-chip"},
		"inventory": {"icon": "archive", "status_indicator": "inventory-status-chip"},
		"prior_auth": {"icon": "file-check", "status_indicator": "prior-auth-status-chip"},
		"agents": {"icon": "cpu", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax", "stream": PHA_EVENT_STREAM, "key": "tenant_id",
	"events": [
		"drug_added_to_formulary", "drug_dispensed", "dispense_verified",
		"drug_interaction_detected", "lasa_alert_triggered", "controlled_substance_dispensed",
		"controlled_substance_wasted", "inventory_low_stock", "inventory_recalled",
		"prior_auth_approved", "prior_auth_denied",
	],
	"guardrails": [
		"contraindicated_dispense_denied", "recalled_drug_dispense_denied",
		"pharmacist_verification_required", "controlled_substance_dual_witness_required",
		"cross_tenant_dispense_denied", "privileged_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "cross_tenant_dispense_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_dispense_prohibited", "required_action": "use_tenant_scoped_query"}},
	{"name": "contraindicated_dispense_denied", "condition": {"operation": "dispense", "interaction_severity": "contraindicated"}, "effect": {"decision": "deny", "reason": "contraindicated_drug_interaction_blocks_dispense", "required_action": "select_alternative_drug"}},
	{"name": "pharmacist_verification_required", "condition": {"operation": "dispense", "pharmacist_verified": False}, "effect": {"decision": "deny", "reason": "pharmacist_verification_required_before_dispense", "required_action": "obtain_pharmacist_verification"}},
	{"name": "recalled_drug_dispense_denied", "condition": {"operation": "dispense", "drug_inventory_status": "recalled"}, "effect": {"decision": "deny", "reason": "recalled_drug_cannot_be_dispensed", "required_action": "quarantine_recalled_drug"}},
	{"name": "expired_drug_dispense_denied", "condition": {"operation": "dispense", "drug_inventory_status": "expired"}, "effect": {"decision": "deny", "reason": "expired_drug_cannot_be_dispensed", "required_action": "remove_expired_drug_from_inventory"}},
	{"name": "out_of_stock_dispense_denied", "condition": {"operation": "dispense", "drug_inventory_status": "out_of_stock"}, "effect": {"decision": "deny", "reason": "drug_out_of_stock", "required_action": "place_inventory_order"}},
	{"name": "controlled_substance_dual_witness_required", "condition": {"operation": "waste_controlled_substance", "dual_witness_present": False}, "effect": {"decision": "deny", "reason": "dual_witness_required_for_controlled_substance_waste", "required_action": "obtain_witness_signature"}},
	{"name": "prior_auth_required_for_non_formulary", "condition": {"operation": "dispense", "formulary_status": "prior_auth_required", "prior_auth_approved": False}, "effect": {"decision": "deny", "reason": "prior_authorization_required", "required_action": "obtain_prior_authorization"}},
	{"name": "drug_type_supported", "condition": {"operation": "add_to_formulary", "drug_type_supported": False}, "effect": {"decision": "deny", "reason": "drug_type_not_supported", "required_action": "select_supported_drug_type"}},
	{"name": "drug_schedule_supported", "condition": {"operation": "add_to_formulary", "drug_schedule_supported": False}, "effect": {"decision": "deny", "reason": "drug_schedule_not_supported", "required_action": "select_supported_drug_schedule"}},
	{"name": "dosage_form_supported", "condition": {"operation": "add_to_formulary", "dosage_form_supported": False}, "effect": {"decision": "deny", "reason": "dosage_form_not_supported", "required_action": "select_supported_dosage_form"}},
	{"name": "lasa_alert_type_supported", "condition": {"operation": "create_lasa_alert", "lasa_alert_type_supported": False}, "effect": {"decision": "deny", "reason": "lasa_alert_type_not_supported", "required_action": "select_supported_lasa_alert_type"}},
	{"name": "inventory_status_supported", "condition": {"operation": "update_inventory", "inventory_status_supported": False}, "effect": {"decision": "deny", "reason": "inventory_status_not_supported", "required_action": "select_supported_inventory_status"}},
	{"name": "controlled_substance_action_supported", "condition": {"operation": "controlled_substance_action", "action_supported": False}, "effect": {"decision": "deny", "reason": "controlled_substance_action_not_supported", "required_action": "select_supported_action"}},
	{"name": "return_reason_required", "condition": {"operation": "return_medication", "return_reason_present": False}, "effect": {"decision": "deny", "reason": "return_reason_required", "required_action": "specify_return_reason"}},
	{"name": "step_therapy_required", "condition": {"operation": "dispense", "formulary_status": "step_therapy", "step_therapy_completed": False}, "effect": {"decision": "deny", "reason": "step_therapy_protocol_not_completed", "required_action": "complete_step_therapy_protocol"}},
	{"name": "agent_privileged_action_requires_approval", "condition": {"agent_action": True, "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "privileged_agent_action_requires_human_approval", "required_action": "record_human_approval"}},
	{"name": "low_stock_warning", "condition": {"operation": "dispense", "inventory_days_remaining": 7}, "effect": {"decision": "warn", "reason": "low_stock_approaching", "required_action": "place_replenishment_order"}},
	{"name": "non_formulary_requires_override", "condition": {"operation": "dispense", "formulary_status": "non_formulary", "formulary_override_present": False}, "effect": {"decision": "deny", "reason": "non_formulary_drug_requires_override", "required_action": "obtain_formulary_override"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {"required": ["tenant_id", "ui", "theme"], "properties": {"tenant_id": {"type": "string"}, "ui": {"type": "object"}, "theme": {"type": "object"}}},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["healthcare/pha/templates"], "routes": UI_ROUTES},
		"theme": THEME, "streaming": STREAMING, "provides": PROVIDES, "requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			effect = rule["effect"]
			return {"rule": rule["name"], "decision": effect["decision"], "reason": effect["reason"], "required_action": effect.get("required_action")}
	return {"rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
