"""Executable capability contract for APG Pharma Product Registration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "pharma_reg"
CAPABILITY_NAME = "Product Registration"
CAPABILITY_VERSION = "1.0.0"
REG_EVENT_STREAM = "apg.pharma.reg.lifecycle"

SUPPORTED_REGISTRATION_TYPES = ["new_application", "renewal", "variation_type_ia", "variation_type_ib", "variation_type_ii", "extension", "transfer_of_ownership", "withdrawal"]
SUPPORTED_DOSSIER_FORMATS = ["ctd_ectd", "ctd_paper", "neesp", "actd", "legacy", "common_technical_document"]
SUPPORTED_APPROVAL_STATUSES = ["not_submitted", "submitted", "under_review", "additional_info_requested", "approved", "approved_with_conditions", "refused", "withdrawn", "expired"]
SUPPORTED_AUTHORITY_INTERACTIONS = ["clarification_request", "scientific_advice", "pre_submission_meeting", "day_80_list", "day_120_list", "oral_explanation", "post_approval_meeting"]
SUPPORTED_LIFECYCLE_EVENTS = ["initial_approval", "renewal", "variation", "suspension", "revocation", "withdrawal", "transfer", "national_procedure", "mutual_recognition", "decentralised"]
SUPPORTED_PRODUCT_TYPES = ["small_molecule", "biologic", "biosimilar", "generic", "hybrid", "well_established_use", "fixed_dose_combination", "combination_product", "atmp"]
SUPPORTED_PROCEDURE_TYPES = ["national", "mutual_recognition", "decentralised", "centralised", "repeat_use", "referral"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["dossier_compiler", "submission_tracker", "authority_liaison", "lifecycle_manager", "registration_analyst"]
SUPPORTED_REGULATORY_REGIONS = ["us_fda", "eu_ema", "uk_mhra", "japan_pmda", "canada_health", "australia_tga", "brazil_anvisa", "india_cdsco", "china_nmpa", "gulf_gcc", "kenya_pharmacy", "sa_sahpra"]
SUPPORTED_DOSSIER_MODULES = ["module_1", "module_2", "module_3", "module_4", "module_5"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"registrations": {"supported_types": SUPPORTED_REGISTRATION_TYPES, "dossier_required": True, "qp_sign_off_required": True, "local_representative_required": True, "tracked_submission_required": True},
	"dossiers": {"supported_formats": SUPPORTED_DOSSIER_FORMATS, "supported_modules": SUPPORTED_DOSSIER_MODULES, "version_control_required": True, "module_completeness_check": True, "ectd_validation_required": True},
	"approvals": {"supported_statuses": SUPPORTED_APPROVAL_STATUSES, "supported_regions": SUPPORTED_REGULATORY_REGIONS, "timeline_tracking_required": True, "condition_tracking_required": True, "certificate_storage_required": True},
	"authority_interactions": {"supported_types": SUPPORTED_AUTHORITY_INTERACTIONS, "minutes_required": True, "action_items_required": True, "follow_up_tracking": True},
	"lifecycle": {"supported_events": SUPPORTED_LIFECYCLE_EVENTS, "renewal_alert_days": 180, "variation_impact_assessment_required": True, "global_dossier_alignment_required": True},
	"procedures": {"supported_types": SUPPORTED_PROCEDURE_TYPES, "reference_member_state_tracking": True, "concern_tracking_required": True, "timeline_tracking_required": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "dossier_integrity_required": True, "approval_before_distribution": True, "cross_tenant_denied": True},
	"observability": {"event_stream": REG_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "compliance": "comp", "scheduler": "schd", "event_stream": "mqeb"},
	"ui": {"enable_dashboard": True, "enable_registrations": True, "enable_dossiers": True, "enable_approvals": True, "enable_authority_interactions": True, "enable_lifecycle": True, "enable_procedures": True},
	"theme": {"default_theme": "pharma_reg_registration", "allow_tenant_overrides": True},
}

PROVIDES = [
	"registration_application_workflow",
	"dossier_compilation_workflow",
	"authority_interaction_workflow",
	"approval_tracking_workflow",
	"lifecycle_maintenance_workflow",
	"variation_management_workflow",
	"renewal_management_workflow",
	"procedure_management_workflow",
	"registration_certificate_workflow",
	"global_dossier_alignment_workflow",
]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "schd", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/pharma-reg/dashboard", "component": "RegDashboard", "permission": "pharma_reg:view", "nav_group": "Overview"},
	{"name": "registrations", "path": "/pharma-reg/registrations", "component": "RegistrationRegistry", "permission": "pharma_reg:registrations", "nav_group": "Registrations"},
	{"name": "registration_detail", "path": "/pharma-reg/registrations/<id>", "component": "RegistrationDetail", "permission": "pharma_reg:registrations", "nav_group": "Registrations"},
	{"name": "dossiers", "path": "/pharma-reg/dossiers", "component": "DossierWorkbench", "permission": "pharma_reg:dossiers", "nav_group": "Dossiers"},
	{"name": "dossier_detail", "path": "/pharma-reg/dossiers/<id>", "component": "DossierDetail", "permission": "pharma_reg:dossiers", "nav_group": "Dossiers"},
	{"name": "approvals", "path": "/pharma-reg/approvals", "component": "ApprovalTracker", "permission": "pharma_reg:approvals", "nav_group": "Approvals"},
	{"name": "authority_interactions", "path": "/pharma-reg/interactions", "component": "AuthorityInteractions", "permission": "pharma_reg:interactions", "nav_group": "Authority"},
	{"name": "procedures", "path": "/pharma-reg/procedures", "component": "ProcedureManagement", "permission": "pharma_reg:procedures", "nav_group": "Procedures"},
	{"name": "lifecycle", "path": "/pharma-reg/lifecycle", "component": "LifecycleManager", "permission": "pharma_reg:lifecycle", "nav_group": "Lifecycle"},
	{"name": "renewals", "path": "/pharma-reg/renewals", "component": "RenewalQueue", "permission": "pharma_reg:renewals", "nav_group": "Lifecycle"},
	{"name": "variations", "path": "/pharma-reg/variations", "component": "VariationQueue", "permission": "pharma_reg:variations", "nav_group": "Lifecycle"},
	{"name": "certificates", "path": "/pharma-reg/certificates", "component": "CertificateVault", "permission": "pharma_reg:certificates", "nav_group": "Documents"},
	{"name": "reports", "path": "/pharma-reg/reports", "component": "RegistrationReports", "permission": "pharma_reg:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/pharma-reg/settings", "component": "RegSettings", "permission": "pharma_reg:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "pharma_reg_registration",
	"tokens": {
		"color.primary": "#0C4A6E",
		"color.accent": "#0369A1",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F0F9FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0C2340",
		"text.secondary": "#334155",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"registrations": {"icon": "file-badge-2", "status_indicator": "registration-type-chip"},
		"dossiers": {"icon": "archive", "status_indicator": "dossier-format-chip"},
		"approvals": {"icon": "check-circle-2", "status_indicator": "approval-status-chip"},
		"authority_interactions": {"icon": "message-square", "status_indicator": "interaction-type-chip"},
		"lifecycle": {"icon": "refresh-cw", "status_indicator": "lifecycle-event-chip"},
		"certificates": {"icon": "award", "status_indicator": "certificate-status-chip"},
		"procedures": {"icon": "git-merge", "status_indicator": "procedure-type-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": REG_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"registration_submitted", "registration_approved", "registration_refused",
		"dossier_compiled", "dossier_updated",
		"authority_interaction_recorded", "clarification_response_submitted",
		"variation_filed", "renewal_filed",
		"approval_expiring", "approval_renewed",
		"lifecycle_event_recorded", "certificate_stored",
	],
	"guardrails": [
		"dossier_completeness_check_required",
		"qp_sign_off_required_before_submission",
		"approval_before_distribution_enforced",
		"renewal_alert_180d_enforced",
		"ectd_validation_required",
		"authority_interaction_minutes_required",
		"cross_tenant_registration_data_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required", "required_action": "attach_policy"}},
	{"name": "registration_type_supported", "condition": {"operation": "create_registration", "registration_type_supported": False}, "effect": {"decision": "deny", "reason": "registration_type_not_supported", "required_action": "select_supported_registration_type"}},
	{"name": "dossier_required_for_submission", "condition": {"operation": "submit_registration", "dossier_attached": False}, "effect": {"decision": "deny", "reason": "dossier_required", "required_action": "compile_dossier"}},
	{"name": "dossier_format_supported", "condition": {"operation": "create_dossier", "dossier_format_supported": False}, "effect": {"decision": "deny", "reason": "dossier_format_not_supported", "required_action": "select_supported_format"}},
	{"name": "ectd_validation_required", "condition": {"operation": "submit_registration", "dossier_format": "ctd_ectd", "ectd_validated": False}, "effect": {"decision": "deny", "reason": "ectd_validation_required", "required_action": "validate_ectd"}},
	{"name": "qp_sign_off_required", "condition": {"operation": "submit_registration", "qp_signed_off": False}, "effect": {"decision": "deny", "reason": "qp_sign_off_required", "required_action": "obtain_qp_sign_off"}},
	{"name": "approval_before_distribution", "condition": {"operation": "distribute_product", "registration_approved": False}, "effect": {"decision": "deny", "reason": "registration_approval_required_before_distribution", "required_action": "obtain_registration_approval"}},
	{"name": "region_supported", "condition": {"operation": "create_registration", "region_supported": False}, "effect": {"decision": "deny", "reason": "regulatory_region_not_supported", "required_action": "select_supported_region"}},
	{"name": "approval_status_supported", "condition": {"operation": "update_approval_status", "approval_status_supported": False}, "effect": {"decision": "deny", "reason": "approval_status_not_supported", "required_action": "select_supported_approval_status"}},
	{"name": "authority_interaction_minutes_required", "condition": {"operation": "record_interaction", "minutes_present": False}, "effect": {"decision": "deny", "reason": "meeting_minutes_required", "required_action": "attach_meeting_minutes"}},
	{"name": "authority_interaction_type_supported", "condition": {"operation": "record_interaction", "interaction_type_supported": False}, "effect": {"decision": "deny", "reason": "interaction_type_not_supported", "required_action": "select_supported_interaction_type"}},
	{"name": "variation_impact_assessment_required", "condition": {"operation": "file_variation", "impact_assessed": False}, "effect": {"decision": "deny", "reason": "variation_impact_assessment_required", "required_action": "complete_impact_assessment"}},
	{"name": "renewal_alert_180d", "condition": {"operation": "check_registration", "expiring_within_180d": True, "renewal_initiated": False}, "effect": {"decision": "deny", "reason": "renewal_required_within_180d", "required_action": "initiate_renewal"}},
	{"name": "lifecycle_event_supported", "condition": {"operation": "record_lifecycle_event", "lifecycle_event_supported": False}, "effect": {"decision": "deny", "reason": "lifecycle_event_not_supported", "required_action": "select_supported_lifecycle_event"}},
	{"name": "procedure_type_supported", "condition": {"operation": "initiate_procedure", "procedure_type_supported": False}, "effect": {"decision": "deny", "reason": "procedure_type_not_supported", "required_action": "select_supported_procedure_type"}},
	{"name": "product_type_supported", "condition": {"operation": "create_registration", "product_type_supported": False}, "effect": {"decision": "deny", "reason": "product_type_not_supported", "required_action": "select_supported_product_type"}},
	{"name": "cross_tenant_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_operation_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "dossier_module_supported", "condition": {"operation": "add_dossier_module", "module_supported": False}, "effect": {"decision": "deny", "reason": "dossier_module_not_supported", "required_action": "select_supported_module"}},
	{"name": "local_representative_required", "condition": {"operation": "submit_registration", "local_representative_present": False}, "effect": {"decision": "deny", "reason": "local_representative_required", "required_action": "designate_local_representative"}},
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
			"api_prefix": "/pharma-reg/api/v1",
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
