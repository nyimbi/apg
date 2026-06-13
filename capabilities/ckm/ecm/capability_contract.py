"""Executable capability contract for APG ECM / Records Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "ckm_ecm"
CAPABILITY_NAME = "ECM / Records Management"
CAPABILITY_VERSION = "1.0.0"
ECM_EVENT_STREAM = "apg.ckm.ecm.lifecycle"

SUPPORTED_DOCUMENT_TYPES = [
	"contract", "policy", "procedure", "report", "form",
	"correspondence", "invoice", "patient_record", "case_file",
	"regulatory_submission", "public_record", "evidence", "certificate",
	"permit", "minutes", "memo", "technical_document", "other",
]
SUPPORTED_DOCUMENT_STATUSES = [
	"draft", "under_review", "approved", "active", "superseded",
	"archived", "pending_disposal", "disposed",
]
SUPPORTED_RETENTION_CATEGORIES = [
	"permanent", "long_term", "medium_term", "short_term",
	"transient", "regulatory", "legal_hold", "vital_record",
]
SUPPORTED_SENSITIVITY_LEVELS = [
	"public", "internal", "confidential", "restricted", "secret",
]
SUPPORTED_REGULATORY_FRAMEWORKS = [
	"hipaa", "gdpr", "sox", "iso_15489", "nara", "ico", "pci_dss",
	"local_government", "custom", "none",
]
SUPPORTED_RETENTION_TRIGGERS = ["creation", "last_access", "last_modified", "event"]
SUPPORTED_DISPOSAL_METHODS = [
	"secure_delete", "physical_destruction", "transfer_to_archives",
	"redaction", "anonymisation", "offsite_shredding",
]
SUPPORTED_WORKFLOW_TYPES = [
	"approval", "review", "compliance_review", "legal_review",
	"quality_check", "publication", "disposal_authorisation",
]
SUPPORTED_WORKFLOW_DECISIONS = ["approved", "rejected", "returned_for_revision", "escalated"]
SUPPORTED_SEARCH_FIELDS = [
	"title", "document_type", "status", "retention_category",
	"sensitivity", "regulatory_framework", "author",
]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"documents": {
		"supported_types": SUPPORTED_DOCUMENT_TYPES,
		"supported_statuses": SUPPORTED_DOCUMENT_STATUSES,
		"title_required": True,
		"content_hash_required": True,
		"retention_category_required": True,
	},
	"versions": {
		"author_required": True,
		"change_summary_required": True,
		"content_hash_required": True,
		"immutable": True,
	},
	"retention_policies": {
		"supported_categories": SUPPORTED_RETENTION_CATEGORIES,
		"supported_triggers": SUPPORTED_RETENTION_TRIGGERS,
		"supported_disposal_methods": SUPPORTED_DISPOSAL_METHODS,
		"years_positive": True,
	},
	"classifications": {
		"supported_sensitivity_levels": SUPPORTED_SENSITIVITY_LEVELS,
		"supported_regulatory_frameworks": SUPPORTED_REGULATORY_FRAMEWORKS,
	},
	"workflows": {
		"supported_types": SUPPORTED_WORKFLOW_TYPES,
		"supported_decisions": SUPPORTED_WORKFLOW_DECISIONS,
		"approvers_required": True,
	},
	"disposal": {
		"supported_methods": SUPPORTED_DISPOSAL_METHODS,
		"authorized_by_required": True,
		"audit_trail_required": True,
	},
	"search": {
		"supported_fields": SUPPORTED_SEARCH_FIELDS,
		"full_text_enabled": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"cross_tenant_access_denied": True,
		"unapproved_disposal_denied": True,
		"version_deletion_denied": True,
		"retention_override_denied": True,
	},
	"observability": {
		"event_stream": ECM_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"search": "srch",
		"esig": "esig",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_documents": True,
		"enable_versions": True,
		"enable_retention": True,
		"enable_classification": True,
		"enable_workflows": True,
		"enable_disposal": True,
		"enable_search": True,
	},
	"theme": {
		"default_theme": "ecm_records_control",
		"allow_tenant_overrides": True,
	},
}

PROVIDES = [
	"document_management",
	"version_control",
	"retention_management",
	"content_workflow",
	"disposal_management",
]
REQUIRES = ["auth", "audl", "ntfy", "srch", "esig"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/ckm-ecm/dashboard", "component": "EcmDashboard", "permission": "ckm_ecm:view", "nav_group": "Overview"},
	{"name": "documents", "path": "/ckm-ecm/documents", "component": "EcmDocumentRegistry", "permission": "ckm_ecm:documents", "nav_group": "Records"},
	{"name": "versions", "path": "/ckm-ecm/versions", "component": "EcmVersionHistory", "permission": "ckm_ecm:documents", "nav_group": "Records"},
	{"name": "retention", "path": "/ckm-ecm/retention", "component": "EcmRetentionPolicyConsole", "permission": "ckm_ecm:retention", "nav_group": "Governance"},
	{"name": "classification", "path": "/ckm-ecm/classification", "component": "EcmClassificationConsole", "permission": "ckm_ecm:classification", "nav_group": "Governance"},
	{"name": "workflows", "path": "/ckm-ecm/workflows", "component": "EcmWorkflowQueue", "permission": "ckm_ecm:workflows", "nav_group": "Operations"},
	{"name": "disposal", "path": "/ckm-ecm/disposal", "component": "EcmDisposalConsole", "permission": "ckm_ecm:disposal", "nav_group": "Lifecycle"},
	{"name": "search", "path": "/ckm-ecm/search", "component": "EcmSearchConsole", "permission": "ckm_ecm:view", "nav_group": "Discovery"},
	{"name": "settings", "path": "/ckm-ecm/settings", "component": "EcmSettings", "permission": "ckm_ecm:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "ecm_records_control",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#0369A1",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F8FAFC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"documents": {"icon": "file-text", "status_indicator": "document-status-chip"},
		"versions": {"icon": "git-branch", "status_indicator": "version-chip"},
		"retention": {"icon": "clock", "status_indicator": "retention-chip"},
		"classification": {"icon": "shield", "status_indicator": "sensitivity-chip"},
		"workflows": {"icon": "check-square", "status_indicator": "workflow-chip"},
		"disposal": {"icon": "trash-2", "status_indicator": "disposal-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": ECM_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"document.created",
		"document.versioned",
		"document.classified",
		"document.status_changed",
		"retention_policy.created",
		"retention_policy.applied",
		"document.due_for_disposal",
		"workflow.started",
		"workflow.step_completed",
		"workflow.completed",
		"disposal.executed",
	],
	"guardrails": [
		"cross_tenant_access_denied",
		"unapproved_disposal_denied",
		"version_deletion_denied",
		"retention_override_denied",
	],
}

RULES: list[dict[str, Any]] = [
	# tenant context
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "ecm_policy_required", "required_action": "attach_ecm_policy"}},
	# document creation
	{"name": "document_title_required", "condition": {"operation": "create_document", "title_present": False}, "effect": {"decision": "deny", "reason": "document_title_required", "required_action": "provide_document_title"}},
	{"name": "document_type_supported", "condition": {"operation": "create_document", "document_type_supported": False}, "effect": {"decision": "deny", "reason": "document_type_not_supported", "required_action": "select_supported_document_type"}},
	{"name": "document_content_hash_required", "condition": {"operation": "create_document", "content_hash_present": False}, "effect": {"decision": "deny", "reason": "content_hash_required", "required_action": "provide_content_hash"}},
	{"name": "document_retention_category_required", "condition": {"operation": "create_document", "retention_category_present": False}, "effect": {"decision": "deny", "reason": "retention_category_required", "required_action": "assign_retention_category"}},
	{"name": "retention_category_supported", "condition": {"operation": "create_document", "retention_category_supported": False}, "effect": {"decision": "deny", "reason": "retention_category_not_supported", "required_action": "select_supported_retention_category"}},
	# versioning
	{"name": "version_document_required", "condition": {"operation": "add_version", "document_present": False}, "effect": {"decision": "deny", "reason": "document_required", "required_action": "select_document"}},
	{"name": "version_author_required", "condition": {"operation": "add_version", "author_present": False}, "effect": {"decision": "deny", "reason": "version_author_required", "required_action": "provide_author"}},
	{"name": "version_change_summary_required", "condition": {"operation": "add_version", "change_summary_present": False}, "effect": {"decision": "deny", "reason": "change_summary_required", "required_action": "provide_change_summary"}},
	{"name": "version_content_hash_required", "condition": {"operation": "add_version", "content_hash_present": False}, "effect": {"decision": "deny", "reason": "content_hash_required", "required_action": "provide_content_hash"}},
	{"name": "version_deletion_denied", "condition": {"operation": "delete_version"}, "effect": {"decision": "deny", "reason": "version_deletion_denied", "required_action": "versions_are_immutable"}},
	# retention policy
	{"name": "retention_policy_category_supported", "condition": {"operation": "create_retention_policy", "retention_category_supported": False}, "effect": {"decision": "deny", "reason": "retention_category_not_supported", "required_action": "select_supported_category"}},
	{"name": "retention_policy_trigger_supported", "condition": {"operation": "create_retention_policy", "trigger_supported": False}, "effect": {"decision": "deny", "reason": "retention_trigger_not_supported", "required_action": "select_supported_trigger"}},
	{"name": "retention_policy_disposal_method_supported", "condition": {"operation": "create_retention_policy", "disposal_method_supported": False}, "effect": {"decision": "deny", "reason": "disposal_method_not_supported", "required_action": "select_supported_disposal_method"}},
	{"name": "retention_policy_years_positive", "condition": {"operation": "create_retention_policy", "years_positive": False}, "effect": {"decision": "deny", "reason": "retention_years_must_be_positive", "required_action": "set_positive_retention_years"}},
	# classification
	{"name": "classification_sensitivity_supported", "condition": {"operation": "classify_document", "sensitivity_supported": False}, "effect": {"decision": "deny", "reason": "sensitivity_level_not_supported", "required_action": "select_supported_sensitivity"}},
	{"name": "classification_framework_supported", "condition": {"operation": "classify_document", "regulatory_framework_supported": False}, "effect": {"decision": "deny", "reason": "regulatory_framework_not_supported", "required_action": "select_supported_framework"}},
	# disposal
	{"name": "disposal_document_required", "condition": {"operation": "dispose_documents", "document_present": False}, "effect": {"decision": "deny", "reason": "document_required", "required_action": "select_documents_for_disposal"}},
	{"name": "disposal_method_supported", "condition": {"operation": "dispose_documents", "disposal_method_supported": False}, "effect": {"decision": "deny", "reason": "disposal_method_not_supported", "required_action": "select_supported_disposal_method"}},
	{"name": "disposal_authorized_by_required", "condition": {"operation": "dispose_documents", "authorized_by_present": False}, "effect": {"decision": "deny", "reason": "disposal_authorization_required", "required_action": "provide_authorized_by"}},
	{"name": "unapproved_disposal_denied", "condition": {"operation": "dispose_documents", "disposal_approved": False}, "effect": {"decision": "deny", "reason": "disposal_requires_approval", "required_action": "obtain_disposal_approval"}},
	# workflow
	{"name": "workflow_type_supported", "condition": {"operation": "start_review_workflow", "workflow_type_supported": False}, "effect": {"decision": "deny", "reason": "workflow_type_not_supported", "required_action": "select_supported_workflow_type"}},
	{"name": "workflow_approvers_required", "condition": {"operation": "start_review_workflow", "approvers_present": False}, "effect": {"decision": "deny", "reason": "approvers_required", "required_action": "assign_approvers"}},
	{"name": "workflow_document_required", "condition": {"operation": "start_review_workflow", "document_present": False}, "effect": {"decision": "deny", "reason": "document_required", "required_action": "select_document"}},
	{"name": "workflow_decision_supported", "condition": {"operation": "approve_workflow_step", "decision_supported": False}, "effect": {"decision": "deny", "reason": "workflow_decision_not_supported", "required_action": "select_supported_decision"}},
	# governance
	{"name": "cross_tenant_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_access"}},
	{"name": "retention_override_denied", "condition": {"operation": "override_retention_policy", "retention_override": True}, "effect": {"decision": "deny", "reason": "retention_policy_override_denied", "required_action": "contact_records_manager"}},
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
			"api_prefix": "/ckm-ecm/api/v1",
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
