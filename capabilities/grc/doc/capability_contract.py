"""Executable capability contract for GRC Document Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "grc_doc"
CAPABILITY_NAME = "Document Management"
CAPABILITY_VERSION = "2.1.0"
DOC_EVENT_STREAM = "apg.grc.doc.lifecycle"

SUPPORTED_DOCUMENT_TYPES = ["policy", "procedure", "evidence", "contract", "report", "record", "template"]
SUPPORTED_CLASSIFICATIONS = ["public", "internal", "confidential", "restricted"]
SUPPORTED_DOCUMENT_STATUSES = ["draft", "in_review", "approved", "published", "archived", "locked"]
SUPPORTED_PERMISSIONS = ["view", "comment", "edit", "approve", "admin"]
SUPPORTED_PROCESSING_JOBS = ["classification", "extraction", "retention_review", "policy_mapping", "quality_review"]
SUPPORTED_DOC_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_DOC_AGENT_ROLES = [
	"document_reviewer",
	"classification_reviewer",
	"retention_reviewer",
	"evidence_reviewer",
	"policy_reviewer",
	"publication_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"documents": {
		"title_required": True,
		"owner_required": True,
		"content_or_template_required": True,
		"supported_document_types": SUPPORTED_DOCUMENT_TYPES,
		"supported_classifications": SUPPORTED_CLASSIFICATIONS,
		"restricted_review_required": True,
	},
	"templates": {
		"name_required": True,
		"body_required": True,
		"owner_required": True,
		"supported_classifications": SUPPORTED_CLASSIFICATIONS,
	},
	"revisions": {
		"document_required": True,
		"editor_required": True,
		"change_summary_required": True,
	},
	"approvals": {
		"approver_required": True,
		"approval_note_required": True,
		"segregation_of_duties": True,
	},
	"publication": {
		"approval_required": True,
		"publisher_required": True,
		"published_documents_locked": True,
	},
	"retention": {
		"policy_required": True,
		"minimum_retention_days": 365,
		"legal_hold_blocks_archive": True,
	},
	"access": {
		"principal_required": True,
		"supported_permissions": SUPPORTED_PERMISSIONS,
		"restricted_documents_require_expiry": True,
	},
	"processing": {
		"supported_jobs": SUPPORTED_PROCESSING_JOBS,
		"processor": "bytewax",
		"document_required": True,
	},
	"doc_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_DOC_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_DOC_AGENT_ROLES,
		"max_autonomous_scope": "review_prepare_and_recommend",
		"human_approval_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_state_changes": True,
		"segregation_of_duties": True,
	},
	"observability": {
		"event_stream": DOC_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_document_events": True,
		"emit_template_events": True,
		"emit_revision_events": True,
		"emit_approval_events": True,
		"emit_access_events": True,
		"emit_processing_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"notification": "adapter",
		"storage": "adapter",
		"search": "adapter",
		"workflow_orchestration": "adapter",
		"policy_management": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_documents": True,
		"enable_templates": True,
		"enable_reviews": True,
		"enable_retention": True,
		"enable_access": True,
		"enable_processing": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {
		"default_theme": "grc_doc_control",
		"allow_tenant_overrides": True,
	},
}


PROVIDES = [
	"document_repository_lifecycle",
	"document_template_lifecycle",
	"document_revision_workflow",
	"document_approval_workflow",
	"document_publication_workflow",
	"document_retention_workflow",
	"document_access_workflow",
	"document_processing_workflow",
	"document_dashboard_service",
	"doc_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"composition_events",
	"composition_config",
	"policy_management",
	"workflow_orchestration",
	"search",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/grc-doc/dashboard", "component": "DocumentDashboard", "permission": "grc_doc:view", "nav_group": "Overview"},
	{"name": "documents", "path": "/grc-doc/documents", "component": "DocumentRepository", "permission": "grc_doc:manage_documents", "nav_group": "Documents"},
	{"name": "templates", "path": "/grc-doc/templates", "component": "DocumentTemplateLibrary", "permission": "grc_doc:manage_templates", "nav_group": "Documents"},
	{"name": "reviews", "path": "/grc-doc/reviews", "component": "DocumentReviewQueue", "permission": "grc_doc:review", "nav_group": "Governance"},
	{"name": "retention", "path": "/grc-doc/retention", "component": "RetentionWorkbench", "permission": "grc_doc:manage_retention", "nav_group": "Governance"},
	{"name": "access", "path": "/grc-doc/access", "component": "DocumentAccessBoard", "permission": "grc_doc:manage_access", "nav_group": "Security"},
	{"name": "processing", "path": "/grc-doc/processing", "component": "DocumentProcessingQueue", "permission": "grc_doc:process", "nav_group": "Automation"},
	{"name": "agents", "path": "/grc-doc/agents", "component": "DocumentAgentWorkbench", "permission": "grc_doc:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/grc-doc/settings", "component": "DocumentSettings", "permission": "grc_doc:admin", "nav_group": "Administration"},
]


THEME = {
	"name": "grc_doc_control",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#4C6F52",
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
		"documents": {"icon": "file-text", "status_indicator": "document-pill", "visual": "repository-list"},
		"templates": {"visual": "template-grid", "status_style": "template-chip"},
		"reviews": {"visual": "review-queue", "status_style": "review-chip"},
		"retention": {"visual": "retention-ledger", "status_style": "retention-chip"},
		"access": {"visual": "access-board", "status_style": "permission-chip"},
		"processing": {"visual": "processing-lane", "status_style": "job-chip"},
		"agents": {"visual": "document-agent-lane", "status_style": "agent-chip"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"stream": DOC_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"document_created",
		"template_registered",
		"document_revised",
		"document_approved",
		"document_published",
		"retention_policy_assigned",
		"document_access_granted",
		"processing_job_registered",
		"processing_job_completed",
		"doc_agent_registered",
	],
	"states": SUPPORTED_DOCUMENT_STATUSES + ["queued", "completed", "expired"],
	"guardrails": [
		"doc_batch_requires_bytewax",
		"doc_event_requires_bytewax",
		"privileged_doc_agent_action_requires_human_approval",
	],
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Document operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "doc_write_requires_policy", "description": "Document writes require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "document_requires_title", "description": "Documents require title.", "condition": {"operation": "create_document", "title_present": False}, "effect": {"decision": "deny", "reason": "document_title_required", "required_action": "set_document_title"}},
	{"name": "document_requires_owner", "description": "Documents require owner.", "condition": {"operation": "create_document", "owner_present": False}, "effect": {"decision": "deny", "reason": "document_owner_required", "required_action": "assign_document_owner"}},
	{"name": "document_type_supported", "description": "Document type must be supported.", "condition": {"operation": "create_document", "document_type_supported": False}, "effect": {"decision": "deny", "reason": "document_type_not_supported", "required_action": "select_supported_document_type"}},
	{"name": "classification_supported", "description": "Document classification must be supported.", "condition": {"operation": "create_document", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "document_requires_content_or_template", "description": "Documents require content or template.", "condition": {"operation": "create_document", "content_or_template_present": False}, "effect": {"decision": "deny", "reason": "document_content_or_template_required", "required_action": "attach_content_or_template"}},
	{"name": "restricted_document_requires_review", "description": "Restricted documents require review.", "condition": {"operation": "create_document", "restricted_classification": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "restricted_document_review_required", "required_action": "record_document_review"}},
	{"name": "template_requires_name", "description": "Templates require name.", "condition": {"operation": "register_template", "name_present": False}, "effect": {"decision": "deny", "reason": "template_name_required", "required_action": "set_template_name"}},
	{"name": "template_requires_body", "description": "Templates require body.", "condition": {"operation": "register_template", "body_present": False}, "effect": {"decision": "deny", "reason": "template_body_required", "required_action": "set_template_body"}},
	{"name": "template_requires_owner", "description": "Templates require owner.", "condition": {"operation": "register_template", "owner_present": False}, "effect": {"decision": "deny", "reason": "template_owner_required", "required_action": "assign_template_owner"}},
	{"name": "template_classification_supported", "description": "Template classification must be supported.", "condition": {"operation": "register_template", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "revision_requires_document", "description": "Revisions require document.", "condition": {"operation": "create_revision", "document_present": False}, "effect": {"decision": "deny", "reason": "document_required", "required_action": "select_document"}},
	{"name": "revision_requires_editor", "description": "Revisions require editor.", "condition": {"operation": "create_revision", "editor_present": False}, "effect": {"decision": "deny", "reason": "editor_required", "required_action": "assign_editor"}},
	{"name": "revision_requires_change_summary", "description": "Revisions require change summary.", "condition": {"operation": "create_revision", "change_summary_present": False}, "effect": {"decision": "deny", "reason": "change_summary_required", "required_action": "record_change_summary"}},
	{"name": "published_documents_are_locked", "description": "Published documents must be revised through controlled workflow.", "condition": {"operation": "create_revision", "published_document": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "published_document_revision_review_required", "required_action": "record_revision_review"}},
	{"name": "approval_requires_document", "description": "Approvals require document.", "condition": {"operation": "approve_document", "document_present": False}, "effect": {"decision": "deny", "reason": "document_required", "required_action": "select_document"}},
	{"name": "approval_requires_approver", "description": "Approvals require approver.", "condition": {"operation": "approve_document", "approver_present": False}, "effect": {"decision": "deny", "reason": "approver_required", "required_action": "assign_approver"}},
	{"name": "approval_requires_note", "description": "Approvals require note.", "condition": {"operation": "approve_document", "approval_note_present": False}, "effect": {"decision": "deny", "reason": "approval_note_required", "required_action": "record_approval_note"}},
	{"name": "approval_segregation_required", "description": "Document owner cannot approve own restricted document.", "condition": {"operation": "approve_document", "owner_is_approver": True, "restricted_classification": True}, "effect": {"decision": "deny", "reason": "segregation_of_duties_required", "required_action": "select_independent_approver"}},
	{"name": "publish_requires_document", "description": "Publishing requires document.", "condition": {"operation": "publish_document", "document_present": False}, "effect": {"decision": "deny", "reason": "document_required", "required_action": "select_document"}},
	{"name": "publish_requires_approval", "description": "Publishing requires approval.", "condition": {"operation": "publish_document", "approved": False}, "effect": {"decision": "deny", "reason": "document_approval_required", "required_action": "approve_document"}},
	{"name": "publish_requires_publisher", "description": "Publishing requires publisher.", "condition": {"operation": "publish_document", "publisher_present": False}, "effect": {"decision": "deny", "reason": "publisher_required", "required_action": "assign_publisher"}},
	{"name": "retention_requires_document", "description": "Retention policies require document.", "condition": {"operation": "assign_retention_policy", "document_present": False}, "effect": {"decision": "deny", "reason": "document_required", "required_action": "select_document"}},
	{"name": "retention_days_minimum", "description": "Retention days must meet minimum.", "condition": {"operation": "assign_retention_policy", "retention_days_lt": 365}, "effect": {"decision": "deny", "reason": "retention_too_short", "required_action": "increase_retention"}},
	{"name": "archive_blocks_legal_hold", "description": "Legal hold blocks archive.", "condition": {"operation": "archive_document", "legal_hold": True}, "effect": {"decision": "deny", "reason": "legal_hold_blocks_archive", "required_action": "release_legal_hold"}},
	{"name": "access_requires_document", "description": "Access grants require document.", "condition": {"operation": "grant_access", "document_present": False}, "effect": {"decision": "deny", "reason": "document_required", "required_action": "select_document"}},
	{"name": "access_requires_principal", "description": "Access grants require principal.", "condition": {"operation": "grant_access", "principal_present": False}, "effect": {"decision": "deny", "reason": "principal_required", "required_action": "select_principal"}},
	{"name": "access_permission_supported", "description": "Access permission must be supported.", "condition": {"operation": "grant_access", "permission_supported": False}, "effect": {"decision": "deny", "reason": "permission_not_supported", "required_action": "select_supported_permission"}},
	{"name": "restricted_access_requires_expiry", "description": "Restricted document grants require expiry.", "condition": {"operation": "grant_access", "restricted_classification": True, "expiry_present": False}, "effect": {"decision": "deny", "reason": "restricted_access_expiry_required", "required_action": "set_access_expiry"}},
	{"name": "processing_requires_document", "description": "Processing jobs require document.", "condition": {"operation": "register_processing_job", "document_present": False}, "effect": {"decision": "deny", "reason": "document_required", "required_action": "select_document"}},
	{"name": "processing_job_supported", "description": "Processing job type must be supported.", "condition": {"operation": "register_processing_job", "job_type_supported": False}, "effect": {"decision": "deny", "reason": "processing_job_not_supported", "required_action": "select_supported_processing_job"}},
	{"name": "processing_requires_bytewax", "description": "Document processing requires Bytewax.", "condition": {"operation": "register_processing_job", "processor_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_processor_required", "required_action": "route_processing_to_bytewax"}},
	{"name": "doc_batch_requires_bytewax", "description": "Document batches require Bytewax coordination.", "condition": {"operation": "doc_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_doc_batch_to_bytewax"}},
	{"name": "doc_event_requires_bytewax", "description": "Document events require Bytewax.", "condition": {"operation": "doc_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_doc_event_to_bytewax"}},
	{"name": "doc_agent_runtime_supported", "description": "Document agents must use an approved runtime.", "condition": {"operation": "register_doc_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "doc_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "doc_agent_role_supported", "description": "Document agents must use an approved role.", "condition": {"operation": "register_doc_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "doc_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_doc_agent_action_requires_human_approval", "description": "Privileged document actions proposed by agents require human approval.", "condition": {"operation": "doc_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
		"display_name": CAPABILITY_NAME,
		"name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": configuration,
		"configuration_schema": _configuration_schema(),
		"provides": PROVIDES,
		"requires": REQUIRES,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/grc-doc/api/v1",
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
