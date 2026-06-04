"""Executable capability contract for GRC Document Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "grc_doc"
CAPABILITY_NAME = "Document Management"
CAPABILITY_VERSION = "2.2.0"
DOC_EVENT_STREAM = "apg.grc.doc.lifecycle"

SUPPORTED_DOCUMENT_TYPES = [
	"policy", "procedure", "evidence", "contract", "report",
	"record", "template", "standard", "guideline", "charter",
	"framework", "memo", "notice", "form",
]
SUPPORTED_CLASSIFICATIONS = ["public", "internal", "confidential", "restricted", "top_secret"]
SUPPORTED_DOCUMENT_STATUSES = [
	"draft", "in_review", "approved", "published", "archived", "locked", "superseded", "withdrawn",
]
SUPPORTED_PERMISSIONS = ["view", "comment", "edit", "approve", "admin", "publish", "archive"]
SUPPORTED_PROCESSING_JOBS = [
	"classification", "extraction", "retention_review", "policy_mapping",
	"quality_review", "ocr", "redaction", "translation", "signature_verification",
]
SUPPORTED_RETENTION_CLASSES = [
	"permanent", "long_term", "medium_term", "short_term", "transient",
]
SUPPORTED_DOC_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_DOC_AGENT_ROLES = [
	"document_reviewer",
	"classification_reviewer",
	"retention_reviewer",
	"evidence_reviewer",
	"policy_reviewer",
	"publication_reviewer",
	"quality_reviewer",
	"redaction_reviewer",
]
SUPPORTED_REVIEW_OUTCOMES = ["approved", "rejected", "deferred", "requires_changes"]
SUPPORTED_ARCHIVE_REASONS = [
	"superseded", "expired_retention", "obsolete", "merged", "regulatory_mandate",
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
		"max_title_length": 512,
		"max_document_size_mb": 100,
	},
	"templates": {
		"name_required": True,
		"body_required": True,
		"owner_required": True,
		"supported_classifications": SUPPORTED_CLASSIFICATIONS,
		"max_template_size_mb": 10,
	},
	"revisions": {
		"document_required": True,
		"editor_required": True,
		"change_summary_required": True,
		"min_change_summary_length": 10,
	},
	"approvals": {
		"approver_required": True,
		"approval_note_required": True,
		"segregation_of_duties": True,
		"multi_approver_for_restricted": True,
		"min_approvers_restricted": 2,
	},
	"publication": {
		"approval_required": True,
		"publisher_required": True,
		"published_documents_locked": True,
		"notification_on_publish": True,
	},
	"retention": {
		"policy_required": True,
		"minimum_retention_days": 365,
		"legal_hold_blocks_archive": True,
		"supported_retention_classes": SUPPORTED_RETENTION_CLASSES,
		"permanent_retention_requires_approval": True,
	},
	"access": {
		"principal_required": True,
		"supported_permissions": SUPPORTED_PERMISSIONS,
		"restricted_documents_require_expiry": True,
		"cross_tenant_access_denied": True,
		"access_grant_audit_required": True,
	},
	"processing": {
		"supported_jobs": SUPPORTED_PROCESSING_JOBS,
		"processor": "bytewax",
		"document_required": True,
		"max_concurrent_jobs": 10,
	},
	"doc_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_DOC_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_DOC_AGENT_ROLES,
		"max_autonomous_scope": "review_prepare_and_recommend",
		"human_approval_required": True,
		"privileged_actions_logged": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_state_changes": True,
		"segregation_of_duties": True,
		"cross_tenant_access_denied": True,
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
		"emit_retention_events": True,
		"emit_archive_events": True,
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
		"multi_tenancy": "adapter",
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
		"enable_audit_trail": True,
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
	"document_audit_trail",
	"doc_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"mten",
	"conf",
	"ntfy",
	"grc_pol",
	"wflo",
	"srch",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/grc-doc/dashboard", "component": "DocumentDashboard", "permission": "grc_doc:view", "nav_group": "Overview"},
	{"name": "documents", "path": "/grc-doc/documents", "component": "DocumentRepository", "permission": "grc_doc:manage_documents", "nav_group": "Documents"},
	{"name": "document_detail", "path": "/grc-doc/documents/:id", "component": "DocumentDetail", "permission": "grc_doc:view", "nav_group": "Documents"},
	{"name": "templates", "path": "/grc-doc/templates", "component": "DocumentTemplateLibrary", "permission": "grc_doc:manage_templates", "nav_group": "Documents"},
	{"name": "reviews", "path": "/grc-doc/reviews", "component": "DocumentReviewQueue", "permission": "grc_doc:review", "nav_group": "Governance"},
	{"name": "retention", "path": "/grc-doc/retention", "component": "RetentionWorkbench", "permission": "grc_doc:manage_retention", "nav_group": "Governance"},
	{"name": "access", "path": "/grc-doc/access", "component": "DocumentAccessBoard", "permission": "grc_doc:manage_access", "nav_group": "Security"},
	{"name": "processing", "path": "/grc-doc/processing", "component": "DocumentProcessingQueue", "permission": "grc_doc:process", "nav_group": "Automation"},
	{"name": "agents", "path": "/grc-doc/agents", "component": "DocumentAgentWorkbench", "permission": "grc_doc:admin", "nav_group": "Automation"},
	{"name": "audit_trail", "path": "/grc-doc/audit-trail", "component": "DocumentAuditTrail", "permission": "grc_doc:view", "nav_group": "Compliance"},
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
		"templates": {"icon": "layout-template", "visual": "template-grid", "status_style": "template-chip"},
		"reviews": {"icon": "clipboard-check", "visual": "review-queue", "status_style": "review-chip"},
		"retention": {"icon": "archive", "visual": "retention-ledger", "status_style": "retention-chip"},
		"access": {"icon": "lock", "visual": "access-board", "status_style": "permission-chip"},
		"processing": {"icon": "cpu", "visual": "processing-lane", "status_style": "job-chip"},
		"agents": {"icon": "bot", "visual": "document-agent-lane", "status_style": "agent-chip"},
		"audit_trail": {"icon": "list", "visual": "audit-timeline", "status_style": "audit-chip"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"stream": DOC_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"document_created",
		"document_updated",
		"document_deleted",
		"template_registered",
		"template_updated",
		"document_revised",
		"document_review_requested",
		"document_review_completed",
		"document_approved",
		"document_rejected",
		"document_published",
		"document_archived",
		"document_superseded",
		"document_withdrawn",
		"retention_policy_assigned",
		"legal_hold_placed",
		"legal_hold_released",
		"document_access_granted",
		"document_access_revoked",
		"processing_job_registered",
		"processing_job_completed",
		"processing_job_failed",
		"doc_agent_registered",
		"doc_agent_action_approved",
		"doc_agent_action_rejected",
	],
	"states": SUPPORTED_DOCUMENT_STATUSES + ["queued", "processing", "completed", "failed", "expired", "on_hold"],
	"guardrails": [
		"doc_batch_requires_bytewax",
		"doc_event_requires_bytewax",
		"privileged_doc_agent_action_requires_human_approval",
		"cross_tenant_event_denied",
		"restricted_document_event_requires_encryption",
	],
}


RULES: list[dict[str, Any]] = [
	# Tenant and policy governance
	{
		"name": "tenant_context_required",
		"description": "Document operations require tenant context.",
		"condition": {"tenant_context_present": False},
		"effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"},
	},
	{
		"name": "cross_tenant_access_denied",
		"description": "Documents may not be accessed across tenant boundaries.",
		"condition": {"cross_tenant_access": True},
		"effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_identity"},
	},
	{
		"name": "doc_write_requires_policy",
		"description": "Document writes require policy attachment.",
		"condition": {"operation_type": "write", "policy_attached": False},
		"effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"},
	},
	{
		"name": "privilege_escalation_denied",
		"description": "Users may not grant document permissions exceeding their own level.",
		"condition": {"operation": "grant_access", "grant_exceeds_grantor_permission": True},
		"effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "reduce_grant_to_grantor_level"},
	},
	{
		"name": "admin_operation_requires_mfa",
		"description": "Admin-level document operations require MFA.",
		"condition": {"permission_required": "admin", "mfa_verified": False},
		"effect": {"decision": "deny", "reason": "mfa_required_for_admin", "required_action": "complete_mfa"},
	},
	# Document create
	{
		"name": "document_requires_title",
		"description": "Documents require title.",
		"condition": {"operation": "create_document", "title_present": False},
		"effect": {"decision": "deny", "reason": "document_title_required", "required_action": "set_document_title"},
	},
	{
		"name": "document_requires_owner",
		"description": "Documents require owner.",
		"condition": {"operation": "create_document", "owner_present": False},
		"effect": {"decision": "deny", "reason": "document_owner_required", "required_action": "assign_document_owner"},
	},
	{
		"name": "document_type_supported",
		"description": "Document type must be supported.",
		"condition": {"operation": "create_document", "document_type_supported": False},
		"effect": {"decision": "deny", "reason": "document_type_not_supported", "required_action": "select_supported_document_type"},
	},
	{
		"name": "classification_supported",
		"description": "Document classification must be supported.",
		"condition": {"operation": "create_document", "classification_supported": False},
		"effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"},
	},
	{
		"name": "document_requires_content_or_template",
		"description": "Documents require content or template.",
		"condition": {"operation": "create_document", "content_or_template_present": False},
		"effect": {"decision": "deny", "reason": "document_content_or_template_required", "required_action": "attach_content_or_template"},
	},
	{
		"name": "restricted_document_requires_review",
		"description": "Restricted documents require review.",
		"condition": {"operation": "create_document", "restricted_classification": True, "review_recorded": False},
		"effect": {"decision": "require_review", "reason": "restricted_document_review_required", "required_action": "record_document_review"},
	},
	# Document update
	{
		"name": "update_requires_document_exists",
		"description": "Updates target an existing document.",
		"condition": {"operation": "update_document", "document_exists": False},
		"effect": {"decision": "deny", "reason": "document_not_found", "required_action": "select_existing_document"},
	},
	{
		"name": "locked_document_update_denied",
		"description": "Locked documents cannot be directly updated.",
		"condition": {"operation": "update_document", "document_status": "locked"},
		"effect": {"decision": "deny", "reason": "document_is_locked", "required_action": "initiate_revision_workflow"},
	},
	# Document delete
	{
		"name": "delete_requires_admin",
		"description": "Document deletion requires admin permission.",
		"condition": {"operation": "delete_document", "has_admin_permission": False},
		"effect": {"decision": "deny", "reason": "admin_permission_required_for_delete", "required_action": "request_admin_permission"},
	},
	{
		"name": "delete_blocked_by_legal_hold",
		"description": "Documents under legal hold cannot be deleted.",
		"condition": {"operation": "delete_document", "legal_hold": True},
		"effect": {"decision": "deny", "reason": "legal_hold_blocks_delete", "required_action": "release_legal_hold"},
	},
	{
		"name": "published_document_delete_denied",
		"description": "Published documents must be archived, not deleted.",
		"condition": {"operation": "delete_document", "document_status": "published"},
		"effect": {"decision": "deny", "reason": "published_document_must_be_archived", "required_action": "archive_document_instead"},
	},
	# Template lifecycle
	{
		"name": "template_requires_name",
		"description": "Templates require name.",
		"condition": {"operation": "register_template", "name_present": False},
		"effect": {"decision": "deny", "reason": "template_name_required", "required_action": "set_template_name"},
	},
	{
		"name": "template_requires_body",
		"description": "Templates require body.",
		"condition": {"operation": "register_template", "body_present": False},
		"effect": {"decision": "deny", "reason": "template_body_required", "required_action": "set_template_body"},
	},
	{
		"name": "template_requires_owner",
		"description": "Templates require owner.",
		"condition": {"operation": "register_template", "owner_present": False},
		"effect": {"decision": "deny", "reason": "template_owner_required", "required_action": "assign_template_owner"},
	},
	{
		"name": "template_classification_supported",
		"description": "Template classification must be supported.",
		"condition": {"operation": "register_template", "classification_supported": False},
		"effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"},
	},
	# Revision workflow
	{
		"name": "revision_requires_document",
		"description": "Revisions require document.",
		"condition": {"operation": "create_revision", "document_present": False},
		"effect": {"decision": "deny", "reason": "document_required", "required_action": "select_document"},
	},
	{
		"name": "revision_requires_editor",
		"description": "Revisions require editor.",
		"condition": {"operation": "create_revision", "editor_present": False},
		"effect": {"decision": "deny", "reason": "editor_required", "required_action": "assign_editor"},
	},
	{
		"name": "revision_requires_change_summary",
		"description": "Revisions require change summary.",
		"condition": {"operation": "create_revision", "change_summary_present": False},
		"effect": {"decision": "deny", "reason": "change_summary_required", "required_action": "record_change_summary"},
	},
	{
		"name": "published_documents_are_locked",
		"description": "Published documents must be revised through controlled workflow.",
		"condition": {"operation": "create_revision", "published_document": True, "review_recorded": False},
		"effect": {"decision": "require_review", "reason": "published_document_revision_review_required", "required_action": "record_revision_review"},
	},
	# Approval workflow
	{
		"name": "approval_requires_document",
		"description": "Approvals require document.",
		"condition": {"operation": "approve_document", "document_present": False},
		"effect": {"decision": "deny", "reason": "document_required", "required_action": "select_document"},
	},
	{
		"name": "approval_requires_approver",
		"description": "Approvals require approver.",
		"condition": {"operation": "approve_document", "approver_present": False},
		"effect": {"decision": "deny", "reason": "approver_required", "required_action": "assign_approver"},
	},
	{
		"name": "approval_requires_note",
		"description": "Approvals require note.",
		"condition": {"operation": "approve_document", "approval_note_present": False},
		"effect": {"decision": "deny", "reason": "approval_note_required", "required_action": "record_approval_note"},
	},
	{
		"name": "approval_segregation_required",
		"description": "Document owner cannot approve own restricted document.",
		"condition": {"operation": "approve_document", "owner_is_approver": True, "restricted_classification": True},
		"effect": {"decision": "deny", "reason": "segregation_of_duties_required", "required_action": "select_independent_approver"},
	},
	{
		"name": "restricted_requires_multi_approver",
		"description": "Restricted/top-secret documents require at least 2 independent approvers.",
		"condition": {"operation": "approve_document", "restricted_classification": True, "approver_count_lt": 2},
		"effect": {"decision": "deny", "reason": "multi_approver_required_for_restricted", "required_action": "add_second_approver"},
	},
	# Reject workflow
	{
		"name": "rejection_requires_reason",
		"description": "Document rejections require a stated reason.",
		"condition": {"operation": "reject_document", "rejection_reason_present": False},
		"effect": {"decision": "deny", "reason": "rejection_reason_required", "required_action": "record_rejection_reason"},
	},
	# Publication workflow
	{
		"name": "publish_requires_document",
		"description": "Publishing requires document.",
		"condition": {"operation": "publish_document", "document_present": False},
		"effect": {"decision": "deny", "reason": "document_required", "required_action": "select_document"},
	},
	{
		"name": "publish_requires_approval",
		"description": "Publishing requires approval.",
		"condition": {"operation": "publish_document", "approved": False},
		"effect": {"decision": "deny", "reason": "document_approval_required", "required_action": "approve_document"},
	},
	{
		"name": "publish_requires_publisher",
		"description": "Publishing requires publisher.",
		"condition": {"operation": "publish_document", "publisher_present": False},
		"effect": {"decision": "deny", "reason": "publisher_required", "required_action": "assign_publisher"},
	},
	{
		"name": "publish_blocks_draft_status",
		"description": "Draft documents cannot be published without completing the review workflow.",
		"condition": {"operation": "publish_document", "document_status": "draft"},
		"effect": {"decision": "deny", "reason": "draft_must_complete_review_before_publish", "required_action": "submit_for_review"},
	},
	# Retention and archive
	{
		"name": "retention_requires_document",
		"description": "Retention policies require document.",
		"condition": {"operation": "assign_retention_policy", "document_present": False},
		"effect": {"decision": "deny", "reason": "document_required", "required_action": "select_document"},
	},
	{
		"name": "retention_days_minimum",
		"description": "Retention days must meet minimum.",
		"condition": {"operation": "assign_retention_policy", "retention_days_lt": 365},
		"effect": {"decision": "deny", "reason": "retention_too_short", "required_action": "increase_retention"},
	},
	{
		"name": "permanent_retention_requires_approval",
		"description": "Permanent retention classification requires explicit approval.",
		"condition": {"operation": "assign_retention_policy", "retention_class": "permanent", "approval_recorded": False},
		"effect": {"decision": "require_review", "reason": "permanent_retention_approval_required", "required_action": "record_retention_approval"},
	},
	{
		"name": "archive_blocks_legal_hold",
		"description": "Legal hold blocks archive.",
		"condition": {"operation": "archive_document", "legal_hold": True},
		"effect": {"decision": "deny", "reason": "legal_hold_blocks_archive", "required_action": "release_legal_hold"},
	},
	{
		"name": "archive_requires_reason",
		"description": "Archiving a document requires a stated reason.",
		"condition": {"operation": "archive_document", "archive_reason_present": False},
		"effect": {"decision": "deny", "reason": "archive_reason_required", "required_action": "record_archive_reason"},
	},
	{
		"name": "archive_reason_supported",
		"description": "Archive reason must be from the supported list.",
		"condition": {"operation": "archive_document", "archive_reason_supported": False},
		"effect": {"decision": "deny", "reason": "unsupported_archive_reason", "required_action": "select_supported_archive_reason"},
	},
	# Access control
	{
		"name": "access_requires_document",
		"description": "Access grants require document.",
		"condition": {"operation": "grant_access", "document_present": False},
		"effect": {"decision": "deny", "reason": "document_required", "required_action": "select_document"},
	},
	{
		"name": "access_requires_principal",
		"description": "Access grants require principal.",
		"condition": {"operation": "grant_access", "principal_present": False},
		"effect": {"decision": "deny", "reason": "principal_required", "required_action": "select_principal"},
	},
	{
		"name": "access_permission_supported",
		"description": "Access permission must be supported.",
		"condition": {"operation": "grant_access", "permission_supported": False},
		"effect": {"decision": "deny", "reason": "permission_not_supported", "required_action": "select_supported_permission"},
	},
	{
		"name": "restricted_access_requires_expiry",
		"description": "Restricted document grants require expiry.",
		"condition": {"operation": "grant_access", "restricted_classification": True, "expiry_present": False},
		"effect": {"decision": "deny", "reason": "restricted_access_expiry_required", "required_action": "set_access_expiry"},
	},
	{
		"name": "cross_tenant_principal_denied",
		"description": "Access cannot be granted to principals from a different tenant.",
		"condition": {"operation": "grant_access", "principal_tenant_mismatch": True},
		"effect": {"decision": "deny", "reason": "cross_tenant_principal_denied", "required_action": "use_same_tenant_principal"},
	},
	# Processing
	{
		"name": "processing_requires_document",
		"description": "Processing jobs require document.",
		"condition": {"operation": "register_processing_job", "document_present": False},
		"effect": {"decision": "deny", "reason": "document_required", "required_action": "select_document"},
	},
	{
		"name": "processing_job_supported",
		"description": "Processing job type must be supported.",
		"condition": {"operation": "register_processing_job", "job_type_supported": False},
		"effect": {"decision": "deny", "reason": "processing_job_not_supported", "required_action": "select_supported_processing_job"},
	},
	{
		"name": "processing_requires_bytewax",
		"description": "Document processing requires Bytewax.",
		"condition": {"operation": "register_processing_job", "processor_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_processor_required", "required_action": "route_processing_to_bytewax"},
	},
	{
		"name": "doc_batch_requires_bytewax",
		"description": "Document batches require Bytewax coordination.",
		"condition": {"operation": "doc_batch", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_doc_batch_to_bytewax"},
	},
	{
		"name": "doc_event_requires_bytewax",
		"description": "Document events require Bytewax.",
		"condition": {"operation": "doc_event", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_doc_event_to_bytewax"},
	},
	# Agent governance
	{
		"name": "doc_agent_runtime_supported",
		"description": "Document agents must use an approved runtime.",
		"condition": {"operation": "register_doc_agent", "agent_runtime_supported": False},
		"effect": {"decision": "deny", "reason": "doc_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"},
	},
	{
		"name": "doc_agent_role_supported",
		"description": "Document agents must use an approved role.",
		"condition": {"operation": "register_doc_agent", "agent_role_supported": False},
		"effect": {"decision": "deny", "reason": "doc_agent_role_not_supported", "required_action": "select_supported_agent_role"},
	},
	{
		"name": "privileged_doc_agent_action_requires_human_approval",
		"description": "Privileged document actions proposed by agents require human approval.",
		"condition": {"operation": "doc_agent_action", "privileged_scope": True, "human_approval_recorded": False},
		"effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"},
	},
	# Data quality
	{
		"name": "document_title_max_length",
		"description": "Document title must not exceed maximum length.",
		"condition": {"operation": "create_document", "title_length_gt": 512},
		"effect": {"decision": "deny", "reason": "document_title_too_long", "required_action": "shorten_document_title"},
	},
	{
		"name": "change_summary_min_length",
		"description": "Change summary must meet minimum length for quality.",
		"condition": {"operation": "create_revision", "change_summary_length_lt": 10},
		"effect": {"decision": "deny", "reason": "change_summary_too_short", "required_action": "expand_change_summary"},
	},
	# Escalation
	{
		"name": "escalate_requires_reason",
		"description": "Document escalations require a stated reason.",
		"condition": {"operation": "escalate_document", "escalation_reason_present": False},
		"effect": {"decision": "deny", "reason": "escalation_reason_required", "required_action": "record_escalation_reason"},
	},
	{
		"name": "escalate_requires_target",
		"description": "Document escalations require a target reviewer or role.",
		"condition": {"operation": "escalate_document", "escalation_target_present": False},
		"effect": {"decision": "deny", "reason": "escalation_target_required", "required_action": "specify_escalation_target"},
	},
	# Domain-specific governance
	{
		"name": "superseded_document_points_to_successor",
		"description": "Superseded documents must reference their successor.",
		"condition": {"operation": "supersede_document", "successor_id_present": False},
		"effect": {"decision": "deny", "reason": "successor_document_required", "required_action": "link_successor_document"},
	},
	{
		"name": "withdrawn_document_requires_rationale",
		"description": "Document withdrawal requires a governance rationale.",
		"condition": {"operation": "withdraw_document", "withdrawal_rationale_present": False},
		"effect": {"decision": "deny", "reason": "withdrawal_rationale_required", "required_action": "record_withdrawal_rationale"},
	},
	{
		"name": "top_secret_document_requires_dual_approval",
		"description": "Top-secret documents require dual approval from separate approvers.",
		"condition": {"operation": "approve_document", "classification": "top_secret", "approver_count_lt": 2},
		"effect": {"decision": "deny", "reason": "dual_approval_required_for_top_secret", "required_action": "obtain_second_independent_approval"},
	},
	{
		"name": "document_access_audit_required",
		"description": "Every access grant to a document must be audited.",
		"condition": {"operation": "grant_access", "audit_trail_enabled": False},
		"effect": {"decision": "deny", "reason": "access_audit_required", "required_action": "enable_audit_trail"},
	},
	{
		"name": "bulk_delete_requires_supervisor_approval",
		"description": "Bulk document deletion requires supervisor-level approval.",
		"condition": {"operation": "bulk_delete_documents", "supervisor_approval_present": False},
		"effect": {"decision": "deny", "reason": "supervisor_approval_required_for_bulk_delete", "required_action": "obtain_supervisor_approval"},
	},
]


def _configuration_schema() -> dict[str, Any]:
	return {
		"type": "object",
		"required": ["tenant_id", "ui", "theme"],
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
		"version": CAPABILITY_VERSION,
		"configuration": configuration,
		"configuration_schema": _configuration_schema(),
		"provides": PROVIDES,
		"requires": REQUIRES,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
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
