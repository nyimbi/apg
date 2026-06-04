"""Executable capability contract for APG Signals Intelligence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_sigint"
CAPABILITY_NAME = "Signals Intelligence"
CAPABILITY_VERSION = "1.1.0"
SIGINT_EVENT_STREAM = "apg.intel.sigint.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["warrant", "consent", "partner_authority", "mission_order", "regulatory_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_SOURCE_TYPES = ["radio", "spectrum_sensor", "satellite_metadata", "telecom_metadata", "iot_telemetry", "maritime_ais", "aviation_adsb", "partner_feed"]
SUPPORTED_BANDS = ["hf", "vhf", "uhf", "l_band", "s_band", "c_band", "x_band", "ku_band", "ka_band", "metadata"]
SUPPORTED_COLLECTION_MODES = ["metadata_only", "spectrum_monitoring", "telemetry_ingest", "partner_feed", "historical_import"]
SUPPORTED_PROCESSING_TYPES = ["normalization", "deduplication", "demodulation_metadata", "entity_resolution", "traffic_analysis", "anomaly_detection"]
SUPPORTED_PATTERN_TYPES = ["beacon", "burst", "route", "contact_graph", "anomaly", "watchlist_match", "trend"]
SUPPORTED_ASSESSMENT_TYPES = ["threat", "pattern_summary", "network_profile", "activity_report", "watchlist_update"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["authority_reviewer", "collection_planner", "signal_processor", "pattern_analyst", "minimization_reviewer", "dissemination_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"sources": {"supported_source_types": SUPPORTED_SOURCE_TYPES, "supported_bands": SUPPORTED_BANDS, "owner_required": True, "authority_required": True, "evidence_required": True},
	"collection_tasks": {"supported_modes": SUPPORTED_COLLECTION_MODES, "authority_required": True, "source_required": True, "retention_positive": True, "minimization_required": True, "approval_required": True, "evidence_required": True},
	"observations": {"task_required": True, "observation_reference_required": True, "fingerprint_required": True, "confidence_required": True, "evidence_required": True},
	"processing": {"supported_types": SUPPORTED_PROCESSING_TYPES, "observation_required": True, "quality_required": True, "analyst_required": True, "evidence_required": True},
	"patterns": {"supported_types": SUPPORTED_PATTERN_TYPES, "batch_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"assessments": {"supported_types": SUPPORTED_ASSESSMENT_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "pattern_required": True, "analyst_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "minimization_required": True, "cross_tenant_sigint_denied": True, "privilege_escalation_denied": True, "unauthorized_content_intercept_denied": True, "minimization_bypass_denied": True, "unapproved_bulk_collection_denied": True, "offensive_sigint_denied": True},
	"observability": {"event_stream": SIGINT_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "radio": "intel_radio", "crawler": "intel_crawler", "graph": "grph", "rag": "ragn", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_sources": True, "enable_collection_tasks": True, "enable_observations": True, "enable_processing": True, "enable_patterns": True, "enable_assessments": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_sigint_control", "allow_tenant_overrides": True},
}

PROVIDES = ["sigint_authority_workflow", "sigint_source_workflow", "sigint_collection_workflow", "sigint_observation_workflow", "sigint_processing_workflow", "sigint_pattern_workflow", "sigint_assessment_workflow", "sigint_review_workflow", "sigint_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "intel_radio", "intel_crawler", "grph", "ragn"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-sigint/dashboard", "component": "SIGINTDashboard", "permission": "intel_sigint:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-sigint/authorities", "component": "SignalAuthorityConsole", "permission": "intel_sigint:authorities", "nav_group": "Governance"},
	{"name": "sources", "path": "/intel-sigint/sources", "component": "SignalSourceRegistry", "permission": "intel_sigint:sources", "nav_group": "Collection"},
	{"name": "collection_tasks", "path": "/intel-sigint/collection-tasks", "component": "SignalCollectionPlanner", "permission": "intel_sigint:collection", "nav_group": "Collection"},
	{"name": "observations", "path": "/intel-sigint/observations", "component": "SignalObservationLedger", "permission": "intel_sigint:observations", "nav_group": "Processing"},
	{"name": "processing", "path": "/intel-sigint/processing", "component": "SignalProcessingWorkbench", "permission": "intel_sigint:processing", "nav_group": "Processing"},
	{"name": "patterns", "path": "/intel-sigint/patterns", "component": "SignalPatternWorkbench", "permission": "intel_sigint:patterns", "nav_group": "Analysis"},
	{"name": "assessments", "path": "/intel-sigint/assessments", "component": "SignalAssessmentWorkbench", "permission": "intel_sigint:assessments", "nav_group": "Analysis"},
	{"name": "reviews", "path": "/intel-sigint/reviews", "component": "SIGINTReviewConsole", "permission": "intel_sigint:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-sigint/agents", "component": "SIGINTAgentWorkbench", "permission": "intel_sigint:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-sigint/settings", "component": "SIGINTSettings", "permission": "intel_sigint:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_sigint_control",
	"tokens": {"color.primary": "#1F4E5F", "color.accent": "#7C2D12", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "sources": {"icon": "radio-tower", "status_indicator": "source-chip"}, "collection_tasks": {"icon": "calendar-clock", "status_indicator": "task-chip"}, "observations": {"icon": "waveform", "status_indicator": "confidence-chip"}, "processing": {"icon": "cpu", "status_indicator": "quality-chip"}, "patterns": {"icon": "activity", "status_indicator": "pattern-chip"}, "assessments": {"icon": "file-search", "status_indicator": "classification-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": SIGINT_EVENT_STREAM, "key": "tenant_id", "events": ["sigint_authority_recorded", "sigint_source_registered", "sigint_collection_task_recorded", "sigint_observation_recorded", "sigint_processing_batch_recorded", "sigint_pattern_recorded", "sigint_assessment_recorded", "sigint_review_recorded", "sigint_agent_registered"], "guardrails": ["sigint_batch_requires_bytewax", "privileged_sigint_agent_action_requires_human_approval", "cross_tenant_sigint_action_denied", "privilege_escalation_action_denied", "unauthorized_content_intercept_action_denied", "minimization_bypass_action_denied", "unapproved_bulk_collection_action_denied", "offensive_sigint_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "sigint_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "sigint_policy_required", "required_action": "attach_sigint_policy"}},
	{"name": "authority_type_supported", "condition": {"operation": "record_authority", "authority_type_supported": False}, "effect": {"decision": "deny", "reason": "authority_type_not_supported", "required_action": "select_supported_authority_type"}},
	{"name": "authority_scope_required", "condition": {"operation": "record_authority", "scope_present": False}, "effect": {"decision": "deny", "reason": "authority_scope_required", "required_action": "attach_scope_reference"}},
	{"name": "authority_classification_supported", "condition": {"operation": "record_authority", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "authority_approver_required", "condition": {"operation": "record_authority", "approver_present": False}, "effect": {"decision": "deny", "reason": "authority_approver_required", "required_action": "record_approver"}},
	{"name": "authority_expiry_required", "condition": {"operation": "record_authority", "expiry_present": False}, "effect": {"decision": "deny", "reason": "authority_expiry_required", "required_action": "set_expiry"}},
	{"name": "authority_evidence_required", "condition": {"operation": "record_authority", "evidence_present": False}, "effect": {"decision": "deny", "reason": "authority_evidence_required", "required_action": "attach_authority_evidence"}},
	{"name": "source_type_supported", "condition": {"operation": "register_source", "source_type_supported": False}, "effect": {"decision": "deny", "reason": "source_type_not_supported", "required_action": "select_supported_source_type"}},
	{"name": "source_band_supported", "condition": {"operation": "register_source", "band_supported": False}, "effect": {"decision": "deny", "reason": "signal_band_not_supported", "required_action": "select_supported_band"}},
	{"name": "source_reference_required", "condition": {"operation": "register_source", "source_reference_present": False}, "effect": {"decision": "deny", "reason": "source_reference_required", "required_action": "attach_source_reference"}},
	{"name": "source_owner_required", "condition": {"operation": "register_source", "owner_present": False}, "effect": {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}},
	{"name": "source_authority_required", "condition": {"operation": "register_source", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "source_evidence_required", "condition": {"operation": "register_source", "evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "task_authority_required", "condition": {"operation": "record_collection_task", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "task_source_required", "condition": {"operation": "record_collection_task", "source_present": False}, "effect": {"decision": "deny", "reason": "source_required", "required_action": "select_source"}},
	{"name": "task_source_authority_match", "condition": {"operation": "record_collection_task", "source_authority_match": False}, "effect": {"decision": "deny", "reason": "source_authority_mismatch", "required_action": "select_source_for_authority"}},
	{"name": "task_mode_supported", "condition": {"operation": "record_collection_task", "collection_mode_supported": False}, "effect": {"decision": "deny", "reason": "collection_mode_not_supported", "required_action": "select_supported_collection_mode"}},
	{"name": "task_retention_positive", "condition": {"operation": "record_collection_task", "retention_days_positive": False}, "effect": {"decision": "deny", "reason": "retention_days_invalid", "required_action": "set_positive_retention_days"}},
	{"name": "task_minimization_required", "condition": {"operation": "record_collection_task", "minimization_present": False}, "effect": {"decision": "deny", "reason": "minimization_reference_required", "required_action": "attach_minimization_reference"}},
	{"name": "task_approval_required", "condition": {"operation": "record_collection_task", "approval_present": False}, "effect": {"decision": "deny", "reason": "collection_approval_required", "required_action": "attach_collection_approval"}},
	{"name": "task_evidence_required", "condition": {"operation": "record_collection_task", "evidence_present": False}, "effect": {"decision": "deny", "reason": "collection_task_evidence_required", "required_action": "attach_task_evidence"}},
	{"name": "observation_task_required", "condition": {"operation": "record_observation", "task_present": False}, "effect": {"decision": "deny", "reason": "collection_task_required", "required_action": "select_collection_task"}},
	{"name": "observation_reference_required", "condition": {"operation": "record_observation", "observation_reference_present": False}, "effect": {"decision": "deny", "reason": "observation_reference_required", "required_action": "attach_observation_reference"}},
	{"name": "observation_fingerprint_required", "condition": {"operation": "record_observation", "fingerprint_present": False}, "effect": {"decision": "deny", "reason": "fingerprint_required", "required_action": "attach_fingerprint"}},
	{"name": "observation_confidence_valid", "condition": {"operation": "record_observation", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "observation_evidence_required", "condition": {"operation": "record_observation", "evidence_present": False}, "effect": {"decision": "deny", "reason": "observation_evidence_required", "required_action": "attach_observation_evidence"}},
	{"name": "processing_observation_required", "condition": {"operation": "record_processing_batch", "observation_present": False}, "effect": {"decision": "deny", "reason": "observation_required", "required_action": "select_observation"}},
	{"name": "processing_type_supported", "condition": {"operation": "record_processing_batch", "processing_type_supported": False}, "effect": {"decision": "deny", "reason": "processing_type_not_supported", "required_action": "select_supported_processing_type"}},
	{"name": "processing_quality_valid", "condition": {"operation": "record_processing_batch", "quality_valid": False}, "effect": {"decision": "deny", "reason": "quality_score_invalid", "required_action": "set_quality_0_to_1"}},
	{"name": "processing_analyst_required", "condition": {"operation": "record_processing_batch", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "processing_evidence_required", "condition": {"operation": "record_processing_batch", "evidence_present": False}, "effect": {"decision": "deny", "reason": "processing_evidence_required", "required_action": "attach_processing_evidence"}},
	{"name": "pattern_batch_required", "condition": {"operation": "record_pattern", "batch_present": False}, "effect": {"decision": "deny", "reason": "processing_batch_required", "required_action": "select_processing_batch"}},
	{"name": "pattern_type_supported", "condition": {"operation": "record_pattern", "pattern_type_supported": False}, "effect": {"decision": "deny", "reason": "pattern_type_not_supported", "required_action": "select_supported_pattern_type"}},
	{"name": "pattern_confidence_valid", "condition": {"operation": "record_pattern", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "pattern_analyst_required", "condition": {"operation": "record_pattern", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "pattern_evidence_required", "condition": {"operation": "record_pattern", "evidence_present": False}, "effect": {"decision": "deny", "reason": "pattern_evidence_required", "required_action": "attach_pattern_evidence"}},
	{"name": "assessment_pattern_required", "condition": {"operation": "record_assessment", "pattern_present": False}, "effect": {"decision": "deny", "reason": "pattern_required", "required_action": "select_pattern"}},
	{"name": "assessment_type_supported", "condition": {"operation": "record_assessment", "assessment_type_supported": False}, "effect": {"decision": "deny", "reason": "assessment_type_not_supported", "required_action": "select_supported_assessment_type"}},
	{"name": "assessment_classification_supported", "condition": {"operation": "record_assessment", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "assessment_analyst_required", "condition": {"operation": "record_assessment", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "assessment_evidence_required", "condition": {"operation": "record_assessment", "evidence_present": False}, "effect": {"decision": "deny", "reason": "assessment_evidence_required", "required_action": "attach_assessment_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "sigint_batch_requires_bytewax", "condition": {"operation": "sigint_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_sigint_batch_to_bytewax"}},
	{"name": "sigint_agent_runtime_supported", "condition": {"operation": "register_sigint_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "sigint_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "sigint_agent_role_supported", "condition": {"operation": "register_sigint_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "sigint_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_sigint_agent_action_requires_human_approval", "condition": {"operation": "sigint_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "cross_tenant_sigint_action_denied", "condition": {"operation": "sigint_agent_action", "cross_tenant_sigint_scope": True}, "effect": {"decision": "deny", "reason": "cross_tenant_sigint_scope_denied", "required_action": "remove_cross_tenant_scope"}},
	{"name": "privilege_escalation_action_denied", "condition": {"operation": "sigint_agent_action", "privilege_escalation_scope": True}, "effect": {"decision": "deny", "reason": "privilege_escalation_scope_denied", "required_action": "remove_privilege_escalation_scope"}},
	{"name": "unauthorized_content_intercept_action_denied", "condition": {"operation": "sigint_agent_action", "unauthorized_content_intercept_scope": True}, "effect": {"decision": "deny", "reason": "unauthorized_content_intercept_scope_denied", "required_action": "remove_intercept_scope"}},
	{"name": "minimization_bypass_action_denied", "condition": {"operation": "sigint_agent_action", "minimization_bypass_scope": True}, "effect": {"decision": "deny", "reason": "minimization_bypass_scope_denied", "required_action": "remove_minimization_bypass_scope"}},
	{"name": "unapproved_bulk_collection_action_denied", "condition": {"operation": "sigint_agent_action", "unapproved_bulk_collection_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_bulk_collection_scope_denied", "required_action": "remove_bulk_collection_scope"}},
	{"name": "offensive_sigint_action_denied", "condition": {"operation": "sigint_agent_action", "offensive_sigint_scope": True}, "effect": {"decision": "deny", "reason": "offensive_sigint_scope_denied", "required_action": "remove_offensive_sigint_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-sigint/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
