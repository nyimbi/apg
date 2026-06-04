"""Executable capability contract for APG Open Source Intelligence (OSINT).

This module is the single authoritative source for:
  - supported enumeration values (used by both service and rules)
  - default configuration schema
  - UI route definitions
  - theme tokens
  - streaming configuration
  - deterministic rule engine
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_osint"
CAPABILITY_NAME = "Open Source Intelligence"
CAPABILITY_VERSION = "2.0.0"
OSINT_EVENT_STREAM = "apg.intel.osint.lifecycle"

# ---------------------------------------------------------------------------
# Supported enumeration values (mirrors models.py enums — kept as plain lists
# so this module has zero imports and can be used standalone).
# ---------------------------------------------------------------------------

SUPPORTED_SOURCE_TYPES = [
	"web", "social_media", "darkweb", "news", "forum", "document",
	"registry", "broadcast", "dataset", "api_feed", "rss_feed",
	"paste_site", "code_repository", "iot_scan",
]
SUPPORTED_SOURCE_STATUSES = ["active", "inactive", "suspended", "under_review", "decommissioned"]
SUPPORTED_RISK_TIERS = ["low", "medium", "high", "critical"]
SUPPORTED_TASK_TYPES = [
	"web_scrape", "social_monitor", "domain_intel", "ip_geolocation",
	"entity_extraction", "document_analysis", "relationship_mapping",
	"dark_web_crawl", "api_collection", "deduplication", "credibility_score",
]
SUPPORTED_TASK_STATUSES = ["pending", "running", "completed", "failed", "cancelled", "retrying"]
SUPPORTED_COLLECTION_METHODS = [
	"crawler", "api_feed", "rss_feed", "manual_upload",
	"partner_feed", "webhook", "headless_browser",
]
SUPPORTED_INTEL_STATUSES = [
	"raw", "triaged", "processed", "verified", "rejected", "archived", "disseminated",
]
SUPPORTED_ENTITY_TYPES = [
	"person", "organization", "location", "object", "event", "facility",
	"vessel", "aircraft", "vehicle", "domain", "ip_address", "email",
	"phone", "cryptocurrency_wallet", "username",
]
SUPPORTED_RELATIONSHIP_TYPES = [
	"affiliated_with", "owns", "operates", "located_at", "communicates_with",
	"member_of", "funds", "directs", "known_alias", "employs",
	"associated_with", "targets", "linked_to",
]
SUPPORTED_CONFIDENCE_LEVELS = ["unconfirmed", "possible", "probable", "confirmed"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_TLP = ["clear", "green", "amber", "amber_strict", "red"]
SUPPORTED_PRIORITIES = ["low", "medium", "high", "critical"]
SUPPORTED_TRIAGE_DECISIONS = ["relevant", "irrelevant", "duplicate", "needs_review", "escalated"]
SUPPORTED_ASSESSMENT_TYPES = [
	"threat", "opportunity", "entity_profile", "event_summary",
	"trend", "watchlist", "network_map", "geospatial",
]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = [
	"source_scout", "collection_planner", "evidence_triage",
	"entity_extractor", "relationship_mapper", "deduplicator",
	"credibility_analyst", "dissemination_reviewer", "watchlist_monitor",
]

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"sources": {
		"supported_types": SUPPORTED_SOURCE_TYPES,
		"supported_risk_tiers": SUPPORTED_RISK_TIERS,
		"supported_collection_methods": SUPPORTED_COLLECTION_METHODS,
		"owner_required": True,
		"terms_review_required": True,
		"evidence_required": True,
	},
	"tasks": {
		"supported_types": SUPPORTED_TASK_TYPES,
		"supported_statuses": SUPPORTED_TASK_STATUSES,
		"approval_required_for_high_risk": True,
		"max_depth_default": 2,
		"max_depth_limit": 10,
	},
	"raw_intelligence": {
		"fingerprint_required": True,
		"content_reference_required": True,
		"confidence_required": True,
		"deduplication_enabled": True,
	},
	"processed_intelligence": {
		"supported_assessment_types": SUPPORTED_ASSESSMENT_TYPES,
		"supported_classifications": SUPPORTED_CLASSIFICATIONS,
		"supported_tlp": SUPPORTED_TLP,
		"analyst_required": True,
		"evidence_required": True,
	},
	"entities": {
		"supported_types": SUPPORTED_ENTITY_TYPES,
		"supported_confidence_levels": SUPPORTED_CONFIDENCE_LEVELS,
		"supported_classifications": SUPPORTED_CLASSIFICATIONS,
		"evidence_required": True,
	},
	"relationships": {
		"supported_types": SUPPORTED_RELATIONSHIP_TYPES,
		"self_loop_denied": True,
		"evidence_required": True,
	},
	"social_profiles": {
		"entity_link_optional": True,
		"evidence_required": True,
	},
	"web_content": {
		"max_crawl_depth": 10,
		"content_hash_required": True,
	},
	"domain_records": {
		"evidence_required": True,
	},
	"ip_intelligence": {
		"evidence_required": True,
	},
	"document_analysis": {
		"evidence_required": True,
	},
	"dissemination": {
		"supported_tlp": SUPPORTED_TLP,
		"supported_classifications": SUPPORTED_CLASSIFICATIONS,
		"approval_required": True,
		"autonomous_dissemination_denied": True,
		"evidence_required": True,
	},
	"reviews": {
		"supported_statuses": SUPPORTED_REVIEW_STATUSES,
		"reviewer_required": True,
		"evidence_required": True,
	},
	"agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AGENT_ROLES,
		"human_approval_required_for_privileged_actions": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"respect_source_terms": True,
		"cross_tenant_osint_denied": True,
		"privilege_escalation_denied": True,
		"evidence_fabrication_denied": True,
		"source_terms_violation_denied": True,
		"autonomous_dissemination_denied": True,
		"unapproved_high_risk_collection_denied": True,
	},
	"observability": {
		"event_stream": OSINT_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"nlp": "nlpc",
		"crawler": "intel_crawler",
		"search": "srch",
		"graph": "grph",
		"rag": "ragn",
		"event_stream": "bytewax",
		"geo": "geoi",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_sources": True,
		"enable_tasks": True,
		"enable_raw_intel": True,
		"enable_processed_intel": True,
		"enable_entities": True,
		"enable_relationships": True,
		"enable_social_profiles": True,
		"enable_web_content": True,
		"enable_domain_records": True,
		"enable_ip_intelligence": True,
		"enable_document_analysis": True,
		"enable_dissemination": True,
		"enable_reviews": True,
		"enable_agents": True,
	},
	"theme": {
		"default_theme": "intel_osint_control",
		"allow_tenant_overrides": True,
	},
}

PROVIDES = [
	"osint_source_workflow",
	"osint_collection_task_workflow",
	"osint_raw_intel_workflow",
	"osint_processed_intel_workflow",
	"osint_entity_workflow",
	"osint_relationship_workflow",
	"osint_social_profile_workflow",
	"osint_web_content_workflow",
	"osint_domain_intel_workflow",
	"osint_ip_intel_workflow",
	"osint_document_analysis_workflow",
	"osint_dissemination_workflow",
	"osint_review_workflow",
	"osint_agent_workflow",
]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "intel_crawler", "srch", "grph", "ragn", "geoi"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-osint/dashboard", "component": "OSINTDashboard", "permission": "intel_osint:view", "nav_group": "Overview"},
	{"name": "sources", "path": "/intel-osint/sources", "component": "OSINTSourceRegistry", "permission": "intel_osint:sources", "nav_group": "Collection"},
	{"name": "tasks", "path": "/intel-osint/tasks", "component": "OSINTTaskConsole", "permission": "intel_osint:tasks", "nav_group": "Collection"},
	{"name": "raw_intel", "path": "/intel-osint/raw-intel", "component": "OSINTRawIntelLedger", "permission": "intel_osint:raw_intel", "nav_group": "Processing"},
	{"name": "triage", "path": "/intel-osint/triage", "component": "OSINTTriageWorkbench", "permission": "intel_osint:triage", "nav_group": "Processing"},
	{"name": "processed_intel", "path": "/intel-osint/processed-intel", "component": "OSINTProcessedIntelWorkbench", "permission": "intel_osint:processed_intel", "nav_group": "Analysis"},
	{"name": "entities", "path": "/intel-osint/entities", "component": "OSINTEntityGraph", "permission": "intel_osint:entities", "nav_group": "Analysis"},
	{"name": "relationships", "path": "/intel-osint/relationships", "component": "OSINTRelationshipMap", "permission": "intel_osint:relationships", "nav_group": "Analysis"},
	{"name": "social_profiles", "path": "/intel-osint/social-profiles", "component": "OSINTSocialProfileConsole", "permission": "intel_osint:social", "nav_group": "Collection"},
	{"name": "web_content", "path": "/intel-osint/web-content", "component": "OSINTWebContentLedger", "permission": "intel_osint:web", "nav_group": "Collection"},
	{"name": "domain_records", "path": "/intel-osint/domain-records", "component": "OSINTDomainIntelConsole", "permission": "intel_osint:domains", "nav_group": "Technical"},
	{"name": "ip_intelligence", "path": "/intel-osint/ip-intelligence", "component": "OSINTIPIntelConsole", "permission": "intel_osint:ips", "nav_group": "Technical"},
	{"name": "document_analysis", "path": "/intel-osint/document-analysis", "component": "OSINTDocumentAnalysisConsole", "permission": "intel_osint:documents", "nav_group": "Analysis"},
	{"name": "dissemination", "path": "/intel-osint/dissemination", "component": "OSINTDisseminationConsole", "permission": "intel_osint:disseminate", "nav_group": "Delivery"},
	{"name": "reviews", "path": "/intel-osint/reviews", "component": "OSINTReviewConsole", "permission": "intel_osint:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-osint/agents", "component": "OSINTAgentWorkbench", "permission": "intel_osint:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-osint/settings", "component": "OSINTSettings", "permission": "intel_osint:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_osint_control",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#C44536",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"sources": {"icon": "radar", "status_indicator": "source-risk-chip"},
		"tasks": {"icon": "activity", "status_indicator": "task-status-chip"},
		"raw_intel": {"icon": "database", "status_indicator": "intel-status-chip"},
		"processed_intel": {"icon": "brain-circuit", "status_indicator": "assessment-chip"},
		"entities": {"icon": "users", "status_indicator": "entity-type-chip"},
		"relationships": {"icon": "network", "status_indicator": "relationship-chip"},
		"social_profiles": {"icon": "share-2", "status_indicator": "platform-chip"},
		"web_content": {"icon": "globe", "status_indicator": "mime-chip"},
		"domain_records": {"icon": "server", "status_indicator": "risk-chip"},
		"ip_intelligence": {"icon": "map-pin", "status_indicator": "threat-chip"},
		"document_analysis": {"icon": "file-text", "status_indicator": "sentiment-chip"},
		"dissemination": {"icon": "send", "status_indicator": "release-chip"},
		"reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": OSINT_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"osint_source_registered",
		"osint_source_updated",
		"osint_task_created",
		"osint_task_status_changed",
		"osint_raw_intel_ingested",
		"osint_raw_intel_triaged",
		"osint_processed_intel_created",
		"osint_entity_extracted",
		"osint_entity_updated",
		"osint_relationship_mapped",
		"osint_social_profile_registered",
		"osint_web_content_scraped",
		"osint_domain_record_created",
		"osint_ip_intel_created",
		"osint_document_analysis_completed",
		"osint_credibility_scored",
		"osint_dissemination_package_created",
		"osint_review_recorded",
		"osint_agent_registered",
		"osint_deduplication_completed",
	],
	"guardrails": [
		"osint_batch_requires_bytewax",
		"privileged_osint_agent_action_requires_human_approval",
		"cross_tenant_osint_action_denied",
		"privilege_escalation_action_denied",
		"evidence_fabrication_action_denied",
		"source_terms_violation_action_denied",
		"autonomous_dissemination_action_denied",
		"unapproved_high_risk_collection_action_denied",
	],
}

# ---------------------------------------------------------------------------
# Deterministic rule engine
# ---------------------------------------------------------------------------

RULES: list[dict[str, Any]] = [
	# Tenant governance
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "osint_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "osint_policy_required", "required_action": "attach_osint_policy"}},
	{"name": "cross_tenant_osint_write_denied", "condition": {"operation_type": "write", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_osint_write_denied", "required_action": "remove_cross_tenant_scope"}},

	# Source registration
	{"name": "source_type_supported", "condition": {"operation": "register_source", "source_type_supported": False}, "effect": {"decision": "deny", "reason": "source_type_not_supported", "required_action": "select_supported_source_type"}},
	{"name": "source_name_required", "condition": {"operation": "register_source", "name_present": False}, "effect": {"decision": "deny", "reason": "source_name_required", "required_action": "provide_source_name"}},
	{"name": "source_owner_required", "condition": {"operation": "register_source", "owner_present": False}, "effect": {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}},
	{"name": "source_terms_review_required", "condition": {"operation": "register_source", "terms_review_present": False}, "effect": {"decision": "deny", "reason": "terms_review_required", "required_action": "complete_terms_review"}},
	{"name": "source_risk_tier_supported", "condition": {"operation": "register_source", "risk_tier_supported": False}, "effect": {"decision": "deny", "reason": "risk_tier_not_supported", "required_action": "select_supported_risk_tier"}},
	{"name": "source_evidence_required", "condition": {"operation": "register_source", "evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},

	# Collection task
	{"name": "task_source_required", "condition": {"operation": "create_task", "source_present": False}, "effect": {"decision": "deny", "reason": "source_required", "required_action": "select_source"}},
	{"name": "task_type_supported", "condition": {"operation": "create_task", "task_type_supported": False}, "effect": {"decision": "deny", "reason": "task_type_not_supported", "required_action": "select_supported_task_type"}},
	{"name": "high_risk_task_requires_approval", "condition": {"operation": "create_task", "high_risk_source": True, "approval_present": False}, "effect": {"decision": "deny", "reason": "high_risk_collection_requires_approval", "required_action": "obtain_collection_approval"}},
	{"name": "task_evidence_required", "condition": {"operation": "create_task", "evidence_present": False}, "effect": {"decision": "deny", "reason": "task_evidence_required", "required_action": "attach_task_evidence"}},

	# Raw intelligence ingestion
	{"name": "raw_intel_task_required", "condition": {"operation": "ingest_raw_intel", "task_present": False}, "effect": {"decision": "deny", "reason": "collection_task_required", "required_action": "select_task"}},
	{"name": "raw_intel_fingerprint_required", "condition": {"operation": "ingest_raw_intel", "fingerprint_present": False}, "effect": {"decision": "deny", "reason": "content_fingerprint_required", "required_action": "compute_fingerprint"}},
	{"name": "raw_intel_confidence_valid", "condition": {"operation": "ingest_raw_intel", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "raw_intel_evidence_required", "condition": {"operation": "ingest_raw_intel", "evidence_present": False}, "effect": {"decision": "deny", "reason": "raw_intel_evidence_required", "required_action": "attach_evidence"}},

	# Processed intelligence
	{"name": "processed_intel_raw_required", "condition": {"operation": "create_processed_intel", "raw_intel_present": False}, "effect": {"decision": "deny", "reason": "raw_intel_required", "required_action": "select_raw_intel"}},
	{"name": "processed_intel_assessment_type_supported", "condition": {"operation": "create_processed_intel", "assessment_type_supported": False}, "effect": {"decision": "deny", "reason": "assessment_type_not_supported", "required_action": "select_supported_assessment_type"}},
	{"name": "processed_intel_analyst_required", "condition": {"operation": "create_processed_intel", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "processed_intel_confidence_valid", "condition": {"operation": "create_processed_intel", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "processed_intel_evidence_required", "condition": {"operation": "create_processed_intel", "evidence_present": False}, "effect": {"decision": "deny", "reason": "processed_intel_evidence_required", "required_action": "attach_evidence"}},

	# Entity extraction
	{"name": "entity_type_supported", "condition": {"operation": "extract_entity", "entity_type_supported": False}, "effect": {"decision": "deny", "reason": "entity_type_not_supported", "required_action": "select_supported_entity_type"}},
	{"name": "entity_name_required", "condition": {"operation": "extract_entity", "name_present": False}, "effect": {"decision": "deny", "reason": "entity_name_required", "required_action": "provide_entity_name"}},
	{"name": "entity_confidence_valid", "condition": {"operation": "extract_entity", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "entity_evidence_required", "condition": {"operation": "extract_entity", "evidence_present": False}, "effect": {"decision": "deny", "reason": "entity_evidence_required", "required_action": "attach_evidence"}},

	# Relationship mapping
	{"name": "relationship_type_supported", "condition": {"operation": "map_relationship", "relationship_type_supported": False}, "effect": {"decision": "deny", "reason": "relationship_type_not_supported", "required_action": "select_supported_relationship_type"}},
	{"name": "relationship_source_required", "condition": {"operation": "map_relationship", "source_entity_present": False}, "effect": {"decision": "deny", "reason": "source_entity_required", "required_action": "select_source_entity"}},
	{"name": "relationship_target_required", "condition": {"operation": "map_relationship", "target_entity_present": False}, "effect": {"decision": "deny", "reason": "target_entity_required", "required_action": "select_target_entity"}},
	{"name": "relationship_self_loop_denied", "condition": {"operation": "map_relationship", "self_loop": True}, "effect": {"decision": "deny", "reason": "relationship_self_loop_denied", "required_action": "select_distinct_entities"}},
	{"name": "relationship_confidence_valid", "condition": {"operation": "map_relationship", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "relationship_evidence_required", "condition": {"operation": "map_relationship", "evidence_present": False}, "effect": {"decision": "deny", "reason": "relationship_evidence_required", "required_action": "attach_evidence"}},

	# Dissemination
	{"name": "dissemination_intel_required", "condition": {"operation": "create_dissemination", "intel_present": False}, "effect": {"decision": "deny", "reason": "processed_intel_required", "required_action": "select_processed_intel"}},
	{"name": "dissemination_approval_required", "condition": {"operation": "create_dissemination", "approval_present": False}, "effect": {"decision": "deny", "reason": "dissemination_approval_required", "required_action": "obtain_approval"}},
	{"name": "dissemination_audience_required", "condition": {"operation": "create_dissemination", "audience_present": False}, "effect": {"decision": "deny", "reason": "audience_required", "required_action": "select_audience"}},
	{"name": "dissemination_evidence_required", "condition": {"operation": "create_dissemination", "evidence_present": False}, "effect": {"decision": "deny", "reason": "dissemination_evidence_required", "required_action": "attach_evidence"}},
	{"name": "autonomous_dissemination_denied", "condition": {"operation": "create_dissemination", "autonomous_dissemination": True}, "effect": {"decision": "deny", "reason": "autonomous_dissemination_denied", "required_action": "require_human_sign_off"}},

	# Review
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_evidence"}},

	# Agent governance
	{"name": "osint_batch_requires_bytewax", "condition": {"operation": "osint_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_osint_batch_to_bytewax"}},
	{"name": "osint_agent_runtime_supported", "condition": {"operation": "register_osint_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "osint_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "osint_agent_role_supported", "condition": {"operation": "register_osint_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "osint_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_osint_agent_action_requires_human_approval", "condition": {"operation": "osint_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "cross_tenant_osint_action_denied", "condition": {"operation": "osint_agent_action", "cross_tenant_osint_scope": True}, "effect": {"decision": "deny", "reason": "cross_tenant_osint_scope_denied", "required_action": "remove_cross_tenant_scope"}},
	{"name": "privilege_escalation_action_denied", "condition": {"operation": "osint_agent_action", "privilege_escalation_scope": True}, "effect": {"decision": "deny", "reason": "privilege_escalation_scope_denied", "required_action": "remove_privilege_escalation_scope"}},
	{"name": "evidence_fabrication_action_denied", "condition": {"operation": "osint_agent_action", "evidence_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "evidence_fabrication_scope_denied", "required_action": "remove_evidence_fabrication_scope"}},
	{"name": "source_terms_violation_action_denied", "condition": {"operation": "osint_agent_action", "source_terms_violation_scope": True}, "effect": {"decision": "deny", "reason": "source_terms_violation_scope_denied", "required_action": "remove_terms_violation_scope"}},
	{"name": "unapproved_high_risk_collection_action_denied", "condition": {"operation": "osint_agent_action", "unapproved_high_risk_collection_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_high_risk_collection_scope_denied", "required_action": "obtain_collection_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	"""Return a deep-copied, tenant-scoped capability contract."""
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
			"properties": (
				{key: {"type": "object"} for key in configuration if key != "tenant_id"}
				| {"tenant_id": {"type": "string", "minLength": 1}}
			),
		},
		"rule_engine": {
			"type": "deterministic",
			"default_decision": "allow",
			"rules": deepcopy(RULES),
		},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/intel-osint/api/v1",
			"requires_theme": True,
			"view_module": "views.py",
			"template_roots": ["templates/", "static/"],
			"routes": deepcopy(UI_ROUTES),
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate all rules against context; return allow/deny decision."""
	actions: list[dict[str, Any]] = []
	for rule in RULES:
		if _matches(rule["condition"], context):
			actions.append(rule["effect"] | {"rule": rule["name"]})
	if not actions:
		return {"decision": "allow", "actions": [], "context": dict(context)}
	return {"decision": "deny", "actions": actions, "context": dict(context)}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	"""True if all condition predicates match the context."""
	for key, expected in condition.items():
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True
