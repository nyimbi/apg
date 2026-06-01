"""Executable capability contract for APG Open Source Intelligence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_osint"
CAPABILITY_NAME = "Open Source Intelligence"
CAPABILITY_VERSION = "1.1.0"
OSINT_EVENT_STREAM = "apg.intel.osint.lifecycle"

SUPPORTED_PRIORITIES = ["low", "medium", "high", "critical"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_SOURCE_TYPES = ["news", "web", "social", "registry", "document", "forum", "broadcast", "dataset"]
SUPPORTED_RISK_TIERS = ["low", "medium", "high", "critical"]
SUPPORTED_COLLECTION_METHODS = ["manual_review", "crawler", "api_feed", "rss_feed", "upload", "partner_feed"]
SUPPORTED_TRIAGE_DECISIONS = ["relevant", "irrelevant", "duplicate", "needs_review", "escalated"]
SUPPORTED_ASSESSMENT_TYPES = ["threat", "opportunity", "entity_profile", "event_summary", "trend", "watchlist"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["source_scout", "collection_planner", "evidence_triage_agent", "assessment_drafter", "watchlist_monitor", "dissemination_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"requirements": {"supported_priorities": SUPPORTED_PRIORITIES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "requester_required": True, "evidence_required": True},
	"sources": {"supported_source_types": SUPPORTED_SOURCE_TYPES, "supported_risk_tiers": SUPPORTED_RISK_TIERS, "owner_required": True, "terms_review_required": True, "evidence_required": True},
	"collection_plans": {"supported_methods": SUPPORTED_COLLECTION_METHODS, "requirement_required": True, "source_required": True, "cadence_required": True, "approval_required_for_high_risk": True, "evidence_required": True},
	"evidence": {"content_reference_required": True, "fingerprint_required": True, "confidence_score_required": True, "evidence_required": True},
	"triage": {"supported_decisions": SUPPORTED_TRIAGE_DECISIONS, "analyst_required": True, "evidence_required": True},
	"assessments": {"supported_types": SUPPORTED_ASSESSMENT_TYPES, "confidence_score_required": True, "analyst_required": True, "evidence_required": True},
	"dissemination": {"audience_required": True, "release_marking_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "respect_source_terms": True},
	"observability": {"event_stream": OSINT_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "crawler": "intel_crawler", "search": "srch", "graph": "grph", "rag": "ragn", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_requirements": True, "enable_sources": True, "enable_collection_plans": True, "enable_evidence": True, "enable_triage": True, "enable_assessments": True, "enable_dissemination": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_osint_control", "allow_tenant_overrides": True},
}

PROVIDES = ["osint_requirement_workflow", "osint_source_workflow", "osint_collection_plan_workflow", "osint_evidence_workflow", "osint_triage_workflow", "osint_assessment_workflow", "osint_dissemination_workflow", "osint_review_workflow", "osint_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "intel_crawler", "srch", "grph", "ragn"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-osint/dashboard", "component": "OSINTDashboard", "permission": "intel_osint:view", "nav_group": "Overview"},
	{"name": "requirements", "path": "/intel-osint/requirements", "component": "OSINTRequirementConsole", "permission": "intel_osint:requirements", "nav_group": "Planning"},
	{"name": "sources", "path": "/intel-osint/sources", "component": "OSINTSourceRegistry", "permission": "intel_osint:sources", "nav_group": "Collection"},
	{"name": "collection_plans", "path": "/intel-osint/collection-plans", "component": "OSINTCollectionPlanner", "permission": "intel_osint:collection", "nav_group": "Collection"},
	{"name": "evidence", "path": "/intel-osint/evidence", "component": "OSINTEvidenceLedger", "permission": "intel_osint:evidence", "nav_group": "Processing"},
	{"name": "triage", "path": "/intel-osint/triage", "component": "OSINTTriageWorkbench", "permission": "intel_osint:triage", "nav_group": "Processing"},
	{"name": "assessments", "path": "/intel-osint/assessments", "component": "OSINTAssessmentWorkbench", "permission": "intel_osint:assessments", "nav_group": "Analysis"},
	{"name": "dissemination", "path": "/intel-osint/dissemination", "component": "OSINTDisseminationConsole", "permission": "intel_osint:disseminate", "nav_group": "Delivery"},
	{"name": "reviews", "path": "/intel-osint/reviews", "component": "OSINTReviewConsole", "permission": "intel_osint:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-osint/agents", "component": "OSINTAgentWorkbench", "permission": "intel_osint:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-osint/settings", "component": "OSINTSettings", "permission": "intel_osint:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_osint_control",
	"tokens": {"color.primary": "#28536B", "color.accent": "#C44536", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"requirements": {"icon": "clipboard-list", "status_indicator": "priority-chip"}, "sources": {"icon": "radar", "status_indicator": "source-risk-chip"}, "collection_plans": {"icon": "route", "status_indicator": "method-chip"}, "evidence": {"icon": "fingerprint", "status_indicator": "confidence-chip"}, "triage": {"icon": "list-checks", "status_indicator": "decision-chip"}, "assessments": {"icon": "brain-circuit", "status_indicator": "assessment-chip"}, "dissemination": {"icon": "send", "status_indicator": "release-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": OSINT_EVENT_STREAM, "key": "tenant_id", "events": ["osint_requirement_registered", "osint_source_registered", "osint_collection_plan_recorded", "osint_evidence_recorded", "osint_triage_recorded", "osint_assessment_recorded", "osint_dissemination_recorded", "osint_review_recorded", "osint_agent_registered"], "guardrails": ["osint_batch_requires_bytewax", "privileged_osint_agent_action_requires_human_approval"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "osint_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "osint_policy_required", "required_action": "attach_osint_policy"}},
	{"name": "requirement_topic_required", "condition": {"operation": "register_requirement", "topic_present": False}, "effect": {"decision": "deny", "reason": "requirement_topic_required", "required_action": "attach_topic"}},
	{"name": "requirement_priority_supported", "condition": {"operation": "register_requirement", "priority_supported": False}, "effect": {"decision": "deny", "reason": "priority_not_supported", "required_action": "select_supported_priority"}},
	{"name": "requirement_requester_required", "condition": {"operation": "register_requirement", "requester_present": False}, "effect": {"decision": "deny", "reason": "requester_required", "required_action": "record_requester"}},
	{"name": "requirement_classification_supported", "condition": {"operation": "register_requirement", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "requirement_evidence_required", "condition": {"operation": "register_requirement", "evidence_present": False}, "effect": {"decision": "deny", "reason": "requirement_evidence_required", "required_action": "attach_requirement_evidence"}},
	{"name": "source_type_supported", "condition": {"operation": "register_source", "source_type_supported": False}, "effect": {"decision": "deny", "reason": "source_type_not_supported", "required_action": "select_supported_source_type"}},
	{"name": "source_reference_required", "condition": {"operation": "register_source", "source_reference_present": False}, "effect": {"decision": "deny", "reason": "source_reference_required", "required_action": "attach_source_reference"}},
	{"name": "source_owner_required", "condition": {"operation": "register_source", "owner_present": False}, "effect": {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}},
	{"name": "source_terms_review_required", "condition": {"operation": "register_source", "terms_review_present": False}, "effect": {"decision": "deny", "reason": "terms_review_required", "required_action": "record_terms_review"}},
	{"name": "source_risk_supported", "condition": {"operation": "register_source", "risk_tier_supported": False}, "effect": {"decision": "deny", "reason": "risk_tier_not_supported", "required_action": "select_supported_risk_tier"}},
	{"name": "source_evidence_required", "condition": {"operation": "register_source", "evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "plan_requirement_required", "condition": {"operation": "record_collection_plan", "requirement_present": False}, "effect": {"decision": "deny", "reason": "requirement_required", "required_action": "select_requirement"}},
	{"name": "plan_source_required", "condition": {"operation": "record_collection_plan", "source_present": False}, "effect": {"decision": "deny", "reason": "source_required", "required_action": "select_source"}},
	{"name": "plan_method_supported", "condition": {"operation": "record_collection_plan", "method_supported": False}, "effect": {"decision": "deny", "reason": "collection_method_not_supported", "required_action": "select_supported_collection_method"}},
	{"name": "plan_cadence_required", "condition": {"operation": "record_collection_plan", "cadence_present": False}, "effect": {"decision": "deny", "reason": "cadence_required", "required_action": "set_cadence"}},
	{"name": "high_risk_plan_requires_approval", "condition": {"operation": "record_collection_plan", "high_risk_source": True, "approval_present": False}, "effect": {"decision": "deny", "reason": "collection_approval_required", "required_action": "attach_approval"}},
	{"name": "plan_evidence_required", "condition": {"operation": "record_collection_plan", "evidence_present": False}, "effect": {"decision": "deny", "reason": "collection_plan_evidence_required", "required_action": "attach_plan_evidence"}},
	{"name": "evidence_plan_required", "condition": {"operation": "record_evidence", "plan_present": False}, "effect": {"decision": "deny", "reason": "collection_plan_required", "required_action": "select_collection_plan"}},
	{"name": "evidence_content_required", "condition": {"operation": "record_evidence", "content_present": False}, "effect": {"decision": "deny", "reason": "content_reference_required", "required_action": "attach_content_reference"}},
	{"name": "evidence_fingerprint_required", "condition": {"operation": "record_evidence", "fingerprint_present": False}, "effect": {"decision": "deny", "reason": "fingerprint_required", "required_action": "attach_fingerprint"}},
	{"name": "evidence_confidence_valid", "condition": {"operation": "record_evidence", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "evidence_record_evidence_required", "condition": {"operation": "record_evidence", "evidence_reference_present": False}, "effect": {"decision": "deny", "reason": "evidence_reference_required", "required_action": "attach_evidence_reference"}},
	{"name": "triage_evidence_required", "condition": {"operation": "record_triage", "evidence_present": False}, "effect": {"decision": "deny", "reason": "evidence_required", "required_action": "select_evidence"}},
	{"name": "triage_decision_supported", "condition": {"operation": "record_triage", "decision_supported": False}, "effect": {"decision": "deny", "reason": "triage_decision_not_supported", "required_action": "select_supported_decision"}},
	{"name": "triage_analyst_required", "condition": {"operation": "record_triage", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "triage_evidence_reference_required", "condition": {"operation": "record_triage", "evidence_reference_present": False}, "effect": {"decision": "deny", "reason": "triage_evidence_required", "required_action": "attach_triage_evidence"}},
	{"name": "assessment_requirement_required", "condition": {"operation": "record_assessment", "requirement_present": False}, "effect": {"decision": "deny", "reason": "requirement_required", "required_action": "select_requirement"}},
	{"name": "assessment_type_supported", "condition": {"operation": "record_assessment", "assessment_type_supported": False}, "effect": {"decision": "deny", "reason": "assessment_type_not_supported", "required_action": "select_supported_assessment_type"}},
	{"name": "assessment_confidence_valid", "condition": {"operation": "record_assessment", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "assessment_analyst_required", "condition": {"operation": "record_assessment", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "assessment_evidence_required", "condition": {"operation": "record_assessment", "evidence_present": False}, "effect": {"decision": "deny", "reason": "assessment_evidence_required", "required_action": "attach_assessment_evidence"}},
	{"name": "dissemination_assessment_required", "condition": {"operation": "record_dissemination", "assessment_present": False}, "effect": {"decision": "deny", "reason": "assessment_required", "required_action": "select_assessment"}},
	{"name": "dissemination_audience_required", "condition": {"operation": "record_dissemination", "audience_present": False}, "effect": {"decision": "deny", "reason": "audience_required", "required_action": "select_audience"}},
	{"name": "dissemination_release_marking_required", "condition": {"operation": "record_dissemination", "release_marking_present": False}, "effect": {"decision": "deny", "reason": "release_marking_required", "required_action": "set_release_marking"}},
	{"name": "dissemination_approval_required", "condition": {"operation": "record_dissemination", "approval_present": False}, "effect": {"decision": "deny", "reason": "dissemination_approval_required", "required_action": "attach_approval"}},
	{"name": "dissemination_evidence_required", "condition": {"operation": "record_dissemination", "evidence_present": False}, "effect": {"decision": "deny", "reason": "dissemination_evidence_required", "required_action": "attach_dissemination_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "osint_batch_requires_bytewax", "condition": {"operation": "osint_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_osint_batch_to_bytewax"}},
	{"name": "osint_agent_runtime_supported", "condition": {"operation": "register_osint_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "osint_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "osint_agent_role_supported", "condition": {"operation": "register_osint_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "osint_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_osint_agent_action_requires_human_approval", "condition": {"operation": "osint_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-osint/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
