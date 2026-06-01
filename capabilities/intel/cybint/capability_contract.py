"""Executable capability contract for APG Cyber Intelligence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_cybint"
CAPABILITY_NAME = "Cyber Intelligence"
CAPABILITY_VERSION = "1.1.0"
CYBINT_EVENT_STREAM = "apg.intel.cybint.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["mission_order", "consent", "partner_authority", "legal_mandate", "defensive_operations_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_INDICATOR_TYPES = ["domain", "ip_address", "url", "file_hash", "email", "certificate", "mutex", "registry_key", "user_agent", "ttp"]
SUPPORTED_TLP = ["clear", "green", "amber", "amber_strict", "red"]
SUPPORTED_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_ENRICHMENT_TYPES = ["reputation", "whois", "passive_dns", "malware_family", "campaign", "vulnerability_context", "asset_context"]
SUPPORTED_PROFILE_TYPES = ["threat_actor", "campaign", "malware_family", "intrusion_set", "vulnerability", "infrastructure_cluster"]
SUPPORTED_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_RESPONSE_PRIORITIES = ["monitor", "triage", "contain", "eradicate", "executive_review"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["authority_reviewer", "indicator_triage", "enrichment_analyst", "threat_profiler", "risk_analyst", "dissemination_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"indicators": {"supported_indicator_types": SUPPORTED_INDICATOR_TYPES, "supported_tlp": SUPPORTED_TLP, "authority_required": True, "confidence_required": True, "evidence_required": True},
	"sightings": {"supported_severities": SUPPORTED_SEVERITIES, "indicator_required": True, "source_reference_required": True, "observed_at_required": True, "evidence_required": True},
	"enrichment": {"supported_types": SUPPORTED_ENRICHMENT_TYPES, "indicator_required": True, "provider_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"profiles": {"supported_types": SUPPORTED_PROFILE_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"risk": {"supported_levels": SUPPORTED_RISK_LEVELS, "indicator_required": True, "profile_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"incidents": {"supported_priorities": SUPPORTED_RESPONSE_PRIORITIES, "assessment_required": True, "incident_reference_required": True, "owner_required": True, "evidence_required": True},
	"dissemination": {"assessment_required": True, "audience_required": True, "release_marking_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "defensive_use_only": True, "exploit_or_offensive_action_denied": True},
	"observability": {"event_stream": CYBINT_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "graph": "grph", "rag": "ragn", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_indicators": True, "enable_sightings": True, "enable_enrichment": True, "enable_profiles": True, "enable_risk": True, "enable_incidents": True, "enable_dissemination": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_cybint_control", "allow_tenant_overrides": True},
}

PROVIDES = ["cybint_authority_workflow", "cybint_indicator_workflow", "cybint_sighting_workflow", "cybint_enrichment_workflow", "cybint_threat_profile_workflow", "cybint_risk_workflow", "cybint_incident_link_workflow", "cybint_dissemination_workflow", "cybint_review_workflow", "cybint_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-cybint/dashboard", "component": "CYBINTDashboard", "permission": "intel_cybint:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-cybint/authorities", "component": "CyberAuthorityConsole", "permission": "intel_cybint:authorities", "nav_group": "Governance"},
	{"name": "indicators", "path": "/intel-cybint/indicators", "component": "IndicatorRegistry", "permission": "intel_cybint:indicators", "nav_group": "Intelligence"},
	{"name": "sightings", "path": "/intel-cybint/sightings", "component": "SightingLedger", "permission": "intel_cybint:sightings", "nav_group": "Intelligence"},
	{"name": "enrichment", "path": "/intel-cybint/enrichment", "component": "EnrichmentWorkbench", "permission": "intel_cybint:enrichment", "nav_group": "Analysis"},
	{"name": "profiles", "path": "/intel-cybint/profiles", "component": "ThreatProfileWorkbench", "permission": "intel_cybint:profiles", "nav_group": "Analysis"},
	{"name": "risk", "path": "/intel-cybint/risk", "component": "CyberRiskWorkbench", "permission": "intel_cybint:risk", "nav_group": "Analysis"},
	{"name": "incidents", "path": "/intel-cybint/incidents", "component": "IncidentLinkConsole", "permission": "intel_cybint:incidents", "nav_group": "Response"},
	{"name": "dissemination", "path": "/intel-cybint/dissemination", "component": "CYBINTDisseminationConsole", "permission": "intel_cybint:dissemination", "nav_group": "Release"},
	{"name": "reviews", "path": "/intel-cybint/reviews", "component": "CYBINTReviewConsole", "permission": "intel_cybint:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-cybint/agents", "component": "CYBINTAgentWorkbench", "permission": "intel_cybint:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-cybint/settings", "component": "CYBINTSettings", "permission": "intel_cybint:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_cybint_control",
	"tokens": {"color.primary": "#1F4E79", "color.accent": "#6D28D9", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "indicators": {"icon": "fingerprint", "status_indicator": "tlp-chip"}, "sightings": {"icon": "radar", "status_indicator": "severity-chip"}, "enrichment": {"icon": "database-zap", "status_indicator": "confidence-chip"}, "profiles": {"icon": "network", "status_indicator": "classification-chip"}, "risk": {"icon": "shield-alert", "status_indicator": "risk-chip"}, "incidents": {"icon": "sirens", "status_indicator": "priority-chip"}, "dissemination": {"icon": "send", "status_indicator": "release-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": CYBINT_EVENT_STREAM, "key": "tenant_id", "events": ["cybint_authority_recorded", "cybint_indicator_recorded", "cybint_sighting_recorded", "cybint_enrichment_recorded", "cybint_profile_recorded", "cybint_risk_recorded", "cybint_incident_link_recorded", "cybint_dissemination_recorded", "cybint_review_recorded", "cybint_agent_registered"], "guardrails": ["cybint_batch_requires_bytewax", "privileged_cybint_agent_action_requires_human_approval", "offensive_cybint_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "cybint_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "cybint_policy_required", "required_action": "attach_cybint_policy"}},
	{"name": "authority_type_supported", "condition": {"operation": "record_authority", "authority_type_supported": False}, "effect": {"decision": "deny", "reason": "authority_type_not_supported", "required_action": "select_supported_authority_type"}},
	{"name": "authority_scope_required", "condition": {"operation": "record_authority", "scope_present": False}, "effect": {"decision": "deny", "reason": "authority_scope_required", "required_action": "attach_scope_reference"}},
	{"name": "authority_classification_supported", "condition": {"operation": "record_authority", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "authority_approver_required", "condition": {"operation": "record_authority", "approver_present": False}, "effect": {"decision": "deny", "reason": "authority_approver_required", "required_action": "record_approver"}},
	{"name": "authority_expiry_required", "condition": {"operation": "record_authority", "expiry_present": False}, "effect": {"decision": "deny", "reason": "authority_expiry_required", "required_action": "set_expiry"}},
	{"name": "authority_evidence_required", "condition": {"operation": "record_authority", "evidence_present": False}, "effect": {"decision": "deny", "reason": "authority_evidence_required", "required_action": "attach_authority_evidence"}},
	{"name": "indicator_type_supported", "condition": {"operation": "record_indicator", "indicator_type_supported": False}, "effect": {"decision": "deny", "reason": "indicator_type_not_supported", "required_action": "select_supported_indicator_type"}},
	{"name": "indicator_value_required", "condition": {"operation": "record_indicator", "indicator_value_present": False}, "effect": {"decision": "deny", "reason": "indicator_value_required", "required_action": "record_indicator_value"}},
	{"name": "indicator_tlp_supported", "condition": {"operation": "record_indicator", "tlp_supported": False}, "effect": {"decision": "deny", "reason": "tlp_not_supported", "required_action": "select_supported_tlp"}},
	{"name": "indicator_confidence_valid", "condition": {"operation": "record_indicator", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "indicator_authority_required", "condition": {"operation": "record_indicator", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "indicator_evidence_required", "condition": {"operation": "record_indicator", "evidence_present": False}, "effect": {"decision": "deny", "reason": "indicator_evidence_required", "required_action": "attach_indicator_evidence"}},
	{"name": "sighting_indicator_required", "condition": {"operation": "record_sighting", "indicator_present": False}, "effect": {"decision": "deny", "reason": "indicator_required", "required_action": "select_indicator"}},
	{"name": "sighting_source_required", "condition": {"operation": "record_sighting", "source_reference_present": False}, "effect": {"decision": "deny", "reason": "source_reference_required", "required_action": "attach_source_reference"}},
	{"name": "sighting_observed_at_required", "condition": {"operation": "record_sighting", "observed_at_present": False}, "effect": {"decision": "deny", "reason": "observed_at_required", "required_action": "record_observed_at"}},
	{"name": "sighting_severity_supported", "condition": {"operation": "record_sighting", "severity_supported": False}, "effect": {"decision": "deny", "reason": "severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "sighting_evidence_required", "condition": {"operation": "record_sighting", "evidence_present": False}, "effect": {"decision": "deny", "reason": "sighting_evidence_required", "required_action": "attach_sighting_evidence"}},
	{"name": "enrichment_indicator_required", "condition": {"operation": "record_enrichment", "indicator_present": False}, "effect": {"decision": "deny", "reason": "indicator_required", "required_action": "select_indicator"}},
	{"name": "enrichment_type_supported", "condition": {"operation": "record_enrichment", "enrichment_type_supported": False}, "effect": {"decision": "deny", "reason": "enrichment_type_not_supported", "required_action": "select_supported_enrichment_type"}},
	{"name": "enrichment_provider_required", "condition": {"operation": "record_enrichment", "provider_present": False}, "effect": {"decision": "deny", "reason": "provider_reference_required", "required_action": "attach_provider_reference"}},
	{"name": "enrichment_confidence_valid", "condition": {"operation": "record_enrichment", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "enrichment_analyst_required", "condition": {"operation": "record_enrichment", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "enrichment_evidence_required", "condition": {"operation": "record_enrichment", "evidence_present": False}, "effect": {"decision": "deny", "reason": "enrichment_evidence_required", "required_action": "attach_enrichment_evidence"}},
	{"name": "profile_type_supported", "condition": {"operation": "record_profile", "profile_type_supported": False}, "effect": {"decision": "deny", "reason": "profile_type_not_supported", "required_action": "select_supported_profile_type"}},
	{"name": "profile_name_required", "condition": {"operation": "record_profile", "name_present": False}, "effect": {"decision": "deny", "reason": "profile_name_required", "required_action": "name_profile"}},
	{"name": "profile_classification_supported", "condition": {"operation": "record_profile", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "profile_confidence_valid", "condition": {"operation": "record_profile", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "profile_analyst_required", "condition": {"operation": "record_profile", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "profile_evidence_required", "condition": {"operation": "record_profile", "evidence_present": False}, "effect": {"decision": "deny", "reason": "profile_evidence_required", "required_action": "attach_profile_evidence"}},
	{"name": "risk_indicator_required", "condition": {"operation": "record_risk", "indicator_present": False}, "effect": {"decision": "deny", "reason": "indicator_required", "required_action": "select_indicator"}},
	{"name": "risk_profile_required", "condition": {"operation": "record_risk", "profile_present": False}, "effect": {"decision": "deny", "reason": "profile_required", "required_action": "select_profile"}},
	{"name": "risk_level_supported", "condition": {"operation": "record_risk", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "risk_confidence_valid", "condition": {"operation": "record_risk", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "risk_analyst_required", "condition": {"operation": "record_risk", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "risk_evidence_required", "condition": {"operation": "record_risk", "evidence_present": False}, "effect": {"decision": "deny", "reason": "risk_evidence_required", "required_action": "attach_risk_evidence"}},
	{"name": "incident_assessment_required", "condition": {"operation": "record_incident_link", "assessment_present": False}, "effect": {"decision": "deny", "reason": "risk_assessment_required", "required_action": "select_risk_assessment"}},
	{"name": "incident_reference_required", "condition": {"operation": "record_incident_link", "incident_reference_present": False}, "effect": {"decision": "deny", "reason": "incident_reference_required", "required_action": "attach_incident_reference"}},
	{"name": "incident_priority_supported", "condition": {"operation": "record_incident_link", "response_priority_supported": False}, "effect": {"decision": "deny", "reason": "response_priority_not_supported", "required_action": "select_supported_priority"}},
	{"name": "incident_owner_required", "condition": {"operation": "record_incident_link", "owner_present": False}, "effect": {"decision": "deny", "reason": "incident_owner_required", "required_action": "assign_incident_owner"}},
	{"name": "incident_evidence_required", "condition": {"operation": "record_incident_link", "evidence_present": False}, "effect": {"decision": "deny", "reason": "incident_link_evidence_required", "required_action": "attach_incident_evidence"}},
	{"name": "dissemination_assessment_required", "condition": {"operation": "record_dissemination", "assessment_present": False}, "effect": {"decision": "deny", "reason": "risk_assessment_required", "required_action": "select_risk_assessment"}},
	{"name": "dissemination_audience_required", "condition": {"operation": "record_dissemination", "audience_present": False}, "effect": {"decision": "deny", "reason": "audience_required", "required_action": "select_audience"}},
	{"name": "dissemination_release_required", "condition": {"operation": "record_dissemination", "release_marking_present": False}, "effect": {"decision": "deny", "reason": "release_marking_required", "required_action": "set_release_marking"}},
	{"name": "dissemination_approval_required", "condition": {"operation": "record_dissemination", "approval_present": False}, "effect": {"decision": "deny", "reason": "dissemination_approval_required", "required_action": "attach_release_approval"}},
	{"name": "dissemination_evidence_required", "condition": {"operation": "record_dissemination", "evidence_present": False}, "effect": {"decision": "deny", "reason": "dissemination_evidence_required", "required_action": "attach_dissemination_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "cybint_batch_requires_bytewax", "condition": {"operation": "cybint_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_cybint_batch_to_bytewax"}},
	{"name": "cybint_agent_runtime_supported", "condition": {"operation": "register_cybint_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "cybint_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "cybint_agent_role_supported", "condition": {"operation": "register_cybint_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "cybint_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_cybint_agent_action_requires_human_approval", "condition": {"operation": "cybint_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "offensive_cybint_action_denied", "condition": {"operation": "cybint_agent_action", "offensive_or_exploit_scope": True}, "effect": {"decision": "deny", "reason": "offensive_or_exploit_scope_denied", "required_action": "remove_offensive_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-cybint/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
