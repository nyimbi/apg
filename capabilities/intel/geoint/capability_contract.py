"""Executable capability contract for APG Geospatial Intelligence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_geoint"
CAPABILITY_NAME = "Geospatial Intelligence"
CAPABILITY_VERSION = "1.1.0"
GEOINT_EVENT_STREAM = "apg.intel.geoint.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["mission_order", "consent", "partner_authority", "legal_mandate", "regulatory_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_SOURCE_TYPES = ["satellite_imagery", "aerial_imagery", "drone_imagery", "public_map", "partner_feed", "field_survey", "open_geodata"]
SUPPORTED_SENSOR_TYPES = ["optical", "sar", "thermal", "multispectral", "hyperspectral", "lidar", "ais_adsb", "manual_survey"]
SUPPORTED_RESOLUTION_CLASSES = ["coarse", "medium", "high", "very_high", "metadata_only"]
SUPPORTED_COLLECTION_MODES = ["catalog_query", "scheduled_refresh", "partner_feed", "historical_import", "manual_upload"]
SUPPORTED_FEATURE_TYPES = ["facility", "route", "boundary", "terrain", "infrastructure", "maritime", "aviation", "environmental", "event"]
SUPPORTED_CHANGE_TYPES = ["new_feature", "removed_feature", "movement", "construction", "damage", "activity_change", "environmental_change"]
SUPPORTED_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_ASSESSMENT_TYPES = ["site_summary", "route_summary", "infrastructure_report", "activity_report", "change_report", "risk_assessment"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["authority_reviewer", "area_planner", "imagery_triage", "feature_analyst", "change_analyst", "dissemination_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"areas": {"supported_classifications": SUPPORTED_CLASSIFICATIONS, "geometry_required": True, "owner_required": True, "authority_required": True, "evidence_required": True},
	"sources": {"supported_source_types": SUPPORTED_SOURCE_TYPES, "supported_sensor_types": SUPPORTED_SENSOR_TYPES, "supported_resolution_classes": SUPPORTED_RESOLUTION_CLASSES, "owner_required": True, "authority_required": True, "evidence_required": True},
	"collection_plans": {"supported_modes": SUPPORTED_COLLECTION_MODES, "authority_required": True, "area_required": True, "source_required": True, "retention_positive": True, "approval_required": True, "evidence_required": True},
	"observations": {"plan_required": True, "observation_reference_required": True, "captured_at_required": True, "accuracy_required": True, "evidence_required": True},
	"features": {"supported_types": SUPPORTED_FEATURE_TYPES, "observation_required": True, "geometry_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"changes": {"supported_types": SUPPORTED_CHANGE_TYPES, "supported_severities": SUPPORTED_SEVERITIES, "feature_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"assessments": {"supported_types": SUPPORTED_ASSESSMENT_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "change_required": True, "analyst_required": True, "evidence_required": True},
	"dissemination": {"assessment_required": True, "audience_required": True, "release_marking_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "targeting_or_harmful_action_denied": True},
	"observability": {"event_stream": GEOINT_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "graph": "grph", "rag": "ragn", "geospatial": "geos", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_areas": True, "enable_sources": True, "enable_collection_plans": True, "enable_observations": True, "enable_features": True, "enable_changes": True, "enable_assessments": True, "enable_dissemination": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_geoint_control", "allow_tenant_overrides": True},
}

PROVIDES = ["geoint_authority_workflow", "geoint_area_workflow", "geoint_source_workflow", "geoint_collection_workflow", "geoint_observation_workflow", "geoint_feature_workflow", "geoint_change_workflow", "geoint_assessment_workflow", "geoint_dissemination_workflow", "geoint_review_workflow", "geoint_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn", "geos"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-geoint/dashboard", "component": "GEOINTDashboard", "permission": "intel_geoint:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-geoint/authorities", "component": "GeospatialAuthorityConsole", "permission": "intel_geoint:authorities", "nav_group": "Governance"},
	{"name": "areas", "path": "/intel-geoint/areas", "component": "AreaOfInterestRegistry", "permission": "intel_geoint:areas", "nav_group": "Planning"},
	{"name": "sources", "path": "/intel-geoint/sources", "component": "ImagerySourceRegistry", "permission": "intel_geoint:sources", "nav_group": "Collection"},
	{"name": "collection_plans", "path": "/intel-geoint/collection-plans", "component": "GeoCollectionPlanner", "permission": "intel_geoint:collection", "nav_group": "Collection"},
	{"name": "observations", "path": "/intel-geoint/observations", "component": "GeoObservationLedger", "permission": "intel_geoint:observations", "nav_group": "Processing"},
	{"name": "features", "path": "/intel-geoint/features", "component": "GeoFeatureWorkbench", "permission": "intel_geoint:features", "nav_group": "Analysis"},
	{"name": "changes", "path": "/intel-geoint/changes", "component": "ChangeDetectionWorkbench", "permission": "intel_geoint:changes", "nav_group": "Analysis"},
	{"name": "assessments", "path": "/intel-geoint/assessments", "component": "GeoAssessmentWorkbench", "permission": "intel_geoint:assessments", "nav_group": "Analysis"},
	{"name": "dissemination", "path": "/intel-geoint/dissemination", "component": "GEOINTDisseminationConsole", "permission": "intel_geoint:dissemination", "nav_group": "Release"},
	{"name": "reviews", "path": "/intel-geoint/reviews", "component": "GEOINTReviewConsole", "permission": "intel_geoint:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-geoint/agents", "component": "GEOINTAgentWorkbench", "permission": "intel_geoint:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-geoint/settings", "component": "GEOINTSettings", "permission": "intel_geoint:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_geoint_control",
	"tokens": {"color.primary": "#22543D", "color.accent": "#1D4ED8", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "areas": {"icon": "map", "status_indicator": "classification-chip"}, "sources": {"icon": "satellite", "status_indicator": "source-chip"}, "collection_plans": {"icon": "calendar-clock", "status_indicator": "plan-chip"}, "observations": {"icon": "scan-eye", "status_indicator": "accuracy-chip"}, "features": {"icon": "map-pin", "status_indicator": "confidence-chip"}, "changes": {"icon": "activity", "status_indicator": "severity-chip"}, "assessments": {"icon": "file-search", "status_indicator": "classification-chip"}, "dissemination": {"icon": "send", "status_indicator": "release-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": GEOINT_EVENT_STREAM, "key": "tenant_id", "events": ["geoint_authority_recorded", "geoint_area_recorded", "geoint_source_registered", "geoint_collection_plan_recorded", "geoint_observation_recorded", "geoint_feature_recorded", "geoint_change_recorded", "geoint_assessment_recorded", "geoint_dissemination_recorded", "geoint_review_recorded", "geoint_agent_registered"], "guardrails": ["geoint_batch_requires_bytewax", "privileged_geoint_agent_action_requires_human_approval", "targeting_geoint_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "geoint_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "geoint_policy_required", "required_action": "attach_geoint_policy"}},
	{"name": "authority_type_supported", "condition": {"operation": "record_authority", "authority_type_supported": False}, "effect": {"decision": "deny", "reason": "authority_type_not_supported", "required_action": "select_supported_authority_type"}},
	{"name": "authority_scope_required", "condition": {"operation": "record_authority", "scope_present": False}, "effect": {"decision": "deny", "reason": "authority_scope_required", "required_action": "attach_scope_reference"}},
	{"name": "authority_classification_supported", "condition": {"operation": "record_authority", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "authority_approver_required", "condition": {"operation": "record_authority", "approver_present": False}, "effect": {"decision": "deny", "reason": "authority_approver_required", "required_action": "record_approver"}},
	{"name": "authority_expiry_required", "condition": {"operation": "record_authority", "expiry_present": False}, "effect": {"decision": "deny", "reason": "authority_expiry_required", "required_action": "set_expiry"}},
	{"name": "authority_evidence_required", "condition": {"operation": "record_authority", "evidence_present": False}, "effect": {"decision": "deny", "reason": "authority_evidence_required", "required_action": "attach_authority_evidence"}},
	{"name": "area_name_required", "condition": {"operation": "record_area", "name_present": False}, "effect": {"decision": "deny", "reason": "area_name_required", "required_action": "name_area"}},
	{"name": "area_geometry_required", "condition": {"operation": "record_area", "geometry_present": False}, "effect": {"decision": "deny", "reason": "geometry_reference_required", "required_action": "attach_geometry_reference"}},
	{"name": "area_classification_supported", "condition": {"operation": "record_area", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "area_owner_required", "condition": {"operation": "record_area", "owner_present": False}, "effect": {"decision": "deny", "reason": "area_owner_required", "required_action": "assign_area_owner"}},
	{"name": "area_authority_required", "condition": {"operation": "record_area", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "area_evidence_required", "condition": {"operation": "record_area", "evidence_present": False}, "effect": {"decision": "deny", "reason": "area_evidence_required", "required_action": "attach_area_evidence"}},
	{"name": "source_type_supported", "condition": {"operation": "register_source", "source_type_supported": False}, "effect": {"decision": "deny", "reason": "source_type_not_supported", "required_action": "select_supported_source_type"}},
	{"name": "sensor_type_supported", "condition": {"operation": "register_source", "sensor_type_supported": False}, "effect": {"decision": "deny", "reason": "sensor_type_not_supported", "required_action": "select_supported_sensor_type"}},
	{"name": "resolution_class_supported", "condition": {"operation": "register_source", "resolution_class_supported": False}, "effect": {"decision": "deny", "reason": "resolution_class_not_supported", "required_action": "select_supported_resolution_class"}},
	{"name": "source_owner_required", "condition": {"operation": "register_source", "owner_present": False}, "effect": {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}},
	{"name": "source_authority_required", "condition": {"operation": "register_source", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "source_evidence_required", "condition": {"operation": "register_source", "evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "plan_authority_required", "condition": {"operation": "record_collection_plan", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "plan_area_required", "condition": {"operation": "record_collection_plan", "area_present": False}, "effect": {"decision": "deny", "reason": "area_required", "required_action": "select_area"}},
	{"name": "plan_source_required", "condition": {"operation": "record_collection_plan", "source_present": False}, "effect": {"decision": "deny", "reason": "source_required", "required_action": "select_source"}},
	{"name": "plan_area_authority_match", "condition": {"operation": "record_collection_plan", "area_authority_match": False}, "effect": {"decision": "deny", "reason": "area_authority_mismatch", "required_action": "select_area_for_authority"}},
	{"name": "plan_source_authority_match", "condition": {"operation": "record_collection_plan", "source_authority_match": False}, "effect": {"decision": "deny", "reason": "source_authority_mismatch", "required_action": "select_source_for_authority"}},
	{"name": "plan_mode_supported", "condition": {"operation": "record_collection_plan", "collection_mode_supported": False}, "effect": {"decision": "deny", "reason": "collection_mode_not_supported", "required_action": "select_supported_collection_mode"}},
	{"name": "plan_retention_positive", "condition": {"operation": "record_collection_plan", "retention_days_positive": False}, "effect": {"decision": "deny", "reason": "retention_days_invalid", "required_action": "set_positive_retention_days"}},
	{"name": "plan_approval_required", "condition": {"operation": "record_collection_plan", "approval_present": False}, "effect": {"decision": "deny", "reason": "collection_approval_required", "required_action": "attach_collection_approval"}},
	{"name": "plan_evidence_required", "condition": {"operation": "record_collection_plan", "evidence_present": False}, "effect": {"decision": "deny", "reason": "collection_plan_evidence_required", "required_action": "attach_collection_plan_evidence"}},
	{"name": "observation_plan_required", "condition": {"operation": "record_observation", "plan_present": False}, "effect": {"decision": "deny", "reason": "collection_plan_required", "required_action": "select_collection_plan"}},
	{"name": "observation_reference_required", "condition": {"operation": "record_observation", "observation_reference_present": False}, "effect": {"decision": "deny", "reason": "observation_reference_required", "required_action": "attach_observation_reference"}},
	{"name": "observation_captured_at_required", "condition": {"operation": "record_observation", "captured_at_present": False}, "effect": {"decision": "deny", "reason": "captured_at_required", "required_action": "record_capture_time"}},
	{"name": "observation_accuracy_valid", "condition": {"operation": "record_observation", "accuracy_valid": False}, "effect": {"decision": "deny", "reason": "geospatial_accuracy_score_invalid", "required_action": "set_accuracy_0_to_1"}},
	{"name": "observation_evidence_required", "condition": {"operation": "record_observation", "evidence_present": False}, "effect": {"decision": "deny", "reason": "observation_evidence_required", "required_action": "attach_observation_evidence"}},
	{"name": "feature_observation_required", "condition": {"operation": "record_feature", "observation_present": False}, "effect": {"decision": "deny", "reason": "observation_required", "required_action": "select_observation"}},
	{"name": "feature_type_supported", "condition": {"operation": "record_feature", "feature_type_supported": False}, "effect": {"decision": "deny", "reason": "feature_type_not_supported", "required_action": "select_supported_feature_type"}},
	{"name": "feature_geometry_required", "condition": {"operation": "record_feature", "geometry_present": False}, "effect": {"decision": "deny", "reason": "geometry_reference_required", "required_action": "attach_feature_geometry"}},
	{"name": "feature_confidence_valid", "condition": {"operation": "record_feature", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "feature_analyst_required", "condition": {"operation": "record_feature", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "feature_evidence_required", "condition": {"operation": "record_feature", "evidence_present": False}, "effect": {"decision": "deny", "reason": "feature_evidence_required", "required_action": "attach_feature_evidence"}},
	{"name": "change_feature_required", "condition": {"operation": "record_change", "feature_present": False}, "effect": {"decision": "deny", "reason": "feature_required", "required_action": "select_feature"}},
	{"name": "change_type_supported", "condition": {"operation": "record_change", "change_type_supported": False}, "effect": {"decision": "deny", "reason": "change_type_not_supported", "required_action": "select_supported_change_type"}},
	{"name": "change_severity_supported", "condition": {"operation": "record_change", "severity_supported": False}, "effect": {"decision": "deny", "reason": "severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "change_confidence_valid", "condition": {"operation": "record_change", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "change_analyst_required", "condition": {"operation": "record_change", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "change_evidence_required", "condition": {"operation": "record_change", "evidence_present": False}, "effect": {"decision": "deny", "reason": "change_evidence_required", "required_action": "attach_change_evidence"}},
	{"name": "assessment_change_required", "condition": {"operation": "record_assessment", "change_present": False}, "effect": {"decision": "deny", "reason": "change_required", "required_action": "select_change"}},
	{"name": "assessment_type_supported", "condition": {"operation": "record_assessment", "assessment_type_supported": False}, "effect": {"decision": "deny", "reason": "assessment_type_not_supported", "required_action": "select_supported_assessment_type"}},
	{"name": "assessment_classification_supported", "condition": {"operation": "record_assessment", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "assessment_analyst_required", "condition": {"operation": "record_assessment", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "assessment_evidence_required", "condition": {"operation": "record_assessment", "evidence_present": False}, "effect": {"decision": "deny", "reason": "assessment_evidence_required", "required_action": "attach_assessment_evidence"}},
	{"name": "dissemination_assessment_required", "condition": {"operation": "record_dissemination", "assessment_present": False}, "effect": {"decision": "deny", "reason": "assessment_required", "required_action": "select_assessment"}},
	{"name": "dissemination_audience_required", "condition": {"operation": "record_dissemination", "audience_present": False}, "effect": {"decision": "deny", "reason": "audience_required", "required_action": "select_audience"}},
	{"name": "dissemination_release_required", "condition": {"operation": "record_dissemination", "release_marking_present": False}, "effect": {"decision": "deny", "reason": "release_marking_required", "required_action": "set_release_marking"}},
	{"name": "dissemination_approval_required", "condition": {"operation": "record_dissemination", "approval_present": False}, "effect": {"decision": "deny", "reason": "dissemination_approval_required", "required_action": "attach_release_approval"}},
	{"name": "dissemination_evidence_required", "condition": {"operation": "record_dissemination", "evidence_present": False}, "effect": {"decision": "deny", "reason": "dissemination_evidence_required", "required_action": "attach_dissemination_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "geoint_batch_requires_bytewax", "condition": {"operation": "geoint_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_geoint_batch_to_bytewax"}},
	{"name": "geoint_agent_runtime_supported", "condition": {"operation": "register_geoint_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "geoint_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "geoint_agent_role_supported", "condition": {"operation": "register_geoint_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "geoint_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_geoint_agent_action_requires_human_approval", "condition": {"operation": "geoint_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "targeting_geoint_action_denied", "condition": {"operation": "geoint_agent_action", "targeting_or_harmful_scope": True}, "effect": {"decision": "deny", "reason": "targeting_or_harmful_scope_denied", "required_action": "remove_targeting_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-geoint/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
