"""Executable capability contract for APG Digital Surveillance."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_surveillance"
CAPABILITY_NAME = "Digital Surveillance"
CAPABILITY_VERSION = "1.1.0"
SURVEILLANCE_EVENT_STREAM = "apg.intel.surveillance.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["legal_mandate", "consent", "security_monitoring_authority", "incident_response_authority", "partner_authority", "public_safety_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_PROGRAM_TYPES = ["asset_protection", "facility_monitoring", "endpoint_monitoring", "fraud_monitoring", "public_safety", "incident_watch", "compliance_monitoring", "executive_protection"]
SUPPORTED_ASSET_TYPES = ["facility", "endpoint", "account", "device", "vehicle", "public_area", "network_segment", "cloud_resource"]
SUPPORTED_SENSOR_TYPES = ["camera", "edr", "network_sensor", "access_control", "location_beacon", "telemetry_feed", "partner_feed", "log_stream"]
SUPPORTED_OBSERVATION_TYPES = ["motion", "access_event", "endpoint_event", "network_event", "location_event", "anomaly", "policy_violation", "safety_event"]
SUPPORTED_ALERT_TYPES = ["intrusion", "tamper", "policy_violation", "anomalous_behavior", "safety_incident", "perimeter_event", "device_compromise", "data_exposure"]
SUPPORTED_ASSESSMENT_TYPES = ["physical_security", "cyber_security", "privacy", "safety", "compliance", "insider_risk"]
SUPPORTED_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_REFERRAL_TYPES = ["incident_response", "legal_review", "public_safety_notice", "compliance_review", "partner_notice", "maintenance_ticket"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["authority_reviewer", "program_planner", "asset_steward", "sensor_steward", "alert_analyst", "dissemination_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"programs": {"supported_program_types": SUPPORTED_PROGRAM_TYPES, "supported_priorities": SUPPORTED_RISK_LEVELS, "authority_required": True, "evidence_required": True},
	"assets": {"supported_asset_types": SUPPORTED_ASSET_TYPES, "owner_required": True, "authority_required": True, "privacy_review_required": True, "evidence_required": True},
	"sensors": {"supported_sensor_types": SUPPORTED_SENSOR_TYPES, "asset_required": True, "custodian_required": True, "calibration_required": True, "evidence_required": True},
	"observations": {"supported_observation_types": SUPPORTED_OBSERVATION_TYPES, "program_required": True, "sensor_required": True, "content_fingerprint_required": True, "confidence_required": True, "observed_at_required": True, "evidence_required": True},
	"alerts": {"supported_alert_types": SUPPORTED_ALERT_TYPES, "supported_risk_levels": SUPPORTED_RISK_LEVELS, "observation_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"risk_assessments": {"supported_assessment_types": SUPPORTED_ASSESSMENT_TYPES, "supported_risk_levels": SUPPORTED_RISK_LEVELS, "alert_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"referrals": {"supported_types": SUPPORTED_REFERRAL_TYPES, "assessment_required": True, "recipient_required": True, "approval_required": True, "evidence_required": True},
	"dissemination": {"assessment_required": True, "audience_required": True, "release_marking_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True, "covert_scope_denied": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "privacy_review_required": True, "covert_tracking_denied": True, "stalking_denied": True, "spyware_denied": True, "credential_capture_denied": True, "bypass_denied": True, "biometric_identification_denied": True, "exfiltration_denied": True},
	"observability": {"event_stream": SURVEILLANCE_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "vision": "cvsn", "graph": "grph", "rag": "ragn", "geospatial": "geos", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_programs": True, "enable_assets": True, "enable_sensors": True, "enable_observations": True, "enable_alerts": True, "enable_risk": True, "enable_referrals": True, "enable_dissemination": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_surveillance_control", "allow_tenant_overrides": True},
}

PROVIDES = ["surveillance_authority_workflow", "surveillance_program_workflow", "surveillance_asset_workflow", "surveillance_sensor_workflow", "surveillance_observation_workflow", "surveillance_alert_workflow", "surveillance_risk_workflow", "surveillance_referral_workflow", "surveillance_dissemination_workflow", "surveillance_review_workflow", "surveillance_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "cvsn", "grph", "ragn", "geos"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-surveillance/dashboard", "component": "SurveillanceDashboard", "permission": "intel_surveillance:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-surveillance/authorities", "component": "SurveillanceAuthorityConsole", "permission": "intel_surveillance:authorities", "nav_group": "Governance"},
	{"name": "programs", "path": "/intel-surveillance/programs", "component": "SurveillanceProgramPlanner", "permission": "intel_surveillance:programs", "nav_group": "Planning"},
	{"name": "assets", "path": "/intel-surveillance/assets", "component": "MonitoredAssetRegistry", "permission": "intel_surveillance:assets", "nav_group": "Assets"},
	{"name": "sensors", "path": "/intel-surveillance/sensors", "component": "SurveillanceSensorRegistry", "permission": "intel_surveillance:sensors", "nav_group": "Collection"},
	{"name": "observations", "path": "/intel-surveillance/observations", "component": "SurveillanceObservationLedger", "permission": "intel_surveillance:observations", "nav_group": "Collection"},
	{"name": "alerts", "path": "/intel-surveillance/alerts", "component": "SurveillanceAlertWorkbench", "permission": "intel_surveillance:alerts", "nav_group": "Analysis"},
	{"name": "risk", "path": "/intel-surveillance/risk", "component": "SurveillanceRiskWorkbench", "permission": "intel_surveillance:risk", "nav_group": "Analysis"},
	{"name": "referrals", "path": "/intel-surveillance/referrals", "component": "SurveillanceReferralConsole", "permission": "intel_surveillance:referrals", "nav_group": "Release"},
	{"name": "dissemination", "path": "/intel-surveillance/dissemination", "component": "SurveillanceDisseminationConsole", "permission": "intel_surveillance:dissemination", "nav_group": "Release"},
	{"name": "reviews", "path": "/intel-surveillance/reviews", "component": "SurveillanceReviewConsole", "permission": "intel_surveillance:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-surveillance/agents", "component": "SurveillanceAgentWorkbench", "permission": "intel_surveillance:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-surveillance/settings", "component": "SurveillanceSettings", "permission": "intel_surveillance:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_surveillance_control",
	"tokens": {"color.primary": "#0F766E", "color.accent": "#0369A1", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "programs": {"icon": "target", "status_indicator": "priority-chip"}, "assets": {"icon": "boxes", "status_indicator": "asset-chip"}, "sensors": {"icon": "cctv", "status_indicator": "sensor-chip"}, "observations": {"icon": "activity", "status_indicator": "evidence-chip"}, "alerts": {"icon": "bell-ring", "status_indicator": "alert-chip"}, "risk": {"icon": "shield-alert", "status_indicator": "risk-chip"}, "referrals": {"icon": "file-output", "status_indicator": "referral-chip"}, "dissemination": {"icon": "send", "status_indicator": "release-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": SURVEILLANCE_EVENT_STREAM, "key": "tenant_id", "events": ["surveillance_authority_recorded", "surveillance_program_recorded", "surveillance_asset_recorded", "surveillance_sensor_registered", "surveillance_observation_recorded", "surveillance_alert_recorded", "surveillance_risk_recorded", "surveillance_referral_recorded", "surveillance_dissemination_recorded", "surveillance_review_recorded", "surveillance_agent_registered"], "guardrails": ["surveillance_batch_requires_bytewax", "privileged_surveillance_agent_action_requires_human_approval", "covert_tracking_action_denied", "stalking_action_denied", "spyware_action_denied", "credential_capture_action_denied", "bypass_action_denied", "biometric_identification_action_denied", "exfiltration_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "surveillance_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "surveillance_policy_required", "required_action": "attach_surveillance_policy"}},
	{"name": "authority_type_supported", "condition": {"operation": "record_authority", "authority_type_supported": False}, "effect": {"decision": "deny", "reason": "authority_type_not_supported", "required_action": "select_supported_authority_type"}},
	{"name": "authority_scope_required", "condition": {"operation": "record_authority", "scope_present": False}, "effect": {"decision": "deny", "reason": "authority_scope_required", "required_action": "attach_scope_reference"}},
	{"name": "authority_classification_supported", "condition": {"operation": "record_authority", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "authority_approver_required", "condition": {"operation": "record_authority", "approver_present": False}, "effect": {"decision": "deny", "reason": "authority_approver_required", "required_action": "record_approver"}},
	{"name": "authority_expiry_required", "condition": {"operation": "record_authority", "expiry_present": False}, "effect": {"decision": "deny", "reason": "authority_expiry_required", "required_action": "set_expiry"}},
	{"name": "authority_evidence_required", "condition": {"operation": "record_authority", "evidence_present": False}, "effect": {"decision": "deny", "reason": "authority_evidence_required", "required_action": "attach_authority_evidence"}},
	{"name": "program_type_supported", "condition": {"operation": "record_program", "program_type_supported": False}, "effect": {"decision": "deny", "reason": "program_type_not_supported", "required_action": "select_supported_program_type"}},
	{"name": "program_name_required", "condition": {"operation": "record_program", "program_name_present": False}, "effect": {"decision": "deny", "reason": "program_name_required", "required_action": "name_program"}},
	{"name": "program_priority_supported", "condition": {"operation": "record_program", "priority_supported": False}, "effect": {"decision": "deny", "reason": "program_priority_not_supported", "required_action": "select_supported_priority"}},
	{"name": "program_authority_required", "condition": {"operation": "record_program", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "program_evidence_required", "condition": {"operation": "record_program", "evidence_present": False}, "effect": {"decision": "deny", "reason": "program_evidence_required", "required_action": "attach_program_evidence"}},
	{"name": "asset_type_supported", "condition": {"operation": "record_asset", "asset_type_supported": False}, "effect": {"decision": "deny", "reason": "asset_type_not_supported", "required_action": "select_supported_asset_type"}},
	{"name": "asset_reference_required", "condition": {"operation": "record_asset", "asset_reference_present": False}, "effect": {"decision": "deny", "reason": "asset_reference_required", "required_action": "attach_asset_reference"}},
	{"name": "asset_owner_required", "condition": {"operation": "record_asset", "owner_present": False}, "effect": {"decision": "deny", "reason": "asset_owner_required", "required_action": "assign_asset_owner"}},
	{"name": "asset_authority_required", "condition": {"operation": "record_asset", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "asset_privacy_review_required", "condition": {"operation": "record_asset", "privacy_review_present": False}, "effect": {"decision": "deny", "reason": "privacy_review_required", "required_action": "record_privacy_review"}},
	{"name": "asset_evidence_required", "condition": {"operation": "record_asset", "evidence_present": False}, "effect": {"decision": "deny", "reason": "asset_evidence_required", "required_action": "attach_asset_evidence"}},
	{"name": "sensor_type_supported", "condition": {"operation": "register_sensor", "sensor_type_supported": False}, "effect": {"decision": "deny", "reason": "sensor_type_not_supported", "required_action": "select_supported_sensor_type"}},
	{"name": "sensor_asset_required", "condition": {"operation": "register_sensor", "asset_present": False}, "effect": {"decision": "deny", "reason": "asset_required", "required_action": "select_asset"}},
	{"name": "sensor_reference_required", "condition": {"operation": "register_sensor", "sensor_reference_present": False}, "effect": {"decision": "deny", "reason": "sensor_reference_required", "required_action": "attach_sensor_reference"}},
	{"name": "sensor_custodian_required", "condition": {"operation": "register_sensor", "custodian_present": False}, "effect": {"decision": "deny", "reason": "sensor_custodian_required", "required_action": "assign_sensor_custodian"}},
	{"name": "sensor_calibration_required", "condition": {"operation": "register_sensor", "calibration_present": False}, "effect": {"decision": "deny", "reason": "sensor_calibration_required", "required_action": "attach_calibration_reference"}},
	{"name": "sensor_evidence_required", "condition": {"operation": "register_sensor", "evidence_present": False}, "effect": {"decision": "deny", "reason": "sensor_evidence_required", "required_action": "attach_sensor_evidence"}},
	{"name": "observation_program_required", "condition": {"operation": "record_observation", "program_present": False}, "effect": {"decision": "deny", "reason": "program_required", "required_action": "select_program"}},
	{"name": "observation_sensor_required", "condition": {"operation": "record_observation", "sensor_present": False}, "effect": {"decision": "deny", "reason": "sensor_required", "required_action": "select_sensor"}},
	{"name": "observation_program_sensor_authority_match", "condition": {"operation": "record_observation", "program_sensor_authority_match": False}, "effect": {"decision": "deny", "reason": "authority_mismatch", "required_action": "align_program_sensor_authority"}},
	{"name": "observation_type_supported", "condition": {"operation": "record_observation", "observation_type_supported": False}, "effect": {"decision": "deny", "reason": "observation_type_not_supported", "required_action": "select_supported_observation_type"}},
	{"name": "observation_reference_required", "condition": {"operation": "record_observation", "observation_reference_present": False}, "effect": {"decision": "deny", "reason": "observation_reference_required", "required_action": "attach_observation_reference"}},
	{"name": "observation_fingerprint_required", "condition": {"operation": "record_observation", "fingerprint_present": False}, "effect": {"decision": "deny", "reason": "content_fingerprint_required", "required_action": "record_content_fingerprint"}},
	{"name": "observation_observed_at_required", "condition": {"operation": "record_observation", "observed_at_present": False}, "effect": {"decision": "deny", "reason": "observed_at_required", "required_action": "record_observed_at"}},
	{"name": "observation_confidence_valid", "condition": {"operation": "record_observation", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "observation_evidence_required", "condition": {"operation": "record_observation", "evidence_present": False}, "effect": {"decision": "deny", "reason": "observation_evidence_required", "required_action": "attach_observation_evidence"}},
	{"name": "alert_observation_required", "condition": {"operation": "record_alert", "observation_present": False}, "effect": {"decision": "deny", "reason": "observation_required", "required_action": "select_observation"}},
	{"name": "alert_type_supported", "condition": {"operation": "record_alert", "alert_type_supported": False}, "effect": {"decision": "deny", "reason": "alert_type_not_supported", "required_action": "select_supported_alert_type"}},
	{"name": "alert_risk_supported", "condition": {"operation": "record_alert", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "alert_confidence_valid", "condition": {"operation": "record_alert", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "alert_analyst_required", "condition": {"operation": "record_alert", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "alert_evidence_required", "condition": {"operation": "record_alert", "evidence_present": False}, "effect": {"decision": "deny", "reason": "alert_evidence_required", "required_action": "attach_alert_evidence"}},
	{"name": "risk_alert_required", "condition": {"operation": "record_risk", "alert_present": False}, "effect": {"decision": "deny", "reason": "alert_required", "required_action": "select_alert"}},
	{"name": "risk_assessment_type_supported", "condition": {"operation": "record_risk", "assessment_type_supported": False}, "effect": {"decision": "deny", "reason": "assessment_type_not_supported", "required_action": "select_supported_assessment_type"}},
	{"name": "risk_level_supported", "condition": {"operation": "record_risk", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "risk_confidence_valid", "condition": {"operation": "record_risk", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "risk_analyst_required", "condition": {"operation": "record_risk", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "risk_evidence_required", "condition": {"operation": "record_risk", "evidence_present": False}, "effect": {"decision": "deny", "reason": "risk_evidence_required", "required_action": "attach_risk_evidence"}},
	{"name": "referral_assessment_required", "condition": {"operation": "record_referral", "assessment_present": False}, "effect": {"decision": "deny", "reason": "assessment_required", "required_action": "select_assessment"}},
	{"name": "referral_type_supported", "condition": {"operation": "record_referral", "referral_type_supported": False}, "effect": {"decision": "deny", "reason": "referral_type_not_supported", "required_action": "select_supported_referral_type"}},
	{"name": "referral_recipient_required", "condition": {"operation": "record_referral", "recipient_present": False}, "effect": {"decision": "deny", "reason": "recipient_required", "required_action": "select_recipient"}},
	{"name": "referral_approval_required", "condition": {"operation": "record_referral", "approval_present": False}, "effect": {"decision": "deny", "reason": "referral_approval_required", "required_action": "attach_referral_approval"}},
	{"name": "referral_evidence_required", "condition": {"operation": "record_referral", "evidence_present": False}, "effect": {"decision": "deny", "reason": "referral_evidence_required", "required_action": "attach_referral_evidence"}},
	{"name": "dissemination_assessment_required", "condition": {"operation": "record_dissemination", "assessment_present": False}, "effect": {"decision": "deny", "reason": "assessment_required", "required_action": "select_assessment"}},
	{"name": "dissemination_audience_required", "condition": {"operation": "record_dissemination", "audience_present": False}, "effect": {"decision": "deny", "reason": "audience_required", "required_action": "select_audience"}},
	{"name": "dissemination_release_required", "condition": {"operation": "record_dissemination", "release_marking_present": False}, "effect": {"decision": "deny", "reason": "release_marking_required", "required_action": "set_release_marking"}},
	{"name": "dissemination_approval_required", "condition": {"operation": "record_dissemination", "approval_present": False}, "effect": {"decision": "deny", "reason": "dissemination_approval_required", "required_action": "attach_release_approval"}},
	{"name": "dissemination_evidence_required", "condition": {"operation": "record_dissemination", "evidence_present": False}, "effect": {"decision": "deny", "reason": "dissemination_evidence_required", "required_action": "attach_dissemination_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "surveillance_batch_requires_bytewax", "condition": {"operation": "surveillance_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_surveillance_batch_to_bytewax"}},
	{"name": "surveillance_agent_runtime_supported", "condition": {"operation": "register_surveillance_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "surveillance_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "surveillance_agent_role_supported", "condition": {"operation": "register_surveillance_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "surveillance_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_surveillance_agent_action_requires_human_approval", "condition": {"operation": "surveillance_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "covert_tracking_action_denied", "condition": {"operation": "surveillance_agent_action", "covert_tracking_scope": True}, "effect": {"decision": "deny", "reason": "covert_tracking_scope_denied", "required_action": "remove_covert_tracking_scope"}},
	{"name": "stalking_action_denied", "condition": {"operation": "surveillance_agent_action", "stalking_scope": True}, "effect": {"decision": "deny", "reason": "stalking_scope_denied", "required_action": "remove_stalking_scope"}},
	{"name": "spyware_action_denied", "condition": {"operation": "surveillance_agent_action", "spyware_scope": True}, "effect": {"decision": "deny", "reason": "spyware_scope_denied", "required_action": "remove_spyware_scope"}},
	{"name": "credential_capture_action_denied", "condition": {"operation": "surveillance_agent_action", "credential_capture_scope": True}, "effect": {"decision": "deny", "reason": "credential_capture_scope_denied", "required_action": "remove_credential_capture_scope"}},
	{"name": "bypass_action_denied", "condition": {"operation": "surveillance_agent_action", "bypass_scope": True}, "effect": {"decision": "deny", "reason": "bypass_scope_denied", "required_action": "remove_bypass_scope"}},
	{"name": "biometric_identification_action_denied", "condition": {"operation": "surveillance_agent_action", "biometric_identification_scope": True}, "effect": {"decision": "deny", "reason": "biometric_identification_scope_denied", "required_action": "remove_biometric_identification_scope"}},
	{"name": "exfiltration_action_denied", "condition": {"operation": "surveillance_agent_action", "exfiltration_scope": True}, "effect": {"decision": "deny", "reason": "exfiltration_scope_denied", "required_action": "remove_exfiltration_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-surveillance/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
