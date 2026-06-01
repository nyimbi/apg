"""Executable capability contract for APG Radio Intelligence Listener."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_radio"
CAPABILITY_NAME = "Radio Intelligence Listener"
CAPABILITY_VERSION = "1.1.0"
RADIO_EVENT_STREAM = "apg.intel.radio.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["legal_mandate", "spectrum_license", "mission_order", "incident_response_authority", "public_safety_authority", "partner_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_BAND_TYPES = ["aviation", "maritime", "public_safety", "amateur", "broadcast", "satellite", "industrial", "defense"]
SUPPORTED_RECEIVER_TYPES = ["sdr", "fixed_station", "mobile_station", "remote_sensor", "spectrum_analyzer", "partner_feed"]
SUPPORTED_SESSION_TYPES = ["spectrum_survey", "incident_watch", "emergency_monitoring", "interference_hunt", "asset_tracking", "training_exercise"]
SUPPORTED_SIGNAL_TYPES = ["voice", "digital", "beacon", "telemetry", "ais", "ads_b", "broadcast", "unknown"]
SUPPORTED_CLASSIFICATION_TYPES = ["licensed_activity", "emergency_signal", "interference", "spoofing_suspected", "anomaly", "unlicensed_activity", "distress_signal"]
SUPPORTED_EVENT_TYPES = ["public_safety_event", "interference_event", "spectrum_violation", "distress_event", "asset_signal", "anomaly_review", "partner_notice"]
SUPPORTED_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_REFERRAL_TYPES = ["public_safety_notice", "regulator_notice", "incident_response", "partner_notice", "compliance_review", "maintenance_ticket"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["authority_reviewer", "band_planner", "receiver_steward", "signal_analyst", "event_analyst", "dissemination_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"band_plans": {"supported_band_types": SUPPORTED_BAND_TYPES, "frequency_bounds_required": True, "authority_required": True, "evidence_required": True},
	"receivers": {"supported_receiver_types": SUPPORTED_RECEIVER_TYPES, "site_reference_required": True, "custodian_required": True, "authority_required": True, "calibration_required": True, "evidence_required": True},
	"sessions": {"supported_session_types": SUPPORTED_SESSION_TYPES, "band_required": True, "receiver_required": True, "collection_plan_required": True, "evidence_required": True},
	"observations": {"supported_signal_types": SUPPORTED_SIGNAL_TYPES, "session_required": True, "frequency_required": True, "signal_fingerprint_required": True, "confidence_required": True, "observed_at_required": True, "evidence_required": True},
	"classifications": {"supported_types": SUPPORTED_CLASSIFICATION_TYPES, "supported_risk_levels": SUPPORTED_RISK_LEVELS, "observation_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"events": {"supported_event_types": SUPPORTED_EVENT_TYPES, "supported_risk_levels": SUPPORTED_RISK_LEVELS, "classification_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"referrals": {"supported_types": SUPPORTED_REFERRAL_TYPES, "assessment_required": True, "recipient_required": True, "approval_required": True, "evidence_required": True},
	"dissemination": {"assessment_required": True, "audience_required": True, "release_marking_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True, "transmit_scope_denied": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "passive_monitoring_only": True, "unauthorized_interception_denied": True, "decryption_denied": True, "jamming_denied": True, "spoofing_denied": True, "interference_denied": True},
	"observability": {"event_stream": RADIO_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "graph": "grph", "rag": "ragn", "geospatial": "geos", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_band_plans": True, "enable_receivers": True, "enable_sessions": True, "enable_observations": True, "enable_classifications": True, "enable_events": True, "enable_referrals": True, "enable_dissemination": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_radio_control", "allow_tenant_overrides": True},
}

PROVIDES = ["radio_authority_workflow", "radio_band_plan_workflow", "radio_receiver_workflow", "radio_collection_session_workflow", "radio_observation_workflow", "radio_classification_workflow", "radio_event_workflow", "radio_referral_workflow", "radio_dissemination_workflow", "radio_review_workflow", "radio_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn", "geos"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-radio/dashboard", "component": "RadioDashboard", "permission": "intel_radio:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-radio/authorities", "component": "RadioAuthorityConsole", "permission": "intel_radio:authorities", "nav_group": "Governance"},
	{"name": "band-plans", "path": "/intel-radio/band-plans", "component": "RadioBandPlanConsole", "permission": "intel_radio:band_plans", "nav_group": "Planning"},
	{"name": "receivers", "path": "/intel-radio/receivers", "component": "RadioReceiverRegistry", "permission": "intel_radio:receivers", "nav_group": "Collection"},
	{"name": "sessions", "path": "/intel-radio/sessions", "component": "RadioCollectionSessionConsole", "permission": "intel_radio:sessions", "nav_group": "Collection"},
	{"name": "observations", "path": "/intel-radio/observations", "component": "RadioSignalObservationLedger", "permission": "intel_radio:observations", "nav_group": "Signals"},
	{"name": "classifications", "path": "/intel-radio/classifications", "component": "RadioTransmissionWorkbench", "permission": "intel_radio:classifications", "nav_group": "Analysis"},
	{"name": "events", "path": "/intel-radio/events", "component": "RadioEventWorkbench", "permission": "intel_radio:events", "nav_group": "Analysis"},
	{"name": "referrals", "path": "/intel-radio/referrals", "component": "RadioReferralConsole", "permission": "intel_radio:referrals", "nav_group": "Release"},
	{"name": "dissemination", "path": "/intel-radio/dissemination", "component": "RadioDisseminationConsole", "permission": "intel_radio:dissemination", "nav_group": "Release"},
	{"name": "reviews", "path": "/intel-radio/reviews", "component": "RadioReviewConsole", "permission": "intel_radio:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-radio/agents", "component": "RadioAgentWorkbench", "permission": "intel_radio:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-radio/settings", "component": "RadioSettings", "permission": "intel_radio:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_radio_control",
	"tokens": {"color.primary": "#0369A1", "color.accent": "#0F766E", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "band-plans": {"icon": "waves", "status_indicator": "band-chip"}, "receivers": {"icon": "radio-receiver", "status_indicator": "receiver-chip"}, "sessions": {"icon": "clock", "status_indicator": "session-chip"}, "observations": {"icon": "activity", "status_indicator": "signal-chip"}, "classifications": {"icon": "list-checks", "status_indicator": "classification-chip"}, "events": {"icon": "radio", "status_indicator": "risk-chip"}, "referrals": {"icon": "file-output", "status_indicator": "referral-chip"}, "dissemination": {"icon": "send", "status_indicator": "release-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": RADIO_EVENT_STREAM, "key": "tenant_id", "events": ["radio_authority_recorded", "radio_band_plan_recorded", "radio_receiver_registered", "radio_session_recorded", "radio_observation_recorded", "radio_classification_recorded", "radio_event_recorded", "radio_referral_recorded", "radio_dissemination_recorded", "radio_review_recorded", "radio_agent_registered"], "guardrails": ["radio_batch_requires_bytewax", "privileged_radio_agent_action_requires_human_approval", "transmit_action_denied", "unauthorized_interception_action_denied", "decryption_action_denied", "jamming_action_denied", "spoofing_action_denied", "interference_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "radio_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "radio_policy_required", "required_action": "attach_radio_policy"}},
	{"name": "authority_type_supported", "condition": {"operation": "record_authority", "authority_type_supported": False}, "effect": {"decision": "deny", "reason": "authority_type_not_supported", "required_action": "select_supported_authority_type"}},
	{"name": "authority_scope_required", "condition": {"operation": "record_authority", "scope_present": False}, "effect": {"decision": "deny", "reason": "authority_scope_required", "required_action": "attach_scope_reference"}},
	{"name": "authority_classification_supported", "condition": {"operation": "record_authority", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "authority_approver_required", "condition": {"operation": "record_authority", "approver_present": False}, "effect": {"decision": "deny", "reason": "authority_approver_required", "required_action": "record_approver"}},
	{"name": "authority_expiry_required", "condition": {"operation": "record_authority", "expiry_present": False}, "effect": {"decision": "deny", "reason": "authority_expiry_required", "required_action": "set_expiry"}},
	{"name": "authority_evidence_required", "condition": {"operation": "record_authority", "evidence_present": False}, "effect": {"decision": "deny", "reason": "authority_evidence_required", "required_action": "attach_authority_evidence"}},
	{"name": "band_type_supported", "condition": {"operation": "record_band_plan", "band_type_supported": False}, "effect": {"decision": "deny", "reason": "band_type_not_supported", "required_action": "select_supported_band_type"}},
	{"name": "band_name_required", "condition": {"operation": "record_band_plan", "band_name_present": False}, "effect": {"decision": "deny", "reason": "band_name_required", "required_action": "name_band_plan"}},
	{"name": "band_frequency_min_valid", "condition": {"operation": "record_band_plan", "frequency_min_valid": False}, "effect": {"decision": "deny", "reason": "frequency_min_invalid", "required_action": "set_nonnegative_frequency_min"}},
	{"name": "band_frequency_max_valid", "condition": {"operation": "record_band_plan", "frequency_max_valid": False}, "effect": {"decision": "deny", "reason": "frequency_max_invalid", "required_action": "set_nonnegative_frequency_max"}},
	{"name": "band_frequency_range_valid", "condition": {"operation": "record_band_plan", "frequency_range_valid": False}, "effect": {"decision": "deny", "reason": "frequency_range_invalid", "required_action": "set_valid_frequency_range"}},
	{"name": "band_authority_required", "condition": {"operation": "record_band_plan", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "band_evidence_required", "condition": {"operation": "record_band_plan", "evidence_present": False}, "effect": {"decision": "deny", "reason": "band_evidence_required", "required_action": "attach_band_evidence"}},
	{"name": "receiver_type_supported", "condition": {"operation": "register_receiver", "receiver_type_supported": False}, "effect": {"decision": "deny", "reason": "receiver_type_not_supported", "required_action": "select_supported_receiver_type"}},
	{"name": "receiver_site_required", "condition": {"operation": "register_receiver", "site_reference_present": False}, "effect": {"decision": "deny", "reason": "receiver_site_required", "required_action": "attach_site_reference"}},
	{"name": "receiver_custodian_required", "condition": {"operation": "register_receiver", "custodian_present": False}, "effect": {"decision": "deny", "reason": "receiver_custodian_required", "required_action": "assign_receiver_custodian"}},
	{"name": "receiver_authority_required", "condition": {"operation": "register_receiver", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "receiver_calibration_required", "condition": {"operation": "register_receiver", "calibration_present": False}, "effect": {"decision": "deny", "reason": "receiver_calibration_required", "required_action": "attach_calibration_reference"}},
	{"name": "receiver_evidence_required", "condition": {"operation": "register_receiver", "evidence_present": False}, "effect": {"decision": "deny", "reason": "receiver_evidence_required", "required_action": "attach_receiver_evidence"}},
	{"name": "session_band_required", "condition": {"operation": "record_session", "band_present": False}, "effect": {"decision": "deny", "reason": "band_required", "required_action": "select_band_plan"}},
	{"name": "session_receiver_required", "condition": {"operation": "record_session", "receiver_present": False}, "effect": {"decision": "deny", "reason": "receiver_required", "required_action": "select_receiver"}},
	{"name": "session_band_receiver_authority_match", "condition": {"operation": "record_session", "band_receiver_authority_match": False}, "effect": {"decision": "deny", "reason": "authority_mismatch", "required_action": "align_band_receiver_authority"}},
	{"name": "session_type_supported", "condition": {"operation": "record_session", "session_type_supported": False}, "effect": {"decision": "deny", "reason": "session_type_not_supported", "required_action": "select_supported_session_type"}},
	{"name": "session_started_at_required", "condition": {"operation": "record_session", "started_at_present": False}, "effect": {"decision": "deny", "reason": "started_at_required", "required_action": "record_started_at"}},
	{"name": "session_plan_required", "condition": {"operation": "record_session", "collection_plan_present": False}, "effect": {"decision": "deny", "reason": "collection_plan_required", "required_action": "attach_collection_plan"}},
	{"name": "session_evidence_required", "condition": {"operation": "record_session", "evidence_present": False}, "effect": {"decision": "deny", "reason": "session_evidence_required", "required_action": "attach_session_evidence"}},
	{"name": "observation_session_required", "condition": {"operation": "record_observation", "session_present": False}, "effect": {"decision": "deny", "reason": "session_required", "required_action": "select_session"}},
	{"name": "observation_frequency_valid", "condition": {"operation": "record_observation", "frequency_valid": False}, "effect": {"decision": "deny", "reason": "frequency_invalid", "required_action": "set_nonnegative_frequency"}},
	{"name": "observation_frequency_in_band", "condition": {"operation": "record_observation", "frequency_in_band": False}, "effect": {"decision": "deny", "reason": "frequency_out_of_band", "required_action": "align_frequency_with_band_plan"}},
	{"name": "observation_signal_type_supported", "condition": {"operation": "record_observation", "signal_type_supported": False}, "effect": {"decision": "deny", "reason": "signal_type_not_supported", "required_action": "select_supported_signal_type"}},
	{"name": "observation_fingerprint_required", "condition": {"operation": "record_observation", "fingerprint_present": False}, "effect": {"decision": "deny", "reason": "signal_fingerprint_required", "required_action": "record_signal_fingerprint"}},
	{"name": "observation_observed_at_required", "condition": {"operation": "record_observation", "observed_at_present": False}, "effect": {"decision": "deny", "reason": "observed_at_required", "required_action": "record_observed_at"}},
	{"name": "observation_confidence_valid", "condition": {"operation": "record_observation", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "observation_evidence_required", "condition": {"operation": "record_observation", "evidence_present": False}, "effect": {"decision": "deny", "reason": "observation_evidence_required", "required_action": "attach_observation_evidence"}},
	{"name": "classification_observation_required", "condition": {"operation": "record_classification", "observation_present": False}, "effect": {"decision": "deny", "reason": "observation_required", "required_action": "select_observation"}},
	{"name": "classification_type_supported", "condition": {"operation": "record_classification", "classification_type_supported": False}, "effect": {"decision": "deny", "reason": "classification_type_not_supported", "required_action": "select_supported_classification_type"}},
	{"name": "classification_risk_supported", "condition": {"operation": "record_classification", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "classification_confidence_valid", "condition": {"operation": "record_classification", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "classification_analyst_required", "condition": {"operation": "record_classification", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "classification_evidence_required", "condition": {"operation": "record_classification", "evidence_present": False}, "effect": {"decision": "deny", "reason": "classification_evidence_required", "required_action": "attach_classification_evidence"}},
	{"name": "event_classification_required", "condition": {"operation": "record_event", "classification_present": False}, "effect": {"decision": "deny", "reason": "classification_required", "required_action": "select_classification"}},
	{"name": "event_type_supported", "condition": {"operation": "record_event", "event_type_supported": False}, "effect": {"decision": "deny", "reason": "event_type_not_supported", "required_action": "select_supported_event_type"}},
	{"name": "event_risk_supported", "condition": {"operation": "record_event", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "event_confidence_valid", "condition": {"operation": "record_event", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "event_analyst_required", "condition": {"operation": "record_event", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "event_evidence_required", "condition": {"operation": "record_event", "evidence_present": False}, "effect": {"decision": "deny", "reason": "event_evidence_required", "required_action": "attach_event_evidence"}},
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
	{"name": "radio_batch_requires_bytewax", "condition": {"operation": "radio_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_radio_batch_to_bytewax"}},
	{"name": "radio_agent_runtime_supported", "condition": {"operation": "register_radio_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "radio_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "radio_agent_role_supported", "condition": {"operation": "register_radio_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "radio_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_radio_agent_action_requires_human_approval", "condition": {"operation": "radio_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "transmit_action_denied", "condition": {"operation": "radio_agent_action", "transmit_scope": True}, "effect": {"decision": "deny", "reason": "transmit_scope_denied", "required_action": "remove_transmit_scope"}},
	{"name": "unauthorized_interception_action_denied", "condition": {"operation": "radio_agent_action", "unauthorized_interception_scope": True}, "effect": {"decision": "deny", "reason": "unauthorized_interception_scope_denied", "required_action": "remove_interception_scope"}},
	{"name": "decryption_action_denied", "condition": {"operation": "radio_agent_action", "decryption_scope": True}, "effect": {"decision": "deny", "reason": "decryption_scope_denied", "required_action": "remove_decryption_scope"}},
	{"name": "jamming_action_denied", "condition": {"operation": "radio_agent_action", "jamming_scope": True}, "effect": {"decision": "deny", "reason": "jamming_scope_denied", "required_action": "remove_jamming_scope"}},
	{"name": "spoofing_action_denied", "condition": {"operation": "radio_agent_action", "spoofing_scope": True}, "effect": {"decision": "deny", "reason": "spoofing_scope_denied", "required_action": "remove_spoofing_scope"}},
	{"name": "interference_action_denied", "condition": {"operation": "radio_agent_action", "interference_scope": True}, "effect": {"decision": "deny", "reason": "interference_scope_denied", "required_action": "remove_interference_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-radio/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
