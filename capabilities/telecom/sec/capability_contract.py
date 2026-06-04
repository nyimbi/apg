"""Executable capability contract for APG Telecom Security."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "telecom_sec"
CAPABILITY_NAME = "Telecom Security"
CAPABILITY_VERSION = "1.0.0"
SEC_EVENT_STREAM = "apg.telecom.sec.lifecycle"

SUPPORTED_FRAUD_TYPES = ["sim_swap_fraud", "wangiri", "irsf", "pbx_hacking", "account_takeover", "subscription_fraud", "roaming_fraud", "bypass_fraud", "voip_fraud", "smishing"]
SUPPORTED_SS7_ATTACK_TYPES = ["location_tracking", "call_interception", "sms_interception", "dos_attack", "spoofing", "eavesdropping", "man_in_middle"]
SUPPORTED_DIAMETER_ATTACK_TYPES = ["eavesdropping", "identity_spoofing", "dos_attack", "replay_attack", "routing_manipulation"]
SUPPORTED_LAWFUL_INTERCEPT_TYPES = ["voice_call", "sms", "data_session", "location", "email", "social_media"]
SUPPORTED_INTERCEPT_STATUSES = ["pending_warrant", "active", "suspended", "completed", "cancelled"]
SUPPORTED_SECURITY_INCIDENT_TYPES = ["fraud_detection", "ss7_attack", "diameter_attack", "voip_fraud", "roaming_abuse", "data_breach", "insider_threat", "ddos", "regulatory_violation"]
SUPPORTED_INCIDENT_SEVERITIES = ["critical", "major", "minor", "informational"]
SUPPORTED_INCIDENT_STATUSES = ["new", "under_investigation", "contained", "eradicated", "recovered", "closed", "regulatory_reported"]
SUPPORTED_THREAT_INTEL_SOURCES = ["internal", "carrier_community", "gsma_fraud_forum", "regulator_feed", "open_source", "commercial_feed", "partner_network"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["fraud_analyst", "ss7_analyst", "intercept_manager", "incident_responder", "threat_intel_analyst"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"fraud": {"supported_types": SUPPORTED_FRAUD_TYPES, "real_time_scoring": True, "block_threshold": 0.85, "evidence_required": True},
	"ss7_security": {"supported_attack_types": SUPPORTED_SS7_ATTACK_TYPES, "firewall_enabled": True, "anomaly_detection": True, "gsma_cat1_enforcement": True},
	"diameter_security": {"supported_attack_types": SUPPORTED_DIAMETER_ATTACK_TYPES, "edge_filtering": True, "realm_validation": True},
	"lawful_intercept": {"supported_types": SUPPORTED_LAWFUL_INTERCEPT_TYPES, "supported_statuses": SUPPORTED_INTERCEPT_STATUSES, "warrant_required": True, "regulatory_authority_required": True, "strict_access_control": True},
	"incidents": {"supported_types": SUPPORTED_SECURITY_INCIDENT_TYPES, "supported_severities": SUPPORTED_INCIDENT_SEVERITIES, "supported_statuses": SUPPORTED_INCIDENT_STATUSES, "evidence_required": True, "regulatory_notification_enabled": True},
	"threat_intel": {"supported_sources": SUPPORTED_THREAT_INTEL_SOURCES, "ioc_sharing_enabled": True, "tlp_enforcement": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "intercept_without_warrant_denied": True, "fraud_block_requires_evidence": True, "cross_tenant_access_denied": True, "evidence_fabrication_denied": True},
	"observability": {"event_stream": SEC_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_fraud": True, "enable_ss7": True, "enable_diameter": True, "enable_lawful_intercept": True, "enable_incidents": True, "enable_threat_intel": True, "enable_agents": True},
	"theme": {"default_theme": "telecom_sec_control", "allow_tenant_overrides": True},
}

PROVIDES = ["fraud_management_workflow", "ss7_security_workflow", "diameter_security_workflow", "lawful_intercept_workflow", "security_incident_workflow", "threat_intel_workflow", "voip_fraud_detection_workflow", "roaming_security_workflow", "sec_agent_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "mqeb", "comp"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/telecom-sec/dashboard", "component": "SecDashboard", "permission": "telecom_sec:view", "nav_group": "Overview"},
	{"name": "fraud_queue", "path": "/telecom-sec/fraud", "component": "SecFraudConsole", "permission": "telecom_sec:fraud", "nav_group": "Fraud"},
	{"name": "fraud_rules", "path": "/telecom-sec/fraud-rules", "component": "SecFraudRuleConsole", "permission": "telecom_sec:fraud_rules", "nav_group": "Fraud"},
	{"name": "ss7_security", "path": "/telecom-sec/ss7", "component": "SecSs7Console", "permission": "telecom_sec:ss7", "nav_group": "Signalling Security"},
	{"name": "diameter_security", "path": "/telecom-sec/diameter", "component": "SecDiameterConsole", "permission": "telecom_sec:diameter", "nav_group": "Signalling Security"},
	{"name": "lawful_intercept", "path": "/telecom-sec/intercept", "component": "SecInterceptConsole", "permission": "telecom_sec:intercept", "nav_group": "Legal"},
	{"name": "incidents", "path": "/telecom-sec/incidents", "component": "SecIncidentQueue", "permission": "telecom_sec:incidents", "nav_group": "Incidents"},
	{"name": "threat_intel", "path": "/telecom-sec/threat-intel", "component": "SecThreatIntelConsole", "permission": "telecom_sec:threat_intel", "nav_group": "Intelligence"},
	{"name": "voip_fraud", "path": "/telecom-sec/voip-fraud", "component": "SecVoipFraudConsole", "permission": "telecom_sec:fraud", "nav_group": "Fraud"},
	{"name": "roaming_security", "path": "/telecom-sec/roaming", "component": "SecRoamingSecurityConsole", "permission": "telecom_sec:roaming", "nav_group": "Signalling Security"},
	{"name": "agents", "path": "/telecom-sec/agents", "component": "SecAgentWorkbench", "permission": "telecom_sec:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/telecom-sec/settings", "component": "SecSettings", "permission": "telecom_sec:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "telecom_sec_control",
	"tokens": {"color.primary": "#7F1D1D", "color.accent": "#1E40AF", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#0F172A", "surface.panel": "#1E293B", "text.primary": "#F1F5F9", "text.secondary": "#94A3B8", "border.radius": "6px", "density": "compact"},
	"components": {"fraud_queue": {"icon": "alert-triangle", "status_indicator": "fraud-type-chip"}, "ss7_security": {"icon": "shield-off", "status_indicator": "ss7-attack-chip"}, "diameter_security": {"icon": "lock", "status_indicator": "diameter-attack-chip"}, "lawful_intercept": {"icon": "eye", "status_indicator": "intercept-status-chip"}, "incidents": {"icon": "alert-octagon", "status_indicator": "incident-severity-chip"}, "threat_intel": {"icon": "crosshair", "status_indicator": "intel-source-chip"}, "voip_fraud": {"icon": "phone-off", "status_indicator": "voip-fraud-chip"}, "roaming_security": {"icon": "globe", "status_indicator": "roaming-security-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": SEC_EVENT_STREAM, "key": "tenant_id", "events": ["fraud_case_raised", "fraud_block_applied", "ss7_attack_detected", "diameter_attack_detected", "intercept_activated", "security_incident_opened", "security_incident_resolved", "threat_ioc_shared", "voip_fraud_detected", "sec_agent_registered"], "guardrails": ["sec_batch_requires_bytewax", "privileged_sec_agent_action_requires_human_approval", "intercept_without_warrant_denied", "fraud_block_requires_evidence", "evidence_fabrication_denied", "cross_tenant_access_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "sec_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "sec_policy_required", "required_action": "attach_sec_policy"}},
	{"name": "fraud_type_supported", "condition": {"operation": "raise_fraud_case", "fraud_type_supported": False}, "effect": {"decision": "deny", "reason": "fraud_type_not_supported", "required_action": "select_supported_fraud_type"}},
	{"name": "fraud_block_requires_evidence", "condition": {"operation": "apply_fraud_block", "evidence_present": False}, "effect": {"decision": "deny", "reason": "fraud_block_requires_evidence", "required_action": "attach_fraud_evidence"}},
	{"name": "fraud_confidence_required", "condition": {"operation": "raise_fraud_case", "confidence_present": False}, "effect": {"decision": "deny", "reason": "fraud_confidence_required", "required_action": "set_fraud_confidence"}},
	{"name": "ss7_attack_type_supported", "condition": {"operation": "record_ss7_attack", "attack_type_supported": False}, "effect": {"decision": "deny", "reason": "ss7_attack_type_not_supported", "required_action": "select_supported_ss7_attack_type"}},
	{"name": "ss7_evidence_required", "condition": {"operation": "record_ss7_attack", "evidence_present": False}, "effect": {"decision": "deny", "reason": "ss7_evidence_required", "required_action": "attach_ss7_evidence"}},
	{"name": "diameter_attack_type_supported", "condition": {"operation": "record_diameter_attack", "attack_type_supported": False}, "effect": {"decision": "deny", "reason": "diameter_attack_type_not_supported", "required_action": "select_supported_diameter_attack_type"}},
	{"name": "intercept_without_warrant_denied", "condition": {"operation": "activate_intercept", "warrant_present": False}, "effect": {"decision": "deny", "reason": "intercept_warrant_required", "required_action": "attach_lawful_warrant"}},
	{"name": "intercept_regulatory_authority_required", "condition": {"operation": "activate_intercept", "regulatory_authority_present": False}, "effect": {"decision": "deny", "reason": "regulatory_authority_required", "required_action": "attach_regulatory_authority"}},
	{"name": "intercept_type_supported", "condition": {"operation": "activate_intercept", "intercept_type_supported": False}, "effect": {"decision": "deny", "reason": "intercept_type_not_supported", "required_action": "select_supported_intercept_type"}},
	{"name": "intercept_status_supported", "condition": {"operation": "update_intercept_status", "intercept_status_supported": False}, "effect": {"decision": "deny", "reason": "intercept_status_not_supported", "required_action": "select_supported_intercept_status"}},
	{"name": "incident_type_supported", "condition": {"operation": "open_incident", "incident_type_supported": False}, "effect": {"decision": "deny", "reason": "incident_type_not_supported", "required_action": "select_supported_incident_type"}},
	{"name": "incident_severity_supported", "condition": {"operation": "open_incident", "severity_supported": False}, "effect": {"decision": "deny", "reason": "incident_severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "incident_evidence_required", "condition": {"operation": "open_incident", "evidence_present": False}, "effect": {"decision": "deny", "reason": "incident_evidence_required", "required_action": "attach_incident_evidence"}},
	{"name": "incident_status_supported", "condition": {"operation": "update_incident_status", "incident_status_supported": False}, "effect": {"decision": "deny", "reason": "incident_status_not_supported", "required_action": "select_supported_incident_status"}},
	{"name": "threat_intel_source_supported", "condition": {"operation": "record_threat_intel", "source_supported": False}, "effect": {"decision": "deny", "reason": "threat_intel_source_not_supported", "required_action": "select_supported_source"}},
	{"name": "evidence_fabrication_denied", "condition": {"operation": "sec_agent_action", "evidence_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "evidence_fabrication_denied", "required_action": "remove_evidence_fabrication_scope"}},
	{"name": "cross_tenant_access_denied", "condition": {"operation": "sec_agent_action", "cross_tenant_access_scope": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "remove_cross_tenant_access_scope"}},
	{"name": "sec_batch_requires_bytewax", "condition": {"operation": "sec_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_sec_batch_to_bytewax"}},
	{"name": "sec_agent_runtime_supported", "condition": {"operation": "register_sec_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "sec_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "sec_agent_role_supported", "condition": {"operation": "register_sec_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "sec_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "sec_agent_name_required", "condition": {"operation": "register_sec_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "sec_agent_name_required", "required_action": "name_sec_agent"}},
	{"name": "sec_agent_scope_required", "condition": {"operation": "register_sec_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "sec_agent_scope_required", "required_action": "bound_sec_agent_scope"}},
	{"name": "privileged_sec_agent_action_requires_human_approval", "condition": {"operation": "sec_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/telecom-sec/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
