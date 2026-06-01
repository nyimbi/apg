"""Executable capability contract for APG Dark Web Monitoring."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_darkweb"
CAPABILITY_NAME = "Dark Web Monitoring"
CAPABILITY_VERSION = "1.1.0"
DARKWEB_EVENT_STREAM = "apg.intel.darkweb.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["mission_order", "legal_mandate", "consent", "partner_authority", "security_monitoring_authority", "incident_response_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_PROGRAM_TYPES = ["brand_protection", "credential_exposure", "data_leak", "fraud_market", "watchlist", "threat_actor", "vulnerability_chatter", "executive_protection"]
SUPPORTED_NETWORK_TYPES = ["tor", "i2p", "paste_site", "invite_forum", "encrypted_market", "public_dump"]
SUPPORTED_SOURCE_TYPES = ["onion_service", "forum", "marketplace", "paste", "leak_site", "chat_channel", "index"]
SUPPORTED_OBSERVATION_TYPES = ["listing", "post", "paste", "profile", "escrow_ad", "threat_claim", "data_dump"]
SUPPORTED_INDICATOR_TYPES = ["credential_exposure", "data_breach", "contraband_signal", "fraud_listing", "exploit_chatter", "malware_ad", "infrastructure_ioc", "brand_abuse"]
SUPPORTED_ASSESSMENT_TYPES = ["marketplace_risk", "threat_actor", "exposure_risk", "fraud_risk", "breach_risk", "infrastructure_risk"]
SUPPORTED_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_REFERRAL_TYPES = ["incident_response", "legal_review", "fraud_review", "brand_protection", "partner_notice", "lawful_request", "compliance_review"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["authority_reviewer", "program_planner", "source_steward", "observation_analyst", "exposure_analyst", "dissemination_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"programs": {"supported_program_types": SUPPORTED_PROGRAM_TYPES, "supported_priorities": SUPPORTED_RISK_LEVELS, "authority_required": True, "evidence_required": True},
	"sources": {"supported_source_types": SUPPORTED_SOURCE_TYPES, "supported_network_types": SUPPORTED_NETWORK_TYPES, "authority_required": True, "access_review_required": True, "passive_monitoring_only": True, "evidence_required": True},
	"observations": {"supported_observation_types": SUPPORTED_OBSERVATION_TYPES, "program_required": True, "source_required": True, "content_fingerprint_required": True, "confidence_required": True, "observed_at_required": True, "evidence_required": True},
	"indicators": {"supported_indicator_types": SUPPORTED_INDICATOR_TYPES, "supported_risk_levels": SUPPORTED_RISK_LEVELS, "observation_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"marketplace_risk": {"supported_assessment_types": SUPPORTED_ASSESSMENT_TYPES, "indicator_required": True, "risk_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"threat_actors": {"indicator_required": True, "actor_reference_required": True, "risk_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"referrals": {"supported_types": SUPPORTED_REFERRAL_TYPES, "assessment_required": True, "recipient_required": True, "approval_required": True, "evidence_required": True},
	"dissemination": {"assessment_required": True, "audience_required": True, "release_marking_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True, "sensitive_action_scope_denied": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "privacy_review_required": True, "credential_use_denied": True, "exploit_procurement_denied": True, "contraband_transaction_denied": True, "evasion_denied": True, "doxxing_denied": True},
	"observability": {"event_stream": DARKWEB_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "graph": "grph", "rag": "ragn", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_programs": True, "enable_sources": True, "enable_observations": True, "enable_indicators": True, "enable_marketplace_risk": True, "enable_threat_actors": True, "enable_referrals": True, "enable_dissemination": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_darkweb_control", "allow_tenant_overrides": True},
}

PROVIDES = ["darkweb_authority_workflow", "darkweb_program_workflow", "darkweb_source_workflow", "darkweb_observation_workflow", "darkweb_indicator_workflow", "darkweb_marketplace_risk_workflow", "darkweb_threat_actor_workflow", "darkweb_referral_workflow", "darkweb_dissemination_workflow", "darkweb_review_workflow", "darkweb_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-darkweb/dashboard", "component": "DarkWebDashboard", "permission": "intel_darkweb:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-darkweb/authorities", "component": "MonitoringAuthorityConsole", "permission": "intel_darkweb:authorities", "nav_group": "Governance"},
	{"name": "programs", "path": "/intel-darkweb/programs", "component": "MonitoringProgramPlanner", "permission": "intel_darkweb:programs", "nav_group": "Planning"},
	{"name": "sources", "path": "/intel-darkweb/sources", "component": "HiddenServiceSourceRegistry", "permission": "intel_darkweb:sources", "nav_group": "Collection"},
	{"name": "observations", "path": "/intel-darkweb/observations", "component": "DarkWebObservationLedger", "permission": "intel_darkweb:observations", "nav_group": "Collection"},
	{"name": "indicators", "path": "/intel-darkweb/indicators", "component": "ExposureIndicatorWorkbench", "permission": "intel_darkweb:indicators", "nav_group": "Analysis"},
	{"name": "marketplace-risk", "path": "/intel-darkweb/marketplace-risk", "component": "MarketplaceRiskWorkbench", "permission": "intel_darkweb:marketplace_risk", "nav_group": "Analysis"},
	{"name": "threat-actors", "path": "/intel-darkweb/threat-actors", "component": "ThreatActorWorkbench", "permission": "intel_darkweb:threat_actors", "nav_group": "Analysis"},
	{"name": "referrals", "path": "/intel-darkweb/referrals", "component": "DarkWebReferralConsole", "permission": "intel_darkweb:referrals", "nav_group": "Release"},
	{"name": "dissemination", "path": "/intel-darkweb/dissemination", "component": "DarkWebDisseminationConsole", "permission": "intel_darkweb:dissemination", "nav_group": "Release"},
	{"name": "reviews", "path": "/intel-darkweb/reviews", "component": "DarkWebReviewConsole", "permission": "intel_darkweb:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-darkweb/agents", "component": "DarkWebAgentWorkbench", "permission": "intel_darkweb:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-darkweb/settings", "component": "DarkWebSettings", "permission": "intel_darkweb:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_darkweb_control",
	"tokens": {"color.primary": "#334155", "color.accent": "#0F766E", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "programs": {"icon": "target", "status_indicator": "priority-chip"}, "sources": {"icon": "network", "status_indicator": "network-chip"}, "observations": {"icon": "file-search", "status_indicator": "evidence-chip"}, "indicators": {"icon": "activity", "status_indicator": "indicator-chip"}, "marketplace-risk": {"icon": "shield-alert", "status_indicator": "risk-chip"}, "threat-actors": {"icon": "user-search", "status_indicator": "actor-chip"}, "referrals": {"icon": "file-output", "status_indicator": "referral-chip"}, "dissemination": {"icon": "send", "status_indicator": "release-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": DARKWEB_EVENT_STREAM, "key": "tenant_id", "events": ["darkweb_authority_recorded", "darkweb_program_recorded", "darkweb_source_registered", "darkweb_observation_recorded", "darkweb_indicator_recorded", "darkweb_marketplace_risk_recorded", "darkweb_threat_actor_recorded", "darkweb_referral_recorded", "darkweb_dissemination_recorded", "darkweb_review_recorded", "darkweb_agent_registered"], "guardrails": ["darkweb_batch_requires_bytewax", "privileged_darkweb_agent_action_requires_human_approval", "credential_use_action_denied", "exploit_procurement_action_denied", "contraband_transaction_action_denied", "evasion_action_denied", "doxxing_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "darkweb_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "darkweb_policy_required", "required_action": "attach_darkweb_policy"}},
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
	{"name": "source_type_supported", "condition": {"operation": "register_source", "source_type_supported": False}, "effect": {"decision": "deny", "reason": "source_type_not_supported", "required_action": "select_supported_source_type"}},
	{"name": "network_type_supported", "condition": {"operation": "register_source", "network_type_supported": False}, "effect": {"decision": "deny", "reason": "network_type_not_supported", "required_action": "select_supported_network_type"}},
	{"name": "source_reference_required", "condition": {"operation": "register_source", "source_reference_present": False}, "effect": {"decision": "deny", "reason": "source_reference_required", "required_action": "attach_source_reference"}},
	{"name": "source_custodian_required", "condition": {"operation": "register_source", "custodian_present": False}, "effect": {"decision": "deny", "reason": "source_custodian_required", "required_action": "assign_source_custodian"}},
	{"name": "source_authority_required", "condition": {"operation": "register_source", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "source_access_review_required", "condition": {"operation": "register_source", "access_review_present": False}, "effect": {"decision": "deny", "reason": "source_access_review_required", "required_action": "record_access_review"}},
	{"name": "source_evidence_required", "condition": {"operation": "register_source", "evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "observation_program_required", "condition": {"operation": "record_observation", "program_present": False}, "effect": {"decision": "deny", "reason": "program_required", "required_action": "select_program"}},
	{"name": "observation_source_required", "condition": {"operation": "record_observation", "source_present": False}, "effect": {"decision": "deny", "reason": "source_required", "required_action": "select_source"}},
	{"name": "observation_program_source_authority_match", "condition": {"operation": "record_observation", "program_source_authority_match": False}, "effect": {"decision": "deny", "reason": "authority_mismatch", "required_action": "align_program_source_authority"}},
	{"name": "observation_type_supported", "condition": {"operation": "record_observation", "observation_type_supported": False}, "effect": {"decision": "deny", "reason": "observation_type_not_supported", "required_action": "select_supported_observation_type"}},
	{"name": "observation_reference_required", "condition": {"operation": "record_observation", "observation_reference_present": False}, "effect": {"decision": "deny", "reason": "observation_reference_required", "required_action": "attach_observation_reference"}},
	{"name": "observation_fingerprint_required", "condition": {"operation": "record_observation", "fingerprint_present": False}, "effect": {"decision": "deny", "reason": "content_fingerprint_required", "required_action": "record_content_fingerprint"}},
	{"name": "observation_observed_at_required", "condition": {"operation": "record_observation", "observed_at_present": False}, "effect": {"decision": "deny", "reason": "observed_at_required", "required_action": "record_observed_at"}},
	{"name": "observation_confidence_valid", "condition": {"operation": "record_observation", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "observation_evidence_required", "condition": {"operation": "record_observation", "evidence_present": False}, "effect": {"decision": "deny", "reason": "observation_evidence_required", "required_action": "attach_observation_evidence"}},
	{"name": "indicator_observation_required", "condition": {"operation": "record_indicator", "observation_present": False}, "effect": {"decision": "deny", "reason": "observation_required", "required_action": "select_observation"}},
	{"name": "indicator_type_supported", "condition": {"operation": "record_indicator", "indicator_type_supported": False}, "effect": {"decision": "deny", "reason": "indicator_type_not_supported", "required_action": "select_supported_indicator_type"}},
	{"name": "indicator_risk_supported", "condition": {"operation": "record_indicator", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "indicator_confidence_valid", "condition": {"operation": "record_indicator", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "indicator_analyst_required", "condition": {"operation": "record_indicator", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "indicator_evidence_required", "condition": {"operation": "record_indicator", "evidence_present": False}, "effect": {"decision": "deny", "reason": "indicator_evidence_required", "required_action": "attach_indicator_evidence"}},
	{"name": "marketplace_indicator_required", "condition": {"operation": "record_marketplace_risk", "indicator_present": False}, "effect": {"decision": "deny", "reason": "indicator_required", "required_action": "select_indicator"}},
	{"name": "marketplace_assessment_type_supported", "condition": {"operation": "record_marketplace_risk", "assessment_type_supported": False}, "effect": {"decision": "deny", "reason": "assessment_type_not_supported", "required_action": "select_supported_assessment_type"}},
	{"name": "marketplace_risk_supported", "condition": {"operation": "record_marketplace_risk", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "marketplace_confidence_valid", "condition": {"operation": "record_marketplace_risk", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "marketplace_analyst_required", "condition": {"operation": "record_marketplace_risk", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "marketplace_evidence_required", "condition": {"operation": "record_marketplace_risk", "evidence_present": False}, "effect": {"decision": "deny", "reason": "marketplace_evidence_required", "required_action": "attach_marketplace_evidence"}},
	{"name": "actor_indicator_required", "condition": {"operation": "record_threat_actor", "indicator_present": False}, "effect": {"decision": "deny", "reason": "indicator_required", "required_action": "select_indicator"}},
	{"name": "actor_reference_required", "condition": {"operation": "record_threat_actor", "actor_reference_present": False}, "effect": {"decision": "deny", "reason": "actor_reference_required", "required_action": "attach_actor_reference"}},
	{"name": "actor_risk_supported", "condition": {"operation": "record_threat_actor", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "actor_confidence_valid", "condition": {"operation": "record_threat_actor", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "actor_analyst_required", "condition": {"operation": "record_threat_actor", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "actor_evidence_required", "condition": {"operation": "record_threat_actor", "evidence_present": False}, "effect": {"decision": "deny", "reason": "actor_evidence_required", "required_action": "attach_actor_evidence"}},
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
	{"name": "darkweb_batch_requires_bytewax", "condition": {"operation": "darkweb_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_darkweb_batch_to_bytewax"}},
	{"name": "darkweb_agent_runtime_supported", "condition": {"operation": "register_darkweb_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "darkweb_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "darkweb_agent_role_supported", "condition": {"operation": "register_darkweb_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "darkweb_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_darkweb_agent_action_requires_human_approval", "condition": {"operation": "darkweb_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "credential_use_action_denied", "condition": {"operation": "darkweb_agent_action", "credential_use_scope": True}, "effect": {"decision": "deny", "reason": "credential_use_scope_denied", "required_action": "remove_credential_use_scope"}},
	{"name": "exploit_procurement_action_denied", "condition": {"operation": "darkweb_agent_action", "exploit_procurement_scope": True}, "effect": {"decision": "deny", "reason": "exploit_procurement_scope_denied", "required_action": "remove_exploit_procurement_scope"}},
	{"name": "contraband_transaction_action_denied", "condition": {"operation": "darkweb_agent_action", "contraband_transaction_scope": True}, "effect": {"decision": "deny", "reason": "contraband_transaction_scope_denied", "required_action": "remove_contraband_transaction_scope"}},
	{"name": "evasion_action_denied", "condition": {"operation": "darkweb_agent_action", "evasion_scope": True}, "effect": {"decision": "deny", "reason": "evasion_scope_denied", "required_action": "remove_evasion_scope"}},
	{"name": "doxxing_action_denied", "condition": {"operation": "darkweb_agent_action", "doxxing_scope": True}, "effect": {"decision": "deny", "reason": "doxxing_scope_denied", "required_action": "remove_doxxing_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-darkweb/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
