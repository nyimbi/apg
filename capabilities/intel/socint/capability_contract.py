"""Executable capability contract for APG Social Media Intelligence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_socint"
CAPABILITY_NAME = "Social Media Intelligence"
CAPABILITY_VERSION = "1.1.0"
SOCINT_EVENT_STREAM = "apg.intel.socint.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["mission_order", "consent", "partner_authority", "legal_mandate", "public_interest_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_TOPIC_TYPES = ["brand", "event", "threat", "public_safety", "disinformation", "fraud", "crisis", "policy"]
SUPPORTED_PLATFORM_TYPES = ["microblog", "social_network", "video", "forum", "messaging_public_channel", "blog", "review_site", "public_web"]
SUPPORTED_SOURCE_TYPES = ["account", "page", "group", "hashtag", "keyword", "public_channel", "site"]
SUPPORTED_POST_TYPES = ["post", "reply", "share", "comment", "video", "image", "article"]
SUPPORTED_SIGNAL_TYPES = ["trend", "sentiment_shift", "coordination", "misinformation", "threat_signal", "fraud_signal", "crisis_signal", "bot_like_activity"]
SUPPORTED_INFLUENCE_TYPES = ["reach", "engagement", "amplification", "authority", "bridge", "coordination"]
SUPPORTED_NETWORK_TYPES = ["community", "amplification_cluster", "hashtag_graph", "account_graph", "narrative_cluster"]
SUPPORTED_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_REFERRAL_TYPES = ["case_escalation", "public_safety_notice", "fraud_review", "policy_review", "partner_notice", "compliance_review"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["authority_reviewer", "topic_planner", "source_steward", "signal_analyst", "network_analyst", "dissemination_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"topics": {"supported_topic_types": SUPPORTED_TOPIC_TYPES, "supported_priorities": SUPPORTED_RISK_LEVELS, "authority_required": True, "evidence_required": True},
	"sources": {"supported_source_types": SUPPORTED_SOURCE_TYPES, "supported_platform_types": SUPPORTED_PLATFORM_TYPES, "authority_required": True, "source_terms_review_required": True, "public_or_authorized_scope_required": True, "evidence_required": True},
	"posts": {"supported_post_types": SUPPORTED_POST_TYPES, "topic_required": True, "source_required": True, "content_fingerprint_required": True, "confidence_required": True, "observed_at_required": True, "evidence_required": True},
	"signals": {"supported_signal_types": SUPPORTED_SIGNAL_TYPES, "supported_risk_levels": SUPPORTED_RISK_LEVELS, "post_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"influence": {"supported_types": SUPPORTED_INFLUENCE_TYPES, "signal_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"networks": {"supported_types": SUPPORTED_NETWORK_TYPES, "supported_risk_levels": SUPPORTED_RISK_LEVELS, "signal_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"referrals": {"supported_types": SUPPORTED_REFERRAL_TYPES, "assessment_required": True, "recipient_required": True, "approval_required": True, "evidence_required": True},
	"dissemination": {"assessment_required": True, "audience_required": True, "release_marking_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True, "platform_abuse_scope_denied": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "privacy_review_required": True, "harassment_denied": True, "doxxing_denied": True, "evasion_denied": True},
	"observability": {"event_stream": SOCINT_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "graph": "grph", "rag": "ragn", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_topics": True, "enable_sources": True, "enable_posts": True, "enable_signals": True, "enable_influence": True, "enable_networks": True, "enable_referrals": True, "enable_dissemination": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_socint_control", "allow_tenant_overrides": True},
}

PROVIDES = ["socint_authority_workflow", "socint_topic_workflow", "socint_source_workflow", "socint_post_workflow", "socint_signal_workflow", "socint_influence_workflow", "socint_network_workflow", "socint_referral_workflow", "socint_dissemination_workflow", "socint_review_workflow", "socint_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-socint/dashboard", "component": "SOCINTDashboard", "permission": "intel_socint:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-socint/authorities", "component": "SocialAuthorityConsole", "permission": "intel_socint:authorities", "nav_group": "Governance"},
	{"name": "topics", "path": "/intel-socint/topics", "component": "SocialTopicPlanner", "permission": "intel_socint:topics", "nav_group": "Planning"},
	{"name": "sources", "path": "/intel-socint/sources", "component": "SocialSourceRegistry", "permission": "intel_socint:sources", "nav_group": "Collection"},
	{"name": "posts", "path": "/intel-socint/posts", "component": "SocialPostEvidenceLedger", "permission": "intel_socint:posts", "nav_group": "Collection"},
	{"name": "signals", "path": "/intel-socint/signals", "component": "SocialSignalWorkbench", "permission": "intel_socint:signals", "nav_group": "Analysis"},
	{"name": "influence", "path": "/intel-socint/influence", "component": "InfluenceAssessmentWorkbench", "permission": "intel_socint:influence", "nav_group": "Analysis"},
	{"name": "networks", "path": "/intel-socint/networks", "component": "NetworkAssessmentWorkbench", "permission": "intel_socint:networks", "nav_group": "Analysis"},
	{"name": "referrals", "path": "/intel-socint/referrals", "component": "SOCINTReferralConsole", "permission": "intel_socint:referrals", "nav_group": "Release"},
	{"name": "dissemination", "path": "/intel-socint/dissemination", "component": "SOCINTDisseminationConsole", "permission": "intel_socint:dissemination", "nav_group": "Release"},
	{"name": "reviews", "path": "/intel-socint/reviews", "component": "SOCINTReviewConsole", "permission": "intel_socint:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-socint/agents", "component": "SOCINTAgentWorkbench", "permission": "intel_socint:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-socint/settings", "component": "SOCINTSettings", "permission": "intel_socint:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_socint_control",
	"tokens": {"color.primary": "#2563EB", "color.accent": "#0F766E", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "topics": {"icon": "target", "status_indicator": "priority-chip"}, "sources": {"icon": "radio-tower", "status_indicator": "platform-chip"}, "posts": {"icon": "message-square", "status_indicator": "evidence-chip"}, "signals": {"icon": "activity", "status_indicator": "signal-chip"}, "influence": {"icon": "megaphone", "status_indicator": "confidence-chip"}, "networks": {"icon": "network", "status_indicator": "risk-chip"}, "referrals": {"icon": "file-output", "status_indicator": "referral-chip"}, "dissemination": {"icon": "send", "status_indicator": "release-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": SOCINT_EVENT_STREAM, "key": "tenant_id", "events": ["socint_authority_recorded", "socint_topic_recorded", "socint_source_registered", "socint_post_recorded", "socint_signal_recorded", "socint_influence_recorded", "socint_network_recorded", "socint_referral_recorded", "socint_dissemination_recorded", "socint_review_recorded", "socint_agent_registered"], "guardrails": ["socint_batch_requires_bytewax", "privileged_socint_agent_action_requires_human_approval", "platform_abuse_action_denied", "harassment_action_denied", "doxxing_action_denied", "evasion_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "socint_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "socint_policy_required", "required_action": "attach_socint_policy"}},
	{"name": "authority_type_supported", "condition": {"operation": "record_authority", "authority_type_supported": False}, "effect": {"decision": "deny", "reason": "authority_type_not_supported", "required_action": "select_supported_authority_type"}},
	{"name": "authority_scope_required", "condition": {"operation": "record_authority", "scope_present": False}, "effect": {"decision": "deny", "reason": "authority_scope_required", "required_action": "attach_scope_reference"}},
	{"name": "authority_classification_supported", "condition": {"operation": "record_authority", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "authority_approver_required", "condition": {"operation": "record_authority", "approver_present": False}, "effect": {"decision": "deny", "reason": "authority_approver_required", "required_action": "record_approver"}},
	{"name": "authority_expiry_required", "condition": {"operation": "record_authority", "expiry_present": False}, "effect": {"decision": "deny", "reason": "authority_expiry_required", "required_action": "set_expiry"}},
	{"name": "authority_evidence_required", "condition": {"operation": "record_authority", "evidence_present": False}, "effect": {"decision": "deny", "reason": "authority_evidence_required", "required_action": "attach_authority_evidence"}},
	{"name": "topic_type_supported", "condition": {"operation": "record_topic", "topic_type_supported": False}, "effect": {"decision": "deny", "reason": "topic_type_not_supported", "required_action": "select_supported_topic_type"}},
	{"name": "topic_name_required", "condition": {"operation": "record_topic", "topic_name_present": False}, "effect": {"decision": "deny", "reason": "topic_name_required", "required_action": "name_topic"}},
	{"name": "topic_priority_supported", "condition": {"operation": "record_topic", "priority_supported": False}, "effect": {"decision": "deny", "reason": "topic_priority_not_supported", "required_action": "select_supported_priority"}},
	{"name": "topic_authority_required", "condition": {"operation": "record_topic", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "topic_evidence_required", "condition": {"operation": "record_topic", "evidence_present": False}, "effect": {"decision": "deny", "reason": "topic_evidence_required", "required_action": "attach_topic_evidence"}},
	{"name": "source_type_supported", "condition": {"operation": "register_source", "source_type_supported": False}, "effect": {"decision": "deny", "reason": "source_type_not_supported", "required_action": "select_supported_source_type"}},
	{"name": "platform_type_supported", "condition": {"operation": "register_source", "platform_type_supported": False}, "effect": {"decision": "deny", "reason": "platform_type_not_supported", "required_action": "select_supported_platform_type"}},
	{"name": "source_reference_required", "condition": {"operation": "register_source", "source_reference_present": False}, "effect": {"decision": "deny", "reason": "source_reference_required", "required_action": "attach_source_reference"}},
	{"name": "source_owner_required", "condition": {"operation": "register_source", "owner_present": False}, "effect": {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}},
	{"name": "source_authority_required", "condition": {"operation": "register_source", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "source_terms_review_required", "condition": {"operation": "register_source", "terms_review_present": False}, "effect": {"decision": "deny", "reason": "source_terms_review_required", "required_action": "record_source_terms_review"}},
	{"name": "source_evidence_required", "condition": {"operation": "register_source", "evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "post_topic_required", "condition": {"operation": "record_post", "topic_present": False}, "effect": {"decision": "deny", "reason": "topic_required", "required_action": "select_topic"}},
	{"name": "post_source_required", "condition": {"operation": "record_post", "source_present": False}, "effect": {"decision": "deny", "reason": "source_required", "required_action": "select_source"}},
	{"name": "post_topic_source_authority_match", "condition": {"operation": "record_post", "topic_source_authority_match": False}, "effect": {"decision": "deny", "reason": "authority_mismatch", "required_action": "align_topic_source_authority"}},
	{"name": "post_type_supported", "condition": {"operation": "record_post", "post_type_supported": False}, "effect": {"decision": "deny", "reason": "post_type_not_supported", "required_action": "select_supported_post_type"}},
	{"name": "post_reference_required", "condition": {"operation": "record_post", "post_reference_present": False}, "effect": {"decision": "deny", "reason": "post_reference_required", "required_action": "attach_post_reference"}},
	{"name": "post_fingerprint_required", "condition": {"operation": "record_post", "fingerprint_present": False}, "effect": {"decision": "deny", "reason": "content_fingerprint_required", "required_action": "record_content_fingerprint"}},
	{"name": "post_observed_at_required", "condition": {"operation": "record_post", "observed_at_present": False}, "effect": {"decision": "deny", "reason": "observed_at_required", "required_action": "record_observed_at"}},
	{"name": "post_confidence_valid", "condition": {"operation": "record_post", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "post_evidence_required", "condition": {"operation": "record_post", "evidence_present": False}, "effect": {"decision": "deny", "reason": "post_evidence_required", "required_action": "attach_post_evidence"}},
	{"name": "signal_post_required", "condition": {"operation": "record_signal", "post_present": False}, "effect": {"decision": "deny", "reason": "post_required", "required_action": "select_post"}},
	{"name": "signal_type_supported", "condition": {"operation": "record_signal", "signal_type_supported": False}, "effect": {"decision": "deny", "reason": "signal_type_not_supported", "required_action": "select_supported_signal_type"}},
	{"name": "signal_risk_supported", "condition": {"operation": "record_signal", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "signal_confidence_valid", "condition": {"operation": "record_signal", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "signal_analyst_required", "condition": {"operation": "record_signal", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "signal_evidence_required", "condition": {"operation": "record_signal", "evidence_present": False}, "effect": {"decision": "deny", "reason": "signal_evidence_required", "required_action": "attach_signal_evidence"}},
	{"name": "influence_signal_required", "condition": {"operation": "record_influence", "signal_present": False}, "effect": {"decision": "deny", "reason": "signal_required", "required_action": "select_signal"}},
	{"name": "influence_type_supported", "condition": {"operation": "record_influence", "influence_type_supported": False}, "effect": {"decision": "deny", "reason": "influence_type_not_supported", "required_action": "select_supported_influence_type"}},
	{"name": "influence_confidence_valid", "condition": {"operation": "record_influence", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "influence_analyst_required", "condition": {"operation": "record_influence", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "influence_evidence_required", "condition": {"operation": "record_influence", "evidence_present": False}, "effect": {"decision": "deny", "reason": "influence_evidence_required", "required_action": "attach_influence_evidence"}},
	{"name": "network_signal_required", "condition": {"operation": "record_network", "signal_present": False}, "effect": {"decision": "deny", "reason": "signal_required", "required_action": "select_signal"}},
	{"name": "network_type_supported", "condition": {"operation": "record_network", "network_type_supported": False}, "effect": {"decision": "deny", "reason": "network_type_not_supported", "required_action": "select_supported_network_type"}},
	{"name": "network_risk_supported", "condition": {"operation": "record_network", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "network_confidence_valid", "condition": {"operation": "record_network", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "network_analyst_required", "condition": {"operation": "record_network", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "network_evidence_required", "condition": {"operation": "record_network", "evidence_present": False}, "effect": {"decision": "deny", "reason": "network_evidence_required", "required_action": "attach_network_evidence"}},
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
	{"name": "socint_batch_requires_bytewax", "condition": {"operation": "socint_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_socint_batch_to_bytewax"}},
	{"name": "socint_agent_runtime_supported", "condition": {"operation": "register_socint_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "socint_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "socint_agent_role_supported", "condition": {"operation": "register_socint_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "socint_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_socint_agent_action_requires_human_approval", "condition": {"operation": "socint_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "platform_abuse_action_denied", "condition": {"operation": "socint_agent_action", "platform_abuse_scope": True}, "effect": {"decision": "deny", "reason": "platform_abuse_scope_denied", "required_action": "remove_platform_abuse_scope"}},
	{"name": "harassment_action_denied", "condition": {"operation": "socint_agent_action", "harassment_scope": True}, "effect": {"decision": "deny", "reason": "harassment_scope_denied", "required_action": "remove_harassment_scope"}},
	{"name": "doxxing_action_denied", "condition": {"operation": "socint_agent_action", "doxxing_scope": True}, "effect": {"decision": "deny", "reason": "doxxing_scope_denied", "required_action": "remove_doxxing_scope"}},
	{"name": "evasion_action_denied", "condition": {"operation": "socint_agent_action", "evasion_scope": True}, "effect": {"decision": "deny", "reason": "evasion_scope_denied", "required_action": "remove_evasion_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-socint/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
