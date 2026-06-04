"""Executable capability contract for APG Financial Intelligence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_finint"
CAPABILITY_NAME = "Financial Intelligence"
CAPABILITY_VERSION = "1.1.0"
FININT_EVENT_STREAM = "apg.intel.finint.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["legal_mandate", "regulatory_authority", "consent", "partner_authority", "mission_order"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_SOURCE_TYPES = ["bank_feed", "payment_network", "crypto_exchange", "trade_registry", "sanctions_feed", "partner_report", "public_filing", "case_system"]
SUPPORTED_SUBJECT_TYPES = ["person", "organization", "account", "wallet", "merchant", "vessel", "case"]
SUPPORTED_RISK_TIERS = ["low", "medium", "high", "critical"]
SUPPORTED_TRANSACTION_TYPES = ["credit", "debit", "transfer", "cash", "card", "crypto", "trade", "asset"]
SUPPORTED_PATTERN_TYPES = ["structuring", "rapid_movement", "round_tripping", "trade_mismatch", "sanctions_proximity", "fraud_ring", "crypto_mixing", "unusual_velocity"]
SUPPORTED_RISK_TYPES = ["aml", "sanctions", "fraud", "corruption", "terror_financing", "tax_evasion", "proliferation_finance", "market_abuse"]
SUPPORTED_REFERRAL_TYPES = ["sar", "str", "case_escalation", "partner_notice", "lawful_request", "compliance_review"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["authority_reviewer", "source_steward", "transaction_analyst", "pattern_analyst", "risk_analyst", "dissemination_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"sources": {"supported_source_types": SUPPORTED_SOURCE_TYPES, "jurisdiction_required": True, "owner_required": True, "authority_required": True, "evidence_required": True},
	"subjects": {"supported_subject_types": SUPPORTED_SUBJECT_TYPES, "supported_risk_tiers": SUPPORTED_RISK_TIERS, "authority_required": True, "subject_reference_required": True, "evidence_required": True},
	"transactions": {"supported_transaction_types": SUPPORTED_TRANSACTION_TYPES, "source_required": True, "subject_required": True, "transaction_reference_required": True, "positive_amount_required": True, "currency_required": True, "occurred_at_required": True, "evidence_required": True},
	"patterns": {"supported_types": SUPPORTED_PATTERN_TYPES, "transaction_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"risk": {"supported_types": SUPPORTED_RISK_TYPES, "supported_levels": SUPPORTED_RISK_TIERS, "pattern_required": True, "confidence_required": True, "analyst_required": True, "evidence_required": True},
	"referrals": {"supported_types": SUPPORTED_REFERRAL_TYPES, "assessment_required": True, "recipient_required": True, "approval_required": True, "evidence_required": True},
	"dissemination": {"assessment_required": True, "audience_required": True, "release_marking_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "privacy_review_required": True, "funds_movement_denied": True},
	"observability": {"event_stream": FININT_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "graph": "grph", "rag": "ragn", "payments": "fintech_payments", "kyc": "fintech_kyc", "aml": "fintech_aml", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_sources": True, "enable_subjects": True, "enable_transactions": True, "enable_patterns": True, "enable_risk": True, "enable_referrals": True, "enable_dissemination": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_finint_control", "allow_tenant_overrides": True},
}

PROVIDES = ["finint_authority_workflow", "finint_source_workflow", "finint_subject_workflow", "finint_transaction_workflow", "finint_pattern_workflow", "finint_risk_workflow", "finint_referral_workflow", "finint_dissemination_workflow", "finint_review_workflow", "finint_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn", "fintech_kyc", "fintech_aml"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-finint/dashboard", "component": "FININTDashboard", "permission": "intel_finint:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-finint/authorities", "component": "FinancialAuthorityConsole", "permission": "intel_finint:authorities", "nav_group": "Governance"},
	{"name": "sources", "path": "/intel-finint/sources", "component": "FinancialSourceRegistry", "permission": "intel_finint:sources", "nav_group": "Data"},
	{"name": "subjects", "path": "/intel-finint/subjects", "component": "FinancialSubjectRegistry", "permission": "intel_finint:subjects", "nav_group": "Data"},
	{"name": "transactions", "path": "/intel-finint/transactions", "component": "FinancialTransactionLedger", "permission": "intel_finint:transactions", "nav_group": "Intelligence"},
	{"name": "patterns", "path": "/intel-finint/patterns", "component": "FinancialPatternWorkbench", "permission": "intel_finint:patterns", "nav_group": "Analysis"},
	{"name": "risk", "path": "/intel-finint/risk", "component": "FinancialRiskWorkbench", "permission": "intel_finint:risk", "nav_group": "Analysis"},
	{"name": "referrals", "path": "/intel-finint/referrals", "component": "FinancialReferralConsole", "permission": "intel_finint:referrals", "nav_group": "Release"},
	{"name": "dissemination", "path": "/intel-finint/dissemination", "component": "FININTDisseminationConsole", "permission": "intel_finint:dissemination", "nav_group": "Release"},
	{"name": "reviews", "path": "/intel-finint/reviews", "component": "FININTReviewConsole", "permission": "intel_finint:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-finint/agents", "component": "FININTAgentWorkbench", "permission": "intel_finint:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-finint/settings", "component": "FININTSettings", "permission": "intel_finint:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_finint_control",
	"tokens": {"color.primary": "#0F766E", "color.accent": "#854D0E", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "sources": {"icon": "database", "status_indicator": "jurisdiction-chip"}, "subjects": {"icon": "id-card", "status_indicator": "risk-tier-chip"}, "transactions": {"icon": "receipt", "status_indicator": "currency-chip"}, "patterns": {"icon": "activity", "status_indicator": "confidence-chip"}, "risk": {"icon": "shield-alert", "status_indicator": "risk-chip"}, "referrals": {"icon": "file-output", "status_indicator": "referral-chip"}, "dissemination": {"icon": "send", "status_indicator": "release-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": FININT_EVENT_STREAM, "key": "tenant_id", "events": ["finint_authority_recorded", "finint_source_registered", "finint_subject_recorded", "finint_transaction_recorded", "finint_pattern_recorded", "finint_risk_recorded", "finint_referral_recorded", "finint_dissemination_recorded", "finint_review_recorded", "finint_agent_registered"], "guardrails": ["finint_batch_requires_bytewax", "privileged_finint_agent_action_requires_human_approval", "funds_movement_action_denied", "cross_tenant_finint_action_denied", "privilege_escalation_action_denied", "autonomous_sar_filing_action_denied", "unapproved_subject_profiling_action_denied", "evidence_fabrication_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "finint_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "finint_policy_required", "required_action": "attach_finint_policy"}},
	{"name": "authority_type_supported", "condition": {"operation": "record_authority", "authority_type_supported": False}, "effect": {"decision": "deny", "reason": "authority_type_not_supported", "required_action": "select_supported_authority_type"}},
	{"name": "authority_scope_required", "condition": {"operation": "record_authority", "scope_present": False}, "effect": {"decision": "deny", "reason": "authority_scope_required", "required_action": "attach_scope_reference"}},
	{"name": "authority_classification_supported", "condition": {"operation": "record_authority", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "authority_approver_required", "condition": {"operation": "record_authority", "approver_present": False}, "effect": {"decision": "deny", "reason": "authority_approver_required", "required_action": "record_approver"}},
	{"name": "authority_expiry_required", "condition": {"operation": "record_authority", "expiry_present": False}, "effect": {"decision": "deny", "reason": "authority_expiry_required", "required_action": "set_expiry"}},
	{"name": "authority_evidence_required", "condition": {"operation": "record_authority", "evidence_present": False}, "effect": {"decision": "deny", "reason": "authority_evidence_required", "required_action": "attach_authority_evidence"}},
	{"name": "source_type_supported", "condition": {"operation": "register_source", "source_type_supported": False}, "effect": {"decision": "deny", "reason": "source_type_not_supported", "required_action": "select_supported_source_type"}},
	{"name": "source_jurisdiction_required", "condition": {"operation": "register_source", "jurisdiction_present": False}, "effect": {"decision": "deny", "reason": "jurisdiction_required", "required_action": "record_jurisdiction"}},
	{"name": "source_owner_required", "condition": {"operation": "register_source", "owner_present": False}, "effect": {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}},
	{"name": "source_authority_required", "condition": {"operation": "register_source", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "source_evidence_required", "condition": {"operation": "register_source", "evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "subject_type_supported", "condition": {"operation": "record_subject", "subject_type_supported": False}, "effect": {"decision": "deny", "reason": "subject_type_not_supported", "required_action": "select_supported_subject_type"}},
	{"name": "subject_reference_required", "condition": {"operation": "record_subject", "subject_reference_present": False}, "effect": {"decision": "deny", "reason": "subject_reference_required", "required_action": "attach_subject_reference"}},
	{"name": "subject_risk_tier_supported", "condition": {"operation": "record_subject", "risk_tier_supported": False}, "effect": {"decision": "deny", "reason": "risk_tier_not_supported", "required_action": "select_supported_risk_tier"}},
	{"name": "subject_authority_required", "condition": {"operation": "record_subject", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "subject_evidence_required", "condition": {"operation": "record_subject", "evidence_present": False}, "effect": {"decision": "deny", "reason": "subject_evidence_required", "required_action": "attach_subject_evidence"}},
	{"name": "transaction_source_required", "condition": {"operation": "record_transaction", "source_present": False}, "effect": {"decision": "deny", "reason": "source_required", "required_action": "select_source"}},
	{"name": "transaction_subject_required", "condition": {"operation": "record_transaction", "subject_present": False}, "effect": {"decision": "deny", "reason": "subject_required", "required_action": "select_subject"}},
	{"name": "transaction_source_subject_authority_match", "condition": {"operation": "record_transaction", "source_subject_authority_match": False}, "effect": {"decision": "deny", "reason": "authority_mismatch", "required_action": "align_source_subject_authority"}},
	{"name": "transaction_reference_required", "condition": {"operation": "record_transaction", "transaction_reference_present": False}, "effect": {"decision": "deny", "reason": "transaction_reference_required", "required_action": "attach_transaction_reference"}},
	{"name": "transaction_amount_positive", "condition": {"operation": "record_transaction", "amount_positive": False}, "effect": {"decision": "deny", "reason": "amount_invalid", "required_action": "set_positive_amount"}},
	{"name": "transaction_currency_required", "condition": {"operation": "record_transaction", "currency_present": False}, "effect": {"decision": "deny", "reason": "currency_required", "required_action": "record_currency"}},
	{"name": "transaction_type_supported", "condition": {"operation": "record_transaction", "transaction_type_supported": False}, "effect": {"decision": "deny", "reason": "transaction_type_not_supported", "required_action": "select_supported_transaction_type"}},
	{"name": "transaction_occurred_at_required", "condition": {"operation": "record_transaction", "occurred_at_present": False}, "effect": {"decision": "deny", "reason": "occurred_at_required", "required_action": "record_occurred_at"}},
	{"name": "transaction_evidence_required", "condition": {"operation": "record_transaction", "evidence_present": False}, "effect": {"decision": "deny", "reason": "transaction_evidence_required", "required_action": "attach_transaction_evidence"}},
	{"name": "pattern_transaction_required", "condition": {"operation": "record_pattern", "transaction_present": False}, "effect": {"decision": "deny", "reason": "transaction_required", "required_action": "select_transaction"}},
	{"name": "pattern_type_supported", "condition": {"operation": "record_pattern", "pattern_type_supported": False}, "effect": {"decision": "deny", "reason": "pattern_type_not_supported", "required_action": "select_supported_pattern_type"}},
	{"name": "pattern_confidence_valid", "condition": {"operation": "record_pattern", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "pattern_analyst_required", "condition": {"operation": "record_pattern", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "pattern_evidence_required", "condition": {"operation": "record_pattern", "evidence_present": False}, "effect": {"decision": "deny", "reason": "pattern_evidence_required", "required_action": "attach_pattern_evidence"}},
	{"name": "risk_pattern_required", "condition": {"operation": "record_risk", "pattern_present": False}, "effect": {"decision": "deny", "reason": "pattern_required", "required_action": "select_pattern"}},
	{"name": "risk_type_supported", "condition": {"operation": "record_risk", "risk_type_supported": False}, "effect": {"decision": "deny", "reason": "risk_type_not_supported", "required_action": "select_supported_risk_type"}},
	{"name": "risk_level_supported", "condition": {"operation": "record_risk", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "risk_confidence_valid", "condition": {"operation": "record_risk", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "risk_analyst_required", "condition": {"operation": "record_risk", "analyst_present": False}, "effect": {"decision": "deny", "reason": "analyst_required", "required_action": "assign_analyst"}},
	{"name": "risk_evidence_required", "condition": {"operation": "record_risk", "evidence_present": False}, "effect": {"decision": "deny", "reason": "risk_evidence_required", "required_action": "attach_risk_evidence"}},
	{"name": "referral_assessment_required", "condition": {"operation": "record_referral", "assessment_present": False}, "effect": {"decision": "deny", "reason": "risk_assessment_required", "required_action": "select_risk_assessment"}},
	{"name": "referral_type_supported", "condition": {"operation": "record_referral", "referral_type_supported": False}, "effect": {"decision": "deny", "reason": "referral_type_not_supported", "required_action": "select_supported_referral_type"}},
	{"name": "referral_recipient_required", "condition": {"operation": "record_referral", "recipient_present": False}, "effect": {"decision": "deny", "reason": "recipient_required", "required_action": "select_recipient"}},
	{"name": "referral_approval_required", "condition": {"operation": "record_referral", "approval_present": False}, "effect": {"decision": "deny", "reason": "referral_approval_required", "required_action": "attach_referral_approval"}},
	{"name": "referral_evidence_required", "condition": {"operation": "record_referral", "evidence_present": False}, "effect": {"decision": "deny", "reason": "referral_evidence_required", "required_action": "attach_referral_evidence"}},
	{"name": "dissemination_assessment_required", "condition": {"operation": "record_dissemination", "assessment_present": False}, "effect": {"decision": "deny", "reason": "risk_assessment_required", "required_action": "select_risk_assessment"}},
	{"name": "dissemination_audience_required", "condition": {"operation": "record_dissemination", "audience_present": False}, "effect": {"decision": "deny", "reason": "audience_required", "required_action": "select_audience"}},
	{"name": "dissemination_release_required", "condition": {"operation": "record_dissemination", "release_marking_present": False}, "effect": {"decision": "deny", "reason": "release_marking_required", "required_action": "set_release_marking"}},
	{"name": "dissemination_approval_required", "condition": {"operation": "record_dissemination", "approval_present": False}, "effect": {"decision": "deny", "reason": "dissemination_approval_required", "required_action": "attach_release_approval"}},
	{"name": "dissemination_evidence_required", "condition": {"operation": "record_dissemination", "evidence_present": False}, "effect": {"decision": "deny", "reason": "dissemination_evidence_required", "required_action": "attach_dissemination_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "finint_batch_requires_bytewax", "condition": {"operation": "finint_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_finint_batch_to_bytewax"}},
	{"name": "finint_agent_runtime_supported", "condition": {"operation": "register_finint_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "finint_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "finint_agent_role_supported", "condition": {"operation": "register_finint_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "finint_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_finint_agent_action_requires_human_approval", "condition": {"operation": "finint_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "funds_movement_action_denied", "condition": {"operation": "finint_agent_action", "funds_movement_scope": True}, "effect": {"decision": "deny", "reason": "funds_movement_scope_denied", "required_action": "remove_funds_movement_scope"}},
	{"name": "cross_tenant_finint_action_denied", "condition": {"operation": "finint_agent_action", "cross_tenant_finint_scope": True}, "effect": {"decision": "deny", "reason": "cross_tenant_finint_scope_denied", "required_action": "remove_cross_tenant_scope"}},
	{"name": "privilege_escalation_action_denied", "condition": {"operation": "finint_agent_action", "privilege_escalation_scope": True}, "effect": {"decision": "deny", "reason": "privilege_escalation_scope_denied", "required_action": "remove_privilege_escalation_scope"}},
	{"name": "autonomous_sar_filing_action_denied", "condition": {"operation": "finint_agent_action", "autonomous_sar_filing_scope": True}, "effect": {"decision": "deny", "reason": "autonomous_sar_filing_scope_denied", "required_action": "remove_autonomous_sar_filing_scope"}},
	{"name": "unapproved_subject_profiling_action_denied", "condition": {"operation": "finint_agent_action", "unapproved_subject_profiling_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_subject_profiling_scope_denied", "required_action": "remove_subject_profiling_scope"}},
	{"name": "evidence_fabrication_action_denied", "condition": {"operation": "finint_agent_action", "evidence_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "evidence_fabrication_scope_denied", "required_action": "remove_evidence_fabrication_scope"}},
	{"name": "cross_tenant_finint_write_denied", "condition": {"operation_type": "write", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_finint_write_denied", "required_action": "remove_cross_tenant_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-finint/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
