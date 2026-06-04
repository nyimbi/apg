"""Executable capability contract for APG Know Your Customer."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_kyc"
CAPABILITY_NAME = "Know Your Customer"
CAPABILITY_VERSION = "1.1.0"
KYC_EVENT_STREAM = "apg.fintech.kyc.lifecycle"

SUPPORTED_CUSTOMER_TYPES = ["individual", "sole_proprietor", "business", "nonprofit", "government"]
SUPPORTED_DOCUMENT_TYPES = ["passport", "national_id", "driver_license", "resident_permit", "business_registration", "tax_id", "utility_bill", "bank_statement"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["kyc_ops_reviewer", "document_reviewer", "sanctions_reviewer", "risk_reviewer", "onboarding_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"profiles": {"supported_customer_types": SUPPORTED_CUSTOMER_TYPES, "legal_name_required": True, "country_required": True, "consent_required": True},
	"documents": {"supported_types": SUPPORTED_DOCUMENT_TYPES, "token_reference_required": True, "minimum_confidence": 0.75, "extracted_subject_required": True},
	"screening": {"sanctions_required": True, "pep_required": True, "adverse_media_required": True, "watchlist_required": True, "review_required_for_hits": True},
	"risk": {"high_risk_threshold": 75, "medium_risk_threshold": 45, "enhanced_due_diligence_required": True},
	"decisions": {"identity_document_required": True, "address_document_required": True, "screening_required": True, "risk_assessment_required": True, "consent_required": True, "expiry_days": 365},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_kyc_events": True, "customer_consent_required": True},
	"observability": {"event_stream": KYC_EVENT_STREAM, "stream_processor": "bytewax", "emit_profile_events": True, "emit_document_events": True, "emit_screening_events": True, "emit_agent_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "consent": "cons", "notifications": "ntfy", "biometrics": "biop", "vision": "cvsn", "nlp": "nlpc", "keys": "keym", "payments": "fintech_payments", "wallets": "fintech_wallets", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_profiles": True, "enable_documents": True, "enable_screening": True, "enable_risk": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "fintech_kyc_control", "allow_tenant_overrides": True},
}

PROVIDES = ["customer_identity_lifecycle", "document_verification_workflow", "sanctions_pep_screening", "kyc_risk_scoring", "customer_due_diligence", "enhanced_due_diligence", "kyc_agent_workflow"]
REQUIRES = ["auth", "audl", "cons", "ntfy", "biop", "cvsn", "nlpc", "keym", "fintech_payments", "fintech_wallets"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-kyc/dashboard", "component": "KycDashboard", "permission": "fintech_kyc:view", "nav_group": "Overview"},
	{"name": "profiles", "path": "/fintech-kyc/profiles", "component": "KycProfileWorkbench", "permission": "fintech_kyc:manage_profiles", "nav_group": "Profiles"},
	{"name": "documents", "path": "/fintech-kyc/documents", "component": "KycDocumentVault", "permission": "fintech_kyc:manage_documents", "nav_group": "Evidence"},
	{"name": "screening", "path": "/fintech-kyc/screening", "component": "KycScreeningConsole", "permission": "fintech_kyc:screen", "nav_group": "Screening"},
	{"name": "risk", "path": "/fintech-kyc/risk", "component": "KycRiskConsole", "permission": "fintech_kyc:review_risk", "nav_group": "Risk"},
	{"name": "reviews", "path": "/fintech-kyc/reviews", "component": "KycReviewQueue", "permission": "fintech_kyc:review", "nav_group": "Reviews"},
	{"name": "agents", "path": "/fintech-kyc/agents", "component": "KycAgentWorkbench", "permission": "fintech_kyc:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-kyc/settings", "component": "KycSettings", "permission": "fintech_kyc:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_kyc_control",
	"tokens": {"color.primary": "#24405F", "color.accent": "#0E7490", "color.success": "#15803D", "color.warning": "#A16207", "color.danger": "#B42318", "surface.canvas": "#F7F9FC", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"profiles": {"icon": "user-check", "status_indicator": "kyc-status-pill"}, "documents": {"icon": "file-check", "status_indicator": "confidence-chip"}, "screening": {"visual": "watchlist-lane", "status_style": "screening-chip"}, "risk": {"visual": "risk-band", "status_style": "risk-chip"}, "reviews": {"visual": "review-queue", "status_style": "decision-chip"}, "agents": {"visual": "agent-lane", "status_style": "agent-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": KYC_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["kyc_profile_opened", "kyc_document_registered", "kyc_screening_recorded", "kyc_risk_scored", "kyc_decision_recorded", "kyc_agent_registered"],
	"guardrails": ["kyc_batch_requires_bytewax", "kyc_event_requires_bytewax", "privileged_kyc_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "KYC operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "kyc_write_requires_policy", "description": "KYC writes require policy evidence.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "kyc_policy_required", "required_action": "attach_kyc_policy"}},
	{"name": "profile_subject_required", "description": "KYC profiles require subject reference.", "condition": {"operation": "open_profile", "subject_present": False}, "effect": {"decision": "deny", "reason": "kyc_subject_required", "required_action": "attach_subject_reference"}},
	{"name": "profile_legal_name_required", "description": "KYC profiles require legal name.", "condition": {"operation": "open_profile", "legal_name_present": False}, "effect": {"decision": "deny", "reason": "legal_name_required", "required_action": "capture_legal_name"}},
	{"name": "profile_customer_type_supported", "description": "KYC customer type must be supported.", "condition": {"operation": "open_profile", "customer_type_supported": False}, "effect": {"decision": "deny", "reason": "customer_type_not_supported", "required_action": "select_supported_customer_type"}},
	{"name": "profile_country_required", "description": "KYC profiles require country code.", "condition": {"operation": "open_profile", "country_present": False}, "effect": {"decision": "deny", "reason": "country_required", "required_action": "capture_country_code"}},
	{"name": "profile_consent_required", "description": "KYC profiles require consent evidence.", "condition": {"operation": "open_profile", "consent_recorded": False}, "effect": {"decision": "deny", "reason": "customer_consent_required", "required_action": "record_customer_consent"}},
	{"name": "document_profile_required", "description": "Documents require an existing KYC profile.", "condition": {"operation": "register_document", "profile_present": False}, "effect": {"decision": "deny", "reason": "kyc_profile_required", "required_action": "select_profile"}},
	{"name": "document_type_supported", "description": "Document type must be supported.", "condition": {"operation": "register_document", "document_type_supported": False}, "effect": {"decision": "deny", "reason": "document_type_not_supported", "required_action": "select_supported_document_type"}},
	{"name": "document_token_required", "description": "Documents require tokenized storage reference.", "condition": {"operation": "register_document", "token_reference_present": False}, "effect": {"decision": "deny", "reason": "document_token_required", "required_action": "attach_document_token"}},
	{"name": "document_subject_required", "description": "Documents require extracted subject evidence.", "condition": {"operation": "register_document", "extracted_subject_present": False}, "effect": {"decision": "deny", "reason": "document_subject_required", "required_action": "extract_document_subject"}},
	{"name": "document_confidence_minimum", "description": "Document confidence must meet the configured minimum.", "condition": {"operation": "register_document", "confidence_below_minimum": True}, "effect": {"decision": "deny", "reason": "document_confidence_below_minimum", "required_action": "reverify_document"}},
	{"name": "screening_profile_required", "description": "Screening requires an existing KYC profile.", "condition": {"operation": "record_screening", "profile_present": False}, "effect": {"decision": "deny", "reason": "kyc_profile_required", "required_action": "select_profile"}},
	{"name": "screening_hits_require_review", "description": "Sanctions, PEP, watchlist, or adverse-media hits require review.", "condition": {"operation": "record_screening", "screening_hit": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "screening_review_required", "required_action": "record_screening_review"}},
	{"name": "risk_score_range", "description": "KYC risk score must be between 0 and 100.", "condition": {"operation": "score_risk", "risk_score_out_of_range": True}, "effect": {"decision": "deny", "reason": "risk_score_out_of_range", "required_action": "set_valid_risk_score"}},
	{"name": "high_risk_requires_edd", "description": "High KYC risk requires enhanced due diligence review.", "condition": {"operation": "score_risk", "high_risk": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "enhanced_due_diligence_required", "required_action": "record_edd_review"}},
	{"name": "decision_identity_document_required", "description": "Verification requires identity document evidence.", "condition": {"operation": "record_decision", "identity_document_present": False}, "effect": {"decision": "deny", "reason": "identity_document_required", "required_action": "attach_identity_document"}},
	{"name": "decision_address_document_required", "description": "Verification requires address document evidence.", "condition": {"operation": "record_decision", "address_document_present": False}, "effect": {"decision": "deny", "reason": "address_document_required", "required_action": "attach_address_document"}},
	{"name": "decision_screening_required", "description": "Verification requires screening evidence.", "condition": {"operation": "record_decision", "screening_present": False}, "effect": {"decision": "deny", "reason": "screening_required", "required_action": "record_screening"}},
	{"name": "decision_risk_required", "description": "Verification requires risk score evidence.", "condition": {"operation": "record_decision", "risk_present": False}, "effect": {"decision": "deny", "reason": "risk_assessment_required", "required_action": "score_risk"}},
	{"name": "decision_blocks_open_reviews", "description": "Verification cannot proceed with unresolved review flags.", "condition": {"operation": "record_decision", "open_review_flags": True}, "effect": {"decision": "deny", "reason": "open_review_flags", "required_action": "resolve_reviews"}},
	{"name": "kyc_batch_requires_bytewax", "description": "KYC batches require Bytewax.", "condition": {"operation": "kyc_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_kyc_batch_to_bytewax"}},
	{"name": "kyc_event_requires_bytewax", "description": "KYC events require Bytewax.", "condition": {"operation": "kyc_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_kyc_event_to_bytewax"}},
	{"name": "kyc_agent_runtime_supported", "description": "KYC agents must use a supported runtime.", "condition": {"operation": "register_kyc_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "kyc_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "kyc_agent_role_supported", "description": "KYC agents must use a supported role.", "condition": {"operation": "register_kyc_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "kyc_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_kyc_agent_action_requires_human_approval", "description": "Privileged KYC-agent actions require human approval.", "condition": {"operation": "kyc_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},

	# Cross-tenant and privilege escalation guards
	{"name": "cross_tenant_kyc_access_denied", "description": "KYC data cannot be accessed across tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_credentials"}},
	{"name": "privilege_escalation_denied", "description": "KYC privilege escalation without approval is denied.", "condition": {"privilege_escalation_attempt": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "obtain_escalation_approval"}},

	# Africa-specific KYC rules
	{"name": "ke_national_id_required_for_individuals", "description": "Kenyan individuals require a valid National ID or Huduma Namba.", "condition": {"operation": "open_profile", "customer_type": "individual", "country": "KE", "national_id_present": False}, "effect": {"decision": "deny", "reason": "ke_national_id_required", "required_action": "capture_national_id_or_huduma_namba"}},
	{"name": "ng_bvn_required_for_individuals", "description": "Nigerian individuals require a Bank Verification Number (BVN).", "condition": {"operation": "open_profile", "customer_type": "individual", "country": "NG", "bvn_present": False}, "effect": {"decision": "deny", "reason": "ng_bvn_required", "required_action": "capture_bvn"}},
	{"name": "gh_ghana_card_required", "description": "Ghanaian individuals require a Ghana Card (national ID).", "condition": {"operation": "open_profile", "customer_type": "individual", "country": "GH", "ghana_card_present": False}, "effect": {"decision": "deny", "reason": "gh_ghana_card_required", "required_action": "capture_ghana_card"}},
	{"name": "za_fica_compliance_required", "description": "South African customers require FICA-compliant identity verification.", "condition": {"operation": "open_profile", "country": "ZA", "fica_compliant": False}, "effect": {"decision": "deny", "reason": "za_fica_compliance_required", "required_action": "complete_fica_verification"}},
	{"name": "mobile_money_simplified_kyc_threshold", "description": "Mobile money simplified KYC is limited to CBK tier thresholds.", "condition": {"operation": "open_profile", "kyc_tier": "simplified", "exceeds_simplified_kyc_threshold": True}, "effect": {"decision": "deny", "reason": "simplified_kyc_threshold_exceeded", "required_action": "upgrade_to_full_kyc"}},
	{"name": "ke_cbk_kyc_tier_required", "description": "Kenya CBK requires tiered KYC for mobile money customers.", "condition": {"operation": "open_profile", "customer_type": "individual", "country": "KE", "kyc_tier_assigned": False}, "effect": {"decision": "deny", "reason": "cbk_kyc_tier_required", "required_action": "assign_cbk_kyc_tier"}},
	{"name": "pep_enhanced_due_diligence_required", "description": "Politically exposed persons require enhanced due diligence.", "condition": {"operation": "record_decision", "pep_identified": True, "edd_completed": False}, "effect": {"decision": "deny", "reason": "pep_edd_required", "required_action": "complete_pep_enhanced_due_diligence"}},
	{"name": "kyc_expiry_review_required", "description": "Expired KYC profiles require re-verification before transacting.", "condition": {"operation": "record_decision", "kyc_expired": True}, "effect": {"decision": "deny", "reason": "kyc_profile_expired", "required_action": "reverify_customer_kyc"}},
	{"name": "business_beneficial_owner_required", "description": "Business KYC requires beneficial owner identification (>25% ownership).", "condition": {"operation": "open_profile", "customer_type": "business", "beneficial_owner_present": False}, "effect": {"decision": "deny", "reason": "beneficial_owner_required", "required_action": "capture_beneficial_owner"}},
	{"name": "kyc_data_quality_phone_required", "description": "KYC profiles require a verified phone number for mobile money.", "condition": {"operation": "open_profile", "mobile_money_enabled": True, "phone_verified": False}, "effect": {"decision": "deny", "reason": "verified_phone_required", "required_action": "verify_phone_number"}},
]


def _configuration_schema() -> dict[str, Any]:
	return {"type": "object", "required": list(DEFAULT_CONFIGURATION), "properties": {key: {"type": "object"} for key in DEFAULT_CONFIGURATION if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}


def _matches_condition(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	if overrides:
		for key, value in overrides.items():
			if isinstance(value, dict) and isinstance(configuration.get(key), dict):
				configuration[key].update(value)
			else:
				configuration[key] = value
	return {"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "configuration": configuration, "configuration_schema": _configuration_schema(), "provides": PROVIDES, "requires": REQUIRES, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-kyc/api/v1", "routes": deepcopy(UI_ROUTES), "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"]}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	contract = get_capability_contract(str(context.get("tenant_id") or "default"))
	matched = [rule for rule in contract["rule_engine"]["rules"] if _matches_condition(rule["condition"], context)]
	decision = "allow"
	for rule in matched:
		effect = rule["effect"]["decision"]
		if effect == "deny":
			decision = "deny"
			break
		if effect == "require_review" and decision == "allow":
			decision = "require_review"
	return {"decision": decision, "matched_rules": [rule["name"] for rule in matched], "actions": [rule["effect"] for rule in matched], "effects": [rule["effect"] for rule in matched]}
