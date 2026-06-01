"""Executable capability contract for APG InsurTech."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_insurance"
CAPABILITY_NAME = "InsurTech"
CAPABILITY_VERSION = "1.1.0"
INSURANCE_EVENT_STREAM = "apg.fintech.insurance.lifecycle"

SUPPORTED_PRODUCT_LINES = ["life", "health", "property", "motor", "travel", "crop", "microinsurance"]
SUPPORTED_CURRENCIES = ["USD", "KES", "EUR", "GBP", "NGN", "GHS", "ZAR"]
SUPPORTED_CLAIM_TYPES = ["medical", "accident", "theft", "damage", "death", "delay", "weather"]
SUPPORTED_DOCUMENT_TYPES = ["policy_schedule", "proof_of_loss", "medical_report", "identity", "invoice", "photo_evidence"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_ALERT_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["underwriting_reviewer", "quote_reviewer", "claim_triage_reviewer", "fraud_review_agent", "reinsurance_reviewer", "insurance_compliance_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"policyholders": {"kyc_required": True, "contact_required": True, "risk_profile_required": True},
	"products": {"supported_lines": SUPPORTED_PRODUCT_LINES, "coverage_terms_required": True, "pricing_reference_required": True},
	"quotes": {"policyholder_required": True, "product_required": True, "positive_premium_required": True, "underwriting_reference_required": True},
	"policies": {"quote_required": True, "effective_date_required": True, "payment_reference_required": True},
	"premiums": {"policy_required": True, "positive_amount_required": True, "supported_currencies": SUPPORTED_CURRENCIES, "payment_reference_required": True},
	"claims": {"policy_required": True, "supported_types": SUPPORTED_CLAIM_TYPES, "positive_amount_required": True, "loss_date_required": True, "evidence_required": True},
	"documents": {"supported_types": SUPPORTED_DOCUMENT_TYPES, "reference_required": True, "evidence_required": True},
	"risk_assessments": {"policyholder_required": True, "score_required": True, "source_required": True},
	"reinsurance": {"policy_required": True, "treaty_reference_required": True, "positive_share_required": True},
	"compliance": {"supported_severities": SUPPORTED_ALERT_SEVERITIES, "evidence_required": True, "review_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True},
	"observability": {"event_stream": INSURANCE_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "payments": "fintech_payments", "wallets": "fintech_wallets", "kyc": "fintech_kyc", "aml": "fintech_aml", "fraud": "fintech_fraud", "analytics": "bia", "reporting": "fin_rpt", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_policyholders": True, "enable_products": True, "enable_quotes": True, "enable_policies": True, "enable_premiums": True, "enable_claims": True, "enable_documents": True, "enable_risk": True, "enable_reinsurance": True, "enable_compliance": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "insurtech_control", "allow_tenant_overrides": True},
}

PROVIDES = ["insurance_policyholder_workflow", "insurance_product_workflow", "insurance_quote_workflow", "insurance_policy_workflow", "insurance_premium_workflow", "insurance_claim_workflow", "insurance_document_workflow", "insurance_risk_workflow", "insurance_reinsurance_workflow", "insurance_compliance_workflow", "insurance_review_workflow", "insurance_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_payments", "fintech_wallets", "fintech_kyc", "fintech_aml", "fintech_fraud", "bia", "fin_rpt"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-insurance/dashboard", "component": "InsuranceDashboard", "permission": "fintech_insurance:view", "nav_group": "Overview"},
	{"name": "policyholders", "path": "/fintech-insurance/policyholders", "component": "PolicyholderConsole", "permission": "fintech_insurance:policyholders", "nav_group": "Customers"},
	{"name": "products", "path": "/fintech-insurance/products", "component": "InsuranceProductConsole", "permission": "fintech_insurance:products", "nav_group": "Products"},
	{"name": "quotes", "path": "/fintech-insurance/quotes", "component": "QuoteWorkbench", "permission": "fintech_insurance:quotes", "nav_group": "Underwriting"},
	{"name": "policies", "path": "/fintech-insurance/policies", "component": "PolicyConsole", "permission": "fintech_insurance:policies", "nav_group": "Policies"},
	{"name": "premiums", "path": "/fintech-insurance/premiums", "component": "PremiumLedger", "permission": "fintech_insurance:premiums", "nav_group": "Policies"},
	{"name": "claims", "path": "/fintech-insurance/claims", "component": "ClaimWorkbench", "permission": "fintech_insurance:claims", "nav_group": "Claims"},
	{"name": "documents", "path": "/fintech-insurance/documents", "component": "InsuranceDocumentConsole", "permission": "fintech_insurance:documents", "nav_group": "Claims"},
	{"name": "risk", "path": "/fintech-insurance/risk", "component": "InsuranceRiskConsole", "permission": "fintech_insurance:risk", "nav_group": "Risk"},
	{"name": "reinsurance", "path": "/fintech-insurance/reinsurance", "component": "ReinsuranceConsole", "permission": "fintech_insurance:reinsurance", "nav_group": "Risk"},
	{"name": "compliance", "path": "/fintech-insurance/compliance", "component": "InsuranceComplianceConsole", "permission": "fintech_insurance:compliance", "nav_group": "Governance"},
	{"name": "reviews", "path": "/fintech-insurance/reviews", "component": "InsuranceReviewConsole", "permission": "fintech_insurance:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/fintech-insurance/agents", "component": "InsuranceAgentWorkbench", "permission": "fintech_insurance:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-insurance/settings", "component": "InsuranceSettings", "permission": "fintech_insurance:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "insurtech_control",
	"tokens": {"color.primary": "#0369A1", "color.accent": "#059669", "color.success": "#15803D", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"policyholders": {"icon": "users", "status_indicator": "policyholder-chip"}, "products": {"icon": "package-check", "status_indicator": "product-chip"}, "quotes": {"icon": "file-signature", "status_indicator": "quote-chip"}, "policies": {"icon": "shield-check", "status_indicator": "policy-chip"}, "premiums": {"icon": "receipt", "status_indicator": "premium-chip"}, "claims": {"icon": "file-warning", "status_indicator": "claim-chip"}, "documents": {"icon": "file-stack", "status_indicator": "document-chip"}, "risk": {"icon": "activity", "status_indicator": "risk-chip"}, "reinsurance": {"icon": "layers-3", "status_indicator": "reinsurance-chip"}, "compliance": {"icon": "scale", "status_indicator": "alert-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": INSURANCE_EVENT_STREAM, "key": "tenant_id", "events": ["policyholder_onboarded", "insurance_product_published", "quote_generated", "policy_bound", "premium_recorded", "claim_opened", "document_recorded", "risk_assessment_recorded", "reinsurance_attachment_recorded", "insurance_compliance_alert_recorded", "insurance_review_recorded", "insurance_agent_registered"], "guardrails": ["insurance_batch_requires_bytewax", "privileged_insurance_agent_action_requires_human_approval"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "insurance_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_evidence_required", "required_action": "attach_policy_evidence"}},
	{"name": "policyholder_kyc_required", "condition": {"operation": "onboard_policyholder", "kyc_present": False}, "effect": {"decision": "deny", "reason": "policyholder_kyc_required", "required_action": "attach_kyc"}},
	{"name": "policyholder_contact_required", "condition": {"operation": "onboard_policyholder", "contact_present": False}, "effect": {"decision": "deny", "reason": "policyholder_contact_required", "required_action": "attach_contact"}},
	{"name": "product_line_supported", "condition": {"operation": "publish_product", "product_line_supported": False}, "effect": {"decision": "deny", "reason": "product_line_not_supported", "required_action": "select_supported_product_line"}},
	{"name": "product_coverage_required", "condition": {"operation": "publish_product", "coverage_terms_present": False}, "effect": {"decision": "deny", "reason": "coverage_terms_required", "required_action": "attach_coverage_terms"}},
	{"name": "quote_policyholder_required", "condition": {"operation": "generate_quote", "policyholder_present": False}, "effect": {"decision": "deny", "reason": "quote_policyholder_required", "required_action": "select_policyholder"}},
	{"name": "quote_product_required", "condition": {"operation": "generate_quote", "product_present": False}, "effect": {"decision": "deny", "reason": "quote_product_required", "required_action": "select_product"}},
	{"name": "quote_positive_premium", "condition": {"operation": "generate_quote", "positive_premium": False}, "effect": {"decision": "deny", "reason": "positive_quote_premium_required", "required_action": "set_positive_premium"}},
	{"name": "quote_underwriting_required", "condition": {"operation": "generate_quote", "underwriting_reference_present": False}, "effect": {"decision": "deny", "reason": "quote_underwriting_reference_required", "required_action": "attach_underwriting_reference"}},
	{"name": "policy_quote_required", "condition": {"operation": "bind_policy", "quote_present": False}, "effect": {"decision": "deny", "reason": "policy_quote_required", "required_action": "select_quote"}},
	{"name": "policy_payment_required", "condition": {"operation": "bind_policy", "payment_reference_present": False}, "effect": {"decision": "deny", "reason": "policy_payment_reference_required", "required_action": "attach_payment_reference"}},
	{"name": "premium_policy_required", "condition": {"operation": "record_premium", "policy_present": False}, "effect": {"decision": "deny", "reason": "premium_policy_required", "required_action": "select_policy"}},
	{"name": "premium_positive_amount", "condition": {"operation": "record_premium", "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_premium_amount_required", "required_action": "set_positive_amount"}},
	{"name": "premium_currency_supported", "condition": {"operation": "record_premium", "currency_supported": False}, "effect": {"decision": "deny", "reason": "premium_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "premium_payment_required", "condition": {"operation": "record_premium", "payment_reference_present": False}, "effect": {"decision": "deny", "reason": "premium_payment_reference_required", "required_action": "attach_payment_reference"}},
	{"name": "claim_policy_required", "condition": {"operation": "open_claim", "policy_present": False}, "effect": {"decision": "deny", "reason": "claim_policy_required", "required_action": "select_policy"}},
	{"name": "claim_type_supported", "condition": {"operation": "open_claim", "claim_type_supported": False}, "effect": {"decision": "deny", "reason": "claim_type_not_supported", "required_action": "select_supported_claim_type"}},
	{"name": "claim_positive_amount", "condition": {"operation": "open_claim", "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_claim_amount_required", "required_action": "set_positive_claim_amount"}},
	{"name": "claim_evidence_required", "condition": {"operation": "open_claim", "evidence_present": False}, "effect": {"decision": "deny", "reason": "claim_evidence_required", "required_action": "attach_claim_evidence"}},
	{"name": "document_type_supported", "condition": {"operation": "record_document", "document_type_supported": False}, "effect": {"decision": "deny", "reason": "document_type_not_supported", "required_action": "select_supported_document_type"}},
	{"name": "document_evidence_required", "condition": {"operation": "record_document", "evidence_present": False}, "effect": {"decision": "deny", "reason": "document_evidence_required", "required_action": "attach_document_evidence"}},
	{"name": "risk_policyholder_required", "condition": {"operation": "record_risk_assessment", "policyholder_present": False}, "effect": {"decision": "deny", "reason": "risk_policyholder_required", "required_action": "select_policyholder"}},
	{"name": "risk_score_required", "condition": {"operation": "record_risk_assessment", "score_present": False}, "effect": {"decision": "deny", "reason": "risk_score_required", "required_action": "set_risk_score"}},
	{"name": "risk_source_required", "condition": {"operation": "record_risk_assessment", "source_present": False}, "effect": {"decision": "deny", "reason": "risk_source_required", "required_action": "attach_risk_source"}},
	{"name": "reinsurance_policy_required", "condition": {"operation": "record_reinsurance_attachment", "policy_present": False}, "effect": {"decision": "deny", "reason": "reinsurance_policy_required", "required_action": "select_policy"}},
	{"name": "reinsurance_treaty_required", "condition": {"operation": "record_reinsurance_attachment", "treaty_reference_present": False}, "effect": {"decision": "deny", "reason": "reinsurance_treaty_required", "required_action": "attach_treaty_reference"}},
	{"name": "reinsurance_positive_share", "condition": {"operation": "record_reinsurance_attachment", "positive_share": False}, "effect": {"decision": "deny", "reason": "positive_reinsurance_share_required", "required_action": "set_positive_reinsurance_share"}},
	{"name": "compliance_severity_supported", "condition": {"operation": "record_compliance_alert", "severity_supported": False}, "effect": {"decision": "deny", "reason": "compliance_severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "compliance_evidence_required", "condition": {"operation": "record_compliance_alert", "evidence_present": False}, "effect": {"decision": "deny", "reason": "compliance_evidence_required", "required_action": "attach_compliance_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "insurance_batch_requires_bytewax", "condition": {"operation": "insurance_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_insurance_batch_to_bytewax"}},
	{"name": "insurance_agent_runtime_supported", "condition": {"operation": "register_insurance_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "insurance_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "insurance_agent_role_supported", "condition": {"operation": "register_insurance_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "insurance_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_insurance_agent_action_requires_human_approval", "condition": {"operation": "insurance_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "requires_theme": True, "api_prefix": "/fintech-insurance/api/v1", "template_roots": ["templates/", "static/"], "view_module": "views.py", "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
