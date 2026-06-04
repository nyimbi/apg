"""Executable APG capability contract for SCM Vendor Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "scm_ven"
CAPABILITY_NAME = "Vendor Management"
CAPABILITY_VERSION = "2.1.0"
VENDOR_EVENT_STREAM = "apg.scm.ven.lifecycle"

SUPPORTED_VENDOR_TYPES = ["manufacturer", "distributor", "service_provider", "contractor", "consultant", "logistics", "technology", "financial", "public_sector", "nonprofit"]
SUPPORTED_LIFECYCLE_STAGES = ["prospect", "onboarding", "qualified", "active", "suspended", "offboarding", "archived"]
SUPPORTED_RISK_TIERS = ["low", "medium", "high", "critical"]
SUPPORTED_COMPLIANCE_STATUSES = ["pending", "compliant", "review_required", "non_compliant", "expired"]
SUPPORTED_PERFORMANCE_DIMENSIONS = ["quality", "delivery", "cost", "service", "sustainability", "innovation"]
SUPPORTED_VENDOR_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_VENDOR_AGENT_ROLES = [
	"vendor_onboarding_reviewer",
	"risk_reviewer",
	"performance_reviewer",
	"compliance_reviewer",
	"contract_reviewer",
	"supplier_query_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"vendors": {
		"code_required": True,
		"name_required": True,
		"supported_types": SUPPORTED_VENDOR_TYPES,
		"category_required": True,
		"country_required": True,
		"owner_required": True,
	},
	"qualification": {
		"criteria_required": True,
		"qualified_by_required": True,
		"minimum_score": 70,
	},
	"onboarding": {
		"vendor_required": True,
		"checklist_required": True,
		"owner_required": True,
	},
	"performance": {
		"period_required": True,
		"supported_dimensions": SUPPORTED_PERFORMANCE_DIMENSIONS,
		"score_range": [0, 100],
		"review_required_below": 60,
	},
	"risk": {
		"supported_tiers": SUPPORTED_RISK_TIERS,
		"owner_required_for_high_or_critical": True,
	},
	"compliance": {
		"supported_statuses": SUPPORTED_COMPLIANCE_STATUSES,
		"evidence_required": True,
		"review_required_for_noncompliance": True,
	},
	"contracts": {
		"value_required": True,
		"currency_required": True,
		"date_range_required": True,
		"approval_required": True,
	},
	"communications": {
		"channel_required": True,
		"subject_required": True,
		"owner_required_for_negative_sentiment": True,
	},
	"portal": {
		"email_required": True,
		"role_required": True,
		"approval_required": True,
	},
	"scorecards": {
		"performance_required": True,
		"risk_required": True,
		"generator_required": True,
	},
	"vendor_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_VENDOR_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_VENDOR_AGENT_ROLES,
		"max_autonomous_scope": "inspect_prepare_and_recommend",
		"human_approval_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_state_changes": True,
		"segregation_of_duties": True,
	},
	"observability": {
		"event_stream": VENDOR_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_vendor_events": True,
		"emit_performance_events": True,
		"emit_risk_events": True,
		"emit_compliance_events": True,
		"emit_contract_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"notification": "adapter",
		"workflow": "adapter",
		"procurement": "adapter",
		"contracts": "adapter",
		"documents": "adapter",
		"risk_policy": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_vendors": True,
		"enable_qualification": True,
		"enable_onboarding": True,
		"enable_performance": True,
		"enable_risk": True,
		"enable_compliance": True,
		"enable_contracts": True,
		"enable_portal": True,
		"enable_scorecards": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "vendor_control", "allow_tenant_overrides": True},
}


PROVIDES = [
	"vendor_profile_lifecycle",
	"vendor_onboarding_workflow",
	"vendor_qualification_lifecycle",
	"vendor_performance_lifecycle",
	"vendor_risk_lifecycle",
	"vendor_contract_lifecycle",
	"vendor_compliance_lifecycle",
	"vendor_communication_lifecycle",
	"vendor_portal_lifecycle",
	"vendor_scorecard_service",
	"vendor_sourcing_integration",
	"vendor_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"composition_events",
	"composition_config",
	"wflo",
	"grc_doc",
	"grc_doc",
	"grc_rsa",
	"mdm",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/scm/vendors/dashboard", "component": "VendorDashboard", "permission": "scm_ven:view", "nav_group": "Overview"},
	{"name": "vendors", "path": "/scm/vendors", "component": "VendorWorkbench", "permission": "scm_ven:manage_vendors", "nav_group": "Master Data"},
	{"name": "qualification", "path": "/scm/vendors/qualification", "component": "VendorQualificationDesk", "permission": "scm_ven:qualify", "nav_group": "Lifecycle"},
	{"name": "onboarding", "path": "/scm/vendors/onboarding", "component": "VendorOnboardingBoard", "permission": "scm_ven:onboard", "nav_group": "Lifecycle"},
	{"name": "performance", "path": "/scm/vendors/performance", "component": "VendorPerformanceWorkbench", "permission": "scm_ven:score", "nav_group": "Performance"},
	{"name": "risk", "path": "/scm/vendors/risk", "component": "VendorRiskCenter", "permission": "scm_ven:govern", "nav_group": "Governance"},
	{"name": "compliance", "path": "/scm/vendors/compliance", "component": "VendorComplianceDesk", "permission": "scm_ven:govern", "nav_group": "Governance"},
	{"name": "contracts", "path": "/scm/vendors/contracts", "component": "VendorContractDesk", "permission": "scm_ven:contract", "nav_group": "Commercial"},
	{"name": "communications", "path": "/scm/vendors/communications", "component": "VendorCommunicationLog", "permission": "scm_ven:communicate", "nav_group": "Engagement"},
	{"name": "portal", "path": "/scm/vendors/portal", "component": "VendorPortalAdmin", "permission": "scm_ven:portal", "nav_group": "Engagement"},
	{"name": "scorecards", "path": "/scm/vendors/scorecards", "component": "VendorScorecardDesk", "permission": "scm_ven:score", "nav_group": "Performance"},
	{"name": "agents", "path": "/scm/vendors/agents", "component": "VendorAgentWorkbench", "permission": "scm_ven:agent_manage", "nav_group": "Automation"},
	{"name": "rules", "path": "/scm/vendors/rules", "component": "VendorRules", "permission": "scm_ven:govern", "nav_group": "Governance"},
	{"name": "settings", "path": "/scm/vendors/settings", "component": "VendorSettings", "permission": "scm_ven:admin", "nav_group": "Administration"},
]


THEME = {
	"name": "vendor_control",
	"tokens": {
		"border.radius": "8px",
		"color.primary": "#234E52",
		"color.accent": "#8A5A20",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"density": "compact",
	},
	"components": {
		"dashboard": {"icon": "layout-dashboard", "status_indicator": "health-pill", "visual": "supplier-grid"},
		"vendors": {"icon": "building-2", "status_style": "vendor-chip", "visual": "vendor-table"},
		"qualification": {"icon": "badge-check", "status_style": "qualification-chip", "visual": "criteria-list"},
		"onboarding": {"icon": "route", "status_style": "stage-chip", "visual": "onboarding-board"},
		"performance": {"icon": "activity", "status_style": "score-chip", "visual": "score-table"},
		"risk": {"icon": "shield-alert", "status_style": "risk-chip", "visual": "risk-register"},
		"compliance": {"icon": "clipboard-check", "status_style": "compliance-chip", "visual": "evidence-ledger"},
		"contracts": {"icon": "file-signature", "status_style": "contract-chip", "visual": "contract-list"},
		"communications": {"icon": "message-square", "status_style": "sentiment-chip", "visual": "communication-log"},
		"portal": {"icon": "door-open", "status_style": "portal-chip", "visual": "portal-users"},
		"scorecards": {"icon": "gauge", "status_style": "scorecard-chip", "visual": "scorecard-list"},
		"agents": {"icon": "bot", "status_style": "agent-chip", "visual": "agent-roster"},
		"rules": {"icon": "list-checks", "status_style": "decision-chip", "visual": "rule-list"},
		"settings": {"icon": "settings", "density": "compact", "visual": "settings-panel"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"event_stream": VENDOR_EVENT_STREAM,
	"events": [
		"vendor_created",
		"vendor_qualified",
		"vendor_onboarded",
		"vendor_performance_recorded",
		"vendor_risk_recorded",
		"vendor_compliance_recorded",
		"vendor_contract_created",
		"vendor_communication_recorded",
		"vendor_portal_user_created",
		"vendor_scorecard_created",
		"vendor_agent_registered",
	],
	"delivery": "at_least_once",
	"ordering_key": "tenant_id",
}


def _rule(name: str, description: str, condition: dict[str, Any], decision: str, reason: str, action: str) -> dict[str, Any]:
	return {"name": name, "description": description, "condition": condition, "effect": {"decision": decision, "reason": reason, "required_action": action}}


RULES = [
	_rule("tenant_context_required", "Vendor operations require tenant context.", {"tenant_context_present": False}, "deny", "tenant_context_required", "attach_tenant_context"),
	_rule("operation_policy_required", "Vendor write operations require policy enforcement.", {"operation_type": "write", "policy_attached": False}, "deny", "operation_policy_required", "attach_operation_policy"),
	_rule("vendor_code_required", "Vendors require a code.", {"operation": "create_vendor", "code_present": False}, "deny", "vendor_code_required", "provide_vendor_code"),
	_rule("vendor_name_required", "Vendors require a name.", {"operation": "create_vendor", "name_present": False}, "deny", "vendor_name_required", "provide_vendor_name"),
	_rule("vendor_type_supported", "Vendors must use a supported type.", {"operation": "create_vendor", "vendor_type_supported": False}, "deny", "vendor_type_not_supported", "choose_supported_vendor_type"),
	_rule("vendor_category_required", "Vendors require a category.", {"operation": "create_vendor", "category_present": False}, "deny", "vendor_category_required", "provide_category"),
	_rule("vendor_country_required", "Vendors require a country.", {"operation": "create_vendor", "country_present": False}, "deny", "vendor_country_required", "provide_country"),
	_rule("vendor_owner_required", "Vendors require an owner.", {"operation": "create_vendor", "owner_present": False}, "deny", "vendor_owner_required", "assign_owner"),
	_rule("qualification_vendor_required", "Qualification requires an existing vendor.", {"operation": "qualify_vendor", "vendor_present": False}, "deny", "qualification_vendor_required", "select_vendor"),
	_rule("qualification_criteria_required", "Qualification requires criteria.", {"operation": "qualify_vendor", "criteria_present": False}, "deny", "qualification_criteria_required", "attach_criteria"),
	_rule("qualification_actor_required", "Qualification requires a reviewer.", {"operation": "qualify_vendor", "qualified_by_present": False}, "deny", "qualification_reviewer_required", "record_reviewer"),
	_rule("qualification_score_required", "Qualification requires a score.", {"operation": "qualify_vendor", "score_present": False}, "deny", "qualification_score_required", "provide_score"),
	_rule("qualification_score_threshold", "Low qualification scores require review.", {"operation": "qualify_vendor", "score_below_threshold": True, "review_recorded": False}, "require_review", "qualification_review_required", "record_qualification_review"),
	_rule("onboarding_vendor_required", "Onboarding requires an existing vendor.", {"operation": "onboard_vendor", "vendor_present": False}, "deny", "onboarding_vendor_required", "select_vendor"),
	_rule("onboarding_checklist_required", "Onboarding requires checklist evidence.", {"operation": "onboard_vendor", "checklist_present": False}, "deny", "onboarding_checklist_required", "attach_checklist"),
	_rule("onboarding_owner_required", "Onboarding requires an owner.", {"operation": "onboard_vendor", "owner_present": False}, "deny", "onboarding_owner_required", "assign_owner"),
	_rule("performance_vendor_required", "Performance records require an existing vendor.", {"operation": "record_performance", "vendor_present": False}, "deny", "performance_vendor_required", "select_vendor"),
	_rule("performance_period_required", "Performance records require a period.", {"operation": "record_performance", "period_present": False}, "deny", "performance_period_required", "provide_period"),
	_rule("performance_dimensions_supported", "Performance dimensions must be supported.", {"operation": "record_performance", "dimensions_supported": False}, "deny", "performance_dimension_not_supported", "choose_supported_dimensions"),
	_rule("performance_scores_in_range", "Performance scores must be between 0 and 100.", {"operation": "record_performance", "scores_in_range": False}, "deny", "performance_score_out_of_range", "correct_scores"),
	_rule("performance_low_score_review", "Low performance scores require review.", {"operation": "record_performance", "low_score": True, "review_recorded": False}, "require_review", "performance_review_required", "record_performance_review"),
	_rule("risk_vendor_required", "Risk records require an existing vendor.", {"operation": "record_risk", "vendor_present": False}, "deny", "risk_vendor_required", "select_vendor"),
	_rule("risk_tier_supported", "Risk tier must be supported.", {"operation": "record_risk", "risk_tier_supported": False}, "deny", "risk_tier_not_supported", "choose_supported_risk_tier"),
	_rule("risk_description_required", "Risk records require description.", {"operation": "record_risk", "description_present": False}, "deny", "risk_description_required", "provide_description"),
	_rule("risk_owner_required", "High or critical risks require an owner.", {"operation": "record_risk", "high_or_critical": True, "owner_present": False}, "deny", "risk_owner_required", "assign_risk_owner"),
	_rule("compliance_vendor_required", "Compliance records require an existing vendor.", {"operation": "record_compliance", "vendor_present": False}, "deny", "compliance_vendor_required", "select_vendor"),
	_rule("compliance_framework_required", "Compliance records require framework.", {"operation": "record_compliance", "framework_present": False}, "deny", "compliance_framework_required", "provide_framework"),
	_rule("compliance_status_supported", "Compliance status must be supported.", {"operation": "record_compliance", "status_supported": False}, "deny", "compliance_status_not_supported", "choose_supported_status"),
	_rule("compliance_evidence_required", "Compliance records require evidence.", {"operation": "record_compliance", "evidence_present": False}, "deny", "compliance_evidence_required", "attach_evidence"),
	_rule("compliance_review_required", "Noncompliant or expired compliance requires review.", {"operation": "record_compliance", "review_required": True, "review_recorded": False}, "require_review", "compliance_review_required", "record_compliance_review"),
	_rule("contract_vendor_required", "Vendor contracts require an existing vendor.", {"operation": "create_contract", "vendor_present": False}, "deny", "contract_vendor_required", "select_vendor"),
	_rule("contract_value_required", "Vendor contracts require value.", {"operation": "create_contract", "value_present": False}, "deny", "contract_value_required", "provide_value"),
	_rule("contract_currency_required", "Vendor contracts require currency.", {"operation": "create_contract", "currency_present": False}, "deny", "contract_currency_required", "provide_currency"),
	_rule("contract_dates_required", "Vendor contracts require start and end dates.", {"operation": "create_contract", "date_range_present": False}, "deny", "contract_date_range_required", "provide_dates"),
	_rule("contract_approval_required", "Vendor contracts require approval.", {"operation": "create_contract", "approval_recorded": False}, "deny", "contract_approval_required", "record_contract_approval"),
	_rule("communication_vendor_required", "Communications require an existing vendor.", {"operation": "record_communication", "vendor_present": False}, "deny", "communication_vendor_required", "select_vendor"),
	_rule("communication_channel_required", "Communications require channel.", {"operation": "record_communication", "channel_present": False}, "deny", "communication_channel_required", "provide_channel"),
	_rule("communication_subject_required", "Communications require subject.", {"operation": "record_communication", "subject_present": False}, "deny", "communication_subject_required", "provide_subject"),
	_rule("negative_sentiment_owner_required", "Negative communication sentiment requires an owner.", {"operation": "record_communication", "negative_sentiment": True, "owner_present": False}, "deny", "communication_owner_required", "assign_owner"),
	_rule("portal_vendor_required", "Portal users require an existing vendor.", {"operation": "create_portal_user", "vendor_present": False}, "deny", "portal_vendor_required", "select_vendor"),
	_rule("portal_email_required", "Portal users require email.", {"operation": "create_portal_user", "email_present": False}, "deny", "portal_email_required", "provide_email"),
	_rule("portal_role_required", "Portal users require role.", {"operation": "create_portal_user", "role_present": False}, "deny", "portal_role_required", "provide_role"),
	_rule("portal_approval_required", "Portal users require approval.", {"operation": "create_portal_user", "approval_recorded": False}, "deny", "portal_approval_required", "record_approval"),
	_rule("scorecard_vendor_required", "Scorecards require an existing vendor.", {"operation": "create_scorecard", "vendor_present": False}, "deny", "scorecard_vendor_required", "select_vendor"),
	_rule("scorecard_performance_required", "Scorecards require performance records.", {"operation": "create_scorecard", "performance_present": False}, "deny", "scorecard_performance_required", "attach_performance"),
	_rule("scorecard_risk_required", "Scorecards require risk records.", {"operation": "create_scorecard", "risk_present": False}, "deny", "scorecard_risk_required", "attach_risk"),
	_rule("scorecard_generator_required", "Scorecards require a generator.", {"operation": "create_scorecard", "generator_present": False}, "deny", "scorecard_generator_required", "record_generator"),
	_rule("bytewax_event_stream_required", "Vendor batches must use Bytewax event stream metadata.", {"operation": "vendor_batch", "event_stream": "queue"}, "deny", "bytewax_event_stream_required", "route_to_bytewax_stream"),
	_rule("agent_runtime_supported", "Vendor agents must use a supported runtime.", {"operation": "register_vendor_agent", "runtime_supported": False}, "deny", "vendor_agent_runtime_not_supported", "choose_supported_runtime"),
	_rule("agent_role_supported", "Vendor agents must use a supported role.", {"operation": "register_vendor_agent", "role_supported": False}, "deny", "vendor_agent_role_not_supported", "choose_supported_role"),
	_rule("agent_scope_limited", "Vendor agents cannot autonomously post privileged state changes.", {"operation": "agent_action", "privileged_action": True, "human_approved": False}, "require_review", "vendor_agent_human_approval_required", "record_human_approval"),
	_rule("audit_required_for_state_change", "Vendor state changes must be auditable.", {"operation_type": "write", "audit_enabled": False}, "deny", "vendor_audit_required", "enable_audit"),
]


CONFIGURATION_SCHEMA = {
	"type": "object",
	"required": ["tenant_id", "ui", "theme"],
	"properties": {
		"tenant_id": {"type": "string"},
		"vendors": {"type": "object"},
		"qualification": {"type": "object"},
		"onboarding": {"type": "object"},
		"performance": {"type": "object"},
		"risk": {"type": "object"},
		"compliance": {"type": "object"},
		"contracts": {"type": "object"},
		"communications": {"type": "object"},
		"portal": {"type": "object"},
		"scorecards": {"type": "object"},
		"vendor_agents": {"type": "object"},
		"governance": {"type": "object"},
		"observability": {"type": "object"},
		"adapters": {"type": "object"},
		"ui": {"type": "object"},
		"theme": {"type": "object"},
	},
}


def _merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
	merged = deepcopy(base)
	for key, value in overrides.items():
		if isinstance(value, dict) and isinstance(merged.get(key), dict):
			merged[key] = _merge_dict(merged[key], value)
		else:
			merged[key] = deepcopy(value)
	return merged


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the executable Vendor Management capability contract."""
	configuration = _merge_dict(DEFAULT_CONFIGURATION, overrides or {})
	configuration["tenant_id"] = tenant_id or configuration.get("tenant_id", "default")
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": deepcopy(PROVIDES),
		"requires": deepcopy(REQUIRES),
		"configuration": configuration,
		"configuration_schema": deepcopy(CONFIGURATION_SCHEMA),
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "api_prefix": "/scm/vendors/api/v1", "requires_theme": True, "template_roots": ["templates/", "static/"], "view_module": "views.py", "routes": deepcopy(UI_ROUTES)},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate deterministic Vendor Management guardrails."""
	matched_rules: list[str] = []
	effects: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched_rules.append(rule["name"])
			effect = deepcopy(rule["effect"])
			effects.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched_rules, "effects": effects, "context": deepcopy(context)}
