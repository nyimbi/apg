"""Executable APG capability contract for Sustainability and ESG Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "ecd_esg"
CAPABILITY_NAME = "Sustainability and ESG Management"
CAPABILITY_VERSION = "2.1.0"
ESG_EVENT_STREAM = "apg.ecd.esg.lifecycle"

SUPPORTED_FRAMEWORKS = ["gri", "sasb", "tcfd", "issb", "csrd", "sec_climate", "local_regulatory"]
SUPPORTED_PILLARS = ["environmental", "social", "governance"]
SUPPORTED_METRIC_TYPES = ["emissions", "energy", "water", "waste", "labor", "safety", "diversity", "ethics", "board", "supply_chain"]
SUPPORTED_UNITS = ["tco2e", "kwh", "m3", "tonnes", "percent", "count", "score", "currency"]
SUPPORTED_MEASUREMENT_SOURCES = ["manual", "meter", "import", "api", "supplier", "calculation"]
SUPPORTED_TARGET_TYPES = ["absolute", "intensity", "reduction", "compliance"]
SUPPORTED_REPORT_TYPES = ["annual", "quarterly", "regulatory", "board", "investor", "supplier"]
SUPPORTED_RISK_TIERS = ["low", "medium", "high", "critical"]
SUPPORTED_ESG_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_ESG_AGENT_ROLES = [
	"sustainability_reviewer",
	"carbon_reviewer",
	"compliance_reviewer",
	"supplier_esg_reviewer",
	"report_reviewer",
	"stakeholder_query_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"profiles": {"name_required": True, "industry_required": True, "country_required": True, "reporting_year_required": True, "owner_required": True},
	"frameworks": {"profile_required": True, "supported_frameworks": SUPPORTED_FRAMEWORKS, "version_required": True, "owner_required": True},
	"metrics": {"profile_required": True, "supported_pillars": SUPPORTED_PILLARS, "supported_types": SUPPORTED_METRIC_TYPES, "supported_units": SUPPORTED_UNITS, "owner_required": True},
	"measurements": {"metric_required": True, "period_required": True, "value_required": True, "supported_sources": SUPPORTED_MEASUREMENT_SOURCES, "evidence_required": True, "review_required_for_calculation_or_supplier": True},
	"targets": {"metric_required": True, "supported_types": SUPPORTED_TARGET_TYPES, "baseline_required": True, "target_required": True, "due_date_required": True, "owner_required": True},
	"supplier_assessments": {"supplier_required": True, "period_required": True, "score_range": [0, 100], "evidence_required": True, "owner_required_for_high_risk": True},
	"initiatives": {"profile_required": True, "name_required": True, "pillar_required": True, "owner_required": True, "expected_impact_required": True},
	"risks": {"profile_required": True, "supported_tiers": SUPPORTED_RISK_TIERS, "owner_required_for_high_or_critical": True},
	"reports": {"profile_required": True, "supported_types": SUPPORTED_REPORT_TYPES, "frameworks_required": True, "measurements_required": True, "approval_required": True},
	"stakeholders": {"profile_required": True, "name_required": True, "type_required": True, "channel_required": True, "consent_required": True},
	"engagements": {"stakeholder_required": True, "topic_required": True, "channel_required": True, "owner_required_for_negative_sentiment": True},
	"esg_agents": {"enabled": True, "supported_runtimes": SUPPORTED_ESG_AGENT_RUNTIMES, "supported_roles": SUPPORTED_ESG_AGENT_ROLES, "max_autonomous_scope": "inspect_prepare_and_recommend", "human_approval_required": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_state_changes": True, "segregation_of_duties": True},
	"observability": {"event_stream": ESG_EVENT_STREAM, "stream_processor": "bytewax", "emit_profile_events": True, "emit_metric_events": True, "emit_measurement_events": True, "emit_report_events": True, "emit_agent_events": True},
	"adapters": {"authorization": "adapter", "audit": "adapter", "notification": "adapter", "workflow": "adapter", "documents": "adapter", "supplier_data": "adapter", "carbon_data": "adapter", "regulatory_content": "adapter", "event_stream": "bytewax", "theme": "adapter"},
	"ui": {"enable_dashboard": True, "enable_profiles": True, "enable_frameworks": True, "enable_metrics": True, "enable_measurements": True, "enable_targets": True, "enable_suppliers": True, "enable_initiatives": True, "enable_risks": True, "enable_reports": True, "enable_stakeholders": True, "enable_agents": True, "enable_settings": True},
	"theme": {"default_theme": "esg_control", "allow_tenant_overrides": True},
}


PROVIDES = [
	"esg_profile_lifecycle",
	"esg_framework_lifecycle",
	"esg_metric_lifecycle",
	"esg_measurement_lifecycle",
	"esg_target_lifecycle",
	"esg_supplier_assessment_lifecycle",
	"esg_initiative_lifecycle",
	"esg_risk_lifecycle",
	"esg_report_workflow",
	"esg_stakeholder_lifecycle",
	"esg_engagement_lifecycle",
	"esg_dashboard_service",
	"esg_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"composition_events",
	"composition_config",
	"workflow",
	"document_management",
	"supplier_master_data",
	"carbon_data_provider",
	"regulatory_content",
	"risk_policy",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/ecd/esg/dashboard", "component": "ESGDashboard", "permission": "ecd_esg:view", "nav_group": "Overview"},
	{"name": "profiles", "path": "/ecd/esg/profiles", "component": "ESGProfileWorkbench", "permission": "ecd_esg:manage_profiles", "nav_group": "Setup"},
	{"name": "frameworks", "path": "/ecd/esg/frameworks", "component": "ESGFrameworkDesk", "permission": "ecd_esg:manage_frameworks", "nav_group": "Setup"},
	{"name": "metrics", "path": "/ecd/esg/metrics", "component": "ESGMetricCatalog", "permission": "ecd_esg:manage_metrics", "nav_group": "Data"},
	{"name": "measurements", "path": "/ecd/esg/measurements", "component": "ESGMeasurementLedger", "permission": "ecd_esg:record_data", "nav_group": "Data"},
	{"name": "targets", "path": "/ecd/esg/targets", "component": "ESGTargetBoard", "permission": "ecd_esg:manage_targets", "nav_group": "Planning"},
	{"name": "suppliers", "path": "/ecd/esg/suppliers", "component": "ESGSupplierAssessmentDesk", "permission": "ecd_esg:assess_suppliers", "nav_group": "Supply Chain"},
	{"name": "initiatives", "path": "/ecd/esg/initiatives", "component": "ESGInitiativeBoard", "permission": "ecd_esg:manage_initiatives", "nav_group": "Planning"},
	{"name": "risks", "path": "/ecd/esg/risks", "component": "ESGRiskRegister", "permission": "ecd_esg:govern", "nav_group": "Governance"},
	{"name": "reports", "path": "/ecd/esg/reports", "component": "ESGReportDesk", "permission": "ecd_esg:report", "nav_group": "Reporting"},
	{"name": "stakeholders", "path": "/ecd/esg/stakeholders", "component": "ESGStakeholderDesk", "permission": "ecd_esg:engage", "nav_group": "Engagement"},
	{"name": "agents", "path": "/ecd/esg/agents", "component": "ESGAgentWorkbench", "permission": "ecd_esg:agent_manage", "nav_group": "Automation"},
	{"name": "rules", "path": "/ecd/esg/rules", "component": "ESGRules", "permission": "ecd_esg:govern", "nav_group": "Governance"},
	{"name": "settings", "path": "/ecd/esg/settings", "component": "ESGSettings", "permission": "ecd_esg:admin", "nav_group": "Administration"},
]


THEME = {
	"name": "esg_control",
	"tokens": {
		"border.radius": "8px",
		"color.primary": "#2F5D50",
		"color.accent": "#7A5C1E",
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
		"dashboard": {"icon": "layout-dashboard", "status_indicator": "health-pill", "visual": "esg-grid"},
		"profiles": {"icon": "building-2", "status_style": "profile-chip", "visual": "profile-table"},
		"frameworks": {"icon": "book-open-check", "status_style": "framework-chip", "visual": "framework-list"},
		"metrics": {"icon": "ruler", "status_style": "metric-chip", "visual": "metric-table"},
		"measurements": {"icon": "database", "status_style": "evidence-chip", "visual": "measurement-ledger"},
		"targets": {"icon": "target", "status_style": "target-chip", "visual": "target-board"},
		"suppliers": {"icon": "network", "status_style": "supplier-chip", "visual": "supplier-assessments"},
		"initiatives": {"icon": "leaf", "status_style": "initiative-chip", "visual": "initiative-board"},
		"risks": {"icon": "shield-alert", "status_style": "risk-chip", "visual": "risk-register"},
		"reports": {"icon": "file-text", "status_style": "report-chip", "visual": "report-list"},
		"stakeholders": {"icon": "users", "status_style": "engagement-chip", "visual": "stakeholder-list"},
		"agents": {"icon": "bot", "status_style": "agent-chip", "visual": "agent-roster"},
		"rules": {"icon": "list-checks", "status_style": "decision-chip", "visual": "rule-list"},
		"settings": {"icon": "settings", "density": "compact", "visual": "settings-panel"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"event_stream": ESG_EVENT_STREAM,
	"events": [
		"esg_profile_created",
		"esg_framework_added",
		"esg_metric_defined",
		"esg_measurement_recorded",
		"esg_target_set",
		"esg_supplier_assessed",
		"esg_initiative_recorded",
		"esg_risk_recorded",
		"esg_report_created",
		"esg_stakeholder_registered",
		"esg_engagement_recorded",
		"esg_agent_registered",
	],
	"delivery": "at_least_once",
	"ordering_key": "tenant_id",
}


def _rule(name: str, description: str, condition: dict[str, Any], decision: str, reason: str, action: str) -> dict[str, Any]:
	return {"name": name, "description": description, "condition": condition, "effect": {"decision": decision, "reason": reason, "required_action": action}}


RULES = [
	_rule("tenant_context_required", "ESG operations require tenant context.", {"tenant_context_present": False}, "deny", "tenant_context_required", "attach_tenant_context"),
	_rule("operation_policy_required", "ESG write operations require policy enforcement.", {"operation_type": "write", "policy_attached": False}, "deny", "operation_policy_required", "attach_operation_policy"),
	_rule("profile_name_required", "ESG profiles require a name.", {"operation": "create_esg_profile", "name_present": False}, "deny", "esg_profile_name_required", "provide_name"),
	_rule("profile_industry_required", "ESG profiles require an industry.", {"operation": "create_esg_profile", "industry_present": False}, "deny", "esg_profile_industry_required", "provide_industry"),
	_rule("profile_country_required", "ESG profiles require a country.", {"operation": "create_esg_profile", "country_present": False}, "deny", "esg_profile_country_required", "provide_country"),
	_rule("profile_year_required", "ESG profiles require a reporting year.", {"operation": "create_esg_profile", "reporting_year_present": False}, "deny", "esg_profile_reporting_year_required", "provide_reporting_year"),
	_rule("profile_owner_required", "ESG profiles require an owner.", {"operation": "create_esg_profile", "owner_present": False}, "deny", "esg_profile_owner_required", "assign_owner"),
	_rule("framework_profile_required", "Frameworks require an ESG profile.", {"operation": "add_framework", "profile_present": False}, "deny", "esg_framework_profile_required", "select_profile"),
	_rule("framework_supported", "Framework code must be supported.", {"operation": "add_framework", "framework_supported": False}, "deny", "esg_framework_not_supported", "choose_supported_framework"),
	_rule("framework_version_required", "Frameworks require a version.", {"operation": "add_framework", "version_present": False}, "deny", "esg_framework_version_required", "provide_version"),
	_rule("framework_owner_required", "Frameworks require an owner.", {"operation": "add_framework", "owner_present": False}, "deny", "esg_framework_owner_required", "assign_owner"),
	_rule("metric_profile_required", "Metrics require an ESG profile.", {"operation": "define_metric", "profile_present": False}, "deny", "esg_metric_profile_required", "select_profile"),
	_rule("metric_pillar_supported", "Metric pillar must be supported.", {"operation": "define_metric", "pillar_supported": False}, "deny", "esg_metric_pillar_not_supported", "choose_supported_pillar"),
	_rule("metric_type_supported", "Metric type must be supported.", {"operation": "define_metric", "metric_type_supported": False}, "deny", "esg_metric_type_not_supported", "choose_supported_metric_type"),
	_rule("metric_unit_supported", "Metric unit must be supported.", {"operation": "define_metric", "unit_supported": False}, "deny", "esg_metric_unit_not_supported", "choose_supported_unit"),
	_rule("metric_name_required", "Metrics require a name.", {"operation": "define_metric", "name_present": False}, "deny", "esg_metric_name_required", "provide_metric_name"),
	_rule("metric_owner_required", "Metrics require an owner.", {"operation": "define_metric", "owner_present": False}, "deny", "esg_metric_owner_required", "assign_owner"),
	_rule("measurement_metric_required", "Measurements require a metric.", {"operation": "record_measurement", "metric_present": False}, "deny", "esg_measurement_metric_required", "select_metric"),
	_rule("measurement_period_required", "Measurements require a period.", {"operation": "record_measurement", "period_present": False}, "deny", "esg_measurement_period_required", "provide_period"),
	_rule("measurement_value_required", "Measurements require a value.", {"operation": "record_measurement", "value_present": False}, "deny", "esg_measurement_value_required", "provide_value"),
	_rule("measurement_source_supported", "Measurement source must be supported.", {"operation": "record_measurement", "source_supported": False}, "deny", "esg_measurement_source_not_supported", "choose_supported_source"),
	_rule("measurement_evidence_required", "Measurements require evidence.", {"operation": "record_measurement", "evidence_present": False}, "deny", "esg_measurement_evidence_required", "attach_evidence"),
	_rule("measurement_review_required", "Supplier or calculated measurements require review.", {"operation": "record_measurement", "review_required": True, "review_recorded": False}, "require_review", "esg_measurement_review_required", "record_measurement_review"),
	_rule("target_metric_required", "Targets require a metric.", {"operation": "set_target", "metric_present": False}, "deny", "esg_target_metric_required", "select_metric"),
	_rule("target_type_supported", "Target type must be supported.", {"operation": "set_target", "target_type_supported": False}, "deny", "esg_target_type_not_supported", "choose_supported_target_type"),
	_rule("target_baseline_required", "Targets require baseline value.", {"operation": "set_target", "baseline_present": False}, "deny", "esg_target_baseline_required", "provide_baseline"),
	_rule("target_value_required", "Targets require target value.", {"operation": "set_target", "target_present": False}, "deny", "esg_target_value_required", "provide_target"),
	_rule("target_due_date_required", "Targets require a due date.", {"operation": "set_target", "due_date_present": False}, "deny", "esg_target_due_date_required", "provide_due_date"),
	_rule("target_owner_required", "Targets require an owner.", {"operation": "set_target", "owner_present": False}, "deny", "esg_target_owner_required", "assign_owner"),
	_rule("supplier_required", "Supplier assessments require supplier id.", {"operation": "record_supplier_assessment", "supplier_present": False}, "deny", "esg_supplier_required", "select_supplier"),
	_rule("supplier_period_required", "Supplier assessments require a period.", {"operation": "record_supplier_assessment", "period_present": False}, "deny", "esg_supplier_period_required", "provide_period"),
	_rule("supplier_score_range", "Supplier ESG scores must be between 0 and 100.", {"operation": "record_supplier_assessment", "score_in_range": False}, "deny", "esg_supplier_score_out_of_range", "correct_score"),
	_rule("supplier_evidence_required", "Supplier assessments require evidence.", {"operation": "record_supplier_assessment", "evidence_present": False}, "deny", "esg_supplier_evidence_required", "attach_evidence"),
	_rule("supplier_owner_required", "High-risk supplier ESG assessments require an owner.", {"operation": "record_supplier_assessment", "high_risk": True, "owner_present": False}, "deny", "esg_supplier_owner_required", "assign_owner"),
	_rule("initiative_profile_required", "Initiatives require an ESG profile.", {"operation": "record_initiative", "profile_present": False}, "deny", "esg_initiative_profile_required", "select_profile"),
	_rule("initiative_name_required", "Initiatives require a name.", {"operation": "record_initiative", "name_present": False}, "deny", "esg_initiative_name_required", "provide_name"),
	_rule("initiative_pillar_supported", "Initiative pillar must be supported.", {"operation": "record_initiative", "pillar_supported": False}, "deny", "esg_initiative_pillar_not_supported", "choose_supported_pillar"),
	_rule("initiative_owner_required", "Initiatives require an owner.", {"operation": "record_initiative", "owner_present": False}, "deny", "esg_initiative_owner_required", "assign_owner"),
	_rule("initiative_impact_required", "Initiatives require expected impact.", {"operation": "record_initiative", "impact_present": False}, "deny", "esg_initiative_impact_required", "provide_expected_impact"),
	_rule("risk_profile_required", "ESG risks require a profile.", {"operation": "record_risk", "profile_present": False}, "deny", "esg_risk_profile_required", "select_profile"),
	_rule("risk_tier_supported", "ESG risk tier must be supported.", {"operation": "record_risk", "risk_tier_supported": False}, "deny", "esg_risk_tier_not_supported", "choose_supported_tier"),
	_rule("risk_description_required", "ESG risks require description.", {"operation": "record_risk", "description_present": False}, "deny", "esg_risk_description_required", "provide_description"),
	_rule("risk_owner_required", "High or critical ESG risks require an owner.", {"operation": "record_risk", "high_or_critical": True, "owner_present": False}, "deny", "esg_risk_owner_required", "assign_owner"),
	_rule("report_profile_required", "Reports require an ESG profile.", {"operation": "create_report", "profile_present": False}, "deny", "esg_report_profile_required", "select_profile"),
	_rule("report_type_supported", "Report type must be supported.", {"operation": "create_report", "report_type_supported": False}, "deny", "esg_report_type_not_supported", "choose_supported_report_type"),
	_rule("report_frameworks_required", "Reports require frameworks.", {"operation": "create_report", "frameworks_present": False}, "deny", "esg_report_frameworks_required", "attach_frameworks"),
	_rule("report_measurements_required", "Reports require measurements.", {"operation": "create_report", "measurements_present": False}, "deny", "esg_report_measurements_required", "attach_measurements"),
	_rule("report_approval_required", "Reports require approval.", {"operation": "create_report", "approval_recorded": False}, "deny", "esg_report_approval_required", "record_report_approval"),
	_rule("stakeholder_profile_required", "Stakeholders require a profile.", {"operation": "register_stakeholder", "profile_present": False}, "deny", "esg_stakeholder_profile_required", "select_profile"),
	_rule("stakeholder_name_required", "Stakeholders require a name.", {"operation": "register_stakeholder", "name_present": False}, "deny", "esg_stakeholder_name_required", "provide_name"),
	_rule("stakeholder_consent_required", "Stakeholders require engagement consent.", {"operation": "register_stakeholder", "consent_recorded": False}, "deny", "esg_stakeholder_consent_required", "record_consent"),
	_rule("engagement_stakeholder_required", "Engagements require a stakeholder.", {"operation": "record_engagement", "stakeholder_present": False}, "deny", "esg_engagement_stakeholder_required", "select_stakeholder"),
	_rule("engagement_topic_required", "Engagements require topic.", {"operation": "record_engagement", "topic_present": False}, "deny", "esg_engagement_topic_required", "provide_topic"),
	_rule("negative_engagement_owner_required", "Negative ESG engagement sentiment requires an owner.", {"operation": "record_engagement", "negative_sentiment": True, "owner_present": False}, "deny", "esg_engagement_owner_required", "assign_owner"),
	_rule("bytewax_event_stream_required", "ESG batches must use Bytewax event stream metadata.", {"operation": "esg_batch", "event_stream": "queue"}, "deny", "bytewax_event_stream_required", "route_to_bytewax_stream"),
	_rule("agent_runtime_supported", "ESG agents must use a supported runtime.", {"operation": "register_esg_agent", "runtime_supported": False}, "deny", "esg_agent_runtime_not_supported", "choose_supported_runtime"),
	_rule("agent_role_supported", "ESG agents must use a supported role.", {"operation": "register_esg_agent", "role_supported": False}, "deny", "esg_agent_role_not_supported", "choose_supported_role"),
	_rule("agent_scope_limited", "ESG agents cannot autonomously post privileged state changes.", {"operation": "agent_action", "privileged_action": True, "human_approved": False}, "require_review", "esg_agent_human_approval_required", "record_human_approval"),
	_rule("audit_required_for_state_change", "ESG state changes must be auditable.", {"operation_type": "write", "audit_enabled": False}, "deny", "esg_audit_required", "enable_audit"),
]


CONFIGURATION_SCHEMA = {"type": "object", "required": ["tenant_id", "ui", "theme"], "properties": {key: {"type": "object"} for key in DEFAULT_CONFIGURATION if key != "tenant_id"} | {"tenant_id": {"type": "string"}}}


def _merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
	merged = deepcopy(base)
	for key, value in overrides.items():
		if isinstance(value, dict) and isinstance(merged.get(key), dict):
			merged[key] = _merge_dict(merged[key], value)
		else:
			merged[key] = deepcopy(value)
	return merged


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	configuration = _merge_dict(DEFAULT_CONFIGURATION, overrides or {})
	configuration["tenant_id"] = tenant_id or configuration.get("tenant_id", "default")
	return {"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": deepcopy(PROVIDES), "requires": deepcopy(REQUIRES), "configuration": configuration, "configuration_schema": deepcopy(CONFIGURATION_SCHEMA), "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/ecd/esg/api/v1", "requires_theme": True, "template_roots": ["templates/", "static/"], "view_module": "views.py", "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
