"""Executable capability contract for APG financial reporting."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_RPT_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_RPT_AGENT_ROLES = [
	"statement_reviewer",
	"consolidation_reviewer",
	"disclosure_reviewer",
	"distribution_reviewer",
	"variance_narrative_reviewer",
	"close_reporting_reviewer",
]
RPT_EVENT_STREAM = "apg.fin.rpt.lifecycle"
SUPPORTED_STATEMENT_TYPES = ["balance_sheet", "income_statement", "cash_flow", "equity_statement", "management_report"]
SUPPORTED_OUTPUT_FORMATS = ["pdf", "xlsx", "html", "json"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"templates": {
		"name_required": True,
		"statement_type_required": True,
		"supported_statement_types": SUPPORTED_STATEMENT_TYPES,
		"line_required_for_generation": True,
	},
	"report_lines": {
		"account_mapping_required": True,
		"sort_order_required": True,
		"line_type_required": True,
	},
	"periods": {
		"period_required": True,
		"period_dates_required": True,
		"close_status_required": True,
	},
	"generation": {
		"template_required": True,
		"period_required": True,
		"output_format_required": True,
		"supported_output_formats": SUPPORTED_OUTPUT_FORMATS,
		"data_quality_review_threshold": 0.97,
	},
	"statements": {
		"balance_check_required": True,
		"approval_required_for_publish": True,
		"narrative_review_required": True,
	},
	"consolidation": {
		"entity_required": True,
		"method_required": True,
		"elimination_review_required": True,
		"ownership_bounds": [0, 100],
	},
	"disclosures": {
		"statement_required": True,
		"owner_required": True,
		"review_required": True,
	},
	"distribution": {
		"statement_required": True,
		"recipient_required": True,
		"approved_statement_required": True,
		"format_required": True,
	},
	"rpt_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_RPT_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_RPT_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "recommend_validate_and_prepare",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_state_changes": True,
		"policy_attached_for_writes": True,
		"segregation_of_duties": True,
	},
	"observability": {
		"event_stream": RPT_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_template_events": True,
		"emit_generation_events": True,
		"emit_statement_events": True,
		"emit_consolidation_events": True,
		"emit_distribution_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"event_stream": "bytewax",
		"notification": "adapter",
		"general_ledger": "adapter",
		"accounts_payable": "adapter",
		"accounts_receivable": "adapter",
		"cash_management": "adapter",
		"document_management": "adapter",
		"business_intelligence": "adapter",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_templates": True,
		"enable_lines": True,
		"enable_periods": True,
		"enable_generation": True,
		"enable_statements": True,
		"enable_consolidation": True,
		"enable_disclosures": True,
		"enable_distribution": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "fin_rpt_control", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"templates",
		"report_lines",
		"periods",
		"generation",
		"statements",
		"consolidation",
		"disclosures",
		"distribution",
		"rpt_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		"tenant_id": {"type": "string", "minLength": 1},
		"templates": {"type": "object"},
		"report_lines": {"type": "object"},
		"periods": {"type": "object"},
		"generation": {"type": "object"},
		"statements": {"type": "object"},
		"consolidation": {"type": "object"},
		"disclosures": {"type": "object"},
		"distribution": {"type": "object"},
		"rpt_agents": {"type": "object"},
		"governance": {"type": "object"},
		"observability": {"type": "object"},
		"adapters": {"type": "object"},
		"ui": {"type": "object"},
		"theme": {"type": "object"},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Financial reporting operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "rpt_write_requires_policy", "description": "Financial reporting writes require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "template_requires_name", "description": "Report templates require a name.", "condition": {"operation": "create_template", "template_name_present": False}, "effect": {"decision": "deny", "reason": "template_name_required", "required_action": "name_template"}},
	{"name": "template_statement_type_supported", "description": "Report template statement type must be supported.", "condition": {"operation": "create_template", "statement_type_supported": False}, "effect": {"decision": "deny", "reason": "statement_type_not_supported", "required_action": "select_supported_statement_type"}},
	{"name": "report_line_requires_template", "description": "Report lines require a template.", "condition": {"operation": "add_report_line", "template_present": False}, "effect": {"decision": "deny", "reason": "report_line_template_required", "required_action": "attach_template"}},
	{"name": "report_line_requires_account_mapping", "description": "Report lines require account mapping.", "condition": {"operation": "add_report_line", "account_mapping_present": False}, "effect": {"decision": "deny", "reason": "report_line_account_mapping_required", "required_action": "attach_account_mapping"}},
	{"name": "report_line_requires_sort_order", "description": "Report lines require sort order.", "condition": {"operation": "add_report_line", "sort_order_present": False}, "effect": {"decision": "deny", "reason": "report_line_sort_order_required", "required_action": "set_sort_order"}},
	{"name": "period_requires_name", "description": "Reporting periods require a name.", "condition": {"operation": "open_period", "period_name_present": False}, "effect": {"decision": "deny", "reason": "period_name_required", "required_action": "name_period"}},
	{"name": "period_requires_dates", "description": "Reporting periods require start and end dates.", "condition": {"operation": "open_period", "period_dates_present": False}, "effect": {"decision": "deny", "reason": "period_dates_required", "required_action": "set_period_dates"}},
	{"name": "period_end_after_start", "description": "Reporting period end must be after start.", "condition": {"operation": "open_period", "period_range_valid": False}, "effect": {"decision": "deny", "reason": "period_range_invalid", "required_action": "set_valid_period_range"}},
	{"name": "generation_requires_template", "description": "Report generation requires a template.", "condition": {"operation": "generate_report", "template_present": False}, "effect": {"decision": "deny", "reason": "generation_template_required", "required_action": "attach_template"}},
	{"name": "generation_requires_period", "description": "Report generation requires reporting period.", "condition": {"operation": "generate_report", "period_present": False}, "effect": {"decision": "deny", "reason": "generation_period_required", "required_action": "attach_period"}},
	{"name": "generation_requires_template_lines", "description": "Report generation requires template lines.", "condition": {"operation": "generate_report", "template_line_count_lte": 0}, "effect": {"decision": "deny", "reason": "template_lines_required", "required_action": "add_template_lines"}},
	{"name": "generation_output_format_supported", "description": "Report generation output format must be supported.", "condition": {"operation": "generate_report", "output_format_supported": False}, "effect": {"decision": "deny", "reason": "output_format_not_supported", "required_action": "select_supported_output_format"}},
	{"name": "generation_quality_requires_review", "description": "Low data quality score requires review.", "condition": {"operation": "generate_report", "data_quality_score_lt": 0.97, "quality_review_recorded": False}, "effect": {"decision": "require_review", "reason": "data_quality_review_required", "required_action": "record_quality_review"}},
	{"name": "statement_requires_generation", "description": "Statements require a generated report.", "condition": {"operation": "publish_statement", "generation_present": False}, "effect": {"decision": "deny", "reason": "statement_generation_required", "required_action": "generate_report"}},
	{"name": "statement_requires_balance_check", "description": "Statements require balance check.", "condition": {"operation": "publish_statement", "balance_check_passed": False}, "effect": {"decision": "deny", "reason": "statement_balance_check_required", "required_action": "resolve_balance_check"}},
	{"name": "statement_requires_approval", "description": "Statements require approval before publish.", "condition": {"operation": "publish_statement", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "statement_approval_required", "required_action": "record_statement_approval"}},
	{"name": "statement_requires_narrative_review", "description": "Statement narratives require review.", "condition": {"operation": "publish_statement", "narrative_review_recorded": False}, "effect": {"decision": "deny", "reason": "narrative_review_required", "required_action": "record_narrative_review"}},
	{"name": "consolidation_requires_parent_entity", "description": "Consolidations require parent entity.", "condition": {"operation": "create_consolidation", "parent_entity_present": False}, "effect": {"decision": "deny", "reason": "parent_entity_required", "required_action": "attach_parent_entity"}},
	{"name": "consolidation_requires_subsidiary_entity", "description": "Consolidations require subsidiary entity.", "condition": {"operation": "create_consolidation", "subsidiary_entity_present": False}, "effect": {"decision": "deny", "reason": "subsidiary_entity_required", "required_action": "attach_subsidiary_entity"}},
	{"name": "consolidation_ownership_within_bounds", "description": "Ownership percentage must be between 0 and 100.", "condition": {"operation": "create_consolidation", "ownership_out_of_bounds": True}, "effect": {"decision": "deny", "reason": "ownership_out_of_bounds", "required_action": "set_valid_ownership"}},
	{"name": "consolidation_requires_elimination_review", "description": "Consolidations require elimination review.", "condition": {"operation": "create_consolidation", "elimination_review_recorded": False}, "effect": {"decision": "require_review", "reason": "elimination_review_required", "required_action": "record_elimination_review"}},
	{"name": "disclosure_requires_statement", "description": "Disclosures require statement linkage.", "condition": {"operation": "record_disclosure", "statement_present": False}, "effect": {"decision": "deny", "reason": "disclosure_statement_required", "required_action": "attach_statement"}},
	{"name": "disclosure_requires_owner", "description": "Disclosures require owner.", "condition": {"operation": "record_disclosure", "owner_present": False}, "effect": {"decision": "deny", "reason": "disclosure_owner_required", "required_action": "assign_owner"}},
	{"name": "disclosure_requires_review", "description": "Disclosures require review.", "condition": {"operation": "record_disclosure", "disclosure_review_recorded": False}, "effect": {"decision": "deny", "reason": "disclosure_review_required", "required_action": "record_disclosure_review"}},
	{"name": "distribution_requires_statement", "description": "Distribution requires statement.", "condition": {"operation": "distribute_statement", "statement_present": False}, "effect": {"decision": "deny", "reason": "distribution_statement_required", "required_action": "attach_statement"}},
	{"name": "distribution_requires_approved_statement", "description": "Distribution requires approved statement.", "condition": {"operation": "distribute_statement", "statement_approved": False}, "effect": {"decision": "deny", "reason": "approved_statement_required", "required_action": "approve_statement"}},
	{"name": "distribution_requires_recipient", "description": "Distribution requires recipient.", "condition": {"operation": "distribute_statement", "recipient_present": False}, "effect": {"decision": "deny", "reason": "distribution_recipient_required", "required_action": "add_recipient"}},
	{"name": "distribution_format_supported", "description": "Distribution format must be supported.", "condition": {"operation": "distribute_statement", "distribution_format_supported": False}, "effect": {"decision": "deny", "reason": "distribution_format_not_supported", "required_action": "select_supported_format"}},
	{"name": "rpt_batch_requires_bytewax", "description": "Financial reporting batches require Bytewax coordination.", "condition": {"operation": "rpt_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_rpt_batch_to_bytewax"}},
	{"name": "rpt_event_requires_bytewax", "description": "Financial reporting lifecycle events require Bytewax.", "condition": {"operation": "rpt_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_rpt_event_to_bytewax"}},
	{"name": "rpt_agent_runtime_supported", "description": "RPT agents must use an approved runtime.", "condition": {"operation": "register_rpt_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "rpt_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "rpt_agent_role_supported", "description": "RPT agents must use an approved role.", "condition": {"operation": "register_rpt_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "rpt_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_rpt_action_requires_human_approval", "description": "Privileged RPT actions proposed by agents require human approval.", "condition": {"operation": "agent_rpt_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/fin-rpt/dashboard", "component": "FinancialReportingDashboard", "permission": "fin_rpt:view", "nav_group": "Overview"},
	{"name": "templates", "path": "/fin-rpt/templates", "component": "ReportTemplateWorkbench", "permission": "fin_rpt:manage_templates", "nav_group": "Templates"},
	{"name": "lines", "path": "/fin-rpt/lines", "component": "ReportLineGrid", "permission": "fin_rpt:manage_templates", "nav_group": "Templates"},
	{"name": "periods", "path": "/fin-rpt/periods", "component": "ReportingPeriodConsole", "permission": "fin_rpt:manage_periods", "nav_group": "Close"},
	{"name": "generation", "path": "/fin-rpt/generation", "component": "ReportGenerationConsole", "permission": "fin_rpt:generate", "nav_group": "Reports"},
	{"name": "statements", "path": "/fin-rpt/statements", "component": "FinancialStatementLibrary", "permission": "fin_rpt:publish", "nav_group": "Reports"},
	{"name": "consolidation", "path": "/fin-rpt/consolidation", "component": "ConsolidationWorkbench", "permission": "fin_rpt:consolidate", "nav_group": "Consolidation"},
	{"name": "disclosures", "path": "/fin-rpt/disclosures", "component": "DisclosureRegister", "permission": "fin_rpt:disclose", "nav_group": "Compliance"},
	{"name": "distribution", "path": "/fin-rpt/distribution", "component": "ReportDistribution", "permission": "fin_rpt:distribute", "nav_group": "Reports"},
	{"name": "agents", "path": "/fin-rpt/agents", "component": "RPTAgentWorkbench", "permission": "fin_rpt:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fin-rpt/settings", "component": "FinancialReportingSettings", "permission": "fin_rpt:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "fin_rpt_control",
	"tokens": {"color.primary": "#28536B", "color.accent": "#C44536", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"templates": {"icon": "file-spreadsheet", "status_indicator": "template-pill", "risk_style": "report-band"},
		"lines": {"visual": "report-grid", "status_style": "line-chip"},
		"periods": {"visual": "period-calendar", "status_style": "period-chip"},
		"generation": {"visual": "generation-queue", "status_style": "quality-chip"},
		"statements": {"visual": "statement-library", "status_style": "publish-chip"},
		"consolidation": {"visual": "entity-tree", "status_style": "entity-chip"},
		"disclosures": {"visual": "disclosure-register", "status_style": "review-chip"},
		"distribution": {"visual": "recipient-list", "status_style": "delivery-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "agent-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "fin_rpt",
		"display_name": "Financial Reporting",
		"provides": [
			"financial_report_template_lifecycle",
			"report_line_mapping",
			"reporting_period_lifecycle",
			"financial_statement_generation",
			"statement_publication_workflow",
			"financial_consolidation",
			"disclosure_management",
			"report_distribution",
			"rpt_agents",
		],
		"requires": ["auth", "audl", "ntfy", "composition_events", "composition_config", "general_ledger", "accounts_payable", "accounts_receivable", "cash_management", "document_management", "business_intelligence"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/fin-rpt/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": RPT_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"template_created",
			"report_line_added",
			"period_opened",
			"report_generated",
			"statement_published",
			"consolidation_created",
			"disclosure_recorded",
			"statement_distributed",
			"rpt_agent_registered",
		],
		"states": ["draft", "mapped", "open", "generated", "reviewed", "published", "distributed", "closed", "blocked"],
		"guardrails": [
			"rpt_batch_requires_bytewax",
			"rpt_event_requires_bytewax",
			"privileged_agent_rpt_action_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return RPT_EVENT_STREAM


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			if not context.get(key[:-4], 0) <= expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gte"):
			if not context.get(key[:-4], 0) >= expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
