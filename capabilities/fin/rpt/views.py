"""View models for APG financial reporting screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_RPT_AGENT_ROLES, SUPPORTED_RPT_AGENT_RUNTIMES, get_capability_contract
	from .context import get_current_user_id, get_tenant_id_from_request
	from .service import FinancialReportingService
except ImportError:
	from capability_contract import SUPPORTED_RPT_AGENT_ROLES, SUPPORTED_RPT_AGENT_RUNTIMES, get_capability_contract
	from context import get_current_user_id, get_tenant_id_from_request
	from service import FinancialReportingService


def navigation_model(tenant_id: str | None = None) -> dict[str, Any]:
	tenant_id = tenant_id or get_tenant_id_from_request()
	contract = get_capability_contract(tenant_id)
	return {"capability": contract["capability"], "routes": contract["ui"]["routes"], "theme": contract["theme"], "api_prefix": contract["ui"]["api_prefix"]}


def dashboard_model(service: FinancialReportingService, tenant_id: str | None = None) -> dict[str, Any]:
	tenant_id = tenant_id or get_tenant_id_from_request()
	user_id = get_current_user_id()
	return {"screen": "dashboard", "title": "Financial Reporting", "tenant_id": tenant_id, "user_id": user_id, "summary": service.dashboard_summary(tenant_id), "sections": ["templates", "generation", "statements", "consolidation", "distribution"]}


def template_model(service: FinancialReportingService, tenant_id: str | None = None) -> dict[str, Any]:
	tenant_id = tenant_id or get_tenant_id_from_request()
	return {"screen": "templates", "records": service.list_templates(tenant_id), "columns": ["template_id", "name", "statement_type", "owner", "line_count", "status"], "actions": ["create_template", "add_report_line"]}


def report_line_model(service: FinancialReportingService, tenant_id: str | None = None) -> dict[str, Any]:
	tenant_id = tenant_id or get_tenant_id_from_request()
	return {"screen": "lines", "records": service.list_report_lines(tenant_id), "columns": ["line_id", "template_record_id", "label", "account_mapping", "sort_order", "line_type"], "actions": ["add_report_line", "review_mapping"]}


def period_model(service: FinancialReportingService, tenant_id: str | None = None) -> dict[str, Any]:
	tenant_id = tenant_id or get_tenant_id_from_request()
	return {"screen": "periods", "records": service.list_periods(tenant_id), "columns": ["period_id", "name", "period_start", "period_end", "close_status", "status"], "actions": ["open_period", "close_period"]}


def generation_model(service: FinancialReportingService, tenant_id: str | None = None) -> dict[str, Any]:
	tenant_id = tenant_id or get_tenant_id_from_request()
	return {"screen": "generation", "records": service.list_generations(tenant_id), "columns": ["generation_id", "template_record_id", "period_record_id", "output_format", "data_quality_score", "status"], "actions": ["generate_report", "publish_statement"]}


def statement_model(service: FinancialReportingService, tenant_id: str | None = None) -> dict[str, Any]:
	tenant_id = tenant_id or get_tenant_id_from_request()
	user_id = get_current_user_id()
	return {"screen": "statements", "records": service.list_statements(tenant_id), "summary": service.statement_summary(tenant_id), "user_id": user_id, "columns": ["statement_id", "title", "approved_by", "narrative_reviewed_by", "status"], "actions": ["publish_statement", "record_disclosure", "distribute_statement"]}


def consolidation_model(service: FinancialReportingService, tenant_id: str | None = None) -> dict[str, Any]:
	tenant_id = tenant_id or get_tenant_id_from_request()
	return {"screen": "consolidation", "records": service.list_consolidations(tenant_id), "columns": ["consolidation_id", "parent_entity", "subsidiary_entity", "method", "ownership_percent", "status"], "actions": ["create_consolidation", "review_eliminations"]}


def disclosure_model(service: FinancialReportingService, tenant_id: str | None = None) -> dict[str, Any]:
	tenant_id = tenant_id or get_tenant_id_from_request()
	return {"screen": "disclosures", "records": service.list_disclosures(tenant_id), "columns": ["disclosure_id", "statement_record_id", "title", "owner", "reviewed_by", "status"], "actions": ["record_disclosure", "review_disclosure"]}


def distribution_model(service: FinancialReportingService, tenant_id: str | None = None) -> dict[str, Any]:
	tenant_id = tenant_id or get_tenant_id_from_request()
	return {"screen": "distribution", "records": service.list_distributions(tenant_id), "summary": service.distribution_summary(tenant_id), "columns": ["distribution_id", "statement_record_id", "recipients", "output_format", "status"], "actions": ["distribute_statement"]}


def agent_workbench_model(service: FinancialReportingService, tenant_id: str | None = None) -> dict[str, Any]:
	tenant_id = tenant_id or get_tenant_id_from_request()
	return {"screen": "agents", "records": service.list_rpt_agents(tenant_id), "supported_runtimes": SUPPORTED_RPT_AGENT_RUNTIMES, "supported_roles": SUPPORTED_RPT_AGENT_ROLES, "actions": ["register_agent", "validate_action", "record_human_approval"]}
