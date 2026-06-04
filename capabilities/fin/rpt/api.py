"""Dependency-light API helpers for APG financial reporting."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .context import get_tenant_id_from_request
	from .service import FinancialReportingService
except ImportError:
	from capability_contract import get_capability_contract
	from context import get_tenant_id_from_request
	from service import FinancialReportingService


_SERVICE = FinancialReportingService()


def capability_status(tenant_id: str | None = None) -> dict[str, Any]:
	tenant_id = tenant_id or get_tenant_id_from_request()
	contract = get_capability_contract(tenant_id)
	return {
		"ok": True,
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"provides": contract["provides"],
		"requires": contract["requires"],
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"streaming": contract["streaming"],
		"summary": _SERVICE.dashboard_summary(tenant_id),
	}


def create_template(payload: dict[str, Any]) -> dict[str, Any]:
	tenant_id = get_tenant_id_from_request(payload)
	return _SERVICE.create_template(payload["template_id"], tenant_id, payload["name"], payload["statement_type"], payload["owner"])


def add_report_line(payload: dict[str, Any]) -> dict[str, Any]:
	tenant_id = get_tenant_id_from_request(payload)
	return _SERVICE.add_report_line(payload["line_id"], tenant_id, payload["template_record_id"], payload["label"], payload["account_mapping"], payload["sort_order"], payload.get("line_type", "detail"))


def open_period(payload: dict[str, Any]) -> dict[str, Any]:
	tenant_id = get_tenant_id_from_request(payload)
	return _SERVICE.open_period(payload["period_id"], tenant_id, payload["name"], payload["period_start"], payload["period_end"], payload.get("close_status", "open"))


def generate_report(payload: dict[str, Any]) -> dict[str, Any]:
	tenant_id = get_tenant_id_from_request(payload)
	return _SERVICE.generate_report(payload["generation_id"], tenant_id, payload["template_record_id"], payload["period_record_id"], payload["output_format"], payload.get("data_quality_score", 1.0), payload.get("quality_reviewed_by"))


def publish_statement(payload: dict[str, Any]) -> dict[str, Any]:
	tenant_id = get_tenant_id_from_request(payload)
	return _SERVICE.publish_statement(payload["statement_id"], tenant_id, payload["generation_record_id"], payload["title"], payload.get("balance_check_passed", True), payload["approved_by"], payload["narrative_reviewed_by"])


def create_consolidation(payload: dict[str, Any]) -> dict[str, Any]:
	tenant_id = get_tenant_id_from_request(payload)
	return _SERVICE.create_consolidation(payload["consolidation_id"], tenant_id, payload["parent_entity"], payload["subsidiary_entity"], payload["method"], payload["ownership_percent"], payload.get("elimination_reviewed_by"))


def distribute_statement(payload: dict[str, Any]) -> dict[str, Any]:
	tenant_id = get_tenant_id_from_request(payload)
	return _SERVICE.distribute_statement(payload["distribution_id"], tenant_id, payload["statement_record_id"], payload["recipients"], payload["output_format"])


def register_rpt_agent(payload: dict[str, Any]) -> dict[str, Any]:
	tenant_id = get_tenant_id_from_request(payload)
	return _SERVICE.register_rpt_agent(tenant_id, payload["name"], payload["runtime"], payload["role"], payload.get("instructions", ""))


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_record(payload)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	tenant_id = tenant_id or get_tenant_id_from_request()
	return _SERVICE.list_records(tenant_id)


def service() -> FinancialReportingService:
	return _SERVICE
