"""Dependency-light API helpers for HCM Payroll."""

from __future__ import annotations

from typing import Any

try:
	from .service import PayrollManagementService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from service import PayrollManagementService  # type: ignore


SERVICE = PayrollManagementService()


def service() -> PayrollManagementService:
	"""Return the process-local payroll service."""
	return SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {"ok": True, "capability": contract["capability"], "display_name": contract["display_name"], "tenant_id": tenant_id, "route_count": len(contract["ui"]["routes"]), "rule_count": len(contract["rule_engine"]["rules"]), "summary": SERVICE.dashboard_summary(tenant_id)}


def create_payroll_period(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_payroll_period(payload.get("period_id", payload.get("id", "period")), payload["tenant_id"], payload["name"], payload.get("frequency", "monthly"), payload["start_date"], payload["end_date"], payload["pay_date"], payload.get("currency", "USD"))


def create_pay_group(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_pay_group(payload.get("pay_group_id", payload.get("id", "pay-group")), payload["tenant_id"], payload["code"], payload["name"], payload.get("frequency", "monthly"), payload.get("currency", "USD"), payload["country"], payload["owner_id"])


def create_employee_pay_profile(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_employee_pay_profile(payload.get("profile_id", payload.get("id", "profile")), payload["tenant_id"], payload["employee_id"], payload["pay_group_id"], payload.get("payment_method", "bank_transfer"), payload["tax_id"], payload.get("currency", "USD"), float(payload.get("base_pay", 0)), payload.get("reviewed_by"))


def create_pay_component(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_pay_component(payload.get("component_id", payload.get("id", "component")), payload["tenant_id"], payload["code"], payload["name"], payload["component_type"], payload.get("currency", "USD"), payload.get("taxable"))


def record_time_import(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_time_import(payload.get("time_import_id", payload.get("id", "time")), payload["tenant_id"], payload["period_id"], payload["profile_id"], float(payload["hours"]), payload["source"], float(payload.get("overtime_hours", 0)), payload.get("approved_by"))


def start_payroll_run(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.start_payroll_run(payload.get("run_id", payload.get("id", "run")), payload["tenant_id"], payload["period_id"], payload["pay_group_id"], payload["initiated_by"])


def add_line_item(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.add_line_item(payload.get("line_id", payload.get("id", "line")), payload["tenant_id"], payload["run_id"], payload["profile_id"], payload["component_id"], payload.get("amount"), payload.get("reviewed_by"))


def record_tax(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_tax(payload.get("tax_id", payload.get("id", "tax")), payload["tenant_id"], payload["run_id"], payload["profile_id"], payload.get("scope", "employee"), payload["authority"], payload.get("amount"))


def record_adjustment(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_adjustment(payload.get("adjustment_id", payload.get("id", "adjustment")), payload["tenant_id"], payload["run_id"], payload["profile_id"], float(payload["amount"]), payload["reason"], payload["approved_by"])


def approve_payroll_run(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_payroll_run(payload["run_id"], payload["tenant_id"], payload["approved_by"])


def post_payroll_run(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.post_payroll_run(payload["run_id"], payload["tenant_id"], payload["posted_by"])


def create_payment_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_payment_batch(payload.get("payment_id", payload.get("id", "payment")), payload["tenant_id"], payload["run_id"], payload["payment_date"], payload.get("approved_by"))


def publish_payslip(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_payslip(payload.get("payslip_id", payload.get("id", "payslip")), payload["tenant_id"], payload["run_id"], payload["profile_id"], payload["privacy_basis"])


def create_tax_filing(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_tax_filing(payload.get("filing_id", payload.get("id", "filing")), payload["tenant_id"], payload["run_id"], payload["authority"], payload["period_ref"], payload["approved_by"])


def register_payroll_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_payroll_agent(payload["tenant_id"], payload["name"], payload["runtime"], payload["role"], payload.get("scope", "review payroll operations"))


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	"""Generic composition helper used by APG package probes."""
	return SERVICE.create_record(str(payload.get("id", "payroll-period")), str(payload.get("tenant_id") or "default"), {"name": payload.get("name", "Payroll Period"), "frequency": payload.get("frequency", "monthly"), "start_date": payload.get("start_date", "2026-01-01"), "end_date": payload.get("end_date", "2026-01-31"), "pay_date": payload.get("pay_date", "2026-02-01"), "currency": payload.get("currency", "USD")}, str(payload.get("status") or "open"))


def list_records(collection: str | None = None, tenant_id: str = "default") -> list[dict[str, Any]]:
	return SERVICE.list_records(collection, tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)


class PayrollPeriodRestApi:
	"""Compatibility shim for older REST endpoint registration."""


class PayrollRunRestApi:
	"""Compatibility shim for older REST endpoint registration."""


class EmployeePayrollRestApi:
	"""Compatibility shim for older REST endpoint registration."""


class PayComponentRestApi:
	"""Compatibility shim for older REST endpoint registration."""


def register_api_endpoints(*_: Any, **__: Any) -> None:
	"""Compatibility hook for older Flask-AppBuilder setup code."""
	return None
