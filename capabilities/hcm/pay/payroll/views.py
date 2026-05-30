"""Screen-model helpers for HCM Payroll."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import PayrollManagementService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import get_capability_contract  # type: ignore
	from service import PayrollManagementService  # type: ignore


NAVIGATION = [
	{"name": "Dashboard", "route": "/hcm/payroll/dashboard", "icon": "layout-dashboard"},
	{"name": "Periods", "route": "/hcm/payroll/periods", "icon": "calendar-days"},
	{"name": "Pay Groups", "route": "/hcm/payroll/pay-groups", "icon": "layers"},
	{"name": "Profiles", "route": "/hcm/payroll/profiles", "icon": "id-card"},
	{"name": "Components", "route": "/hcm/payroll/components", "icon": "list-plus"},
	{"name": "Time Imports", "route": "/hcm/payroll/time-imports", "icon": "clock"},
	{"name": "Runs", "route": "/hcm/payroll/runs", "icon": "calculator"},
	{"name": "Line Items", "route": "/hcm/payroll/line-items", "icon": "receipt-text"},
	{"name": "Taxes", "route": "/hcm/payroll/taxes", "icon": "landmark"},
	{"name": "Adjustments", "route": "/hcm/payroll/adjustments", "icon": "sliders-horizontal"},
	{"name": "Payments", "route": "/hcm/payroll/payments", "icon": "wallet-cards"},
	{"name": "Payslips", "route": "/hcm/payroll/payslips", "icon": "file-text"},
	{"name": "Filings", "route": "/hcm/payroll/tax-filings", "icon": "archive"},
	{"name": "Agents", "route": "/hcm/payroll/agents", "icon": "bot"},
	{"name": "Settings", "route": "/hcm/payroll/settings", "icon": "settings"},
]


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def _base(screen: str, tenant_id: str) -> dict[str, Any]:
	return {"screen": screen, "tenant_id": tenant_id, "navigation": NAVIGATION}


def dashboard_model(service: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("dashboard", tenant_id)
	model["summary"] = service.dashboard_summary(tenant_id)
	model["work_queue"] = {"open_periods": len([r for r in service.periods.values() if r["tenant_id"] == tenant_id and r["status"] == "open"]), "runs_needing_approval": len([r for r in service.runs.values() if r["tenant_id"] == tenant_id and not r.get("approved_by")]), "payments_ready": len([r for r in service.runs.values() if r["tenant_id"] == tenant_id and r.get("posted_by")]), "published_payslips": len(service.list_records("payslips", tenant_id))}
	return model


def _records(service: PayrollManagementService, tenant_id: str, screen: str, collection: str, columns: list[str]) -> dict[str, Any]:
	model = _base(screen, tenant_id)
	model["records"] = service.list_records(collection, tenant_id)
	model["columns"] = columns
	return model


def period_model(service: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records(service, tenant_id, "periods", "periods", ["name", "frequency", "start_date", "end_date", "pay_date", "currency", "status"])


def pay_group_model(service: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records(service, tenant_id, "pay_groups", "pay_groups", ["code", "name", "frequency", "currency", "country", "owner_id", "status"])


def profile_model(service: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records(service, tenant_id, "profiles", "employee_pay_profiles", ["employee_id", "pay_group_id", "payment_method", "currency", "base_pay", "status"])


def component_model(service: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records(service, tenant_id, "components", "components", ["code", "name", "component_type", "currency", "taxable", "status"])


def time_import_model(service: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records(service, tenant_id, "time_imports", "time_imports", ["period_id", "profile_id", "hours", "overtime_hours", "source", "status"])


def run_model(service: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records(service, tenant_id, "runs", "runs", ["period_id", "pay_group_id", "initiated_by", "approved_by", "posted_by", "status"])


def line_item_model(service: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records(service, tenant_id, "line_items", "line_items", ["run_id", "employee_id", "component_id", "component_type", "amount", "status"])


def tax_model(service: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records(service, tenant_id, "taxes", "taxes", ["run_id", "employee_id", "scope", "authority", "amount", "status"])


def adjustment_model(service: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records(service, tenant_id, "adjustments", "adjustments", ["run_id", "employee_id", "amount", "reason", "approved_by", "status"])


def payment_model(service: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records(service, tenant_id, "payments", "payment_batches", ["run_id", "payment_date", "approved_by", "net_pay", "status"])


def payslip_model(service: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records(service, tenant_id, "payslips", "payslips", ["run_id", "employee_id", "privacy_basis", "net_pay", "status"])


def filing_model(service: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records(service, tenant_id, "filings", "tax_filings", ["run_id", "authority", "period_ref", "tax_total", "status"])


def agent_workbench_model(service: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("agents", tenant_id)
	model["records"] = service.list_records("agents", tenant_id)
	model["actions"] = ["review_run", "review_tax", "review_payment", "review_variance", "review_employee_query"]
	return model


class PayrollPeriodModelView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class PayrollRunModelView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class EmployeePayrollModelView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class PayComponentModelView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class PayrollDashboardView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""
