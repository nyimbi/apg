"""View models for APG accounts payable screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AP_AGENT_ROLES, SUPPORTED_AP_AGENT_RUNTIMES, get_capability_contract
	from .service import AccountsPayableService
except ImportError:
	from capability_contract import SUPPORTED_AP_AGENT_ROLES, SUPPORTED_AP_AGENT_RUNTIMES, get_capability_contract
	from service import AccountsPayableService


def navigation_model(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"capability": contract["capability"], "routes": contract["ui"]["routes"], "theme": contract["theme"], "api_prefix": contract["ui"]["api_prefix"]}


def dashboard_model(service: AccountsPayableService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "dashboard",
		"title": "Accounts Payable",
		"summary": service.dashboard_summary(tenant_id),
		"sections": ["vendors", "invoices", "approvals", "payments", "aging"],
	}


def vendor_model(service: AccountsPayableService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "vendors",
		"records": service.list_vendors(tenant_id),
		"columns": ["vendor_id", "name", "owner", "payment_method", "status"],
		"actions": ["register_vendor", "review_bank_change", "record_invoice"],
	}


def invoice_model(service: AccountsPayableService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "invoices",
		"records": service.list_invoices(tenant_id),
		"columns": ["invoice_id", "vendor_id", "invoice_number", "amount", "currency", "status"],
		"actions": ["record_invoice", "match_invoice", "approve_invoice", "place_hold"],
	}


def matching_model(service: AccountsPayableService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "matching",
		"records": service.list_invoices(tenant_id),
		"columns": ["invoice_id", "matched", "receipt_reference", "variance_rate", "status"],
		"actions": ["match_invoice", "review_variance"],
	}


def approval_model(service: AccountsPayableService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "approvals",
		"records": [invoice for invoice in service.list_invoices(tenant_id) if invoice["status"] in {"matched", "captured"}],
		"columns": ["invoice_id", "amount", "requested_by", "approved_by", "status"],
		"actions": ["approve_invoice", "place_invoice_hold"],
	}


def payment_model(service: AccountsPayableService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "payments",
		"records": service.list_payments(tenant_id),
		"columns": ["payment_id", "vendor_id", "amount", "cash_account", "scheduled_date", "status"],
		"actions": ["schedule_payment", "release_payment_batch"],
	}


def expense_model(service: AccountsPayableService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "expenses",
		"records": service.list_expenses(tenant_id),
		"columns": ["report_id", "employee_id", "amount", "policy_exception", "status"],
		"actions": ["record_expense_report", "review_policy_exception"],
	}


def aging_model(service: AccountsPayableService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "aging",
		"summary": service.aging_summary(tenant_id),
		"columns": ["open_invoice_count", "open_amount", "held_invoice_count"],
		"actions": ["review_aging", "schedule_payments", "close_period"],
	}


def close_model(service: AccountsPayableService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "close",
		"records": service.list_period_closes(tenant_id),
		"columns": ["close_id", "period", "aging_reviewed_by", "status"],
		"actions": ["close_period", "review_exceptions"],
	}


def agent_workbench_model(service: AccountsPayableService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "agents",
		"records": service.list_ap_agents(tenant_id),
		"supported_runtimes": SUPPORTED_AP_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AP_AGENT_ROLES,
		"actions": ["register_agent", "validate_action", "record_human_approval"],
	}
