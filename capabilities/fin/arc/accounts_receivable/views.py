"""Screen-model helpers for the Accounts Receivable capability."""

from __future__ import annotations

from typing import Any

try:
	from .service import AccountsReceivableService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from service import AccountsReceivableService  # type: ignore

from .context import get_current_user_id, get_tenant_id_from_request


NAVIGATION = [
	{"name": "Dashboard", "route": "/arc-accounts-receivable/dashboard", "icon": "layout-dashboard"},
	{"name": "Customers", "route": "/arc-accounts-receivable/customers", "icon": "users"},
	{"name": "Credit", "route": "/arc-accounts-receivable/credit", "icon": "shield-check"},
	{"name": "Invoices", "route": "/arc-accounts-receivable/invoices", "icon": "file-text"},
	{"name": "Payments", "route": "/arc-accounts-receivable/payments", "icon": "receipt"},
	{"name": "Cash Application", "route": "/arc-accounts-receivable/cash-application", "icon": "combine"},
	{"name": "Collections", "route": "/arc-accounts-receivable/collections", "icon": "phone-call"},
	{"name": "Disputes", "route": "/arc-accounts-receivable/disputes", "icon": "message-square-warning"},
	{"name": "Aging", "route": "/arc-accounts-receivable/aging", "icon": "calendar-clock"},
	{"name": "Agents", "route": "/arc-accounts-receivable/agents", "icon": "bot"},
	{"name": "Settings", "route": "/arc-accounts-receivable/settings", "icon": "settings"},
]


def _base(screen: str, tenant_id: str) -> dict[str, Any]:
	return {"screen": screen, "tenant_id": tenant_id, "navigation": NAVIGATION}


def dashboard_model(service: AccountsReceivableService, tenant_id: str) -> dict[str, Any]:
	model = _base("dashboard", tenant_id)
	model["summary"] = service.dashboard_summary(tenant_id)
	model["work_queue"] = {
		"open_disputes": len([record for record in service.disputes.values() if record["tenant_id"] == tenant_id and record["status"] != "resolved"]),
		"open_collections": len([record for record in service.collection_activities.values() if record["tenant_id"] == tenant_id and record["status"] == "recorded"]),
	}
	return model


def customer_model(service: AccountsReceivableService, tenant_id: str) -> dict[str, Any]:
	model = _base("customers", tenant_id)
	model["records"] = service.list_records("customers", tenant_id)
	model["columns"] = ["customer_code", "legal_name", "customer_type", "currency", "credit_hold", "status"]
	return model


def credit_model(service: AccountsReceivableService, tenant_id: str) -> dict[str, Any]:
	model = _base("credit", tenant_id)
	model["records"] = service.list_records("credit_assessments", tenant_id)
	model["columns"] = ["customer_id", "credit_limit", "credit_score", "credit_hold", "reviewed_by", "status"]
	return model


def invoice_model(service: AccountsReceivableService, tenant_id: str) -> dict[str, Any]:
	model = _base("invoices", tenant_id)
	model["records"] = service.list_records("invoices", tenant_id)
	model["columns"] = ["invoice_number", "customer_id", "invoice_date", "due_date", "total_amount", "outstanding_amount", "status"]
	return model


def payment_model(service: AccountsReceivableService, tenant_id: str) -> dict[str, Any]:
	model = _base("payments", tenant_id)
	model["records"] = service.list_records("payments", tenant_id)
	model["columns"] = ["payment_reference", "customer_id", "payment_date", "amount", "unapplied_amount", "method", "status"]
	return model


def cash_application_model(service: AccountsReceivableService, tenant_id: str) -> dict[str, Any]:
	model = _base("cash_application", tenant_id)
	model["records"] = service.list_records("cash_applications", tenant_id)
	model["columns"] = ["payment_id", "invoice_id", "allocation_amount", "reviewed_by", "status"]
	return model


def collection_model(service: AccountsReceivableService, tenant_id: str) -> dict[str, Any]:
	model = _base("collections", tenant_id)
	model["records"] = service.list_records("collection_activities", tenant_id)
	model["columns"] = ["invoice_id", "contact_method", "priority", "outcome", "status"]
	return model


def dispute_model(service: AccountsReceivableService, tenant_id: str) -> dict[str, Any]:
	model = _base("disputes", tenant_id)
	model["records"] = service.list_records("disputes", tenant_id)
	model["columns"] = ["invoice_id", "reason", "owner", "resolution", "reviewed_by", "status"]
	return model


def aging_model(service: AccountsReceivableService, tenant_id: str) -> dict[str, Any]:
	model = _base("aging", tenant_id)
	model["summary"] = service.aging_summary(tenant_id)
	model["records"] = service.list_records("invoices", tenant_id)
	return model


def agent_workbench_model(service: AccountsReceivableService, tenant_id: str) -> dict[str, Any]:
	model = _base("agents", tenant_id)
	model["records"] = service.list_records("agents", tenant_id)
	model["actions"] = ["review_credit", "review_invoice", "prepare_cash_application", "prepare_collection_activity", "review_dispute"]
	return model


# ---------------------------------------------------------------------------
# Context-aware constructors — resolve tenant and user from request context
# ---------------------------------------------------------------------------


def dashboard_ctx(service: AccountsReceivableService) -> dict[str, Any]:
	"""Build dashboard model resolving tenant from request context."""
	tenant_id = get_tenant_id_from_request()
	user_id = get_current_user_id()
	model = dashboard_model(service, tenant_id)
	model["current_user_id"] = user_id
	return model


def customer_ctx(service: AccountsReceivableService) -> dict[str, Any]:
	"""Build customer model resolving tenant from request context."""
	tenant_id = get_tenant_id_from_request()
	user_id = get_current_user_id()
	model = customer_model(service, tenant_id)
	model["current_user_id"] = user_id
	return model


def credit_ctx(service: AccountsReceivableService) -> dict[str, Any]:
	"""Build credit model resolving tenant from request context."""
	tenant_id = get_tenant_id_from_request()
	user_id = get_current_user_id()
	model = credit_model(service, tenant_id)
	model["current_user_id"] = user_id
	return model


def invoice_ctx(service: AccountsReceivableService) -> dict[str, Any]:
	"""Build invoice model resolving tenant from request context."""
	tenant_id = get_tenant_id_from_request()
	user_id = get_current_user_id()
	model = invoice_model(service, tenant_id)
	model["current_user_id"] = user_id
	return model


def payment_ctx(service: AccountsReceivableService) -> dict[str, Any]:
	"""Build payment model resolving tenant from request context."""
	tenant_id = get_tenant_id_from_request()
	user_id = get_current_user_id()
	model = payment_model(service, tenant_id)
	model["current_user_id"] = user_id
	return model


def cash_application_ctx(service: AccountsReceivableService) -> dict[str, Any]:
	"""Build cash application model resolving tenant from request context."""
	tenant_id = get_tenant_id_from_request()
	user_id = get_current_user_id()
	model = cash_application_model(service, tenant_id)
	model["current_user_id"] = user_id
	return model


def collection_ctx(service: AccountsReceivableService) -> dict[str, Any]:
	"""Build collection model resolving tenant from request context."""
	tenant_id = get_tenant_id_from_request()
	user_id = get_current_user_id()
	model = collection_model(service, tenant_id)
	model["current_user_id"] = user_id
	return model


def dispute_ctx(service: AccountsReceivableService) -> dict[str, Any]:
	"""Build dispute model resolving tenant from request context."""
	tenant_id = get_tenant_id_from_request()
	user_id = get_current_user_id()
	model = dispute_model(service, tenant_id)
	model["current_user_id"] = user_id
	return model


def aging_ctx(service: AccountsReceivableService) -> dict[str, Any]:
	"""Build aging model resolving tenant from request context."""
	tenant_id = get_tenant_id_from_request()
	user_id = get_current_user_id()
	model = aging_model(service, tenant_id)
	model["current_user_id"] = user_id
	return model


def agent_workbench_ctx(service: AccountsReceivableService) -> dict[str, Any]:
	"""Build agent workbench model resolving tenant from request context."""
	tenant_id = get_tenant_id_from_request()
	user_id = get_current_user_id()
	model = agent_workbench_model(service, tenant_id)
	model["current_user_id"] = user_id
	return model
