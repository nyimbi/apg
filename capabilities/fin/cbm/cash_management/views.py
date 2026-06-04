"""Screen-model helpers for the Cash Management capability."""

from __future__ import annotations

from typing import Any

try:
	from .service import CashManagementService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from service import CashManagementService  # type: ignore


def _clean_text(value):
	if value is None:
		return None
	return str(value).strip() or None


def _object_value(source, name):
	if source is None:
		return None
	if isinstance(source, dict):
		return source.get(name)
	return getattr(source, name, None)


def _mapping_value(source, name):
	if source is None:
		return None
	getter = getattr(source, "get", None)
	return getter(name) if getter else None


def resolve_tenant_id(appbuilder=None):
	"""Resolve tenant id from Flask g, AppBuilder user, request headers/query, or env."""
	import os as _cbm_os  # noqa: PLC0415

	# 1. Flask g.current_user (highest priority — runtime context)
	try:
		from flask import g, has_request_context, request, session
		if has_request_context():
			cu = getattr(g, "current_user", None)
			t = _object_value(cu, "tenant_id") if isinstance(cu, dict) else None
			if t:
				return str(t)
			t = getattr(g, "tenant_id", None)
			if t:
				return str(t)
			# Headers
			t = _mapping_value(request.headers, "X-APG-Tenant-ID") or _mapping_value(request.headers, "X-Tenant-ID")
			if t:
				return str(t)
			# Query
			t = _mapping_value(request.args, "tenant_id") or _mapping_value(request.args, "tenant")
			if t:
				return str(t)
	except Exception:
		pass

	# 2. AppBuilder security manager user
	if appbuilder is not None:
		try:
			sm = getattr(appbuilder, "sm", None)
			cu = getattr(sm, "user", None) if sm else None
			t = _object_value(cu, "tenant_id")
			if t:
				return str(t)
		except Exception:
			pass

	# 3. Env fallback
	return _cbm_os.getenv("APG_DEFAULT_TENANT_ID", "default")


# ============================================================================
# Custom Widgets
# ============================================================================


NAVIGATION = [
	{"name": "Dashboard", "route": "/cbm-cash-management/dashboard", "icon": "layout-dashboard"},
	{"name": "Banks", "route": "/cbm-cash-management/banks", "icon": "landmark"},
	{"name": "Accounts", "route": "/cbm-cash-management/accounts", "icon": "wallet-cards"},
	{"name": "Positions", "route": "/cbm-cash-management/positions", "icon": "scale"},
	{"name": "Flows", "route": "/cbm-cash-management/flows", "icon": "arrow-left-right"},
	{"name": "Forecasts", "route": "/cbm-cash-management/forecasts", "icon": "chart-no-axes-combined"},
	{"name": "Liquidity", "route": "/cbm-cash-management/liquidity", "icon": "gauge"},
	{"name": "Reconciliation", "route": "/cbm-cash-management/reconciliation", "icon": "list-checks"},
	{"name": "Investments", "route": "/cbm-cash-management/investments", "icon": "trending-up"},
	{"name": "Payment Runs", "route": "/cbm-cash-management/payment-runs", "icon": "send"},
	{"name": "Agents", "route": "/cbm-cash-management/agents", "icon": "bot"},
	{"name": "Settings", "route": "/cbm-cash-management/settings", "icon": "settings"},
]


def _base(screen: str, tenant_id: str) -> dict[str, Any]:
	return {"screen": screen, "tenant_id": tenant_id, "navigation": NAVIGATION}


def dashboard_model(service: CashManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("dashboard", tenant_id)
	model["summary"] = service.dashboard_summary(tenant_id)
	model["work_queue"] = {
		"open_reconciliations": len([record for record in service.reconciliations.values() if record["tenant_id"] == tenant_id and record["status"] != "matched"]),
		"pending_payment_runs": len([record for record in service.payment_runs.values() if record["tenant_id"] == tenant_id and record["status"] != "funded"]),
	}
	return model


def bank_model(service: CashManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("banks", tenant_id)
	model["records"] = service.list_records("banks", tenant_id)
	model["columns"] = ["code", "name", "connectivity_status", "status"]
	return model


def account_model(service: CashManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("accounts", tenant_id)
	model["records"] = service.list_records("cash_accounts", tenant_id)
	model["columns"] = ["account_number", "name", "account_type", "currency", "minimum_buffer", "status"]
	return model


def position_model(service: CashManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("positions", tenant_id)
	model["records"] = service.list_records("cash_positions", tenant_id)
	model["columns"] = ["account_id", "as_of_date", "available_balance", "ledger_balance", "status"]
	return model


def flow_model(service: CashManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("flows", tenant_id)
	model["records"] = service.list_records("cash_flows", tenant_id)
	model["columns"] = ["account_id", "flow_type", "amount", "category", "expected_date", "status"]
	return model


def forecast_model(service: CashManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("forecasts", tenant_id)
	model["records"] = service.list_records("cash_forecasts", tenant_id)
	model["columns"] = ["horizon_days", "scenario", "confidence_score", "projected_net_cash", "status"]
	return model


def liquidity_model(service: CashManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("liquidity", tenant_id)
	model["records"] = service.list_records("liquidity_reviews", tenant_id)
	model["summary"] = service.dashboard_summary(tenant_id)
	return model


def reconciliation_model(service: CashManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("reconciliation", tenant_id)
	model["records"] = service.list_records("reconciliations", tenant_id)
	model["columns"] = ["account_id", "bank_statement_balance", "ledger_balance", "variance", "reviewed_by", "status"]
	return model


def investment_model(service: CashManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("investments", tenant_id)
	model["records"] = service.list_records("investments", tenant_id)
	model["columns"] = ["investment_type", "counterparty", "principal", "maturity_date", "yield_rate", "status"]
	return model


def payment_run_model(service: CashManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("payment_runs", tenant_id)
	model["records"] = service.list_records("payment_runs", tenant_id)
	model["columns"] = ["funding_account_id", "payment_total", "approved_by", "status"]
	return model


def agent_workbench_model(service: CashManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("agents", tenant_id)
	model["records"] = service.list_records("agents", tenant_id)
	model["actions"] = ["review_cash_position", "review_forecast", "review_reconciliation", "prepare_investment"]
	return model


# ============================================================================
# Flask-AppBuilder View Base
# ============================================================================

class CashManagementBaseView:
	"""Base mixin for FAB views — resolves tenant from AppBuilder context."""

	appbuilder = None

	def _get_tenant_id(self) -> str:
		return resolve_tenant_id(self.appbuilder)
