"""Screen-model helpers for the General Ledger capability."""

from __future__ import annotations

from typing import Any

try:
	from .service import GeneralLedgerService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from service import GeneralLedgerService  # type: ignore


NAVIGATION = [
	{"name": "Dashboard", "route": "/glr-general-ledger/dashboard", "icon": "layout-dashboard"},
	{"name": "Accounts", "route": "/glr-general-ledger/accounts", "icon": "book-open"},
	{"name": "Dimensions", "route": "/glr-general-ledger/dimensions", "icon": "tags"},
	{"name": "Periods", "route": "/glr-general-ledger/periods", "icon": "calendar-days"},
	{"name": "Batches", "route": "/glr-general-ledger/batches", "icon": "layers"},
	{"name": "Journals", "route": "/glr-general-ledger/journals", "icon": "receipt-text"},
	{"name": "Postings", "route": "/glr-general-ledger/postings", "icon": "send"},
	{"name": "Trial Balance", "route": "/glr-general-ledger/trial-balance", "icon": "scale"},
	{"name": "Allocations", "route": "/glr-general-ledger/allocations", "icon": "git-branch"},
	{"name": "Reversals", "route": "/glr-general-ledger/reversals", "icon": "undo-2"},
	{"name": "Agents", "route": "/glr-general-ledger/agents", "icon": "bot"},
	{"name": "Settings", "route": "/glr-general-ledger/settings", "icon": "settings"},
]


def _base(screen: str, tenant_id: str) -> dict[str, Any]:
	return {"screen": screen, "tenant_id": tenant_id, "navigation": NAVIGATION}


def dashboard_model(service: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	model = _base("dashboard", tenant_id)
	model["summary"] = service.dashboard_summary(tenant_id)
	model["work_queue"] = {
		"unposted_journals": len([record for record in service.journal_entries.values() if record["tenant_id"] == tenant_id and record["status"] != "posted"]),
		"open_periods": len([record for record in service.periods.values() if record["tenant_id"] == tenant_id and record["status"] == "open"]),
	}
	return model


def account_model(service: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	model = _base("accounts", tenant_id)
	model["records"] = service.list_records("accounts", tenant_id)
	model["columns"] = ["code", "name", "account_type", "currency", "allow_posting", "status"]
	return model


def dimension_model(service: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	model = _base("dimensions", tenant_id)
	model["records"] = service.list_records("dimensions", tenant_id)
	model["columns"] = ["name", "value", "owner", "status"]
	return model


def period_model(service: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	model = _base("periods", tenant_id)
	model["records"] = service.list_records("periods", tenant_id)
	model["actions"] = ["open", "close", "review"]
	return model


def journal_batch_model(service: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	model = _base("journal_batches", tenant_id)
	model["records"] = service.list_records("journal_batches", tenant_id)
	model["columns"] = ["period_id", "source", "currency", "status"]
	return model


def journal_model(service: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	model = _base("journals", tenant_id)
	model["records"] = service.list_records("journal_entries", tenant_id)
	model["columns"] = ["description", "total_debits", "total_credits", "prepared_by", "approved_by", "status"]
	return model


def posting_model(service: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	model = _base("postings", tenant_id)
	model["records"] = service.list_records("postings", tenant_id)
	model["columns"] = ["journal_id", "posted_by", "status", "created_at"]
	return model


def trial_balance_model(service: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	model = _base("trial_balance", tenant_id)
	model["summary"] = service.dashboard_summary(tenant_id)
	model["audit_events"] = service.audit_events(tenant_id)
	return model


def allocation_model(service: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	model = _base("allocations", tenant_id)
	model["records"] = service.list_records("allocations", tenant_id)
	model["columns"] = ["source_account_id", "target_account_ids", "basis", "reviewed_by", "status"]
	return model


def reversal_model(service: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	model = _base("reversals", tenant_id)
	model["records"] = service.list_records("reversals", tenant_id)
	model["columns"] = ["journal_id", "reason", "approved_by", "status"]
	return model


def agent_workbench_model(service: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	model = _base("agents", tenant_id)
	model["records"] = service.list_records("agents", tenant_id)
	model["actions"] = ["review_journal", "prepare_reversal", "review_trial_balance", "recommend_allocation"]
	return model
