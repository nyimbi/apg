"""Process-local API helpers for APG Time & Expense Management (tex)."""

from __future__ import annotations

try:
	from .service import TimeExpenseService
except ImportError:  # pragma: no cover
	import sys as _sys, pathlib as _pl
	_here = str(_pl.Path(__file__).parent)
	if _here not in _sys.path:
		_sys.path.insert(0, _here)
	from service import TimeExpenseService  # type: ignore

_SERVICE = TimeExpenseService()


def service() -> TimeExpenseService:
	return _SERVICE


def submit_timesheet(payload: dict):
	return _SERVICE.submit_timesheet(
		payload["timesheet_id"], payload.get("tenant_id", "default"),
		payload["resource_id"], payload["project_id"],
		payload.get("period_type", "weekly"), payload.get("period_reference", ""),
		payload.get("status", "submitted"), payload["submitted_by"],
		payload["reviewer_id"], payload.get("policy_attached", True),
	)


def record_time_entry(payload: dict):
	return _SERVICE.record_time_entry(
		payload["entry_id"], payload.get("tenant_id", "default"),
		payload["timesheet_id"], payload["project_id"],
		payload.get("task_id", ""), payload.get("entry_type", "regular"),
		payload.get("billable_status", "billable"),
		float(payload["hours"]), payload["entry_date"],
		payload.get("description", ""),
		payload.get("backdated", False), payload.get("justification", ""),
	)


def approve_timesheet(payload: dict):
	return _SERVICE.approve_timesheet(
		payload["approval_id"], payload.get("tenant_id", "default"),
		payload["timesheet_id"], payload["reviewer_id"],
		payload["status"], payload.get("comments", ""),
		payload.get("evidence_reference", ""),
	)


def submit_expense(payload: dict):
	return _SERVICE.submit_expense(
		payload["expense_id"], payload.get("tenant_id", "default"),
		payload["resource_id"], payload["project_id"],
		payload["category"], payload.get("currency", "USD"),
		float(payload["amount"]), payload.get("receipt_status", "pending_upload"),
		payload["expense_date"], payload.get("description", ""),
		payload["approval_reference"], payload.get("evidence_reference", ""),
	)


def approve_expense(payload: dict):
	return _SERVICE.approve_expense(
		payload["approval_id"], payload.get("tenant_id", "default"),
		payload["expense_claim_id"], payload["reviewer_id"],
		payload["status"], payload.get("comments", ""),
		payload.get("evidence_reference", ""),
	)


def process_reimbursement(payload: dict):
	return _SERVICE.process_reimbursement(
		payload["reimb_id"], payload.get("tenant_id", "default"),
		payload["expense_claim_id"], payload["resource_id"],
		payload["method"], float(payload["amount"]),
		payload.get("currency", "USD"), payload["approval_reference"],
		payload.get("processed_date", ""),
	)


def set_billing_rate(payload: dict):
	return _SERVICE.set_billing_rate(
		payload["rate_id"], payload.get("tenant_id", "default"),
		payload["resource_id"], payload["project_id"],
		payload["rate_type"], float(payload["rate_amount"]),
		payload.get("currency", "USD"), payload["effective_date"],
		payload["approval_reference"],
	)


def billable_hours_summary(payload: dict):
	return _SERVICE.billable_hours_summary(
		payload.get("tenant_id", "default"),
		payload.get("project_id"),
	)


def register_agent(payload: dict):
	return _SERVICE.register_agent(
		payload["agent_id"], payload.get("tenant_id", "default"),
		payload["name"], payload["runtime"], payload["role"],
		payload.get("scope", "time and expense operations"),
	)


def validate_agent_action(payload: dict):
	return _SERVICE.validate_agent_action(
		payload.get("tenant_id", "default"),
		payload.get("privileged_scope", False),
		payload.get("human_approval_recorded", False),
	)


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(
		payload.get("tenant_id", "default"),
		payload["item_count"],
		payload.get("event_stream", "bytewax"),
	)


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
