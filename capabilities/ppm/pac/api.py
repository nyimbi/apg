"""Process-local API helpers for APG Project Accounting (pac)."""

from __future__ import annotations

try:
	from .service import ProjectAccountingService
except ImportError:  # pragma: no cover
	import sys as _sys, pathlib as _pl
	_here = str(_pl.Path(__file__).parent)
	if _here not in _sys.path:
		_sys.path.insert(0, _here)
	from service import ProjectAccountingService  # type: ignore

_SERVICE = ProjectAccountingService()


def service() -> ProjectAccountingService:
	return _SERVICE


def create_account(payload: dict):
	return _SERVICE.create_account(
		payload["account_id"], payload.get("tenant_id", "default"),
		payload["project_id"], payload["name"], payload.get("status", "active"),
		payload.get("currency", "USD"), float(payload["budget_amount"]),
		payload["owner_id"], payload["evidence_reference"],
		payload.get("policy_attached", True),
	)


def record_cost(payload: dict):
	return _SERVICE.record_cost(
		payload["cost_id"], payload.get("tenant_id", "default"),
		payload["account_id"], payload["cost_type"], payload["transaction_type"],
		float(payload["amount"]), payload.get("description", ""),
		payload.get("period_reference", ""), payload["evidence_reference"],
		payload.get("backdated", False), payload.get("justification", ""),
	)


def recognise_revenue(payload: dict):
	return _SERVICE.recognise_revenue(
		payload["recognition_id"], payload.get("tenant_id", "default"),
		payload["account_id"], payload["revenue_type"], payload["wip_method"],
		float(payload["amount"]), payload.get("recognition_period", ""),
		payload["approval_reference"], payload["evidence_reference"],
	)


def post_wip_adjustment(payload: dict):
	return _SERVICE.post_wip_adjustment(
		payload["wip_id"], payload.get("tenant_id", "default"),
		payload["account_id"], float(payload["adjustment_amount"]),
		payload.get("description", ""), payload["auditor_id"],
		payload["evidence_reference"],
	)


def raise_invoice(payload: dict):
	return _SERVICE.raise_invoice(
		payload["invoice_id"], payload.get("tenant_id", "default"),
		payload["account_id"], payload["billing_type"],
		float(payload["amount"]), payload.get("milestone_reference", ""),
		payload["approval_reference"], payload["evidence_reference"],
	)


def override_budget(payload: dict):
	return _SERVICE.override_budget(
		payload["override_id"], payload.get("tenant_id", "default"),
		payload["account_id"], float(payload["original_budget"]),
		float(payload["revised_budget"]), payload.get("reason", ""),
		payload["controller_approval_reference"], payload["evidence_reference"],
	)


def record_approval(payload: dict):
	return _SERVICE.record_approval(
		payload["approval_id"], payload.get("tenant_id", "default"),
		payload["reference_id"], payload["approval_type"],
		payload["reviewer_id"], payload["status"], payload["evidence_reference"],
	)


def register_agent(payload: dict):
	return _SERVICE.register_agent(
		payload["agent_id"], payload.get("tenant_id", "default"),
		payload["name"], payload["runtime"], payload["role"],
		payload.get("scope", "project accounting operations"),
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


def profitability_report(payload: dict):
	return _SERVICE.profitability_report(
		payload.get("tenant_id", "default"),
		payload["account_id"],
		payload.get("method", "gross_margin"),
	)


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
