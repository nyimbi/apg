"""Legal Billing & Time Tracking — Flask Blueprint REST API."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import LegalBillingService

_log = logging.getLogger(__name__)

bp = Blueprint("leg_bil", __name__, url_prefix="/api/legal/bil")
_svc: LegalBillingService | None = None


def get_service() -> LegalBillingService:
	global _svc
	if _svc is None:
		_svc = LegalBillingService()
	return _svc


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


@bp.get("/health")
def health():
	return jsonify(_run(get_service().health_check()))


@bp.get("/describe")
def describe():
	return jsonify(_run(get_service().describe()))


@bp.get("/time-entries")
def list_time_entries():
	tenant = request.args.get("tenant_id", "default")
	billable = request.args.get("billable")
	try:
		items = _run(get_service().list_time_entries(
			tenant_id=tenant,
			matter_id=request.args.get("matter_id"),
			attorney_id=request.args.get("attorney_id"),
			status=request.args.get("status"),
			billable=None if billable is None else billable.lower() == "true",
			date_from=request.args.get("date_from"),
			date_to=request.args.get("date_to"),
		))
		total_hours = sum(te["hours"] for te in items)
		total_amount = sum(te["amount"] for te in items)
		return jsonify({"items": items, "total": len(items), "total_hours": total_hours, "total_amount": total_amount})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/time-entries/<entry_id>")
def get_time_entry(entry_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().get_time_entry(tenant, entry_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/time-entries")
def create_time_entry():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_time_entry(**body))), 201
	except Exception as exc:
		_log.error("create_time_entry: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.put("/time-entries/<entry_id>")
def update_time_entry(entry_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_time_entry(tenant, entry_id, **body)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/time-entries/<entry_id>")
def delete_time_entry(entry_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_time_entry(tenant, entry_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/time-entries/<entry_id>/submit")
def submit_time_entry(entry_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().submit_time_entry(tenant, entry_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/time-entries/<entry_id>/approve")
def approve_time_entry(entry_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().approve_time_entry(tenant, entry_id, body.get("approved_by", ""))))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/disbursements")
def list_disbursements():
	tenant = request.args.get("tenant_id", "default")
	billable = request.args.get("billable")
	try:
		items = _run(get_service().list_disbursements(
			tenant_id=tenant,
			matter_id=request.args.get("matter_id"),
			disbursement_type=request.args.get("disbursement_type"),
			billable=None if billable is None else billable.lower() == "true",
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/disbursements")
def create_disbursement():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_disbursement(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/disbursements/<disbursement_id>")
def update_disbursement(disbursement_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_disbursement(tenant, disbursement_id, **body)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/disbursements/<disbursement_id>")
def delete_disbursement(disbursement_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_disbursement(tenant, disbursement_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/invoices")
def list_invoices():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_invoices(
			tenant_id=tenant,
			matter_id=request.args.get("matter_id"),
			client_id=request.args.get("client_id"),
			status=request.args.get("status"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/invoices/<invoice_id>")
def get_invoice(invoice_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().get_invoice(tenant, invoice_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/invoices")
def create_invoice():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_invoice(**body))), 201
	except Exception as exc:
		_log.error("create_invoice: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.put("/invoices/<invoice_id>")
def update_invoice(invoice_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_invoice(tenant, invoice_id, **body)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/invoices/<invoice_id>")
def delete_invoice(invoice_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_invoice(tenant, invoice_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/invoices/<invoice_id>/approve")
def approve_invoice(invoice_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().approve_invoice(tenant, invoice_id, body.get("approved_by_id", ""))))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/invoices/<invoice_id>/send")
def send_invoice(invoice_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().send_invoice(tenant, invoice_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/invoices/<invoice_id>/pay")
def record_payment(invoice_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().record_payment(tenant, invoice_id, body.get("payment_reference", ""))))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/trust-accounts")
def list_trust_accounts():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_trust_accounts(tenant, request.args.get("client_id")))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/trust-accounts")
def create_trust_account():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_trust_account(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/trust-accounts/<account_id>/transactions")
def trust_transaction(account_id: str):
	body = request.get_json(force=True) or {}
	body["trust_account_id"] = account_id
	try:
		return jsonify(_run(get_service().trust_transaction(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/trust-accounts/<account_id>/transactions")
def list_trust_transactions(account_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_trust_transactions(tenant, account_id))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/dashboard")
def dashboard():
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().billing_dashboard(tenant)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/matters/<matter_id>/billing-summary")
def matter_billing(matter_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().matter_billing_summary(tenant, matter_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/audit")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	limit = int(request.args.get("limit", 100))
	try:
		return jsonify(_run(get_service().get_audit_events(tenant, limit)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400
