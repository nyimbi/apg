"""Complete REST API for APG Telecom Billing.

Flask Blueprint — url_prefix=/api/telecom/bil

All endpoints are tenant-scoped via X-Tenant-ID header.
Actor identity via X-Actor-ID header.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

import asyncio
from decimal import Decimal, InvalidOperation
from typing import Any

from flask import Blueprint, jsonify, request

from .domain.rules import RuleViolation
from .service import TelecomBillingService

bil_api = Blueprint("telecom_bil_api", __name__, url_prefix="/api/telecom/bil")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tenant() -> str:
	tid = request.headers.get("X-Tenant-ID", "").strip()
	if not tid:
		tid = request.args.get("tenant_id", "default").strip()
	return tid or "default"


def _actor() -> str:
	return request.headers.get("X-Actor-ID", "system").strip() or "system"


def _svc(tenant_id: str | None = None, actor_id: str | None = None) -> TelecomBillingService:
	return TelecomBillingService(
		tenant_id=tenant_id or _tenant(),
		actor_id=actor_id or _actor(),
	)


def _run(coro: Any) -> Any:
	"""Run an async coroutine from sync Flask context."""
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
				future = pool.submit(asyncio.run, coro)
				return future.result()
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


def _ok(data: Any, status: int = 200):
	return jsonify({"status": "ok", "data": data}), status


def _err(message: str, status: int = 400, code: str = "error"):
	return jsonify({"status": "error", "code": code, "message": message}), status


def _paginate(items: list[Any], page: int, per_page: int) -> dict[str, Any]:
	total = len(items)
	start = (page - 1) * per_page
	return {
		"items": items[start : start + per_page],
		"total": total,
		"page": page,
		"per_page": per_page,
		"pages": max(1, (total + per_page - 1) // per_page),
	}


def _handle(fn):
	"""Decorator: catch RuleViolation, ValueError, KeyError uniformly."""
	from functools import wraps

	@wraps(fn)
	def wrapper(*args, **kwargs):
		try:
			return fn(*args, **kwargs)
		except RuleViolation as e:
			return _err(str(e), 422, e.rule_name)
		except PermissionError as e:
			return _err(str(e), 403, "permission_denied")
		except (KeyError, LookupError) as e:
			return _err(str(e), 404, "not_found")
		except (ValueError, TypeError, InvalidOperation) as e:
			return _err(str(e), 400, "validation_error")
		except Exception as e:
			return _err(f"Internal error: {e}", 500, "internal_error")

	return wrapper


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@bil_api.get("/health")
def health():
	return _ok({"capability": "telecom_bil", "status": "healthy"})


# ---------------------------------------------------------------------------
# CDR endpoints
# ---------------------------------------------------------------------------

@bil_api.get("/cdrs")
@_handle
def list_cdrs():
	svc = _svc()
	page = int(request.args.get("page", 1))
	per_page = int(request.args.get("per_page", 50))
	status_filter = request.args.get("status")
	cdr_type_filter = request.args.get("cdr_type")
	msisdn_filter = request.args.get("msisdn")

	items = [
		v.to_dict() for (tid, _), v in svc.cdrs.items()
		if tid == svc.tenant_id
	]
	if status_filter:
		items = [i for i in items if i.get("mediation_status") == status_filter]
	if cdr_type_filter:
		items = [i for i in items if i.get("cdr_type") == cdr_type_filter]
	if msisdn_filter:
		items = [i for i in items if i.get("msisdn") == msisdn_filter]

	sort_by = request.args.get("sort_by", "recorded_at")
	items.sort(key=lambda x: x.get(sort_by, ""), reverse=True)
	return _ok(_paginate(items, page, per_page))


@bil_api.post("/cdrs")
@_handle
def create_cdr():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = svc.record_cdr(
		cdr_id=body["cdr_id"],
		source=body["source"],
		mediation_status=body.get("mediation_status", "raw"),
		msisdn=body["msisdn"],
		duration_seconds=int(body.get("duration_seconds", 0)),
		data_volume_bytes=int(body.get("data_volume_bytes", 0)),
		recorded_at=body.get("recorded_at", ""),
		policy_attached=bool(body.get("policy_attached", True)),
	)
	return _ok(result, 201)


@bil_api.get("/cdrs/<cdr_id>")
@_handle
def get_cdr(cdr_id: str):
	svc = _svc()
	item = svc.cdrs.get((svc.tenant_id, cdr_id))
	if not item:
		return _err(f"CDR {cdr_id} not found", 404, "not_found")
	return _ok(item.to_dict())


@bil_api.post("/cdrs/<cdr_id>/rate")
@_handle
def rate_cdr(cdr_id: str):
	body = request.get_json(force=True) or {}
	svc = _svc()
	body["subscriber_id"] = body.get("subscriber_id", cdr_id)
	cdr_type = body.get("cdr_type", "voice").lower()
	if cdr_type == "voice":
		result = _run(svc.rate_voice_call(body))
	elif cdr_type == "data":
		result = _run(svc.rate_data_session(body))
	elif cdr_type == "sms":
		result = _run(svc.rate_sms(body))
	elif cdr_type in {"roaming", "roaming_voice", "roaming_data"}:
		result = _run(svc.rate_roaming_event(body))
	else:
		return _err(f"Unsupported rating type: {cdr_type}", 400, "unsupported_cdr_type")
	return _ok(result)


# ---------------------------------------------------------------------------
# Usage Event endpoints
# ---------------------------------------------------------------------------

@bil_api.post("/usage-events")
@_handle
def create_usage_event():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = _run(svc.real_time_balance_check(
		subscriber_id=body["subscriber_id"],
		service_type=body.get("service_type", "voice"),
		amount=Decimal(str(body.get("amount", "0"))),
	))
	return _ok(result, 201)


@bil_api.post("/usage-events/bundle-consume")
@_handle
def consume_bundle():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = _run(svc.bundle_consumption(
		subscriber_id=body["subscriber_id"],
		event_type=body["event_type"],
		units=Decimal(str(body["units"])),
	))
	return _ok(result)


# ---------------------------------------------------------------------------
# Billing Account endpoints
# ---------------------------------------------------------------------------

@bil_api.get("/accounts")
@_handle
def list_accounts():
	svc = _svc()
	summary = svc.dashboard_summary()
	return _ok({"summary": summary, "message": "Use dashboard for account aggregates"})


@bil_api.post("/accounts/convergent")
@_handle
def create_convergent_account():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = svc.setup_convergent(
		account_id=body["account_id"],
		convergent_mode=body["convergent_mode"],
		master_account_id=body["master_account_id"],
		member_account_ids=body.get("member_account_ids", ""),
		currency=body.get("currency", "KES"),
	)
	return _ok(result, 201)


@bil_api.get("/accounts/<account_id>/balance")
@_handle
def get_balance(account_id: str):
	svc = _svc()
	balance = svc._balances.get(account_id, {})
	return _ok({
		"account_id": account_id,
		"tenant_id": svc.tenant_id,
		"balances": {k: str(v) for k, v in balance.items()},
	})


@bil_api.post("/accounts/<account_id>/suspend")
@_handle
def suspend_account(account_id: str):
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = _run(svc.service_suspension(
		account_id=account_id,
		reason=body.get("reason", "non_payment"),
	))
	return _ok(result)


@bil_api.post("/accounts/<account_id>/restore")
@_handle
def restore_account(account_id: str):
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = _run(svc.service_restoration(
		account_id=account_id,
		payment_id=body["payment_id"],
	))
	return _ok(result)


# ---------------------------------------------------------------------------
# Tariff Plan endpoints
# ---------------------------------------------------------------------------

@bil_api.get("/tariff-plans")
@_handle
def list_tariff_plans():
	return _ok({"message": "Tariff plans stored in external store; use /api/telecom/bil/reports/tariff-summary"})


# ---------------------------------------------------------------------------
# Invoice endpoints
# ---------------------------------------------------------------------------

@bil_api.get("/invoices")
@_handle
def list_invoices():
	svc = _svc()
	page = int(request.args.get("page", 1))
	per_page = int(request.args.get("per_page", 50))
	status_filter = request.args.get("status")
	customer_filter = request.args.get("customer_id")

	items = [
		v.to_dict() for (tid, _), v in svc.invoices.items()
		if tid == svc.tenant_id
	]
	if status_filter:
		items = [i for i in items if i.get("status") == status_filter]
	if customer_filter:
		items = [i for i in items if i.get("customer_id") == customer_filter]

	items.sort(key=lambda x: x.get("due_date", ""), reverse=True)
	return _ok(_paginate(items, page, per_page))


@bil_api.post("/invoices")
@_handle
def create_invoice():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = svc.generate_invoice(
		invoice_id=body["invoice_id"],
		customer_id=body["customer_id"],
		cycle_id=body["cycle_id"],
		total_amount=float(body.get("total_amount", 0)),
		currency=body.get("currency", "KES"),
		due_date=body["due_date"],
	)
	return _ok(result, 201)


@bil_api.get("/invoices/<invoice_id>")
@_handle
def get_invoice(invoice_id: str):
	svc = _svc()
	result = _run(svc.view_bill(invoice_id))
	if not result.get("found"):
		return _err(f"Invoice {invoice_id} not found", 404, "not_found")
	return _ok(result)


@bil_api.put("/invoices/<invoice_id>")
@_handle
def update_invoice(invoice_id: str):
	body = request.get_json(force=True) or {}
	svc = _svc()
	invoice = svc.invoices.get((svc.tenant_id, invoice_id))
	if not invoice:
		return _err(f"Invoice {invoice_id} not found", 404, "not_found")
	if "notes" in body:
		invoice.notes = body["notes"]  # type: ignore[attr-defined]
	return _ok(invoice.to_dict())


@bil_api.delete("/invoices/<invoice_id>")
@_handle
def cancel_invoice(invoice_id: str):
	body = request.get_json(force=True) or {}
	svc = _svc()
	invoice = svc.invoices.get((svc.tenant_id, invoice_id))
	if not invoice:
		return _err(f"Invoice {invoice_id} not found", 404, "not_found")
	invoice.status = "cancelled"  # type: ignore[attr-defined]
	svc._emit("invoice_cancelled", invoice_id, {"actor": svc.actor_id})
	return _ok({"invoice_id": invoice_id, "status": "cancelled"})


@bil_api.post("/invoices/<invoice_id>/approve")
@_handle
def approve_invoice(invoice_id: str):
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = svc.approve_invoice(invoice_id, body.get("approval_reference", ""))
	return _ok(result)


@bil_api.post("/invoices/<invoice_id>/reject")
@_handle
def reject_invoice(invoice_id: str):
	body = request.get_json(force=True) or {}
	svc = _svc()
	invoice = svc.invoices.get((svc.tenant_id, invoice_id))
	if not invoice:
		return _err(f"Invoice {invoice_id} not found", 404, "not_found")
	invoice.status = "cancelled"  # type: ignore[attr-defined]
	invoice.notes = body.get("reason", "rejected")  # type: ignore[attr-defined]
	svc._emit("invoice_rejected", invoice_id, {"reason": body.get("reason", "")})
	return _ok(invoice.to_dict())


@bil_api.post("/invoices/<invoice_id>/post")
@_handle
def post_invoice(invoice_id: str):
	svc = _svc()
	invoice = svc.invoices.get((svc.tenant_id, invoice_id))
	if not invoice:
		return _err(f"Invoice {invoice_id} not found", 404, "not_found")
	if getattr(invoice, "status", "") != "approved":
		return _err("Invoice must be approved before posting", 422, "invoice_not_approved")
	invoice.status = "sent"  # type: ignore[attr-defined]
	svc._emit("invoice_posted", invoice_id, {})
	return _ok(invoice.to_dict())


@bil_api.post("/invoices/<invoice_id>/write-off")
@_handle
def write_off_invoice(invoice_id: str):
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = svc.write_off_invoice(invoice_id, body.get("approval_reference", ""))
	return _ok(result)


@bil_api.post("/invoices/<invoice_id>/deliver")
@_handle
def deliver_invoice(invoice_id: str):
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = _run(svc.bill_delivery(invoice_id, body.get("channel", "email")))
	return _ok(result)


@bil_api.post("/invoices/<invoice_id>/adjust")
@_handle
def adjust_invoice(invoice_id: str):
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = _run(svc.apply_adjustments(
		invoice_id=invoice_id,
		adjustment_type=body["adjustment_type"],
		amount=Decimal(str(body["amount"])),
		reason=body["reason"],
	))
	return _ok(result)


# ---------------------------------------------------------------------------
# Bill cycle endpoints
# ---------------------------------------------------------------------------

@bil_api.get("/cycles")
@_handle
def list_cycles():
	svc = _svc()
	items = [
		v.to_dict() for (tid, _), v in svc.cycles.items()
		if tid == svc.tenant_id
	]
	return _ok(items)


@bil_api.post("/cycles")
@_handle
def create_cycle():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = svc.create_bill_cycle(
		cycle_id=body["cycle_id"],
		cycle_type=body.get("cycle_type", "monthly"),
		cutoff_date=body["cutoff_date"],
		start_date=body["start_date"],
		end_date=body["end_date"],
		status=body.get("status", "active"),
	)
	return _ok(result, 201)


@bil_api.post("/cycles/run")
@_handle
def run_billing():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = _run(svc.generate_bill_run(
		billing_date=body["billing_date"],
		segment=body.get("segment"),
	))
	return _ok(result)


# ---------------------------------------------------------------------------
# Payment endpoints
# ---------------------------------------------------------------------------

@bil_api.get("/payments")
@_handle
def list_payments():
	svc = _svc()
	page = int(request.args.get("page", 1))
	per_page = int(request.args.get("per_page", 50))
	items = [
		v.to_dict() for (tid, _), v in svc.payments.items()
		if tid == svc.tenant_id
	]
	items.sort(key=lambda x: x.get("paid_at", ""), reverse=True)
	return _ok(_paginate(items, page, per_page))


@bil_api.post("/payments")
@_handle
def create_payment():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = svc.record_payment(
		payment_id=body["payment_id"],
		invoice_id=body["invoice_id"],
		payment_method=body["payment_method"],
		amount=float(body["amount"]),
		currency=body.get("currency", "KES"),
		reference=body["reference"],
		paid_at=body.get("paid_at", ""),
	)
	return _ok(result, 201)


@bil_api.post("/payments/process")
@_handle
def process_payment():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = _run(svc.payment_processing(
		account_id=body["account_id"],
		amount=Decimal(str(body["amount"])),
		payment_method=body["payment_method"],
		reference=body["reference"],
	))
	return _ok(result)


@bil_api.post("/payments/<payment_id>/allocate")
@_handle
def allocate_payment(payment_id: str):
	svc = _svc()
	result = _run(svc.allocate_payment(payment_id))
	return _ok(result)


# ---------------------------------------------------------------------------
# Dunning endpoints
# ---------------------------------------------------------------------------

@bil_api.get("/dunning")
@_handle
def list_dunning():
	svc = _svc()
	items = [
		v.to_dict() for (tid, _), v in svc.dunning_steps.items()
		if tid == svc.tenant_id
	]
	return _ok(items)


@bil_api.post("/dunning")
@_handle
def create_dunning():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = svc.trigger_dunning(
		dunning_id=body["dunning_id"],
		invoice_id=body["invoice_id"],
		step=body["step"],
		triggered_at=body.get("triggered_at", ""),
		next_step_date=body.get("next_step_date"),
	)
	return _ok(result, 201)


@bil_api.post("/dunning/workflow")
@_handle
def run_dunning_workflow():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = _run(svc.dunning_workflow(
		account_id=body["account_id"],
		dpd_days=int(body["dpd_days"]),
	))
	return _ok(result)


# ---------------------------------------------------------------------------
# Discount endpoints
# ---------------------------------------------------------------------------

@bil_api.get("/discounts")
@_handle
def list_discounts():
	svc = _svc()
	items = [
		v.to_dict() for (tid, _), v in svc.discounts.items()
		if tid == svc.tenant_id
	]
	return _ok(items)


@bil_api.post("/discounts")
@_handle
def create_discount():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = svc.apply_discount(
		discount_id=body["discount_id"],
		customer_id=body["customer_id"],
		discount_type=body["discount_type"],
		discount_pct=float(body["discount_pct"]),
		approval_reference=body["approval_reference"],
		valid_from=body["valid_from"],
		valid_to=body["valid_to"],
	)
	return _ok(result, 201)


@bil_api.post("/promotions/apply")
@_handle
def apply_promotion():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = _run(svc.apply_promotion(
		subscriber_id=body["subscriber_id"],
		promo_code=body["promo_code"],
		valid_from=body.get("valid_from", ""),
		valid_to=body.get("valid_to", ""),
	))
	return _ok(result)


# ---------------------------------------------------------------------------
# Dispute endpoints
# ---------------------------------------------------------------------------

@bil_api.get("/disputes")
@_handle
def list_disputes():
	svc = _svc()
	page = int(request.args.get("page", 1))
	per_page = int(request.args.get("per_page", 50))
	status_filter = request.args.get("status")

	items = [
		d for d in svc._disputes.values()
		if d.get("tenant_id") == svc.tenant_id
	]
	if status_filter:
		items = [d for d in items if d.get("status") == status_filter]
	items.sort(key=lambda x: x.get("raised_at", ""), reverse=True)
	return _ok(_paginate(items, page, per_page))


@bil_api.post("/disputes")
@_handle
def create_dispute():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = _run(svc.raise_billing_dispute(
		account_id=body["account_id"],
		invoice_id=body["invoice_id"],
		disputed_amount=Decimal(str(body["disputed_amount"])),
		reason=body["reason"],
	))
	return _ok(result, 201)


@bil_api.get("/disputes/<dispute_id>")
@_handle
def get_dispute(dispute_id: str):
	svc = _svc()
	d = svc._disputes.get(dispute_id)
	if not d or d.get("tenant_id") != svc.tenant_id:
		return _err(f"Dispute {dispute_id} not found", 404, "not_found")
	return _ok(d)


@bil_api.put("/disputes/<dispute_id>")
@_handle
def update_dispute(dispute_id: str):
	body = request.get_json(force=True) or {}
	svc = _svc()
	d = svc._disputes.get(dispute_id)
	if not d or d.get("tenant_id") != svc.tenant_id:
		return _err(f"Dispute {dispute_id} not found", 404, "not_found")
	if "evidence_refs" in body:
		d["evidence_refs"] = body["evidence_refs"]
	return _ok(d)


@bil_api.post("/disputes/<dispute_id>/investigate")
@_handle
def investigate_dispute(dispute_id: str):
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = _run(svc.investigate_dispute(dispute_id, body.get("cdr_analysis", {})))
	return _ok(result)


@bil_api.post("/disputes/<dispute_id>/resolve")
@_handle
def resolve_dispute(dispute_id: str):
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = _run(svc.resolve_dispute(
		dispute_id=dispute_id,
		resolution=body["resolution"],
		credit_amount=Decimal(str(body.get("credit_amount", "0"))),
	))
	return _ok(result)


@bil_api.post("/disputes/<dispute_id>/cancel")
@_handle
def cancel_dispute(dispute_id: str):
	svc = _svc()
	d = svc._disputes.get(dispute_id)
	if not d or d.get("tenant_id") != svc.tenant_id:
		return _err(f"Dispute {dispute_id} not found", 404, "not_found")
	d["status"] = "withdrawn"
	svc._emit("dispute_withdrawn", dispute_id, {})
	return _ok(d)


# ---------------------------------------------------------------------------
# Interconnect Settlement endpoints
# ---------------------------------------------------------------------------

@bil_api.get("/settlements")
@_handle
def list_settlements():
	return _ok({"message": "Settlements stored in external store; use /reports/interconnect"})


@bil_api.post("/settlements/reconcile")
@_handle
def reconcile_settlement():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = _run(svc.interconnect_reconciliation(
		carrier=body["carrier"],
		period={"start": body["period_start"], "end": body["period_end"]},
	))
	return _ok(result)


# ---------------------------------------------------------------------------
# Agent endpoints
# ---------------------------------------------------------------------------

@bil_api.get("/agents")
@_handle
def list_agents():
	svc = _svc()
	items = [
		v.to_dict() for (tid, _), v in svc.agents.items()
		if tid == svc.tenant_id
	]
	return _ok(items)


@bil_api.post("/agents")
@_handle
def register_agent():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = svc.register_agent(
		agent_id=body["agent_id"],
		name=body["name"],
		runtime=body["runtime"],
		role=body["role"],
		scope=body.get("scope", "billing operations"),
	)
	return _ok(result, 201)


# ---------------------------------------------------------------------------
# Report endpoints
# ---------------------------------------------------------------------------

@bil_api.get("/reports/revenue")
@_handle
def report_revenue():
	svc = _svc()
	period = {
		"start": request.args.get("start", ""),
		"end": request.args.get("end", ""),
	}
	segment = request.args.get("segment")
	result = _run(svc.revenue_report(period, segment))
	return _ok(result)


@bil_api.get("/reports/arpu")
@_handle
def report_arpu():
	svc = _svc()
	period = {
		"start": request.args.get("start", ""),
		"end": request.args.get("end", ""),
	}
	result = _run(svc.arpu_analysis(period))
	return _ok(result)


@bil_api.get("/reports/disputes")
@_handle
def report_disputes():
	svc = _svc()
	period = {
		"start": request.args.get("start", ""),
		"end": request.args.get("end", ""),
	}
	result = _run(svc.dispute_analytics(period))
	return _ok(result)


@bil_api.get("/reports/leakage")
@_handle
def report_leakage():
	svc = _svc()
	period = {
		"start": request.args.get("start", ""),
		"end": request.args.get("end", ""),
	}
	result = _run(svc.revenue_leakage_detection(period))
	return _ok(result)


@bil_api.get("/reports/churn")
@_handle
def report_churn():
	svc = _svc()
	period = {
		"start": request.args.get("start", ""),
		"end": request.args.get("end", ""),
	}
	result = _run(svc.churn_revenue_impact(period))
	return _ok(result)


@bil_api.get("/reports/interconnect")
@_handle
def report_interconnect():
	svc = _svc()
	carrier = request.args.get("carrier", "")
	period = {
		"start": request.args.get("start", ""),
		"end": request.args.get("end", ""),
	}
	result = _run(svc.interconnect_reconciliation(carrier, period))
	return _ok(result)


@bil_api.get("/reports/dashboard")
@_handle
def report_dashboard():
	svc = _svc()
	result = svc.dashboard_summary()
	return _ok(result)


@bil_api.get("/reports/cdr-rating")
@_handle
def report_cdr_rating():
	svc = _svc()
	result = _run(svc.revenue_leakage_detection({
		"start": request.args.get("start", ""),
		"end": request.args.get("end", ""),
	}))
	return _ok(result)


# ---------------------------------------------------------------------------
# Real-time charging
# ---------------------------------------------------------------------------

@bil_api.post("/realtime/charge")
@_handle
def realtime_charge():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = _run(svc.real_time_balance_check(
		subscriber_id=body["subscriber_id"],
		service_type=body.get("service_type", "voice"),
		amount=Decimal(str(body.get("amount", "0"))),
	))
	return _ok(result)


@bil_api.post("/realtime/overage")
@_handle
def realtime_overage():
	body = request.get_json(force=True) or {}
	svc = _svc()
	result = _run(svc.overage_charging(
		subscriber_id=body["subscriber_id"],
		bundle_id=body["bundle_id"],
		excess_units=Decimal(str(body["excess_units"])),
	))
	return _ok(result)


# ---------------------------------------------------------------------------
# Convergent billing
# ---------------------------------------------------------------------------

@bil_api.post("/convergent/bill")
@_handle
def convergent_bill():
	body = request.get_json(force=True) or {}
	svc = _svc()
	account_id = body["account_id"]
	period = {"start": body.get("period_start", ""), "end": body.get("period_end", "")}
	result = _run(svc.generate_bill(account_id, period))
	return _ok(result)


# ---------------------------------------------------------------------------
# Tax calculation
# ---------------------------------------------------------------------------

@bil_api.post("/tax/calculate")
@_handle
def calculate_tax():
	body = request.get_json(force=True) or {}
	from .domain.calculations import calculate_jurisdiction_tax
	pre_tax = Decimal(str(body["amount"]))
	jurisdiction = body.get("jurisdiction", "KE")
	result = calculate_jurisdiction_tax(pre_tax, jurisdiction)
	# Ensure all values are JSON-serialisable
	return _ok({
		k: str(v) if isinstance(v, Decimal) else v
		for k, v in result.items()
	})
