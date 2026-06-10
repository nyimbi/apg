"""Flask Blueprint REST API for SACCO Lending."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import SaccoLendingService

_log = logging.getLogger(__name__)

bp = Blueprint("sacco_lnd", __name__, url_prefix="/api/fintech/sacco/lnd")
_svc = SaccoLendingService()


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


# ── Health ────────────────────────────────────────────────────────────────────

@bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check())), 200


# ── Loan Products ─────────────────────────────────────────────────────────────

@bp.get("/products")
def list_products():
	result = _run(_svc.list_products(
		tenant_id=_tenant(),
		product_type=request.args.get("product_type"),
		active_only=request.args.get("active_only", "true").lower() == "true",
	))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/products/<product_id>")
def get_product(product_id: str):
	try:
		return jsonify(_run(_svc.get_product(product_id, tenant_id=_tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/products")
def create_product():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.create_product(
			product_code=body["product_code"],
			product_name=body["product_name"],
			product_type=body["product_type"],
			interest_rate_pa=body["interest_rate_pa"],
			min_amount=body["min_amount"],
			max_amount=body["max_amount"],
			min_term_months=body["min_term_months"],
			max_term_months=body["max_term_months"],
			tenant_id=_tenant(),
			interest_method=body.get("interest_method", "reducing_balance"),
			max_multiplier=body.get("max_multiplier", 3.0),
			grace_period_months=body.get("grace_period_months", 0),
			processing_fee_pct=body.get("processing_fee_pct", 0.0),
			insurance_fee_pct=body.get("insurance_fee_pct", 0.0),
			min_guarantors=body.get("min_guarantors", 2),
			requires_collateral=body.get("requires_collateral", False),
			description=body.get("description"),
		))
		return jsonify(result), 201
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 422
	except Exception as exc:
		_log.error("create_product error: %s", exc)
		return jsonify({"error": str(exc)}), 500


@bp.put("/products/<product_id>")
def update_product(product_id: str):
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.update_product(product_id, tenant_id=_tenant(), **body))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.delete("/products/<product_id>")
def delete_product(product_id: str):
	try:
		return jsonify(_run(_svc.delete_product(product_id, tenant_id=_tenant()))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


# ── Loans ─────────────────────────────────────────────────────────────────────

@bp.get("/loans")
def list_loans():
	result = _run(_svc.list_loans(
		tenant_id=_tenant(),
		member_id=request.args.get("member_id"),
		product_id=request.args.get("product_id"),
		status=request.args.get("status"),
	))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/loans/<loan_id>")
def get_loan(loan_id: str):
	try:
		return jsonify(_run(_svc.get_loan(loan_id, tenant_id=_tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/loans")
def apply_for_loan():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.apply_for_loan(
			member_id=body["member_id"],
			product_id=body["product_id"],
			amount_requested=body["amount_requested"],
			term_months=body["term_months"],
			purpose=body["purpose"],
			tenant_id=_tenant(),
			guarantor_ids=body.get("guarantor_ids", []),
			collateral_description=body.get("collateral_description"),
			collateral_value=body.get("collateral_value"),
		))
		return jsonify(result), 201
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 422
	except Exception as exc:
		_log.error("apply_for_loan error: %s", exc)
		return jsonify({"error": str(exc)}), 500


@bp.put("/loans/<loan_id>")
def update_loan(loan_id: str):
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.update_loan(loan_id, tenant_id=_tenant(), **body))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.delete("/loans/<loan_id>")
def cancel_loan(loan_id: str):
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.delete_loan(loan_id, tenant_id=_tenant(), reason=body.get("reason", "admin_cancel")))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/loans/<loan_id>/approve")
def approve_loan(loan_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.approve_loan(
			loan_id=loan_id,
			approved_amount=body["approved_amount"],
			approved_term_months=body["approved_term_months"],
			approved_by=body["approved_by"],
			tenant_id=_tenant(),
			approved_rate=body.get("approved_rate"),
			approval_notes=body.get("approval_notes"),
			conditions=body.get("conditions"),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/loans/<loan_id>/reject")
def reject_loan(loan_id: str):
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.reject_loan(loan_id, rejected_by=body["rejected_by"], rejection_reason=body["rejection_reason"], tenant_id=_tenant()))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/loans/<loan_id>/disburse")
def disburse_loan(loan_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.disburse_loan(
			loan_id=loan_id,
			disbursement_method=body["disbursement_method"],
			disbursement_reference=body["disbursement_reference"],
			disbursed_by=body["disbursed_by"],
			tenant_id=_tenant(),
			disbursement_account=body.get("disbursement_account"),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.get("/loans/<loan_id>/schedule")
def get_schedule(loan_id: str):
	try:
		return jsonify(_run(_svc.get_repayment_schedule(loan_id, tenant_id=_tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Repayments ────────────────────────────────────────────────────────────────

@bp.post("/repayments")
def record_repayment():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.record_repayment(
			loan_id=body["loan_id"],
			amount=body["amount"],
			payment_reference=body["payment_reference"],
			recorded_by=body["recorded_by"],
			tenant_id=_tenant(),
			payment_method=body.get("payment_method", "cash"),
			payment_date=body.get("payment_date"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.get("/repayments")
def list_repayments():
	result = _run(_svc.list_repayments(
		loan_id=request.args.get("loan_id"),
		member_id=request.args.get("member_id"),
		tenant_id=_tenant(),
	))
	return jsonify({"items": result, "total": len(result)}), 200


# ── Credit Scoring ────────────────────────────────────────────────────────────

@bp.post("/credit-score")
def compute_credit_score():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.compute_credit_score(
			member_id=body["member_id"],
			savings_balance=body["savings_balance"],
			share_capital=body["share_capital"],
			months_as_member=body["months_as_member"],
			existing_loan_balance=body.get("existing_loan_balance", 0.0),
			repayment_record_pct=body.get("repayment_record_pct", 100.0),
			tenant_id=_tenant(),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.get("/credit-score/<member_id>")
def get_credit_score(member_id: str):
	result = _run(_svc.get_latest_credit_score(member_id, tenant_id=_tenant()))
	if result is None:
		return jsonify({"error": "no_credit_score_found"}), 404
	return jsonify(result), 200


# ── Arrears ───────────────────────────────────────────────────────────────────

@bp.post("/arrears/check")
def run_arrears_check():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.run_arrears_check(as_of_date=body.get("as_of_date", ""), tenant_id=_tenant()))), 200
	except Exception as exc:
		return jsonify({"error": str(exc)}), 500


@bp.get("/arrears")
def list_arrears():
	result = _run(_svc.list_arrears(min_days=int(request.args.get("min_days", 1)), tenant_id=_tenant()))
	return jsonify({"items": result, "total": len(result)}), 200


# ── CRB ───────────────────────────────────────────────────────────────────────

@bp.post("/crb")
def submit_crb():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.submit_crb_report(
			member_id=body["member_id"],
			report_type=body["report_type"],
			reason=body["reason"],
			reported_by=body["reported_by"],
			tenant_id=_tenant(),
			crb_reference=body.get("crb_reference"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.get("/crb")
def list_crb():
	result = _run(_svc.list_crb_reports(member_id=request.args.get("member_id"), tenant_id=_tenant()))
	return jsonify({"items": result, "total": len(result)}), 200


# ── Summary & Audit ───────────────────────────────────────────────────────────

@bp.get("/summary")
def summary():
	return jsonify(_run(_svc.portfolio_summary(tenant_id=_tenant()))), 200


@bp.get("/audit")
def audit_events():
	result = _run(_svc.get_audit_events(tenant_id=_tenant()))
	return jsonify({"items": result, "total": len(result)}), 200
