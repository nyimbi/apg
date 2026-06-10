"""Flask Blueprint REST API for SACCO Dividend & Distribution."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import SaccoDividendService

_log = logging.getLogger(__name__)

bp = Blueprint("sacco_div", __name__, url_prefix="/api/fintech/sacco/div")
_svc = SaccoDividendService()


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


# ── Financial Years ───────────────────────────────────────────────────────────

@bp.get("/years")
def list_years():
	result = _run(_svc.list_financial_years(tenant_id=_tenant(), status=request.args.get("status")))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/years/<year_id>")
def get_year(year_id: str):
	try:
		return jsonify(_run(_svc.get_financial_year(year_id, tenant_id=_tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/years")
def create_year():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.create_financial_year(
			year_code=body["year_code"],
			start_date=body["start_date"],
			end_date=body["end_date"],
			tenant_id=_tenant(),
			description=body.get("description"),
		))
		return jsonify(result), 201
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 422
	except Exception as exc:
		_log.error("create_year error: %s", exc)
		return jsonify({"error": str(exc)}), 500


@bp.put("/years/<year_id>")
def update_year(year_id: str):
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.update_financial_year(year_id, tenant_id=_tenant(), **body))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.delete("/years/<year_id>")
def delete_year(year_id: str):
	try:
		return jsonify(_run(_svc.delete_financial_year(year_id, tenant_id=_tenant()))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/years/<year_id>/close")
def close_year(year_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.close_financial_year(
			year_id=year_id,
			closed_by=body["closed_by"],
			approved_by=body["approved_by"],
			tenant_id=_tenant(),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


# ── Surplus Allocation ────────────────────────────────────────────────────────

@bp.post("/years/<year_id>/allocate")
def allocate_surplus(year_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.allocate_surplus(
			year_id=year_id,
			total_income=body["total_income"],
			total_expenses=body["total_expenses"],
			statutory_reserve_pct=body.get("statutory_reserve_pct", 20.0),
			education_fund_pct=body.get("education_fund_pct", 5.0),
			dividend_pool_pct=body.get("dividend_pool_pct", 50.0),
			rebate_pool_pct=body.get("rebate_pool_pct", 15.0),
			allocation_approved_by=body["allocation_approved_by"],
			allocation_date=body["allocation_date"],
			tenant_id=_tenant(),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.get("/years/<year_id>/allocations")
def list_allocations(year_id: str):
	result = _run(_svc.list_surplus_allocations(year_id=year_id, tenant_id=_tenant()))
	return jsonify({"items": result, "total": len(result)}), 200


# ── Declarations ──────────────────────────────────────────────────────────────

@bp.get("/declarations")
def list_declarations():
	result = _run(_svc.list_declarations(tenant_id=_tenant(), year_id=request.args.get("year_id")))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/declarations/<declaration_id>")
def get_declaration(declaration_id: str):
	try:
		return jsonify(_run(_svc.get_declaration(declaration_id, tenant_id=_tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/declarations")
def declare_dividend():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.declare_dividend(
			year_id=body["year_id"],
			dividend_rate_pct=body["dividend_rate_pct"],
			rebate_rate_pct=body["rebate_rate_pct"],
			declared_by=body["declared_by"],
			board_resolution_ref=body["board_resolution_ref"],
			declaration_date=body["declaration_date"],
			payment_date=body["payment_date"],
			tenant_id=_tenant(),
		))
		return jsonify(result), 201
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 422
	except Exception as exc:
		_log.error("declare_dividend error: %s", exc)
		return jsonify({"error": str(exc)}), 500


@bp.put("/declarations/<declaration_id>")
def update_declaration(declaration_id: str):
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.update_declaration(declaration_id, tenant_id=_tenant(), **body))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/declarations/<declaration_id>/reverse")
def reverse_declaration(declaration_id: str):
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.reverse_declaration(declaration_id, reversed_by=body["reversed_by"], reason=body["reason"], tenant_id=_tenant()))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.get("/declarations/<declaration_id>/summary")
def declaration_summary(declaration_id: str):
	try:
		return jsonify(_run(_svc.dividend_summary(declaration_id, tenant_id=_tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Distributions ─────────────────────────────────────────────────────────────

@bp.get("/distributions")
def list_distributions():
	result = _run(_svc.list_distributions(
		declaration_id=request.args.get("declaration_id"),
		member_id=request.args.get("member_id"),
		status=request.args.get("status"),
		tenant_id=_tenant(),
	))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/distributions/<distribution_id>")
def get_distribution(distribution_id: str):
	try:
		return jsonify(_run(_svc.get_distribution(distribution_id, tenant_id=_tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/distributions/compute")
def compute_distribution():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.compute_member_distribution(
			declaration_id=body["declaration_id"],
			member_id=body["member_id"],
			share_capital=body["share_capital"],
			savings_balance=body["savings_balance"],
			payment_method=body.get("payment_method", "savings_credit"),
			tenant_id=_tenant(),
			member_number=body.get("member_number"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/distributions/bulk-compute")
def bulk_compute():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.bulk_compute_distributions(
			declaration_id=body["declaration_id"],
			members=body["members"],
			tenant_id=_tenant(),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/distributions/<distribution_id>/pay")
def pay_distribution(distribution_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.pay_distribution(
			distribution_id=distribution_id,
			payment_reference=body["payment_reference"],
			paid_by=body["paid_by"],
			tenant_id=_tenant(),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/distributions/<distribution_id>/reverse")
def reverse_distribution(distribution_id: str):
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.reverse_distribution(distribution_id, reversed_by=body["reversed_by"], reason=body["reason"], tenant_id=_tenant()))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/declarations/<declaration_id>/pay-all")
def payment_batch(declaration_id: str):
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.run_payment_batch(declaration_id=declaration_id, run_by=body["run_by"], tenant_id=_tenant()))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


# ── WHT ───────────────────────────────────────────────────────────────────────

@bp.post("/wht")
def generate_wht():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.generate_wht_return(
			declaration_id=body["declaration_id"],
			filed_by=body["filed_by"],
			tenant_id=_tenant(),
			kra_return_reference=body.get("kra_return_reference"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.get("/wht")
def list_wht():
	result = _run(_svc.list_wht_records(tenant_id=_tenant()))
	return jsonify({"items": result, "total": len(result)}), 200


# ── Reports & Audit ───────────────────────────────────────────────────────────

@bp.get("/years/<year_id>/report")
def annual_report(year_id: str):
	try:
		return jsonify(_run(_svc.annual_report(year_id, tenant_id=_tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/members/<member_id>/history")
def member_history(member_id: str):
	return jsonify(_run(_svc.member_dividend_history(member_id, tenant_id=_tenant()))), 200


@bp.get("/audit")
def audit_events():
	result = _run(_svc.get_audit_events(tenant_id=_tenant()))
	return jsonify({"items": result, "total": len(result)}), 200
