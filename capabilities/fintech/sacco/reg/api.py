"""Flask Blueprint REST API for SASRA Regulatory Reporting.

All endpoints expect X-Tenant-ID header. Amounts in KES (Decimal).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .models import ReturnType
from .service import SACCARegulatoryService

_log = logging.getLogger(__name__)

bp = Blueprint("sacco_reg", __name__, url_prefix="/api/fintech/sacco/reg")
_svc = SACCARegulatoryService()


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _err(msg: str, code: int = 422):
	return jsonify({"error": str(msg)}), code


# ── Health ─────────────────────────────────────────────────────────────────────

@bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check())), 200


@bp.get("/describe")
def describe():
	return jsonify(_run(_svc.describe())), 200


# ── Ledger Seed (test/dev only) ───────────────────────────────────────────────

@bp.post("/ledger/seed")
def seed_ledger():
	"""Inject a ledger snapshot for a given date (dev/test use)."""
	body = request.get_json(force=True) or {}
	try:
		_svc.seed_ledger(
			tenant_id=_tenant(),
			as_of_date=body["as_of_date"],
			data=body["data"],
		)
		return jsonify({"ok": True, "as_of_date": body["as_of_date"]}), 201
	except KeyError as exc:
		return _err(f"missing field: {exc}")


# ── Quarterly Return ──────────────────────────────────────────────────────────

@bp.get("/returns/quarterly")
def quarterly_return():
	"""Generate SASRA quarterly prudential return (Forms 1-5)."""
	try:
		year = int(request.args["year"])
		quarter = int(request.args["quarter"])
	except (KeyError, ValueError) as exc:
		return _err(f"year and quarter (1-4) required: {exc}")
	try:
		result = _run(_svc.generate_quarterly_return(_tenant(), year, quarter))
		return jsonify(result.model_dump(mode="json")), 200
	except AssertionError as exc:
		return _err(str(exc))
	except Exception as exc:
		_log.error("quarterly_return error: %s", exc)
		return _err(str(exc), 500)


# ── Annual Report ─────────────────────────────────────────────────────────────

@bp.get("/returns/annual")
def annual_return():
	try:
		year = int(request.args["year"])
	except (KeyError, ValueError):
		return _err("year required")
	try:
		result = _run(_svc.generate_annual_report(_tenant(), year))
		return jsonify(result.model_dump(mode="json")), 200
	except Exception as exc:
		_log.error("annual_return error: %s", exc)
		return _err(str(exc), 500)


# ── Capital Adequacy ──────────────────────────────────────────────────────────

@bp.get("/ratios/capital-adequacy")
def capital_adequacy():
	as_of = request.args.get("as_of_date")
	result = _run(_svc.calculate_capital_adequacy(_tenant(), as_of))
	return jsonify(result.model_dump(mode="json")), 200


# ── Liquidity ─────────────────────────────────────────────────────────────────

@bp.get("/ratios/liquidity")
def liquidity():
	as_of = request.args.get("as_of_date")
	result = _run(_svc.calculate_liquidity_ratio(_tenant(), as_of))
	return jsonify(result.model_dump(mode="json")), 200


# ── Loan to Deposit ───────────────────────────────────────────────────────────

@bp.get("/ratios/loan-to-deposit")
def ldr():
	as_of = request.args.get("as_of_date")
	result = _run(_svc.calculate_loan_to_deposit_ratio(_tenant(), as_of))
	return jsonify({"loan_to_deposit_ratio_pct": str(result), "maximum_pct": "70.00"}), 200


# ── NPL & PAR ─────────────────────────────────────────────────────────────────

@bp.get("/ratios/npl")
def npl_ratio():
	as_of = request.args.get("as_of_date")
	result = _run(_svc.calculate_npl_ratio(_tenant(), as_of))
	return jsonify({"npl_ratio_pct": str(result)}), 200


@bp.get("/ratios/par")
def par():
	as_of = request.args.get("as_of_date")
	days = int(request.args.get("days", 30))
	result = _run(_svc.calculate_par(_tenant(), as_of, days))
	return jsonify({"par_pct": str(result), "days": days}), 200


# ── Loan Classification ───────────────────────────────────────────────────────

@bp.get("/loan-classification")
def loan_classification():
	as_of = request.args.get("as_of_date")
	result = _run(_svc.classify_loan_portfolio(_tenant(), as_of))
	return jsonify(result.model_dump(mode="json")), 200


@bp.get("/provisions/required")
def required_provisions():
	as_of = request.args.get("as_of_date")
	result = _run(_svc.calculate_required_provisions(_tenant(), as_of))
	return jsonify({"required_provisions": str(result)}), 200


@bp.get("/provisions/coverage")
def provisioning_coverage():
	as_of = request.args.get("as_of_date")
	result = _run(_svc.calculate_provisioning_coverage(_tenant(), as_of))
	return jsonify({"provisioning_coverage_pct": str(result)}), 200


# ── Compliance ────────────────────────────────────────────────────────────────

@bp.get("/compliance")
def compliance():
	as_of = request.args.get("as_of_date")
	result = _run(_svc.check_regulatory_compliance(_tenant(), as_of))
	return jsonify(result.model_dump(mode="json")), 200


@bp.get("/compliance/dashboard")
def dashboard():
	as_of = request.args.get("as_of_date")
	result = _run(_svc.get_compliance_dashboard(_tenant(), as_of))
	return jsonify(result.model_dump(mode="json")), 200


# ── Board Report ──────────────────────────────────────────────────────────────

@bp.get("/board-report")
def board_report():
	period = request.args.get("period")
	result = _run(_svc.generate_board_report(_tenant(), period))
	return jsonify(result), 200


# ── SASRA XML ─────────────────────────────────────────────────────────────────

@bp.get("/returns/xml")
def xml_return():
	try:
		year = int(request.args["year"])
		quarter = int(request.args["quarter"])
	except (KeyError, ValueError) as exc:
		return _err(f"year and quarter required: {exc}")
	try:
		xml = _run(_svc.generate_sasra_xml_return(_tenant(), year, quarter))
		from flask import Response
		return Response(xml, mimetype="application/xml"), 200
	except Exception as exc:
		_log.error("xml_return error: %s", exc)
		return _err(str(exc), 500)


# ── Filing Registry ───────────────────────────────────────────────────────────

@bp.post("/filings")
def file_return():
	body = request.get_json(force=True) or {}
	try:
		rt = ReturnType(body["return_type"])
		result = _run(_svc.file_return(
			tenant_id=_tenant(),
			return_type=rt,
			period=body["period"],
			data=body.get("data", {}),
			filing_officer=body["filing_officer"],
			submitted_at=body.get("submitted_at"),
		))
		return jsonify(result.model_dump(mode="json")), 201
	except (KeyError, ValueError) as exc:
		return _err(str(exc))
	except AssertionError as exc:
		return _err(str(exc))
	except Exception as exc:
		_log.error("file_return error: %s", exc)
		return _err(str(exc), 500)


@bp.get("/filings")
def filing_history():
	result = _run(_svc.get_filing_history(
		tenant_id=_tenant(),
		from_date=request.args.get("from_date"),
		to_date=request.args.get("to_date"),
	))
	items = [f.model_dump(mode="json") for f in result]
	return jsonify({"items": items, "total": len(items)}), 200


# ── Regulatory Calendar ───────────────────────────────────────────────────────

@bp.get("/calendar")
def regulatory_calendar():
	try:
		year = int(request.args.get("year", 0)) or None
	except ValueError:
		year = None
	result = _run(_svc.get_regulatory_calendar(_tenant(), year))
	items = [d.model_dump(mode="json") for d in result]
	return jsonify({"items": items, "total": len(items)}), 200


@bp.get("/calendar/pending")
def pending_filings():
	result = _run(_svc.get_pending_filings(_tenant()))
	items = [d.model_dump(mode="json") for d in result]
	return jsonify({"items": items, "total": len(items)}), 200


# ── Audit ─────────────────────────────────────────────────────────────────────

@bp.get("/audit")
def audit_events():
	result = _run(_svc.get_audit_events(_tenant()))
	return jsonify({"items": result, "total": len(result)}), 200
