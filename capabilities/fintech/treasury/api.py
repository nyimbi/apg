"""Flask Blueprint REST API for fintech_treasury capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

blueprint = Blueprint("fintech_treasury_api", __name__, url_prefix="/api/v1/fintech/treasury")


def _svc():
	from .service import CorporateTreasuryService
	return CorporateTreasuryService()


# ── Cash position ─────────────────────────────────────────────────────────────

@blueprint.post("/cash-position")
def cash_position():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.cash_position(
			entity_id=data["entity_id"],
			as_of_date=data["as_of_date"],
			currencies=data["currencies"],
		)
	)
	return jsonify(result)


@blueprint.post("/liquidity-forecast")
def liquidity_forecast():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.liquidity_forecast(
			entity_id=data["entity_id"],
			days=data.get("days", 90),
			method=data.get("method", "ar_ap_driven"),
		)
	)
	return jsonify(result)


@blueprint.get("/fx-exposure")
def fx_exposure():
	import asyncio
	entity_id = request.args["entity_id"]
	as_of_date = request.args["as_of_date"]
	svc = _svc()
	result = asyncio.run(
		svc.fx_exposure_report(entity_id=entity_id, as_of_date=as_of_date)
	)
	return jsonify(result)


# ── Hedge instruments ─────────────────────────────────────────────────────────

@blueprint.get("/hedge-instruments")
def list_hedge_instruments():
	return jsonify({"instruments": []})


@blueprint.post("/hedge-instruments")
def create_hedge_instrument():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.hedge_instrument_create(
			instrument_type=data["instrument_type"],
			notional=float(data["notional"]),
			currency_pair=data["currency_pair"],
			strike=float(data["strike"]),
			maturity=data["maturity"],
			entity_id=data.get("entity_id"),
			counterparty_id=data.get("counterparty_id"),
		)
	)
	return jsonify(result), 201


@blueprint.get("/hedge-instruments/<hedge_id>")
def get_hedge_instrument(hedge_id: str):
	import asyncio
	svc = _svc()
	result = asyncio.run(
		svc._store.get("hedge_instruments", hedge_id)
	)
	return jsonify(result or {})


@blueprint.put("/hedge-instruments/<hedge_id>")
def test_hedge_effectiveness(hedge_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.hedge_effectiveness_test(
			hedge_id=hedge_id,
			period=data["period"],
			method=data.get("method", "dollar_offset"),
		)
	)
	return jsonify(result)


# ── Intercompany loans ────────────────────────────────────────────────────────

@blueprint.get("/intercompany-loans")
def list_loans():
	return jsonify({"loans": []})


@blueprint.post("/intercompany-loans")
def create_loan():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.intercompany_loan(
			lender_entity=data["lender_entity"],
			borrower_entity=data["borrower_entity"],
			amount=float(data["amount"]),
			currency=data["currency"],
			rate=float(data["rate"]),
			tenor_months=int(data["tenor_months"]),
		)
	)
	return jsonify(result), 201


@blueprint.get("/intercompany-loans/<loan_id>")
def get_loan(loan_id: str):
	import asyncio
	svc = _svc()
	result = asyncio.run(
		svc._store.get("intercompany_loans", loan_id)
	)
	return jsonify(result or {})


# ── Dealing ───────────────────────────────────────────────────────────────────

@blueprint.post("/fx-forward")
def fx_forward():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.fx_forward_booking(
			entity_id=data["entity_id"],
			buy_currency=data["buy_currency"],
			sell_currency=data["sell_currency"],
			amount=float(data["amount"]),
			settlement_date=data["settlement_date"],
			forward_rate=float(data["forward_rate"]),
		)
	)
	return jsonify(result), 201


@blueprint.post("/mm-placement")
def mm_placement():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.money_market_placement(
			entity_id=data["entity_id"],
			bank_id=data["bank_id"],
			amount=float(data["amount"]),
			currency=data["currency"],
			tenor_days=int(data["tenor_days"]),
			rate=float(data["rate"]),
		)
	)
	return jsonify(result), 201


@blueprint.post("/payment-factory")
def payment_factory():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.payment_factory(
			entity_id=data["entity_id"],
			payments=data["payments"],
			payment_date=data["payment_date"],
		)
	)
	return jsonify(result), 201


@blueprint.post("/netting")
def netting():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.netting_calculation(
			entities=data["entities"],
			currency=data["currency"],
			period=data["period"],
		)
	)
	return jsonify(result)


# ── KPI and reporting ─────────────────────────────────────────────────────────

@blueprint.get("/kpi")
def kpi():
	import asyncio
	entity_id = request.args["entity_id"]
	svc = _svc()
	result = asyncio.run(svc.treasury_kpi_dashboard(entity_id))
	return jsonify(result)


@blueprint.post("/scenario-analysis")
def scenario_analysis():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.scenario_analysis(
			entity_id=data["entity_id"],
			scenario_type=data["scenario_type"],
			parameters=data.get("parameters", {}),
		)
	)
	return jsonify(result)


@blueprint.post("/covenant-monitoring")
def covenant_monitoring():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.covenant_monitoring(
			facility_id=data["facility_id"],
			financial_ratios=data["financial_ratios"],
		)
	)
	return jsonify(result)


@blueprint.get("/analytics")
def analytics():
	import asyncio
	entity_id = request.args["entity_id"]
	period = request.args.get("period", "2026-06")
	svc = _svc()
	result = asyncio.run(
		svc.treasury_analytics(entity_id=entity_id, period=period)
	)
	return jsonify(result)


@blueprint.get("/regulatory-capital")
def regulatory_capital():
	import asyncio
	entity_id = request.args["entity_id"]
	period = request.args.get("period", "2026")
	svc = _svc()
	result = asyncio.run(
		svc.regulatory_capital_report(entity_id=entity_id, period=period)
	)
	return jsonify(result)
