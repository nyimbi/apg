"""Flask Blueprint views for fintech_treasury capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

blueprint = Blueprint("fintech_treasury", __name__, url_prefix="/fintech/treasury")

_ROUTES = [
	{"path": "/", "methods": ["GET"], "description": "Capability index"},
	{"path": "/cash-position", "methods": ["GET"], "description": "Cash position dashboard"},
	{"path": "/hedge-instruments", "methods": ["GET"], "description": "Hedge instrument list"},
	{"path": "/intercompany-loans", "methods": ["GET"], "description": "Intercompany loans"},
	{"path": "/kpi", "methods": ["GET"], "description": "Treasury KPI dashboard"},
	{"path": "/fx-exposure", "methods": ["GET"], "description": "FX exposure report"},
	{"path": "/analytics", "methods": ["GET"], "description": "Treasury analytics"},
]


@blueprint.get("/")
def index():
	return jsonify({
		"capability": "fintech_treasury",
		"display_name": "Corporate Treasury Management",
		"version": "1.0.0",
		"routes": _ROUTES,
	})


@blueprint.get("/cash-position")
def cash_position():
	return jsonify({
		"view": "cash_position",
		"entity_id": request.args.get("entity_id"),
		"as_of_date": request.args.get("as_of_date"),
		"currencies": request.args.getlist("currency"),
	})


@blueprint.get("/hedge-instruments")
def hedge_instruments():
	return jsonify({
		"view": "hedge_instrument_list",
		"entity_id": request.args.get("entity_id"),
		"status": request.args.get("status"),
	})


@blueprint.get("/hedge-instruments/<hedge_id>")
def hedge_instrument_detail(hedge_id: str):
	return jsonify({"view": "hedge_instrument_detail", "hedge_id": hedge_id})


@blueprint.get("/intercompany-loans")
def intercompany_loans():
	return jsonify({
		"view": "intercompany_loan_list",
		"entity_id": request.args.get("entity_id"),
	})


@blueprint.get("/kpi")
def kpi_dashboard():
	return jsonify({
		"view": "treasury_kpi_dashboard",
		"entity_id": request.args.get("entity_id"),
	})


@blueprint.get("/fx-exposure")
def fx_exposure():
	return jsonify({
		"view": "fx_exposure_report",
		"entity_id": request.args.get("entity_id"),
		"as_of_date": request.args.get("as_of_date"),
	})


@blueprint.get("/analytics")
def analytics():
	return jsonify({
		"view": "treasury_analytics",
		"entity_id": request.args.get("entity_id"),
		"period": request.args.get("period", "2026-06"),
	})
