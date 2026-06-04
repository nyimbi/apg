"""Flask Blueprint views for fintech_switch capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

blueprint = Blueprint("fintech_switch", __name__, url_prefix="/fintech/switch")

_ROUTES = [
	{"path": "/", "methods": ["GET"], "description": "Capability index"},
	{"path": "/transactions", "methods": ["GET"], "description": "Transaction list"},
	{"path": "/transactions/<txn_id>", "methods": ["GET"], "description": "Transaction detail"},
	{"path": "/schemes", "methods": ["GET"], "description": "Registered schemes"},
	{"path": "/clearing", "methods": ["GET"], "description": "Clearing files"},
	{"path": "/analytics", "methods": ["GET"], "description": "Switch analytics"},
	{"path": "/health", "methods": ["GET"], "description": "Switch health"},
]


@blueprint.get("/")
def index():
	return jsonify({
		"capability": "fintech_switch",
		"display_name": "Payment Switch",
		"version": "1.0.0",
		"routes": _ROUTES,
	})


@blueprint.get("/transactions")
def transaction_list():
	return jsonify({
		"view": "switch_transaction_list",
		"filters": {
			"network": request.args.get("network"),
			"channel": request.args.get("channel"),
			"date_from": request.args.get("date_from"),
			"date_to": request.args.get("date_to"),
		},
	})


@blueprint.get("/transactions/<txn_id>")
def transaction_detail(txn_id: str):
	return jsonify({"view": "switch_transaction_detail", "transaction_id": txn_id})


@blueprint.get("/schemes")
def schemes():
	return jsonify({"view": "scheme_list"})


@blueprint.get("/clearing")
def clearing():
	return jsonify({
		"view": "clearing_file_list",
		"settlement_date": request.args.get("date"),
		"scheme": request.args.get("scheme"),
	})


@blueprint.get("/analytics")
def analytics():
	return jsonify({
		"view": "switch_analytics",
		"period": request.args.get("period", "2026-06"),
	})


@blueprint.get("/health")
def health():
	return jsonify({"view": "switch_health"})
