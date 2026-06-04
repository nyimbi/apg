"""Flask Blueprint REST API for fintech_switch capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

blueprint = Blueprint("fintech_switch_api", __name__, url_prefix="/api/v1/fintech/switch")


def _svc():
	from .service import PaymentSwitchService
	return PaymentSwitchService()


# ── Routing ───────────────────────────────────────────────────────────────────

@blueprint.get("/transactions")
def list_transactions():
	import asyncio
	data = {k: v for k, v in request.args.items()}
	limit = int(request.args.get("limit", 100))
	svc = _svc()
	result = asyncio.run(
		svc.transaction_history_switch(data, limit=limit)
	)
	return jsonify(result)


@blueprint.post("/transactions/route")
def route_transaction():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.route_transaction(data["transaction_data"], data["routing_rules"])
	)
	return jsonify(result), 201


@blueprint.get("/transactions/<txn_id>")
def get_transaction(txn_id: str):
	import asyncio
	svc = _svc()
	result = asyncio.run(
		svc._store.get("switch_transactions", txn_id)
	)
	return jsonify(result or {})


@blueprint.put("/transactions/<txn_id>")
def update_transaction(txn_id: str):
	return jsonify({"message": "use domain-specific endpoints to update switch transactions"})


@blueprint.delete("/transactions/<txn_id>")
def delete_transaction(txn_id: str):
	return jsonify({"message": "switch transactions are immutable"}), 405


# ── Authorisation ─────────────────────────────────────────────────────────────

@blueprint.post("/authorise")
def authorise():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.switch_authorisation(
			pan_or_phone=data["pan_or_phone"],
			amount=float(data["amount"]),
			merchant_id=data["merchant_id"],
			currency=data["currency"],
			transaction_type=data.get("transaction_type", "purchase"),
			channel=data.get("channel", "pos"),
		)
	)
	return jsonify(result), 201


@blueprint.post("/authorise/cnp")
def card_not_present():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.card_not_present_auth(
			token=data["token"],
			amount=float(data["amount"]),
			cvv_result=data["cvv_result"],
			avs_result=data["avs_result"],
		)
	)
	return jsonify(result), 201


@blueprint.post("/authorise/3ds")
def authenticate_3ds():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.authenticate_3ds(
			pan=data["pan"],
			amount=float(data["amount"]),
			eci=data["eci"],
			cavv=data.get("cavv", ""),
		)
	)
	return jsonify(result), 201


# ── Schemes ───────────────────────────────────────────────────────────────────

@blueprint.get("/schemes")
def list_schemes():
	return jsonify({"schemes": []})


@blueprint.post("/schemes")
def register_scheme():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.scheme_registration(
			scheme_name=data["scheme_name"],
			credentials=data["credentials"],
			effective_date=data["effective_date"],
		)
	)
	return jsonify(result), 201


@blueprint.get("/schemes/<scheme_id>")
def get_scheme(scheme_id: str):
	import asyncio
	svc = _svc()
	result = asyncio.run(
		svc._store.get("switch_schemes", scheme_id)
	)
	return jsonify(result or {})


# ── Clearing ──────────────────────────────────────────────────────────────────

@blueprint.post("/clearing")
def generate_clearing():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.clearing_file_generation(
			settlement_date=data["settlement_date"],
			scheme=data["scheme"],
		)
	)
	return jsonify(result), 201


@blueprint.get("/clearing/<file_id>")
def get_clearing_file(file_id: str):
	import asyncio
	svc = _svc()
	result = asyncio.run(
		svc._store.get("clearing_files", file_id)
	)
	return jsonify(result or {})


# ── Operations ────────────────────────────────────────────────────────────────

@blueprint.post("/transactions/<txn_id>/replay")
def replay_transaction(txn_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.transaction_replay(txn_id, target_system=data["target_system"])
	)
	return jsonify(result)


@blueprint.post("/failover")
def failover():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.downtime_failover(
			primary_route=data["primary_route"],
			failover_route=data["failover_route"],
		)
	)
	return jsonify(result)


@blueprint.post("/fraud/velocity-check")
def velocity_check():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.fraud_velocity_check(
			pan_or_phone=data["pan_or_phone"],
			window_seconds=int(data["window_seconds"]),
			max_attempts=int(data["max_attempts"]),
		)
	)
	return jsonify(result)


@blueprint.post("/compliance/check")
def compliance_check():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.scheme_compliance_check(
			transaction_id=data["transaction_id"],
			scheme=data["scheme"],
		)
	)
	return jsonify(result)


@blueprint.get("/health")
def health():
	import asyncio
	svc = _svc()
	result = asyncio.run(svc.switch_health_check())
	return jsonify(result)


@blueprint.get("/analytics")
def analytics():
	import asyncio
	period = request.args.get("period", "2026-06")
	svc = _svc()
	result = asyncio.run(svc.switch_analytics(period))
	return jsonify(result)


@blueprint.post("/simulate")
def simulate():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.switch_simulator(
			scenario=data["scenario"],
			expected_response=data["expected_response"],
		)
	)
	return jsonify(result)
