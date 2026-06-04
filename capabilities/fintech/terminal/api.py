"""Flask Blueprint REST API for fintech_terminal capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

blueprint = Blueprint("fintech_terminal_api", __name__, url_prefix="/api/v1/fintech/terminal")


def _svc():
	from .service import TerminalBankingService
	return TerminalBankingService()


# ── Terminals CRUD ────────────────────────────────────────────────────────────

@blueprint.get("/terminals")
def list_terminals():
	return jsonify({"terminals": [], "count": 0})


@blueprint.post("/terminals")
def create_terminal():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.register_terminal(
			terminal_id=data["terminal_id"],
			location=data["location"],
			agent_id=data["agent_id"],
			terminal_type=data["terminal_type"],
			connectivity=data["connectivity"],
			serial_number=data.get("serial_number"),
			merchant_id=data.get("merchant_id"),
			model=data.get("model"),
			tenant_id=data.get("tenant_id"),
		)
	)
	return jsonify(result), 201


@blueprint.get("/terminals/<terminal_id>")
def get_terminal(terminal_id: str):
	import asyncio
	svc = _svc()
	result = asyncio.run(svc._get_terminal(terminal_id))
	return jsonify(result)


@blueprint.post("/terminals/<terminal_id>/activate")
def activate_terminal(terminal_id: str):
	import asyncio
	data = request.get_json(force=True) or {}
	svc = _svc()
	result = asyncio.run(
		svc.activate_terminal(
			terminal_id,
			activated_by=data["activated_by"],
			pci_dss_compliant=data.get("pci_dss_compliant", True),
			tamper_detection_enabled=data.get("tamper_detection_enabled", True),
			software_integrity_verified=data.get("software_integrity_verified", True),
		)
	)
	return jsonify(result)


# ── Transactions ──────────────────────────────────────────────────────────────

@blueprint.post("/terminals/<terminal_id>/transactions")
def post_transaction(terminal_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.terminal_transaction(
			terminal_id,
			transaction_type=data["transaction_type"],
			amount=float(data["amount"]),
			currency=data["currency"],
			customer_id=data["customer_id"],
			reference=data["reference"],
			metadata=data.get("metadata"),
		)
	)
	return jsonify(result), 201


@blueprint.post("/terminals/<terminal_id>/deposit")
def cash_deposit(terminal_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.cash_deposit(terminal_id, data["customer_id"], float(data["amount"]), data["currency"])
	)
	return jsonify(result), 201


@blueprint.post("/terminals/<terminal_id>/withdrawal")
def cash_withdrawal(terminal_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.cash_withdrawal(
			terminal_id,
			data["customer_id"],
			float(data["amount"]),
			data["currency"],
			pin_verified=data.get("pin_verified", True),
		)
	)
	return jsonify(result), 201


@blueprint.post("/terminals/<terminal_id>/transfer")
def fund_transfer(terminal_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.fund_transfer_terminal(
			terminal_id,
			from_account=data["from_account"],
			to_account=data["to_account"],
			amount=float(data["amount"]),
			currency=data.get("currency", "KES"),
		)
	)
	return jsonify(result), 201


@blueprint.post("/terminals/<terminal_id>/bill-payment")
def bill_payment(terminal_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.bill_payment_terminal(
			terminal_id,
			customer_id=data["customer_id"],
			biller_code=data["biller_code"],
			amount=float(data["amount"]),
			currency=data.get("currency", "KES"),
		)
	)
	return jsonify(result), 201


# ── Float ─────────────────────────────────────────────────────────────────────

@blueprint.post("/terminals/<terminal_id>/float")
def manage_float(terminal_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.float_management(
			terminal_id,
			float_amount=float(data["float_amount"]),
			operation_type=data["operation_type"],
			authorised_by=data.get("authorised_by"),
		)
	)
	return jsonify(result)


# ── Operations ────────────────────────────────────────────────────────────────

@blueprint.get("/terminals/<terminal_id>/health")
def terminal_health(terminal_id: str):
	import asyncio
	svc = _svc()
	result = asyncio.run(svc.terminal_health_check(terminal_id))
	return jsonify(result)


@blueprint.post("/terminals/<terminal_id>/sync")
def offline_sync(terminal_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.offline_queue_sync(terminal_id, data["queued_transactions"])
	)
	return jsonify(result)


@blueprint.get("/terminals/<terminal_id>/reconciliation")
def reconciliation(terminal_id: str):
	import asyncio
	recon_date = request.args.get("date", "")
	svc = _svc()
	result = asyncio.run(
		svc.terminal_reconciliation(terminal_id, recon_date)
	)
	return jsonify(result)


@blueprint.get("/terminals/<terminal_id>/commission")
def commission_report(terminal_id: str):
	import asyncio
	period = request.args.get("period", "")
	svc = _svc()
	result = asyncio.run(
		svc.terminal_commission_report(terminal_id, period)
	)
	return jsonify(result)


@blueprint.post("/terminals/<terminal_id>/fraud-alert")
def fraud_alert(terminal_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.fraud_alert_terminal(terminal_id, data["event_type"], data.get("details", {}))
	)
	return jsonify(result), 201


@blueprint.get("/analytics")
def analytics():
	import asyncio
	network_id = request.args.get("network_id", "default")
	period = request.args.get("period", "2026-06")
	svc = _svc()
	result = asyncio.run(
		svc.terminal_analytics(network_id, period)
	)
	return jsonify(result)


@blueprint.get("/regulatory-report")
def regulatory_report():
	import asyncio
	period = request.args.get("period", "")
	jurisdiction = request.args.get("jurisdiction", "CBK")
	svc = _svc()
	result = asyncio.run(
		svc.regulatory_report(period, jurisdiction)
	)
	return jsonify(result)
