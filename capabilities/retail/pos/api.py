"""REST API Blueprint for APG Point of Sale."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, g, jsonify, request

from .service import PosService
from .capability_contract import get_capability_contract, evaluate_capability_rules

api = Blueprint("retail_pos_api", __name__, url_prefix="/retail-pos/api/v1")
_svc = PosService()


def _tenant_id() -> str:
	return getattr(g, "tenant_id", request.headers.get("X-Tenant-ID", "default"))


def _run(coro: Any) -> Any:
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		return asyncio.run(coro)
	finally:
		loop.close()


def _err(msg: str, code: int = 400) -> Any:
	return jsonify({"error": msg, "status": code}), code


@api.get("/contract")
def contract() -> Any:
	"""Return capability contract. GET /retail-pos/api/v1/contract"""
	return jsonify(get_capability_contract(_tenant_id()))


@api.post("/rules/evaluate")
def evaluate_rules() -> Any:
	"""Evaluate rules. POST /retail-pos/api/v1/rules/evaluate"""
	return jsonify(evaluate_capability_rules(request.get_json(force=True) or {}))


# Terminals
@api.get("/terminals")
def list_terminals() -> Any:
	"""List terminals. GET /retail-pos/api/v1/terminals?store_id=<id>"""
	recs = _run(_svc.list_terminals(_tenant_id(), request.args.get("store_id")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/terminals")
def register_terminal() -> Any:
	"""Register terminal. POST /retail-pos/api/v1/terminals"""
	from .models import PosTerminalCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.register_terminal(PosTerminalCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.get("/terminals/<terminal_id>")
def get_terminal(terminal_id: str) -> Any:
	"""Get terminal. GET /retail-pos/api/v1/terminals/<terminal_id>"""
	rec = _run(_svc.get_terminal(_tenant_id(), terminal_id))
	return jsonify(rec.model_dump()) if rec else _err("not_found", 404)


@api.post("/terminals/<terminal_id>/heartbeat")
def terminal_heartbeat(terminal_id: str) -> Any:
	"""Terminal heartbeat. POST /retail-pos/api/v1/terminals/<terminal_id>/heartbeat"""
	rec = _run(_svc.heartbeat_terminal(_tenant_id(), terminal_id))
	return jsonify(rec.model_dump()) if rec else _err("not_found", 404)


# Sessions
@api.get("/sessions")
def list_sessions() -> Any:
	"""List sessions. GET /retail-pos/api/v1/sessions?store_id=<id>&status=<s>"""
	recs = _run(_svc.list_sessions(_tenant_id(), request.args.get("store_id"), request.args.get("status")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/sessions")
def open_session() -> Any:
	"""Open session. POST /retail-pos/api/v1/sessions"""
	from .models import PosSessionCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.open_session(PosSessionCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.get("/sessions/<session_id>")
def get_session(session_id: str) -> Any:
	"""Get session summary. GET /retail-pos/api/v1/sessions/<session_id>"""
	summary = _run(_svc.session_summary(_tenant_id(), session_id))
	return jsonify(summary) if summary else _err("not_found", 404)


@api.put("/sessions/<session_id>")
def update_session(session_id: str) -> Any:
	"""Update session status. PUT /retail-pos/api/v1/sessions/<session_id>"""
	from .models import PosSessionUpdate
	body = request.get_json(force=True) or {}
	try:
		status = body.get("status")
		if status == "closed":
			rec = _run(_svc.close_session(_tenant_id(), session_id, float(body.get("closing_cash", 0))))
		elif status == "suspended":
			rec = _run(_svc.suspend_session(_tenant_id(), session_id))
		elif status == "open":
			rec = _run(_svc.resume_session(_tenant_id(), session_id))
		else:
			return _err(f"unsupported status transition: {status}")
		return jsonify(rec.model_dump()) if rec else _err("not_found", 404)
	except Exception as exc:
		return _err(str(exc))


# Transactions
@api.get("/transactions")
def list_transactions() -> Any:
	"""List transactions. GET /retail-pos/api/v1/transactions?session_id=<id>"""
	recs = _run(_svc.list_transactions(_tenant_id(), request.args.get("session_id")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/transactions")
def post_transaction() -> Any:
	"""Post transaction. POST /retail-pos/api/v1/transactions"""
	from .models import PosTransactionCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.post_transaction(PosTransactionCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.get("/transactions/<transaction_id>")
def get_transaction(transaction_id: str) -> Any:
	"""Get transaction. GET /retail-pos/api/v1/transactions/<transaction_id>"""
	rec = _run(_svc.get_transaction(_tenant_id(), transaction_id))
	return jsonify(rec.model_dump()) if rec else _err("not_found", 404)


@api.delete("/transactions/<transaction_id>")
def void_transaction_by_id(transaction_id: str) -> Any:
	"""Void transaction. DELETE /retail-pos/api/v1/transactions/<transaction_id>"""
	from .models import PosVoidCreate
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant_id())
	body["original_transaction_id"] = transaction_id
	try:
		rec = _run(_svc.void_transaction(PosVoidCreate(**body)))
		return jsonify(rec.model_dump())
	except Exception as exc:
		return _err(str(exc))


# Voids
@api.post("/voids")
def create_void() -> Any:
	"""Post a void. POST /retail-pos/api/v1/voids"""
	from .models import PosVoidCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.void_transaction(PosVoidCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


# Cash events
@api.get("/cash")
def list_cash_events() -> Any:
	"""List cash events. GET /retail-pos/api/v1/cash?session_id=<id>"""
	session_id = request.args.get("session_id", "")
	recs = _run(_svc.list_cash_events(_tenant_id(), session_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/cash")
def record_cash_event() -> Any:
	"""Record cash event. POST /retail-pos/api/v1/cash"""
	from .models import PosCashEventCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.record_cash_event(PosCashEventCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


# Reconciliation
@api.post("/reconcile")
def create_reconciliation() -> Any:
	"""Create reconciliation. POST /retail-pos/api/v1/reconcile"""
	from .models import PosReconciliationCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.create_reconciliation(PosReconciliationCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.get("/reconcile/<reconciliation_id>")
def get_reconciliation(reconciliation_id: str) -> Any:
	"""Get reconciliation. GET /retail-pos/api/v1/reconcile/<reconciliation_id>"""
	rec = _run(_svc.get_reconciliation(_tenant_id(), reconciliation_id))
	return jsonify(rec.model_dump()) if rec else _err("not_found", 404)


@api.put("/reconcile/<reconciliation_id>/approve")
def approve_reconciliation(reconciliation_id: str) -> Any:
	"""Approve reconciliation. PUT /retail-pos/api/v1/reconcile/<reconciliation_id>/approve"""
	body = request.get_json(force=True) or {}
	rec = _run(_svc.approve_reconciliation(_tenant_id(), reconciliation_id, body.get("by","system")))
	return jsonify(rec.model_dump()) if rec else _err("not_found", 404)


# Receipts
@api.post("/receipts")
def issue_receipt() -> Any:
	"""Issue receipt. POST /retail-pos/api/v1/receipts"""
	from .models import PosReceiptCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.issue_receipt(PosReceiptCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.get("/receipts")
def list_receipts() -> Any:
	"""List receipts. GET /retail-pos/api/v1/receipts?transaction_id=<id>"""
	transaction_id = request.args.get("transaction_id", "")
	recs = _run(_svc.list_receipts(_tenant_id(), transaction_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})
