"""Flask Blueprint views for APG Point of Sale."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from flask import Blueprint, g, jsonify, request

from .service import PosService

bp = Blueprint("retail_pos_views", __name__, url_prefix="/retail-pos")
_svc = PosService()


def _tenant_id() -> str:
	return getattr(g, "tenant_id", request.headers.get("X-Tenant-ID", "default"))


def has_access(permission: str) -> Callable:
	def decorator(fn: Callable) -> Callable:
		@wraps(fn)
		def wrapper(*args: Any, **kwargs: Any) -> Any:
			perms: set[str] = getattr(g, "permissions", set())
			if permission not in perms and "superadmin" not in perms:
				return jsonify({"error": "forbidden", "required_permission": permission}), 403
			return fn(*args, **kwargs)
		return wrapper
	return decorator


def _run(coro: Any) -> Any:
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


@bp.get("/dashboard")
@has_access("retail_pos:view")
def dashboard() -> Any:
	tid = _tenant_id()
	sessions = _run(_svc.list_sessions(tid, status="open"))
	return jsonify({"tenant_id": tid, "open_sessions": len(sessions), "sessions": [s.model_dump() for s in sessions]})


@bp.get("/terminals")
@has_access("retail_pos:admin")
def list_terminals() -> Any:
	tid = _tenant_id()
	store_id = request.args.get("store_id")
	recs = _run(_svc.list_terminals(tid, store_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/terminals")
@has_access("retail_pos:admin")
def register_terminal() -> Any:
	from .models import PosTerminalCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.register_terminal(PosTerminalCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/terminals/<terminal_id>/heartbeat")
@has_access("retail_pos:transact")
def terminal_heartbeat(terminal_id: str) -> Any:
	tid = _tenant_id()
	rec = _run(_svc.heartbeat_terminal(tid, terminal_id))
	if rec is None:
		return jsonify({"error": "not_found"}), 404
	return jsonify(rec.model_dump())


@bp.get("/sessions")
@has_access("retail_pos:view")
def list_sessions() -> Any:
	tid = _tenant_id()
	store_id = request.args.get("store_id")
	status = request.args.get("status")
	recs = _run(_svc.list_sessions(tid, store_id, status))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/sessions")
@has_access("retail_pos:transact")
def open_session() -> Any:
	from .models import PosSessionCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.open_session(PosSessionCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/sessions/<session_id>")
@has_access("retail_pos:view")
def session_detail(session_id: str) -> Any:
	tid = _tenant_id()
	summary = _run(_svc.session_summary(tid, session_id))
	if not summary:
		return jsonify({"error": "not_found"}), 404
	return jsonify(summary)


@bp.post("/sessions/<session_id>/close")
@has_access("retail_pos:transact")
def close_session(session_id: str) -> Any:
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	try:
		rec = _run(_svc.close_session(tid, session_id, float(body.get("closing_cash", 0))))
		if rec is None:
			return jsonify({"error": "not_found"}), 404
		return jsonify(rec.model_dump())
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/sessions/<session_id>/suspend")
@has_access("retail_pos:transact")
def suspend_session(session_id: str) -> Any:
	tid = _tenant_id()
	rec = _run(_svc.suspend_session(tid, session_id))
	if rec is None:
		return jsonify({"error": "not_found"}), 404
	return jsonify(rec.model_dump())


@bp.get("/transactions")
@has_access("retail_pos:view")
def list_transactions() -> Any:
	tid = _tenant_id()
	session_id = request.args.get("session_id")
	recs = _run(_svc.list_transactions(tid, session_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/transactions")
@has_access("retail_pos:transact")
def post_transaction() -> Any:
	from .models import PosTransactionCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.post_transaction(PosTransactionCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/transactions/<transaction_id>")
@has_access("retail_pos:view")
def transaction_detail(transaction_id: str) -> Any:
	tid = _tenant_id()
	rec = _run(_svc.get_transaction(tid, transaction_id))
	if rec is None:
		return jsonify({"error": "not_found"}), 404
	return jsonify(rec.model_dump())


@bp.post("/voids")
@has_access("retail_pos:void")
def void_transaction() -> Any:
	from .models import PosVoidCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.void_transaction(PosVoidCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/cash")
@has_access("retail_pos:view")
def list_cash_events() -> Any:
	tid = _tenant_id()
	session_id = request.args.get("session_id", "")
	recs = _run(_svc.list_cash_events(tid, session_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/cash")
@has_access("retail_pos:transact")
def record_cash_event() -> Any:
	from .models import PosCashEventCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.record_cash_event(PosCashEventCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/reconcile")
@has_access("retail_pos:reconcile")
def create_reconciliation() -> Any:
	from .models import PosReconciliationCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.create_reconciliation(PosReconciliationCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/reconcile/<reconciliation_id>/approve")
@has_access("retail_pos:admin")
def approve_reconciliation(reconciliation_id: str) -> Any:
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	rec = _run(_svc.approve_reconciliation(tid, reconciliation_id, body.get("by", "system")))
	if rec is None:
		return jsonify({"error": "not_found"}), 404
	return jsonify(rec.model_dump())


@bp.post("/receipts")
@has_access("retail_pos:transact")
def issue_receipt() -> Any:
	from .models import PosReceiptCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.issue_receipt(PosReceiptCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400
