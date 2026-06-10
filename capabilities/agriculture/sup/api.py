"""Agricultural Supply Chain Flask Blueprint — agr_sup."""
from __future__ import annotations
import logging
from flask import Blueprint, jsonify, request
from .service import SupplyChainService

_log = logging.getLogger(__name__)
bp = Blueprint("agr_sup", __name__, url_prefix="/api/agriculture/sup")
_svc: dict[str, SupplyChainService] = {}


def _get_svc(t: str = "default") -> SupplyChainService:
	if t not in _svc:
		_svc[t] = SupplyChainService(tenant_id=t)
	return _svc[t]


def _t() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@bp.get("/health")
async def health():
	return jsonify(await _get_svc(_t()).health_check()), 200


@bp.get("/batches")
async def list_batches():
	svc = _get_svc(_t())
	items = await svc.list_batches(
		farmer_id=request.args.get("farmer_id"),
		buyer_id=request.args.get("buyer_id"),
		status=request.args.get("status"),
		product_type=request.args.get("product_type"),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.get("/batches/<batch_id>")
async def get_batch(batch_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_batch(batch_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/batches")
async def create_batch():
	try:
		return jsonify(await _get_svc(_t()).create_batch(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/batches/<batch_id>")
async def update_batch(batch_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_batch(batch_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/batches/<batch_id>")
async def delete_batch(batch_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_batch(batch_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/batches/<batch_id>/trace")
async def batch_trace(batch_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_batch_trace(batch_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/batches/<batch_id>/export-readiness")
async def export_readiness(batch_id: str):
	try:
		return jsonify(await _get_svc(_t()).check_export_readiness(batch_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/procurement")
async def list_procurement():
	svc = _get_svc(_t())
	items = await svc.list_procurement(supplier_id=request.args.get("supplier_id"), status=request.args.get("status"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/procurement")
async def create_procurement():
	try:
		return jsonify(await _get_svc(_t()).create_procurement_order(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/procurement/<order_id>")
async def update_procurement(order_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_procurement_order(order_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/procurement/<order_id>")
async def delete_procurement(order_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_procurement_order(order_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/cold-chain")
async def list_cold_chain():
	svc = _get_svc(_t())
	items = await svc.list_cold_chain_logs(batch_id=request.args.get("batch_id"), status=request.args.get("status"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/cold-chain")
async def log_cold_chain():
	try:
		return jsonify(await _get_svc(_t()).log_cold_chain(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/cold-chain/<batch_id>/summary")
async def cold_chain_summary(batch_id: str):
	return jsonify(await _get_svc(_t()).get_cold_chain_summary(batch_id)), 200


@bp.get("/export-docs")
async def list_export_docs():
	svc = _get_svc(_t())
	items = await svc.list_export_docs(batch_id=request.args.get("batch_id"), document_type=request.args.get("document_type"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/export-docs")
async def create_export_doc():
	try:
		return jsonify(await _get_svc(_t()).create_export_doc(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/export-docs/<doc_id>")
async def delete_export_doc(doc_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_export_doc(doc_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/audit")
async def get_audit():
	events = await _get_svc(_t()).get_audit_events(int(request.args.get("limit", 100)))
	return jsonify({"events": events, "count": len(events)}), 200
