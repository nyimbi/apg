"""Crop Insurance Flask Blueprint — agr_ins."""
from __future__ import annotations
import logging
from flask import Blueprint, jsonify, request
from .service import CropInsuranceService

_log = logging.getLogger(__name__)
bp = Blueprint("agr_ins", __name__, url_prefix="/api/agriculture/ins")
_svc: dict[str, CropInsuranceService] = {}


def _get_svc(t: str = "default") -> CropInsuranceService:
	if t not in _svc:
		_svc[t] = CropInsuranceService(tenant_id=t)
	return _svc[t]


def _t() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@bp.get("/health")
async def health():
	return jsonify(await _get_svc(_t()).health_check()), 200


@bp.get("/products")
async def list_products():
	svc = _get_svc(_t())
	active_str = request.args.get("active")
	active = None if active_str is None else active_str.lower() == "true"
	items = await svc.list_products(trigger_type=request.args.get("trigger_type"), active=active)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/products")
async def create_product():
	try:
		return jsonify(await _get_svc(_t()).create_product(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/products/<product_id>")
async def get_product(product_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_product(product_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/products/<product_id>")
async def update_product(product_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_product(product_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/products/<product_id>")
async def delete_product(product_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_product(product_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/premium-calc")
async def calc_premium():
	try:
		svc = _get_svc(_t())
		result = await svc.calculate_premium(
			product_id=request.args.get("product_id", ""),
			farmer_id=request.args.get("farmer_id", ""),
			sum_insured=float(request.args.get("sum_insured", 0)),
			risk_modifier=float(request.args.get("risk_modifier", 1.0)),
		)
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/policies")
async def list_policies():
	svc = _get_svc(_t())
	items = await svc.list_policies(
		farmer_id=request.args.get("farmer_id"),
		status=request.args.get("status"),
		season=request.args.get("season"),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/policies")
async def create_policy():
	try:
		return jsonify(await _get_svc(_t()).create_policy(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/policies/<policy_id>")
async def get_policy(policy_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_policy(policy_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/policies/<policy_id>")
async def update_policy(policy_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_policy(policy_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/policies/<policy_id>/activate")
async def activate_policy(policy_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _get_svc(_t()).activate_policy(policy_id, body.get("payment_reference", ""))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/policies/<policy_id>")
async def delete_policy(policy_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_policy(policy_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/claims")
async def list_claims():
	svc = _get_svc(_t())
	items = await svc.list_claims(
		policy_id=request.args.get("policy_id"),
		farmer_id=request.args.get("farmer_id"),
		status=request.args.get("status"),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/claims")
async def submit_claim():
	try:
		return jsonify(await _get_svc(_t()).submit_claim(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/claims/<claim_id>")
async def get_claim(claim_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_claim(claim_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/claims/<claim_id>")
async def update_claim(claim_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_claim(claim_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/claims/<claim_id>/verify")
async def verify_claim(claim_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _get_svc(_t()).verify_trigger(claim_id, float(body["verified_value"]), body.get("source", "manual"))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/portfolio")
async def portfolio_stats():
	return jsonify(await _get_svc(_t()).get_portfolio_stats()), 200


@bp.get("/coverage/<farmer_id>")
async def farmer_coverage(farmer_id: str):
	return jsonify(await _get_svc(_t()).get_farmer_coverage(farmer_id)), 200


@bp.get("/audit")
async def get_audit():
	events = await _get_svc(_t()).get_audit_events(int(request.args.get("limit", 100)))
	return jsonify({"events": events, "count": len(events)}), 200
