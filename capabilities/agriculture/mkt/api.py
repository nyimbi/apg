"""Agri-Marketplace Flask Blueprint — agr_mkt."""
from __future__ import annotations
import logging
from flask import Blueprint, jsonify, request
from .service import AgriMarketplaceService

_log = logging.getLogger(__name__)
bp = Blueprint("agr_mkt", __name__, url_prefix="/api/agriculture/mkt")
_svc: dict[str, AgriMarketplaceService] = {}


def _get_svc(t: str = "default") -> AgriMarketplaceService:
	if t not in _svc:
		_svc[t] = AgriMarketplaceService(tenant_id=t)
	return _svc[t]


def _t() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@bp.get("/health")
async def health():
	return jsonify(await _get_svc(_t()).health_check()), 200


@bp.get("/listings")
async def list_listings():
	svc = _get_svc(_t())
	items = await svc.list_listings(
		product_type=request.args.get("product_type"),
		status=request.args.get("status"),
		farmer_id=request.args.get("farmer_id"),
		location=request.args.get("location"),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.get("/listings/<listing_id>")
async def get_listing(listing_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_listing(listing_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/listings")
async def create_listing():
	try:
		return jsonify(await _get_svc(_t()).create_listing(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/listings/<listing_id>")
async def update_listing(listing_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_listing(listing_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/listings/<listing_id>")
async def delete_listing(listing_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_listing(listing_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/listings/<listing_id>/publish")
async def publish_listing(listing_id: str):
	try:
		return jsonify(await _get_svc(_t()).publish_listing(listing_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/listings/<listing_id>/matches")
async def match_buyers(listing_id: str):
	try:
		buyers = await _get_svc(_t()).match_buyers(listing_id)
		return jsonify({"buyers": buyers, "count": len(buyers)}), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/bids")
async def list_bids():
	svc = _get_svc(_t())
	items = await svc.list_bids(
		listing_id=request.args.get("listing_id"),
		buyer_id=request.args.get("buyer_id"),
		status=request.args.get("status"),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/bids")
async def place_bid():
	try:
		return jsonify(await _get_svc(_t()).place_bid(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/bids/<bid_id>/respond")
async def respond_to_bid(bid_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _get_svc(_t()).respond_to_bid(bid_id, body.get("action", ""), body.get("counter_price"))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/prices")
async def price_discovery():
	svc = _get_svc(_t())
	result = await svc.get_price_discovery(
		product_type=request.args.get("product_type", ""),
		region=request.args.get("region"),
	)
	return jsonify(result), 200


@bp.get("/escrows")
async def list_escrows():
	svc = _get_svc(_t())
	items = await svc.list_escrows(buyer_id=request.args.get("buyer_id"), farmer_id=request.args.get("farmer_id"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/escrows")
async def create_escrow():
	try:
		return jsonify(await _get_svc(_t()).create_escrow(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/escrows/<escrow_id>/release")
async def release_escrow(escrow_id: str):
	try:
		return jsonify(await _get_svc(_t()).release_escrow(escrow_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/auctions")
async def list_auctions():
	items = await _get_svc(_t()).list_auctions(status=request.args.get("status"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/auctions")
async def create_auction():
	try:
		return jsonify(await _get_svc(_t()).create_auction(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/auctions/<auction_id>/bid")
async def auction_bid(auction_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _get_svc(_t()).place_auction_bid(auction_id, body["bidder_id"], float(body["amount"]))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/auctions/<auction_id>/close")
async def close_auction(auction_id: str):
	try:
		return jsonify(await _get_svc(_t()).close_auction(auction_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/summary")
async def summary():
	return jsonify(await _get_svc(_t()).get_marketplace_summary()), 200


@bp.get("/audit")
async def get_audit():
	events = await _get_svc(_t()).get_audit_events(int(request.args.get("limit", 100)))
	return jsonify({"events": events, "count": len(events)}), 200
