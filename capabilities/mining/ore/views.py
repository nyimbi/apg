"""Flask Blueprint views for APG Ore Processing & Metallurgy."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from flask import Blueprint, abort, g, jsonify, request

from .service import OreService

views_bp = Blueprint("mining_ore_views", __name__, url_prefix="/mining-ore")


def has_access(permission: str) -> Callable:
	def decorator(fn: Callable) -> Callable:
		@wraps(fn)
		def wrapper(*args: Any, **kwargs: Any) -> Any:
			user = getattr(g, "current_user", None)
			if user is None:
				abort(401)
			perms = getattr(user, "permissions", [])
			if permission not in perms and "mining_ore:admin" not in perms:
				abort(403)
			return fn(*args, **kwargs)
		return wrapper
	return decorator


def _get_service() -> OreService:
	return OreService(tenant_id=getattr(g, "tenant_id", "default"))


# ── Dashboard ──────────────────────────────────────────────────────────────────

@views_bp.get("/dashboard")
@has_access("mining_ore:view")
def dashboard():
	"""Process KPI dashboard — recovery, throughput, open deviations."""
	import asyncio
	svc = _get_service()
	kpis = asyncio.get_event_loop().run_until_complete(svc.get_process_kpi_summary())
	return jsonify({"view": "ore_dashboard", "data": kpis})


# ── Plant Feed ─────────────────────────────────────────────────────────────────

@views_bp.get("/plant-feed")
@has_access("mining_ore:view")
def plant_feed_ledger():
	"""Plant feed records ledger."""
	import asyncio
	svc = _get_service()
	feed_source = request.args.get("feed_source")
	limit = int(request.args.get("limit", 100))
	results = asyncio.get_event_loop().run_until_complete(
		svc.list_plant_feeds(feed_source=feed_source, limit=limit)
	)
	return jsonify({"view": "plant_feed", "count": len(results), "items": [r.model_dump() for r in results]})


# ── Circuit Status ─────────────────────────────────────────────────────────────

@views_bp.get("/circuits")
@has_access("mining_ore:view")
def circuit_status_board():
	"""Real-time circuit status board."""
	import asyncio
	svc = _get_service()
	statuses = asyncio.get_event_loop().run_until_complete(svc.get_current_circuit_statuses())
	return jsonify({
		"view": "circuit_status",
		"count": len(statuses),
		"circuits": [s.model_dump() for s in statuses],
	})


# ── Reagents ───────────────────────────────────────────────────────────────────

@views_bp.get("/reagents")
@has_access("mining_ore:view")
def reagent_inventory():
	"""Reagent inventory levels."""
	import asyncio
	svc = _get_service()
	inventory = asyncio.get_event_loop().run_until_complete(svc.get_reagent_inventory())
	return jsonify({"view": "reagent_inventory", "inventory_kg": inventory})


@views_bp.get("/reagents/usage")
@has_access("mining_ore:view")
def reagent_usage():
	"""Reagent usage history."""
	import asyncio
	svc = _get_service()
	reagent_type = request.args.get("reagent_type")
	results = asyncio.get_event_loop().run_until_complete(svc.list_reagent_usage(reagent_type=reagent_type))
	return jsonify({"view": "reagent_usage", "count": len(results), "items": [r.model_dump() for r in results]})


# ── Metallurgical Balance ──────────────────────────────────────────────────────

@views_bp.get("/met-balance")
@has_access("mining_ore:met_balance")
def met_balance_list():
	"""Metallurgical balance list."""
	import asyncio
	svc = _get_service()
	balance_type = request.args.get("balance_type")
	commodity = request.args.get("commodity")
	published_only = request.args.get("published_only", "false").lower() == "true"
	results = asyncio.get_event_loop().run_until_complete(
		svc.list_metallurgical_balances(balance_type=balance_type, commodity=commodity, published_only=published_only)
	)
	return jsonify({"view": "met_balance", "count": len(results), "items": [r.model_dump() for r in results]})


@views_bp.get("/met-balance/<string:id>")
@has_access("mining_ore:met_balance")
def met_balance_detail(id: str):
	"""Metallurgical balance detail view."""
	import asyncio
	svc = _get_service()
	balance = asyncio.get_event_loop().run_until_complete(svc.get_metallurgical_balance(id))
	if balance is None:
		abort(404)
	return jsonify({"view": "met_balance_detail", "balance": balance.model_dump()})


# ── Product Quality ────────────────────────────────────────────────────────────

@views_bp.get("/product-quality")
@has_access("mining_ore:view")
def product_quality_ledger():
	"""Product quality records."""
	import asyncio
	svc = _get_service()
	product_type = request.args.get("product_type")
	on_spec_only = request.args.get("on_spec_only", "false").lower() == "true"
	results = asyncio.get_event_loop().run_until_complete(
		svc.list_product_quality(product_type=product_type, on_spec_only=on_spec_only)
	)
	return jsonify({"view": "product_quality", "count": len(results), "items": [r.model_dump() for r in results]})


# ── Reconciliation ─────────────────────────────────────────────────────────────

@views_bp.get("/reconciliation")
@has_access("mining_ore:reconciliation")
def reconciliation_console():
	"""Ore reconciliation console."""
	import asyncio
	svc = _get_service()
	kpis = asyncio.get_event_loop().run_until_complete(svc.get_process_kpi_summary())
	return jsonify({"view": "reconciliation", "process_kpis": kpis})


# ── Deviation Alerts ───────────────────────────────────────────────────────────

@views_bp.get("/deviations")
@has_access("mining_ore:view")
def deviation_alerts():
	"""Open process deviation alerts."""
	import asyncio
	svc = _get_service()
	open_only = request.args.get("open_only", "true").lower() == "true"
	alert_level = request.args.get("alert_level")
	results = asyncio.get_event_loop().run_until_complete(
		svc.list_deviation_alerts(open_only=open_only, alert_level=alert_level)
	)
	return jsonify({"view": "deviation_alerts", "count": len(results), "items": [r.model_dump() for r in results]})
