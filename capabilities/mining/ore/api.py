"""REST API Blueprint for APG Ore Processing & Metallurgy."""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, g, jsonify, request

from .models import (
	CircuitStatusUpdateCreate,
	DeviationAlertCreate,
	MetallurgicalBalanceCreate,
	PlantFeedCreate,
	ProductQualityCreate,
	ReagentUsageCreate,
)
from .service import OreService

api_bp = Blueprint("mining_ore_api", __name__, url_prefix="/api/mining-ore")


def _svc() -> OreService:
	return OreService(tenant_id=getattr(g, "tenant_id", "default"))


def _loop() -> asyncio.AbstractEventLoop:
	return asyncio.get_event_loop()


def _err(msg: str, code: int = 400) -> tuple[Any, int]:
	return jsonify({"error": msg}), code


# ── Plant Feed ─────────────────────────────────────────────────────────────────

@api_bp.get("/plant-feed")
def list_plant_feed():
	"""List plant feed records."""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_plant_feeds(
			feed_source=request.args.get("feed_source"),
			limit=int(request.args.get("limit", 100)),
			offset=int(request.args.get("offset", 0)),
		)
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/plant-feed")
def record_plant_feed():
	"""Record plant feed data."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = PlantFeedCreate(**data)
		result = _loop().run_until_complete(
			svc.record_plant_feed(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.get("/plant-feed/<string:id>")
def get_plant_feed(id: str):
	"""Get a plant feed record."""
	svc = _svc()
	result = _loop().run_until_complete(svc.get_plant_feed(id))
	if result is None:
		return _err("Not found", 404)
	return jsonify(result.model_dump())


# ── Circuit Status ─────────────────────────────────────────────────────────────

@api_bp.post("/circuits/status")
def update_circuit_status():
	"""Update process circuit status."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = CircuitStatusUpdateCreate(**data)
		result = _loop().run_until_complete(
			svc.update_circuit_status(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.get("/circuits/current")
def get_current_circuit_statuses():
	"""Get current status of all circuits."""
	svc = _svc()
	results = _loop().run_until_complete(svc.get_current_circuit_statuses())
	return jsonify({"count": len(results), "circuits": [r.model_dump() for r in results]})


# ── Reagents ───────────────────────────────────────────────────────────────────

@api_bp.get("/reagents/inventory")
def reagent_inventory():
	"""Get current reagent inventory."""
	svc = _svc()
	inventory = _loop().run_until_complete(svc.get_reagent_inventory())
	return jsonify({"inventory_kg": inventory})


@api_bp.post("/reagents/stock")
def add_reagent_stock():
	"""Add reagent stock (delivery received)."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	reagent_type = data.get("reagent_type")
	quantity_kg = data.get("quantity_kg")
	if not reagent_type or quantity_kg is None:
		return _err("reagent_type and quantity_kg required")
	try:
		result = _loop().run_until_complete(svc.add_reagent_stock(reagent_type, float(quantity_kg)))
		return jsonify(result), 201
	except AssertionError as exc:
		return _err(str(exc))


@api_bp.get("/reagents/usage")
def list_reagent_usage():
	"""List reagent usage records."""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_reagent_usage(
			reagent_type=request.args.get("reagent_type"),
			circuit_id=request.args.get("circuit_id"),
		)
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/reagents/usage")
def record_reagent_usage():
	"""Record reagent usage."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = ReagentUsageCreate(**data)
		result = _loop().run_until_complete(
			svc.record_reagent_usage(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


# ── Metallurgical Balance ──────────────────────────────────────────────────────

@api_bp.get("/met-balance")
def list_met_balances():
	"""List metallurgical balances."""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_metallurgical_balances(
			balance_type=request.args.get("balance_type"),
			commodity=request.args.get("commodity"),
			published_only=request.args.get("published_only", "false").lower() == "true",
		)
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/met-balance")
def submit_met_balance():
	"""Submit a metallurgical balance."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = MetallurgicalBalanceCreate(**data)
		result = _loop().run_until_complete(
			svc.submit_metallurgical_balance(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.get("/met-balance/<string:id>")
def get_met_balance(id: str):
	"""Get a metallurgical balance."""
	svc = _svc()
	result = _loop().run_until_complete(svc.get_metallurgical_balance(id))
	if result is None:
		return _err("Not found", 404)
	return jsonify(result.model_dump())


@api_bp.post("/met-balance/<string:id>/approve")
def approve_met_balance(id: str):
	"""Approve a metallurgical balance."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	approver_id = data.get("approver_id", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(svc.approve_metallurgical_balance(id, approver_id))
		return jsonify(result.model_dump())
	except KeyError as exc:
		return _err(str(exc), 404)


@api_bp.post("/met-balance/<string:id>/publish")
def publish_met_balance(id: str):
	"""Publish an approved metallurgical balance."""
	svc = _svc()
	try:
		result = _loop().run_until_complete(svc.publish_metallurgical_balance(id))
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as exc:
		return _err(str(exc), 403 if isinstance(exc, PermissionError) else 404)


# ── Product Quality ────────────────────────────────────────────────────────────

@api_bp.get("/product-quality")
def list_product_quality():
	"""List product quality records."""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_product_quality(
			product_type=request.args.get("product_type"),
			on_spec_only=request.args.get("on_spec_only", "false").lower() == "true",
		)
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/product-quality")
def record_product_quality():
	"""Record product quality data."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = ProductQualityCreate(**data)
		result = _loop().run_until_complete(
			svc.record_product_quality(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.post("/product-quality/<string:id>/approve-dispatch")
def approve_product_dispatch(id: str):
	"""Approve a product lot for dispatch."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	approved_by = data.get("approved_by", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(svc.approve_product_dispatch(id, approved_by))
		return jsonify(result.model_dump())
	except KeyError as exc:
		return _err(str(exc), 404)


# ── Deviation Alerts ───────────────────────────────────────────────────────────

@api_bp.get("/deviations")
def list_deviations():
	"""List process deviation alerts."""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_deviation_alerts(
			open_only=request.args.get("open_only", "true").lower() == "true",
			alert_level=request.args.get("alert_level"),
		)
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/deviations")
def raise_deviation():
	"""Raise a process deviation alert."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = DeviationAlertCreate(**data)
		result = _loop().run_until_complete(
			svc.raise_deviation_alert(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.post("/deviations/<string:id>/acknowledge")
def acknowledge_deviation(id: str):
	"""Acknowledge a deviation alert."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	acknowledged_by = data.get("acknowledged_by", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(svc.acknowledge_deviation(id, acknowledged_by))
		return jsonify(result.model_dump())
	except KeyError as exc:
		return _err(str(exc), 404)


@api_bp.post("/deviations/<string:id>/resolve")
def resolve_deviation(id: str):
	"""Resolve a deviation alert."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	resolution_notes = data.get("resolution_notes", "")
	try:
		result = _loop().run_until_complete(svc.resolve_deviation(id, resolution_notes))
		return jsonify(result.model_dump())
	except KeyError as exc:
		return _err(str(exc), 404)


# ── KPI Summary ────────────────────────────────────────────────────────────────

@api_bp.get("/kpis")
def process_kpis():
	"""Process KPI summary."""
	svc = _svc()
	kpis = _loop().run_until_complete(svc.get_process_kpi_summary())
	return jsonify(kpis)
