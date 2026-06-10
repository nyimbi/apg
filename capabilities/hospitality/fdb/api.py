"""Flask Blueprint — REST API for F&B Management."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import FDBService

_log = logging.getLogger(__name__)

fdb_bp = Blueprint("hos_fdb", __name__, url_prefix="/api/hospitality/fdb")
_svc = FDBService()


def _run(coro: Any) -> Any:
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@fdb_bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@fdb_bp.get("/menu-items")
def list_menu_items():
	category = request.args.get("category")
	available_only = request.args.get("available_only", "false").lower() == "true"
	return jsonify(_run(_svc.list_menu_items(_tenant(), category=category, available_only=available_only)))


@fdb_bp.get("/menu-items/<item_id>")
def get_menu_item(item_id: str):
	try:
		return jsonify(_run(_svc.get_menu_item(item_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@fdb_bp.post("/menu-items")
def create_menu_item():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_menu_item(
			name=data["name"],
			category=data.get("category", "main"),
			price=float(data["price"]),
			cost=float(data.get("cost", 0)),
			description=data.get("description"),
			allergens=data.get("allergens", []),
			prep_time_mins=data.get("prep_time_mins", 15),
			is_available=data.get("is_available", True),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@fdb_bp.put("/menu-items/<item_id>")
def update_menu_item(item_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.update_menu_item(item_id, data, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@fdb_bp.delete("/menu-items/<item_id>")
def delete_menu_item(item_id: str):
	try:
		return jsonify(_run(_svc.delete_menu_item(item_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@fdb_bp.get("/tables")
def list_tables():
	status = request.args.get("status")
	section = request.args.get("section")
	return jsonify(_run(_svc.list_tables(_tenant(), status=status, section=section)))


@fdb_bp.post("/tables")
def create_table():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_table(
			table_number=data["table_number"],
			section=data.get("section", "main"),
			capacity=int(data.get("capacity", 4)),
			notes=data.get("notes"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@fdb_bp.get("/tables/<table_id>")
def get_table(table_id: str):
	try:
		return jsonify(_run(_svc.get_table(table_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@fdb_bp.post("/tables/<table_id>/seat")
def seat_guests(table_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.seat_guests(table_id, int(data.get("covers", 1)), _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@fdb_bp.get("/orders")
def list_orders():
	status = request.args.get("status")
	table_id = request.args.get("table_id")
	return jsonify(_run(_svc.list_orders(_tenant(), status=status, table_id=table_id)))


@fdb_bp.get("/orders/<order_id>")
def get_order(order_id: str):
	try:
		return jsonify(_run(_svc.get_order(order_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@fdb_bp.post("/orders")
def create_order():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_order(
			table_id=data["table_id"],
			server_id=data.get("server_id", "pos"),
			order_type=data.get("order_type", "dine_in"),
			items=data.get("items", []),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@fdb_bp.post("/orders/<order_id>/send-to-kitchen")
def send_to_kitchen(order_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.send_to_kitchen(order_id, data.get("priority", "normal"), _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@fdb_bp.post("/orders/<order_id>/settle")
def settle_order(order_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.settle_order(
			order_id,
			data.get("payment_method", "cash"),
			float(data.get("amount_paid", 0)),
			float(data.get("discount", 0)),
			_tenant(),
		)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@fdb_bp.delete("/orders/<order_id>")
def void_order(order_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.delete_order(order_id, data.get("reason", "voided"), _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@fdb_bp.post("/kitchen-tickets/<ticket_id>/complete")
def complete_kitchen_ticket(ticket_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.complete_kitchen_ticket(ticket_id, data.get("completed_by", "chef"), _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@fdb_bp.post("/recipes")
def create_recipe():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_recipe(
			menu_item_id=data["menu_item_id"],
			ingredients=data.get("ingredients", []),
			yield_portions=int(data.get("yield_portions", 1)),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@fdb_bp.get("/inventory")
def list_inventory():
	category = request.args.get("category")
	low_stock = request.args.get("low_stock", "false").lower() == "true"
	return jsonify(_run(_svc.list_inventory(_tenant(), category=category, low_stock=low_stock)))


@fdb_bp.post("/inventory")
def create_inventory_item():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_inventory_item(
			name=data["name"],
			category=data.get("category", "food"),
			unit=data.get("unit", "kg"),
			quantity=float(data.get("quantity", 0)),
			unit_cost=float(data.get("unit_cost", 0)),
			reorder_level=float(data.get("reorder_level", 10)),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@fdb_bp.post("/inventory/<item_id>/adjust")
def adjust_inventory(item_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.adjust_inventory(item_id, float(data["quantity_delta"]), data.get("reason", "adjustment"), _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@fdb_bp.get("/menu-engineering")
def menu_engineering():
	return jsonify(_run(_svc.menu_engineering_report(_tenant())))


@fdb_bp.get("/revenue-report")
def revenue_report():
	from datetime import date
	date_str = request.args.get("date", date.today().isoformat())
	return jsonify(_run(_svc.daily_revenue_report(date_str, _tenant())))


@fdb_bp.get("/dashboard")
def dashboard():
	return jsonify(_run(_svc.dashboard_summary(_tenant())))


@fdb_bp.get("/audit-events")
def audit_events():
	return jsonify(_run(_svc.get_audit_events(_tenant())))
