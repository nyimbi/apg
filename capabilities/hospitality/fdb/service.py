"""F&B Management service — restaurant POS, table management, menu engineering, kitchen display, recipe costing, inventory."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

TAX_RATE = 0.16  # Kenya VAT 16%


def _uid() -> str:
	return uuid4().hex[:12]


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class FDBService:
	"""F&B Management service."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.menu_items: dict[str, dict[str, Any]] = {}
		self.tables: dict[str, dict[str, Any]] = {}
		self.orders: dict[str, dict[str, Any]] = {}
		self.order_items: dict[str, dict[str, Any]] = {}
		self.kitchen_tickets: dict[str, dict[str, Any]] = {}
		self.recipes: dict[str, dict[str, Any]] = {}
		self.inventory: dict[str, dict[str, Any]] = {}
		self.inventory_transactions: dict[str, dict[str, Any]] = {}
		self.reservations: dict[str, dict[str, Any]] = {}
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _emit(self, tenant_id: str, event_type: str, record_id: str, record_type: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": _uid(),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"details": details or {},
			"created_at": _now(),
		})

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "hos_fdb",
			"status": "healthy",
			"menu_items": sum(1 for m in self.menu_items.values() if m["is_available"]),
			"occupied_tables": sum(1 for t in self.tables.values() if t["status"] == "occupied"),
			"open_orders": sum(1 for o in self.orders.values() if o["status"] in {"open", "sent_to_kitchen"}),
			"checked_at": _now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": "hos_fdb",
			"name": "F&B Management",
			"domain": "hospitality",
			"version": "1.0.0",
			"description": "Restaurant POS, table management, menu engineering, kitchen display, recipe costing, inventory",
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Menu Items ────────────────────────────────────────────────────────────

	async def list_menu_items(self, tenant_id: str | None = None, category: str | None = None, available_only: bool = False) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(m) for m in self.menu_items.values() if m["tenant_id"] == tenant]
		if category:
			items = [m for m in items if m["category"] == category]
		if available_only:
			items = [m for m in items if m["is_available"]]
		return items

	async def get_menu_item(self, item_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		item = self.menu_items.get(item_id)
		if not item or item["tenant_id"] != tenant:
			raise KeyError(f"menu_item_not_found:{item_id}")
		return deepcopy(item)

	async def create_menu_item(self, name: str, category: str, price: float, cost: float = 0.0,
	                            description: str | None = None, allergens: list[str] | None = None,
	                            prep_time_mins: int = 15, is_available: bool = True,
	                            tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		if not name:
			raise ValueError("item_name_required")
		margin = ((price - cost) / price * 100) if price > 0 else 0.0
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"name": name,
			"category": category,
			"description": description,
			"price": price,
			"cost": cost,
			"gross_margin_pct": round(margin, 2),
			"allergens": allergens or [],
			"is_available": is_available,
			"prep_time_mins": prep_time_mins,
			"order_count": 0,
			"status": "active",
			"created_at": _now(),
		}
		self.menu_items[record["id"]] = record
		self._emit(tenant, "menu_item_created", record["id"], "menu_item", {"category": category})
		return deepcopy(record)

	async def update_menu_item(self, item_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		item = self.menu_items.get(item_id)
		if not item or item["tenant_id"] != tenant:
			raise KeyError(f"menu_item_not_found:{item_id}")
		allowed = {"name", "price", "cost", "is_available", "description", "prep_time_mins", "allergens"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				item[k] = v
		# Recalculate margin
		if item["price"] > 0:
			item["gross_margin_pct"] = round((item["price"] - item["cost"]) / item["price"] * 100, 2)
		item["updated_at"] = _now()
		self._emit(tenant, "menu_item_updated", item_id, "menu_item")
		return deepcopy(item)

	async def delete_menu_item(self, item_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		item = self.menu_items.get(item_id)
		if not item or item["tenant_id"] != tenant:
			raise KeyError(f"menu_item_not_found:{item_id}")
		item["is_available"] = False
		item["status"] = "deactivated"
		self._emit(tenant, "menu_item_deactivated", item_id, "menu_item")
		return {"deactivated": True, "item_id": item_id}

	async def set_item_availability(self, item_id: str, is_available: bool, tenant_id: str | None = None) -> dict[str, Any]:
		return await self.update_menu_item(item_id, {"is_available": is_available}, tenant_id)

	# ── Tables ────────────────────────────────────────────────────────────────

	async def list_tables(self, tenant_id: str | None = None, status: str | None = None, section: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		tables = [deepcopy(t) for t in self.tables.values() if t["tenant_id"] == tenant]
		if status:
			tables = [t for t in tables if t["status"] == status]
		if section:
			tables = [t for t in tables if t["section"] == section]
		return tables

	async def get_table(self, table_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		table = self.tables.get(table_id)
		if not table or table["tenant_id"] != tenant:
			raise KeyError(f"table_not_found:{table_id}")
		return deepcopy(table)

	async def create_table(self, table_number: str, section: str, capacity: int, notes: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		for t in self.tables.values():
			if t["tenant_id"] == tenant and t["table_number"] == table_number:
				raise ValueError(f"table_number_exists:{table_number}")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"table_number": table_number,
			"section": section,
			"capacity": capacity,
			"notes": notes,
			"status": "available",  # available|occupied|reserved|cleaning
			"current_order_id": None,
			"seated_at": None,
			"created_at": _now(),
		}
		self.tables[record["id"]] = record
		self._emit(tenant, "table_created", record["id"], "table")
		return deepcopy(record)

	async def update_table(self, table_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		table = self.tables.get(table_id)
		if not table or table["tenant_id"] != tenant:
			raise KeyError(f"table_not_found:{table_id}")
		for k, v in updates.items():
			if v is not None:
				table[k] = v
		return deepcopy(table)

	async def delete_table(self, table_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		table = self.tables.get(table_id)
		if not table or table["tenant_id"] != tenant:
			raise KeyError(f"table_not_found:{table_id}")
		if table["status"] == "occupied":
			raise ValueError("cannot_delete_occupied_table")
		del self.tables[table_id]
		return {"deleted": True, "table_id": table_id}

	async def seat_guests(self, table_id: str, covers: int, tenant_id: str | None = None) -> dict[str, Any]:
		"""Seat guests at a table."""
		tenant = self._tenant(tenant_id)
		table = self.tables.get(table_id)
		if not table or table["tenant_id"] != tenant:
			raise KeyError(f"table_not_found:{table_id}")
		if table["status"] not in {"available"}:
			raise ValueError(f"table_not_available:{table['status']}")
		if covers > table["capacity"]:
			raise ValueError(f"covers_exceeds_capacity:{covers}>{table['capacity']}")
		table["status"] = "occupied"
		table["covers"] = covers
		table["seated_at"] = _now()
		self._emit(tenant, "table_seated", table_id, "table", {"covers": covers})
		return deepcopy(table)

	# ── Orders ────────────────────────────────────────────────────────────────

	async def list_orders(self, tenant_id: str | None = None, status: str | None = None, table_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		orders = [deepcopy(o) for o in self.orders.values() if o["tenant_id"] == tenant]
		if status:
			orders = [o for o in orders if o["status"] == status]
		if table_id:
			orders = [o for o in orders if o["table_id"] == table_id]
		return orders

	async def get_order(self, order_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order_not_found:{order_id}")
		return deepcopy(order)

	async def create_order(self, table_id: str, server_id: str, order_type: str = "dine_in",
	                        items: list[dict[str, Any]] | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		table = self.tables.get(table_id)
		if not table or table["tenant_id"] != tenant:
			raise KeyError(f"table_not_found:{table_id}")
		# Resolve items and compute subtotal
		resolved_items = []
		subtotal = 0.0
		for item_ref in (items or []):
			item_id = item_ref.get("item_id", "")
			qty = item_ref.get("quantity", 1)
			menu_item = self.menu_items.get(item_id)
			if not menu_item or menu_item["tenant_id"] != tenant:
				raise KeyError(f"menu_item_not_found:{item_id}")
			line_total = menu_item["price"] * qty
			subtotal += line_total
			resolved_items.append({
				"item_id": item_id,
				"name": menu_item["name"],
				"quantity": qty,
				"unit_price": menu_item["price"],
				"line_total": line_total,
				"notes": item_ref.get("notes"),
				"kitchen_status": "pending",
			})
		tax = round(subtotal * TAX_RATE, 2)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"table_id": table_id,
			"table_number": table.get("table_number", ""),
			"server_id": server_id,
			"order_type": order_type,
			"items": resolved_items,
			"subtotal": round(subtotal, 2),
			"tax": tax,
			"total": round(subtotal + tax, 2),
			"discount": 0.0,
			"covers": table.get("covers", 1),
			"status": "open",
			"kitchen_status": "pending",
			"payment_status": "unpaid",
			"created_at": _now(),
		}
		self.orders[record["id"]] = record
		table["current_order_id"] = record["id"]
		# Update menu item order counts
		for ri in resolved_items:
			mi = self.menu_items.get(ri["item_id"])
			if mi:
				mi["order_count"] = mi.get("order_count", 0) + ri["quantity"]
		self._emit(tenant, "order_created", record["id"], "order", {"table_id": table_id, "total": record["total"]})
		return deepcopy(record)

	async def update_order(self, order_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order_not_found:{order_id}")
		if order["status"] in {"settled", "cancelled"}:
			raise ValueError("cannot_modify_closed_order")
		for k, v in updates.items():
			if v is not None:
				order[k] = v
		self._emit(tenant, "order_updated", order_id, "order")
		return deepcopy(order)

	async def delete_order(self, order_id: str, reason: str = "voided", tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order_not_found:{order_id}")
		if order["status"] == "settled":
			raise ValueError("cannot_void_settled_order")
		order["status"] = "cancelled"
		order["cancellation_reason"] = reason
		self._emit(tenant, "order_voided", order_id, "order", {"reason": reason})
		return deepcopy(order)

	async def send_to_kitchen(self, order_id: str, priority: str = "normal", tenant_id: str | None = None) -> dict[str, Any]:
		"""Send an order to the kitchen display system."""
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order_not_found:{order_id}")
		ticket: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"order_id": order_id,
			"table_number": order["table_number"],
			"covers": order["covers"],
			"items": deepcopy([i for i in order["items"] if i["kitchen_status"] == "pending"]),
			"priority": priority,
			"status": "pending",
			"sent_at": _now(),
			"completed_at": None,
		}
		self.kitchen_tickets[ticket["id"]] = ticket
		order["kitchen_status"] = "in_kitchen"
		order["status"] = "sent_to_kitchen"
		self._emit(tenant, "order_sent_to_kitchen", ticket["id"], "kitchen_ticket")
		return deepcopy(ticket)

	async def complete_kitchen_ticket(self, ticket_id: str, completed_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Mark a kitchen ticket as ready for service."""
		tenant = self._tenant(tenant_id)
		ticket = self.kitchen_tickets.get(ticket_id)
		if not ticket or ticket["tenant_id"] != tenant:
			raise KeyError(f"ticket_not_found:{ticket_id}")
		ticket["status"] = "completed"
		ticket["completed_at"] = _now()
		ticket["completed_by"] = completed_by
		order = self.orders.get(ticket["order_id"])
		if order:
			order["kitchen_status"] = "ready"
			for item in order["items"]:
				item["kitchen_status"] = "ready"
		self._emit(tenant, "kitchen_ticket_completed", ticket_id, "kitchen_ticket")
		return deepcopy(ticket)

	async def settle_order(self, order_id: str, payment_method: str, amount_paid: float,
	                        discount: float = 0.0, tenant_id: str | None = None) -> dict[str, Any]:
		"""Settle an order (bill payment)."""
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order_not_found:{order_id}")
		if order["status"] == "settled":
			raise ValueError("order_already_settled")
		if discount > 0:
			order["discount"] = discount
			order["total"] = round(order["total"] - discount, 2)
		order["status"] = "settled"
		order["payment_method"] = payment_method
		order["amount_paid"] = amount_paid
		order["change"] = round(amount_paid - order["total"], 2)
		order["payment_status"] = "paid"
		order["settled_at"] = _now()
		# Free table
		table = self.tables.get(order["table_id"])
		if table:
			table["status"] = "cleaning"
			table["current_order_id"] = None
		self._emit(tenant, "order_settled", order_id, "order", {"payment_method": payment_method, "total": order["total"]})
		return deepcopy(order)

	# ── Recipes & Costing ─────────────────────────────────────────────────────

	async def create_recipe(self, menu_item_id: str, ingredients: list[dict[str, Any]], yield_portions: int = 1,
	                         tenant_id: str | None = None) -> dict[str, Any]:
		"""Create a recipe with ingredient costs for a menu item."""
		tenant = self._tenant(tenant_id)
		item = self.menu_items.get(menu_item_id)
		if not item or item["tenant_id"] != tenant:
			raise KeyError(f"menu_item_not_found:{menu_item_id}")
		total_cost = sum(i.get("cost", 0.0) * i.get("quantity", 1) for i in ingredients)
		cost_per_portion = total_cost / yield_portions if yield_portions else 0.0
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"menu_item_id": menu_item_id,
			"menu_item_name": item["name"],
			"ingredients": deepcopy(ingredients),
			"yield_portions": yield_portions,
			"total_cost": round(total_cost, 2),
			"cost_per_portion": round(cost_per_portion, 2),
			"status": "active",
			"created_at": _now(),
		}
		self.recipes[record["id"]] = record
		# Update menu item cost
		item["cost"] = cost_per_portion
		if item["price"] > 0:
			item["gross_margin_pct"] = round((item["price"] - cost_per_portion) / item["price"] * 100, 2)
		self._emit(tenant, "recipe_created", record["id"], "recipe", {"menu_item_id": menu_item_id})
		return deepcopy(record)

	async def list_recipes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.recipes.values() if r["tenant_id"] == tenant]

	# ── Inventory ─────────────────────────────────────────────────────────────

	async def list_inventory(self, tenant_id: str | None = None, category: str | None = None, low_stock: bool = False) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(i) for i in self.inventory.values() if i["tenant_id"] == tenant]
		if category:
			items = [i for i in items if i["category"] == category]
		if low_stock:
			items = [i for i in items if i["quantity"] <= i["reorder_level"]]
		return items

	async def create_inventory_item(self, name: str, category: str, unit: str, quantity: float,
	                                 unit_cost: float, reorder_level: float = 10.0,
	                                 tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"name": name,
			"category": category,
			"unit": unit,
			"quantity": quantity,
			"unit_cost": unit_cost,
			"reorder_level": reorder_level,
			"total_value": round(quantity * unit_cost, 2),
			"status": "active",
			"created_at": _now(),
		}
		self.inventory[record["id"]] = record
		self._emit(tenant, "inventory_item_created", record["id"], "inventory")
		return deepcopy(record)

	async def adjust_inventory(self, item_id: str, quantity_delta: float, reason: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Adjust inventory stock level."""
		tenant = self._tenant(tenant_id)
		inv = self.inventory.get(item_id)
		if not inv or inv["tenant_id"] != tenant:
			raise KeyError(f"inventory_item_not_found:{item_id}")
		old_qty = inv["quantity"]
		inv["quantity"] = max(0.0, inv["quantity"] + quantity_delta)
		inv["total_value"] = round(inv["quantity"] * inv["unit_cost"], 2)
		txn: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"item_id": item_id,
			"delta": quantity_delta,
			"old_quantity": old_qty,
			"new_quantity": inv["quantity"],
			"reason": reason,
			"created_at": _now(),
		}
		self.inventory_transactions[txn["id"]] = txn
		self._emit(tenant, "inventory_adjusted", item_id, "inventory", {"delta": quantity_delta, "reason": reason})
		return deepcopy(inv)

	# ── Menu Engineering ──────────────────────────────────────────────────────

	async def menu_engineering_report(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""BCG-style menu engineering: Stars/Plowhorses/Puzzles/Dogs."""
		tenant = self._tenant(tenant_id)
		items = [m for m in self.menu_items.values() if m["tenant_id"] == tenant and m["is_available"]]
		if not items:
			return {"tenant_id": tenant, "items": [], "generated_at": _now()}
		avg_orders = sum(m["order_count"] for m in items) / len(items)
		avg_margin = sum(m["gross_margin_pct"] for m in items) / len(items)
		classified = []
		for m in items:
			high_popularity = m["order_count"] >= avg_orders
			high_margin = m["gross_margin_pct"] >= avg_margin
			if high_popularity and high_margin:
				classification = "star"
			elif high_popularity and not high_margin:
				classification = "plowhorse"
			elif not high_popularity and high_margin:
				classification = "puzzle"
			else:
				classification = "dog"
			classified.append({
				"item_id": m["id"],
				"name": m["name"],
				"category": m["category"],
				"order_count": m["order_count"],
				"gross_margin_pct": m["gross_margin_pct"],
				"classification": classification,
			})
		return {
			"tenant_id": tenant,
			"total_items": len(items),
			"items": classified,
			"stars": [c for c in classified if c["classification"] == "star"],
			"plowhorses": [c for c in classified if c["classification"] == "plowhorse"],
			"puzzles": [c for c in classified if c["classification"] == "puzzle"],
			"dogs": [c for c in classified if c["classification"] == "dog"],
			"generated_at": _now(),
		}

	async def daily_revenue_report(self, date: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		orders = [o for o in self.orders.values() if o["tenant_id"] == tenant and o["status"] == "settled" and o["settled_at"][:10] == date]
		by_type: dict[str, float] = {}
		for o in orders:
			by_type[o["order_type"]] = by_type.get(o["order_type"], 0.0) + o["total"]
		return {
			"tenant_id": tenant,
			"date": date,
			"total_orders": len(orders),
			"total_revenue": round(sum(o["total"] for o in orders), 2),
			"total_tax": round(sum(o["tax"] for o in orders), 2),
			"total_discounts": round(sum(o.get("discount", 0.0) for o in orders), 2),
			"revenue_by_type": by_type,
			"avg_cover": round(sum(o["total"] for o in orders) / max(sum(o.get("covers", 1) for o in orders), 1), 2),
			"generated_at": _now(),
		}

	async def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return {
			"tenant_id": tenant,
			"active_menu_items": sum(1 for m in self.menu_items.values() if m["tenant_id"] == tenant and m["is_available"]),
			"tables_total": sum(1 for t in self.tables.values() if t["tenant_id"] == tenant),
			"tables_occupied": sum(1 for t in self.tables.values() if t["tenant_id"] == tenant and t["status"] == "occupied"),
			"open_orders": sum(1 for o in self.orders.values() if o["tenant_id"] == tenant and o["status"] in {"open", "sent_to_kitchen"}),
			"kitchen_pending": sum(1 for k in self.kitchen_tickets.values() if k["tenant_id"] == tenant and k["status"] == "pending"),
			"low_stock_items": sum(1 for i in self.inventory.values() if i["tenant_id"] == tenant and i["quantity"] <= i["reorder_level"]),
			"generated_at": _now(),
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

