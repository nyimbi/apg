"""Order Management & Tracking async service (scm_omt)."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

_log = logging.getLogger(__name__)

CAPABILITY_ID = "scm_omt"
ORDER_STATUSES = {
	"draft", "confirmed", "allocated", "picking", "packed",
	"shipped", "delivered", "cancelled", "on_hold",
}
NOTIFICATION_CHANNELS = {"email", "sms", "push", "webhook"}
ORDER_PRIORITIES = {"urgent", "high", "normal", "low"}


class OrderManagementService:
	"""Async service for order lifecycle, ATP, backorder management,
	split shipments, order promising, and customer notifications."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.orders: dict[str, dict[str, Any]] = {}
		self.backorders: dict[str, dict[str, Any]] = {}
		self.split_shipments: dict[str, dict[str, Any]] = {}
		self.order_promises: dict[str, dict[str, Any]] = {}
		self.notifications: dict[str, dict[str, Any]] = {}
		self.atp_records: dict[str, dict[str, Any]] = {}  # available-to-promise
		self.holds: dict[str, dict[str, Any]] = {}
		self._order_seq: int = 1000
		self._audit_events: list[dict[str, Any]] = []

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _tenant(self, tenant_id: str | None = None) -> str:
		t = tenant_id or self.tenant_id
		if not t:
			raise PermissionError("tenant_context_required")
		return t

	def _next_order_number(self, tenant_id: str) -> str:
		self._order_seq += 1
		return f"ORD-{tenant_id[:4].upper()}-{self._order_seq:06d}"

	def _emit(self, tenant_id: str, event_type: str, record_id: str, record_type: str, status: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"status": status,
			"capability_id": CAPABILITY_ID,
			"emitted_at": self._now(),
		})

	# ── Health & describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"order_count": len(self.orders),
			"open_backorders": sum(1 for b in self.backorders.values() if b["status"] == "open"),
			"pending_notifications": sum(1 for n in self.notifications.values() if n["status"] == "pending"),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "scm",
			"version": "1.0.0",
			"description": "Order lifecycle, ATP, backorder management, split shipments, order promising, customer notifications",
			"supported_statuses": sorted(ORDER_STATUSES),
			"notification_channels": sorted(NOTIFICATION_CHANNELS),
			"order_priorities": sorted(ORDER_PRIORITIES),
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Order CRUD ────────────────────────────────────────────────────────────

	async def create_order(
		self,
		customer_id: str,
		lines: list[dict[str, Any]],
		shipping_address: dict[str, Any] | None = None,
		billing_address: dict[str, Any] | None = None,
		customer_reference: str | None = None,
		requested_delivery_date: str | None = None,
		priority: str = "normal",
		notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a new customer order."""
		tenant = self._tenant(tenant_id)
		if priority not in ORDER_PRIORITIES:
			raise ValueError(f"priority must be one of {ORDER_PRIORITIES}")
		if not lines:
			raise ValueError("order must have at least one line")
		enriched_lines = []
		total_value = 0.0
		for line in lines:
			line_total = float(line.get("quantity", 0)) * float(line.get("unit_price", 0))
			enriched_lines.append({
				**line,
				"line_total": round(line_total, 4),
				"status": "draft",
			})
			total_value += line_total
		order_number = self._next_order_number(tenant)
		record: dict[str, Any] = {
			"id": self._id("ord"),
			"type": "scm_omt_order",
			"tenant_id": tenant,
			"order_number": order_number,
			"customer_id": customer_id,
			"customer_reference": customer_reference,
			"lines": enriched_lines,
			"total_value": round(total_value, 4),
			"currency": lines[0].get("currency", "USD") if lines else "USD",
			"requested_delivery_date": requested_delivery_date,
			"promised_delivery_date": None,
			"shipping_address": deepcopy(shipping_address or {}),
			"billing_address": deepcopy(billing_address or {}),
			"priority": priority,
			"notes": notes,
			"status": "draft",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.orders[record["id"]] = record
		self._emit(tenant, "order_created", record["id"], "scm_omt_order", "draft")
		return deepcopy(record)

	async def list_orders(
		self,
		tenant_id: str | None = None,
		status: str | None = None,
		customer_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List orders with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(o) for o in self.orders.values() if o["tenant_id"] == tenant]
		if status:
			items = [o for o in items if o["status"] == status]
		if customer_id:
			items = [o for o in items if o["customer_id"] == customer_id]
		return items

	async def get_order(self, order_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Fetch a single order."""
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order '{order_id}' not found")
		return deepcopy(order)

	async def update_order(
		self,
		order_id: str,
		updates: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Update order fields."""
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order '{order_id}' not found")
		allowed = {"status", "requested_delivery_date", "shipping_address", "priority", "notes"}
		for k, v in updates.items():
			if k in allowed:
				order[k] = v
		order["updated_at"] = self._now()
		self._emit(tenant, "order_updated", order_id, "scm_omt_order", order["status"])
		return deepcopy(order)

	async def delete_order(self, order_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Cancel and soft-delete a draft order."""
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order '{order_id}' not found")
		if order["status"] not in {"draft", "on_hold"}:
			raise ValueError(f"cannot delete an order in status '{order['status']}'")
		order["status"] = "cancelled"
		order["updated_at"] = self._now()
		self._emit(tenant, "order_deleted", order_id, "scm_omt_order", "cancelled")
		return deepcopy(order)

	async def confirm_order(self, order_id: str, confirmed_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Confirm a draft order."""
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order '{order_id}' not found")
		if order["status"] != "draft":
			raise ValueError("only draft orders can be confirmed")
		order["status"] = "confirmed"
		order["confirmed_by"] = confirmed_by
		order["confirmed_at"] = self._now()
		order["updated_at"] = self._now()
		self._emit(tenant, "order_confirmed", order_id, "scm_omt_order", "confirmed")
		return deepcopy(order)

	async def cancel_order(self, order_id: str, reason: str, cancelled_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Cancel an order (any non-shipped status)."""
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order '{order_id}' not found")
		if order["status"] in {"shipped", "delivered", "cancelled"}:
			raise ValueError(f"cannot cancel a {order['status']} order")
		order["status"] = "cancelled"
		order["cancellation_reason"] = reason
		order["cancelled_by"] = cancelled_by
		order["cancelled_at"] = self._now()
		order["updated_at"] = self._now()
		self._emit(tenant, "order_cancelled", order_id, "scm_omt_order", "cancelled")
		return deepcopy(order)

	async def place_order_on_hold(self, order_id: str, reason: str, held_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Place an order on hold pending investigation."""
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order '{order_id}' not found")
		hold_record: dict[str, Any] = {
			"id": self._id("hold"),
			"type": "scm_omt_hold",
			"tenant_id": tenant,
			"order_id": order_id,
			"reason": reason,
			"held_by": held_by,
			"status": "active",
			"created_at": self._now(),
		}
		self.holds[hold_record["id"]] = hold_record
		order["status"] = "on_hold"
		order["hold_reason"] = reason
		order["updated_at"] = self._now()
		self._emit(tenant, "order_placed_on_hold", order_id, "scm_omt_order", "on_hold")
		return deepcopy(order)

	async def release_order_hold(self, order_id: str, released_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Release an order from hold."""
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order '{order_id}' not found")
		if order["status"] != "on_hold":
			raise ValueError("order is not on hold")
		order["status"] = "confirmed"
		order["released_by"] = released_by
		order["released_at"] = self._now()
		order["updated_at"] = self._now()
		self._emit(tenant, "order_hold_released", order_id, "scm_omt_order", "confirmed")
		return deepcopy(order)

	# ── ATP (Available-to-Promise) ────────────────────────────────────────────

	async def check_atp(
		self,
		sku: str,
		requested_quantity: float,
		requested_date: str | None = None,
		warehouse_id: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Check available-to-promise quantity for a SKU."""
		tenant = self._tenant(tenant_id)
		key = f"{tenant}:{sku}:{warehouse_id or 'any'}"
		atp_entry = self.atp_records.get(key, {})
		available = float(atp_entry.get("available_quantity", 0))
		can_fulfil = available >= requested_quantity
		shortage = max(0.0, requested_quantity - available) if not can_fulfil else 0.0
		result: dict[str, Any] = {
			"tenant_id": tenant,
			"sku": sku,
			"warehouse_id": warehouse_id,
			"requested_quantity": requested_quantity,
			"available_quantity": available,
			"can_fulfil": can_fulfil,
			"shortage_quantity": shortage,
			"checked_at": self._now(),
		}
		if requested_date:
			result["requested_date"] = requested_date
		return result

	async def update_atp(
		self,
		sku: str,
		available_quantity: float,
		warehouse_id: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Update ATP record for a SKU."""
		tenant = self._tenant(tenant_id)
		key = f"{tenant}:{sku}:{warehouse_id or 'any'}"
		record: dict[str, Any] = {
			"id": key,
			"tenant_id": tenant,
			"sku": sku,
			"warehouse_id": warehouse_id,
			"available_quantity": available_quantity,
			"updated_at": self._now(),
		}
		self.atp_records[key] = record
		self._emit(tenant, "atp_updated", key, "scm_omt_atp", "updated")
		return deepcopy(record)

	# ── Backorder management ──────────────────────────────────────────────────

	async def create_backorder(
		self,
		order_id: str,
		sku: str,
		backordered_quantity: float,
		reason: str,
		expected_fulfilment_date: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Register a backorder for an unfulfilled order line."""
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order '{order_id}' not found")
		record: dict[str, Any] = {
			"id": self._id("bo"),
			"type": "scm_omt_backorder",
			"tenant_id": tenant,
			"order_id": order_id,
			"sku": sku,
			"backordered_quantity": backordered_quantity,
			"reason": reason,
			"expected_fulfilment_date": expected_fulfilment_date,
			"status": "open",
			"created_at": self._now(),
		}
		self.backorders[record["id"]] = record
		self._emit(tenant, "backorder_created", record["id"], "scm_omt_backorder", "open")
		return deepcopy(record)

	async def fulfil_backorder(
		self,
		backorder_id: str,
		fulfilled_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Mark a backorder as fulfilled."""
		tenant = self._tenant(tenant_id)
		bo = self.backorders.get(backorder_id)
		if not bo or bo["tenant_id"] != tenant:
			raise KeyError(f"backorder '{backorder_id}' not found")
		bo["status"] = "fulfilled"
		bo["fulfilled_by"] = fulfilled_by
		bo["fulfilled_at"] = self._now()
		self._emit(tenant, "backorder_fulfilled", backorder_id, "scm_omt_backorder", "fulfilled")
		return deepcopy(bo)

	async def list_backorders(
		self,
		tenant_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List backorders."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(b) for b in self.backorders.values() if b["tenant_id"] == tenant]
		if status:
			items = [b for b in items if b["status"] == status]
		return items

	# ── Split shipments ───────────────────────────────────────────────────────

	async def create_split_shipment(
		self,
		order_id: str,
		split_lines: list[dict[str, Any]],
		reason: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Split an order into multiple partial shipments."""
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order '{order_id}' not found")
		if not split_lines:
			raise ValueError("split_lines must not be empty")
		record: dict[str, Any] = {
			"id": self._id("split"),
			"type": "scm_omt_split_shipment",
			"tenant_id": tenant,
			"order_id": order_id,
			"split_lines": deepcopy(split_lines),
			"reason": reason,
			"status": "pending",
			"created_at": self._now(),
		}
		self.split_shipments[record["id"]] = record
		self._emit(tenant, "split_shipment_created", record["id"], "scm_omt_split_shipment", "pending")
		return deepcopy(record)

	async def list_split_shipments(
		self,
		order_id: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List split shipments."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(s) for s in self.split_shipments.values() if s["tenant_id"] == tenant]
		if order_id:
			items = [s for s in items if s["order_id"] == order_id]
		return items

	# ── Order promising ───────────────────────────────────────────────────────

	async def promise_order(
		self,
		order_id: str,
		promised_date: str,
		promised_by: str,
		confidence_pct: float = 95.0,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Commit an order delivery promise."""
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order '{order_id}' not found")
		record: dict[str, Any] = {
			"id": self._id("promise"),
			"type": "scm_omt_order_promise",
			"tenant_id": tenant,
			"order_id": order_id,
			"promised_date": promised_date,
			"promised_by": promised_by,
			"confidence_pct": confidence_pct,
			"status": "active",
			"created_at": self._now(),
		}
		self.order_promises[record["id"]] = record
		order["promised_delivery_date"] = promised_date
		order["updated_at"] = self._now()
		self._emit(tenant, "order_promised", record["id"], "scm_omt_order_promise", "active")
		return deepcopy(record)

	async def revoke_order_promise(
		self,
		promise_id: str,
		reason: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Revoke a delivery promise."""
		tenant = self._tenant(tenant_id)
		promise = self.order_promises.get(promise_id)
		if not promise or promise["tenant_id"] != tenant:
			raise KeyError(f"promise '{promise_id}' not found")
		promise["status"] = "revoked"
		promise["revocation_reason"] = reason
		promise["revoked_at"] = self._now()
		self._emit(tenant, "order_promise_revoked", promise_id, "scm_omt_order_promise", "revoked")
		return deepcopy(promise)

	# ── Customer notifications ────────────────────────────────────────────────

	async def send_notification(
		self,
		order_id: str,
		channel: str,
		event_type: str,
		message: str,
		recipient: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Send a customer notification for an order event."""
		tenant = self._tenant(tenant_id)
		if channel not in NOTIFICATION_CHANNELS:
			raise ValueError(f"channel must be one of {NOTIFICATION_CHANNELS}")
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order '{order_id}' not found")
		record: dict[str, Any] = {
			"id": self._id("notif"),
			"type": "scm_omt_notification",
			"tenant_id": tenant,
			"order_id": order_id,
			"channel": channel,
			"event_type": event_type,
			"message": message,
			"recipient": recipient,
			"status": "sent",
			"sent_at": self._now(),
			"created_at": self._now(),
		}
		self.notifications[record["id"]] = record
		self._emit(tenant, "notification_sent", record["id"], "scm_omt_notification", "sent")
		return deepcopy(record)

	async def list_notifications(
		self,
		order_id: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List notifications, optionally filtered by order."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(n) for n in self.notifications.values() if n["tenant_id"] == tenant]
		if order_id:
			items = [n for n in items if n["order_id"] == order_id]
		return items

	# ── Analytics ─────────────────────────────────────────────────────────────

	async def order_analytics(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return aggregate order metrics."""
		tenant = self._tenant(tenant_id)
		all_orders = [o for o in self.orders.values() if o["tenant_id"] == tenant]
		by_status: dict[str, int] = {}
		total_value = 0.0
		for o in all_orders:
			by_status[o["status"]] = by_status.get(o["status"], 0) + 1
			total_value += o.get("total_value", 0)
		return {
			"tenant_id": tenant,
			"total_orders": len(all_orders),
			"by_status": by_status,
			"total_order_value": round(total_value, 2),
			"open_backorders": sum(1 for b in self.backorders.values() if b["tenant_id"] == tenant and b["status"] == "open"),
			"active_promises": sum(1 for p in self.order_promises.values() if p["tenant_id"] == tenant and p["status"] == "active"),
			"notifications_sent": sum(1 for n in self.notifications.values() if n["tenant_id"] == tenant),
			"generated_at": self._now(),
		}

	async def fulfilment_rate(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Calculate order fulfilment rate."""
		tenant = self._tenant(tenant_id)
		all_orders = [o for o in self.orders.values() if o["tenant_id"] == tenant]
		delivered = sum(1 for o in all_orders if o["status"] == "delivered")
		total = len(all_orders) or 1
		rate = round(delivered / total * 100, 2)
		backordered = sum(1 for b in self.backorders.values() if b["tenant_id"] == tenant and b["status"] == "open")
		return {
			"tenant_id": tenant,
			"total_orders": total,
			"delivered": delivered,
			"fulfilment_rate_pct": rate,
			"open_backorders": backordered,
			"generated_at": self._now(),
		}

	async def bulk_confirm_orders(
		self,
		order_ids: list[str],
		confirmed_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Bulk-confirm multiple orders."""
		tenant = self._tenant(tenant_id)
		tasks = [self.confirm_order(oid, confirmed_by, tenant_id=tenant) for oid in order_ids]
		raw = await asyncio.gather(*tasks, return_exceptions=True)
		results, errors = [], []
		for item in raw:
			if isinstance(item, Exception):
				errors.append(str(item))
			else:
				results.append(item)
		return {"confirmed": len(results), "failed": len(errors), "orders": results, "errors": errors}
