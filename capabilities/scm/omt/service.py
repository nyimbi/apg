"""Order Management & Tracking async service (scm_omt)."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

CAPABILITY_ID = "scm_omt"
ORDER_STATUSES = {
	"draft", "confirmed", "allocated", "picking", "packed",
	"shipped", "partially_shipped", "delivered", "cancelled", "on_hold",
}
NOTIFICATION_CHANNELS = {"email", "sms", "push", "webhook"}
ORDER_PRIORITIES = {"urgent", "high", "normal", "low"}
PRIORITY_WEIGHTS = {"urgent": 4.0, "high": 3.0, "normal": 2.0, "low": 1.0}
CUSTOMER_TIER_WEIGHTS = {"strategic": 3.0, "preferred": 2.0, "standard": 1.0}

# Formal state-machine adjacency — only these transitions are permitted.
TRANSITIONS: dict[str, set[str]] = {
	"draft":            {"confirmed", "cancelled", "on_hold"},
	"confirmed":        {"allocated", "cancelled", "on_hold"},
	"allocated":        {"picking", "cancelled", "on_hold"},
	"picking":          {"packed", "on_hold"},
	"packed":           {"shipped", "partially_shipped", "on_hold"},
	"shipped":          {"delivered"},
	"partially_shipped":{"shipped", "delivered", "on_hold"},
	"delivered":        set(),
	"cancelled":        set(),
	"on_hold":          {"confirmed", "cancelled"},
}

# Maximum concurrent tasks for bounded bulk operations.
_BULK_CONCURRENCY = 10


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
		self.rmas: dict[str, dict[str, Any]] = {}  # return merchandise authorizations
		self.customer_tiers: dict[str, str] = {}  # customer_id → tier
		self._order_seq: int = 1000
		self._audit_events: list[dict[str, Any]] = []
		self._idempotency_cache: dict[str, str] = {}  # key → order_id

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

	def _emit(
		self,
		tenant_id: str,
		event_type: str,
		record_id: str,
		record_type: str,
		status: str,
		causation_id: str | None = None,
		correlation_id: str | None = None,
	) -> str:
		"""Append an audit event; return its generated id for causal chaining."""
		event_id = self._id("evt")
		self._audit_events.append({
			"id": event_id,
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"status": status,
			"capability_id": CAPABILITY_ID,
			"causation_id": causation_id,
			"correlation_id": correlation_id,
			"emitted_at": self._now(),
		})
		return event_id

	def _assert_transition(self, order: dict[str, Any], target_status: str) -> None:
		"""Raise ValueError if the target status is not a valid next state."""
		current = order["status"]
		allowed = TRANSITIONS.get(current, set())
		if target_status not in allowed:
			raise ValueError(
				f"cannot transition order '{order['id']}' from '{current}' to '{target_status}'; "
				f"allowed: {sorted(allowed) or '(none)'}"
			)

	async def _bounded_gather(self, coros: list, concurrency: int = _BULK_CONCURRENCY) -> list:
		"""Run coroutines with a bounded semaphore to prevent resource exhaustion."""
		sem = asyncio.Semaphore(concurrency)

		async def _wrap(coro):
			async with sem:
				return await coro

		return list(await asyncio.gather(*[_wrap(c) for c in coros], return_exceptions=True))

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
		idempotency_key: str | None = None,
	) -> dict[str, Any]:
		"""Create a new customer order.

		If *idempotency_key* is supplied and a previous call with the same key succeeded,
		the original order is returned without creating a duplicate.
		"""
		tenant = self._tenant(tenant_id)
		if idempotency_key:
			cache_key = f"{tenant}:{idempotency_key}"
			if cache_key in self._idempotency_cache:
				existing_id = self._idempotency_cache[cache_key]
				if existing_id in self.orders:
					_log.debug("idempotency hit for key=%s order=%s", idempotency_key, existing_id)
					return deepcopy(self.orders[existing_id])
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
		if idempotency_key:
			self._idempotency_cache[f"{tenant}:{idempotency_key}"] = record["id"]
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
		concurrency: int = _BULK_CONCURRENCY,
	) -> dict[str, Any]:
		"""Bulk-confirm multiple orders with bounded concurrency."""
		tenant = self._tenant(tenant_id)
		coros = [self.confirm_order(oid, confirmed_by, tenant_id=tenant) for oid in order_ids]
		raw = await self._bounded_gather(coros, concurrency=concurrency)
		results, errors = [], []
		for item in raw:
			if isinstance(item, Exception):
				errors.append(str(item))
			else:
				results.append(item)
		return {"confirmed": len(results), "failed": len(errors), "orders": results, "errors": errors}

	# ── Order scoring & priority queue ────────────────────────────────────────

	async def set_customer_tier(
		self,
		customer_id: str,
		tier: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Assign a SLA tier to a customer (strategic | preferred | standard)."""
		self._tenant(tenant_id)
		if tier not in CUSTOMER_TIER_WEIGHTS:
			raise ValueError(f"tier must be one of {set(CUSTOMER_TIER_WEIGHTS)}")
		self.customer_tiers[customer_id] = tier
		return {"customer_id": customer_id, "tier": tier, "updated_at": self._now()}

	async def get_order_queue(
		self,
		tenant_id: str | None = None,
		status_filter: str = "confirmed",
	) -> list[dict[str, Any]]:
		"""Return confirmed orders sorted by composite score (revenue × priority × tier).

		The warehouse picks from the top of this list to maximise value delivery.
		"""
		tenant = self._tenant(tenant_id)
		candidates = [
			deepcopy(o) for o in self.orders.values()
			if o["tenant_id"] == tenant and o["status"] == status_filter
		]
		for order in candidates:
			p_weight = PRIORITY_WEIGHTS.get(order.get("priority", "normal"), 2.0)
			tier = self.customer_tiers.get(order["customer_id"], "standard")
			t_weight = CUSTOMER_TIER_WEIGHTS.get(tier, 1.0)
			order["_score"] = round(order.get("total_value", 0.0) * p_weight * t_weight, 4)
		candidates.sort(key=lambda o: o["_score"], reverse=True)
		return candidates

	# ── Order routing ─────────────────────────────────────────────────────────

	async def route_order(
		self,
		order_id: str,
		warehouse_atp_snapshots: list[dict[str, Any]],
		policy: str = "consolidate",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Assign order lines to warehouses according to a routing policy.

		*warehouse_atp_snapshots* — list of dicts with keys:
		  ``warehouse_id``, ``sku``, ``available_quantity``.

		*policy* options:
		  - ``consolidate`` — prefer fewest warehouses (minimize shipment count).
		  - ``fastest``     — use the warehouse with the most surplus stock.

		Returns an assignment plan: list of ``{line_index, sku, warehouse_id, quantity}``.
		"""
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order '{order_id}' not found")
		if policy not in {"consolidate", "fastest"}:
			raise ValueError("policy must be 'consolidate' or 'fastest'")

		# Build a lookup: sku → [(warehouse_id, available_qty)] sorted by available_qty desc
		wh_index: dict[str, list[tuple[str, float]]] = {}
		for snap in warehouse_atp_snapshots:
			wh = snap["warehouse_id"]
			sku = snap["sku"]
			qty = float(snap.get("available_quantity", 0))
			wh_index.setdefault(sku, []).append((wh, qty))
		for sku in wh_index:
			wh_index[sku].sort(key=lambda x: x[1], reverse=True)

		assignments: list[dict[str, Any]] = []
		unroutable: list[dict[str, Any]] = []

		for idx, line in enumerate(order["lines"]):
			sku = line["sku"]
			needed = float(line["quantity"])
			options = wh_index.get(sku, [])

			if policy == "fastest":
				# Pick the single warehouse with the most stock, regardless of whether it
				# can fully cover the line.
				if options:
					wh_id, avail = options[0]
					assignments.append({
						"line_index": idx,
						"sku": sku,
						"warehouse_id": wh_id,
						"quantity": min(needed, avail),
					})
				else:
					unroutable.append({"line_index": idx, "sku": sku, "reason": "no_stock"})
			else:  # consolidate — prefer one warehouse for all lines
				# Find any single warehouse that can cover the full quantity.
				covered = False
				for wh_id, avail in options:
					if avail >= needed:
						assignments.append({
							"line_index": idx,
							"sku": sku,
							"warehouse_id": wh_id,
							"quantity": needed,
						})
						covered = True
						break
				if not covered:
					unroutable.append({"line_index": idx, "sku": sku, "reason": "insufficient_stock"})

		record: dict[str, Any] = {
			"id": self._id("route"),
			"type": "scm_omt_route_plan",
			"tenant_id": tenant,
			"order_id": order_id,
			"policy": policy,
			"assignments": assignments,
			"unroutable_lines": unroutable,
			"created_at": self._now(),
		}
		self._emit(tenant, "order_routed", record["id"], "scm_omt_route_plan", "created")
		return record

	# ── Delivery window negotiation ───────────────────────────────────────────

	async def get_available_delivery_windows(
		self,
		order_id: str,
		candidate_dates: list[str],
		warehouse_calendar: dict[str, Any] | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Return feasible delivery windows for an order from a list of candidates.

		*candidate_dates* — ISO-8601 date strings the customer is considering.
		*warehouse_calendar* — optional dict with ``blackout_dates`` (list[str]) and
		  ``cutoff_time`` (``HH:MM`` UTC).  Dates in the blackout list are excluded.

		Returns feasible and infeasible windows with reasons.
		"""
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order '{order_id}' not found")

		blackout: set[str] = set((warehouse_calendar or {}).get("blackout_dates", []))
		feasible, infeasible = [], []

		for date_str in candidate_dates:
			if date_str in blackout:
				infeasible.append({"date": date_str, "reason": "warehouse_blackout"})
				continue
			# ATP horizon check — if we have an ATP record for any line sku, verify stock.
			shortage_skus: list[str] = []
			for line in order["lines"]:
				sku = line["sku"]
				key = f"{tenant}:{sku}:any"
				atp = self.atp_records.get(key, {})
				if float(atp.get("available_quantity", 0)) < float(line["quantity"]):
					shortage_skus.append(sku)
			if shortage_skus:
				infeasible.append({"date": date_str, "reason": "insufficient_atp", "skus": shortage_skus})
			else:
				feasible.append({"date": date_str, "status": "available"})

		return {
			"order_id": order_id,
			"feasible_windows": feasible,
			"infeasible_windows": infeasible,
			"checked_at": self._now(),
		}

	# ── SLA breach detection ──────────────────────────────────────────────────

	async def detect_sla_breaches(
		self,
		tenant_id: str | None = None,
		escalate: bool = False,
		escalation_recipient: str | None = None,
	) -> dict[str, Any]:
		"""Scan active orders for SLA breaches (now > promised_date, not yet delivered).

		If *escalate* is True and *escalation_recipient* is given, a notification is
		queued for each breached order.

		Returns a summary with the list of breached order ids.
		"""
		tenant = self._tenant(tenant_id)
		now_str = self._now()
		breached: list[dict[str, Any]] = []

		for order in self.orders.values():
			if order["tenant_id"] != tenant:
				continue
			if order["status"] in {"delivered", "cancelled"}:
				continue
			promised = order.get("promised_delivery_date")
			if not promised:
				continue
			if promised < now_str:
				breach = {
					"order_id": order["id"],
					"order_number": order["order_number"],
					"promised_delivery_date": promised,
					"current_status": order["status"],
					"customer_id": order["customer_id"],
				}
				breached.append(breach)
				causation = self._emit(
					tenant, "sla_breach_detected", order["id"], "scm_omt_order", order["status"]
				)
				if escalate and escalation_recipient:
					await self.send_notification(
						order_id=order["id"],
						channel="email",
						event_type="sla_breach",
						message=(
							f"Order {order['order_number']} promised by {promised} "
							f"is still in status '{order['status']}'."
						),
						recipient=escalation_recipient,
						tenant_id=tenant,
					)

		return {
			"tenant_id": tenant,
			"total_breached": len(breached),
			"breached_orders": breached,
			"scanned_at": now_str,
		}

	# ── Re-promising engine ───────────────────────────────────────────────────

	async def re_promise_breached_orders(
		self,
		tenant_id: str | None = None,
		auto_revoke: bool = False,
		new_promise_offset_days: int = 3,
	) -> dict[str, Any]:
		"""Scan active promises, revoke any whose promised_date has passed, and
		optionally re-promise with *new_promise_offset_days* added to today.

		Returns counts of revoked and re-promised records.
		"""
		tenant = self._tenant(tenant_id)
		now_str = self._now()
		today = now_str[:10]
		revoked, repromised = [], []

		for promise in list(self.order_promises.values()):
			if promise["tenant_id"] != tenant or promise["status"] != "active":
				continue
			if promise["promised_date"] < today:
				if auto_revoke:
					promise["status"] = "revoked"
					promise["revocation_reason"] = "system_re_promise_sweep"
					promise["revoked_at"] = self._now()
					self._emit(tenant, "order_promise_revoked", promise["id"], "scm_omt_order_promise", "revoked")
					revoked.append(promise["id"])

					# Re-promise: compute new date (naive offset, no calendar awareness)
					from datetime import timedelta
					new_date_obj = datetime.fromisoformat(today) + timedelta(days=new_promise_offset_days)
					new_date = new_date_obj.date().isoformat()
					new_promise = await self.promise_order(
						order_id=promise["order_id"],
						promised_date=new_date,
						promised_by="system_re_promise",
						confidence_pct=70.0,
						tenant_id=tenant,
					)
					repromised.append(new_promise["id"])

		return {
			"tenant_id": tenant,
			"revoked_count": len(revoked),
			"repromised_count": len(repromised),
			"revoked_promise_ids": revoked,
			"new_promise_ids": repromised,
			"processed_at": self._now(),
		}

	# ── Return Merchandise Authorization (RMA) ────────────────────────────────

	async def create_rma(
		self,
		order_id: str,
		lines: list[dict[str, Any]],
		reason: str,
		requested_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Initiate a return / RMA against a delivered order.

		*lines* — list of dicts with ``sku``, ``return_quantity``, and optional
		``condition`` (``new`` | ``damaged`` | ``missing_parts``).
		"""
		tenant = self._tenant(tenant_id)
		order = self.orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"order '{order_id}' not found")
		if order["status"] != "delivered":
			raise ValueError("RMAs can only be raised against delivered orders")
		if not lines:
			raise ValueError("return lines must not be empty")
		record: dict[str, Any] = {
			"id": self._id("rma"),
			"type": "scm_omt_rma",
			"tenant_id": tenant,
			"order_id": order_id,
			"order_number": order["order_number"],
			"lines": deepcopy(lines),
			"reason": reason,
			"requested_by": requested_by,
			"status": "pending",
			"created_at": self._now(),
		}
		self.rmas[record["id"]] = record
		self._emit(tenant, "rma_created", record["id"], "scm_omt_rma", "pending")
		return deepcopy(record)

	async def approve_rma(
		self,
		rma_id: str,
		approved_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Approve a pending RMA, authorising the customer to ship goods back."""
		tenant = self._tenant(tenant_id)
		rma = self.rmas.get(rma_id)
		if not rma or rma["tenant_id"] != tenant:
			raise KeyError(f"RMA '{rma_id}' not found")
		if rma["status"] != "pending":
			raise ValueError(f"RMA is already '{rma['status']}'")
		rma["status"] = "approved"
		rma["approved_by"] = approved_by
		rma["approved_at"] = self._now()
		self._emit(tenant, "rma_approved", rma_id, "scm_omt_rma", "approved")
		return deepcopy(rma)

	async def receive_return(
		self,
		rma_id: str,
		received_by: str,
		condition_notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record physical receipt of returned goods against an approved RMA."""
		tenant = self._tenant(tenant_id)
		rma = self.rmas.get(rma_id)
		if not rma or rma["tenant_id"] != tenant:
			raise KeyError(f"RMA '{rma_id}' not found")
		if rma["status"] != "approved":
			raise ValueError("can only receive goods against an approved RMA")
		rma["status"] = "received"
		rma["received_by"] = received_by
		rma["received_at"] = self._now()
		if condition_notes:
			rma["condition_notes"] = condition_notes
		self._emit(tenant, "return_received", rma_id, "scm_omt_rma", "received")
		return deepcopy(rma)

	async def list_rmas(
		self,
		tenant_id: str | None = None,
		status: str | None = None,
		order_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List RMAs with optional status and order filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.rmas.values() if r["tenant_id"] == tenant]
		if status:
			items = [r for r in items if r["status"] == status]
		if order_id:
			items = [r for r in items if r["order_id"] == order_id]
		return items

	# ── ATP horizon (date-bucketed) ───────────────────────────────────────────

	async def update_atp_horizon(
		self,
		sku: str,
		supply_events: list[dict[str, Any]],
		demand_events: list[dict[str, Any]],
		opening_stock: float = 0.0,
		warehouse_id: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Build a rolling ATP profile from supply and demand events.

		*supply_events* — list of ``{date: str, quantity: float}`` (PO receipts,
		  production completions, etc.).
		*demand_events* — list of ``{date: str, quantity: float}`` (confirmed orders,
		  forecast consumption, etc.).

		Returns a date-sorted list of buckets showing cumulative ATP at each event date,
		stored under the ATP record and returned for inspection.
		"""
		tenant = self._tenant(tenant_id)
		key = f"{tenant}:{sku}:{warehouse_id or 'any'}"

		# Merge and sort all events chronologically.
		all_events: list[tuple[str, float]] = []
		for e in supply_events:
			all_events.append((e["date"], +float(e["quantity"])))
		for e in demand_events:
			all_events.append((e["date"], -float(e["quantity"])))
		all_events.sort(key=lambda x: x[0])

		buckets: list[dict[str, Any]] = []
		running = opening_stock
		for date_str, delta in all_events:
			running += delta
			buckets.append({"date": date_str, "delta": delta, "cumulative_atp": round(running, 6)})

		record: dict[str, Any] = {
			"id": key,
			"tenant_id": tenant,
			"sku": sku,
			"warehouse_id": warehouse_id,
			"opening_stock": opening_stock,
			"available_quantity": round(running, 6),  # final balance
			"atp_horizon": buckets,
			"updated_at": self._now(),
		}
		self.atp_records[key] = record
		self._emit(tenant, "atp_horizon_updated", key, "scm_omt_atp", "updated")
		return deepcopy(record)

	async def check_atp_by_date(
		self,
		sku: str,
		requested_quantity: float,
		requested_date: str,
		warehouse_id: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Check whether ATP will be >= *requested_quantity* by *requested_date*.

		Uses the stored ATP horizon if available; falls back to point-in-time ATP.
		"""
		tenant = self._tenant(tenant_id)
		key = f"{tenant}:{sku}:{warehouse_id or 'any'}"
		atp_entry = self.atp_records.get(key, {})
		horizon: list[dict[str, Any]] = atp_entry.get("atp_horizon", [])

		if horizon:
			# Find the cumulative ATP at or just before the requested date.
			atp_at_date = float(atp_entry.get("opening_stock", 0))
			for bucket in horizon:
				if bucket["date"] <= requested_date:
					atp_at_date = bucket["cumulative_atp"]
				else:
					break
		else:
			atp_at_date = float(atp_entry.get("available_quantity", 0))

		can_fulfil = atp_at_date >= requested_quantity
		return {
			"tenant_id": tenant,
			"sku": sku,
			"warehouse_id": warehouse_id,
			"requested_quantity": requested_quantity,
			"requested_date": requested_date,
			"atp_at_date": atp_at_date,
			"can_fulfil": can_fulfil,
			"shortage_quantity": max(0.0, requested_quantity - atp_at_date),
			"checked_at": self._now(),
		}
