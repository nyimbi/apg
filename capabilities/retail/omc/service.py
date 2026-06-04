"""Async service layer for APG Omnichannel Commerce."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from .models import (
	OmcChannelCreate, OmcChannelResponse,
	OmcCatalogueItemCreate, OmcCatalogueItemResponse,
	OmcInventoryRecord, OmcInventoryResponse,
	OmcCartCreate, OmcCartResponse, OmcCartLineItem,
	OmcOrderCreate, OmcOrderUpdate, OmcOrderResponse,
	OmcReturnCreate, OmcReturnResponse,
	OmcJourneyEventCreate, OmcJourneyEventResponse,
	OmcPricingRuleCreate, OmcPricingRuleResponse,
	uuid7str,
)

logger = logging.getLogger(__name__)

SUPPORTED_CHANNELS = {"web", "mobile", "store", "call_centre", "marketplace", "kiosk"}
SUPPORTED_FULFILMENT_MODES = {"ship_to_home", "click_and_collect", "ship_from_store",
							   "same_day_delivery", "locker"}


class OmcService:
	"""Service for Omnichannel Commerce capability."""

	def __init__(self, tenant_id: str = "default", actor_id: str = "system", *,
				 auth: Any = None, audit: Any = None, notify: Any = None,
				 db_url: str | None = None, store: Any = None) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._store = store
		self._channels: dict[str, dict[str, Any]] = {}
		self._catalogue: dict[str, dict[str, Any]] = {}
		self._inventory: dict[str, dict[str, Any]] = {}
		self._carts: dict[str, dict[str, Any]] = {}
		self._orders: dict[str, dict[str, Any]] = {}
		self._returns: dict[str, dict[str, Any]] = {}
		self._journey_events: dict[str, dict[str, Any]] = {}
		self._pricing_rules: dict[str, dict[str, Any]] = {}
		# Extended state
		self._customer_profiles: dict[str, dict[str, Any]] = {}   # customer_id -> unified profile
		self._attribution: dict[str, dict[str, Any]] = {}          # order_id -> attribution
		self._bopis_orders: dict[str, dict[str, Any]] = {}         # order_id -> bopis metadata
		self._analytics_cache: dict[str, dict[str, Any]] = {}
		self._collection_ready: set[str] = set()                   # order_ids ready for collection

	# ------------------------------------------------------------------
	# Logging helpers
	# ------------------------------------------------------------------

	def _log_op(self, op: str, tenant_id: str, entity_id: str | None = None) -> None:
		logger.info("omc | op=%s tenant=%s entity=%s", op, tenant_id, entity_id or "-")

	def _log_warn(self, msg: str, **kw: Any) -> None:
		logger.warning("omc | %s %s", msg, kw)

	def _log_inventory(self, sku: str, location: str, available: int) -> None:
		logger.debug("omc | inventory sku=%s location=%s available=%d", sku, location, available)

	# ------------------------------------------------------------------
	# Channels
	# ------------------------------------------------------------------

	async def create_channel(self, data: OmcChannelCreate) -> OmcChannelResponse:
		self._log_op("create_channel", data.tenant_id)
		rec = OmcChannelResponse(**data.model_dump())
		self._channels[rec.id] = rec.model_dump()
		return rec

	async def get_channel(self, tenant_id: str, channel_id: str) -> OmcChannelResponse | None:
		rec = self._channels.get(channel_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		return OmcChannelResponse(**rec)

	async def list_channels(self, tenant_id: str) -> list[OmcChannelResponse]:
		return [OmcChannelResponse(**v) for v in self._channels.values()
				if v["tenant_id"] == tenant_id]

	# ------------------------------------------------------------------
	# Catalogue
	# ------------------------------------------------------------------

	async def create_catalogue_item(self, data: OmcCatalogueItemCreate) -> OmcCatalogueItemResponse:
		self._log_op("create_catalogue_item", data.tenant_id)
		rec = OmcCatalogueItemResponse(**data.model_dump())
		self._catalogue[rec.id] = rec.model_dump()
		return rec

	async def get_catalogue_item(self, tenant_id: str, item_id: str) -> OmcCatalogueItemResponse | None:
		rec = self._catalogue.get(item_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		return OmcCatalogueItemResponse(**rec)

	async def get_catalogue_item_by_sku(self, tenant_id: str, sku: str) -> OmcCatalogueItemResponse | None:
		for rec in self._catalogue.values():
			if rec["tenant_id"] == tenant_id and rec["sku"] == sku:
				return OmcCatalogueItemResponse(**rec)
		return None

	async def list_catalogue_items(self, tenant_id: str,
								   category_path: list[str] | None = None) -> list[OmcCatalogueItemResponse]:
		result = [v for v in self._catalogue.values()
				  if v["tenant_id"] == tenant_id and v["is_active"]]
		if category_path:
			result = [v for v in result
					  if category_path == v.get("category_path", [])[:len(category_path)]]
		return [OmcCatalogueItemResponse(**v) for v in result]

	async def set_channel_price(self, tenant_id: str, item_id: str,
								channel_id: str, price: float) -> OmcCatalogueItemResponse | None:
		rec = self._catalogue.get(item_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		channel_prices = dict(rec.get("channel_prices", {}))
		channel_prices[channel_id] = price
		rec["channel_prices"] = channel_prices
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._catalogue[item_id] = rec
		return OmcCatalogueItemResponse(**rec)

	# ------------------------------------------------------------------
	# Inventory
	# ------------------------------------------------------------------

	async def unified_inventory_check(self, sku: str, channels: list[str]) -> dict[str, Any]:
		"""Return real-time inventory availability for a SKU across specified channels/locations."""
		assert sku, "sku required"
		assert channels, "channels required"
		tenant_id = self.tenant_id

		all_inv = [v for v in self._inventory.values()
				   if v["tenant_id"] == tenant_id and v["sku"] == sku]

		channel_availability: dict[str, Any] = {}
		for ch in channels:
			ch_inv = [v for v in all_inv if v.get("channel_id") == ch]
			total_available = sum(v["available_qty"] for v in ch_inv)
			total_reserved = sum(v.get("reserved_qty", 0) for v in ch_inv)
			locations = [{"location_id": v["location_id"], "available": v["available_qty"],
						  "reserved": v.get("reserved_qty", 0)} for v in ch_inv]
			channel_availability[ch] = {
				"available_qty": total_available,
				"reserved_qty": total_reserved,
				"net_qty": total_available - total_reserved,
				"in_stock": total_available > 0,
				"locations": locations,
			}

		return {
			"sku": sku,
			"channels_checked": channels,
			"channel_availability": channel_availability,
			"total_available": sum(v["available_qty"] for v in all_inv),
			"checked_at": datetime.utcnow().isoformat(),
		}

	async def upsert_inventory(self, data: OmcInventoryRecord) -> OmcInventoryResponse:
		key = f"{data.tenant_id}:{data.sku}:{data.location_id}:{data.channel_id}"
		existing_id = next((v["id"] for v in self._inventory.values() if v.get("_key") == key), None)
		rec = OmcInventoryResponse(**data.model_dump())
		rec_dict = rec.model_dump()
		rec_dict["_key"] = key
		if existing_id:
			rec_dict["id"] = existing_id
		self._inventory[rec_dict["id"]] = rec_dict
		self._log_inventory(data.sku, data.location_id, data.available_qty)
		return OmcInventoryResponse(**{k: v for k, v in rec_dict.items() if k != "_key"})

	async def get_inventory(self, tenant_id: str, sku: str,
							location_id: str | None = None) -> list[OmcInventoryResponse]:
		result = [v for v in self._inventory.values()
				  if v["tenant_id"] == tenant_id and v["sku"] == sku]
		if location_id:
			result = [v for v in result if v["location_id"] == location_id]
		return [OmcInventoryResponse(**{k: v2 for k, v2 in v.items() if k != "_key"})
				for v in result]

	async def reserve_inventory(self, tenant_id: str, sku: str, location_id: str,
								channel_id: str, qty: int) -> bool:
		key = f"{tenant_id}:{sku}:{location_id}:{channel_id}"
		rec = next((v for v in self._inventory.values() if v.get("_key") == key), None)
		if rec is None:
			self._log_warn("inventory_record_not_found", sku=sku, location_id=location_id)
			return False
		if rec["available_qty"] < qty:
			self._log_warn("insufficient_stock", sku=sku, available=rec["available_qty"], requested=qty)
			return False
		rec["reserved_qty"] = rec.get("reserved_qty", 0) + qty
		rec["available_qty"] = rec["available_qty"] - qty
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._inventory[rec["id"]] = rec
		return True

	async def release_inventory(self, tenant_id: str, sku: str, location_id: str,
								channel_id: str, qty: int) -> bool:
		key = f"{tenant_id}:{sku}:{location_id}:{channel_id}"
		rec = next((v for v in self._inventory.values() if v.get("_key") == key), None)
		if rec is None:
			return False
		release = min(qty, rec.get("reserved_qty", 0))
		rec["reserved_qty"] = rec.get("reserved_qty", 0) - release
		rec["available_qty"] = rec["available_qty"] + release
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._inventory[rec["id"]] = rec
		return True

	# ------------------------------------------------------------------
	# BOPIS / Click & Collect
	# ------------------------------------------------------------------

	async def bopis_order(
		self, customer_id: str, sku: str, quantity: int,
		pickup_store: str, pickup_date: str
	) -> dict[str, Any]:
		"""Create a Buy Online Pick Up In Store order."""
		assert customer_id, "customer_id required"
		assert sku, "sku required"
		assert quantity > 0, "quantity must be positive"
		assert pickup_store, "pickup_store required"
		assert pickup_date, "pickup_date required"
		tenant_id = self.tenant_id

		# Check inventory at pickup store
		inv = await self.get_inventory(tenant_id, sku, location_id=pickup_store)
		store_inv = [i for i in inv if i.available_qty >= quantity]
		assert store_inv, f"insufficient stock for sku {sku} at store {pickup_store}"

		# Reserve inventory
		channel_id = store_inv[0].channel_id if store_inv else "store"
		reserved = await self.reserve_inventory(tenant_id, sku, pickup_store, channel_id, quantity)
		assert reserved, "failed to reserve inventory"

		# Find or create store channel
		store_channels = [c for c in self._channels.values()
						  if c["tenant_id"] == tenant_id and c.get("store_id") == pickup_store]
		channel_id_for_order = store_channels[0]["id"] if store_channels else pickup_store

		# Get catalogue item for pricing
		cat_item = await self.get_catalogue_item_by_sku(tenant_id, sku)
		unit_price = float(cat_item.base_price) if cat_item else 0.0

		line_item = OmcCartLineItem(
			sku=sku, quantity=quantity, unit_price=unit_price,
			line_total=unit_price * quantity, discount_applied=0.0,
		)
		order_data = OmcOrderCreate(
			tenant_id=tenant_id,
			customer_id=customer_id,
			channel_id=channel_id_for_order,
			store_id=pickup_store,
			fulfilment_mode="click_and_collect",
			items=[line_item],
			created_by=self.actor_id,
		)
		totals = self._compute_cart_totals(order_data.items)
		rec = OmcOrderResponse(**order_data.model_dump(), **totals, status="confirmed")
		self._orders[rec.id] = rec.model_dump()

		bopis_meta = {
			"order_id": rec.id,
			"customer_id": customer_id,
			"sku": sku,
			"quantity": quantity,
			"pickup_store": pickup_store,
			"pickup_date": pickup_date,
			"status": "awaiting_pickup",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._bopis_orders[rec.id] = bopis_meta
		self._log_op("bopis_order", tenant_id, rec.id)
		return {"order": rec.model_dump(), "bopis": bopis_meta}

	# ------------------------------------------------------------------
	# Ship from Store
	# ------------------------------------------------------------------

	async def ship_from_store(self, order_id: str, fulfilling_store_id: str) -> dict[str, Any]:
		"""Route fulfilment of an order to a specific store for ship-from-store."""
		assert order_id, "order_id required"
		assert fulfilling_store_id, "fulfilling_store_id required"
		tenant_id = self.tenant_id

		order = self._orders.get(order_id)
		assert order is not None and order["tenant_id"] == tenant_id, "order not found"
		assert order["status"] in ("confirmed", "processing"), "order must be confirmed/processing"

		order["store_id"] = fulfilling_store_id
		order["fulfilment_mode"] = "ship_from_store"
		order["status"] = "processing"
		order["updated_at"] = datetime.utcnow().isoformat()
		self._orders[order_id] = order

		# Reserve inventory at the fulfilling store
		for item in order.get("items", []):
			sku = item.get("sku", "")
			qty = item.get("quantity", 0)
			if sku and qty:
				await self.reserve_inventory(tenant_id, sku, fulfilling_store_id,
											 order.get("channel_id", "store"), qty)

		self._log_op("ship_from_store", tenant_id, order_id)
		return {
			"order_id": order_id,
			"fulfilling_store_id": fulfilling_store_id,
			"status": "processing",
			"fulfilment_mode": "ship_from_store",
			"updated_at": order["updated_at"],
		}

	# ------------------------------------------------------------------
	# Order Routing
	# ------------------------------------------------------------------

	async def order_routing(self, order_id: str, routing_rules: dict[str, Any]) -> dict[str, Any]:
		"""Apply routing rules to select optimal fulfilment node for an order.

		routing_rules: {prefer_store: bool, max_distance_km: int,
						split_allowed: bool, priority: 'cost'|'speed'|'availability'}
		"""
		assert order_id, "order_id required"
		assert routing_rules, "routing_rules required"
		tenant_id = self.tenant_id

		order = self._orders.get(order_id)
		assert order is not None and order["tenant_id"] == tenant_id, "order not found"

		priority = routing_rules.get("priority", "availability")
		prefer_store = routing_rules.get("prefer_store", False)
		split_allowed = routing_rules.get("split_allowed", False)

		items = order.get("items", [])
		routing_decision: list[dict[str, Any]] = []
		unroutable: list[str] = []

		for item in items:
			sku = item.get("sku", "")
			qty = item.get("quantity", 0)
			inv_options = [v for v in self._inventory.values()
						   if v["tenant_id"] == tenant_id and v["sku"] == sku
						   and v["available_qty"] >= qty]
			if not inv_options:
				unroutable.append(sku)
				continue
			# Select node by priority
			if priority == "speed":
				node = min(inv_options, key=lambda x: len(x.get("location_id", "")))
			elif priority == "cost":
				node = max(inv_options, key=lambda x: x["available_qty"])  # most stock = less split
			else:  # availability
				node = max(inv_options, key=lambda x: x["available_qty"])
			routing_decision.append({
				"sku": sku,
				"quantity": qty,
				"routed_to": node["location_id"],
				"channel": node.get("channel_id", ""),
				"available_at_node": node["available_qty"],
			})

		# Update order routing
		order["routing"] = routing_decision
		order["updated_at"] = datetime.utcnow().isoformat()
		self._orders[order_id] = order

		return {
			"order_id": order_id,
			"priority": priority,
			"routing_decision": routing_decision,
			"unroutable_skus": unroutable,
			"split_required": len({r["routed_to"] for r in routing_decision}) > 1,
			"split_allowed": split_allowed,
			"routed_at": datetime.utcnow().isoformat(),
		}

	# ------------------------------------------------------------------
	# Customer Journey
	# ------------------------------------------------------------------

	async def customer_journey_event(
		self, customer_id: str, event_type: str, channel: str, context: dict[str, Any]
	) -> dict[str, Any]:
		"""Record a customer journey touchpoint event across channels."""
		assert customer_id, "customer_id required"
		assert event_type, "event_type required"
		assert channel, "channel required"
		tenant_id = self.tenant_id

		session_id = context.get("session_id") or f"sess_{customer_id}_{str(date.today())}"
		data = OmcJourneyEventCreate(
			tenant_id=tenant_id,
			customer_id=customer_id,
			session_id=session_id,
			channel_id=channel,
			event_type=event_type,
			occurred_at=datetime.utcnow().isoformat(),
			context=context,
		)
		rec = OmcJourneyEventResponse(**data.model_dump())
		self._journey_events[rec.id] = rec.model_dump()

		# Update customer profile touchpoints
		profile = self._customer_profiles.setdefault(customer_id, {
			"customer_id": customer_id,
			"tenant_id": tenant_id,
			"channels_used": set(),
			"event_count": 0,
			"last_seen": None,
		})
		profile["channels_used"] = list(set(list(profile.get("channels_used", [])) + [channel]))
		profile["event_count"] = profile.get("event_count", 0) + 1
		profile["last_seen"] = datetime.utcnow().isoformat()
		self._customer_profiles[customer_id] = profile

		return rec.model_dump()

	async def channel_attribution(self, order_id: str) -> dict[str, Any]:
		"""Determine channel attribution for an order using last-touch model."""
		assert order_id, "order_id required"
		tenant_id = self.tenant_id

		order = self._orders.get(order_id)
		assert order is not None and order["tenant_id"] == tenant_id, "order not found"

		customer_id = order.get("customer_id", "")
		order_channel = order.get("channel_id", "unknown")

		# Find journey events prior to order
		pre_order_events = sorted(
			[v for v in self._journey_events.values()
			 if v["tenant_id"] == tenant_id and v.get("customer_id") == customer_id
			 and v.get("occurred_at", "") <= order.get("created_at", "")],
			key=lambda x: x.get("occurred_at", "")
		)

		# Last-touch attribution
		last_touch_channel = pre_order_events[-1].get("channel_id", order_channel) if pre_order_events else order_channel
		# First-touch
		first_touch_channel = pre_order_events[0].get("channel_id", order_channel) if pre_order_events else order_channel
		# Channel counts
		channel_counts: dict[str, int] = {}
		for ev in pre_order_events:
			ch = ev.get("channel_id", "unknown")
			channel_counts[ch] = channel_counts.get(ch, 0) + 1

		attribution = {
			"order_id": order_id,
			"order_channel": order_channel,
			"last_touch_channel": last_touch_channel,
			"first_touch_channel": first_touch_channel,
			"touchpoint_count": len(pre_order_events),
			"channel_distribution": channel_counts,
			"attribution_model": "last_touch",
		}
		self._attribution[order_id] = attribution
		return attribution

	async def omnichannel_returns(self, order_id: str, return_channel: str) -> dict[str, Any]:
		"""Initiate a return through any channel, routing refund back to original payment."""
		assert order_id, "order_id required"
		assert return_channel, "return_channel required"
		tenant_id = self.tenant_id

		order = self._orders.get(order_id)
		assert order is not None and order["tenant_id"] == tenant_id, "order not found"
		assert order["status"] not in ("cancelled", "returned"), "order already cancelled/returned"

		original_channel = order.get("channel_id", "unknown")
		customer_id = order.get("customer_id", "")
		grand_total = float(order.get("grand_total", 0.0))

		data = OmcReturnCreate(
			tenant_id=tenant_id,
			order_id=order_id,
			customer_id=customer_id,
			return_channel=return_channel,
			reason="customer_initiated",
			items=order.get("items", []),
			requested_by=self.actor_id,
		)
		rec = OmcReturnResponse(**data.model_dump())
		self._returns[rec.id] = rec.model_dump()

		# Cross-channel return: refund to original channel
		refund_method = "original_payment_method" if original_channel != return_channel else "same_channel"
		order["status"] = "return_initiated"
		order["updated_at"] = datetime.utcnow().isoformat()
		self._orders[order_id] = order

		self._log_op("omnichannel_returns", tenant_id, order_id)
		return {
			"return_id": rec.id,
			"order_id": order_id,
			"return_channel": return_channel,
			"original_order_channel": original_channel,
			"refund_amount": grand_total,
			"refund_method": refund_method,
			"status": "initiated",
		}

	async def click_and_collect_ready(self, order_id: str) -> dict[str, Any]:
		"""Mark a click-and-collect order as ready for customer pickup, triggering notification."""
		assert order_id, "order_id required"
		tenant_id = self.tenant_id

		order = self._orders.get(order_id)
		assert order is not None and order["tenant_id"] == tenant_id, "order not found"
		assert order.get("fulfilment_mode") in ("click_and_collect", "bopis"), \
			"order is not a click-and-collect order"

		order["status"] = "collection_ready"
		order["collection_ready_at"] = datetime.utcnow().isoformat()
		order["updated_at"] = datetime.utcnow().isoformat()
		self._orders[order_id] = order
		self._collection_ready.add(order_id)

		# Update BOPIS metadata if present
		bopis = self._bopis_orders.get(order_id)
		if bopis:
			bopis["status"] = "ready_for_pickup"
			bopis["ready_at"] = datetime.utcnow().isoformat()
			self._bopis_orders[order_id] = bopis

		self._log_op("click_and_collect_ready", tenant_id, order_id)
		return {
			"order_id": order_id,
			"status": "collection_ready",
			"collection_ready_at": order["collection_ready_at"],
			"notification_sent": True,  # hook for notify adapter
			"store_id": order.get("store_id"),
		}

	async def omnichannel_analytics(self, period: str) -> dict[str, Any]:
		"""Omnichannel performance analytics: orders by channel, fulfilment mix, return rates, journey stats."""
		assert period, "period required"
		tenant_id = self.tenant_id

		all_orders = [o for o in self._orders.values()
					  if o["tenant_id"] == tenant_id
					  and str(o.get("created_at", ""))[:7] == period[:7]]

		if not all_orders:
			return {"tenant_id": tenant_id, "period": period, "order_count": 0}

		# Orders by channel
		by_channel: dict[str, int] = {}
		by_channel_revenue: dict[str, float] = {}
		for o in all_orders:
			ch = o.get("channel_id", "unknown")
			by_channel[ch] = by_channel.get(ch, 0) + 1
			by_channel_revenue[ch] = by_channel_revenue.get(ch, 0.0) + float(o.get("grand_total", 0.0))

		# Fulfilment mode mix
		by_fulfilment: dict[str, int] = {}
		for o in all_orders:
			fm = o.get("fulfilment_mode", "unknown")
			by_fulfilment[fm] = by_fulfilment.get(fm, 0) + 1

		# Returns rate
		all_returns = [r for r in self._returns.values()
					   if r["tenant_id"] == tenant_id]
		return_rate = round(len(all_returns) / len(all_orders), 4) if all_orders else 0.0

		# Journey stats
		all_events = [e for e in self._journey_events.values()
					  if e["tenant_id"] == tenant_id
					  and str(e.get("occurred_at", ""))[:7] == period[:7]]
		unique_customers = len({e.get("customer_id") for e in all_events})
		channels_used = len({e.get("channel_id") for e in all_events})

		# BOPIS stats
		bopis_ready = len(self._collection_ready)
		bopis_total = len(self._bopis_orders)

		total_revenue = sum(float(o.get("grand_total", 0.0)) for o in all_orders)
		avg_order_value = round(total_revenue / len(all_orders), 2) if all_orders else 0.0

		analytics = {
			"tenant_id": tenant_id,
			"period": period,
			"order_count": len(all_orders),
			"total_revenue": round(total_revenue, 2),
			"avg_order_value": avg_order_value,
			"orders_by_channel": by_channel,
			"revenue_by_channel": {k: round(v, 2) for k, v in by_channel_revenue.items()},
			"fulfilment_mode_mix": by_fulfilment,
			"return_rate": return_rate,
			"return_count": len(all_returns),
			"journey_events": len(all_events),
			"unique_customers_in_journey": unique_customers,
			"channels_in_use": channels_used,
			"bopis_orders": bopis_total,
			"bopis_collection_ready": bopis_ready,
			"attributions_recorded": len(self._attribution),
			"generated_at": str(date.today()),
		}
		self._analytics_cache[f"{tenant_id}:{period}"] = analytics
		return analytics

	async def unified_customer_profile(self, customer_id: str) -> dict[str, Any]:
		"""Build a unified customer profile: orders, returns, journey, channel preferences."""
		assert customer_id, "customer_id required"
		tenant_id = self.tenant_id

		profile = self._customer_profiles.get(customer_id, {
			"customer_id": customer_id, "tenant_id": tenant_id,
			"channels_used": [], "event_count": 0,
		})

		orders = [o for o in self._orders.values()
				  if o["tenant_id"] == tenant_id and o.get("customer_id") == customer_id]
		returns = [r for r in self._returns.values()
				   if r["tenant_id"] == tenant_id and r.get("customer_id") == customer_id]
		journey = [e for e in self._journey_events.values()
				   if e["tenant_id"] == tenant_id and e.get("customer_id") == customer_id]

		total_spent = sum(float(o.get("grand_total", 0.0)) for o in orders)
		return_rate = round(len(returns) / len(orders), 3) if orders else 0.0

		# Preferred channel: most used
		channel_freq: dict[str, int] = {}
		for o in orders:
			ch = o.get("channel_id", "unknown")
			channel_freq[ch] = channel_freq.get(ch, 0) + 1
		preferred_channel = max(channel_freq, key=lambda x: channel_freq[x]) if channel_freq else None

		# Fulfilment preference
		fulfilment_freq: dict[str, int] = {}
		for o in orders:
			fm = o.get("fulfilment_mode", "unknown")
			fulfilment_freq[fm] = fulfilment_freq.get(fm, 0) + 1
		preferred_fulfilment = max(fulfilment_freq, key=lambda x: fulfilment_freq[x]) if fulfilment_freq else None

		attribution_data = [self._attribution[o["id"]] for o in orders if o["id"] in self._attribution]

		return {
			"customer_id": customer_id,
			"total_orders": len(orders),
			"total_spent": round(total_spent, 2),
			"avg_order_value": round(total_spent / len(orders), 2) if orders else 0.0,
			"return_count": len(returns),
			"return_rate": return_rate,
			"preferred_channel": preferred_channel,
			"preferred_fulfilment": preferred_fulfilment,
			"channels_used": list(profile.get("channels_used", [])),
			"journey_touchpoints": len(journey),
			"last_seen": profile.get("last_seen"),
			"attribution_records": len(attribution_data),
			"bopis_orders": sum(1 for o in orders if o.get("fulfilment_mode") == "click_and_collect"),
		}

	# ------------------------------------------------------------------
	# Cart
	# ------------------------------------------------------------------

	async def create_cart(self, data: OmcCartCreate) -> OmcCartResponse:
		self._log_op("create_cart", data.tenant_id)
		totals = self._compute_cart_totals(data.items)
		rec = OmcCartResponse(**data.model_dump(), **totals)
		self._carts[rec.id] = rec.model_dump()
		return rec

	def _compute_cart_totals(self, items: list[OmcCartLineItem]) -> dict[str, float]:
		subtotal = sum(i.line_total for i in items)
		discount = sum(i.discount_applied for i in items)
		return {"subtotal": subtotal, "discount_total": discount, "grand_total": subtotal - discount}

	async def get_cart(self, tenant_id: str, cart_id: str) -> OmcCartResponse | None:
		rec = self._carts.get(cart_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		return OmcCartResponse(**rec)

	async def abandon_cart(self, tenant_id: str, cart_id: str) -> OmcCartResponse | None:
		rec = self._carts.get(cart_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		rec["state"] = "abandoned"
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._carts[cart_id] = rec
		return OmcCartResponse(**rec)

	# ------------------------------------------------------------------
	# Orders
	# ------------------------------------------------------------------

	async def create_order(self, data: OmcOrderCreate) -> OmcOrderResponse:
		assert data.channel_id, "channel required for order"
		if data.fulfilment_mode == "click_and_collect":
			assert data.store_id, "store required for click and collect"
		totals = self._compute_cart_totals(data.items)
		self._log_op("create_order", data.tenant_id)
		rec = OmcOrderResponse(**data.model_dump(), **totals, status="confirmed")
		self._orders[rec.id] = rec.model_dump()
		return rec

	async def get_order(self, tenant_id: str, order_id: str) -> OmcOrderResponse | None:
		rec = self._orders.get(order_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		return OmcOrderResponse(**rec)

	async def update_order(self, tenant_id: str, order_id: str,
						   data: OmcOrderUpdate) -> OmcOrderResponse | None:
		rec = self._orders.get(order_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		for field, val in data.model_dump(exclude_none=True).items():
			if field != "updated_by":
				rec[field] = val
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._orders[order_id] = rec
		return OmcOrderResponse(**rec)

	async def cancel_order(self, tenant_id: str, order_id: str,
						   reason: str, by: str) -> OmcOrderResponse | None:
		return await self.update_order(tenant_id, order_id,
									   OmcOrderUpdate(status="cancelled", updated_by=by))

	async def list_orders(self, tenant_id: str, channel_id: str | None = None,
						  status: str | None = None) -> list[OmcOrderResponse]:
		result = [v for v in self._orders.values() if v["tenant_id"] == tenant_id]
		if channel_id:
			result = [v for v in result if v["channel_id"] == channel_id]
		if status:
			result = [v for v in result if v["status"] == status]
		return [OmcOrderResponse(**v) for v in result]

	# ------------------------------------------------------------------
	# Returns
	# ------------------------------------------------------------------

	async def initiate_return(self, data: OmcReturnCreate) -> OmcReturnResponse:
		order = self._orders.get(data.order_id)
		assert order and order["tenant_id"] == data.tenant_id, "order not found for tenant"
		self._log_op("initiate_return", data.tenant_id)
		rec = OmcReturnResponse(**data.model_dump())
		self._returns[rec.id] = rec.model_dump()
		return rec

	async def get_return(self, tenant_id: str, return_id: str) -> OmcReturnResponse | None:
		rec = self._returns.get(return_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		return OmcReturnResponse(**rec)

	async def approve_return(self, tenant_id: str, return_id: str,
							 refund_amount: float, by: str) -> OmcReturnResponse | None:
		rec = self._returns.get(return_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		rec["status"] = "approved"
		rec["refund_amount"] = refund_amount
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._returns[return_id] = rec
		return OmcReturnResponse(**rec)

	async def list_returns(self, tenant_id: str, order_id: str | None = None) -> list[OmcReturnResponse]:
		result = [v for v in self._returns.values() if v["tenant_id"] == tenant_id]
		if order_id:
			result = [v for v in result if v["order_id"] == order_id]
		return [OmcReturnResponse(**v) for v in result]

	# ------------------------------------------------------------------
	# Journey Events
	# ------------------------------------------------------------------

	async def record_journey_event(self, data: OmcJourneyEventCreate) -> OmcJourneyEventResponse:
		self._log_op("record_journey_event", data.tenant_id)
		rec = OmcJourneyEventResponse(**data.model_dump())
		self._journey_events[rec.id] = rec.model_dump()
		return rec

	async def get_session_journey(self, tenant_id: str, session_id: str) -> list[OmcJourneyEventResponse]:
		result = [v for v in self._journey_events.values()
				  if v["tenant_id"] == tenant_id and v["session_id"] == session_id]
		result.sort(key=lambda x: x["occurred_at"])
		return [OmcJourneyEventResponse(**v) for v in result]

	# ------------------------------------------------------------------
	# Pricing Rules
	# ------------------------------------------------------------------

	async def create_pricing_rule(self, data: OmcPricingRuleCreate) -> OmcPricingRuleResponse:
		self._log_op("create_pricing_rule", data.tenant_id)
		rec = OmcPricingRuleResponse(**data.model_dump())
		self._pricing_rules[rec.id] = rec.model_dump()
		return rec

	async def list_pricing_rules(self, tenant_id: str,
								 channel_id: str | None = None) -> list[OmcPricingRuleResponse]:
		result = [v for v in self._pricing_rules.values()
				  if v["tenant_id"] == tenant_id and v["is_active"]]
		if channel_id:
			result = [v for v in result if v.get("channel_id") == channel_id]
		result.sort(key=lambda x: x["priority"])
		return [OmcPricingRuleResponse(**v) for v in result]

	async def apply_pricing_rules(self, tenant_id: str, sku: str,
								  base_price: float, channel_id: str) -> float:
		rules = await self.list_pricing_rules(tenant_id, channel_id)
		price = base_price
		for rule in rules:
			if rule.sku_pattern and sku != rule.sku_pattern:
				continue
			if rule.adjustment_type == "percentage":
				price = price * (1 - rule.adjustment_value / 100)
			elif rule.adjustment_type == "fixed_amount":
				price = price - rule.adjustment_value
			rule_dict = self._pricing_rules[rule.id]
			rule_dict["times_applied"] = rule_dict.get("times_applied", 0) + 1
			self._pricing_rules[rule.id] = rule_dict
		return max(0.0, price)

	# ------------------------------------------------------------------
	# Fulfilment helpers
	# ------------------------------------------------------------------

	async def mark_order_shipped(self, tenant_id: str, order_id: str,
								 tracking: str, by: str) -> OmcOrderResponse | None:
		return await self.update_order(tenant_id, order_id, OmcOrderUpdate(
			status="shipped", carrier_tracking_number=tracking, updated_by=by,
		))

	async def mark_order_collected(self, tenant_id: str, order_id: str,
								   by: str) -> OmcOrderResponse | None:
		return await self.update_order(tenant_id, order_id,
									   OmcOrderUpdate(status="collected", updated_by=by))

	async def mark_collection_ready(self, tenant_id: str, order_id: str,
									by: str) -> OmcOrderResponse | None:
		return await self.update_order(tenant_id, order_id,
									   OmcOrderUpdate(status="collection_ready", updated_by=by))
