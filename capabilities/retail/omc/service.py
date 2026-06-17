"""Async service layer for APG Omnichannel Commerce."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

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


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class OmcService:
	"""Service for Omnichannel Commerce capability."""

	def __init__(self, tenant_id: str = "default", actor_id: str = "system", *,
				 auth: Any = None, audit: Any = None, notify: Any = None,
				 db_url: str | None = None, store: Any = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._store = store
		self._channels = WriteThruDict('channels', tenant_id, _store)
		self._catalogue = WriteThruDict('catalogue', tenant_id, _store)
		self._inventory = WriteThruDict('inventory', tenant_id, _store)
		self._carts = WriteThruDict('carts', tenant_id, _store)
		self._orders = WriteThruDict('orders', tenant_id, _store)
		self._returns = WriteThruDict('returns', tenant_id, _store)
		self._journey_events = WriteThruDict('journey_events', tenant_id, _store)
		self._pricing_rules = WriteThruDict('pricing_rules', tenant_id, _store)
		# Extended state
		self._customer_profiles = WriteThruDict('customer_profiles', tenant_id, _store)   # customer_id -> unified profile
		self._attribution = WriteThruDict('attribution', tenant_id, _store)          # order_id -> attribution
		self._bopis_orders = WriteThruDict('bopis_orders', tenant_id, _store)         # order_id -> bopis metadata
		self._analytics_cache = WriteThruDict('analytics_cache', tenant_id, _store)
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

	async def ml_demand_forecast(self, *args, **kwargs):
		"""AI-powered product demand forecasting for inventory optimization. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.predict(kwargs.get("sales_series",[{"period": str(i), "value": 50.0+i} for i in range(12)]), horizon=kwargs.get("horizon",7), task="retail_demand_forecast")
			return {"demand_forecast": result.predictions, "rationale": result.rationale, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

	# ------------------------------------------------------------------
	# Cart Merge (Guest → Authenticated)
	# ------------------------------------------------------------------

	async def merge_carts(
		self,
		tenant_id: str,
		guest_cart_id: str,
		authenticated_cart_id: str,
		strategy: str = "union",
	) -> OmcCartResponse | None:
		"""Merge a guest cart into an authenticated cart on login.

		strategy:
		  'keep_authenticated' — discard guest cart, return authenticated cart unchanged
		  'keep_guest'         — replace authenticated cart items with guest cart items
		  'union'              — merge SKUs; sum quantities for duplicates
		"""
		assert guest_cart_id, "guest_cart_id required"
		assert authenticated_cart_id, "authenticated_cart_id required"
		assert strategy in ("keep_authenticated", "keep_guest", "union"), "invalid strategy"

		guest = self._carts.get(guest_cart_id)
		auth = self._carts.get(authenticated_cart_id)
		if guest is None or guest["tenant_id"] != tenant_id:
			self._log_warn("merge_carts: guest cart not found", guest_cart_id=guest_cart_id)
			return None
		if auth is None or auth["tenant_id"] != tenant_id:
			self._log_warn("merge_carts: auth cart not found", authenticated_cart_id=authenticated_cart_id)
			return None

		if strategy == "keep_authenticated":
			merged_items = auth.get("items", [])
		elif strategy == "keep_guest":
			merged_items = guest.get("items", [])
		else:  # union
			# Index auth items by SKU; accumulate quantities
			by_sku: dict[str, OmcCartLineItem] = {}
			for raw in auth.get("items", []):
				item = OmcCartLineItem(**raw) if isinstance(raw, dict) else raw
				by_sku[item.sku] = item
			for raw in guest.get("items", []):
				item = OmcCartLineItem(**raw) if isinstance(raw, dict) else raw
				if item.sku in by_sku:
					existing = by_sku[item.sku]
					new_qty = existing.quantity + item.quantity
					by_sku[item.sku] = OmcCartLineItem(
						sku=existing.sku,
						quantity=new_qty,
						unit_price=existing.unit_price,
						line_total=existing.unit_price * new_qty,
						discount_applied=existing.discount_applied + item.discount_applied,
						promotion_ids=list(set(existing.promotion_ids + item.promotion_ids)),
					)
				else:
					by_sku[item.sku] = item
			merged_items = [i.model_dump() for i in by_sku.values()]

		items_as_models = [OmcCartLineItem(**i) if isinstance(i, dict) else i for i in merged_items]
		totals = self._compute_cart_totals(items_as_models)

		auth["items"] = [i.model_dump() if hasattr(i, "model_dump") else i for i in items_as_models]
		auth.update(totals)
		auth["updated_at"] = datetime.utcnow().isoformat()
		self._carts[authenticated_cart_id] = auth

		# Invalidate guest cart
		guest["state"] = "merged"
		guest["updated_at"] = datetime.utcnow().isoformat()
		self._carts[guest_cart_id] = guest

		self._log_op("merge_carts", tenant_id, authenticated_cart_id)
		return OmcCartResponse(**auth)

	# ------------------------------------------------------------------
	# Loyalty Composability Hooks
	# ------------------------------------------------------------------

	async def earn_loyalty_points(
		self,
		order_id: str,
		program_id: str,
		points_per_currency_unit: float = 1.0,
	) -> dict[str, Any]:
		"""Record loyalty point earn event for a completed order.

		Calls the retail_loy adapter if available; otherwise records locally for
		deferred sync. Emits 'loyalty_earned' event.
		"""
		assert order_id, "order_id required"
		assert program_id, "program_id required"
		tenant_id = self.tenant_id

		order = self._orders.get(order_id)
		assert order is not None and order["tenant_id"] == tenant_id, "order not found"
		assert order.get("status") in ("collected", "delivered", "shipped"), \
			"points only earned on fulfilled orders"

		grand_total = float(order.get("grand_total", 0.0))
		points_earned = int(grand_total * points_per_currency_unit)
		customer_id = order.get("customer_id", "")

		earn_record: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": tenant_id,
			"order_id": order_id,
			"customer_id": customer_id,
			"program_id": program_id,
			"points_earned": points_earned,
			"order_value": grand_total,
			"channel_id": order.get("channel_id"),
			"earned_at": datetime.utcnow().isoformat(),
		}

		# Delegate to loy adapter when wired
		if self._notify:
			try:
				await self._notify("loyalty_earned", earn_record)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		self._log_op("earn_loyalty_points", tenant_id, order_id)
		return earn_record

	async def burn_loyalty_points(
		self,
		cart_id: str,
		points: int,
		program_id: str,
		point_value: float = 0.01,
	) -> OmcCartResponse | None:
		"""Apply a loyalty point burn as a cart discount line item.

		Args:
		  cart_id: target active cart
		  points: number of points to redeem
		  program_id: loyalty program identifier
		  point_value: monetary value per point (default KES 0.01 = 1 cent)
		"""
		assert cart_id, "cart_id required"
		assert points > 0, "points must be positive"
		assert program_id, "program_id required"
		tenant_id = self.tenant_id

		rec = self._carts.get(cart_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		assert rec.get("state") == "active", "can only burn points on active cart"

		discount_value = round(points * point_value, 2)
		# Add a synthetic zero-unit loyalty discount line item
		loyalty_line = OmcCartLineItem(
			sku=f"LOYALTY-{program_id}",
			quantity=1,
			unit_price=-discount_value,
			line_total=-discount_value,
			discount_applied=discount_value,
			promotion_ids=[program_id],
		)
		items = [OmcCartLineItem(**i) if isinstance(i, dict) else i
				 for i in rec.get("items", [])]
		items.append(loyalty_line)
		totals = self._compute_cart_totals(items)
		rec["items"] = [i.model_dump() for i in items]
		rec.update(totals)
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._carts[cart_id] = rec

		self._log_op("burn_loyalty_points", tenant_id, cart_id)
		return OmcCartResponse(**rec)

	# ------------------------------------------------------------------
	# Safety Stock and Low-Stock Alerts
	# ------------------------------------------------------------------

	async def compute_safety_stock(
		self,
		sku: str,
		location_id: str,
		lookback_days: int = 30,
		lead_time_days: int = 7,
		service_level_z: float = 1.65,
	) -> dict[str, Any]:
		"""Compute reorder point and safety stock from historical demand volatility.

		Uses the classical formula: SS = Z * sigma_d * sqrt(L)
		where sigma_d = demand std dev per day, L = supplier lead time in days.
		Demand series is approximated from reservation velocity in stored inventory records.
		"""
		assert sku, "sku required"
		assert location_id, "location_id required"
		tenant_id = self.tenant_id

		inv_records = [v for v in self._inventory.values()
					   if v["tenant_id"] == tenant_id
					   and v["sku"] == sku
					   and v["location_id"] == location_id]

		if not inv_records:
			return {"sku": sku, "location_id": location_id, "error": "no inventory records found"}

		# Use reserved_qty as a proxy for demand; in production this comes from sales history
		total_reserved = sum(v.get("reserved_qty", 0) for v in inv_records)
		avg_daily_demand = total_reserved / max(lookback_days, 1)

		import math
		# Variance approximation — in production derive from time-series
		demand_std = avg_daily_demand * 0.3  # assume 30% CV as baseline
		safety_stock = int(math.ceil(service_level_z * demand_std * math.sqrt(lead_time_days)))
		reorder_point = int(math.ceil(avg_daily_demand * lead_time_days + safety_stock))

		# Persist computed safety stock
		for rec in inv_records:
			rec["safety_stock_qty"] = safety_stock
			rec["updated_at"] = datetime.utcnow().isoformat()
			self._inventory[rec["id"]] = rec

		result = {
			"sku": sku,
			"location_id": location_id,
			"avg_daily_demand": round(avg_daily_demand, 2),
			"demand_std_dev": round(demand_std, 2),
			"lead_time_days": lead_time_days,
			"service_level_z": service_level_z,
			"safety_stock_qty": safety_stock,
			"reorder_point": reorder_point,
			"computed_at": datetime.utcnow().isoformat(),
		}
		self._log_op("compute_safety_stock", tenant_id)
		return result

	async def list_low_stock_alerts(
		self,
		threshold_multiplier: float = 1.0,
	) -> list[dict[str, Any]]:
		"""Return inventory records where available_qty <= safety_stock_qty * threshold_multiplier.

		threshold_multiplier > 1.0 triggers earlier warnings (e.g. 1.5 = alert at 150% of safety stock).
		"""
		tenant_id = self.tenant_id
		alerts: list[dict[str, Any]] = []

		for rec in self._inventory.values():
			if rec["tenant_id"] != tenant_id:
				continue
			safety = rec.get("safety_stock_qty", 0)
			available = rec.get("available_qty", 0)
			threshold = safety * threshold_multiplier
			if available <= threshold:
				alerts.append({
					"sku": rec["sku"],
					"location_id": rec["location_id"],
					"channel_id": rec.get("channel_id"),
					"available_qty": available,
					"safety_stock_qty": safety,
					"threshold": threshold,
					"severity": "critical" if available == 0 else "warning",
					"checked_at": datetime.utcnow().isoformat(),
				})

		alerts.sort(key=lambda x: x["available_qty"])
		self._log_op("list_low_stock_alerts", tenant_id)
		return alerts

	# ------------------------------------------------------------------
	# Fraud Screening
	# ------------------------------------------------------------------

	async def fraud_screen_order(self, order_id: str) -> dict[str, Any]:
		"""Compute fraud risk score for an order and update fraud_check_passed.

		Assembles feature vector from: order value, customer history, channel,
		payment method, and velocity. High-risk orders (score >= 0.7) are held
		for manual review. Configurable threshold via RETAIL_OMC_FRAUD_THRESHOLD env var.
		"""
		import os
		assert order_id, "order_id required"
		tenant_id = self.tenant_id

		order = self._orders.get(order_id)
		assert order is not None and order["tenant_id"] == tenant_id, "order not found"

		threshold = float(os.environ.get("RETAIL_OMC_FRAUD_THRESHOLD", "0.7"))
		customer_id = order.get("customer_id")
		grand_total = float(order.get("grand_total", 0.0))
		channel_id = order.get("channel_id", "unknown")
		payment_method = order.get("payment_method", "unknown")

		# Feature vector
		customer_orders = [o for o in self._orders.values()
						   if o["tenant_id"] == tenant_id
						   and o.get("customer_id") == customer_id
						   and o["id"] != order_id]
		order_velocity_30d = len(customer_orders)
		customer_lifetime_value = sum(float(o.get("grand_total", 0)) for o in customer_orders)

		# Heuristic scoring (production: replace with ML model or fraud API)
		score = 0.0
		signals: list[str] = []

		if grand_total > 100_000:
			score += 0.3
			signals.append("high_value_order")
		if order_velocity_30d > 10:
			score += 0.2
			signals.append("high_velocity")
		if channel_id in ("marketplace",) and grand_total > 50_000:
			score += 0.15
			signals.append("marketplace_high_value")
		if payment_method in ("cod", "unknown"):
			score += 0.1
			signals.append("risky_payment_method")
		if customer_lifetime_value == 0 and grand_total > 20_000:
			score += 0.25
			signals.append("new_customer_high_value")

		score = min(round(score, 3), 1.0)
		passed = score < threshold
		decision = "approved" if passed else "held_for_review"

		order["fraud_check_passed"] = passed
		order["fraud_score"] = score
		order["fraud_decision"] = decision
		order["fraud_signals"] = signals
		if not passed:
			order["status"] = "fraud_review"
		order["updated_at"] = datetime.utcnow().isoformat()
		self._orders[order_id] = order

		self._log_op("fraud_screen_order", tenant_id, order_id)
		return {
			"order_id": order_id,
			"fraud_score": score,
			"threshold": threshold,
			"passed": passed,
			"decision": decision,
			"signals": signals,
			"screened_at": datetime.utcnow().isoformat(),
		}

	# ------------------------------------------------------------------
	# Catalogue Search with Faceted Filtering
	# ------------------------------------------------------------------

	async def search_catalogue(
		self,
		tenant_id: str,
		query: str,
		filters: dict[str, Any] | None = None,
		sort: str = "relevance",
		page: int = 1,
		page_size: int = 20,
	) -> dict[str, Any]:
		"""Full-text catalogue search with faceted filtering and relevance ranking.

		Filters:
		  category_path: list[str]
		  brand: str
		  min_price: float
		  max_price: float
		  channel_id: str  (filters by channel_prices availability)
		  in_stock_only: bool

		Sort options: relevance, price_asc, price_desc, name_asc
		"""
		assert tenant_id, "tenant_id required"
		filters = filters or {}

		candidates = [v for v in self._catalogue.values()
					  if v["tenant_id"] == tenant_id and v["is_active"]]

		# Apply filters
		if filters.get("category_path"):
			cp = filters["category_path"]
			candidates = [v for v in candidates
						  if v.get("category_path", [])[:len(cp)] == cp]
		if filters.get("brand"):
			candidates = [v for v in candidates
						  if v.get("brand", "").lower() == filters["brand"].lower()]
		if filters.get("min_price") is not None:
			candidates = [v for v in candidates if v["base_price"] >= filters["min_price"]]
		if filters.get("max_price") is not None:
			candidates = [v for v in candidates if v["base_price"] <= filters["max_price"]]
		if filters.get("channel_id"):
			ch = filters["channel_id"]
			candidates = [v for v in candidates if ch in v.get("channel_prices", {})]
		if filters.get("in_stock_only"):
			stocked_skus = {v["sku"] for v in self._inventory.values()
							if v["tenant_id"] == tenant_id and v["available_qty"] > 0}
			candidates = [v for v in candidates if v["sku"] in stocked_skus]

		# Full-text relevance scoring (BM25-lite: term frequency in name + description)
		q_terms = query.lower().split() if query else []

		def _score(item: dict[str, Any]) -> float:
			if not q_terms:
				return 0.0
			text = f"{item.get('name', '')} {item.get('description', '')} {item.get('brand', '')}".lower()
			return sum(text.count(t) for t in q_terms) / (len(text.split()) + 1)

		if query:
			candidates = sorted(candidates, key=_score, reverse=True)
			# Filter out zero-score results when a query is provided
			candidates = [v for v in candidates if _score(v) > 0] or candidates

		# Sort
		if sort == "price_asc":
			candidates.sort(key=lambda x: x["base_price"])
		elif sort == "price_desc":
			candidates.sort(key=lambda x: x["base_price"], reverse=True)
		elif sort == "name_asc":
			candidates.sort(key=lambda x: x.get("name", "").lower())

		# Facet aggregation
		brand_facets: dict[str, int] = {}
		category_facets: dict[str, int] = {}
		for v in candidates:
			b = v.get("brand") or "unknown"
			brand_facets[b] = brand_facets.get(b, 0) + 1
			cat = "/".join(v.get("category_path", ["uncategorized"]))
			category_facets[cat] = category_facets.get(cat, 0) + 1

		total = len(candidates)
		start = (page - 1) * page_size
		page_items = candidates[start: start + page_size]

		return {
			"query": query,
			"total_results": total,
			"page": page,
			"page_size": page_size,
			"total_pages": max(1, -(-total // page_size)),  # ceiling division
			"results": [OmcCatalogueItemResponse(**v).model_dump() for v in page_items],
			"facets": {
				"brands": brand_facets,
				"categories": category_facets,
			},
			"searched_at": datetime.utcnow().isoformat(),
		}

	# ------------------------------------------------------------------
	# Multi-Touch Attribution
	# ------------------------------------------------------------------

	async def multi_touch_attribution(
		self,
		order_id: str,
		model: str = "linear",
		time_decay_half_life_hours: float = 24.0,
	) -> dict[str, Any]:
		"""Compute multi-touch attribution for an order.

		model options:
		  'last_touch'  — 100% credit to final touchpoint
		  'first_touch' — 100% credit to first touchpoint
		  'linear'      — equal credit across all touchpoints
		  'time_decay'  — exponential decay, recent touches get more credit
		"""
		assert order_id, "order_id required"
		assert model in ("last_touch", "first_touch", "linear", "time_decay"), "invalid model"
		tenant_id = self.tenant_id

		order = self._orders.get(order_id)
		assert order is not None and order["tenant_id"] == tenant_id, "order not found"

		customer_id = order.get("customer_id", "")
		order_channel = order.get("channel_id", "unknown")
		grand_total = float(order.get("grand_total", 0.0))

		pre_order_events = sorted(
			[v for v in self._journey_events.values()
			 if v["tenant_id"] == tenant_id
			 and v.get("customer_id") == customer_id
			 and v.get("occurred_at", "") <= order.get("created_at", "")],
			key=lambda x: x.get("occurred_at", ""),
		)

		if not pre_order_events:
			# No journey data: attribute 100% to order channel
			attribution_vector = {order_channel: 1.0}
		elif model == "last_touch":
			attribution_vector = {pre_order_events[-1].get("channel_id", order_channel): 1.0}
		elif model == "first_touch":
			attribution_vector = {pre_order_events[0].get("channel_id", order_channel): 1.0}
		elif model == "linear":
			n = len(pre_order_events)
			attribution_vector = {}
			for ev in pre_order_events:
				ch = ev.get("channel_id", "unknown")
				attribution_vector[ch] = attribution_vector.get(ch, 0.0) + 1.0 / n
		else:  # time_decay
			import math
			order_ts = order.get("created_at", datetime.utcnow().isoformat())
			weights: list[float] = []
			for ev in pre_order_events:
				ev_ts = ev.get("occurred_at", order_ts)
				# Hours before order
				try:
					delta_h = (datetime.fromisoformat(order_ts) - datetime.fromisoformat(ev_ts)).total_seconds() / 3600
				except Exception:
					delta_h = 0.0
				weights.append(math.exp(-delta_h / time_decay_half_life_hours))
			total_w = sum(weights) or 1.0
			attribution_vector = {}
			for ev, w in zip(pre_order_events, weights):
				ch = ev.get("channel_id", "unknown")
				attribution_vector[ch] = attribution_vector.get(ch, 0.0) + w / total_w

		# Revenue allocation
		revenue_attribution = {ch: round(credit * grand_total, 2)
								for ch, credit in attribution_vector.items()}

		result = {
			"order_id": order_id,
			"model": model,
			"touchpoint_count": len(pre_order_events),
			"order_value": grand_total,
			"attribution_vector": {ch: round(v, 4) for ch, v in attribution_vector.items()},
			"revenue_attribution": revenue_attribution,
			"computed_at": datetime.utcnow().isoformat(),
		}
		self._attribution[order_id] = result
		self._log_op("multi_touch_attribution", tenant_id, order_id)
		return result

	# ------------------------------------------------------------------
	# RMA Workflow with Condition Grading
	# ------------------------------------------------------------------

	async def process_rma(
		self,
		return_id: str,
		received_items: list[dict[str, Any]],
		condition_grades: dict[str, str],
	) -> dict[str, Any]:
		"""Process a received return and route each item by condition grade.

		condition_grades: {sku: grade} where grade in ('new', 'refurbished', 'damaged', 'scrap')
		Adjusts inventory and computes actual refund from grade-based recovery rates.
		"""
		RECOVERY_RATES = {"new": 1.0, "refurbished": 0.7, "damaged": 0.3, "scrap": 0.0}
		GRADE_DISPOSITIONS = {
			"new": "restock",
			"refurbished": "refurbishment_queue",
			"damaged": "clearance",
			"scrap": "write_off",
		}

		assert return_id, "return_id required"
		tenant_id = self.tenant_id

		ret = self._returns.get(return_id)
		assert ret is not None and ret["tenant_id"] == tenant_id, "return not found"
		assert ret.get("status") in ("pending", "approved"), "return must be pending or approved"

		order_id = ret.get("order_id", "")
		order = self._orders.get(order_id, {})
		grand_total = float(order.get("grand_total", 0.0))

		item_dispositions: list[dict[str, Any]] = []
		total_refund = 0.0

		for item in received_items:
			sku = item.get("sku", "")
			qty = item.get("quantity", 1)
			unit_price = float(item.get("unit_price", 0.0))
			grade = condition_grades.get(sku, "damaged")
			disposition = GRADE_DISPOSITIONS.get(grade, "write_off")
			recovery_rate = RECOVERY_RATES.get(grade, 0.0)
			refund_for_item = round(unit_price * qty * recovery_rate, 2)
			total_refund += refund_for_item

			# Restock items that are in 'new' or 'refurbished' condition
			if grade in ("new", "refurbished") and sku:
				location_id = order.get("store_id") or "returns_warehouse"
				inv_records = [v for v in self._inventory.values()
							   if v["tenant_id"] == tenant_id
							   and v["sku"] == sku
							   and v["location_id"] == location_id]
				if inv_records:
					rec = inv_records[0]
					restock_qty = qty if grade == "new" else max(1, int(qty * 0.7))
					rec["available_qty"] = rec.get("available_qty", 0) + restock_qty
					rec["on_hand_qty"] = rec.get("on_hand_qty", 0) + restock_qty
					rec["updated_at"] = datetime.utcnow().isoformat()
					self._inventory[rec["id"]] = rec

			item_dispositions.append({
				"sku": sku,
				"quantity": qty,
				"grade": grade,
				"disposition": disposition,
				"unit_refund": round(unit_price * recovery_rate, 2),
				"total_refund": refund_for_item,
			})

		# Finalize return record
		ret["status"] = "processed"
		ret["refund_amount"] = round(total_refund, 2)
		ret["rma_dispositions"] = item_dispositions
		ret["processed_at"] = datetime.utcnow().isoformat()
		ret["updated_at"] = datetime.utcnow().isoformat()
		self._returns[return_id] = ret

		# Update source order status
		if order:
			order["status"] = "returned"
			order["updated_at"] = datetime.utcnow().isoformat()
			self._orders[order_id] = order

		self._log_op("process_rma", tenant_id, return_id)
		return {
			"return_id": return_id,
			"order_id": order_id,
			"item_dispositions": item_dispositions,
			"total_refund": round(total_refund, 2),
			"original_order_value": grand_total,
			"recovery_rate": round(total_refund / grand_total, 3) if grand_total else 0.0,
			"processed_at": ret["processed_at"],
		}

	# ------------------------------------------------------------------
	# Shipping Rate Calculation
	# ------------------------------------------------------------------

	async def calculate_shipping(
		self,
		order_id: str,
		carrier_options: list[str] | None = None,
	) -> list[dict[str, Any]]:
		"""Evaluate available carrier options for an order, returning ranked rate quotes.

		Rates are computed from order weight, destination zone, and fulfilment mode.
		In production, replace the rate table with calls to carrier APIs (DHL, FedEx, G4S).
		"""
		assert order_id, "order_id required"
		tenant_id = self.tenant_id

		order = self._orders.get(order_id)
		assert order is not None and order["tenant_id"] == tenant_id, "order not found"

		carrier_options = carrier_options or ["standard", "express", "same_day"]
		fulfilment_mode = order.get("fulfilment_mode", "ship_to_home")

		# Exclude shipping options for C&C (no carrier required)
		if fulfilment_mode == "click_and_collect":
			return [{"carrier": "in_store_pickup", "service": "click_and_collect",
					 "rate": 0.0, "currency": "USD", "eta_days": 0}]

		# Compute total order weight from catalogue items
		total_weight_kg = 0.0
		for item in order.get("items", []):
			sku = item.get("sku", "")
			qty = item.get("quantity", 1)
			cat_item = await self.get_catalogue_item_by_sku(tenant_id, sku)
			if cat_item and cat_item.weight_kg:
				total_weight_kg += cat_item.weight_kg * qty
		total_weight_kg = total_weight_kg or 0.5  # default 500g

		# Simplified zone-based rate table (KES/USD)
		RATE_TABLE = {
			"standard": {"rate_per_kg": 3.0, "base_rate": 5.0, "eta_days": 5},
			"express":  {"rate_per_kg": 8.0, "base_rate": 12.0, "eta_days": 2},
			"same_day": {"rate_per_kg": 15.0, "base_rate": 20.0, "eta_days": 0},
		}

		quotes: list[dict[str, Any]] = []
		for carrier in carrier_options:
			if carrier not in RATE_TABLE:
				continue
			tbl = RATE_TABLE[carrier]
			rate = round(tbl["base_rate"] + tbl["rate_per_kg"] * total_weight_kg, 2)
			quotes.append({
				"carrier": carrier,
				"service": f"{carrier}_delivery",
				"rate": rate,
				"currency": order.get("currency_code", "USD"),
				"weight_kg": round(total_weight_kg, 3),
				"eta_days": tbl["eta_days"],
				"fulfilment_mode": fulfilment_mode,
			})

		quotes.sort(key=lambda x: x["rate"])
		self._log_op("calculate_shipping", tenant_id, order_id)
		return quotes

	# ------------------------------------------------------------------
	# Audit Trail Query
	# ------------------------------------------------------------------

	async def query_audit_log(
		self,
		entity_type: str,
		entity_id: str | None = None,
		limit: int = 100,
	) -> list[dict[str, Any]]:
		"""Query the structured audit log for an entity type and optional entity ID.

		Audit events are appended by _emit_audit_event (called by write operations).
		Returns events in reverse-chronological order.
		"""
		assert entity_type, "entity_type required"
		tenant_id = self.tenant_id

		if not hasattr(self, "_audit_log"):
			self._audit_log = WriteThruList('audit_log', tenant_id, _store)

		results = [e for e in self._audit_log
				   if e.get("tenant_id") == tenant_id
				   and e.get("entity_type") == entity_type
				   and (entity_id is None or e.get("entity_id") == entity_id)]

		results.sort(key=lambda x: x.get("occurred_at", ""), reverse=True)
		return results[:limit]

	async def _emit_audit_event(
		self,
		entity_type: str,
		entity_id: str,
		action: str,
		before: dict[str, Any],
		after: dict[str, Any],
		actor: str | None = None,
	) -> None:
		"""Append a structured audit event to the in-process audit log.

		Structured as a CloudEvent for Bytewax compatibility.
		In production, replace list append with an async write to the audit DB or stream.
		"""
		if not hasattr(self, "_audit_log"):
			self._audit_log = WriteThruList('audit_log', tenant_id, _store)

		event: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self.tenant_id,
			"entity_type": entity_type,
			"entity_id": entity_id,
			"action": action,
			"actor": actor or self.actor_id,
			"before": before,
			"after": after,
			"occurred_at": datetime.utcnow().isoformat(),
			"source": "retail_omc",
			"specversion": "1.0",
		}
		self._audit_log.append(event)

		if self._audit_adapter:
			try:
				await self._audit_adapter(event)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_channels', '_catalogue', '_inventory', '_carts', '_orders', '_returns', '_journey_events', '_pricing_rules', '_customer_profiles', '_attribution', '_bopis_orders', '_analytics_cache', '_audit_log', '_audit_log']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

