"""Agri-Marketplace service — agr_mkt."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

_log = logging.getLogger(__name__)
_CAPABILITY_ID = "agr_mkt"


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_id(prefix: str = "") -> str:
	suffix = uuid4().hex[:12]
	return f"{prefix}-{suffix}" if prefix else suffix


class AgriMarketplaceService:
	"""Async service for agri-marketplace: farmer produce listing, buyer matching,
	price discovery, escrow, and auction management."""

	def __init__(self, tenant_id: str = "default") -> None:
		if not tenant_id:
			raise ValueError("tenant_id required")
		self.tenant_id = tenant_id
		self._listings: dict[str, dict[str, Any]] = {}
		self._bids: dict[str, dict[str, Any]] = {}
		self._escrows: dict[str, dict[str, Any]] = {}
		self._auctions: dict[str, dict[str, Any]] = {}
		self._auction_bids: dict[str, list[dict[str, Any]]] = {}
		self._audit: list[dict[str, Any]] = []

	def _emit(self, event_type: str, entity_type: str, entity_id: str, payload: dict[str, Any]) -> None:
		self._audit.append({
			"id": _new_id("evt"),
			"tenant_id": self.tenant_id,
			"event_type": event_type,
			"entity_type": entity_type,
			"entity_id": entity_id,
			"payload": payload,
			"occurred_at": _now(),
		})

	# ------------------------------------------------------------------ health

	async def health_check(self) -> dict[str, Any]:
		return {
			"status": "ok",
			"capability": _CAPABILITY_ID,
			"tenant_id": self.tenant_id,
			"counts": {
				"listings": len(self._listings),
				"bids": len(self._bids),
				"escrows": len(self._escrows),
				"auctions": len(self._auctions),
			},
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": _CAPABILITY_ID,
			"name": "Agri-Marketplace",
			"domain": "agriculture",
			"version": "1.0.0",
			"description": "Farmer produce listing, buyer matching, price discovery, escrow, auction management.",
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		return self._audit[-limit:]

	# ------------------------------------------------------------------ listings

	async def list_listings(self, product_type: str | None = None, status: str | None = None,
							farmer_id: str | None = None, location: str | None = None,
							limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		items = list(self._listings.values())
		if product_type:
			items = [l for l in items if l.get("product_type") == product_type]
		if status:
			items = [l for l in items if l.get("status") == status]
		if farmer_id:
			items = [l for l in items if l.get("farmer_id") == farmer_id]
		if location:
			items = [l for l in items if location.lower() in str(l.get("location", "")).lower()]
		return items[offset: offset + limit]

	async def get_listing(self, listing_id: str) -> dict[str, Any]:
		if listing_id not in self._listings:
			raise KeyError(f"listing_not_found:{listing_id}")
		self._listings[listing_id]["views_count"] = self._listings[listing_id].get("views_count", 0) + 1
		return self._listings[listing_id]

	async def create_listing(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			lid = _new_id("lst")
			ts = _now()
			record: dict[str, Any] = {
				"id": lid,
				"tenant_id": self.tenant_id,
				"farmer_id": payload["farmer_id"],
				"product_type": payload["product_type"],
				"variety": payload.get("variety"),
				"quantity_kg": float(payload["quantity_kg"]),
				"asking_price_per_kg": float(payload["asking_price_per_kg"]),
				"currency": payload.get("currency", "KES"),
				"harvest_date": payload["harvest_date"],
				"available_from": payload["available_from"],
				"available_to": payload["available_to"],
				"location": payload["location"],
				"quality_grade": payload.get("quality_grade"),
				"description": payload.get("description"),
				"images": list(payload.get("images", [])),
				"status": "draft",
				"views_count": 0,
				"bids_count": 0,
				"created_at": ts,
				"updated_at": ts,
			}
			self._listings[lid] = record
			self._emit("listing.created", "listing", lid, record)
			return record
		except Exception as exc:
			_log.error("create_listing failed: %s", exc)
			raise

	async def update_listing(self, listing_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if listing_id not in self._listings:
				raise KeyError(f"listing_not_found:{listing_id}")
			record = self._listings[listing_id]
			for field in ["asking_price_per_kg", "quantity_kg", "available_to", "status", "description"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("listing.updated", "listing", listing_id, payload)
			return record
		except Exception as exc:
			_log.error("update_listing failed: %s", exc)
			raise

	async def delete_listing(self, listing_id: str) -> dict[str, Any]:
		try:
			if listing_id not in self._listings:
				raise KeyError(f"listing_not_found:{listing_id}")
			self._listings.pop(listing_id)
			self._emit("listing.deleted", "listing", listing_id, {"id": listing_id})
			return {"deleted": True, "id": listing_id}
		except Exception as exc:
			_log.error("delete_listing failed: %s", exc)
			raise

	async def publish_listing(self, listing_id: str) -> dict[str, Any]:
		"""Move a listing from draft to active."""
		if listing_id not in self._listings:
			raise KeyError(f"listing_not_found:{listing_id}")
		self._listings[listing_id]["status"] = "active"
		self._listings[listing_id]["updated_at"] = _now()
		self._emit("listing.published", "listing", listing_id, {"id": listing_id})
		return self._listings[listing_id]

	async def match_buyers(self, listing_id: str) -> list[dict[str, Any]]:
		"""Find buyers who have placed bids on similar product types."""
		if listing_id not in self._listings:
			raise KeyError(f"listing_not_found:{listing_id}")
		listing = self._listings[listing_id]
		product_type = listing.get("product_type")
		buyers = {}
		for bid in self._bids.values():
			bid_listing = self._listings.get(bid.get("listing_id", ""))
			if bid_listing and bid_listing.get("product_type") == product_type:
				buyer_id = bid.get("buyer_id")
				if buyer_id and buyer_id not in buyers:
					buyers[buyer_id] = {
						"buyer_id": buyer_id,
						"bids_on_product": 0,
						"avg_offered_price": 0,
					}
				if buyer_id:
					buyers[buyer_id]["bids_on_product"] += 1
		return list(buyers.values())

	# ------------------------------------------------------------------ bids

	async def list_bids(self, listing_id: str | None = None, buyer_id: str | None = None,
						status: str | None = None) -> list[dict[str, Any]]:
		items = list(self._bids.values())
		if listing_id:
			items = [b for b in items if b.get("listing_id") == listing_id]
		if buyer_id:
			items = [b for b in items if b.get("buyer_id") == buyer_id]
		if status:
			items = [b for b in items if b.get("status") == status]
		return items

	async def get_bid(self, bid_id: str) -> dict[str, Any]:
		if bid_id not in self._bids:
			raise KeyError(f"bid_not_found:{bid_id}")
		return self._bids[bid_id]

	async def place_bid(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Place a bid on an active listing."""
		try:
			listing_id = payload["listing_id"]
			if listing_id not in self._listings:
				raise KeyError(f"listing_not_found:{listing_id}")
			if self._listings[listing_id].get("status") != "active":
				raise ValueError("listing_not_active")
			bid_id = _new_id("bid")
			ts = _now()
			qty = float(payload["quantity_kg"])
			price = float(payload["offered_price_per_kg"])
			record: dict[str, Any] = {
				"id": bid_id,
				"tenant_id": self.tenant_id,
				"listing_id": listing_id,
				"buyer_id": payload["buyer_id"],
				"offered_price_per_kg": price,
				"quantity_kg": qty,
				"total_value": round(price * qty, 2),
				"currency": self._listings[listing_id].get("currency", "KES"),
				"message": payload.get("message"),
				"status": "pending",
				"counter_price": None,
				"created_at": ts,
				"updated_at": ts,
			}
			self._bids[bid_id] = record
			self._listings[listing_id]["bids_count"] = self._listings[listing_id].get("bids_count", 0) + 1
			self._emit("bid.placed", "bid", bid_id, record)
			return record
		except Exception as exc:
			_log.error("place_bid failed: %s", exc)
			raise

	async def respond_to_bid(self, bid_id: str, action: str, counter_price: float | None = None) -> dict[str, Any]:
		"""Accept, reject, or counter a bid."""
		try:
			if bid_id not in self._bids:
				raise KeyError(f"bid_not_found:{bid_id}")
			if action not in ("accept", "reject", "counter"):
				raise ValueError(f"invalid_action:{action}")
			bid = self._bids[bid_id]
			if action == "accept":
				bid["status"] = "accepted"
				listing_id = bid["listing_id"]
				if listing_id in self._listings:
					self._listings[listing_id]["status"] = "matched"
			elif action == "reject":
				bid["status"] = "rejected"
			elif action == "counter":
				if counter_price is None:
					raise ValueError("counter_price required for counter action")
				bid["status"] = "countered"
				bid["counter_price"] = counter_price
			bid["updated_at"] = _now()
			self._emit(f"bid.{action}ed", "bid", bid_id, {"action": action})
			return bid
		except Exception as exc:
			_log.error("respond_to_bid failed: %s", exc)
			raise

	# ------------------------------------------------------------------ price discovery

	async def get_price_discovery(self, product_type: str, region: str | None = None) -> dict[str, Any]:
		"""Compute market prices from accepted bids."""
		bids = [b for b in self._bids.values() if b.get("status") == "accepted"]
		if region:
			listing_ids = {l["id"] for l in self._listings.values()
						if region.lower() in str(l.get("location", "")).lower()}
			bids = [b for b in bids if b.get("listing_id") in listing_ids]
		product_bids = []
		for b in bids:
			listing = self._listings.get(b.get("listing_id", ""))
			if listing and listing.get("product_type") == product_type:
				product_bids.append(b["offered_price_per_kg"])
		if not product_bids:
			return {"product_type": product_type, "region": region, "sample_size": 0, "currency": "KES"}
		product_bids_sorted = sorted(product_bids)
		n = len(product_bids_sorted)
		median = product_bids_sorted[n // 2] if n % 2 else (product_bids_sorted[n // 2 - 1] + product_bids_sorted[n // 2]) / 2
		return {
			"product_type": product_type,
			"region": region,
			"period": _now()[:7],
			"avg_price_per_kg": round(sum(product_bids) / n, 2),
			"min_price_per_kg": min(product_bids),
			"max_price_per_kg": max(product_bids),
			"median_price_per_kg": round(median, 2),
			"sample_size": n,
			"currency": "KES",
		}

	# ------------------------------------------------------------------ escrow

	async def list_escrows(self, buyer_id: str | None = None, farmer_id: str | None = None) -> list[dict[str, Any]]:
		items = list(self._escrows.values())
		if buyer_id:
			items = [e for e in items if e.get("buyer_id") == buyer_id]
		if farmer_id:
			items = [e for e in items if e.get("farmer_id") == farmer_id]
		return items

	async def create_escrow(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			esc_id = _new_id("esc")
			ts = _now()
			record: dict[str, Any] = {
				"id": esc_id,
				"tenant_id": self.tenant_id,
				"bid_id": payload["bid_id"],
				"listing_id": payload["listing_id"],
				"buyer_id": payload["buyer_id"],
				"farmer_id": payload["farmer_id"],
				"amount": float(payload["amount"]),
				"currency": payload.get("currency", "KES"),
				"status": "funded",
				"funded_at": ts,
				"released_at": None,
				"notes": payload.get("notes"),
				"created_at": ts,
			}
			self._escrows[esc_id] = record
			self._emit("escrow.funded", "escrow", esc_id, record)
			return record
		except Exception as exc:
			_log.error("create_escrow failed: %s", exc)
			raise

	async def release_escrow(self, escrow_id: str) -> dict[str, Any]:
		try:
			if escrow_id not in self._escrows:
				raise KeyError(f"escrow_not_found:{escrow_id}")
			self._escrows[escrow_id]["status"] = "released"
			self._escrows[escrow_id]["released_at"] = _now()
			self._emit("escrow.released", "escrow", escrow_id, {"id": escrow_id})
			return self._escrows[escrow_id]
		except Exception as exc:
			_log.error("release_escrow failed: %s", exc)
			raise

	async def dispute_escrow(self, escrow_id: str, reason: str) -> dict[str, Any]:
		try:
			if escrow_id not in self._escrows:
				raise KeyError(f"escrow_not_found:{escrow_id}")
			self._escrows[escrow_id]["status"] = "disputed"
			self._escrows[escrow_id]["dispute_reason"] = reason
			self._emit("escrow.disputed", "escrow", escrow_id, {"reason": reason})
			return self._escrows[escrow_id]
		except Exception as exc:
			_log.error("dispute_escrow failed: %s", exc)
			raise

	# ------------------------------------------------------------------ auctions

	async def list_auctions(self, status: str | None = None) -> list[dict[str, Any]]:
		items = list(self._auctions.values())
		if status:
			items = [a for a in items if a.get("status") == status]
		return items

	async def get_auction(self, auction_id: str) -> dict[str, Any]:
		if auction_id not in self._auctions:
			raise KeyError(f"auction_not_found:{auction_id}")
		return self._auctions[auction_id]

	async def create_auction(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			auc_id = _new_id("auc")
			ts = _now()
			record: dict[str, Any] = {
				"id": auc_id,
				"tenant_id": self.tenant_id,
				"listing_id": payload["listing_id"],
				"start_at": payload["start_at"],
				"end_at": payload["end_at"],
				"reserve_price": float(payload["reserve_price"]),
				"increment": float(payload.get("increment", 0.5)),
				"current_bid": None,
				"current_bidder": None,
				"bid_count": 0,
				"status": "scheduled",
				"winner_id": None,
				"winning_bid": None,
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._auctions[auc_id] = record
			self._auction_bids[auc_id] = []
			self._emit("auction.created", "auction", auc_id, record)
			return record
		except Exception as exc:
			_log.error("create_auction failed: %s", exc)
			raise

	async def place_auction_bid(self, auction_id: str, bidder_id: str, amount: float) -> dict[str, Any]:
		"""Place a bid in an open auction."""
		try:
			if auction_id not in self._auctions:
				raise KeyError(f"auction_not_found:{auction_id}")
			auction = self._auctions[auction_id]
			if auction.get("status") != "open":
				raise ValueError("auction_not_open")
			current = auction.get("current_bid") or auction.get("reserve_price", 0)
			min_bid = current + auction.get("increment", 0.5)
			if amount < min_bid:
				raise ValueError(f"bid_too_low:min={min_bid}")
			auction["current_bid"] = amount
			auction["current_bidder"] = bidder_id
			auction["bid_count"] = auction.get("bid_count", 0) + 1
			auction["updated_at"] = _now()
			bid_record = {"bidder_id": bidder_id, "amount": amount, "placed_at": _now()}
			self._auction_bids[auction_id].append(bid_record)
			self._emit("auction.bid_placed", "auction", auction_id, bid_record)
			return {**auction, "your_bid": amount}
		except Exception as exc:
			_log.error("place_auction_bid failed: %s", exc)
			raise

	async def close_auction(self, auction_id: str) -> dict[str, Any]:
		"""Close auction and determine winner."""
		try:
			if auction_id not in self._auctions:
				raise KeyError(f"auction_not_found:{auction_id}")
			auction = self._auctions[auction_id]
			reserve = auction.get("reserve_price", 0)
			current_bid = auction.get("current_bid")
			if current_bid and current_bid >= reserve:
				auction["status"] = "settled"
				auction["winner_id"] = auction.get("current_bidder")
				auction["winning_bid"] = current_bid
			else:
				auction["status"] = "closed"
			auction["updated_at"] = _now()
			self._emit("auction.closed", "auction", auction_id, {"status": auction["status"], "winning_bid": current_bid})
			return auction
		except Exception as exc:
			_log.error("close_auction failed: %s", exc)
			raise

	async def delete_auction(self, auction_id: str) -> dict[str, Any]:
		try:
			if auction_id not in self._auctions:
				raise KeyError(f"auction_not_found:{auction_id}")
			self._auctions.pop(auction_id)
			self._auction_bids.pop(auction_id, None)
			self._emit("auction.deleted", "auction", auction_id, {"id": auction_id})
			return {"deleted": True, "id": auction_id}
		except Exception as exc:
			_log.error("delete_auction failed: %s", exc)
			raise

	async def get_marketplace_summary(self) -> dict[str, Any]:
		"""High-level marketplace statistics."""
		active = len([l for l in self._listings.values() if l.get("status") == "active"])
		matched = len([l for l in self._listings.values() if l.get("status") == "matched"])
		total_value = sum(b["total_value"] for b in self._bids.values() if b.get("status") == "accepted")
		return {
			"total_listings": len(self._listings),
			"active_listings": active,
			"matched_listings": matched,
			"total_bids": len(self._bids),
			"active_auctions": len([a for a in self._auctions.values() if a.get("status") == "open"]),
			"funded_escrows": len([e for e in self._escrows.values() if e.get("status") == "funded"]),
			"total_accepted_value": round(total_value, 2),
		}
