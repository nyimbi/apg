"""Reservations & Channel Manager service — CRS, OTA distribution, GDS, availability sync, booking engine."""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

_log = logging.getLogger(__name__)


def _uid() -> str:
	return uuid4().hex[:12]


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _nights(check_in: str, check_out: str) -> int:
	try:
		fmt = "%Y-%m-%d"
		return max(0, (datetime.strptime(check_out, fmt) - datetime.strptime(check_in, fmt)).days)
	except Exception:
		return 0


class RSVService:
	"""Reservations & Channel Manager service."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.channels: dict[str, dict[str, Any]] = {}
		self.bookings: dict[str, dict[str, Any]] = {}
		self.availability: dict[str, dict[str, Any]] = {}  # key: tenant:room_type:date
		self.gds_connections: dict[str, dict[str, Any]] = {}
		self.sync_logs: dict[str, dict[str, Any]] = {}
		self.booking_rules: dict[str, dict[str, Any]] = {}
		self.rate_restrictions: dict[str, dict[str, Any]] = {}
		self.waitlists: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

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
			"service": "hos_rsv",
			"status": "healthy",
			"active_channels": sum(1 for c in self.channels.values() if c["is_active"]),
			"pending_bookings": sum(1 for b in self.bookings.values() if b["status"] == "pending"),
			"confirmed_bookings": sum(1 for b in self.bookings.values() if b["status"] == "confirmed"),
			"checked_at": _now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": "hos_rsv",
			"name": "Reservations & Channel Manager",
			"domain": "hospitality",
			"version": "1.0.0",
			"description": "CRS, OTA channel distribution, GDS connectivity, availability sync, booking engine",
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Channels ──────────────────────────────────────────────────────────────

	async def list_channels(self, tenant_id: str | None = None, channel_type: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		channels = [deepcopy(c) for c in self.channels.values() if c["tenant_id"] == tenant]
		if channel_type:
			channels = [c for c in channels if c["channel_type"] == channel_type]
		return channels

	async def get_channel(self, channel_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		ch = self.channels.get(channel_id)
		if not ch or ch["tenant_id"] != tenant:
			raise KeyError(f"channel_not_found:{channel_id}")
		return deepcopy(ch)

	async def create_channel(self, code: str, name: str, channel_type: str, commission_pct: float = 0.0,
	                          api_endpoint: str | None = None, credentials_ref: str | None = None,
	                          is_active: bool = True, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		for ch in self.channels.values():
			if ch["tenant_id"] == tenant and ch["code"] == code:
				raise ValueError(f"channel_code_exists:{code}")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"code": code,
			"name": name,
			"channel_type": channel_type,
			"commission_pct": commission_pct,
			"api_endpoint": api_endpoint,
			"credentials_ref": credentials_ref,
			"is_active": is_active,
			"bookings_count": 0,
			"status": "active",
			"created_at": _now(),
		}
		self.channels[record["id"]] = record
		self._emit(tenant, "channel_created", record["id"], "channel", {"code": code, "type": channel_type})
		return deepcopy(record)

	async def update_channel(self, channel_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		ch = self.channels.get(channel_id)
		if not ch or ch["tenant_id"] != tenant:
			raise KeyError(f"channel_not_found:{channel_id}")
		allowed = {"name", "commission_pct", "api_endpoint", "is_active", "credentials_ref"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				ch[k] = v
		self._emit(tenant, "channel_updated", channel_id, "channel")
		return deepcopy(ch)

	async def delete_channel(self, channel_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		ch = self.channels.get(channel_id)
		if not ch or ch["tenant_id"] != tenant:
			raise KeyError(f"channel_not_found:{channel_id}")
		ch["is_active"] = False
		ch["status"] = "deactivated"
		self._emit(tenant, "channel_deactivated", channel_id, "channel")
		return {"deactivated": True, "channel_id": channel_id}

	async def pause_channel(self, channel_id: str, reason: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Pause a channel without deactivating it."""
		tenant = self._tenant(tenant_id)
		ch = self.channels.get(channel_id)
		if not ch or ch["tenant_id"] != tenant:
			raise KeyError(f"channel_not_found:{channel_id}")
		ch["status"] = "paused"
		ch["pause_reason"] = reason
		self._emit(tenant, "channel_paused", channel_id, "channel", {"reason": reason})
		return deepcopy(ch)

	async def resume_channel(self, channel_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Resume a paused channel."""
		tenant = self._tenant(tenant_id)
		ch = self.channels.get(channel_id)
		if not ch or ch["tenant_id"] != tenant:
			raise KeyError(f"channel_not_found:{channel_id}")
		ch["status"] = "active"
		ch.pop("pause_reason", None)
		self._emit(tenant, "channel_resumed", channel_id, "channel")
		return deepcopy(ch)

	# ── Bookings ──────────────────────────────────────────────────────────────

	async def list_bookings(self, tenant_id: str | None = None, channel_id: str | None = None,
	                         status: str | None = None, date_from: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(b) for b in self.bookings.values() if b["tenant_id"] == tenant]
		if channel_id:
			items = [b for b in items if b["channel_id"] == channel_id]
		if status:
			items = [b for b in items if b["status"] == status]
		if date_from:
			items = [b for b in items if b["check_in_date"] >= date_from]
		return items

	async def get_booking(self, booking_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		booking = self.bookings.get(booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"booking_not_found:{booking_id}")
		return deepcopy(booking)

	async def create_booking(self, channel_id: str, guest_name: str, guest_email: str,
	                          room_type: str, check_in_date: str, check_out_date: str,
	                          rate: float, adults: int = 1, children: int = 0,
	                          guest_phone: str | None = None, external_booking_ref: str | None = None,
	                          special_requests: str | None = None, currency: str = "KES",
	                          tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		ch = self.channels.get(channel_id)
		if not ch or ch["tenant_id"] != tenant:
			raise KeyError(f"channel_not_found:{channel_id}")
		if not ch["is_active"]:
			raise ValueError(f"channel_not_active:{channel_id}")
		n = _nights(check_in_date, check_out_date)
		if n <= 0:
			raise ValueError("check_out_must_be_after_check_in")
		# Check availability
		avail_key = f"{tenant}:{room_type}:{check_in_date}"
		avail = self.availability.get(avail_key)
		if avail and (avail["stop_sell"] or avail["available_count"] <= 0):
			raise ValueError(f"room_not_available:{room_type}:{check_in_date}")
		total = rate * n
		commission = total * ch["commission_pct"] / 100
		net_revenue = total - commission
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"channel_id": channel_id,
			"channel_code": ch["code"],
			"external_booking_ref": external_booking_ref or f"EXT-{_uid()}",
			"guest_name": guest_name,
			"guest_email": guest_email,
			"guest_phone": guest_phone,
			"room_type": room_type,
			"check_in_date": check_in_date,
			"check_out_date": check_out_date,
			"nights": n,
			"adults": adults,
			"children": children,
			"rate": rate,
			"currency": currency,
			"total_amount": total,
			"commission": commission,
			"net_revenue": net_revenue,
			"special_requests": special_requests,
			"status": "confirmed",
			"created_at": _now(),
		}
		self.bookings[record["id"]] = record
		ch["bookings_count"] = ch.get("bookings_count", 0) + 1
		# Decrement availability
		if avail:
			avail["available_count"] = max(0, avail["available_count"] - 1)
		self._emit(tenant, "booking_created", record["id"], "booking", {"channel": ch["code"], "room_type": room_type})
		return deepcopy(record)

	async def update_booking(self, booking_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		booking = self.bookings.get(booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"booking_not_found:{booking_id}")
		if booking["status"] in {"cancelled", "checked_out"}:
			raise ValueError("cannot_modify_closed_booking")
		allowed = {"check_in_date", "check_out_date", "adults", "children", "special_requests", "status"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				booking[k] = v
		# Recalculate if dates changed
		if "check_in_date" in updates or "check_out_date" in updates:
			booking["nights"] = _nights(booking["check_in_date"], booking["check_out_date"])
			booking["total_amount"] = booking["rate"] * booking["nights"]
		self._emit(tenant, "booking_updated", booking_id, "booking")
		return deepcopy(booking)

	async def cancel_booking(self, booking_id: str, reason: str, cancelled_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		booking = self.bookings.get(booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"booking_not_found:{booking_id}")
		booking["status"] = "cancelled"
		booking["cancellation_reason"] = reason
		booking["cancelled_by"] = cancelled_by
		booking["cancelled_at"] = _now()
		self._emit(tenant, "booking_cancelled", booking_id, "booking", {"reason": reason})
		return deepcopy(booking)

	async def delete_booking(self, booking_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		return await self.cancel_booking(booking_id, "admin_delete", "admin", tenant_id)

	async def get_booking_by_external_ref(self, external_ref: str, tenant_id: str | None = None) -> dict[str, Any] | None:
		"""Lookup booking by external channel reference."""
		tenant = self._tenant(tenant_id)
		for b in self.bookings.values():
			if b["tenant_id"] == tenant and b["external_booking_ref"] == external_ref:
				return deepcopy(b)
		return None

	# ── Availability Management ───────────────────────────────────────────────

	async def set_availability(self, room_type: str, date: str, available_count: int,
	                            stop_sell: bool = False, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		key = f"{tenant}:{room_type}:{date}"
		record: dict[str, Any] = {
			"id": key,
			"tenant_id": tenant,
			"room_type": room_type,
			"date": date,
			"available_count": available_count,
			"stop_sell": stop_sell,
			"updated_at": _now(),
		}
		self.availability[key] = record
		self._emit(tenant, "availability_updated", key, "availability", {"room_type": room_type, "date": date, "count": available_count})
		return deepcopy(record)

	async def get_availability(self, room_type: str, date: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		key = f"{tenant}:{room_type}:{date}"
		return deepcopy(self.availability.get(key, {
			"id": key, "tenant_id": tenant, "room_type": room_type,
			"date": date, "available_count": 0, "stop_sell": False, "updated_at": _now(),
		}))

	async def bulk_set_availability(self, room_type: str, date_from: str, date_to: str,
	                                 available_count: int, stop_sell: bool = False,
	                                 tenant_id: str | None = None) -> dict[str, Any]:
		"""Set availability for a date range in one call."""
		tenant = self._tenant(tenant_id)
		from datetime import timedelta
		fmt = "%Y-%m-%d"
		try:
			start = datetime.strptime(date_from, fmt)
			end = datetime.strptime(date_to, fmt)
		except ValueError as exc:
			raise ValueError(f"invalid_date_format: {exc}") from exc
		updated = []
		current = start
		while current <= end:
			date_str = current.strftime(fmt)
			rec = await self.set_availability(room_type, date_str, available_count, stop_sell, tenant)
			updated.append(date_str)
			current += timedelta(days=1)
		return {"room_type": room_type, "dates_updated": len(updated), "available_count": available_count, "stop_sell": stop_sell}

	async def stop_sell(self, room_type: str, date: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Apply stop-sell restriction for a room type on a date."""
		return await self.set_availability(room_type, date, 0, stop_sell=True, tenant_id=tenant_id)

	async def lift_stop_sell(self, room_type: str, date: str, available_count: int, tenant_id: str | None = None) -> dict[str, Any]:
		"""Lift stop-sell restriction."""
		return await self.set_availability(room_type, date, available_count, stop_sell=False, tenant_id=tenant_id)

	# ── GDS Connections ───────────────────────────────────────────────────────

	async def create_gds_connection(self, gds_provider: str, property_code: str, credentials_ref: str,
	                                 chain_code: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		valid_gds = {"amadeus", "sabre", "travelport", "galileo", "worldspan"}
		if gds_provider not in valid_gds:
			raise ValueError(f"unsupported_gds:{gds_provider}")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"gds_provider": gds_provider,
			"property_code": property_code,
			"chain_code": chain_code,
			"credentials_ref": credentials_ref,
			"status": "active",
			"last_sync_at": None,
			"created_at": _now(),
		}
		self.gds_connections[record["id"]] = record
		self._emit(tenant, "gds_connection_created", record["id"], "gds_connection", {"provider": gds_provider})
		return deepcopy(record)

	async def list_gds_connections(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(c) for c in self.gds_connections.values() if c["tenant_id"] == tenant]

	async def sync_gds_availability(self, connection_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Trigger a GDS availability sync for a connection."""
		tenant = self._tenant(tenant_id)
		conn = self.gds_connections.get(connection_id)
		if not conn or conn["tenant_id"] != tenant:
			raise KeyError(f"gds_connection_not_found:{connection_id}")
		conn["last_sync_at"] = _now()
		sync_record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"connection_id": connection_id,
			"gds_provider": conn["gds_provider"],
			"sync_type": "availability",
			"status": "completed",
			"records_synced": len(self.availability),
			"synced_at": _now(),
		}
		self.sync_logs[sync_record["id"]] = sync_record
		self._emit(tenant, "gds_sync_completed", sync_record["id"], "sync_log")
		return deepcopy(sync_record)

	# ── Rate Restrictions ─────────────────────────────────────────────────────

	async def set_rate_restriction(self, room_type: str, date_from: str, date_to: str,
	                                restriction_type: str, value: Any,
	                                tenant_id: str | None = None) -> dict[str, Any]:
		"""Set booking restrictions (min_stay, max_stay, closed_to_arrival, etc.)."""
		tenant = self._tenant(tenant_id)
		valid_restrictions = {"min_stay", "max_stay", "closed_to_arrival", "closed_to_departure", "must_stay"}
		if restriction_type not in valid_restrictions:
			raise ValueError(f"invalid_restriction_type:{restriction_type}")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"room_type": room_type,
			"date_from": date_from,
			"date_to": date_to,
			"restriction_type": restriction_type,
			"value": value,
			"is_active": True,
			"created_at": _now(),
		}
		self.rate_restrictions[record["id"]] = record
		self._emit(tenant, "rate_restriction_set", record["id"], "rate_restriction")
		return deepcopy(record)

	async def list_rate_restrictions(self, tenant_id: str | None = None, room_type: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.rate_restrictions.values() if r["tenant_id"] == tenant and r["is_active"]]
		if room_type:
			items = [r for r in items if r["room_type"] == room_type]
		return items

	# ── Waitlist ──────────────────────────────────────────────────────────────

	async def add_to_waitlist(self, guest_name: str, guest_email: str, room_type: str,
	                           check_in_date: str, check_out_date: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"guest_name": guest_name,
			"guest_email": guest_email,
			"room_type": room_type,
			"check_in_date": check_in_date,
			"check_out_date": check_out_date,
			"position": len([w for w in self.waitlists.values() if w["tenant_id"] == tenant and w["status"] == "waiting"]) + 1,
			"status": "waiting",
			"created_at": _now(),
		}
		self.waitlists[record["id"]] = record
		self._emit(tenant, "waitlist_entry_added", record["id"], "waitlist")
		return deepcopy(record)

	async def list_waitlist(self, tenant_id: str | None = None, room_type: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(w) for w in self.waitlists.values() if w["tenant_id"] == tenant and w["status"] == "waiting"]
		if room_type:
			items = [w for w in items if w["room_type"] == room_type]
		return sorted(items, key=lambda x: x["position"])

	# ── Analytics ─────────────────────────────────────────────────────────────

	async def channel_performance(self, tenant_id: str | None = None, date_from: str | None = None) -> dict[str, Any]:
		"""Compute booking volume and revenue by channel."""
		tenant = self._tenant(tenant_id)
		bookings = [b for b in self.bookings.values() if b["tenant_id"] == tenant and b["status"] != "cancelled"]
		if date_from:
			bookings = [b for b in bookings if b["check_in_date"] >= date_from]
		by_channel: dict[str, dict[str, Any]] = {}
		for b in bookings:
			code = b.get("channel_code", b["channel_id"])
			if code not in by_channel:
				by_channel[code] = {"bookings": 0, "total_revenue": 0.0, "net_revenue": 0.0, "commission": 0.0}
			by_channel[code]["bookings"] += 1
			by_channel[code]["total_revenue"] += b["total_amount"]
			by_channel[code]["net_revenue"] += b["net_revenue"]
			by_channel[code]["commission"] += b["commission"]
		return {
			"tenant_id": tenant,
			"total_bookings": len(bookings),
			"total_revenue": sum(b["total_amount"] for b in bookings),
			"by_channel": by_channel,
			"generated_at": _now(),
		}

	async def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return {
			"tenant_id": tenant,
			"active_channels": sum(1 for c in self.channels.values() if c["tenant_id"] == tenant and c["is_active"]),
			"total_bookings": sum(1 for b in self.bookings.values() if b["tenant_id"] == tenant),
			"confirmed_bookings": sum(1 for b in self.bookings.values() if b["tenant_id"] == tenant and b["status"] == "confirmed"),
			"cancelled_bookings": sum(1 for b in self.bookings.values() if b["tenant_id"] == tenant and b["status"] == "cancelled"),
			"gds_connections": len([c for c in self.gds_connections.values() if c["tenant_id"] == tenant]),
			"waitlist_count": len([w for w in self.waitlists.values() if w["tenant_id"] == tenant and w["status"] == "waiting"]),
			"generated_at": _now(),
		}
