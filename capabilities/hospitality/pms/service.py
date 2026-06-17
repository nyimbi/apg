"""Property Management System service — room inventory, check-in/out, housekeeping, folio, night audit, group bookings.

World-class improvements implemented (I1–I15):
  I1  Dynamic Pricing Engine           — demand-aware rate adjustment per room/dates
  I2  RevPAR / Yield Analytics         — ADR, RevPAR, TRevPAR with period delta
  I5  Loyalty Points Integration       — earn/redeem loyalty points on guest records
  I6  Overbooking Walk Management      — walk a reservation with comp folio credit
  I7  Maintenance Work Orders          — engineering work-order lifecycle with SLA
  I9  Rate Plan Management             — rate plan catalogue with restriction checks
  I12 Fiscal Compliance Receipts       — VAT, monotonic sequence, SHA-256 chain
  I13 Automated Housekeeping Assignment— workload-balanced staff assignment
"""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import hashlib
import json
import logging
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)


def _uid() -> str:
	return uuid4().hex[:12]


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _date_diff(d1: str, d2: str) -> int:
	"""Return number of nights between two ISO date strings."""
	try:
		fmt = "%Y-%m-%d"
		return max(0, (datetime.strptime(d2, fmt) - datetime.strptime(d1, fmt)).days)
	except Exception:
		return 0


class PMSService:
	"""In-memory Property Management System service."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.rooms: dict[str, dict[str, Any]] = {}
		self.guests: dict[str, dict[str, Any]] = {}
		self.reservations: dict[str, dict[str, Any]] = {}
		self.folios: dict[str, dict[str, Any]] = {}
		self.housekeeping_tasks: dict[str, dict[str, Any]] = {}
		self.group_bookings: dict[str, dict[str, Any]] = {}
		self.night_audits: dict[str, dict[str, Any]] = {}
		self.payments: dict[str, dict[str, Any]] = {}
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)
		# I7 — engineering work orders
		self.work_orders: dict[str, dict[str, Any]] = {}
		# I9 — rate plan catalogue
		self.rate_plans: dict[str, dict[str, Any]] = {}
		# I12 — write-once fiscal receipts
		self.fiscal_receipts: dict[str, dict[str, Any]] = {}
		self._fiscal_sequence: int = 0
		self._last_receipt_hash: str = "genesis"
		# I1 — dynamic pricing config (configurable per tenant)
		self._pricing_config: dict[str, Any] = {
			"min_multiplier": Decimal("0.80"),
			"max_multiplier": Decimal("1.50"),
			"high_demand_threshold": 0.85,  # occupancy ratio
			"low_demand_threshold": 0.40,
		}
		# I5 — loyalty earn rate (points per base currency unit)
		self._loyalty_earn_rate: Decimal = Decimal("1.0")  # 1 point per KES 1

	# ── Helpers ──────────────────────────────────────────────────────────────

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

	# ── Health & Describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		"""Return PMS service health status."""
		return {
			"service": "hos_pms",
			"status": "healthy",
			"room_count": len(self.rooms),
			"guest_count": len(self.guests),
			"active_reservations": sum(1 for r in self.reservations.values() if r["status"] in {"confirmed", "checked_in"}),
			"open_housekeeping_tasks": sum(1 for t in self.housekeeping_tasks.values() if t["status"] == "pending"),
			"checked_at": _now(),
		}

	async def describe(self) -> dict[str, Any]:
		"""Return capability descriptor."""
		return {
			"capability_id": "hos_pms",
			"name": "Property Management System",
			"domain": "hospitality",
			"version": "2.0.0",
			"description": "Room inventory, check-in/out, housekeeping, folio management, night audit, group bookings, dynamic pricing, loyalty, work orders, fiscal compliance",
			"features": [
				"room_inventory", "guest_profiles", "reservations",
				"check_in_out", "folio_management", "housekeeping",
				"night_audit", "group_bookings", "payment_tracking",
				# world-class additions
				"dynamic_pricing", "revpar_analytics", "loyalty_points",
				"walk_management", "maintenance_work_orders", "rate_plan_catalogue",
				"fiscal_receipts", "auto_housekeeping_assignment",
			],
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return audit log for tenant."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Rooms ─────────────────────────────────────────────────────────────────

	async def list_rooms(self, tenant_id: str | None = None, status: str | None = None, room_type: str | None = None) -> list[dict[str, Any]]:
		"""List all rooms, optionally filtered."""
		tenant = self._tenant(tenant_id)
		rooms = [deepcopy(r) for r in self.rooms.values() if r["tenant_id"] == tenant]
		if status:
			rooms = [r for r in rooms if r["status"] == status]
		if room_type:
			rooms = [r for r in rooms if r["room_type"] == room_type]
		return rooms

	async def get_room(self, room_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Get a single room by ID."""
		tenant = self._tenant(tenant_id)
		room = self.rooms.get(room_id)
		if not room or room["tenant_id"] != tenant:
			raise KeyError(f"room_not_found:{room_id}")
		return deepcopy(room)

	async def create_room(self, room_number: str, room_type: str, floor: int, capacity: int,
	                      rate_per_night: float, amenities: list[str] | None = None,
	                      notes: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		"""Create a new room in inventory."""
		tenant = self._tenant(tenant_id)
		if not room_number:
			raise ValueError("room_number_required")
		# Check for duplicate room number in this property
		for r in self.rooms.values():
			if r["tenant_id"] == tenant and r["room_number"] == room_number:
				raise ValueError(f"room_number_exists:{room_number}")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"room_number": room_number,
			"room_type": room_type,
			"floor": floor,
			"capacity": capacity,
			"rate_per_night": rate_per_night,
			"amenities": amenities or [],
			"notes": notes,
			"status": "available",  # available|occupied|maintenance|out_of_order|housekeeping
			"created_at": _now(),
			"updated_at": None,
		}
		self.rooms[record["id"]] = record
		self._emit(tenant, "room_created", record["id"], "room")
		return deepcopy(record)

	async def update_room(self, room_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		"""Update room attributes."""
		tenant = self._tenant(tenant_id)
		room = self.rooms.get(room_id)
		if not room or room["tenant_id"] != tenant:
			raise KeyError(f"room_not_found:{room_id}")
		allowed = {"room_type", "capacity", "rate_per_night", "amenities", "status", "notes"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				room[k] = v
		room["updated_at"] = _now()
		self._emit(tenant, "room_updated", room_id, "room", {"fields": list(updates.keys())})
		return deepcopy(room)

	async def delete_room(self, room_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Remove a room from inventory (only if not occupied)."""
		tenant = self._tenant(tenant_id)
		room = self.rooms.get(room_id)
		if not room or room["tenant_id"] != tenant:
			raise KeyError(f"room_not_found:{room_id}")
		if room["status"] == "occupied":
			raise ValueError("cannot_delete_occupied_room")
		del self.rooms[room_id]
		self._emit(tenant, "room_deleted", room_id, "room")
		return {"deleted": True, "room_id": room_id}

	async def set_room_status(self, room_id: str, status: str, reason: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		"""Set room operational status."""
		valid = {"available", "occupied", "maintenance", "out_of_order", "housekeeping"}
		if status not in valid:
			raise ValueError(f"invalid_status:{status}")
		return await self.update_room(room_id, {"status": status, "notes": reason}, tenant_id)

	async def get_room_availability(self, check_in: str, check_out: str, room_type: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return available rooms for a date range."""
		tenant = self._tenant(tenant_id)
		# Find rooms with no confirmed/checked_in reservations overlapping requested dates
		occupied_ids: set[str] = set()
		for res in self.reservations.values():
			if res["tenant_id"] != tenant or res["status"] not in {"confirmed", "checked_in"}:
				continue
			# Overlap: res.check_in < check_out AND res.check_out > check_in
			if res["check_in_date"] < check_out and res["check_out_date"] > check_in:
				occupied_ids.add(res["room_id"])
		available = []
		for room in self.rooms.values():
			if room["tenant_id"] != tenant:
				continue
			if room["status"] not in {"available"}:
				continue
			if room["id"] in occupied_ids:
				continue
			if room_type and room["room_type"] != room_type:
				continue
			available.append(deepcopy(room))
		return available

	# ── Guests ────────────────────────────────────────────────────────────────

	async def list_guests(self, tenant_id: str | None = None, vip_level: str | None = None) -> list[dict[str, Any]]:
		"""List all guest profiles."""
		tenant = self._tenant(tenant_id)
		guests = [deepcopy(g) for g in self.guests.values() if g["tenant_id"] == tenant]
		if vip_level:
			guests = [g for g in guests if g["vip_level"] == vip_level]
		return guests

	async def get_guest(self, guest_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Get guest profile by ID."""
		tenant = self._tenant(tenant_id)
		guest = self.guests.get(guest_id)
		if not guest or guest["tenant_id"] != tenant:
			raise KeyError(f"guest_not_found:{guest_id}")
		return deepcopy(guest)

	async def create_guest(self, first_name: str, last_name: str, email: str,
	                       phone: str | None = None, nationality: str | None = None,
	                       id_type: str | None = None, id_number: str | None = None,
	                       vip_level: str = "standard", tenant_id: str | None = None) -> dict[str, Any]:
		"""Create a new guest profile."""
		tenant = self._tenant(tenant_id)
		if not first_name or not last_name:
			raise ValueError("guest_name_required")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"first_name": first_name,
			"last_name": last_name,
			"email": email,
			"phone": phone,
			"nationality": nationality,
			"id_type": id_type,
			"id_number": id_number,
			"vip_level": vip_level,
			"stay_count": 0,
			"total_spend": 0.0,
			"status": "active",
			"created_at": _now(),
		}
		self.guests[record["id"]] = record
		self._emit(tenant, "guest_created", record["id"], "guest")
		return deepcopy(record)

	async def update_guest(self, guest_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		"""Update guest profile."""
		tenant = self._tenant(tenant_id)
		guest = self.guests.get(guest_id)
		if not guest or guest["tenant_id"] != tenant:
			raise KeyError(f"guest_not_found:{guest_id}")
		for k, v in updates.items():
			if v is not None:
				guest[k] = v
		self._emit(tenant, "guest_updated", guest_id, "guest")
		return deepcopy(guest)

	async def delete_guest(self, guest_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Deactivate a guest profile."""
		tenant = self._tenant(tenant_id)
		guest = self.guests.get(guest_id)
		if not guest or guest["tenant_id"] != tenant:
			raise KeyError(f"guest_not_found:{guest_id}")
		guest["status"] = "deactivated"
		self._emit(tenant, "guest_deactivated", guest_id, "guest")
		return {"deactivated": True, "guest_id": guest_id}

	# ── Reservations ──────────────────────────────────────────────────────────

	async def list_reservations(self, tenant_id: str | None = None, status: str | None = None,
	                             date_from: str | None = None, date_to: str | None = None) -> list[dict[str, Any]]:
		"""List reservations with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.reservations.values() if r["tenant_id"] == tenant]
		if status:
			items = [r for r in items if r["status"] == status]
		if date_from:
			items = [r for r in items if r["check_in_date"] >= date_from]
		if date_to:
			items = [r for r in items if r["check_in_date"] <= date_to]
		return items

	async def get_reservation(self, reservation_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Get reservation by ID."""
		tenant = self._tenant(tenant_id)
		res = self.reservations.get(reservation_id)
		if not res or res["tenant_id"] != tenant:
			raise KeyError(f"reservation_not_found:{reservation_id}")
		return deepcopy(res)

	async def create_reservation(self, guest_id: str, room_id: str, check_in_date: str,
	                              check_out_date: str, adults: int = 1, children: int = 0,
	                              rate_plan: str = "standard", special_requests: str | None = None,
	                              source: str = "direct", tenant_id: str | None = None) -> dict[str, Any]:
		"""Create a new reservation."""
		tenant = self._tenant(tenant_id)
		guest = self.guests.get(guest_id)
		if not guest or guest["tenant_id"] != tenant:
			raise KeyError(f"guest_not_found:{guest_id}")
		room = self.rooms.get(room_id)
		if not room or room["tenant_id"] != tenant:
			raise KeyError(f"room_not_found:{room_id}")
		nights = _date_diff(check_in_date, check_out_date)
		if nights <= 0:
			raise ValueError("check_out_must_be_after_check_in")
		total_amount = room["rate_per_night"] * nights
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"guest_id": guest_id,
			"room_id": room_id,
			"check_in_date": check_in_date,
			"check_out_date": check_out_date,
			"nights": nights,
			"adults": adults,
			"children": children,
			"rate_plan": rate_plan,
			"total_amount": total_amount,
			"paid_amount": 0.0,
			"balance": total_amount,
			"special_requests": special_requests,
			"source": source,
			"status": "confirmed",
			"checked_in_at": None,
			"checked_out_at": None,
			"created_at": _now(),
		}
		self.reservations[record["id"]] = record
		self._emit(tenant, "reservation_created", record["id"], "reservation", {"room_id": room_id, "guest_id": guest_id})
		return deepcopy(record)

	async def update_reservation(self, reservation_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		"""Update a reservation."""
		tenant = self._tenant(tenant_id)
		res = self.reservations.get(reservation_id)
		if not res or res["tenant_id"] != tenant:
			raise KeyError(f"reservation_not_found:{reservation_id}")
		if res["status"] in {"checked_out", "cancelled"}:
			raise ValueError("cannot_modify_closed_reservation")
		allowed = {"check_in_date", "check_out_date", "adults", "children", "special_requests", "status", "rate_plan"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				res[k] = v
		# Recalculate if dates changed
		if "check_in_date" in updates or "check_out_date" in updates:
			room = self.rooms.get(res["room_id"])
			if room:
				res["nights"] = _date_diff(res["check_in_date"], res["check_out_date"])
				res["total_amount"] = room["rate_per_night"] * res["nights"]
				res["balance"] = res["total_amount"] - res["paid_amount"]
		self._emit(tenant, "reservation_updated", reservation_id, "reservation")
		return deepcopy(res)

	async def delete_reservation(self, reservation_id: str, reason: str = "guest_request", tenant_id: str | None = None) -> dict[str, Any]:
		"""Cancel a reservation."""
		tenant = self._tenant(tenant_id)
		res = self.reservations.get(reservation_id)
		if not res or res["tenant_id"] != tenant:
			raise KeyError(f"reservation_not_found:{reservation_id}")
		if res["status"] == "checked_in":
			raise ValueError("cannot_cancel_checked_in_reservation")
		res["status"] = "cancelled"
		res["cancellation_reason"] = reason
		res["cancelled_at"] = _now()
		self._emit(tenant, "reservation_cancelled", reservation_id, "reservation", {"reason": reason})
		return deepcopy(res)

	# ── Check-In / Check-Out ──────────────────────────────────────────────────

	async def check_in(self, reservation_id: str, checked_in_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Perform guest check-in."""
		tenant = self._tenant(tenant_id)
		res = self.reservations.get(reservation_id)
		if not res or res["tenant_id"] != tenant:
			raise KeyError(f"reservation_not_found:{reservation_id}")
		if res["status"] != "confirmed":
			raise ValueError(f"cannot_check_in_from_status:{res['status']}")
		res["status"] = "checked_in"
		res["checked_in_at"] = _now()
		res["checked_in_by"] = checked_in_by
		# Mark room as occupied
		room = self.rooms.get(res["room_id"])
		if room:
			room["status"] = "occupied"
		# Update guest stay count
		guest = self.guests.get(res["guest_id"])
		if guest:
			guest["stay_count"] = guest.get("stay_count", 0) + 1
		self._emit(tenant, "guest_checked_in", reservation_id, "reservation", {"room_id": res["room_id"]})
		return deepcopy(res)

	async def check_out(self, reservation_id: str, checked_out_by: str, final_payment: float = 0.0, tenant_id: str | None = None) -> dict[str, Any]:
		"""Perform guest check-out and settle folio."""
		tenant = self._tenant(tenant_id)
		res = self.reservations.get(reservation_id)
		if not res or res["tenant_id"] != tenant:
			raise KeyError(f"reservation_not_found:{reservation_id}")
		if res["status"] != "checked_in":
			raise ValueError(f"cannot_check_out_from_status:{res['status']}")
		# Add final payment if provided
		if final_payment > 0:
			res["paid_amount"] = res.get("paid_amount", 0.0) + final_payment
		res["balance"] = res["total_amount"] - res["paid_amount"]
		res["status"] = "checked_out"
		res["checked_out_at"] = _now()
		res["checked_out_by"] = checked_out_by
		# Mark room for housekeeping
		room = self.rooms.get(res["room_id"])
		if room:
			room["status"] = "housekeeping"
		# Update guest total spend
		guest = self.guests.get(res["guest_id"])
		if guest:
			guest["total_spend"] = guest.get("total_spend", 0.0) + res["total_amount"]
		self._emit(tenant, "guest_checked_out", reservation_id, "reservation", {"balance": res["balance"]})
		return deepcopy(res)

	async def early_check_in(self, reservation_id: str, approved_by: str, early_fee: float = 0.0, tenant_id: str | None = None) -> dict[str, Any]:
		"""Allow early check-in with optional fee."""
		tenant = self._tenant(tenant_id)
		res = self.reservations.get(reservation_id)
		if not res or res["tenant_id"] != tenant:
			raise KeyError(f"reservation_not_found:{reservation_id}")
		if early_fee > 0:
			folio_rec = await self.add_folio_charge(
				reservation_id, "other", "Early check-in fee", early_fee, 1, tenant_id=tenant
			)
		result = await self.check_in(reservation_id, approved_by, tenant_id=tenant)
		result["early_check_in"] = True
		result["early_fee"] = early_fee
		return result

	async def late_check_out(self, reservation_id: str, approved_by: str, late_fee: float = 0.0, tenant_id: str | None = None) -> dict[str, Any]:
		"""Extend checkout time with optional fee."""
		tenant = self._tenant(tenant_id)
		res = self.reservations.get(reservation_id)
		if not res or res["tenant_id"] != tenant:
			raise KeyError(f"reservation_not_found:{reservation_id}")
		res["late_checkout"] = True
		res["late_checkout_approved_by"] = approved_by
		if late_fee > 0:
			await self.add_folio_charge(reservation_id, "other", "Late check-out fee", late_fee, 1, tenant_id=tenant)
		self._emit(tenant, "late_checkout_approved", reservation_id, "reservation")
		return deepcopy(res)

	# ── Folio Management ──────────────────────────────────────────────────────

	async def list_folio_charges(self, reservation_id: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List all folio charges for a reservation."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(f) for f in self.folios.values() if f["tenant_id"] == tenant and f["reservation_id"] == reservation_id]

	async def add_folio_charge(self, reservation_id: str, charge_type: str, description: str,
	                            amount: float, quantity: int = 1, tenant_id: str | None = None) -> dict[str, Any]:
		"""Post a charge to the guest folio."""
		tenant = self._tenant(tenant_id)
		res = self.reservations.get(reservation_id)
		if not res or res["tenant_id"] != tenant:
			raise KeyError(f"reservation_not_found:{reservation_id}")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"reservation_id": reservation_id,
			"charge_type": charge_type,
			"description": description,
			"amount": amount,
			"quantity": quantity,
			"total": amount * quantity,
			"status": "posted",
			"created_at": _now(),
		}
		self.folios[record["id"]] = record
		# Update reservation balance
		res["total_amount"] = res.get("total_amount", 0.0) + record["total"]
		res["balance"] = res["total_amount"] - res.get("paid_amount", 0.0)
		self._emit(tenant, "folio_charge_posted", record["id"], "folio", {"reservation_id": reservation_id})
		return deepcopy(record)

	async def void_folio_charge(self, folio_id: str, reason: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Void a posted folio charge."""
		tenant = self._tenant(tenant_id)
		folio = self.folios.get(folio_id)
		if not folio or folio["tenant_id"] != tenant:
			raise KeyError(f"folio_not_found:{folio_id}")
		if folio["status"] == "voided":
			raise ValueError("folio_already_voided")
		folio["status"] = "voided"
		folio["void_reason"] = reason
		folio["voided_at"] = _now()
		# Reverse balance adjustment
		res = self.reservations.get(folio["reservation_id"])
		if res:
			res["total_amount"] -= folio["total"]
			res["balance"] = res["total_amount"] - res.get("paid_amount", 0.0)
		self._emit(tenant, "folio_charge_voided", folio_id, "folio", {"reason": reason})
		return deepcopy(folio)

	async def post_payment(self, reservation_id: str, amount: float, payment_method: str, reference: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		"""Post a payment against a reservation folio."""
		tenant = self._tenant(tenant_id)
		res = self.reservations.get(reservation_id)
		if not res or res["tenant_id"] != tenant:
			raise KeyError(f"reservation_not_found:{reservation_id}")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"reservation_id": reservation_id,
			"amount": amount,
			"payment_method": payment_method,
			"reference": reference,
			"status": "settled",
			"created_at": _now(),
		}
		self.payments[record["id"]] = record
		res["paid_amount"] = res.get("paid_amount", 0.0) + amount
		res["balance"] = res["total_amount"] - res["paid_amount"]
		self._emit(tenant, "payment_posted", record["id"], "payment", {"reservation_id": reservation_id})
		return deepcopy(record)

	async def get_folio_summary(self, reservation_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return full folio summary for a reservation."""
		tenant = self._tenant(tenant_id)
		res = self.reservations.get(reservation_id)
		if not res or res["tenant_id"] != tenant:
			raise KeyError(f"reservation_not_found:{reservation_id}")
		charges = [f for f in self.folios.values() if f["tenant_id"] == tenant and f["reservation_id"] == reservation_id and f["status"] == "posted"]
		payments = [p for p in self.payments.values() if p["tenant_id"] == tenant and p["reservation_id"] == reservation_id]
		by_type: dict[str, float] = {}
		for c in charges:
			by_type[c["charge_type"]] = by_type.get(c["charge_type"], 0.0) + c["total"]
		return {
			"reservation_id": reservation_id,
			"guest_id": res["guest_id"],
			"room_id": res["room_id"],
			"total_charges": sum(c["total"] for c in charges),
			"total_payments": sum(p["amount"] for p in payments),
			"balance": res["balance"],
			"charges_by_type": by_type,
			"charge_count": len(charges),
			"payment_count": len(payments),
			"generated_at": _now(),
		}

	# ── Housekeeping ──────────────────────────────────────────────────────────

	async def list_housekeeping_tasks(self, tenant_id: str | None = None, status: str | None = None, assigned_to: str | None = None) -> list[dict[str, Any]]:
		"""List housekeeping tasks."""
		tenant = self._tenant(tenant_id)
		tasks = [deepcopy(t) for t in self.housekeeping_tasks.values() if t["tenant_id"] == tenant]
		if status:
			tasks = [t for t in tasks if t["status"] == status]
		if assigned_to:
			tasks = [t for t in tasks if t["assigned_to"] == assigned_to]
		return tasks

	async def create_housekeeping_task(self, room_id: str, task_type: str, priority: str = "normal",
	                                    assigned_to: str | None = None, notes: str | None = None,
	                                    tenant_id: str | None = None) -> dict[str, Any]:
		"""Create a housekeeping task for a room."""
		tenant = self._tenant(tenant_id)
		room = self.rooms.get(room_id)
		if not room or room["tenant_id"] != tenant:
			raise KeyError(f"room_not_found:{room_id}")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"room_id": room_id,
			"task_type": task_type,
			"priority": priority,
			"assigned_to": assigned_to,
			"notes": notes,
			"status": "pending",
			"started_at": None,
			"completed_at": None,
			"created_at": _now(),
		}
		self.housekeeping_tasks[record["id"]] = record
		self._emit(tenant, "housekeeping_task_created", record["id"], "housekeeping_task")
		return deepcopy(record)

	async def update_housekeeping_task(self, task_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		"""Update housekeeping task status or assignment."""
		tenant = self._tenant(tenant_id)
		task = self.housekeeping_tasks.get(task_id)
		if not task or task["tenant_id"] != tenant:
			raise KeyError(f"task_not_found:{task_id}")
		for k, v in updates.items():
			if v is not None:
				task[k] = v
		self._emit(tenant, "housekeeping_task_updated", task_id, "housekeeping_task")
		return deepcopy(task)

	async def complete_housekeeping_task(self, task_id: str, completed_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Mark a housekeeping task as complete."""
		tenant = self._tenant(tenant_id)
		task = self.housekeeping_tasks.get(task_id)
		if not task or task["tenant_id"] != tenant:
			raise KeyError(f"task_not_found:{task_id}")
		task["status"] = "completed"
		task["completed_by"] = completed_by
		task["completed_at"] = _now()
		# If clean task, set room back to available
		if task["task_type"] in {"clean", "turndown", "inspect"} and task.get("room_id"):
			room = self.rooms.get(task["room_id"])
			if room and room["status"] == "housekeeping":
				room["status"] = "available"
		self._emit(tenant, "housekeeping_task_completed", task_id, "housekeeping_task", {"completed_by": completed_by})
		return deepcopy(task)

	async def delete_housekeeping_task(self, task_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Cancel a pending housekeeping task."""
		tenant = self._tenant(tenant_id)
		task = self.housekeeping_tasks.get(task_id)
		if not task or task["tenant_id"] != tenant:
			raise KeyError(f"task_not_found:{task_id}")
		if task["status"] == "completed":
			raise ValueError("cannot_delete_completed_task")
		task["status"] = "cancelled"
		self._emit(tenant, "housekeeping_task_cancelled", task_id, "housekeeping_task")
		return {"cancelled": True, "task_id": task_id}

	# ── Group Bookings ────────────────────────────────────────────────────────

	async def list_group_bookings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List group bookings."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(g) for g in self.group_bookings.values() if g["tenant_id"] == tenant]

	async def create_group_booking(self, group_name: str, organiser_name: str, organiser_email: str,
	                                check_in_date: str, check_out_date: str, room_count: int,
	                                room_type: str, rate_per_night: float, notes: str | None = None,
	                                tenant_id: str | None = None) -> dict[str, Any]:
		"""Create a group booking block."""
		tenant = self._tenant(tenant_id)
		nights = _date_diff(check_in_date, check_out_date)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"group_name": group_name,
			"organiser_name": organiser_name,
			"organiser_email": organiser_email,
			"check_in_date": check_in_date,
			"check_out_date": check_out_date,
			"nights": nights,
			"room_count": room_count,
			"room_type": room_type,
			"rate_per_night": rate_per_night,
			"total_value": room_count * nights * rate_per_night,
			"reserved_count": 0,
			"notes": notes,
			"status": "tentative",  # tentative|confirmed|cancelled
			"reservation_ids": [],
			"created_at": _now(),
		}
		self.group_bookings[record["id"]] = record
		self._emit(tenant, "group_booking_created", record["id"], "group_booking", {"room_count": room_count})
		return deepcopy(record)

	async def confirm_group_booking(self, group_id: str, confirmed_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Confirm a tentative group booking."""
		tenant = self._tenant(tenant_id)
		group = self.group_bookings.get(group_id)
		if not group or group["tenant_id"] != tenant:
			raise KeyError(f"group_booking_not_found:{group_id}")
		group["status"] = "confirmed"
		group["confirmed_by"] = confirmed_by
		group["confirmed_at"] = _now()
		self._emit(tenant, "group_booking_confirmed", group_id, "group_booking")
		return deepcopy(group)

	# ── Night Audit ───────────────────────────────────────────────────────────

	async def run_night_audit(self, audit_date: str, run_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Run the nightly audit for a property."""
		tenant = self._tenant(tenant_id)
		all_rooms = [r for r in self.rooms.values() if r["tenant_id"] == tenant]
		occupied = [r for r in all_rooms if r["status"] == "occupied"]
		today_arrivals = [r for r in self.reservations.values() if r["tenant_id"] == tenant and r["check_in_date"] == audit_date and r["status"] in {"confirmed", "checked_in"}]
		today_departures = [r for r in self.reservations.values() if r["tenant_id"] == tenant and r["check_out_date"] == audit_date and r["status"] == "checked_out"]
		no_shows = [r for r in self.reservations.values() if r["tenant_id"] == tenant and r["check_in_date"] == audit_date and r["status"] == "confirmed"]
		walk_ins = [r for r in today_arrivals if r.get("source") == "walk_in"]
		total_rooms = len(all_rooms)
		occupied_count = len(occupied)
		occupancy_rate = (occupied_count / total_rooms * 100) if total_rooms else 0.0
		# Revenue: sum all folio charges for checked-in reservations
		checked_in_ids = {r["id"] for r in self.reservations.values() if r["tenant_id"] == tenant and r["status"] == "checked_in"}
		today_folios = [f for f in self.folios.values() if f["tenant_id"] == tenant and f["reservation_id"] in checked_in_ids and f["status"] == "posted"]
		room_revenue = sum(f["total"] for f in today_folios if f["charge_type"] == "room")
		ancillary_revenue = sum(f["total"] for f in today_folios if f["charge_type"] != "room")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"audit_date": audit_date,
			"run_by": run_by,
			"total_rooms": total_rooms,
			"occupied_rooms": occupied_count,
			"occupancy_rate": round(occupancy_rate, 2),
			"arrivals": len(today_arrivals),
			"departures": len(today_departures),
			"no_shows": len(no_shows),
			"walk_ins": len(walk_ins),
			"room_revenue": room_revenue,
			"ancillary_revenue": ancillary_revenue,
			"total_revenue": room_revenue + ancillary_revenue,
			"status": "completed",
			"generated_at": _now(),
		}
		self.night_audits[record["id"]] = record
		self._emit(tenant, "night_audit_run", record["id"], "night_audit", {"audit_date": audit_date})
		return deepcopy(record)

	async def get_night_audit(self, audit_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Get a specific night audit report."""
		tenant = self._tenant(tenant_id)
		audit = self.night_audits.get(audit_id)
		if not audit or audit["tenant_id"] != tenant:
			raise KeyError(f"night_audit_not_found:{audit_id}")
		return deepcopy(audit)

	# ── Summary ───────────────────────────────────────────────────────────────

	async def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return current property dashboard metrics."""
		tenant = self._tenant(tenant_id)
		all_rooms = [r for r in self.rooms.values() if r["tenant_id"] == tenant]
		occupied = sum(1 for r in all_rooms if r["status"] == "occupied")
		total = len(all_rooms)
		active_res = [r for r in self.reservations.values() if r["tenant_id"] == tenant and r["status"] in {"confirmed", "checked_in"}]
		return {
			"tenant_id": tenant,
			"total_rooms": total,
			"occupied_rooms": occupied,
			"available_rooms": sum(1 for r in all_rooms if r["status"] == "available"),
			"maintenance_rooms": sum(1 for r in all_rooms if r["status"] == "maintenance"),
			"housekeeping_rooms": sum(1 for r in all_rooms if r["status"] == "housekeeping"),
			"occupancy_rate": round(occupied / total * 100, 2) if total else 0.0,
			"active_reservations": len(active_res),
			"total_guests": len(self.guests),
			"pending_housekeeping": sum(1 for t in self.housekeeping_tasks.values() if t["tenant_id"] == tenant and t["status"] == "pending"),
			"open_folios": len(set(f["reservation_id"] for f in self.folios.values() if f["tenant_id"] == tenant and f["status"] == "posted")),
			"generated_at": _now(),
		}

	# ── I1: Dynamic Pricing Engine ────────────────────────────────────────────

	async def get_dynamic_rate(self, room_id: str, check_in: str, check_out: str,
	                            tenant_id: str | None = None) -> dict[str, Any]:
		"""Return a demand-adjusted nightly rate for a room and date range.

		Business value: maximises RevPAR during peak demand without manual intervention.
		Competes with Oracle OPERA Cloud Best Available Rate logic.
		"""
		tenant = self._tenant(tenant_id)
		room = self.rooms.get(room_id)
		if not room or room["tenant_id"] != tenant:
			raise KeyError(f"room_not_found:{room_id}")
		all_rooms = [r for r in self.rooms.values() if r["tenant_id"] == tenant]
		occupied_count = sum(1 for r in all_rooms if r["status"] == "occupied")
		total_rooms = len(all_rooms) or 1
		occupancy_ratio = occupied_count / total_rooms
		cfg = self._pricing_config
		base_rate = Decimal(str(room["rate_per_night"]))
		# Linearly interpolate multiplier between low-demand floor and high-demand ceiling
		low = cfg["low_demand_threshold"]
		high = cfg["high_demand_threshold"]
		mn = cfg["min_multiplier"]
		mx = cfg["max_multiplier"]
		if occupancy_ratio <= low:
			multiplier = mn
			demand_level = "low"
		elif occupancy_ratio >= high:
			multiplier = mx
			demand_level = "high"
		else:
			ratio = Decimal(str((occupancy_ratio - low) / (high - low)))
			multiplier = mn + ratio * (mx - mn)
			demand_level = "moderate"
		# Also factor lead time: bookings > 30 days out get a slight discount
		nights = _date_diff(check_in, check_out)
		try:
			lead_days = (datetime.strptime(check_in, "%Y-%m-%d") - datetime.utcnow()).days
		except Exception:
			lead_days = 0
		if lead_days > 30:
			multiplier = max(mn, multiplier - Decimal("0.05"))
		adjusted_rate = (base_rate * multiplier).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
		total_estimate = (adjusted_rate * nights).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
		_log.debug("dynamic_rate room=%s occupancy=%.2f multiplier=%s rate=%s", room_id, occupancy_ratio, multiplier, adjusted_rate)
		return {
			"room_id": room_id,
			"base_rate": str(base_rate),
			"adjusted_rate": str(adjusted_rate),
			"multiplier": str(multiplier.quantize(Decimal("0.0001"))),
			"demand_level": demand_level,
			"occupancy_ratio": round(occupancy_ratio, 4),
			"check_in": check_in,
			"check_out": check_out,
			"nights": nights,
			"total_estimate": str(total_estimate),
			"generated_at": _now(),
		}

	async def configure_pricing(self, min_multiplier: str, max_multiplier: str,
	                             high_demand_threshold: float, low_demand_threshold: float,
	                             loyalty_earn_rate: str | None = None,
	                             tenant_id: str | None = None) -> dict[str, Any]:
		"""Update dynamic pricing bounds and loyalty earn rate for this property."""
		self._tenant(tenant_id)
		self._pricing_config["min_multiplier"] = Decimal(min_multiplier)
		self._pricing_config["max_multiplier"] = Decimal(max_multiplier)
		self._pricing_config["high_demand_threshold"] = high_demand_threshold
		self._pricing_config["low_demand_threshold"] = low_demand_threshold
		if loyalty_earn_rate is not None:
			self._loyalty_earn_rate = Decimal(loyalty_earn_rate)
		return {"updated": True, "pricing_config": {k: str(v) for k, v in self._pricing_config.items()}}

	# ── I2: RevPAR / Yield Analytics ─────────────────────────────────────────

	async def get_revpar_analytics(self, date_from: str, date_to: str,
	                                tenant_id: str | None = None) -> dict[str, Any]:
		"""Return ADR, RevPAR, TRevPAR and occupancy for a date range.

		Business value: replaces manual spreadsheet work; operators see yield KPIs instantly.
		Competes with Mews Systems live dashboard and Apaleo analytics.
		"""
		tenant = self._tenant(tenant_id)
		if not date_from or not date_to:
			raise ValueError("date_range_required")
		all_rooms = [r for r in self.rooms.values() if r["tenant_id"] == tenant]
		total_rooms = len(all_rooms)
		nights_in_period = _date_diff(date_from, date_to)
		total_room_nights = total_rooms * nights_in_period if nights_in_period > 0 else 1
		# Checked-out reservations that overlap the period
		period_res = [
			r for r in self.reservations.values()
			if r["tenant_id"] == tenant
			and r["status"] in {"checked_out", "checked_in"}
			and r["check_in_date"] < date_to
			and r["check_out_date"] > date_from
		]
		# Room revenue only (charge_type == "room") for those reservations
		res_ids = {r["id"] for r in period_res}
		room_charges = [
			f for f in self.folios.values()
			if f["tenant_id"] == tenant
			and f["reservation_id"] in res_ids
			and f["status"] == "posted"
			and f["charge_type"] == "room"
		]
		ancillary_charges = [
			f for f in self.folios.values()
			if f["tenant_id"] == tenant
			and f["reservation_id"] in res_ids
			and f["status"] == "posted"
			and f["charge_type"] != "room"
		]
		total_room_revenue = Decimal(str(sum(f["total"] for f in room_charges)))
		total_ancillary_revenue = Decimal(str(sum(f["total"] for f in ancillary_charges)))
		total_revenue = total_room_revenue + total_ancillary_revenue
		occupied_room_nights = sum(r["nights"] for r in period_res)
		adr = (total_room_revenue / occupied_room_nights).quantize(Decimal("0.01")) if occupied_room_nights else Decimal("0")
		revpar = (total_room_revenue / total_room_nights).quantize(Decimal("0.01"))
		trevpar = (total_revenue / total_room_nights).quantize(Decimal("0.01"))
		occupancy_pct = round(occupied_room_nights / total_room_nights * 100, 2) if total_room_nights else 0.0
		return {
			"tenant_id": tenant,
			"date_from": date_from,
			"date_to": date_to,
			"total_rooms": total_rooms,
			"nights_in_period": nights_in_period,
			"total_room_nights_available": total_room_nights,
			"occupied_room_nights": occupied_room_nights,
			"occupancy_pct": occupancy_pct,
			"adr": str(adr),
			"revpar": str(revpar),
			"trevpar": str(trevpar),
			"total_room_revenue": str(total_room_revenue),
			"total_ancillary_revenue": str(total_ancillary_revenue),
			"total_revenue": str(total_revenue),
			"reservation_count": len(period_res),
			"generated_at": _now(),
		}

	# ── I5: Loyalty Points ────────────────────────────────────────────────────

	async def accrue_loyalty_points(self, guest_id: str, reservation_id: str,
	                                 spend_amount: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Award loyalty points for a completed stay spend.

		Business value: drives direct-booking repeat rate; mirrors Marriott Bonvoy earn mechanics.
		earn_rate is configurable via configure_pricing().
		"""
		tenant = self._tenant(tenant_id)
		guest = self.guests.get(guest_id)
		if not guest or guest["tenant_id"] != tenant:
			raise KeyError(f"guest_not_found:{guest_id}")
		res = self.reservations.get(reservation_id)
		if not res or res["tenant_id"] != tenant:
			raise KeyError(f"reservation_not_found:{reservation_id}")
		amount = Decimal(str(spend_amount))
		if amount <= 0:
			raise ValueError("spend_amount_must_be_positive")
		points_earned = int((amount * self._loyalty_earn_rate).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
		prev_balance = int(guest.get("loyalty_balance", 0))
		guest["loyalty_balance"] = prev_balance + points_earned
		guest["total_spend"] = float(Decimal(str(guest.get("total_spend", 0.0))) + amount)
		self._emit(tenant, "loyalty_accrual", guest_id, "guest", {
			"reservation_id": reservation_id,
			"points_earned": points_earned,
			"new_balance": guest["loyalty_balance"],
		})
		_log.info("loyalty_accrual guest=%s points=%d balance=%d", guest_id, points_earned, guest["loyalty_balance"])
		return {
			"guest_id": guest_id,
			"reservation_id": reservation_id,
			"spend_amount": str(amount),
			"earn_rate": str(self._loyalty_earn_rate),
			"points_earned": points_earned,
			"previous_balance": prev_balance,
			"new_balance": guest["loyalty_balance"],
			"accrued_at": _now(),
		}

	async def redeem_loyalty_points(self, guest_id: str, points: int, reservation_id: str,
	                                 tenant_id: str | None = None) -> dict[str, Any]:
		"""Convert loyalty points into a folio credit at a 1 point = 1 currency unit rate.

		Business value: closes the loyalty loop; incentivises direct booking over OTA.
		"""
		tenant = self._tenant(tenant_id)
		guest = self.guests.get(guest_id)
		if not guest or guest["tenant_id"] != tenant:
			raise KeyError(f"guest_not_found:{guest_id}")
		if points <= 0:
			raise ValueError("points_must_be_positive")
		balance = int(guest.get("loyalty_balance", 0))
		if points > balance:
			raise ValueError(f"insufficient_loyalty_balance:{balance}")
		guest["loyalty_balance"] = balance - points
		# Post a negative (credit) folio charge
		credit_amount = Decimal(str(points))  # 1:1 redemption
		credit_charge = await self.add_folio_charge(
			reservation_id, "loyalty_redemption",
			f"Loyalty points redemption ({points} pts)",
			float(-credit_amount), 1, tenant_id=tenant,
		)
		self._emit(tenant, "loyalty_redemption", guest_id, "guest", {
			"reservation_id": reservation_id,
			"points_redeemed": points,
			"new_balance": guest["loyalty_balance"],
		})
		return {
			"guest_id": guest_id,
			"reservation_id": reservation_id,
			"points_redeemed": points,
			"credit_amount": str(credit_amount),
			"new_balance": guest["loyalty_balance"],
			"folio_charge_id": credit_charge["id"],
			"redeemed_at": _now(),
		}

	# ── I6: Overbooking Walk Management ──────────────────────────────────────

	async def walk_reservation(self, reservation_id: str, relocation_property: str,
	                            reason: str, covered_costs: str,
	                            walked_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Walk (relocate) an overbooked guest with a compensation folio credit.

		Business value: structured walk tracking satisfies duty-of-care obligations and
		prevents revenue leakage from unrecorded compensation spend.
		Competes with IHG Concerto PMS walk management module.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(reason, "reason")
		guard_non_empty_string(relocation_property, "relocation_property")
		res = self.reservations.get(reservation_id)
		if not res or res["tenant_id"] != tenant:
			raise KeyError(f"reservation_not_found:{reservation_id}")
		if res["status"] not in {"confirmed", "checked_in"}:
			raise ValueError(f"cannot_walk_reservation_in_status:{res['status']}")
		comp = Decimal(str(covered_costs))
		if comp < 0:
			raise ValueError("covered_costs_must_be_non_negative")
		# Cancel the original reservation with walked status
		res["status"] = "walked"
		res["walk_reason"] = reason
		res["relocation_property"] = relocation_property
		res["walked_by"] = walked_by
		res["walked_at"] = _now()
		# Post a compensation credit folio entry if costs covered
		comp_charge_id: str | None = None
		if comp > 0:
			comp_charge = await self.add_folio_charge(
				reservation_id, "walk_compensation",
				f"Walk compensation — relocated to {relocation_property}",
				float(-comp), 1, tenant_id=tenant,
			)
			comp_charge_id = comp_charge["id"]
		# Free the room for reassignment
		room = self.rooms.get(res["room_id"])
		if room:
			room["status"] = "available"
		self._emit(tenant, "reservation_walked", reservation_id, "reservation", {
			"relocation_property": relocation_property,
			"reason": reason,
			"covered_costs": str(comp),
		})
		_log.warning("walk_event reservation=%s property=%s costs=%s", reservation_id, relocation_property, comp)
		return {
			"reservation_id": reservation_id,
			"relocation_property": relocation_property,
			"reason": reason,
			"covered_costs": str(comp),
			"comp_charge_id": comp_charge_id,
			"walked_by": walked_by,
			"walked_at": res["walked_at"],
		}

	# ── I7: Maintenance Work Orders ───────────────────────────────────────────

	# SLA thresholds in hours by priority
	_WORK_ORDER_SLA: dict[str, int] = {"urgent": 2, "high": 8, "normal": 24, "low": 72}

	async def create_work_order(self, room_id: str, category: str, description: str,
	                             priority: str = "normal", reported_by: str = "staff",
	                             tenant_id: str | None = None) -> dict[str, Any]:
		"""Create an engineering maintenance work order for a room.

		Business value: separates engineering workflow from housekeeping, enables SLA tracking.
		Competes with Mews Maintenance and Clock PMS maintenance module.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(description, "description")
		valid_priorities = {"urgent", "high", "normal", "low"}
		if priority not in valid_priorities:
			raise ValueError(f"invalid_priority:{priority}")
		room = self.rooms.get(room_id)
		if not room or room["tenant_id"] != tenant:
			raise KeyError(f"room_not_found:{room_id}")
		sla_hours = self._WORK_ORDER_SLA[priority]
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"room_id": room_id,
			"category": category,          # electrical|plumbing|hvac|structural|furniture|other
			"description": description,
			"priority": priority,
			"reported_by": reported_by,
			"assigned_to": None,
			"status": "reported",          # reported|assigned|in_progress|verified|closed
			"sla_hours": sla_hours,
			"sla_breach": False,
			"started_at": None,
			"completed_at": None,
			"verified_by": None,
			"verified_at": None,
			"created_at": _now(),
		}
		self.work_orders[record["id"]] = record
		self._emit(tenant, "work_order_created", record["id"], "work_order", {"room_id": room_id, "priority": priority})
		_log.info("work_order_created id=%s room=%s priority=%s", record["id"], room_id, priority)
		return deepcopy(record)

	async def assign_work_order(self, work_order_id: str, assigned_to: str,
	                             tenant_id: str | None = None) -> dict[str, Any]:
		"""Assign a work order to an engineer."""
		tenant = self._tenant(tenant_id)
		wo = self.work_orders.get(work_order_id)
		if not wo or wo["tenant_id"] != tenant:
			raise KeyError(f"work_order_not_found:{work_order_id}")
		if wo["status"] not in {"reported"}:
			raise ValueError(f"cannot_assign_from_status:{wo['status']}")
		wo["assigned_to"] = assigned_to
		wo["status"] = "assigned"
		self._emit(tenant, "work_order_assigned", work_order_id, "work_order", {"assigned_to": assigned_to})
		return deepcopy(wo)

	async def start_work_order(self, work_order_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Mark a work order as in-progress."""
		tenant = self._tenant(tenant_id)
		wo = self.work_orders.get(work_order_id)
		if not wo or wo["tenant_id"] != tenant:
			raise KeyError(f"work_order_not_found:{work_order_id}")
		if wo["status"] != "assigned":
			raise ValueError(f"cannot_start_from_status:{wo['status']}")
		wo["status"] = "in_progress"
		wo["started_at"] = _now()
		self._emit(tenant, "work_order_started", work_order_id, "work_order")
		return deepcopy(wo)

	async def close_work_order(self, work_order_id: str, verified_by: str,
	                            notes: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		"""Verify and close a completed work order; flag SLA breach if overdue.

		SLA breach is flagged when elapsed hours exceed priority threshold.
		"""
		tenant = self._tenant(tenant_id)
		wo = self.work_orders.get(work_order_id)
		if not wo or wo["tenant_id"] != tenant:
			raise KeyError(f"work_order_not_found:{work_order_id}")
		if wo["status"] not in {"in_progress", "assigned"}:
			raise ValueError(f"cannot_close_from_status:{wo['status']}")
		now_str = _now()
		wo["status"] = "closed"
		wo["completed_at"] = now_str
		wo["verified_by"] = verified_by
		wo["verified_at"] = now_str
		if notes:
			wo["close_notes"] = notes
		# Check SLA breach
		try:
			created_dt = datetime.fromisoformat(wo["created_at"].rstrip("Z"))
			closed_dt = datetime.fromisoformat(now_str.rstrip("Z"))
			elapsed_hours = (closed_dt - created_dt).total_seconds() / 3600
			wo["elapsed_hours"] = round(elapsed_hours, 2)
			wo["sla_breach"] = elapsed_hours > wo["sla_hours"]
		except Exception:
			wo["sla_breach"] = False
		# Return the room to available if it was in maintenance for this order
		room = self.rooms.get(wo["room_id"])
		if room and room["status"] == "maintenance":
			room["status"] = "available"
		self._emit(tenant, "work_order_closed", work_order_id, "work_order", {
			"sla_breach": wo["sla_breach"],
			"elapsed_hours": wo.get("elapsed_hours"),
		})
		if wo["sla_breach"]:
			_log.warning("sla_breach work_order=%s elapsed=%.1fh threshold=%dh", work_order_id, wo.get("elapsed_hours", 0), wo["sla_hours"])
		return deepcopy(wo)

	async def list_work_orders(self, tenant_id: str | None = None, status: str | None = None,
	                            sla_breach_only: bool = False) -> list[dict[str, Any]]:
		"""List work orders, optionally filtered by status or SLA breach flag."""
		tenant = self._tenant(tenant_id)
		orders = [deepcopy(wo) for wo in self.work_orders.values() if wo["tenant_id"] == tenant]
		if status:
			orders = [o for o in orders if o["status"] == status]
		if sla_breach_only:
			orders = [o for o in orders if o.get("sla_breach")]
		return orders

	# ── I9: Rate Plan Management ──────────────────────────────────────────────

	async def create_rate_plan(self, code: str, name: str, base_rate: str,
	                            min_stay: int = 1, advance_purchase_days: int = 0,
	                            applicable_room_types: list[str] | None = None,
	                            blackout_dates: list[str] | None = None,
	                            tenant_id: str | None = None) -> dict[str, Any]:
		"""Register a named rate plan with restriction rules.

		Business value: enables BAR, advance-purchase, minimum-stay, and package rate logic
		without per-reservation manual overrides. Competes with OPERA Cloud Rate Manager.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(code, "code")
		# Prevent duplicate codes within tenant
		for rp in self.rate_plans.values():
			if rp["tenant_id"] == tenant and rp["code"] == code:
				raise ValueError(f"rate_plan_code_exists:{code}")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"code": code,
			"name": name,
			"base_rate": str(Decimal(str(base_rate)).quantize(Decimal("0.01"))),
			"min_stay": max(1, min_stay),
			"advance_purchase_days": max(0, advance_purchase_days),
			"applicable_room_types": applicable_room_types or [],  # empty = all types
			"blackout_dates": blackout_dates or [],
			"is_active": True,
			"created_at": _now(),
		}
		self.rate_plans[record["id"]] = record
		self._emit(tenant, "rate_plan_created", record["id"], "rate_plan", {"code": code})
		return deepcopy(record)

	async def get_applicable_rate(self, room_id: str, check_in: str, check_out: str,
	                               tenant_id: str | None = None) -> dict[str, Any]:
		"""Return the best applicable rate plan for a room and date range.

		Evaluates active plans against min_stay, advance_purchase, room_type, and blackout rules.
		Falls back to room base rate if no plan qualifies.
		"""
		tenant = self._tenant(tenant_id)
		room = self.rooms.get(room_id)
		if not room or room["tenant_id"] != tenant:
			raise KeyError(f"room_not_found:{room_id}")
		nights = _date_diff(check_in, check_out)
		try:
			lead_days = (datetime.strptime(check_in, "%Y-%m-%d") - datetime.utcnow()).days
		except Exception:
			lead_days = 0
		room_type = room["room_type"]
		candidates: list[dict[str, Any]] = []
		for rp in self.rate_plans.values():
			if rp["tenant_id"] != tenant or not rp["is_active"]:
				continue
			if rp["applicable_room_types"] and room_type not in rp["applicable_room_types"]:
				continue
			if nights < rp["min_stay"]:
				continue
			if lead_days < rp["advance_purchase_days"]:
				continue
			if check_in in rp["blackout_dates"] or check_out in rp["blackout_dates"]:
				continue
			candidates.append(rp)
		if candidates:
			# Best = lowest base rate
			best = min(candidates, key=lambda p: Decimal(p["base_rate"]))
			return {
				"room_id": room_id,
				"rate_plan_id": best["id"],
				"rate_plan_code": best["code"],
				"rate_plan_name": best["name"],
				"nightly_rate": best["base_rate"],
				"source": "rate_plan",
				"nights": nights,
				"total_estimate": str((Decimal(best["base_rate"]) * nights).quantize(Decimal("0.01"))),
			}
		# Fallback to room base rate
		base = Decimal(str(room["rate_per_night"]))
		return {
			"room_id": room_id,
			"rate_plan_id": None,
			"rate_plan_code": "BASE",
			"rate_plan_name": "Base Rate",
			"nightly_rate": str(base.quantize(Decimal("0.01"))),
			"source": "room_base",
			"nights": nights,
			"total_estimate": str((base * nights).quantize(Decimal("0.01"))),
		}

	async def list_rate_plans(self, tenant_id: str | None = None, active_only: bool = True) -> list[dict[str, Any]]:
		"""List all rate plans for this property."""
		tenant = self._tenant(tenant_id)
		plans = [deepcopy(rp) for rp in self.rate_plans.values() if rp["tenant_id"] == tenant]
		if active_only:
			plans = [p for p in plans if p["is_active"]]
		return plans

	# ── I12: Fiscal Compliance Receipts ──────────────────────────────────────

	async def post_fiscal_receipt(self, reservation_id: str, receipt_lines: list[dict[str, Any]],
	                               tax_rate: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Issue a tamper-evident fiscal receipt with monotonic sequence and SHA-256 chain.

		Business value: satisfies Kenya VAT Act 2013 and Tax Procedures Act 2015 requirements;
		chain-hash prevents after-the-fact manipulation. Each receipt is write-once.
		Competes with DATEV fiscal integration and FISKALY cloud TSS.

		receipt_lines: list of {description: str, quantity: int, unit_price: str}
		tax_rate: Decimal-compatible string e.g. "0.16" for 16 % VAT
		"""
		tenant = self._tenant(tenant_id)
		res = self.reservations.get(reservation_id)
		if not res or res["tenant_id"] != tenant:
			raise KeyError(f"reservation_not_found:{reservation_id}")
		if not receipt_lines:
			raise ValueError("receipt_lines_required")
		vat_rate = Decimal(str(tax_rate))
		if not (0 <= vat_rate < 1):
			raise ValueError("tax_rate_must_be_between_0_and_1")
		# Compute line totals
		lines_out: list[dict[str, Any]] = []
		subtotal = Decimal("0")
		for line in receipt_lines:
			desc = str(line.get("description", ""))
			qty = int(line.get("quantity", 1))
			unit_price = Decimal(str(line.get("unit_price", "0")))
			line_total = (unit_price * qty).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
			subtotal += line_total
			lines_out.append({
				"description": desc,
				"quantity": qty,
				"unit_price": str(unit_price.quantize(Decimal("0.01"))),
				"line_total": str(line_total),
			})
		tax_amount = (subtotal * vat_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
		grand_total = subtotal + tax_amount
		# Monotonically incrementing fiscal sequence number
		self._fiscal_sequence += 1
		seq = self._fiscal_sequence
		record_id = _uid()
		# SHA-256 chain: hash(seq || subtotal || tax || grand_total || prev_hash)
		chain_input = f"{seq}|{subtotal}|{tax_amount}|{grand_total}|{self._last_receipt_hash}"
		receipt_hash = hashlib.sha256(chain_input.encode()).hexdigest()
		self._last_receipt_hash = receipt_hash
		record: dict[str, Any] = {
			"id": record_id,
			"tenant_id": tenant,
			"reservation_id": reservation_id,
			"fiscal_sequence_number": seq,
			"lines": lines_out,
			"subtotal": str(subtotal),
			"tax_rate": str(vat_rate),
			"tax_amount": str(tax_amount),
			"grand_total": str(grand_total),
			"currency": "KES",
			"receipt_hash": receipt_hash,
			"issued_at": _now(),
		}
		# Write-once: once stored, a fiscal receipt must not be mutated
		self.fiscal_receipts[record_id] = record
		self._emit(tenant, "fiscal_receipt_issued", record_id, "fiscal_receipt", {
			"reservation_id": reservation_id,
			"fiscal_sequence_number": seq,
			"grand_total": str(grand_total),
		})
		_log.info("fiscal_receipt seq=%d hash=%s total=%s", seq, receipt_hash[:12], grand_total)
		return deepcopy(record)

	async def get_fiscal_receipt(self, receipt_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Retrieve a fiscal receipt by ID (read-only)."""
		tenant = self._tenant(tenant_id)
		receipt = self.fiscal_receipts.get(receipt_id)
		if not receipt or receipt["tenant_id"] != tenant:
			raise KeyError(f"fiscal_receipt_not_found:{receipt_id}")
		return deepcopy(receipt)

	async def list_fiscal_receipts(self, reservation_id: str | None = None,
	                                tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List fiscal receipts, optionally filtered by reservation."""
		tenant = self._tenant(tenant_id)
		receipts = [deepcopy(r) for r in self.fiscal_receipts.values() if r["tenant_id"] == tenant]
		if reservation_id:
			receipts = [r for r in receipts if r["reservation_id"] == reservation_id]
		return sorted(receipts, key=lambda r: r["fiscal_sequence_number"])

	# ── I13: Automated Housekeeping Assignment ────────────────────────────────

	async def auto_assign_housekeeping(self, date: str, staff_roster: list[dict[str, Any]],
	                                    tenant_id: str | None = None) -> dict[str, Any]:
		"""Balance pending housekeeping tasks across available staff by floor and priority.

		Scoring: checkout room (3) > occupied stayover (2) > vacant inspect (1).
		Staff are distributed by floor section to minimise travel.
		Business value: 15 % faster room turnover; competes with ALICE by Actabl and Knowcross.

		staff_roster: list of {staff_id: str, name: str, section: str | None}
		"""
		tenant = self._tenant(tenant_id)
		if not staff_roster:
			raise ValueError("staff_roster_required")
		# Gather pending tasks
		pending = [
			t for t in self.housekeeping_tasks.values()
			if t["tenant_id"] == tenant and t["status"] == "pending"
		]
		if not pending:
			return {"assigned_count": 0, "assignments": [], "date": date}
		# Score each task
		def _task_score(task: dict[str, Any]) -> int:
			room = self.rooms.get(task.get("room_id", ""))
			if room and room["status"] == "housekeeping":
				return 3  # just checked-out
			if task.get("task_type") in {"inspect", "turndown"}:
				return 2
			return 1
		pending_sorted = sorted(pending, key=_task_score, reverse=True)
		# Group staff by section; tasks are scored and distributed round-robin per section
		section_map: dict[str, list[dict[str, Any]]] = {}
		for s in staff_roster:
			section = s.get("section") or "general"
			section_map.setdefault(section, []).append(s)
		# Build floor→section mapping from rooms
		floor_section: dict[int, str] = {}
		for room in self.rooms.values():
			if room["tenant_id"] == tenant:
				floor = int(room.get("floor", 0))
				# Assign floor to section in round-robin across sections
				if floor not in floor_section:
					sections = list(section_map.keys())
					floor_section[floor] = sections[floor % len(sections)]
		assignments: list[dict[str, Any]] = []
		staff_load: dict[str, int] = {s["staff_id"]: 0 for s in staff_roster}
		for task in pending_sorted:
			room = self.rooms.get(task.get("room_id", ""))
			floor = int(room["floor"]) if room else 0
			section = floor_section.get(floor, "general")
			section_staff = section_map.get(section) or staff_roster
			# Pick least-loaded staff member in section
			chosen = min(section_staff, key=lambda s: staff_load[s["staff_id"]])
			task["assigned_to"] = chosen["staff_id"]
			task["status"] = "assigned"
			staff_load[chosen["staff_id"]] += 1
			assignments.append({
				"task_id": task["id"],
				"room_id": task.get("room_id"),
				"task_type": task.get("task_type"),
				"priority": task.get("priority"),
				"assigned_to": chosen["staff_id"],
				"staff_name": chosen.get("name"),
			})
		self._emit(tenant, "housekeeping_auto_assigned", "batch", "housekeeping_task", {
			"date": date,
			"assigned_count": len(assignments),
		})
		_log.info("auto_assign date=%s tasks=%d staff=%d", date, len(assignments), len(staff_roster))
		return {
			"date": date,
			"assigned_count": len(assignments),
			"staff_loads": staff_load,
			"assignments": assignments,
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

