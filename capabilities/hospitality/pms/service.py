"""Property Management System service — room inventory, check-in/out, housekeeping, folio, night audit, group bookings."""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

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

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.rooms: dict[str, dict[str, Any]] = {}
		self.guests: dict[str, dict[str, Any]] = {}
		self.reservations: dict[str, dict[str, Any]] = {}
		self.folios: dict[str, dict[str, Any]] = {}
		self.housekeeping_tasks: dict[str, dict[str, Any]] = {}
		self.group_bookings: dict[str, dict[str, Any]] = {}
		self.night_audits: dict[str, dict[str, Any]] = {}
		self.payments: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

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
			"version": "1.0.0",
			"description": "Room inventory, check-in/out, housekeeping, folio management, night audit, group bookings",
			"features": [
				"room_inventory", "guest_profiles", "reservations",
				"check_in_out", "folio_management", "housekeeping",
				"night_audit", "group_bookings", "payment_tracking",
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
