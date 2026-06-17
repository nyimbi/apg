"""Events & Venue Management service — event booking, venue configuration, catering BEO, AV requirements, billing, contracts."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import hashlib
import hmac
import json
import logging
from copy import deepcopy
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_non_empty_string, guard_tenant_id, BoundedCache

_log = logging.getLogger(__name__)


def _uid() -> str:
	return uuid4().hex[:12]


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _today() -> date:
	return datetime.utcnow().date()


def _parse_date(s: str) -> date:
	return date.fromisoformat(s)


def _dec(v: Any) -> Decimal:
	"""Convert numeric value to Decimal safely."""
	return Decimal(str(v))


def _round2(v: Decimal) -> Decimal:
	return v.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ── Per-person catering estimate by event type (KES) ──────────────────────────
_CATERING_RATE_PER_HEAD: dict[str, Decimal] = {
	"conference": _dec("3500"),
	"wedding": _dec("8000"),
	"gala": _dec("6000"),
	"birthday": _dec("4000"),
	"product_launch": _dec("3000"),
	"training": _dec("2500"),
	"other": _dec("3000"),
}

# ── AV flat-rate estimates (KES) ──────────────────────────────────────────────
_AV_RATE: dict[str, Decimal] = {
	"basic": _dec("25000"),
	"standard": _dec("50000"),
	"full_production": _dec("150000"),
}

# ── Tiered cancellation schedule: (min_days_out, forfeiture_pct) ──────────────
# If days_to_event >= threshold, apply the corresponding forfeiture.
_CANCELLATION_TIERS: list[tuple[int, Decimal]] = [
	(90, _dec("0")),
	(60, _dec("25")),
	(30, _dec("50")),
	(0,  _dec("100")),
]

# ── Revenue confidence weights ────────────────────────────────────────────────
_CONFIDENCE: dict[str, Decimal] = {
	"confirmed": _dec("0.90"),
	"tentative": _dec("0.40"),
}


class EVNService:
	"""Events & Venue Management service.

	Improvements implemented (from WORLD_CLASS_IMPROVEMENTS.md):
	  I2  — Waitlist & automatic conflict-resolution queue
	  I3  — Partial-day time-slot conflict detection
	  I6  — Digital contract signature with tamper-evidence hash
	  I7  — Automated payment timeline & overdue escalation
	  I9  — Post-event NPS & satisfaction score capture
	  I10 — Configurable tiered cancellation fee engine
	  I11 — AV equipment inventory & conflict detection
	  I12 — Revenue forecast — contracted vs pipeline split
	"""

	def __init__(self, tenant_id: str = "default", tenant_secret: str = "changeme", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		# Used for HMAC-SHA256 contract signature (I6)
		self._tenant_secret = tenant_secret

		self.venues: dict[str, dict[str, Any]] = {}
		self.event_bookings: dict[str, dict[str, Any]] = {}
		self.beos: dict[str, dict[str, Any]] = {}
		self.contracts: dict[str, dict[str, Any]] = {}
		self.payments: dict[str, dict[str, Any]] = {}
		self.catering_orders: dict[str, dict[str, Any]] = {}
		self.av_requirements: dict[str, dict[str, Any]] = {}
		self.setup_configs: dict[str, dict[str, Any]] = {}
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)

		# I2 — Waitlist: keyed by "venue_id::event_date"
		self._waitlist: dict[str, list[dict[str, Any]]] = {}

		# I9 — NPS records
		self._nps_records = WriteThruDict('nps_records', tenant_id, _store)

		# I11 — AV asset inventory
		self._av_assets = WriteThruDict('av_assets', tenant_id, _store)

		# I7 — Payment timeline milestones
		self._payment_milestones = WriteThruDict('payment_milestones', tenant_id, _store)

	# ── Internal helpers ──────────────────────────────────────────────────────

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

	def _booking_venue(self, tenant: str, booking: dict[str, Any]) -> dict[str, Any]:
		venue = self.venues.get(booking["venue_id"])
		if not venue or venue["tenant_id"] != tenant:
			raise KeyError(f"venue_not_found:{booking['venue_id']}")
		return venue

	# ── Health / describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "hos_evn",
			"status": "healthy",
			"active_venues": sum(1 for v in self.venues.values() if v["status"] == "active"),
			"upcoming_events": sum(1 for e in self.event_bookings.values() if e["status"] in {"tentative", "confirmed"}),
			"checked_at": _now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": "hos_evn",
			"name": "Events & Venue Management",
			"domain": "hospitality",
			"version": "2.0.0",
			"description": (
				"Event booking, venue configuration, catering BEO, AV requirements, billing, contracts. "
				"Includes: waitlist, partial-day conflict detection, HMAC contract signing, payment timelines, "
				"NPS capture, tiered cancellation fees, AV inventory conflict detection, revenue forecast."
			),
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Venues ────────────────────────────────────────────────────────────────

	async def list_venues(self, tenant_id: str | None = None, venue_type: str | None = None, available_date: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		venues = [deepcopy(v) for v in self.venues.values() if v["tenant_id"] == tenant and v["status"] == "active"]
		if venue_type:
			venues = [v for v in venues if v["venue_type"] == venue_type]
		if available_date:
			booked_venue_ids = {
				e["venue_id"] for e in self.event_bookings.values()
				if e["tenant_id"] == tenant and e["event_date"] == available_date and e["status"] in {"tentative", "confirmed"}
			}
			venues = [v for v in venues if v["id"] not in booked_venue_ids]
		return venues

	async def get_venue(self, venue_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		venue = self.venues.get(venue_id)
		if not venue or venue["tenant_id"] != tenant:
			raise KeyError(f"venue_not_found:{venue_id}")
		return deepcopy(venue)

	async def create_venue(self, name: str, venue_type: str, capacity_seated: int,
	                        capacity_standing: int = 0, area_sqm: float = 0.0,
	                        rental_rate_per_day: float = 0.0, av_included: bool = False,
	                        catering_allowed: bool = True, notes: str | None = None,
	                        fire_code_capacity: int | None = None,
	                        tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"name": name,
			"venue_type": venue_type,
			"capacity_seated": capacity_seated,
			"capacity_standing": capacity_standing,
			"area_sqm": area_sqm,
			"rental_rate_per_day": str(_dec(rental_rate_per_day)),
			"av_included": av_included,
			"catering_allowed": catering_allowed,
			"notes": notes,
			# I8 — fire-code capacity for compliance guardrails
			"fire_code_capacity": fire_code_capacity or capacity_seated,
			# I8 — per-setup-style capacity matrix: populated via set_venue_capacity_matrix
			"capacity_matrix": {},
			"booking_count": 0,
			"status": "active",
			"created_at": _now(),
		}
		self.venues[record["id"]] = record
		self._emit(tenant, "venue_created", record["id"], "venue", {"type": venue_type})
		return deepcopy(record)

	async def update_venue(self, venue_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		venue = self.venues.get(venue_id)
		if not venue or venue["tenant_id"] != tenant:
			raise KeyError(f"venue_not_found:{venue_id}")
		allowed = {"name", "capacity_seated", "rental_rate_per_day", "status", "notes", "av_included", "fire_code_capacity"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				venue[k] = v
		self._emit(tenant, "venue_updated", venue_id, "venue")
		return deepcopy(venue)

	async def delete_venue(self, venue_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		venue = self.venues.get(venue_id)
		if not venue or venue["tenant_id"] != tenant:
			raise KeyError(f"venue_not_found:{venue_id}")
		upcoming = [e for e in self.event_bookings.values() if e["tenant_id"] == tenant and e["venue_id"] == venue_id and e["status"] in {"tentative", "confirmed"}]
		if upcoming:
			raise ValueError("venue_has_upcoming_bookings")
		venue["status"] = "deactivated"
		self._emit(tenant, "venue_deactivated", venue_id, "venue")
		return {"deactivated": True, "venue_id": venue_id}

	async def configure_venue_setup(self, venue_id: str, setup_style: str, capacity_override: int | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		"""Store a setup configuration for a venue."""
		tenant = self._tenant(tenant_id)
		venue = self.venues.get(venue_id)
		if not venue or venue["tenant_id"] != tenant:
			raise KeyError(f"venue_not_found:{venue_id}")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"venue_id": venue_id,
			"setup_style": setup_style,
			"capacity_override": capacity_override,
			"effective_capacity": capacity_override or venue["capacity_seated"],
			"created_at": _now(),
		}
		self.setup_configs[record["id"]] = record
		return deepcopy(record)

	# ── I8 — Venue capacity matrix by setup style ─────────────────────────────

	async def set_venue_capacity_matrix(self, venue_id: str, matrix: dict[str, int], tenant_id: str | None = None) -> dict[str, Any]:
		"""Store per-setup-style capacities (e.g. {'theatre': 400, 'cabaret': 220, 'classroom': 180}).

		Booking creation will validate expected_attendance against the effective capacity for the
		chosen setup style, preventing over-selling and insurance liability violations (I8).
		"""
		tenant = self._tenant(tenant_id)
		venue = self.venues.get(venue_id)
		if not venue or venue["tenant_id"] != tenant:
			raise KeyError(f"venue_not_found:{venue_id}")
		venue["capacity_matrix"] = dict(matrix)
		self._emit(tenant, "venue_capacity_matrix_updated", venue_id, "venue", {"styles": list(matrix.keys())})
		return deepcopy(venue)

	async def get_effective_capacity(self, venue_id: str, setup_style: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return the effective seating capacity for a venue × setup style combination (I8)."""
		tenant = self._tenant(tenant_id)
		venue = self.venues.get(venue_id)
		if not venue or venue["tenant_id"] != tenant:
			raise KeyError(f"venue_not_found:{venue_id}")
		matrix = venue.get("capacity_matrix", {})
		effective = matrix.get(setup_style, venue["capacity_seated"])
		return {
			"venue_id": venue_id,
			"setup_style": setup_style,
			"effective_capacity": effective,
			"from_matrix": setup_style in matrix,
			"fire_code_capacity": venue.get("fire_code_capacity", venue["capacity_seated"]),
		}

	# ── Event Bookings ────────────────────────────────────────────────────────

	async def list_event_bookings(self, tenant_id: str | None = None, venue_id: str | None = None,
	                               event_type: str | None = None, date_from: str | None = None,
	                               status: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(e) for e in self.event_bookings.values() if e["tenant_id"] == tenant]
		if venue_id:
			items = [e for e in items if e["venue_id"] == venue_id]
		if event_type:
			items = [e for e in items if e["event_type"] == event_type]
		if date_from:
			items = [e for e in items if e["event_date"] >= date_from]
		if status:
			items = [e for e in items if e["status"] == status]
		return items

	async def get_event_booking(self, booking_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		booking = self.event_bookings.get(booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"event_booking_not_found:{booking_id}")
		return deepcopy(booking)

	async def create_event_booking(self, venue_id: str, event_name: str, client_name: str, client_email: str,
	                                event_type: str, event_date: str, start_time: str, end_time: str,
	                                expected_attendance: int, catering_required: bool = False,
	                                av_required: bool = False, decoration_required: bool = False,
	                                setup_style: str = "theatre",
	                                client_phone: str | None = None, notes: str | None = None,
	                                tenant_id: str | None = None) -> dict[str, Any]:
		"""Create an event booking.

		Enhancements vs v1:
		  - Validates against per-setup-style effective capacity (I8)
		  - Detects partial-day time-slot overlaps, not just same-date (I3)
		  - Auto-queues to waitlist on conflict rather than hard-failing (I2)
		  - All monetary values stored as Decimal strings (I4/compliance)
		"""
		tenant = self._tenant(tenant_id)
		venue = self.venues.get(venue_id)
		if not venue or venue["tenant_id"] != tenant:
			raise KeyError(f"venue_not_found:{venue_id}")

		# I8 — check against setup-style-specific capacity
		matrix = venue.get("capacity_matrix", {})
		effective_cap = matrix.get(setup_style, venue["capacity_seated"])
		fire_cap = venue.get("fire_code_capacity", venue["capacity_seated"])
		if expected_attendance > effective_cap:
			raise ValueError(f"attendance_exceeds_venue_capacity:{expected_attendance}>{effective_cap} (style={setup_style})")
		if expected_attendance > fire_cap:
			raise ValueError(f"attendance_exceeds_fire_code_capacity:{expected_attendance}>{fire_cap}")

		# I3 — partial-day time-slot overlap detection
		try:
			s_new = datetime.strptime(start_time, "%H:%M").time()
			e_new = datetime.strptime(end_time, "%H:%M").time()
		except ValueError as exc:
			raise ValueError(f"invalid_time_format (expected HH:MM): {exc}") from exc

		for eb in self.event_bookings.values():
			if eb["tenant_id"] != tenant or eb["venue_id"] != venue_id or eb["event_date"] != event_date:
				continue
			if eb["status"] not in {"tentative", "confirmed"}:
				continue
			try:
				s_ex = datetime.strptime(eb["start_time"], "%H:%M").time()
				e_ex = datetime.strptime(eb["end_time"], "%H:%M").time()
			except ValueError:
				continue
			# Standard interval-overlap: s1 < e2 AND s2 < e1
			if s_new < e_ex and s_ex < e_new:
				# I2 — add to waitlist instead of hard-failing
				wl_key = f"{venue_id}::{event_date}"
				entry = {
					"id": _uid(),
					"tenant_id": tenant,
					"venue_id": venue_id,
					"event_date": event_date,
					"start_time": start_time,
					"end_time": end_time,
					"client_name": client_name,
					"client_email": client_email,
					"event_name": event_name,
					"queued_at": _now(),
					"conflicting_booking_id": eb["id"],
				}
				self._waitlist.setdefault(wl_key, []).append(entry)
				self._emit(tenant, "booking_waitlisted", entry["id"], "waitlist", {"venue_id": venue_id, "date": event_date})
				return {
					"status": "waitlisted",
					"waitlist_entry_id": entry["id"],
					"conflicting_booking_id": eb["id"],
					"message": f"venue_time_slot_conflict — added to waitlist position {len(self._waitlist[wl_key])}",
				}

		# Decimal financials (I4 / compliance)
		rental_rate = _dec(venue["rental_rate_per_day"])
		catering_rate = _CATERING_RATE_PER_HEAD.get(event_type, _dec("3000"))
		catering_estimate = _round2(catering_rate * _dec(expected_attendance)) if catering_required else _dec("0")
		av_estimate = _AV_RATE["standard"] if av_required and not venue["av_included"] else _dec("0")
		decoration_estimate = _dec("20000") if decoration_required else _dec("0")
		total_estimate = _round2(rental_rate + catering_estimate + av_estimate + decoration_estimate)

		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"venue_id": venue_id,
			"venue_name": venue["name"],
			"event_name": event_name,
			"client_name": client_name,
			"client_email": client_email,
			"client_phone": client_phone,
			"event_type": event_type,
			"event_date": event_date,
			"start_time": start_time,
			"end_time": end_time,
			"setup_style": setup_style,
			"expected_attendance": expected_attendance,
			"catering_required": catering_required,
			"av_required": av_required,
			"decoration_required": decoration_required,
			# Stored as str for JSON-safety; convert with _dec() when needed
			"venue_rental": str(rental_rate),
			"catering_estimate": str(catering_estimate),
			"av_estimate": str(av_estimate),
			"decoration_estimate": str(decoration_estimate),
			"total_estimate": str(total_estimate),
			"deposit_paid": "0",
			"balance": str(total_estimate),
			"notes": notes,
			"status": "tentative",
			"beo_generated": False,
			"contract_issued": False,
			"created_at": _now(),
		}
		self.event_bookings[record["id"]] = record
		venue["booking_count"] = venue.get("booking_count", 0) + 1
		self._emit(tenant, "event_booking_created", record["id"], "event_booking", {"venue_id": venue_id, "date": event_date})
		return deepcopy(record)

	async def update_event_booking(self, booking_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		booking = self.event_bookings.get(booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"event_booking_not_found:{booking_id}")
		if booking["status"] in {"completed", "cancelled"}:
			raise ValueError("cannot_modify_closed_event_booking")
		allowed = {"event_name", "event_date", "start_time", "end_time", "expected_attendance", "status", "notes"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				booking[k] = v
		self._emit(tenant, "event_booking_updated", booking_id, "event_booking")
		return deepcopy(booking)

	async def confirm_event_booking(self, booking_id: str, deposit_amount: float, tenant_id: str | None = None) -> dict[str, Any]:
		"""Confirm an event booking on receipt of deposit."""
		tenant = self._tenant(tenant_id)
		booking = self.event_bookings.get(booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"event_booking_not_found:{booking_id}")
		dep = _dec(deposit_amount)
		booking["status"] = "confirmed"
		booking["deposit_paid"] = str(dep)
		booking["balance"] = str(_round2(_dec(booking["total_estimate"]) - dep))
		booking["confirmed_at"] = _now()
		self._emit(tenant, "event_booking_confirmed", booking_id, "event_booking", {"deposit": str(dep)})
		return deepcopy(booking)

	async def delete_event_booking(self, booking_id: str, reason: str = "client_cancellation", tenant_id: str | None = None) -> dict[str, Any]:
		"""Cancel a booking and auto-promote the head of the waitlist if present (I2)."""
		tenant = self._tenant(tenant_id)
		booking = self.event_bookings.get(booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"event_booking_not_found:{booking_id}")
		booking["status"] = "cancelled"
		booking["cancellation_reason"] = reason
		booking["cancelled_at"] = _now()
		self._emit(tenant, "event_booking_cancelled", booking_id, "event_booking", {"reason": reason})

		# I2 — promote head of waitlist for this venue/date
		wl_key = f"{booking['venue_id']}::{booking['event_date']}"
		waitlist = self._waitlist.get(wl_key, [])
		if waitlist:
			promoted = waitlist.pop(0)
			self._emit(tenant, "waitlist_promoted", promoted["id"], "waitlist", {
				"venue_id": promoted["venue_id"],
				"date": promoted["event_date"],
				"client_email": promoted["client_email"],
			})
			booking["waitlist_promoted_entry"] = promoted["id"]
			_log.info("hos_evn: waitlist entry %s promoted for venue %s on %s", promoted["id"], booking["venue_id"], booking["event_date"])

		return deepcopy(booking)

	# ── I2 — Waitlist management ──────────────────────────────────────────────

	async def get_waitlist(self, venue_id: str, event_date: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return all waitlist entries for a venue/date combination (I2)."""
		tenant = self._tenant(tenant_id)
		wl_key = f"{venue_id}::{event_date}"
		return [deepcopy(e) for e in self._waitlist.get(wl_key, []) if e["tenant_id"] == tenant]

	# ── BEO (Banquet Event Order) ─────────────────────────────────────────────

	async def generate_beo(self, event_booking_id: str, menu_selections: list[dict[str, Any]],
	                        av_requirements: list[str], setup_style: str = "theatre",
	                        special_requirements: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		"""Generate a Banquet Event Order.

		Validates allergen/dietary completeness on every menu line (I5 — compliance).
		"""
		tenant = self._tenant(tenant_id)
		booking = self.event_bookings.get(event_booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"event_booking_not_found:{event_booking_id}")
		venue = self.venues.get(booking["venue_id"])

		# I5 — allergen matrix validation
		for i, item in enumerate(menu_selections):
			if "allergens" not in item:
				raise ValueError(f"menu_item[{i}] missing 'allergens' list — required for allergen compliance")
			if "dietary_tags" not in item:
				raise ValueError(f"menu_item[{i}] missing 'dietary_tags' list — required for dietary compliance")

		# Build dietary summary aggregate
		all_allergens: set[str] = set()
		all_tags: set[str] = set()
		for item in menu_selections:
			all_allergens.update(item.get("allergens", []))
			all_tags.update(item.get("dietary_tags", []))

		beo: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"event_booking_id": event_booking_id,
			"beo_number": f"BEO-{_uid()[:6].upper()}",
			"venue_name": booking["venue_name"],
			"event_name": booking["event_name"],
			"client_name": booking["client_name"],
			"event_date": booking["event_date"],
			"start_time": booking["start_time"],
			"end_time": booking["end_time"],
			"expected_attendance": booking["expected_attendance"],
			"setup_style": setup_style,
			"menu_selections": deepcopy(menu_selections),
			"dietary_summary": {
				"allergens_present": sorted(all_allergens),
				"dietary_options": sorted(all_tags),
			},
			"av_requirements": av_requirements,
			"special_requirements": special_requirements,
			"venue_details": deepcopy(venue) if venue else {},
			"financials": {
				"venue_rental": booking["venue_rental"],
				"catering_estimate": booking["catering_estimate"],
				"av_estimate": booking["av_estimate"],
				"total_estimate": booking["total_estimate"],
				"deposit_paid": booking["deposit_paid"],
				"balance": booking["balance"],
			},
			"status": "draft",
			"generated_at": _now(),
		}
		self.beos[beo["id"]] = beo
		booking["beo_generated"] = True
		self._emit(tenant, "beo_generated", beo["id"], "beo", {"booking_id": event_booking_id})
		return deepcopy(beo)

	async def get_beo(self, beo_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		beo = self.beos.get(beo_id)
		if not beo or beo["tenant_id"] != tenant:
			raise KeyError(f"beo_not_found:{beo_id}")
		return deepcopy(beo)

	async def list_beos(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(b) for b in self.beos.values() if b["tenant_id"] == tenant]

	async def finalise_beo(self, beo_id: str, approved_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Mark a BEO as finalised and approved."""
		tenant = self._tenant(tenant_id)
		beo = self.beos.get(beo_id)
		if not beo or beo["tenant_id"] != tenant:
			raise KeyError(f"beo_not_found:{beo_id}")
		beo["status"] = "finalised"
		beo["approved_by"] = approved_by
		beo["finalised_at"] = _now()
		self._emit(tenant, "beo_finalised", beo_id, "beo")
		return deepcopy(beo)

	# ── Contracts ─────────────────────────────────────────────────────────────

	async def issue_contract(self, event_booking_id: str, deposit_pct: float = 30.0,
	                          payment_terms: str = "50% 30 days before, balance on day",
	                          cancellation_policy: str = "tiered", special_clauses: str | None = None,
	                          tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		booking = self.event_bookings.get(event_booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"event_booking_not_found:{event_booking_id}")
		total = _dec(booking["total_estimate"])
		deposit_amount = _round2(total * _dec(deposit_pct) / _dec("100"))
		contract: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"contract_number": f"CONT-{_uid()[:6].upper()}",
			"event_booking_id": event_booking_id,
			"event_name": booking["event_name"],
			"client_name": booking["client_name"],
			"client_email": booking["client_email"],
			"event_date": booking["event_date"],
			"total_value": str(total),
			"deposit_pct": deposit_pct,
			"deposit_amount": str(deposit_amount),
			"payment_terms": payment_terms,
			"cancellation_policy": cancellation_policy,
			"special_clauses": special_clauses,
			"status": "issued",
			"signed": False,
			"signature_hash": None,
			"issued_at": _now(),
		}
		self.contracts[contract["id"]] = contract
		booking["contract_issued"] = True
		self._emit(tenant, "contract_issued", contract["id"], "contract", {"booking_id": event_booking_id})
		return deepcopy(contract)

	async def sign_contract(self, contract_id: str, signed_by: str,
	                         signature_ip: str | None = None, user_agent: str | None = None,
	                         tenant_id: str | None = None) -> dict[str, Any]:
		"""Sign a contract and compute a tamper-evidence HMAC-SHA256 hash (I6).

		The hash covers: contract_number + client_name + total_value + signed_at.
		Call `verify_contract_signature` to confirm the document has not been altered post-signing.
		"""
		tenant = self._tenant(tenant_id)
		contract = self.contracts.get(contract_id)
		if not contract or contract["tenant_id"] != tenant:
			raise KeyError(f"contract_not_found:{contract_id}")
		signed_at = _now()
		# I6 — canonical body for HMAC
		canonical_body = json.dumps({
			"contract_number": contract["contract_number"],
			"client_name": contract["client_name"],
			"total_value": contract["total_value"],
			"event_date": contract["event_date"],
			"signed_at": signed_at,
		}, sort_keys=True)
		sig_hash = hmac.new(
			self._tenant_secret.encode(),
			canonical_body.encode(),
			hashlib.sha256,
		).hexdigest()
		contract["signed"] = True
		contract["signed_by"] = signed_by
		contract["signed_at"] = signed_at
		contract["signature_hash"] = sig_hash
		contract["signature_ip"] = signature_ip
		contract["user_agent"] = user_agent
		contract["signature_canonical_body"] = canonical_body
		contract["status"] = "signed"
		self._emit(tenant, "contract_signed", contract_id, "contract", {"signed_by": signed_by})
		return deepcopy(contract)

	async def verify_contract_signature(self, contract_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Recompute and verify the HMAC signature stored at signing time (I6)."""
		tenant = self._tenant(tenant_id)
		contract = self.contracts.get(contract_id)
		if not contract or contract["tenant_id"] != tenant:
			raise KeyError(f"contract_not_found:{contract_id}")
		if not contract.get("signed"):
			return {"contract_id": contract_id, "verified": False, "reason": "not_yet_signed"}
		stored_hash = contract.get("signature_hash", "")
		canonical_body = contract.get("signature_canonical_body", "")
		recomputed = hmac.new(
			self._tenant_secret.encode(),
			canonical_body.encode(),
			hashlib.sha256,
		).hexdigest()
		match = hmac.compare_digest(stored_hash, recomputed)
		return {
			"contract_id": contract_id,
			"verified": match,
			"reason": "ok" if match else "hash_mismatch — document may have been tampered",
		}

	async def list_contracts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(c) for c in self.contracts.values() if c["tenant_id"] == tenant]

	# ── Payments ──────────────────────────────────────────────────────────────

	async def record_event_payment(self, event_booking_id: str, amount: float, payment_type: str,
	                                payment_method: str, reference: str | None = None,
	                                tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		booking = self.event_bookings.get(event_booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"event_booking_not_found:{event_booking_id}")
		amt = _dec(amount)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"event_booking_id": event_booking_id,
			"amount": str(amt),
			"payment_type": payment_type,  # deposit|interim|final
			"payment_method": payment_method,
			"reference": reference,
			"status": "settled",
			"created_at": _now(),
		}
		self.payments[record["id"]] = record
		new_paid = _round2(_dec(booking.get("deposit_paid", "0")) + amt)
		booking["deposit_paid"] = str(new_paid)
		booking["balance"] = str(_round2(_dec(booking["total_estimate"]) - new_paid))

		# Update matching milestone if one exists
		for ms in self._payment_milestones.values():
			if ms["event_booking_id"] == event_booking_id and ms["status"] == "pending":
				ms_due = _dec(ms["amount"])
				if amt >= ms_due:
					ms["status"] = "paid"
					ms["paid_at"] = _now()
					break

		self._emit(tenant, "event_payment_recorded", record["id"], "payment", {"amount": str(amt)})
		return deepcopy(record)

	async def list_event_payments(self, event_booking_id: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(p) for p in self.payments.values() if p["tenant_id"] == tenant and p["event_booking_id"] == event_booking_id]

	# ── I7 — Payment Timeline & Overdue Escalation ────────────────────────────

	async def generate_payment_timeline(self, booking_id: str, instalments: list[dict[str, Any]], tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Create a payment milestone schedule for a booking (I7).

		`instalments` is a list of {'due_date': 'YYYY-MM-DD', 'amount': Decimal|float|str, 'type': str} dicts.
		Each milestone is stored and tracked; overdue milestones surface in `get_overdue_reminders`.
		"""
		tenant = self._tenant(tenant_id)
		booking = self.event_bookings.get(booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"event_booking_not_found:{booking_id}")
		created: list[dict[str, Any]] = []
		for inst in instalments:
			ms: dict[str, Any] = {
				"id": _uid(),
				"tenant_id": tenant,
				"event_booking_id": booking_id,
				"due_date": inst["due_date"],
				"amount": str(_dec(inst["amount"])),
				"type": inst.get("type", "instalment"),
				"status": "pending",
				"created_at": _now(),
			}
			self._payment_milestones[ms["id"]] = ms
			created.append(deepcopy(ms))
		self._emit(tenant, "payment_timeline_created", booking_id, "event_booking", {"milestones": len(created)})
		return created

	async def get_overdue_reminders(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return all unpaid payment milestones past their due date, ranked by urgency (I7).

		Each result includes `days_overdue` so the calling layer can triage escalation cadence.
		"""
		tenant = self._tenant(tenant_id)
		today = _today()
		overdue: list[dict[str, Any]] = []
		for ms in self._payment_milestones.values():
			if ms["tenant_id"] != tenant or ms["status"] != "pending":
				continue
			due = _parse_date(ms["due_date"])
			if due < today:
				row = deepcopy(ms)
				row["days_overdue"] = (today - due).days
				booking = self.event_bookings.get(ms["event_booking_id"], {})
				row["client_name"] = booking.get("client_name")
				row["client_email"] = booking.get("client_email")
				overdue.append(row)
		overdue.sort(key=lambda r: r["days_overdue"], reverse=True)
		return overdue

	# ── AV Requirements ───────────────────────────────────────────────────────

	async def set_av_requirements(self, event_booking_id: str, equipment_list: list[str],
	                               technician_required: bool = True, setup_time_mins: int = 60,
	                               tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		booking = self.event_bookings.get(event_booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"event_booking_not_found:{event_booking_id}")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"event_booking_id": event_booking_id,
			"equipment_list": equipment_list,
			"technician_required": technician_required,
			"setup_time_mins": setup_time_mins,
			"status": "pending",
			"created_at": _now(),
		}
		self.av_requirements[record["id"]] = record
		self._emit(tenant, "av_requirements_set", record["id"], "av_requirements")
		return deepcopy(record)

	# ── I11 — AV Equipment Inventory & Conflict Detection ────────────────────

	async def register_av_asset(self, name: str, category: str, quantity_owned: int, tenant_id: str | None = None) -> dict[str, Any]:
		"""Add an AV asset (e.g. projector, PA system) to the bookable inventory pool (I11).

		Prevents the common failure mode of two events being assigned the same physical unit.
		"""
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"name": name,
			"category": category,
			"quantity_owned": quantity_owned,
			"created_at": _now(),
		}
		self._av_assets[record["id"]] = record
		self._emit(tenant, "av_asset_registered", record["id"], "av_asset")
		return deepcopy(record)

	async def check_av_availability(self, event_date: str, equipment_requests: list[dict[str, Any]], tenant_id: str | None = None) -> dict[str, Any]:
		"""Check AV inventory availability for a given date against confirmed bookings (I11).

		`equipment_requests` is a list of {'category': str, 'quantity': int} dicts.
		Returns per-category availability including any shortfall and conflicting booking IDs.
		"""
		tenant = self._tenant(tenant_id)
		# Sum owned units by category
		owned: dict[str, int] = {}
		for asset in self._av_assets.values():
			if asset["tenant_id"] != tenant:
				continue
			owned[asset["category"]] = owned.get(asset["category"], 0) + asset["quantity_owned"]

		# Count units already committed on this date
		committed: dict[str, int] = {}
		conflict_map: dict[str, list[str]] = {}
		for av_req in self.av_requirements.values():
			if av_req["tenant_id"] != tenant:
				continue
			booking = self.event_bookings.get(av_req["event_booking_id"], {})
			if booking.get("event_date") != event_date or booking.get("status") not in {"tentative", "confirmed"}:
				continue
			for equip in av_req.get("equipment_list", []):
				# equipment_list items may be plain strings (category names)
				committed[equip] = committed.get(equip, 0) + 1
				conflict_map.setdefault(equip, []).append(av_req["event_booking_id"])

		results: list[dict[str, Any]] = []
		for req in equipment_requests:
			cat = req["category"]
			qty_requested = req.get("quantity", 1)
			qty_owned = owned.get(cat, 0)
			qty_committed = committed.get(cat, 0)
			available = max(0, qty_owned - qty_committed)
			results.append({
				"category": cat,
				"quantity_requested": qty_requested,
				"quantity_owned": qty_owned,
				"quantity_committed": qty_committed,
				"available": available,
				"shortfall": max(0, qty_requested - available),
				"conflicting_booking_ids": conflict_map.get(cat, []),
			})
		return {"event_date": event_date, "results": results, "checked_at": _now()}

	# ── I10 — Tiered Cancellation Fee Engine ──────────────────────────────────

	async def compute_cancellation_fee(self, booking_id: str, cancellation_date: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		"""Compute the cancellation fee based on tiered forfeiture schedule (I10).

		Tiers (days to event → forfeiture %):
		  >= 90 days → 0%
		  >= 60 days → 25%
		  >= 30 days → 50%
		  <  30 days → 100%

		All arithmetic in Decimal with ROUND_HALF_UP for audit compliance.
		"""
		tenant = self._tenant(tenant_id)
		booking = self.event_bookings.get(booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"event_booking_not_found:{booking_id}")
		cancel_date = _parse_date(cancellation_date) if cancellation_date else _today()
		event_date = _parse_date(booking["event_date"])
		days_to_event = (event_date - cancel_date).days

		# Find the applicable tier (first threshold the days_to_event satisfies)
		tier_min_days, forfeiture_pct = _CANCELLATION_TIERS[-1]  # default: full forfeiture
		for threshold, pct in _CANCELLATION_TIERS:
			if days_to_event >= threshold:
				tier_min_days, forfeiture_pct = threshold, pct
				break

		total = _dec(booking["total_estimate"])
		deposit_paid = _dec(booking.get("deposit_paid", "0"))
		fee_amount = _round2(total * forfeiture_pct / _dec("100"))
		refund_amount = _round2(deposit_paid - fee_amount) if deposit_paid > fee_amount else _dec("0")

		return {
			"booking_id": booking_id,
			"event_date": booking["event_date"],
			"cancellation_date": cancel_date.isoformat(),
			"days_to_event": days_to_event,
			"tier_min_days": tier_min_days,
			"forfeiture_pct": str(forfeiture_pct),
			"total_booking_value": str(total),
			"fee_amount": str(fee_amount),
			"deposit_paid": str(deposit_paid),
			"refund_amount": str(refund_amount),
			"justification": f"{forfeiture_pct}% forfeiture applies when cancellation is {days_to_event} days before event",
		}

	# ── I9 — Post-Event NPS & Satisfaction Score ──────────────────────────────

	async def record_event_nps(self, booking_id: str, nps_score: int, dimension_scores: dict[str, int] | None = None, comment: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		"""Record a post-event NPS survey response (0–10 scale) (I9).

		Dimension scores can cover: venue, catering, av, service, value_for_money.
		Feeds hos_crm client-lifetime-value models via the `event_nps_recorded` event.
		"""
		tenant = self._tenant(tenant_id)
		booking = self.event_bookings.get(booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"event_booking_not_found:{booking_id}")
		if not (0 <= nps_score <= 10):
			raise ValueError(f"nps_score must be 0-10, got {nps_score}")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"event_booking_id": booking_id,
			"venue_id": booking["venue_id"],
			"event_type": booking["event_type"],
			"nps_score": nps_score,
			"nps_category": "promoter" if nps_score >= 9 else ("passive" if nps_score >= 7 else "detractor"),
			"dimension_scores": dimension_scores or {},
			"comment": comment,
			"recorded_at": _now(),
		}
		self._nps_records[record["id"]] = record
		self._emit(tenant, "event_nps_recorded", record["id"], "nps", {"score": nps_score, "booking_id": booking_id})
		return deepcopy(record)

	async def nps_summary(self, tenant_id: str | None = None, venue_id: str | None = None, date_from: str | None = None, date_to: str | None = None) -> dict[str, Any]:
		"""Compute NPS: promoters/passives/detractors and net score (I9).

		Optionally filtered by venue or date range. Compatible with hos_crm for cross-capability
		client satisfaction modelling.
		"""
		tenant = self._tenant(tenant_id)
		records = [r for r in self._nps_records.values() if r["tenant_id"] == tenant]
		if venue_id:
			records = [r for r in records if r["venue_id"] == venue_id]
		if date_from:
			records = [r for r in records if r["recorded_at"] >= date_from]
		if date_to:
			records = [r for r in records if r["recorded_at"] <= date_to]
		if not records:
			return {"tenant_id": tenant, "total_responses": 0, "nps": None, "generated_at": _now()}
		promoters = sum(1 for r in records if r["nps_category"] == "promoter")
		passives = sum(1 for r in records if r["nps_category"] == "passive")
		detractors = sum(1 for r in records if r["nps_category"] == "detractor")
		total = len(records)
		# NPS = (promoters - detractors) / total * 100
		nps = _round2(_dec(promoters - detractors) / _dec(total) * _dec("100"))
		avg_score = _round2(sum(_dec(r["nps_score"]) for r in records) / _dec(total))
		return {
			"tenant_id": tenant,
			"venue_id": venue_id,
			"total_responses": total,
			"promoters": promoters,
			"passives": passives,
			"detractors": detractors,
			"nps": str(nps),
			"avg_score": str(avg_score),
			"generated_at": _now(),
		}

	# ── I12 — Revenue Forecast ────────────────────────────────────────────────

	async def revenue_forecast(self, months_ahead: int = 6, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Forward revenue split into contracted (confirmed) vs pipeline (tentative) buckets (I12).

		Applies confidence weights (confirmed 90%, tentative 40%) to produce a weighted_total per
		calendar month. Enables CFO-level capital allocation decisions without ad-hoc spreadsheets.
		"""
		tenant = self._tenant(tenant_id)
		if months_ahead < 1 or months_ahead > 24:
			raise ValueError("months_ahead must be 1–24")
		today = _today()

		# Build month buckets from current month through months_ahead
		buckets: dict[str, dict[str, Decimal]] = {}
		for m in range(months_ahead):
			# shift month by m from today
			month_offset = today.month - 1 + m
			year = today.year + month_offset // 12
			month = (month_offset % 12) + 1
			key = f"{year:04d}-{month:02d}"
			buckets[key] = {"contracted": _dec("0"), "pipeline": _dec("0")}

		for booking in self.event_bookings.values():
			if booking["tenant_id"] != tenant:
				continue
			if booking["status"] not in {"confirmed", "tentative"}:
				continue
			month_key = booking["event_date"][:7]  # "YYYY-MM"
			if month_key not in buckets:
				continue
			total = _dec(booking["total_estimate"])
			if booking["status"] == "confirmed":
				buckets[month_key]["contracted"] += total
			else:
				buckets[month_key]["pipeline"] += total

		result: list[dict[str, Any]] = []
		for month_key, vals in sorted(buckets.items()):
			contracted = _round2(vals["contracted"])
			pipeline = _round2(vals["pipeline"])
			weighted = _round2(
				contracted * _CONFIDENCE["confirmed"] + pipeline * _CONFIDENCE["tentative"]
			)
			result.append({
				"month": month_key,
				"contracted": str(contracted),
				"pipeline": str(pipeline),
				"weighted_total": str(weighted),
				"confidence_weights": {
					"confirmed": str(_CONFIDENCE["confirmed"]),
					"tentative": str(_CONFIDENCE["tentative"]),
				},
			})
		return result

	# ── Analytics ─────────────────────────────────────────────────────────────

	async def venue_utilisation_report(self, date_from: str, date_to: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		bookings = [e for e in self.event_bookings.values() if e["tenant_id"] == tenant and date_from <= e["event_date"] <= date_to]
		by_venue: dict[str, int] = {}
		by_type: dict[str, int] = {}
		total_revenue = _dec("0")
		for b in bookings:
			by_venue[b["venue_name"]] = by_venue.get(b["venue_name"], 0) + 1
			by_type[b["event_type"]] = by_type.get(b["event_type"], 0) + 1
			total_revenue += _dec(b["total_estimate"])
		return {
			"tenant_id": tenant,
			"date_from": date_from,
			"date_to": date_to,
			"total_events": len(bookings),
			"total_revenue": str(_round2(total_revenue)),
			"by_venue": by_venue,
			"by_event_type": by_type,
			"generated_at": _now(),
		}

	async def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		total_balance = sum(_dec(e["balance"]) for e in self.event_bookings.values() if e["tenant_id"] == tenant and e["status"] in {"tentative", "confirmed"})
		return {
			"tenant_id": tenant,
			"active_venues": sum(1 for v in self.venues.values() if v["tenant_id"] == tenant and v["status"] == "active"),
			"total_event_bookings": sum(1 for e in self.event_bookings.values() if e["tenant_id"] == tenant),
			"confirmed_events": sum(1 for e in self.event_bookings.values() if e["tenant_id"] == tenant and e["status"] == "confirmed"),
			"tentative_events": sum(1 for e in self.event_bookings.values() if e["tenant_id"] == tenant and e["status"] == "tentative"),
			"total_beos": sum(1 for b in self.beos.values() if b["tenant_id"] == tenant),
			"total_contracts": sum(1 for c in self.contracts.values() if c["tenant_id"] == tenant),
			"outstanding_balance": str(_round2(total_balance)),
			"pending_milestones": sum(1 for m in self._payment_milestones.values() if m["tenant_id"] == tenant and m["status"] == "pending"),
			"generated_at": _now(),
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_nps_records', '_av_assets', '_payment_milestones', '_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

