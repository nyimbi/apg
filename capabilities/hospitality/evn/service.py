"""Events & Venue Management service — event booking, venue configuration, catering BEO, AV requirements, billing, contracts."""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)


def _uid() -> str:
	return uuid4().hex[:12]


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# Per-person catering estimate by meal type
_CATERING_RATE_PER_HEAD = {
	"conference": 3500.0,
	"wedding": 8000.0,
	"gala": 6000.0,
	"birthday": 4000.0,
	"product_launch": 3000.0,
	"training": 2500.0,
	"other": 3000.0,
}

# AV flat rate estimates
_AV_RATE = {
	"basic": 25000.0,
	"standard": 50000.0,
	"full_production": 150000.0,
}


class EVNService:
	"""Events & Venue Management service."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.venues: dict[str, dict[str, Any]] = {}
		self.event_bookings: dict[str, dict[str, Any]] = {}
		self.beos: dict[str, dict[str, Any]] = {}
		self.contracts: dict[str, dict[str, Any]] = {}
		self.payments: dict[str, dict[str, Any]] = {}
		self.catering_orders: dict[str, dict[str, Any]] = {}
		self.av_requirements: dict[str, dict[str, Any]] = {}
		self.setup_configs: dict[str, dict[str, Any]] = {}
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
			"version": "1.0.0",
			"description": "Event booking, venue configuration, catering BEO, AV requirements, billing, contracts",
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
			"rental_rate_per_day": rental_rate_per_day,
			"av_included": av_included,
			"catering_allowed": catering_allowed,
			"notes": notes,
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
		allowed = {"name", "capacity_seated", "rental_rate_per_day", "status", "notes", "av_included"}
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
		# Check no upcoming bookings
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
	                                client_phone: str | None = None, notes: str | None = None,
	                                tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		venue = self.venues.get(venue_id)
		if not venue or venue["tenant_id"] != tenant:
			raise KeyError(f"venue_not_found:{venue_id}")
		if expected_attendance > venue["capacity_seated"]:
			raise ValueError(f"attendance_exceeds_venue_capacity:{expected_attendance}>{venue['capacity_seated']}")
		# Check venue availability
		for eb in self.event_bookings.values():
			if eb["tenant_id"] == tenant and eb["venue_id"] == venue_id and eb["event_date"] == event_date and eb["status"] in {"tentative", "confirmed"}:
				raise ValueError(f"venue_already_booked_on:{event_date}")
		# Compute estimates
		venue_rental = venue["rental_rate_per_day"]
		catering_estimate = _CATERING_RATE_PER_HEAD.get(event_type, 3000.0) * expected_attendance if catering_required else 0.0
		av_estimate = _AV_RATE["standard"] if av_required and not venue["av_included"] else 0.0
		decoration_estimate = 20000.0 if decoration_required else 0.0
		total_estimate = venue_rental + catering_estimate + av_estimate + decoration_estimate
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
			"expected_attendance": expected_attendance,
			"catering_required": catering_required,
			"av_required": av_required,
			"decoration_required": decoration_required,
			"venue_rental": venue_rental,
			"catering_estimate": round(catering_estimate, 2),
			"av_estimate": av_estimate,
			"decoration_estimate": decoration_estimate,
			"total_estimate": round(total_estimate, 2),
			"deposit_paid": 0.0,
			"balance": round(total_estimate, 2),
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
		booking["status"] = "confirmed"
		booking["deposit_paid"] = deposit_amount
		booking["balance"] = round(booking["total_estimate"] - deposit_amount, 2)
		booking["confirmed_at"] = _now()
		self._emit(tenant, "event_booking_confirmed", booking_id, "event_booking", {"deposit": deposit_amount})
		return deepcopy(booking)

	async def delete_event_booking(self, booking_id: str, reason: str = "client_cancellation", tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		booking = self.event_bookings.get(booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"event_booking_not_found:{booking_id}")
		booking["status"] = "cancelled"
		booking["cancellation_reason"] = reason
		booking["cancelled_at"] = _now()
		self._emit(tenant, "event_booking_cancelled", booking_id, "event_booking", {"reason": reason})
		return deepcopy(booking)

	# ── BEO (Banquet Event Order) ─────────────────────────────────────────────

	async def generate_beo(self, event_booking_id: str, menu_selections: list[dict[str, Any]],
	                        av_requirements: list[str], setup_style: str = "theatre",
	                        special_requirements: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		"""Generate a Banquet Event Order document."""
		tenant = self._tenant(tenant_id)
		booking = self.event_bookings.get(event_booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"event_booking_not_found:{event_booking_id}")
		venue = self.venues.get(booking["venue_id"])
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
	                          cancellation_policy: str = "standard", special_clauses: str | None = None,
	                          tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		booking = self.event_bookings.get(event_booking_id)
		if not booking or booking["tenant_id"] != tenant:
			raise KeyError(f"event_booking_not_found:{event_booking_id}")
		deposit_amount = round(booking["total_estimate"] * deposit_pct / 100, 2)
		contract: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"contract_number": f"CONT-{_uid()[:6].upper()}",
			"event_booking_id": event_booking_id,
			"event_name": booking["event_name"],
			"client_name": booking["client_name"],
			"client_email": booking["client_email"],
			"event_date": booking["event_date"],
			"total_value": booking["total_estimate"],
			"deposit_pct": deposit_pct,
			"deposit_amount": deposit_amount,
			"payment_terms": payment_terms,
			"cancellation_policy": cancellation_policy,
			"special_clauses": special_clauses,
			"status": "issued",
			"signed": False,
			"issued_at": _now(),
		}
		self.contracts[contract["id"]] = contract
		booking["contract_issued"] = True
		self._emit(tenant, "contract_issued", contract["id"], "contract", {"booking_id": event_booking_id})
		return deepcopy(contract)

	async def sign_contract(self, contract_id: str, signed_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		contract = self.contracts.get(contract_id)
		if not contract or contract["tenant_id"] != tenant:
			raise KeyError(f"contract_not_found:{contract_id}")
		contract["signed"] = True
		contract["signed_by"] = signed_by
		contract["signed_at"] = _now()
		contract["status"] = "signed"
		self._emit(tenant, "contract_signed", contract_id, "contract", {"signed_by": signed_by})
		return deepcopy(contract)

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
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"event_booking_id": event_booking_id,
			"amount": amount,
			"payment_type": payment_type,  # deposit|interim|final
			"payment_method": payment_method,
			"reference": reference,
			"status": "settled",
			"created_at": _now(),
		}
		self.payments[record["id"]] = record
		booking["deposit_paid"] = booking.get("deposit_paid", 0.0) + amount
		booking["balance"] = round(booking["total_estimate"] - booking["deposit_paid"], 2)
		self._emit(tenant, "event_payment_recorded", record["id"], "payment", {"amount": amount})
		return deepcopy(record)

	async def list_event_payments(self, event_booking_id: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(p) for p in self.payments.values() if p["tenant_id"] == tenant and p["event_booking_id"] == event_booking_id]

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

	# ── Analytics ─────────────────────────────────────────────────────────────

	async def venue_utilisation_report(self, date_from: str, date_to: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		bookings = [e for e in self.event_bookings.values() if e["tenant_id"] == tenant and date_from <= e["event_date"] <= date_to]
		by_venue: dict[str, int] = {}
		by_type: dict[str, int] = {}
		for b in bookings:
			by_venue[b["venue_name"]] = by_venue.get(b["venue_name"], 0) + 1
			by_type[b["event_type"]] = by_type.get(b["event_type"], 0) + 1
		return {
			"tenant_id": tenant,
			"date_from": date_from,
			"date_to": date_to,
			"total_events": len(bookings),
			"total_revenue": sum(b["total_estimate"] for b in bookings),
			"by_venue": by_venue,
			"by_event_type": by_type,
			"generated_at": _now(),
		}

	async def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return {
			"tenant_id": tenant,
			"active_venues": sum(1 for v in self.venues.values() if v["tenant_id"] == tenant and v["status"] == "active"),
			"total_event_bookings": sum(1 for e in self.event_bookings.values() if e["tenant_id"] == tenant),
			"confirmed_events": sum(1 for e in self.event_bookings.values() if e["tenant_id"] == tenant and e["status"] == "confirmed"),
			"tentative_events": sum(1 for e in self.event_bookings.values() if e["tenant_id"] == tenant and e["status"] == "tentative"),
			"total_beos": sum(1 for b in self.beos.values() if b["tenant_id"] == tenant),
			"total_contracts": sum(1 for c in self.contracts.values() if c["tenant_id"] == tenant),
			"generated_at": _now(),
		}
