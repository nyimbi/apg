"""Spa & Activities Management service — treatment booking, therapist scheduling, inventory, retail, memberships."""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)


def _uid() -> str:
	return uuid4().hex[:12]


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _add_minutes(time_str: str, minutes: int) -> str:
	"""Add minutes to HH:MM time string."""
	try:
		h, m = map(int, time_str.split(":"))
		total = h * 60 + m + minutes
		return f"{total // 60:02d}:{total % 60:02d}"
	except Exception:
		return time_str


class SPAService:
	"""Spa & Activities Management service."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.treatments: dict[str, dict[str, Any]] = {}
		self.therapists: dict[str, dict[str, Any]] = {}
		self.appointments: dict[str, dict[str, Any]] = {}
		self.memberships: dict[str, dict[str, Any]] = {}
		self.retail_items: dict[str, dict[str, Any]] = {}
		self.retail_sales: dict[str, dict[str, Any]] = {}
		self.therapist_schedules: dict[str, dict[str, Any]] = {}
		self.activities: dict[str, dict[str, Any]] = {}
		self.activity_bookings: dict[str, dict[str, Any]] = {}
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
			"service": "hos_spa",
			"status": "healthy",
			"active_treatments": sum(1 for t in self.treatments.values() if t["is_active"]),
			"active_therapists": sum(1 for t in self.therapists.values() if t["status"] == "active"),
			"today_appointments": sum(1 for a in self.appointments.values() if a["status"] in {"confirmed", "in_progress"}),
			"checked_at": _now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": "hos_spa",
			"name": "Spa & Activities Management",
			"domain": "hospitality",
			"version": "1.0.0",
			"description": "Treatment booking, therapist scheduling, inventory, retail, membership management",
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Treatments ────────────────────────────────────────────────────────────

	async def list_treatments(self, tenant_id: str | None = None, category: str | None = None, active_only: bool = False) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(t) for t in self.treatments.values() if t["tenant_id"] == tenant]
		if category:
			items = [t for t in items if t["category"] == category]
		if active_only:
			items = [t for t in items if t["is_active"]]
		return items

	async def get_treatment(self, treatment_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		t = self.treatments.get(treatment_id)
		if not t or t["tenant_id"] != tenant:
			raise KeyError(f"treatment_not_found:{treatment_id}")
		return deepcopy(t)

	async def create_treatment(self, name: str, category: str, duration_mins: int, price: float,
	                            therapist_required: int = 1, description: str | None = None,
	                            is_active: bool = True, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"name": name,
			"category": category,
			"duration_mins": duration_mins,
			"price": price,
			"therapist_required": therapist_required,
			"description": description,
			"is_active": is_active,
			"booking_count": 0,
			"created_at": _now(),
		}
		self.treatments[record["id"]] = record
		self._emit(tenant, "treatment_created", record["id"], "treatment", {"category": category})
		return deepcopy(record)

	async def update_treatment(self, treatment_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		t = self.treatments.get(treatment_id)
		if not t or t["tenant_id"] != tenant:
			raise KeyError(f"treatment_not_found:{treatment_id}")
		allowed = {"name", "price", "duration_mins", "is_active", "description", "therapist_required"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				t[k] = v
		self._emit(tenant, "treatment_updated", treatment_id, "treatment")
		return deepcopy(t)

	async def delete_treatment(self, treatment_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		t = self.treatments.get(treatment_id)
		if not t or t["tenant_id"] != tenant:
			raise KeyError(f"treatment_not_found:{treatment_id}")
		t["is_active"] = False
		return {"deactivated": True, "treatment_id": treatment_id}

	# ── Therapists ────────────────────────────────────────────────────────────

	async def list_therapists(self, tenant_id: str | None = None, specialisation: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		therapists = [deepcopy(t) for t in self.therapists.values() if t["tenant_id"] == tenant and t["status"] == "active"]
		if specialisation:
			therapists = [t for t in therapists if specialisation in t["specialisations"]]
		return therapists

	async def get_therapist(self, therapist_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		t = self.therapists.get(therapist_id)
		if not t or t["tenant_id"] != tenant:
			raise KeyError(f"therapist_not_found:{therapist_id}")
		return deepcopy(t)

	async def create_therapist(self, first_name: str, last_name: str, specialisations: list[str] | None = None,
	                            employment_type: str = "full_time", phone: str | None = None,
	                            email: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"first_name": first_name,
			"last_name": last_name,
			"full_name": f"{first_name} {last_name}",
			"specialisations": specialisations or [],
			"employment_type": employment_type,
			"phone": phone,
			"email": email,
			"appointment_count": 0,
			"status": "active",
			"created_at": _now(),
		}
		self.therapists[record["id"]] = record
		self._emit(tenant, "therapist_created", record["id"], "therapist")
		return deepcopy(record)

	async def update_therapist(self, therapist_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		t = self.therapists.get(therapist_id)
		if not t or t["tenant_id"] != tenant:
			raise KeyError(f"therapist_not_found:{therapist_id}")
		for k, v in updates.items():
			if v is not None:
				t[k] = v
		return deepcopy(t)

	async def delete_therapist(self, therapist_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		t = self.therapists.get(therapist_id)
		if not t or t["tenant_id"] != tenant:
			raise KeyError(f"therapist_not_found:{therapist_id}")
		t["status"] = "deactivated"
		return {"deactivated": True, "therapist_id": therapist_id}

	async def get_therapist_schedule(self, therapist_id: str, date: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Get all appointments for a therapist on a given date."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(a) for a in self.appointments.values()
		        if a["tenant_id"] == tenant and a["therapist_id"] == therapist_id
		        and a["appointment_date"] == date and a["status"] != "cancelled"]

	def _find_available_therapist(self, tenant: str, treatment_id: str, date: str, start_time: str) -> str | None:
		"""Find an available therapist for a given treatment and timeslot."""
		treatment = self.treatments.get(treatment_id)
		if not treatment:
			return None
		end_time = _add_minutes(start_time, treatment["duration_mins"])
		for therapist in self.therapists.values():
			if therapist["tenant_id"] != tenant or therapist["status"] != "active":
				continue
			# Check for conflicting appointments
			conflict = False
			for appt in self.appointments.values():
				if (appt["tenant_id"] == tenant and appt["therapist_id"] == therapist["id"]
				        and appt["appointment_date"] == date and appt["status"] != "cancelled"):
					# Overlap check: existing.start < new.end AND existing.end > new.start
					if appt["start_time"] < end_time and appt["end_time"] > start_time:
						conflict = True
						break
			if not conflict:
				return therapist["id"]
		return None

	# ── Appointments ──────────────────────────────────────────────────────────

	async def list_appointments(self, tenant_id: str | None = None, date: str | None = None,
	                             therapist_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(a) for a in self.appointments.values() if a["tenant_id"] == tenant]
		if date:
			items = [a for a in items if a["appointment_date"] == date]
		if therapist_id:
			items = [a for a in items if a["therapist_id"] == therapist_id]
		if status:
			items = [a for a in items if a["status"] == status]
		return sorted(items, key=lambda x: (x["appointment_date"], x["start_time"]))

	async def get_appointment(self, appointment_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		appt = self.appointments.get(appointment_id)
		if not appt or appt["tenant_id"] != tenant:
			raise KeyError(f"appointment_not_found:{appointment_id}")
		return deepcopy(appt)

	async def create_appointment(self, guest_name: str, guest_email: str, treatment_id: str,
	                              appointment_date: str, start_time: str,
	                              therapist_id: str | None = None, reservation_id: str | None = None,
	                              special_notes: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		treatment = self.treatments.get(treatment_id)
		if not treatment or treatment["tenant_id"] != tenant:
			raise KeyError(f"treatment_not_found:{treatment_id}")
		if not treatment["is_active"]:
			raise ValueError(f"treatment_not_active:{treatment_id}")
		# Auto-assign therapist if not specified
		assigned_therapist_id = therapist_id
		if not assigned_therapist_id:
			assigned_therapist_id = self._find_available_therapist(tenant, treatment_id, appointment_date, start_time)
		if not assigned_therapist_id:
			raise ValueError("no_therapist_available_for_timeslot")
		end_time = _add_minutes(start_time, treatment["duration_mins"])
		# Verify therapist exists
		therapist = self.therapists.get(assigned_therapist_id)
		if not therapist or therapist["tenant_id"] != tenant:
			raise KeyError(f"therapist_not_found:{assigned_therapist_id}")
		# Apply membership discount
		discount_pct = 0.0
		for membership in self.memberships.values():
			if membership["tenant_id"] == tenant and membership["guest_email"] == guest_email and membership["status"] == "active":
				discount_pct = membership["discount_pct"]
				break
		price = round(treatment["price"] * (1 - discount_pct / 100), 2)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"guest_name": guest_name,
			"guest_email": guest_email,
			"treatment_id": treatment_id,
			"treatment_name": treatment["name"],
			"therapist_id": assigned_therapist_id,
			"therapist_name": therapist["full_name"],
			"appointment_date": appointment_date,
			"start_time": start_time,
			"end_time": end_time,
			"duration_mins": treatment["duration_mins"],
			"price": price,
			"discount_pct": discount_pct,
			"reservation_id": reservation_id,
			"special_notes": special_notes,
			"status": "confirmed",
			"payment_status": "unpaid",
			"created_at": _now(),
		}
		self.appointments[record["id"]] = record
		treatment["booking_count"] = treatment.get("booking_count", 0) + 1
		therapist["appointment_count"] = therapist.get("appointment_count", 0) + 1
		self._emit(tenant, "appointment_created", record["id"], "appointment", {"treatment": treatment["name"]})
		return deepcopy(record)

	async def update_appointment(self, appointment_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		appt = self.appointments.get(appointment_id)
		if not appt or appt["tenant_id"] != tenant:
			raise KeyError(f"appointment_not_found:{appointment_id}")
		if appt["status"] in {"completed", "cancelled"}:
			raise ValueError("cannot_modify_closed_appointment")
		allowed = {"appointment_date", "start_time", "therapist_id", "status", "special_notes"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				appt[k] = v
		if "start_time" in updates:
			treatment = self.treatments.get(appt["treatment_id"])
			if treatment:
				appt["end_time"] = _add_minutes(appt["start_time"], treatment["duration_mins"])
		self._emit(tenant, "appointment_updated", appointment_id, "appointment")
		return deepcopy(appt)

	async def delete_appointment(self, appointment_id: str, reason: str = "cancelled", tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		appt = self.appointments.get(appointment_id)
		if not appt or appt["tenant_id"] != tenant:
			raise KeyError(f"appointment_not_found:{appointment_id}")
		appt["status"] = "cancelled"
		appt["cancellation_reason"] = reason
		self._emit(tenant, "appointment_cancelled", appointment_id, "appointment")
		return {"cancelled": True, "appointment_id": appointment_id}

	async def complete_appointment(self, appointment_id: str, payment_method: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Mark an appointment as complete and settle payment."""
		tenant = self._tenant(tenant_id)
		appt = self.appointments.get(appointment_id)
		if not appt or appt["tenant_id"] != tenant:
			raise KeyError(f"appointment_not_found:{appointment_id}")
		appt["status"] = "completed"
		appt["payment_status"] = "paid"
		appt["payment_method"] = payment_method
		appt["completed_at"] = _now()
		self._emit(tenant, "appointment_completed", appointment_id, "appointment", {"price": appt["price"]})
		return deepcopy(appt)

	# ── Memberships ───────────────────────────────────────────────────────────

	async def list_memberships(self, tenant_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(m) for m in self.memberships.values() if m["tenant_id"] == tenant]
		if status:
			items = [m for m in items if m["status"] == status]
		return items

	async def get_membership(self, membership_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		m = self.memberships.get(membership_id)
		if not m or m["tenant_id"] != tenant:
			raise KeyError(f"membership_not_found:{membership_id}")
		return deepcopy(m)

	async def create_membership(self, guest_name: str, guest_email: str, membership_type: str,
	                             price: float, valid_months: int = 12, included_treatments: int = 0,
	                             discount_pct: float = 0.0, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		now_dt = datetime.utcnow()
		valid_from = now_dt.strftime("%Y-%m-%d")
		valid_to = (now_dt + timedelta(days=30 * valid_months)).strftime("%Y-%m-%d")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"guest_name": guest_name,
			"guest_email": guest_email,
			"membership_type": membership_type,
			"valid_from": valid_from,
			"valid_to": valid_to,
			"price": price,
			"included_treatments": included_treatments,
			"treatments_used": 0,
			"discount_pct": discount_pct,
			"status": "active",
			"created_at": _now(),
		}
		self.memberships[record["id"]] = record
		self._emit(tenant, "membership_created", record["id"], "membership", {"type": membership_type})
		return deepcopy(record)

	async def update_membership(self, membership_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		m = self.memberships.get(membership_id)
		if not m or m["tenant_id"] != tenant:
			raise KeyError(f"membership_not_found:{membership_id}")
		for k, v in updates.items():
			if v is not None:
				m[k] = v
		return deepcopy(m)

	async def delete_membership(self, membership_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		m = self.memberships.get(membership_id)
		if not m or m["tenant_id"] != tenant:
			raise KeyError(f"membership_not_found:{membership_id}")
		m["status"] = "cancelled"
		return {"cancelled": True, "membership_id": membership_id}

	async def renew_membership(self, membership_id: str, months: int = 12, tenant_id: str | None = None) -> dict[str, Any]:
		"""Renew a membership by extending the validity period."""
		tenant = self._tenant(tenant_id)
		m = self.memberships.get(membership_id)
		if not m or m["tenant_id"] != tenant:
			raise KeyError(f"membership_not_found:{membership_id}")
		try:
			current_end = datetime.strptime(m["valid_to"], "%Y-%m-%d")
			new_end = current_end + timedelta(days=30 * months)
			m["valid_to"] = new_end.strftime("%Y-%m-%d")
		except Exception as exc:
			_log.error("renew_membership date error: %s", exc)
		m["status"] = "active"
		self._emit(tenant, "membership_renewed", membership_id, "membership", {"months": months})
		return deepcopy(m)

	# ── Retail ────────────────────────────────────────────────────────────────

	async def list_retail_items(self, tenant_id: str | None = None, category: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.retail_items.values() if r["tenant_id"] == tenant]
		if category:
			items = [r for r in items if r["category"] == category]
		return items

	async def create_retail_item(self, name: str, category: str, price: float, cost: float,
	                              stock_quantity: int, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"name": name,
			"category": category,
			"price": price,
			"cost": cost,
			"margin_pct": round((price - cost) / price * 100, 2) if price > 0 else 0.0,
			"stock_quantity": stock_quantity,
			"status": "active",
			"created_at": _now(),
		}
		self.retail_items[record["id"]] = record
		self._emit(tenant, "retail_item_created", record["id"], "retail_item")
		return deepcopy(record)

	async def sell_retail_item(self, item_id: str, quantity: int, guest_name: str, payment_method: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		item = self.retail_items.get(item_id)
		if not item or item["tenant_id"] != tenant:
			raise KeyError(f"retail_item_not_found:{item_id}")
		if item["stock_quantity"] < quantity:
			raise ValueError(f"insufficient_stock:{item['stock_quantity']}<{quantity}")
		item["stock_quantity"] -= quantity
		sale: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"item_id": item_id,
			"item_name": item["name"],
			"quantity": quantity,
			"unit_price": item["price"],
			"total": item["price"] * quantity,
			"guest_name": guest_name,
			"payment_method": payment_method,
			"created_at": _now(),
		}
		self.retail_sales[sale["id"]] = sale
		self._emit(tenant, "retail_sale", sale["id"], "retail_sale", {"item": item["name"], "qty": quantity})
		return deepcopy(sale)

	# ── Analytics ─────────────────────────────────────────────────────────────

	async def revenue_report(self, date: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		appts = [a for a in self.appointments.values() if a["tenant_id"] == tenant and a["appointment_date"] == date and a["status"] == "completed"]
		retail = [s for s in self.retail_sales.values() if s["tenant_id"] == tenant and s["created_at"][:10] == date]
		return {
			"tenant_id": tenant,
			"date": date,
			"treatment_revenue": round(sum(a["price"] for a in appts), 2),
			"treatment_appointments": len(appts),
			"retail_revenue": round(sum(s["total"] for s in retail), 2),
			"retail_transactions": len(retail),
			"total_revenue": round(sum(a["price"] for a in appts) + sum(s["total"] for s in retail), 2),
			"generated_at": _now(),
		}

	async def therapist_utilisation(self, date: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		therapists = [t for t in self.therapists.values() if t["tenant_id"] == tenant and t["status"] == "active"]
		result = []
		for therapist in therapists:
			appts = [a for a in self.appointments.values()
			         if a["tenant_id"] == tenant and a["therapist_id"] == therapist["id"]
			         and a["appointment_date"] == date and a["status"] != "cancelled"]
			total_mins = sum(a["duration_mins"] for a in appts)
			result.append({
				"therapist_id": therapist["id"],
				"therapist_name": therapist["full_name"],
				"appointments": len(appts),
				"total_mins": total_mins,
				"utilisation_pct": round(total_mins / 480 * 100, 1),  # 8-hour day
			})
		return result

	async def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return {
			"tenant_id": tenant,
			"active_treatments": sum(1 for t in self.treatments.values() if t["tenant_id"] == tenant and t["is_active"]),
			"active_therapists": sum(1 for t in self.therapists.values() if t["tenant_id"] == tenant and t["status"] == "active"),
			"upcoming_appointments": sum(1 for a in self.appointments.values() if a["tenant_id"] == tenant and a["status"] == "confirmed"),
			"active_memberships": sum(1 for m in self.memberships.values() if m["tenant_id"] == tenant and m["status"] == "active"),
			"retail_items": sum(1 for r in self.retail_items.values() if r["tenant_id"] == tenant),
			"generated_at": _now(),
		}
