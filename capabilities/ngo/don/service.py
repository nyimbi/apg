"""Donor Relationship Management Service — registry, communications, pledges, receipts, stewardship."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

_log = logging.getLogger(__name__)

CAPABILITY_ID = "ngo_don"

SUPPORTED_DONOR_TYPES = {"individual", "corporate", "foundation", "government", "bilateral", "multilateral"}
SUPPORTED_CHANNELS = {"email", "phone", "sms", "whatsapp", "meeting", "letter", "event"}
SUPPORTED_DIRECTIONS = {"inbound", "outbound"}
SUPPORTED_FREQUENCIES = {"one_time", "monthly", "quarterly", "annual"}
SUPPORTED_STEWARDSHIP_TIERS = {"standard", "major", "principal", "legacy"}
SUPPORTED_PAYMENT_METHODS = {"bank_transfer", "cheque", "mpesa", "swift", "eft", "card", "cash"}


class DonorRelationshipService:
	"""Async service for NGO donor relationship management."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self._donors: dict[str, dict[str, Any]] = {}
		self._communications: dict[str, dict[str, Any]] = {}
		self._pledges: dict[str, dict[str, Any]] = {}
		self._receipts: dict[str, dict[str, Any]] = {}
		self._stewardship_plans: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []
		self._receipt_seq: int = 0

	# ── helpers ───────────────────────────────────────────────────────────────

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _tenant(self) -> str:
		if not self.tenant_id:
			raise PermissionError("tenant_context_required")
		return self.tenant_id

	def _emit(self, event_type: str, record_id: str, record_type: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._id("evt"),
			"tenant_id": self._tenant(),
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"details": details or {},
			"emitted_at": self._now(),
		})

	def _guard_donor(self, donor_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		donor = self._donors.get(donor_id)
		if not donor or donor["tenant_id"] != tenant:
			raise KeyError(f"donor_not_found:{donor_id}")
		return donor

	def _next_receipt_number(self) -> str:
		self._receipt_seq += 1
		year = datetime.utcnow().year
		return f"RCP-{year}-{self._receipt_seq:06d}"

	# ── health / describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"donor_count": len(self._donors),
			"active_donors": sum(1 for d in self._donors.values() if d["status"] == "active"),
			"open_pledges": sum(1 for p in self._pledges.values() if p["status"] == "open"),
			"receipts_issued": len(self._receipts),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "ngo",
			"version": "1.0.0",
			"description": "Donor registry, communication history, pledge tracking, receipt generation, stewardship plans",
			"donor_types": list(SUPPORTED_DONOR_TYPES),
			"stewardship_tiers": list(SUPPORTED_STEWARDSHIP_TIERS),
			"tenant_id": self.tenant_id,
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		tenant = self._tenant()
		events = [e for e in self._audit_events if e["tenant_id"] == tenant]
		return [deepcopy(e) for e in events[-limit:]]

	# ── donors ────────────────────────────────────────────────────────────────

	async def list_donors(self, status: str | None = None, donor_type: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(d) for d in self._donors.values() if d["tenant_id"] == tenant]
		if status:
			items = [d for d in items if d["status"] == status]
		if donor_type:
			items = [d for d in items if d["donor_type"] == donor_type]
		return items

	async def get_donor(self, donor_id: str) -> dict[str, Any]:
		return deepcopy(self._guard_donor(donor_id))

	async def create_donor(
		self,
		name: str,
		donor_type: str = "individual",
		email: str = "",
		phone: str = "",
		country: str = "KE",
		address: str = "",
		tax_id: str = "",
		notes: str = "",
		tags: list[str] | None = None,
	) -> dict[str, Any]:
		"""Register a new donor."""
		tenant = self._tenant()
		if not name:
			raise ValueError("name_required")
		if donor_type not in SUPPORTED_DONOR_TYPES:
			raise ValueError(f"unsupported_donor_type:{donor_type}")
		record: dict[str, Any] = {
			"id": self._id("don"),
			"type": "ngo_donor",
			"tenant_id": tenant,
			"name": name,
			"donor_type": donor_type,
			"email": email,
			"phone": phone,
			"country": country,
			"address": address,
			"tax_id": tax_id,
			"notes": notes,
			"tags": tags or [],
			"total_pledged": Decimal("0"),
			"total_received": Decimal("0"),
			"status": "active",
			"created_at": self._now(),
			"updated_at": None,
		}
		self._donors[record["id"]] = record
		self._emit("donor_created", record["id"], "ngo_donor", {"name": name, "type": donor_type})
		_log.info("Donor created: %s (%s)", record["id"], name)
		return deepcopy(record)

	async def update_donor(self, donor_id: str, **kwargs: Any) -> dict[str, Any]:
		donor = self._guard_donor(donor_id)
		allowed = {"name", "email", "phone", "address", "status", "notes", "tags"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				donor[k] = v
		donor["updated_at"] = self._now()
		self._emit("donor_updated", donor_id, "ngo_donor", kwargs)
		return deepcopy(donor)

	async def delete_donor(self, donor_id: str) -> dict[str, Any]:
		"""Soft-delete a donor by marking inactive."""
		donor = self._guard_donor(donor_id)
		donor["status"] = "inactive"
		donor["updated_at"] = self._now()
		self._emit("donor_deactivated", donor_id, "ngo_donor")
		return deepcopy(donor)

	async def search_donors(self, query: str) -> list[dict[str, Any]]:
		"""Search donors by name, email or phone (case-insensitive)."""
		tenant = self._tenant()
		q = query.lower()
		return [
			deepcopy(d) for d in self._donors.values()
			if d["tenant_id"] == tenant and (
				q in d["name"].lower()
				or q in d.get("email", "").lower()
				or q in d.get("phone", "")
			)
		]

	# ── communications ────────────────────────────────────────────────────────

	async def list_communications(self, donor_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(c) for c in self._communications.values() if c["tenant_id"] == tenant]
		if donor_id:
			items = [c for c in items if c["donor_id"] == donor_id]
		return items

	async def get_communication(self, comm_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		comm = self._communications.get(comm_id)
		if not comm or comm["tenant_id"] != tenant:
			raise KeyError(f"communication_not_found:{comm_id}")
		return deepcopy(comm)

	async def log_communication(
		self,
		donor_id: str,
		subject: str,
		body: str,
		staff_member: str,
		communication_date: str,
		channel: str = "email",
		direction: str = "outbound",
		tags: list[str] | None = None,
	) -> dict[str, Any]:
		"""Log a communication with a donor."""
		self._guard_donor(donor_id)
		if channel not in SUPPORTED_CHANNELS:
			raise ValueError(f"unsupported_channel:{channel}")
		if direction not in SUPPORTED_DIRECTIONS:
			raise ValueError(f"unsupported_direction:{direction}")
		record: dict[str, Any] = {
			"id": self._id("com"),
			"type": "ngo_communication",
			"tenant_id": self._tenant(),
			"donor_id": donor_id,
			"channel": channel,
			"direction": direction,
			"subject": subject,
			"body": body,
			"staff_member": staff_member,
			"communication_date": communication_date,
			"tags": tags or [],
			"created_at": self._now(),
		}
		self._communications[record["id"]] = record
		self._emit("communication_logged", record["id"], "ngo_communication", {"donor_id": donor_id, "channel": channel})
		return deepcopy(record)

	async def delete_communication(self, comm_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		comm = self._communications.get(comm_id)
		if not comm or comm["tenant_id"] != tenant:
			raise KeyError(f"communication_not_found:{comm_id}")
		removed = self._communications.pop(comm_id)
		self._emit("communication_deleted", comm_id, "ngo_communication")
		return deepcopy(removed)

	# ── pledges ───────────────────────────────────────────────────────────────

	async def list_pledges(self, donor_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(p) for p in self._pledges.values() if p["tenant_id"] == tenant]
		if donor_id:
			items = [p for p in items if p["donor_id"] == donor_id]
		if status:
			items = [p for p in items if p["status"] == status]
		return items

	async def get_pledge(self, pledge_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		pledge = self._pledges.get(pledge_id)
		if not pledge or pledge["tenant_id"] != tenant:
			raise KeyError(f"pledge_not_found:{pledge_id}")
		return deepcopy(pledge)

	async def create_pledge(
		self,
		donor_id: str,
		amount: Decimal,
		pledge_date: str,
		due_date: str,
		currency: str = "KES",
		purpose: str = "",
		frequency: str = "one_time",
		notes: str = "",
	) -> dict[str, Any]:
		"""Record a donor pledge."""
		donor = self._guard_donor(donor_id)
		if amount <= 0:
			raise ValueError("pledge_amount_must_be_positive")
		if frequency not in SUPPORTED_FREQUENCIES:
			raise ValueError(f"unsupported_frequency:{frequency}")
		record: dict[str, Any] = {
			"id": self._id("ple"),
			"type": "ngo_pledge",
			"tenant_id": self._tenant(),
			"donor_id": donor_id,
			"amount": amount,
			"received_amount": Decimal("0"),
			"currency": currency,
			"pledge_date": pledge_date,
			"due_date": due_date,
			"purpose": purpose,
			"frequency": frequency,
			"notes": notes,
			"status": "open",
			"created_at": self._now(),
		}
		self._pledges[record["id"]] = record
		donor["total_pledged"] += amount
		self._emit("pledge_created", record["id"], "ngo_pledge", {"donor_id": donor_id, "amount": str(amount)})
		return deepcopy(record)

	async def update_pledge(self, pledge_id: str, **kwargs: Any) -> dict[str, Any]:
		tenant = self._tenant()
		pledge = self._pledges.get(pledge_id)
		if not pledge or pledge["tenant_id"] != tenant:
			raise KeyError(f"pledge_not_found:{pledge_id}")
		allowed = {"notes", "due_date", "status", "purpose"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				pledge[k] = v
		self._emit("pledge_updated", pledge_id, "ngo_pledge", kwargs)
		return deepcopy(pledge)

	async def cancel_pledge(self, pledge_id: str, reason: str) -> dict[str, Any]:
		"""Cancel an open pledge."""
		tenant = self._tenant()
		pledge = self._pledges.get(pledge_id)
		if not pledge or pledge["tenant_id"] != tenant:
			raise KeyError(f"pledge_not_found:{pledge_id}")
		if pledge["status"] != "open":
			raise ValueError(f"cannot_cancel_{pledge['status']}_pledge")
		pledge["status"] = "cancelled"
		pledge["cancellation_reason"] = reason
		pledge["cancelled_at"] = self._now()
		self._emit("pledge_cancelled", pledge_id, "ngo_pledge", {"reason": reason})
		return deepcopy(pledge)

	# ── receipts ──────────────────────────────────────────────────────────────

	async def list_receipts(self, donor_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(r) for r in self._receipts.values() if r["tenant_id"] == tenant]
		if donor_id:
			items = [r for r in items if r["donor_id"] == donor_id]
		return items

	async def get_receipt(self, receipt_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		receipt = self._receipts.get(receipt_id)
		if not receipt or receipt["tenant_id"] != tenant:
			raise KeyError(f"receipt_not_found:{receipt_id}")
		return deepcopy(receipt)

	async def generate_receipt(
		self,
		donor_id: str,
		amount: Decimal,
		receipt_date: str,
		reference: str,
		issued_by: str,
		pledge_id: str | None = None,
		currency: str = "KES",
		payment_method: str = "bank_transfer",
	) -> dict[str, Any]:
		"""Generate and record a donation receipt."""
		donor = self._guard_donor(donor_id)
		if payment_method not in SUPPORTED_PAYMENT_METHODS:
			raise ValueError(f"unsupported_payment_method:{payment_method}")
		if pledge_id:
			pledge = self._pledges.get(pledge_id)
			if pledge and pledge["tenant_id"] == self._tenant():
				pledge["received_amount"] += amount
				if pledge["received_amount"] >= pledge["amount"]:
					pledge["status"] = "fulfilled"
		record: dict[str, Any] = {
			"id": self._id("rcp"),
			"type": "ngo_receipt",
			"tenant_id": self._tenant(),
			"receipt_number": self._next_receipt_number(),
			"donor_id": donor_id,
			"pledge_id": pledge_id,
			"amount": amount,
			"currency": currency,
			"receipt_date": receipt_date,
			"payment_method": payment_method,
			"reference": reference,
			"issued_by": issued_by,
			"status": "issued",
			"created_at": self._now(),
		}
		self._receipts[record["id"]] = record
		donor["total_received"] += amount
		self._emit("receipt_generated", record["id"], "ngo_receipt", {"donor_id": donor_id, "amount": str(amount)})
		return deepcopy(record)

	# ── stewardship plans ─────────────────────────────────────────────────────

	async def list_stewardship_plans(self, tier: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(p) for p in self._stewardship_plans.values() if p["tenant_id"] == tenant]
		if tier:
			items = [p for p in items if p["tier"] == tier]
		return items

	async def get_stewardship_plan(self, plan_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		plan = self._stewardship_plans.get(plan_id)
		if not plan or plan["tenant_id"] != tenant:
			raise KeyError(f"stewardship_plan_not_found:{plan_id}")
		return deepcopy(plan)

	async def create_stewardship_plan(
		self,
		donor_id: str,
		tier: str = "standard",
		touchpoints_per_year: int = 4,
		assigned_to: str = "",
		notes: str = "",
	) -> dict[str, Any]:
		"""Create a stewardship plan for a donor."""
		self._guard_donor(donor_id)
		if tier not in SUPPORTED_STEWARDSHIP_TIERS:
			raise ValueError(f"invalid_stewardship_tier:{tier}")
		if touchpoints_per_year < 1:
			raise ValueError("touchpoints_must_be_at_least_1")
		record: dict[str, Any] = {
			"id": self._id("stp"),
			"type": "ngo_stewardship_plan",
			"tenant_id": self._tenant(),
			"donor_id": donor_id,
			"tier": tier,
			"touchpoints_per_year": touchpoints_per_year,
			"assigned_to": assigned_to,
			"notes": notes,
			"completed_touchpoints": 0,
			"status": "active",
			"created_at": self._now(),
		}
		self._stewardship_plans[record["id"]] = record
		self._emit("stewardship_plan_created", record["id"], "ngo_stewardship_plan", {"donor_id": donor_id, "tier": tier})
		return deepcopy(record)

	async def record_stewardship_touchpoint(self, plan_id: str, notes: str = "") -> dict[str, Any]:
		"""Record a completed stewardship touchpoint."""
		tenant = self._tenant()
		plan = self._stewardship_plans.get(plan_id)
		if not plan or plan["tenant_id"] != tenant:
			raise KeyError(f"stewardship_plan_not_found:{plan_id}")
		plan["completed_touchpoints"] += 1
		plan["last_touchpoint_at"] = self._now()
		if notes:
			plan["last_touchpoint_notes"] = notes
		self._emit("stewardship_touchpoint_recorded", plan_id, "ngo_stewardship_plan")
		return deepcopy(plan)

	async def update_stewardship_plan(self, plan_id: str, **kwargs: Any) -> dict[str, Any]:
		tenant = self._tenant()
		plan = self._stewardship_plans.get(plan_id)
		if not plan or plan["tenant_id"] != tenant:
			raise KeyError(f"stewardship_plan_not_found:{plan_id}")
		allowed = {"tier", "touchpoints_per_year", "assigned_to", "notes", "status"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				plan[k] = v
		self._emit("stewardship_plan_updated", plan_id, "ngo_stewardship_plan", kwargs)
		return deepcopy(plan)

	# ── analytics ─────────────────────────────────────────────────────────────

	async def donor_giving_history(self, donor_id: str) -> dict[str, Any]:
		"""Return complete giving history for a donor."""
		donor = self._guard_donor(donor_id)
		pledges = [p for p in self._pledges.values() if p["donor_id"] == donor_id]
		receipts = [r for r in self._receipts.values() if r["donor_id"] == donor_id]
		comms = [c for c in self._communications.values() if c["donor_id"] == donor_id]
		return {
			"donor_id": donor_id,
			"donor_name": donor["name"],
			"total_pledged": donor["total_pledged"],
			"total_received": donor["total_received"],
			"pledge_count": len(pledges),
			"receipt_count": len(receipts),
			"communication_count": len(comms),
			"open_pledges": len([p for p in pledges if p["status"] == "open"]),
			"generated_at": self._now(),
		}

	async def portfolio_summary(self) -> dict[str, Any]:
		"""Return donor portfolio summary."""
		tenant = self._tenant()
		donors = [d for d in self._donors.values() if d["tenant_id"] == tenant]
		by_type: dict[str, int] = {}
		for d in donors:
			by_type[d["donor_type"]] = by_type.get(d["donor_type"], 0) + 1
		total_pledged = sum(d["total_pledged"] for d in donors)
		total_received = sum(d["total_received"] for d in donors)
		return {
			"tenant_id": tenant,
			"total_donors": len(donors),
			"active_donors": sum(1 for d in donors if d["status"] == "active"),
			"by_type": by_type,
			"total_pledged": total_pledged,
			"total_received": total_received,
			"open_pledges": sum(1 for p in self._pledges.values() if p["tenant_id"] == tenant and p["status"] == "open"),
			"generated_at": self._now(),
		}

	async def retention_analysis(self) -> dict[str, Any]:
		"""Analyse donor retention — donors who gave in prior year vs current."""
		tenant = self._tenant()
		current_year = str(datetime.utcnow().year)
		prior_year = str(datetime.utcnow().year - 1)
		current_donors = {r["donor_id"] for r in self._receipts.values() if r["tenant_id"] == tenant and r["receipt_date"][:4] == current_year}
		prior_donors = {r["donor_id"] for r in self._receipts.values() if r["tenant_id"] == tenant and r["receipt_date"][:4] == prior_year}
		retained = current_donors & prior_donors
		lapsed = prior_donors - current_donors
		new_donors = current_donors - prior_donors
		retention_rate = len(retained) / len(prior_donors) * 100 if prior_donors else 0.0
		return {
			"tenant_id": tenant,
			"current_year": current_year,
			"prior_year": prior_year,
			"current_givers": len(current_donors),
			"prior_givers": len(prior_donors),
			"retained": len(retained),
			"lapsed": len(lapsed),
			"new_donors": len(new_donors),
			"retention_rate_pct": round(retention_rate, 2),
			"generated_at": self._now(),
		}

	async def bulk_import_donors(self, donors: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-import donors from an external list."""
		tasks = [
			self.create_donor(
				name=d["name"],
				donor_type=d.get("donor_type", "individual"),
				email=d.get("email", ""),
				phone=d.get("phone", ""),
				country=d.get("country", "KE"),
				address=d.get("address", ""),
				tax_id=d.get("tax_id", ""),
				notes=d.get("notes", ""),
				tags=d.get("tags", []),
			)
			for d in donors
		]
		outcomes = await asyncio.gather(*tasks, return_exceptions=True)
		results, errors = [], []
		for donor, outcome in zip(donors, outcomes):
			if isinstance(outcome, Exception):
				errors.append({"input": donor, "error": str(outcome)})
			else:
				results.append(outcome)
		return {"created": len(results), "failed": len(errors), "donors": results, "errors": errors}

	async def overdue_pledges(self) -> list[dict[str, Any]]:
		"""Return pledges past their due date that remain open."""
		tenant = self._tenant()
		today = self._now()[:10]
		return [
			deepcopy(p) for p in self._pledges.values()
			if p["tenant_id"] == tenant and p["status"] == "open" and p["due_date"] < today
		]
