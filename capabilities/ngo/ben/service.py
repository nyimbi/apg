"""Beneficiary Registry Service — profiling, enrolment, vulnerability scoring, transfers, deduplication."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

_log = logging.getLogger(__name__)

CAPABILITY_ID = "ngo_ben"

SUPPORTED_GENDERS = {"male", "female", "non_binary", "unknown"}
SUPPORTED_PAYMENT_METHODS = {"mpesa", "bank_transfer", "cheque", "cash", "voucher"}
VULNERABILITY_CATEGORIES = {"critical", "high", "medium", "low", "none"}


def _compute_vulnerability_score(
	food: float, shelter: float, health: float, income: float, protection: float
) -> tuple[float, str]:
	"""Compute composite vulnerability score (0–100) and category."""
	composite = (food + shelter + health + income + protection) / 5.0
	if composite >= 80:
		category = "critical"
	elif composite >= 60:
		category = "high"
	elif composite >= 40:
		category = "medium"
	elif composite >= 20:
		category = "low"
	else:
		category = "none"
	return round(composite, 2), category


class BeneficiaryRegistryService:
	"""Async service for NGO beneficiary management."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self._beneficiaries: dict[str, dict[str, Any]] = {}
		self._enrolments: dict[str, dict[str, Any]] = {}
		self._assessments: dict[str, dict[str, Any]] = {}
		self._transfers: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

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

	def _guard_beneficiary(self, beneficiary_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		b = self._beneficiaries.get(beneficiary_id)
		if not b or b["tenant_id"] != tenant:
			raise KeyError(f"beneficiary_not_found:{beneficiary_id}")
		return b

	# ── health / describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"beneficiary_count": len(self._beneficiaries),
			"active_beneficiaries": sum(1 for b in self._beneficiaries.values() if b["status"] == "active"),
			"enrolment_count": len(self._enrolments),
			"pending_transfers": sum(1 for t in self._transfers.values() if t["status"] == "pending"),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "ngo",
			"version": "1.0.0",
			"description": "Beneficiary profiling, programme enrolment, vulnerability scoring, transfer management, deduplication",
			"vulnerability_categories": list(VULNERABILITY_CATEGORIES),
			"tenant_id": self.tenant_id,
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		tenant = self._tenant()
		events = [e for e in self._audit_events if e["tenant_id"] == tenant]
		return [deepcopy(e) for e in events[-limit:]]

	# ── beneficiaries ─────────────────────────────────────────────────────────

	async def list_beneficiaries(
		self,
		status: str | None = None,
		county: str | None = None,
		vulnerability_category: str | None = None,
	) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(b) for b in self._beneficiaries.values() if b["tenant_id"] == tenant]
		if status:
			items = [b for b in items if b["status"] == status]
		if county:
			items = [b for b in items if b.get("county") == county]
		if vulnerability_category:
			items = [b for b in items if b.get("vulnerability_category") == vulnerability_category]
		return items

	async def get_beneficiary(self, beneficiary_id: str) -> dict[str, Any]:
		return deepcopy(self._guard_beneficiary(beneficiary_id))

	async def create_beneficiary(
		self,
		first_name: str,
		last_name: str,
		national_id: str = "",
		date_of_birth: str = "",
		gender: str = "unknown",
		phone: str = "",
		location: str = "",
		county: str = "",
		household_size: int = 1,
		vulnerability_category: str = "",
		notes: str = "",
	) -> dict[str, Any]:
		"""Register a new beneficiary."""
		tenant = self._tenant()
		if not first_name or not last_name:
			raise ValueError("first_name_and_last_name_required")
		if gender not in SUPPORTED_GENDERS:
			raise ValueError(f"unsupported_gender:{gender}")
		record: dict[str, Any] = {
			"id": self._id("ben"),
			"type": "ngo_beneficiary",
			"tenant_id": tenant,
			"first_name": first_name,
			"last_name": last_name,
			"national_id": national_id,
			"date_of_birth": date_of_birth,
			"gender": gender,
			"phone": phone,
			"location": location,
			"county": county,
			"household_size": household_size,
			"vulnerability_category": vulnerability_category,
			"vulnerability_score": 0.0,
			"status": "active",
			"notes": notes,
			"created_at": self._now(),
			"updated_at": None,
		}
		self._beneficiaries[record["id"]] = record
		self._emit("beneficiary_created", record["id"], "ngo_beneficiary", {"name": f"{first_name} {last_name}"})
		_log.info("Beneficiary created: %s (%s %s)", record["id"], first_name, last_name)
		return deepcopy(record)

	async def update_beneficiary(self, beneficiary_id: str, **kwargs: Any) -> dict[str, Any]:
		b = self._guard_beneficiary(beneficiary_id)
		allowed = {"first_name", "last_name", "phone", "location", "county", "household_size",
				   "vulnerability_category", "status", "notes"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				b[k] = v
		b["updated_at"] = self._now()
		self._emit("beneficiary_updated", beneficiary_id, "ngo_beneficiary", kwargs)
		return deepcopy(b)

	async def delete_beneficiary(self, beneficiary_id: str) -> dict[str, Any]:
		"""Soft-delete a beneficiary."""
		b = self._guard_beneficiary(beneficiary_id)
		b["status"] = "inactive"
		b["updated_at"] = self._now()
		self._emit("beneficiary_deactivated", beneficiary_id, "ngo_beneficiary")
		return deepcopy(b)

	# ── enrolments ────────────────────────────────────────────────────────────

	async def list_enrolments(self, beneficiary_id: str | None = None, programme_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(e) for e in self._enrolments.values() if e["tenant_id"] == tenant]
		if beneficiary_id:
			items = [e for e in items if e["beneficiary_id"] == beneficiary_id]
		if programme_id:
			items = [e for e in items if e["programme_id"] == programme_id]
		return items

	async def get_enrolment(self, enrolment_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		e = self._enrolments.get(enrolment_id)
		if not e or e["tenant_id"] != tenant:
			raise KeyError(f"enrolment_not_found:{enrolment_id}")
		return deepcopy(e)

	async def enrol_beneficiary(
		self,
		beneficiary_id: str,
		programme_id: str,
		enrolment_date: str,
		enrolled_by: str,
		notes: str = "",
	) -> dict[str, Any]:
		"""Enrol a beneficiary in a programme."""
		self._guard_beneficiary(beneficiary_id)
		# prevent duplicate active enrolments for same programme
		existing = [
			e for e in self._enrolments.values()
			if e["beneficiary_id"] == beneficiary_id
			and e["programme_id"] == programme_id
			and e["status"] == "active"
		]
		if existing:
			raise ValueError(f"beneficiary_already_enrolled_in_programme:{programme_id}")
		record: dict[str, Any] = {
			"id": self._id("enr"),
			"type": "ngo_enrolment",
			"tenant_id": self._tenant(),
			"beneficiary_id": beneficiary_id,
			"programme_id": programme_id,
			"enrolment_date": enrolment_date,
			"enrolled_by": enrolled_by,
			"notes": notes,
			"status": "active",
			"created_at": self._now(),
		}
		self._enrolments[record["id"]] = record
		self._emit("beneficiary_enrolled", record["id"], "ngo_enrolment", {"beneficiary_id": beneficiary_id, "programme_id": programme_id})
		return deepcopy(record)

	async def exit_beneficiary(self, enrolment_id: str, reason: str, exited_by: str) -> dict[str, Any]:
		"""Exit a beneficiary from a programme."""
		tenant = self._tenant()
		e = self._enrolments.get(enrolment_id)
		if not e or e["tenant_id"] != tenant:
			raise KeyError(f"enrolment_not_found:{enrolment_id}")
		e["status"] = "exited"
		e["exit_reason"] = reason
		e["exited_by"] = exited_by
		e["exited_at"] = self._now()
		self._emit("beneficiary_exited", enrolment_id, "ngo_enrolment", {"reason": reason})
		return deepcopy(e)

	# ── vulnerability assessments ─────────────────────────────────────────────

	async def list_assessments(self, beneficiary_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(a) for a in self._assessments.values() if a["tenant_id"] == tenant]
		if beneficiary_id:
			items = [a for a in items if a["beneficiary_id"] == beneficiary_id]
		return items

	async def create_vulnerability_assessment(
		self,
		beneficiary_id: str,
		assessor: str,
		assessment_date: str,
		food_security_score: float = 0.0,
		shelter_score: float = 0.0,
		health_score: float = 0.0,
		income_score: float = 0.0,
		protection_score: float = 0.0,
		notes: str = "",
	) -> dict[str, Any]:
		"""Run a vulnerability assessment and update beneficiary score."""
		b = self._guard_beneficiary(beneficiary_id)
		composite, category = _compute_vulnerability_score(
			food_security_score, shelter_score, health_score, income_score, protection_score
		)
		record: dict[str, Any] = {
			"id": self._id("vas"),
			"type": "ngo_vulnerability_assessment",
			"tenant_id": self._tenant(),
			"beneficiary_id": beneficiary_id,
			"assessor": assessor,
			"assessment_date": assessment_date,
			"food_security_score": food_security_score,
			"shelter_score": shelter_score,
			"health_score": health_score,
			"income_score": income_score,
			"protection_score": protection_score,
			"composite_score": composite,
			"category": category,
			"notes": notes,
			"created_at": self._now(),
		}
		self._assessments[record["id"]] = record
		# update beneficiary with latest assessment
		b["vulnerability_score"] = composite
		b["vulnerability_category"] = category
		b["updated_at"] = self._now()
		self._emit("vulnerability_assessed", record["id"], "ngo_vulnerability_assessment",
				   {"composite_score": composite, "category": category})
		return deepcopy(record)

	# ── transfers ─────────────────────────────────────────────────────────────

	async def list_transfers(self, beneficiary_id: str | None = None, programme_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(t) for t in self._transfers.values() if t["tenant_id"] == tenant]
		if beneficiary_id:
			items = [t for t in items if t["beneficiary_id"] == beneficiary_id]
		if programme_id:
			items = [t for t in items if t["programme_id"] == programme_id]
		return items

	async def create_transfer(
		self,
		beneficiary_id: str,
		programme_id: str,
		amount: Decimal,
		transfer_date: str,
		reference: str,
		approved_by: str,
		currency: str = "KES",
		payment_method: str = "mpesa",
		notes: str = "",
	) -> dict[str, Any]:
		"""Create a transfer for a beneficiary."""
		self._guard_beneficiary(beneficiary_id)
		if payment_method not in SUPPORTED_PAYMENT_METHODS:
			raise ValueError(f"unsupported_payment_method:{payment_method}")
		if not approved_by:
			raise ValueError("approved_by_required")
		if amount <= 0:
			raise ValueError("amount_must_be_positive")
		record: dict[str, Any] = {
			"id": self._id("trf"),
			"type": "ngo_transfer",
			"tenant_id": self._tenant(),
			"beneficiary_id": beneficiary_id,
			"programme_id": programme_id,
			"amount": amount,
			"currency": currency,
			"transfer_date": transfer_date,
			"payment_method": payment_method,
			"reference": reference,
			"approved_by": approved_by,
			"notes": notes,
			"status": "pending",
			"created_at": self._now(),
		}
		self._transfers[record["id"]] = record
		self._emit("transfer_created", record["id"], "ngo_transfer",
				   {"beneficiary_id": beneficiary_id, "amount": str(amount)})
		return deepcopy(record)

	async def confirm_transfer(self, transfer_id: str, confirmed_by: str) -> dict[str, Any]:
		"""Confirm a transfer as paid."""
		tenant = self._tenant()
		t = self._transfers.get(transfer_id)
		if not t or t["tenant_id"] != tenant:
			raise KeyError(f"transfer_not_found:{transfer_id}")
		if t["status"] != "pending":
			raise ValueError(f"cannot_confirm_{t['status']}_transfer")
		t["status"] = "confirmed"
		t["confirmed_by"] = confirmed_by
		t["confirmed_at"] = self._now()
		self._emit("transfer_confirmed", transfer_id, "ngo_transfer", {"confirmed_by": confirmed_by})
		return deepcopy(t)

	async def reverse_transfer(self, transfer_id: str, reason: str) -> dict[str, Any]:
		"""Reverse a confirmed transfer."""
		tenant = self._tenant()
		t = self._transfers.get(transfer_id)
		if not t or t["tenant_id"] != tenant:
			raise KeyError(f"transfer_not_found:{transfer_id}")
		if t["status"] not in {"confirmed", "pending"}:
			raise ValueError(f"cannot_reverse_{t['status']}_transfer")
		t["status"] = "reversed"
		t["reversal_reason"] = reason
		t["reversed_at"] = self._now()
		self._emit("transfer_reversed", transfer_id, "ngo_transfer", {"reason": reason})
		return deepcopy(t)

	# ── deduplication ─────────────────────────────────────────────────────────

	async def check_duplicate(self, beneficiary_id: str) -> dict[str, Any]:
		"""Fuzzy-match beneficiary against registry to detect duplicates."""
		tenant = self._tenant()
		b = self._guard_beneficiary(beneficiary_id)
		candidates = []
		for other_id, other in self._beneficiaries.items():
			if other_id == beneficiary_id or other["tenant_id"] != tenant:
				continue
			score = 0.0
			# name match
			if other["first_name"].lower() == b["first_name"].lower():
				score += 40.0
			if other["last_name"].lower() == b["last_name"].lower():
				score += 30.0
			# national ID exact match
			if b.get("national_id") and other.get("national_id") and b["national_id"] == other["national_id"]:
				score += 80.0
			# phone match
			if b.get("phone") and other.get("phone") and b["phone"] == other["phone"]:
				score += 50.0
			if score >= 60.0:
				candidates.append({"beneficiary_id": other_id, "name": f"{other['first_name']} {other['last_name']}", "match_score": score})
		is_duplicate = bool(candidates)
		candidates.sort(key=lambda x: x["match_score"], reverse=True)
		return {
			"beneficiary_id": beneficiary_id,
			"duplicate_candidates": candidates[:5],
			"is_duplicate": is_duplicate,
			"checked_at": self._now(),
		}

	async def bulk_deduplication_scan(self) -> dict[str, Any]:
		"""Scan entire beneficiary registry for duplicates."""
		tenant = self._tenant()
		beneficiaries = [b for b in self._beneficiaries.values() if b["tenant_id"] == tenant and b["status"] == "active"]
		tasks = [self.check_duplicate(b["id"]) for b in beneficiaries]
		outcomes = await asyncio.gather(*tasks, return_exceptions=True)
		flagged = [o for o in outcomes if not isinstance(o, Exception) and o.get("is_duplicate")]
		return {
			"tenant_id": tenant,
			"total_scanned": len(beneficiaries),
			"flagged_duplicates": len(flagged),
			"duplicate_sets": flagged[:50],
			"generated_at": self._now(),
		}

	# ── analytics ─────────────────────────────────────────────────────────────

	async def vulnerability_distribution(self) -> dict[str, Any]:
		"""Return vulnerability category distribution."""
		tenant = self._tenant()
		beneficiaries = [b for b in self._beneficiaries.values() if b["tenant_id"] == tenant and b["status"] == "active"]
		by_category: dict[str, int] = {}
		for b in beneficiaries:
			cat = b.get("vulnerability_category", "unknown")
			by_category[cat] = by_category.get(cat, 0) + 1
		return {
			"tenant_id": tenant,
			"total_active": len(beneficiaries),
			"by_category": by_category,
			"generated_at": self._now(),
		}

	async def programme_reach_summary(self, programme_id: str) -> dict[str, Any]:
		"""Return summary of beneficiaries enrolled in a programme."""
		tenant = self._tenant()
		enrolments = [e for e in self._enrolments.values() if e["tenant_id"] == tenant and e["programme_id"] == programme_id and e["status"] == "active"]
		transfers = [t for t in self._transfers.values() if t["tenant_id"] == tenant and t["programme_id"] == programme_id]
		confirmed_transfers = [t for t in transfers if t["status"] == "confirmed"]
		return {
			"programme_id": programme_id,
			"active_enrolments": len(enrolments),
			"total_transfers": len(transfers),
			"confirmed_transfers": len(confirmed_transfers),
			"total_transferred": sum(t["amount"] for t in confirmed_transfers),
			"generated_at": self._now(),
		}

	async def bulk_enrol(self, beneficiary_ids: list[str], programme_id: str, enrolment_date: str, enrolled_by: str) -> dict[str, Any]:
		"""Bulk-enrol multiple beneficiaries in a programme."""
		tasks = [
			self.enrol_beneficiary(bid, programme_id, enrolment_date, enrolled_by)
			for bid in beneficiary_ids
		]
		outcomes = await asyncio.gather(*tasks, return_exceptions=True)
		results, errors = [], []
		for bid, outcome in zip(beneficiary_ids, outcomes):
			if isinstance(outcome, Exception):
				errors.append({"beneficiary_id": bid, "error": str(outcome)})
			else:
				results.append(outcome)
		return {"enrolled": len(results), "failed": len(errors), "enrolments": results, "errors": errors}
