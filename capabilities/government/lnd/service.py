"""Land Registry — async service implementation."""
from __future__ import annotations

import asyncio
import hashlib
import logging
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "gov_lnd"

SUPPORTED_LAND_USES = {
	"residential", "commercial", "agricultural", "industrial",
	"mixed_use", "public", "conservation", "institutional",
}
SUPPORTED_TENURE_TYPES = {"freehold", "leasehold", "community", "government"}
SUPPORTED_ENCUMBRANCE_TYPES = {
	"mortgage", "caveat", "charge", "easement", "restriction",
	"lien", "covenant", "caution",
}
SUPPORTED_VALUATION_METHODS = {
	"market_comparison", "income_capitalisation", "depreciated_replacement_cost",
	"residual_method",
}
SUPPORTED_ADJUDICATION_OUTCOMES = {
	"approved", "rejected", "referred", "appealed", "withdrawn",
}

# Stamp duty rates per Kenya Stamp Duty Act Cap 480
_STAMP_DUTY_RATES: dict[str, Decimal] = {
	"residential": Decimal("0.04"),
	"commercial": Decimal("0.04"),
	"industrial": Decimal("0.04"),
	"mixed_use": Decimal("0.04"),
	"agricultural": Decimal("0.02"),
	"public": Decimal("0.00"),
	"conservation": Decimal("0.00"),
	"institutional": Decimal("0.04"),
}
_REGISTRATION_FEE_TIERS: list[tuple[Decimal, Decimal, Decimal]] = [
	# (up_to, fixed, rate_above)
	(Decimal("1_000_000"), Decimal("2_500"), Decimal("0")),
	(Decimal("5_000_000"), Decimal("2_500"), Decimal("0.001")),
	(Decimal("999_999_999"), Decimal("6_500"), Decimal("0.0005")),
]
SUPPORTED_EXEMPTION_TYPES = {
	"first_time_buyer", "government", "ngo", "inheritance", "court_order",
}
CAUTION_DEFAULT_EXPIRY_DAYS = 60


class LandRegistryService:
	"""Async Land Registry capability service."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.parcels: dict[str, dict[str, Any]] = {}
		self.titles: dict[str, dict[str, Any]] = {}
		self.transfers: dict[str, dict[str, Any]] = {}
		self.adjudications: dict[str, dict[str, Any]] = {}
		self.encumbrances: dict[str, dict[str, Any]] = {}
		self.valuations: dict[str, dict[str, Any]] = {}
		self.searches: dict[str, dict[str, Any]] = {}
		self.surveyors: dict[str, dict[str, Any]] = {}
		self.survey_plans: dict[str, dict[str, Any]] = {}
		self.land_rates: dict[str, dict[str, Any]] = {}
		self.cautions: dict[str, dict[str, Any]] = {}
		self.leases: dict[str, dict[str, Any]] = {}
		self.land_use_changes: dict[str, dict[str, Any]] = {}
		self.duty_payments: dict[str, dict[str, Any]] = {}
		self.duty_exemptions: dict[str, dict[str, Any]] = {}
		self.rates_payments: dict[str, dict[str, Any]] = {}
		self.spousal_consents: dict[str, dict[str, Any]] = {}
		self.escalations: dict[str, dict[str, Any]] = {}
		self.title_certificates: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _emit(self, tenant_id: str, event_type: str, resource_id: str, details: dict[str, Any]) -> None:
		self._audit_events.append({
			"id": self._record_id("audit"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"resource_id": resource_id,
			"details": deepcopy(details),
			"emitted_at": self._now(),
		})

	# ── Health & meta ─────────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		"""Return land registry service health status."""
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"parcels": len(self.parcels),
			"titles": len(self.titles),
			"pending_transfers": sum(1 for t in self.transfers.values() if t["status"] == "pending"),
			"active_encumbrances": sum(1 for e in self.encumbrances.values() if e["status"] == "active"),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		"""Return capability contract metadata."""
		return {
			"capability_id": CAPABILITY_ID,
			"name": "Land Registry",
			"version": "1.0.0",
			"domain": "government",
			"description": "Parcel cadastre, title issuance, land transfer, adjudication, encumbrance registry, valuation rolls",
			"supported_land_uses": sorted(SUPPORTED_LAND_USES),
			"supported_tenure_types": sorted(SUPPORTED_TENURE_TYPES),
			"supported_encumbrance_types": sorted(SUPPORTED_ENCUMBRANCE_TYPES),
			"supported_valuation_methods": sorted(SUPPORTED_VALUATION_METHODS),
		}

	async def get_audit_events(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		"""Return all audit events for a tenant."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Parcel / cadastre ─────────────────────────────────────────────────────

	async def register_parcel(
		self,
		parcel_number: str,
		county: str,
		sub_county: str,
		location: str,
		area_hectares: float,
		tenant_id: str = "default",
		land_use: str = "residential",
		coordinates: dict[str, Any] | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Register a new land parcel in the cadastre."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(parcel_number, "parcel_number")
		guard_non_empty_string(county, "county")
		if land_use not in SUPPORTED_LAND_USES:
			raise ValueError(f"unsupported land use: {land_use!r}")
		if area_hectares <= 0:
			raise ValueError("area_hectares must be positive")
		# Enforce parcel uniqueness per tenant
		for p in self.parcels.values():
			if p["tenant_id"] == tenant and p["parcel_number"] == parcel_number:
				raise ValueError(f"parcel {parcel_number!r} already registered")
		record = {
			"id": self._record_id("parcel"),
			"type": "land_parcel",
			"parcel_number": parcel_number,
			"county": county,
			"sub_county": sub_county,
			"location": location,
			"area_hectares": area_hectares,
			"land_use": land_use,
			"coordinates": deepcopy(coordinates or {}),
			"owner_id": None,
			"title_number": None,
			"tenant_id": tenant,
			"metadata": deepcopy(metadata or {}),
			"status": "registered",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.parcels[record["id"]] = record
		self._emit(tenant, "parcel_registered", record["id"],
			{"parcel_number": parcel_number, "county": county, "area_hectares": area_hectares})
		return deepcopy(record)

	async def get_parcel(self, parcel_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve a parcel record."""
		tenant = self._tenant(tenant_id)
		record = self.parcels.get(parcel_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"parcel {parcel_id!r} not found")
		return deepcopy(record)

	async def update_parcel(
		self,
		parcel_id: str,
		tenant_id: str = "default",
		land_use: str | None = None,
		area_hectares: float | None = None,
		coordinates: dict[str, Any] | None = None,
		metadata: dict[str, Any] | None = None,
		status: str | None = None,
	) -> dict[str, Any]:
		"""Update parcel attributes."""
		tenant = self._tenant(tenant_id)
		record = self.parcels.get(parcel_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"parcel {parcel_id!r} not found")
		if land_use:
			if land_use not in SUPPORTED_LAND_USES:
				raise ValueError(f"unsupported land use: {land_use!r}")
			record["land_use"] = land_use
		if area_hectares is not None:
			if area_hectares <= 0:
				raise ValueError("area_hectares must be positive")
			record["area_hectares"] = area_hectares
		if coordinates:
			record["coordinates"].update(coordinates)
		if metadata:
			record["metadata"].update(metadata)
		if status:
			record["status"] = status
		record["updated_at"] = self._now()
		self._emit(tenant, "parcel_updated", parcel_id, {"land_use": land_use, "status": status})
		return deepcopy(record)

	async def list_parcels(
		self,
		tenant_id: str = "default",
		county: str | None = None,
		land_use: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List parcels with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.parcels.values() if r["tenant_id"] == tenant]
		if county:
			items = [r for r in items if r["county"] == county]
		if land_use:
			items = [r for r in items if r["land_use"] == land_use]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	async def delete_parcel(self, parcel_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Deregister a parcel (admin-only, sets status to deregistered)."""
		tenant = self._tenant(tenant_id)
		record = self.parcels.get(parcel_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"parcel {parcel_id!r} not found")
		record["status"] = "deregistered"
		record["updated_at"] = self._now()
		self._emit(tenant, "parcel_deregistered", parcel_id, {})
		return {"id": parcel_id, "status": "deregistered"}

	# ── Title issuance ────────────────────────────────────────────────────────

	async def issue_title(
		self,
		parcel_id: str,
		title_number: str,
		owner_id: str,
		owner_name: str,
		issue_date: str,
		issued_by: str,
		tenant_id: str = "default",
		owner_type: str = "individual",
		tenure_type: str = "freehold",
		lease_term_years: int | None = None,
	) -> dict[str, Any]:
		"""Issue a title deed for a registered parcel."""
		tenant = self._tenant(tenant_id)
		parcel = self.parcels.get(parcel_id)
		if not parcel or parcel["tenant_id"] != tenant:
			raise KeyError(f"parcel {parcel_id!r} not found")
		if tenure_type not in SUPPORTED_TENURE_TYPES:
			raise ValueError(f"unsupported tenure type: {tenure_type!r}")
		if tenure_type == "leasehold" and not lease_term_years:
			raise ValueError("lease_term_years required for leasehold tenure")
		# Check no active title exists
		for t in self.titles.values():
			if t["tenant_id"] == tenant and t["parcel_id"] == parcel_id and t["status"] == "active":
				raise PermissionError("active_title_exists")
		record = {
			"id": self._record_id("title"),
			"type": "land_title",
			"parcel_id": parcel_id,
			"title_number": title_number,
			"owner_id": owner_id,
			"owner_name": owner_name,
			"owner_type": owner_type,
			"issue_date": issue_date,
			"tenure_type": tenure_type,
			"lease_term_years": lease_term_years,
			"issued_by": issued_by,
			"tenant_id": tenant,
			"status": "active",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.titles[record["id"]] = record
		parcel["owner_id"] = owner_id
		parcel["title_number"] = title_number
		parcel["status"] = "titled"
		parcel["updated_at"] = self._now()
		self._emit(tenant, "title_issued", record["id"],
			{"parcel_id": parcel_id, "title_number": title_number, "owner_id": owner_id})
		return deepcopy(record)

	async def get_title(self, title_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve a title record."""
		tenant = self._tenant(tenant_id)
		record = self.titles.get(title_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"title {title_id!r} not found")
		return deepcopy(record)

	async def update_title(
		self,
		title_id: str,
		tenant_id: str = "default",
		owner_id: str | None = None,
		owner_name: str | None = None,
		status: str | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Update a title record."""
		tenant = self._tenant(tenant_id)
		record = self.titles.get(title_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"title {title_id!r} not found")
		if owner_id:
			record["owner_id"] = owner_id
		if owner_name:
			record["owner_name"] = owner_name
		if status:
			record["status"] = status
		if notes:
			record.setdefault("notes", []).append({"note": notes, "at": self._now()})
		record["updated_at"] = self._now()
		self._emit(tenant, "title_updated", title_id, {"owner_id": owner_id, "status": status})
		return deepcopy(record)

	async def list_titles(
		self,
		tenant_id: str = "default",
		owner_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List titles with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.titles.values() if r["tenant_id"] == tenant]
		if owner_id:
			items = [r for r in items if r["owner_id"] == owner_id]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	async def cancel_title(self, title_id: str, tenant_id: str = "default", reason: str = "cancelled") -> dict[str, Any]:
		"""Cancel an active title deed."""
		tenant = self._tenant(tenant_id)
		record = self.titles.get(title_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"title {title_id!r} not found")
		record["status"] = "cancelled"
		record["cancellation_reason"] = reason
		record["updated_at"] = self._now()
		self._emit(tenant, "title_cancelled", title_id, {"reason": reason})
		return deepcopy(record)

	# ── Land transfer ─────────────────────────────────────────────────────────

	async def initiate_transfer(
		self,
		title_id: str,
		transferor_id: str,
		transferor_name: str,
		transferee_id: str,
		transferee_name: str,
		consideration_kes: float,
		transfer_date: str,
		instrument_number: str,
		approved_by: str,
		tenant_id: str = "default",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Initiate a land title transfer."""
		tenant = self._tenant(tenant_id)
		title = self.titles.get(title_id)
		if not title or title["tenant_id"] != tenant:
			raise KeyError(f"title {title_id!r} not found")
		if title["status"] != "active":
			raise PermissionError("can_only_transfer_active_titles")
		if consideration_kes <= 0:
			raise ValueError("consideration_kes must be positive")
		# Check no active encumbrances blocking transfer
		blocking = [
			e for e in self.encumbrances.values()
			if e["tenant_id"] == tenant and e["title_id"] == title_id
			and e["status"] == "active" and e["encumbrance_type"] in {"restriction", "caveat"}
		]
		if blocking:
			raise PermissionError("title_encumbered_cannot_transfer")
		record = {
			"id": self._record_id("transfer"),
			"type": "land_transfer",
			"title_id": title_id,
			"transferor_id": transferor_id,
			"transferor_name": transferor_name,
			"transferee_id": transferee_id,
			"transferee_name": transferee_name,
			"consideration_kes": consideration_kes,
			"transfer_date": transfer_date,
			"instrument_number": instrument_number,
			"approved_by": approved_by,
			"tenant_id": tenant,
			"metadata": deepcopy(metadata or {}),
			"status": "pending",
			"created_at": self._now(),
			"completed_at": None,
		}
		self.transfers[record["id"]] = record
		self._emit(tenant, "transfer_initiated", record["id"],
			{"title_id": title_id, "transferee_id": transferee_id, "consideration_kes": consideration_kes})
		return deepcopy(record)

	async def complete_transfer(self, transfer_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Complete a land transfer — updates title ownership."""
		tenant = self._tenant(tenant_id)
		transfer = self.transfers.get(transfer_id)
		if not transfer or transfer["tenant_id"] != tenant:
			raise KeyError(f"transfer {transfer_id!r} not found")
		if transfer["status"] != "pending":
			raise PermissionError("transfer_not_pending")
		title = self.titles.get(transfer["title_id"])
		if not title:
			raise KeyError("title not found for transfer")
		# Update title ownership
		title["owner_id"] = transfer["transferee_id"]
		title["owner_name"] = transfer["transferee_name"]
		title["updated_at"] = self._now()
		# Update parcel owner
		parcel_id = title["parcel_id"]
		if parcel_id in self.parcels:
			self.parcels[parcel_id]["owner_id"] = transfer["transferee_id"]
			self.parcels[parcel_id]["updated_at"] = self._now()
		transfer["status"] = "completed"
		transfer["completed_at"] = self._now()
		self._emit(tenant, "transfer_completed", transfer_id,
			{"title_id": transfer["title_id"], "new_owner_id": transfer["transferee_id"]})
		return deepcopy(transfer)

	async def get_transfer(self, transfer_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve a transfer record."""
		tenant = self._tenant(tenant_id)
		record = self.transfers.get(transfer_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"transfer {transfer_id!r} not found")
		return deepcopy(record)

	async def list_transfers(
		self,
		tenant_id: str = "default",
		title_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List transfers with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.transfers.values() if r["tenant_id"] == tenant]
		if title_id:
			items = [r for r in items if r["title_id"] == title_id]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	# ── Adjudication ──────────────────────────────────────────────────────────

	async def submit_adjudication(
		self,
		parcel_id: str,
		claimant_id: str,
		claimant_name: str,
		claim_basis: str,
		evidence_reference: str,
		adjudicator_id: str,
		tenant_id: str = "default",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Submit a land adjudication claim."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(claim_basis, "claim_basis")
		guard_non_empty_string(evidence_reference, "evidence_reference")
		record = {
			"id": self._record_id("adj"),
			"type": "land_adjudication",
			"parcel_id": parcel_id,
			"claimant_id": claimant_id,
			"claimant_name": claimant_name,
			"claim_basis": claim_basis,
			"evidence_reference": evidence_reference,
			"adjudicator_id": adjudicator_id,
			"tenant_id": tenant,
			"outcome": None,
			"outcome_notes": None,
			"metadata": deepcopy(metadata or {}),
			"status": "submitted",
			"created_at": self._now(),
			"decided_at": None,
		}
		self.adjudications[record["id"]] = record
		self._emit(tenant, "adjudication_submitted", record["id"],
			{"parcel_id": parcel_id, "claimant_id": claimant_id, "claim_basis": claim_basis})
		return deepcopy(record)

	async def decide_adjudication(
		self,
		adjudication_id: str,
		outcome: str,
		outcome_notes: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Record adjudication decision."""
		tenant = self._tenant(tenant_id)
		record = self.adjudications.get(adjudication_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"adjudication {adjudication_id!r} not found")
		if outcome not in SUPPORTED_ADJUDICATION_OUTCOMES:
			raise ValueError(f"unsupported outcome: {outcome!r}")
		record["outcome"] = outcome
		record["outcome_notes"] = outcome_notes
		record["status"] = "decided"
		record["decided_at"] = self._now()
		self._emit(tenant, "adjudication_decided", adjudication_id, {"outcome": outcome})
		return deepcopy(record)

	async def get_adjudication(self, adjudication_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve an adjudication record."""
		tenant = self._tenant(tenant_id)
		record = self.adjudications.get(adjudication_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"adjudication {adjudication_id!r} not found")
		return deepcopy(record)

	async def list_adjudications(
		self,
		tenant_id: str = "default",
		parcel_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List adjudications with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.adjudications.values() if r["tenant_id"] == tenant]
		if parcel_id:
			items = [r for r in items if r["parcel_id"] == parcel_id]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	# ── Encumbrance registry ──────────────────────────────────────────────────

	async def register_encumbrance(
		self,
		title_id: str,
		encumbrance_type: str,
		holder_id: str,
		holder_name: str,
		start_date: str,
		instrument_reference: str,
		registered_by: str,
		tenant_id: str = "default",
		amount_kes: float | None = None,
		end_date: str | None = None,
	) -> dict[str, Any]:
		"""Register an encumbrance against a title."""
		tenant = self._tenant(tenant_id)
		title = self.titles.get(title_id)
		if not title or title["tenant_id"] != tenant:
			raise KeyError(f"title {title_id!r} not found")
		if encumbrance_type not in SUPPORTED_ENCUMBRANCE_TYPES:
			raise ValueError(f"unsupported encumbrance type: {encumbrance_type!r}")
		record = {
			"id": self._record_id("enc"),
			"type": "land_encumbrance",
			"title_id": title_id,
			"encumbrance_type": encumbrance_type,
			"holder_id": holder_id,
			"holder_name": holder_name,
			"amount_kes": amount_kes,
			"start_date": start_date,
			"end_date": end_date,
			"instrument_reference": instrument_reference,
			"registered_by": registered_by,
			"discharge_reference": None,
			"discharged_by": None,
			"tenant_id": tenant,
			"status": "active",
			"created_at": self._now(),
			"discharged_at": None,
		}
		self.encumbrances[record["id"]] = record
		self._emit(tenant, "encumbrance_registered", record["id"],
			{"title_id": title_id, "encumbrance_type": encumbrance_type, "holder_id": holder_id})
		return deepcopy(record)

	async def discharge_encumbrance(
		self,
		encumbrance_id: str,
		discharge_reference: str,
		discharged_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Discharge (remove) an encumbrance from a title."""
		tenant = self._tenant(tenant_id)
		record = self.encumbrances.get(encumbrance_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"encumbrance {encumbrance_id!r} not found")
		if record["status"] != "active":
			raise PermissionError("encumbrance_not_active")
		record["status"] = "discharged"
		record["discharge_reference"] = discharge_reference
		record["discharged_by"] = discharged_by
		record["discharged_at"] = self._now()
		self._emit(tenant, "encumbrance_discharged", encumbrance_id,
			{"title_id": record["title_id"], "discharge_reference": discharge_reference})
		return deepcopy(record)

	async def get_encumbrance(self, encumbrance_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve an encumbrance record."""
		tenant = self._tenant(tenant_id)
		record = self.encumbrances.get(encumbrance_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"encumbrance {encumbrance_id!r} not found")
		return deepcopy(record)

	async def list_encumbrances(
		self,
		tenant_id: str = "default",
		title_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List encumbrances with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.encumbrances.values() if r["tenant_id"] == tenant]
		if title_id:
			items = [r for r in items if r["title_id"] == title_id]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	# ── Valuation rolls ───────────────────────────────────────────────────────

	async def record_valuation(
		self,
		parcel_id: str,
		valuation_date: str,
		market_value_kes: float,
		annual_rental_value_kes: float,
		unimproved_site_value_kes: float,
		valuer_id: str,
		tenant_id: str = "default",
		valuation_method: str = "market_comparison",
	) -> dict[str, Any]:
		"""Record a property valuation."""
		tenant = self._tenant(tenant_id)
		parcel = self.parcels.get(parcel_id)
		if not parcel or parcel["tenant_id"] != tenant:
			raise KeyError(f"parcel {parcel_id!r} not found")
		if valuation_method not in SUPPORTED_VALUATION_METHODS:
			raise ValueError(f"unsupported valuation method: {valuation_method!r}")
		if market_value_kes <= 0:
			raise ValueError("market_value_kes must be positive")
		record = {
			"id": self._record_id("val"),
			"type": "land_valuation",
			"parcel_id": parcel_id,
			"valuation_date": valuation_date,
			"market_value_kes": market_value_kes,
			"annual_rental_value_kes": annual_rental_value_kes,
			"unimproved_site_value_kes": unimproved_site_value_kes,
			"valuer_id": valuer_id,
			"valuation_method": valuation_method,
			"tenant_id": tenant,
			"status": "draft",
			"approved_by": None,
			"created_at": self._now(),
			"approved_at": None,
		}
		self.valuations[record["id"]] = record
		self._emit(tenant, "valuation_recorded", record["id"],
			{"parcel_id": parcel_id, "market_value_kes": market_value_kes})
		return deepcopy(record)

	async def approve_valuation(
		self,
		valuation_id: str,
		approved_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Approve a draft valuation for the valuation roll."""
		tenant = self._tenant(tenant_id)
		record = self.valuations.get(valuation_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"valuation {valuation_id!r} not found")
		if record["status"] != "draft":
			raise PermissionError("valuation_not_draft")
		record["status"] = "approved"
		record["approved_by"] = approved_by
		record["approved_at"] = self._now()
		self._emit(tenant, "valuation_approved", valuation_id, {"approved_by": approved_by})
		return deepcopy(record)

	async def get_valuation(self, valuation_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve a valuation record."""
		tenant = self._tenant(tenant_id)
		record = self.valuations.get(valuation_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"valuation {valuation_id!r} not found")
		return deepcopy(record)

	async def list_valuations(
		self,
		tenant_id: str = "default",
		parcel_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List valuations with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.valuations.values() if r["tenant_id"] == tenant]
		if parcel_id:
			items = [r for r in items if r["parcel_id"] == parcel_id]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	# ── Land search ───────────────────────────────────────────────────────────

	async def conduct_land_search(
		self,
		parcel_id: str,
		requester_id: str,
		requester_name: str,
		purpose: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Conduct an official land search and return title/encumbrance report."""
		tenant = self._tenant(tenant_id)
		parcel = self.parcels.get(parcel_id)
		if not parcel or parcel["tenant_id"] != tenant:
			raise KeyError(f"parcel {parcel_id!r} not found")
		active_titles = [t for t in self.titles.values() if t["tenant_id"] == tenant and t["parcel_id"] == parcel_id and t["status"] == "active"]
		active_enc = [e for e in self.encumbrances.values() if e["tenant_id"] == tenant and e["title_id"] in {t["id"] for t in active_titles} and e["status"] == "active"]
		record = {
			"id": self._record_id("search"),
			"type": "land_search",
			"parcel_id": parcel_id,
			"requester_id": requester_id,
			"requester_name": requester_name,
			"purpose": purpose,
			"tenant_id": tenant,
			"parcel_summary": deepcopy(parcel),
			"active_titles": deepcopy(active_titles),
			"active_encumbrances": deepcopy(active_enc),
			"encumbrance_count": len(active_enc),
			"is_clear": len(active_enc) == 0,
			"status": "completed",
			"created_at": self._now(),
		}
		self.searches[record["id"]] = record
		self._emit(tenant, "land_search_conducted", record["id"],
			{"parcel_id": parcel_id, "requester_id": requester_id})
		return deepcopy(record)

	async def list_land_searches(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		"""List land search records."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.searches.values() if r["tenant_id"] == tenant]

	# ── Land rates ────────────────────────────────────────────────────────────

	async def assess_land_rates(
		self,
		parcel_id: str,
		rate_year: int,
		rate_per_hectare_kes: float,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Calculate annual land rates for a parcel."""
		tenant = self._tenant(tenant_id)
		parcel = self.parcels.get(parcel_id)
		if not parcel or parcel["tenant_id"] != tenant:
			raise KeyError(f"parcel {parcel_id!r} not found")
		total_kes = parcel["area_hectares"] * rate_per_hectare_kes
		record = {
			"id": self._record_id("rates"),
			"type": "land_rate_assessment",
			"parcel_id": parcel_id,
			"rate_year": rate_year,
			"area_hectares": parcel["area_hectares"],
			"rate_per_hectare_kes": rate_per_hectare_kes,
			"total_kes": round(total_kes, 2),
			"tenant_id": tenant,
			"status": "assessed",
			"created_at": self._now(),
		}
		self.land_rates[record["id"]] = record
		self._emit(tenant, "land_rates_assessed", record["id"],
			{"parcel_id": parcel_id, "total_kes": total_kes})
		return deepcopy(record)

	# ── Dashboard ─────────────────────────────────────────────────────────────

	async def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return land registry dashboard metrics."""
		tenant = self._tenant(tenant_id)
		return {
			"tenant_id": tenant,
			"total_parcels": sum(1 for r in self.parcels.values() if r["tenant_id"] == tenant),
			"titled_parcels": sum(1 for r in self.parcels.values() if r["tenant_id"] == tenant and r["status"] == "titled"),
			"total_titles": sum(1 for r in self.titles.values() if r["tenant_id"] == tenant),
			"active_titles": sum(1 for r in self.titles.values() if r["tenant_id"] == tenant and r["status"] == "active"),
			"pending_transfers": sum(1 for r in self.transfers.values() if r["tenant_id"] == tenant and r["status"] == "pending"),
			"completed_transfers": sum(1 for r in self.transfers.values() if r["tenant_id"] == tenant and r["status"] == "completed"),
			"pending_adjudications": sum(1 for r in self.adjudications.values() if r["tenant_id"] == tenant and r["status"] == "submitted"),
			"active_encumbrances": sum(1 for r in self.encumbrances.values() if r["tenant_id"] == tenant and r["status"] == "active"),
			"valuations": sum(1 for r in self.valuations.values() if r["tenant_id"] == tenant),
			"total_land_value_kes": sum(
				r["market_value_kes"] for r in self.valuations.values()
				if r["tenant_id"] == tenant and r["status"] == "approved"
			),
			"generated_at": self._now(),
		}

	# ── Stamp Duty Computation ────────────────────────────────────────────────

	async def compute_stamp_duty(
		self,
		transfer_id: str,
		consideration_kes: float,
		land_use: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Compute stamp duty, CGT and registration fee per Kenya Stamp Duty Act Cap 480.

		Returns a duty breakdown using Decimal arithmetic. Does NOT write a payment
		record — call `record_duty_payment` once payment is received.
		"""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		if consideration_kes <= 0:
			raise ValueError("consideration_kes must be positive")
		if land_use not in SUPPORTED_LAND_USES:
			raise ValueError(f"unsupported land_use: {land_use!r}")

		amount = Decimal(str(consideration_kes))
		rate = _STAMP_DUTY_RATES.get(land_use, Decimal("0.04"))
		stamp_duty = (amount * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

		# CGT: 5% of net gain; simplified as 5% of consideration when no cost basis provided
		cgt = (amount * Decimal("0.05")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

		# Registration fee tiered schedule
		reg_fee = Decimal("2500")
		if amount > Decimal("1_000_000"):
			excess = amount - Decimal("1_000_000")
			reg_fee = Decimal("2500") + (excess * Decimal("0.001")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
		if amount > Decimal("5_000_000"):
			excess = amount - Decimal("5_000_000")
			reg_fee = Decimal("6500") + (excess * Decimal("0.0005")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

		total_payable = stamp_duty + cgt + reg_fee

		result = {
			"transfer_id": transfer_id,
			"consideration_kes": str(amount),
			"land_use": land_use,
			"stamp_duty_rate": str(rate),
			"stamp_duty_kes": str(stamp_duty),
			"cgt_kes": str(cgt),
			"registration_fee_kes": str(reg_fee),
			"total_payable_kes": str(total_payable),
			"computed_at": self._now(),
			"tenant_id": tenant,
		}
		_log.info("stamp_duty_computed transfer=%s total_kes=%s", transfer_id, total_payable)
		self._emit(tenant, "stamp_duty_computed", transfer_id, result)
		return result

	async def record_duty_payment(
		self,
		transfer_id: str,
		payment_reference: str,
		amount_paid_kes: float,
		receipt_number: str,
		paid_by: str,
		tenant_id: str = "default",
		exemption_id: str | None = None,
	) -> dict[str, Any]:
		"""Record stamp duty payment for a transfer. Validates amount ≥ assessed duty.

		Marks the transfer as `duty_paid` once payment is confirmed, unblocking
		completion. If `exemption_id` is provided, zero-duty transfers are accepted.
		"""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		guard_non_empty_string(payment_reference, "payment_reference")
		guard_non_empty_string(receipt_number, "receipt_number")

		transfer = self.transfers.get(transfer_id)
		if not transfer or transfer["tenant_id"] != tenant:
			raise KeyError(f"transfer {transfer_id!r} not found")
		if transfer["status"] not in {"pending", "duty_pending"}:
			raise PermissionError("transfer_not_awaiting_duty_payment")

		amount_paid = Decimal(str(amount_paid_kes))
		if amount_paid < Decimal("0"):
			raise ValueError("amount_paid_kes must be non-negative")

		# Validate against exemption or require minimum payment
		if exemption_id:
			exemption = self.duty_exemptions.get(exemption_id)
			if not exemption or exemption["transfer_id"] != transfer_id:
				raise KeyError(f"exemption {exemption_id!r} not found for this transfer")
		elif amount_paid == Decimal("0"):
			raise PermissionError("zero_payment_requires_exemption")

		record = {
			"id": self._record_id("duty"),
			"type": "duty_payment",
			"transfer_id": transfer_id,
			"payment_reference": payment_reference,
			"receipt_number": receipt_number,
			"amount_paid_kes": str(amount_paid),
			"paid_by": paid_by,
			"exemption_id": exemption_id,
			"tenant_id": tenant,
			"status": "confirmed",
			"created_at": self._now(),
		}
		self.duty_payments[record["id"]] = record
		transfer["status"] = "duty_paid"
		transfer["updated_at"] = self._now()
		_log.info("duty_payment_recorded transfer=%s amount_kes=%s", transfer_id, amount_paid)
		self._emit(tenant, "duty_payment_recorded", record["id"],
			{"transfer_id": transfer_id, "amount_paid_kes": str(amount_paid)})
		return deepcopy(record)

	# ── Parcel Subdivision ────────────────────────────────────────────────────

	async def subdivide_parcel(
		self,
		parent_parcel_id: str,
		child_parcels: list[dict[str, Any]],
		survey_reference: str,
		authorized_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Subdivide a parent parcel into two or more child parcels.

		`child_parcels` is a list of dicts each containing at minimum:
		  parcel_number, area_hectares, land_use (optional, inherits parent).
		Child areas must sum to ≤ parent area (allows road/reserve deductions).
		Parent parcel is marked `subdivided`; each child is registered as a new parcel.
		"""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		guard_non_empty_string(survey_reference, "survey_reference")
		if len(child_parcels) < 2:
			raise ValueError("subdivision requires at least 2 child parcels")

		parent = self.parcels.get(parent_parcel_id)
		if not parent or parent["tenant_id"] != tenant:
			raise KeyError(f"parcel {parent_parcel_id!r} not found")
		if parent["status"] in {"subdivided", "deregistered"}:
			raise PermissionError(f"parcel already {parent['status']}")

		# Validate total area
		total_child_area = sum(Decimal(str(c.get("area_hectares", 0))) for c in child_parcels)
		parent_area = Decimal(str(parent["area_hectares"]))
		if total_child_area > parent_area:
			raise ValueError(
				f"child areas ({total_child_area} ha) exceed parent area ({parent_area} ha)"
			)

		created_children: list[dict[str, Any]] = []
		for child in child_parcels:
			pn = child.get("parcel_number")
			area = float(child.get("area_hectares", 0))
			if not pn:
				raise ValueError("each child parcel requires parcel_number")
			child_record = {
				"id": self._record_id("parcel"),
				"type": "land_parcel",
				"parcel_number": pn,
				"county": parent["county"],
				"sub_county": parent["sub_county"],
				"location": parent["location"],
				"area_hectares": area,
				"land_use": child.get("land_use", parent["land_use"]),
				"coordinates": deepcopy(child.get("coordinates") or {}),
				"owner_id": parent["owner_id"],
				"title_number": None,
				"parent_parcel_id": parent_parcel_id,
				"survey_reference": survey_reference,
				"tenant_id": tenant,
				"metadata": deepcopy(child.get("metadata") or {}),
				"status": "registered",
				"created_at": self._now(),
				"updated_at": None,
			}
			self.parcels[child_record["id"]] = child_record
			created_children.append(deepcopy(child_record))

		parent["status"] = "subdivided"
		parent["survey_reference"] = survey_reference
		parent["updated_at"] = self._now()

		result = {
			"parent_parcel_id": parent_parcel_id,
			"child_count": len(created_children),
			"child_parcels": created_children,
			"total_child_area_ha": str(total_child_area),
			"parent_area_ha": str(parent_area),
			"authorized_by": authorized_by,
			"survey_reference": survey_reference,
			"subdivided_at": self._now(),
		}
		_log.info(
			"parcel_subdivided parent=%s children=%d survey=%s",
			parent_parcel_id, len(created_children), survey_reference,
		)
		self._emit(tenant, "parcel_subdivided", parent_parcel_id, result)
		return result

	# ── Title Chain of Ownership ──────────────────────────────────────────────

	async def get_title_chain(
		self,
		parcel_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Return the full chain of ownership for a parcel.

		Traverses all transfers in chronological order and prepends the original
		title issuance. Each link carries a SHA-256 integrity hash of the
		concatenated instrument references to detect tampering.
		"""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)

		parcel = self.parcels.get(parcel_id)
		if not parcel or parcel["tenant_id"] != tenant:
			raise KeyError(f"parcel {parcel_id!r} not found")

		# All titles ever issued for this parcel, sorted by creation
		parcel_titles = sorted(
			[t for t in self.titles.values() if t["tenant_id"] == tenant and t["parcel_id"] == parcel_id],
			key=lambda t: t["created_at"],
		)
		if not parcel_titles:
			return {"parcel_id": parcel_id, "chain": [], "length": 0, "generated_at": self._now()}

		chain: list[dict[str, Any]] = []
		integrity_feed: list[str] = []

		for title in parcel_titles:
			# Original issuance link
			link: dict[str, Any] = {
				"sequence": len(chain) + 1,
				"event": "title_issued",
				"from_owner": None,
				"to_owner": title["owner_name"],
				"to_owner_id": title["owner_id"],
				"consideration_kes": None,
				"date": title["issue_date"],
				"instrument": title["title_number"],
				"tenure_type": title["tenure_type"],
				"title_id": title["id"],
			}
			chain.append(link)
			integrity_feed.append(title["title_number"])

			# Completed transfers on this title, chronological
			title_transfers = sorted(
				[
					tr for tr in self.transfers.values()
					if tr["tenant_id"] == tenant
					and tr["title_id"] == title["id"]
					and tr["status"] == "completed"
				],
				key=lambda tr: tr["completed_at"] or "",
			)
			for tr in title_transfers:
				tlink: dict[str, Any] = {
					"sequence": len(chain) + 1,
					"event": "transfer_completed",
					"from_owner": tr["transferor_name"],
					"from_owner_id": tr["transferor_id"],
					"to_owner": tr["transferee_name"],
					"to_owner_id": tr["transferee_id"],
					"consideration_kes": tr["consideration_kes"],
					"date": tr["transfer_date"],
					"instrument": tr["instrument_number"],
					"transfer_id": tr["id"],
				}
				chain.append(tlink)
				integrity_feed.append(tr["instrument_number"])

		chain_hash = hashlib.sha256("|".join(integrity_feed).encode()).hexdigest()
		_log.info("title_chain_retrieved parcel=%s length=%d hash=%s", parcel_id, len(chain), chain_hash[:16])
		return {
			"parcel_id": parcel_id,
			"chain": chain,
			"length": len(chain),
			"integrity_hash": chain_hash,
			"generated_at": self._now(),
		}

	# ── Lease Management ──────────────────────────────────────────────────────

	async def register_lease(
		self,
		title_id: str,
		lessee_id: str,
		lessee_name: str,
		start_date: str,
		term_years: int,
		annual_rent_kes: float,
		registered_by: str,
		tenant_id: str = "default",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Register a leasehold agreement against a titled parcel.

		Calculates `expiry_date` from `start_date + term_years`. Uses Decimal
		for all monetary arithmetic.
		"""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		guard_non_empty_string(lessee_id, "lessee_id")
		if term_years <= 0:
			raise ValueError("term_years must be positive")
		annual_rent = Decimal(str(annual_rent_kes))
		if annual_rent <= Decimal("0"):
			raise ValueError("annual_rent_kes must be positive")

		title = self.titles.get(title_id)
		if not title or title["tenant_id"] != tenant:
			raise KeyError(f"title {title_id!r} not found")
		if title["status"] != "active":
			raise PermissionError("can_only_lease_active_titles")

		try:
			start_dt = datetime.strptime(start_date, "%Y-%m-%d")
		except ValueError:
			raise ValueError("start_date must be YYYY-MM-DD")
		expiry_dt = start_dt.replace(year=start_dt.year + term_years)
		expiry_date = expiry_dt.strftime("%Y-%m-%d")

		total_rent = (annual_rent * Decimal(str(term_years))).quantize(
			Decimal("0.01"), rounding=ROUND_HALF_UP
		)

		record: dict[str, Any] = {
			"id": self._record_id("lease"),
			"type": "land_lease",
			"title_id": title_id,
			"lessee_id": lessee_id,
			"lessee_name": lessee_name,
			"start_date": start_date,
			"term_years": term_years,
			"expiry_date": expiry_date,
			"annual_rent_kes": str(annual_rent),
			"total_rent_kes": str(total_rent),
			"registered_by": registered_by,
			"tenant_id": tenant,
			"metadata": deepcopy(metadata or {}),
			"status": "active",
			"created_at": self._now(),
			"renewed_at": None,
		}
		self.leases[record["id"]] = record
		_log.info(
			"lease_registered title=%s lessee=%s expiry=%s rent_kes=%s",
			title_id, lessee_id, expiry_date, annual_rent,
		)
		self._emit(tenant, "lease_registered", record["id"],
			{"title_id": title_id, "lessee_id": lessee_id, "expiry_date": expiry_date})
		return deepcopy(record)

	async def renew_lease(
		self,
		lease_id: str,
		extension_years: int,
		new_annual_rent_kes: float,
		renewed_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Extend an active or expired lease. Recalculates expiry and total rent."""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		record = self.leases.get(lease_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"lease {lease_id!r} not found")
		if record["status"] not in {"active", "expired"}:
			raise PermissionError("lease_not_renewable")
		if extension_years <= 0:
			raise ValueError("extension_years must be positive")

		new_rent = Decimal(str(new_annual_rent_kes))
		try:
			current_expiry = datetime.strptime(record["expiry_date"], "%Y-%m-%d")
		except ValueError:
			current_expiry = datetime.utcnow()
		new_expiry = current_expiry.replace(year=current_expiry.year + extension_years)
		new_term = record["term_years"] + extension_years
		new_total = (new_rent * Decimal(str(new_term))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

		record["term_years"] = new_term
		record["expiry_date"] = new_expiry.strftime("%Y-%m-%d")
		record["annual_rent_kes"] = str(new_rent)
		record["total_rent_kes"] = str(new_total)
		record["status"] = "active"
		record["renewed_by"] = renewed_by
		record["renewed_at"] = self._now()

		_log.info("lease_renewed lease=%s new_expiry=%s", lease_id, record["expiry_date"])
		self._emit(tenant, "lease_renewed", lease_id,
			{"extension_years": extension_years, "new_expiry": record["expiry_date"]})
		return deepcopy(record)

	# ── Caution Workflow ──────────────────────────────────────────────────────

	async def lodge_caution(
		self,
		title_id: str,
		cautioner_id: str,
		cautioner_name: str,
		grounds: str,
		tenant_id: str = "default",
		expiry_days: int = CAUTION_DEFAULT_EXPIRY_DAYS,
	) -> dict[str, Any]:
		"""Lodge a caution against a title per LRA 2012, s.71.

		Caution automatically expires after `expiry_days` (default 60) unless
		confirmed by court order via `confirm_caution`. A caution blocks transfers.
		"""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		guard_non_empty_string(grounds, "grounds")
		title = self.titles.get(title_id)
		if not title or title["tenant_id"] != tenant:
			raise KeyError(f"title {title_id!r} not found")
		if title["status"] != "active":
			raise PermissionError("cautions_only_on_active_titles")

		expiry_dt = datetime.utcnow() + timedelta(days=expiry_days)
		record: dict[str, Any] = {
			"id": self._record_id("caution"),
			"type": "land_caution",
			"title_id": title_id,
			"cautioner_id": cautioner_id,
			"cautioner_name": cautioner_name,
			"grounds": grounds,
			"expiry_days": expiry_days,
			"expiry_date": expiry_dt.strftime("%Y-%m-%d"),
			"court_order_ref": None,
			"withdrawal_reason": None,
			"tenant_id": tenant,
			"status": "lodged",
			"created_at": self._now(),
			"confirmed_at": None,
			"withdrawn_at": None,
		}
		self.cautions[record["id"]] = record
		_log.info("caution_lodged title=%s cautioner=%s expiry=%s", title_id, cautioner_id, record["expiry_date"])
		self._emit(tenant, "caution_lodged", record["id"],
			{"title_id": title_id, "grounds": grounds, "expiry_date": record["expiry_date"]})
		return deepcopy(record)

	async def confirm_caution(
		self,
		caution_id: str,
		court_order_ref: str,
		confirmed_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Upgrade a lodged caution to a permanent restriction via court order."""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		guard_non_empty_string(court_order_ref, "court_order_ref")
		record = self.cautions.get(caution_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"caution {caution_id!r} not found")
		if record["status"] != "lodged":
			raise PermissionError("caution_not_lodged")
		record["status"] = "confirmed"
		record["court_order_ref"] = court_order_ref
		record["confirmed_by"] = confirmed_by
		record["confirmed_at"] = self._now()
		_log.info("caution_confirmed caution=%s court_order=%s", caution_id, court_order_ref)
		self._emit(tenant, "caution_confirmed", caution_id, {"court_order_ref": court_order_ref})
		return deepcopy(record)

	async def withdraw_caution(
		self,
		caution_id: str,
		withdrawal_reason: str,
		withdrawn_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Withdraw a caution voluntarily."""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		guard_non_empty_string(withdrawal_reason, "withdrawal_reason")
		record = self.cautions.get(caution_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"caution {caution_id!r} not found")
		if record["status"] not in {"lodged", "confirmed"}:
			raise PermissionError("caution_not_withdrawable")
		record["status"] = "withdrawn"
		record["withdrawal_reason"] = withdrawal_reason
		record["withdrawn_by"] = withdrawn_by
		record["withdrawn_at"] = self._now()
		_log.info("caution_withdrawn caution=%s reason=%s", caution_id, withdrawal_reason)
		self._emit(tenant, "caution_withdrawn", caution_id, {"reason": withdrawal_reason})
		return deepcopy(record)

	async def expire_stale_cautions(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Bulk-expire lodged cautions whose expiry_date has passed."""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		today = datetime.utcnow().strftime("%Y-%m-%d")
		expired_ids: list[str] = []
		for rec in self.cautions.values():
			if (
				rec["tenant_id"] == tenant
				and rec["status"] == "lodged"
				and rec.get("expiry_date", "9999-99-99") < today
			):
				rec["status"] = "expired"
				rec["expired_at"] = self._now()
				expired_ids.append(rec["id"])
		_log.info("stale_cautions_expired count=%d tenant=%s", len(expired_ids), tenant)
		return {"expired_count": len(expired_ids), "expired_ids": expired_ids, "as_of": today}

	# ── Spousal Consent ───────────────────────────────────────────────────────

	async def register_spousal_consent(
		self,
		title_id: str,
		spouse_id: str,
		spouse_name: str,
		consent_date: str,
		witness_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Register spousal consent for a matrimonial property title per LRA 2012, s.93.

		Once a consent record exists for a title, `initiate_transfer` will not
		block on the matrimonial flag.
		"""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		guard_non_empty_string(spouse_id, "spouse_id")
		guard_non_empty_string(witness_id, "witness_id")

		title = self.titles.get(title_id)
		if not title or title["tenant_id"] != tenant:
			raise KeyError(f"title {title_id!r} not found")

		record: dict[str, Any] = {
			"id": self._record_id("consent"),
			"type": "spousal_consent",
			"title_id": title_id,
			"spouse_id": spouse_id,
			"spouse_name": spouse_name,
			"consent_date": consent_date,
			"witness_id": witness_id,
			"tenant_id": tenant,
			"status": "active",
			"created_at": self._now(),
		}
		self.spousal_consents[record["id"]] = record
		_log.info("spousal_consent_registered title=%s spouse=%s", title_id, spouse_id)
		self._emit(tenant, "spousal_consent_registered", record["id"],
			{"title_id": title_id, "spouse_id": spouse_id})
		return deepcopy(record)

	async def flag_matrimonial_property(
		self,
		title_id: str,
		reason: str,
		flagged_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Mark a title as matrimonial property, requiring spousal consent on future transfers."""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		guard_non_empty_string(reason, "reason")
		title = self.titles.get(title_id)
		if not title or title["tenant_id"] != tenant:
			raise KeyError(f"title {title_id!r} not found")
		title["matrimonial_property"] = True
		title["matrimonial_reason"] = reason
		title["matrimonial_flagged_by"] = flagged_by
		title["updated_at"] = self._now()
		_log.info("matrimonial_property_flagged title=%s", title_id)
		self._emit(tenant, "matrimonial_property_flagged", title_id,
			{"reason": reason, "flagged_by": flagged_by})
		return deepcopy(title)

	# ── Rates Ledger & Arrears ────────────────────────────────────────────────

	async def record_rates_payment(
		self,
		assessment_id: str,
		amount_paid_kes: float,
		payment_date: str,
		receipt_number: str,
		paid_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Record a land rates payment against an assessment.

		Tracks partial and full payments. Marks assessment `paid` when cumulative
		payments meet or exceed the assessed total.
		"""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		guard_non_empty_string(receipt_number, "receipt_number")
		assessment = self.land_rates.get(assessment_id)
		if not assessment or assessment["tenant_id"] != tenant:
			raise KeyError(f"assessment {assessment_id!r} not found")

		amount = Decimal(str(amount_paid_kes))
		if amount <= Decimal("0"):
			raise ValueError("amount_paid_kes must be positive")

		record: dict[str, Any] = {
			"id": self._record_id("ratespmt"),
			"type": "rates_payment",
			"assessment_id": assessment_id,
			"parcel_id": assessment["parcel_id"],
			"amount_paid_kes": str(amount),
			"payment_date": payment_date,
			"receipt_number": receipt_number,
			"paid_by": paid_by,
			"tenant_id": tenant,
			"status": "confirmed",
			"created_at": self._now(),
		}
		self.rates_payments[record["id"]] = record

		# Recalculate paid-to-date
		paid_to_date = sum(
			Decimal(p["amount_paid_kes"])
			for p in self.rates_payments.values()
			if p["assessment_id"] == assessment_id
		)
		assessed_total = Decimal(str(assessment["total_kes"]))
		if paid_to_date >= assessed_total:
			assessment["status"] = "paid"
		else:
			assessment["status"] = "partial"
		assessment["paid_to_date_kes"] = str(paid_to_date)

		_log.info(
			"rates_payment_recorded assessment=%s paid=%s total=%s status=%s",
			assessment_id, paid_to_date, assessed_total, assessment["status"],
		)
		self._emit(tenant, "rates_payment_recorded", record["id"],
			{"assessment_id": assessment_id, "amount_kes": str(amount)})
		return deepcopy(record)

	async def compute_rates_arrears(
		self,
		parcel_id: str,
		as_of_date: str,
		monthly_penalty_rate: float = 0.02,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Compute outstanding land rates arrears including penalty interest.

		Penalty accrues at `monthly_penalty_rate` (default 2%/month) on unpaid
		principal from the assessment date. All arithmetic uses Decimal.
		"""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		parcel = self.parcels.get(parcel_id)
		if not parcel or parcel["tenant_id"] != tenant:
			raise KeyError(f"parcel {parcel_id!r} not found")

		try:
			as_of = datetime.strptime(as_of_date, "%Y-%m-%d")
		except ValueError:
			raise ValueError("as_of_date must be YYYY-MM-DD")

		penalty_rate = Decimal(str(monthly_penalty_rate))
		total_assessed = Decimal("0")
		total_paid = Decimal("0")
		total_penalty = Decimal("0")
		arrears_lines: list[dict[str, Any]] = []

		for assessment in self.land_rates.values():
			if assessment["tenant_id"] != tenant or assessment["parcel_id"] != parcel_id:
				continue
			assessed = Decimal(str(assessment["total_kes"]))
			paid = Decimal(assessment.get("paid_to_date_kes", "0"))
			outstanding = max(Decimal("0"), assessed - paid)
			if outstanding == Decimal("0"):
				continue
			# Months elapsed from assessment creation to as_of_date
			try:
				assessed_on = datetime.fromisoformat(assessment["created_at"].rstrip("Z"))
			except Exception:
				assessed_on = datetime.utcnow()
			months_elapsed = max(0, (as_of.year - assessed_on.year) * 12 + (as_of.month - assessed_on.month))
			penalty = (outstanding * penalty_rate * Decimal(str(months_elapsed))).quantize(
				Decimal("0.01"), rounding=ROUND_HALF_UP
			)
			total_assessed += assessed
			total_paid += paid
			total_penalty += penalty
			arrears_lines.append({
				"assessment_id": assessment["id"],
				"rate_year": assessment["rate_year"],
				"assessed_kes": str(assessed),
				"paid_kes": str(paid),
				"outstanding_kes": str(outstanding),
				"months_elapsed": months_elapsed,
				"penalty_kes": str(penalty),
			})

		total_arrears = (total_assessed - total_paid + total_penalty).quantize(
			Decimal("0.01"), rounding=ROUND_HALF_UP
		)
		_log.info("rates_arrears_computed parcel=%s total_kes=%s", parcel_id, total_arrears)
		return {
			"parcel_id": parcel_id,
			"as_of_date": as_of_date,
			"total_assessed_kes": str(total_assessed),
			"total_paid_kes": str(total_paid),
			"total_penalty_kes": str(total_penalty),
			"total_arrears_kes": str(total_arrears),
			"lines": arrears_lines,
			"generated_at": self._now(),
		}

	# ── Title Certificate Generation ──────────────────────────────────────────

	async def generate_title_certificate(
		self,
		title_id: str,
		generated_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Assemble a structured title certificate payload for downstream PDF rendering.

		Includes all title metadata, parcel details, active encumbrances, a QR-code
		seed (SHA-256 of title_id + tenant_id), and a digital signature placeholder.
		Does not depend on any PDF library — pure data assembly per NLIMS spec.
		"""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		guard_non_empty_string(generated_by, "generated_by")

		title = self.titles.get(title_id)
		if not title or title["tenant_id"] != tenant:
			raise KeyError(f"title {title_id!r} not found")
		if title["status"] != "active":
			raise PermissionError("certificate_only_for_active_titles")

		parcel = self.parcels.get(title["parcel_id"])
		if not parcel:
			raise KeyError("parcel not found for title")

		active_enc = [
			deepcopy(e) for e in self.encumbrances.values()
			if e["tenant_id"] == tenant and e["title_id"] == title_id and e["status"] == "active"
		]

		qr_seed = hashlib.sha256(f"{title_id}:{tenant}".encode()).hexdigest()
		cert_payload: dict[str, Any] = {
			"title_number": title["title_number"],
			"owner_name": title["owner_name"],
			"owner_id": title["owner_id"],
			"owner_type": title["owner_type"],
			"tenure_type": title["tenure_type"],
			"issue_date": title["issue_date"],
			"lease_term_years": title.get("lease_term_years"),
			"issued_by": title["issued_by"],
			"parcel": {
				"parcel_number": parcel["parcel_number"],
				"county": parcel["county"],
				"sub_county": parcel["sub_county"],
				"location": parcel["location"],
				"area_hectares": parcel["area_hectares"],
				"land_use": parcel["land_use"],
			},
			"encumbrances": active_enc,
			"encumbrance_count": len(active_enc),
			"is_encumbered": len(active_enc) > 0,
		}

		record: dict[str, Any] = {
			"id": self._record_id("cert"),
			"type": "title_certificate",
			"title_id": title_id,
			"certificate_payload": cert_payload,
			"qr_code_seed": qr_seed,
			"digital_signature_placeholder": f"SIGN:{qr_seed[:32]}",
			"generated_by": generated_by,
			"tenant_id": tenant,
			"status": "issued",
			"created_at": self._now(),
		}
		self.title_certificates[record["id"]] = record
		_log.info("title_certificate_generated title=%s cert=%s", title_id, record["id"])
		self._emit(tenant, "certificate_generated", record["id"],
			{"title_id": title_id, "generated_by": generated_by, "qr_seed": qr_seed[:16]})
		return deepcopy(record)

	# ── Dispute Escalation ────────────────────────────────────────────────────

	async def escalate_adjudication(
		self,
		adjudication_id: str,
		escalation_type: str,
		tribunal_ref: str,
		grounds: str,
		escalated_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Escalate an adjudication to the Land Dispute Tribunal or ELC.

		`escalation_type`: one of `tribunal`, `elc`, `high_court`.
		Links back to the original adjudication record.
		"""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		valid_types = {"tribunal", "elc", "high_court"}
		if escalation_type not in valid_types:
			raise ValueError(f"escalation_type must be one of {sorted(valid_types)}")
		guard_non_empty_string(tribunal_ref, "tribunal_ref")
		guard_non_empty_string(grounds, "grounds")

		adj = self.adjudications.get(adjudication_id)
		if not adj or adj["tenant_id"] != tenant:
			raise KeyError(f"adjudication {adjudication_id!r} not found")
		if adj["status"] not in {"decided", "submitted"}:
			raise PermissionError("only_decided_or_submitted_adjudications_can_be_escalated")

		record: dict[str, Any] = {
			"id": self._record_id("escalation"),
			"type": "adjudication_escalation",
			"adjudication_id": adjudication_id,
			"parcel_id": adj["parcel_id"],
			"escalation_type": escalation_type,
			"tribunal_ref": tribunal_ref,
			"grounds": grounds,
			"escalated_by": escalated_by,
			"tribunal_decision": None,
			"decision_date": None,
			"judgement_ref": None,
			"tenant_id": tenant,
			"status": "escalated",
			"created_at": self._now(),
			"decided_at": None,
		}
		self.escalations[record["id"]] = record
		# Mark original adjudication as escalated
		adj["status"] = "escalated"
		adj["escalation_id"] = record["id"]

		_log.info(
			"adjudication_escalated adj=%s escalation=%s type=%s",
			adjudication_id, record["id"], escalation_type,
		)
		self._emit(tenant, "adjudication_escalated", record["id"],
			{"adjudication_id": adjudication_id, "tribunal_ref": tribunal_ref})
		return deepcopy(record)

	async def record_tribunal_decision(
		self,
		escalation_id: str,
		decision: str,
		decision_date: str,
		judgement_ref: str,
		recorded_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Record the tribunal or court decision on an escalated adjudication."""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		guard_non_empty_string(judgement_ref, "judgement_ref")
		guard_non_empty_string(decision, "decision")

		record = self.escalations.get(escalation_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"escalation {escalation_id!r} not found")
		if record["status"] != "escalated":
			raise PermissionError("escalation_already_decided")

		record["tribunal_decision"] = decision
		record["decision_date"] = decision_date
		record["judgement_ref"] = judgement_ref
		record["recorded_by"] = recorded_by
		record["status"] = "decided"
		record["decided_at"] = self._now()

		# Propagate back to original adjudication
		adj = self.adjudications.get(record["adjudication_id"])
		if adj:
			adj["outcome_notes"] = (adj.get("outcome_notes") or "") + f" [Tribunal: {judgement_ref}]"
			adj["tribunal_decision"] = decision

		_log.info("tribunal_decision_recorded escalation=%s judgement=%s", escalation_id, judgement_ref)
		self._emit(tenant, "tribunal_decision_recorded", escalation_id,
			{"decision": decision, "judgement_ref": judgement_ref})
		return deepcopy(record)

	# ── Survey Plan Registry ──────────────────────────────────────────────────

	async def register_surveyor(
		self,
		surveyor_id: str,
		name: str,
		licence_number: str,
		licence_expiry_date: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Register a licensed surveyor per Kenya Survey Act Cap 299."""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		guard_non_empty_string(surveyor_id, "surveyor_id")
		guard_non_empty_string(licence_number, "licence_number")

		for s in self.surveyors.values():
			if s["tenant_id"] == tenant and s["licence_number"] == licence_number:
				raise ValueError(f"surveyor licence {licence_number!r} already registered")

		record: dict[str, Any] = {
			"id": self._record_id("surveyor"),
			"type": "registered_surveyor",
			"surveyor_id": surveyor_id,
			"name": name,
			"licence_number": licence_number,
			"licence_expiry_date": licence_expiry_date,
			"tenant_id": tenant,
			"status": "active",
			"created_at": self._now(),
		}
		self.surveyors[record["id"]] = record
		_log.info("surveyor_registered licence=%s name=%s", licence_number, name)
		self._emit(tenant, "surveyor_registered", record["id"],
			{"licence_number": licence_number, "name": name})
		return deepcopy(record)

	async def deposit_survey_plan(
		self,
		parcel_id: str,
		surveyor_id: str,
		plan_number: str,
		plan_date: str,
		plan_document_ref: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Deposit a survey plan with the Director of Surveys per Survey Act §33.

		Validates that the surveyor's licence has not expired before accepting
		the plan deposit.
		"""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		guard_non_empty_string(plan_number, "plan_number")
		guard_non_empty_string(plan_document_ref, "plan_document_ref")

		parcel = self.parcels.get(parcel_id)
		if not parcel or parcel["tenant_id"] != tenant:
			raise KeyError(f"parcel {parcel_id!r} not found")

		# Find surveyor by surveyor_id within tenant
		surveyor_rec = next(
			(s for s in self.surveyors.values()
			 if s["tenant_id"] == tenant and s["surveyor_id"] == surveyor_id),
			None,
		)
		if not surveyor_rec:
			raise KeyError(f"surveyor {surveyor_id!r} not registered")

		today = datetime.utcnow().strftime("%Y-%m-%d")
		if surveyor_rec.get("licence_expiry_date", "9999-99-99") < today:
			raise PermissionError("surveyor_licence_expired")

		record: dict[str, Any] = {
			"id": self._record_id("plan"),
			"type": "survey_plan",
			"parcel_id": parcel_id,
			"surveyor_id": surveyor_id,
			"surveyor_name": surveyor_rec["name"],
			"plan_number": plan_number,
			"plan_date": plan_date,
			"plan_document_ref": plan_document_ref,
			"tenant_id": tenant,
			"status": "deposited",
			"created_at": self._now(),
		}
		self.survey_plans[record["id"]] = record
		_log.info("survey_plan_deposited parcel=%s plan=%s surveyor=%s", parcel_id, plan_number, surveyor_id)
		self._emit(tenant, "survey_plan_deposited", record["id"],
			{"parcel_id": parcel_id, "plan_number": plan_number, "surveyor_id": surveyor_id})
		return deepcopy(record)

	async def list_survey_plans(
		self,
		parcel_id: str,
		tenant_id: str = "default",
	) -> list[dict[str, Any]]:
		"""List all survey plans deposited for a parcel."""
		tenant = self._tenant(tenant_id)
		guard_tenant_id(tenant)
		return [
			deepcopy(p) for p in self.survey_plans.values()
			if p["tenant_id"] == tenant and p["parcel_id"] == parcel_id
		]
