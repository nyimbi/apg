"""Land Registry — async service implementation."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
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
