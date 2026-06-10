"""County / Devolved Services — async service implementation."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "gov_cty"

SUPPORTED_REVENUE_TYPES = {
	"land_rates", "business_permit", "parking", "market_fee",
	"advertisement", "building_permit", "health_certificate",
	"liquor_licence", "billboard", "estate_charges",
}
SUPPORTED_PERMIT_TYPES = {
	"business_permit", "building_permit", "health_certificate",
	"liquor_licence", "food_hygiene", "fire_clearance",
	"signage_permit", "outdoor_advertising",
}
SUPPORTED_WELFARE_PROGRAMMES = {
	"cash_transfer", "food_subsidy", "elderly_grant",
	"disability_grant", "orphan_support", "school_bursary",
}
SUPPORTED_HEALTH_FACILITY_TYPES = {
	"dispensary", "health_centre", "county_hospital",
	"sub_county_hospital", "nursing_home", "maternity",
}
SUPPORTED_TICKET_TYPES = {
	"road_repair", "drainage", "streetlight", "water_supply",
	"waste_collection", "park_maintenance", "sewer_blockage",
	"bridge_repair", "bus_shelter",
}
SUPPORTED_PRIORITIES = {"low", "normal", "high", "critical"}


class CountyServicesService:
	"""Async County / Devolved Services capability service."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.revenues: dict[str, dict[str, Any]] = {}
		self.permits: dict[str, dict[str, Any]] = {}
		self.welfare_applications: dict[str, dict[str, Any]] = {}
		self.health_facilities: dict[str, dict[str, Any]] = {}
		self.patients: dict[str, dict[str, Any]] = {}
		self.tickets: dict[str, dict[str, Any]] = {}
		self.budgets: dict[str, dict[str, Any]] = {}
		self.expenditures: dict[str, dict[str, Any]] = {}
		self.contractors: dict[str, dict[str, Any]] = {}
		self.market_stalls: dict[str, dict[str, Any]] = {}
		self.wards: dict[str, dict[str, Any]] = {}
		self.inspections: dict[str, dict[str, Any]] = {}
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
		"""Return county services health status."""
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"revenues": len(self.revenues),
			"permits": len(self.permits),
			"open_tickets": sum(1 for t in self.tickets.values() if t["status"] == "open"),
			"welfare_applications": len(self.welfare_applications),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		"""Return capability contract metadata."""
		return {
			"capability_id": CAPABILITY_ID,
			"name": "County / Devolved Services",
			"version": "1.0.0",
			"domain": "government",
			"description": "County revenue collection, permit issuance, social welfare, devolved health, public works ticketing",
			"supported_revenue_types": sorted(SUPPORTED_REVENUE_TYPES),
			"supported_permit_types": sorted(SUPPORTED_PERMIT_TYPES),
			"supported_welfare_programmes": sorted(SUPPORTED_WELFARE_PROGRAMMES),
			"supported_health_facility_types": sorted(SUPPORTED_HEALTH_FACILITY_TYPES),
			"supported_ticket_types": sorted(SUPPORTED_TICKET_TYPES),
		}

	async def get_audit_events(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		"""Return all audit events for a tenant."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Revenue collection ────────────────────────────────────────────────────

	async def collect_revenue(
		self,
		payer_id: str,
		payer_name: str,
		revenue_type: str,
		amount_kes: float,
		period: str,
		tenant_id: str = "default",
		payment_method: str = "mpesa",
		receipt_number: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Record a county revenue payment."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(payer_id, "payer_id")
		if revenue_type not in SUPPORTED_REVENUE_TYPES:
			raise ValueError(f"unsupported revenue type: {revenue_type!r}")
		if amount_kes <= 0:
			raise ValueError("amount_kes must be positive")
		record = {
			"id": self._record_id("rev"),
			"type": "county_revenue",
			"payer_id": payer_id,
			"payer_name": payer_name,
			"revenue_type": revenue_type,
			"amount_kes": amount_kes,
			"period": period,
			"payment_method": payment_method,
			"receipt_number": receipt_number or self._record_id("REC").upper(),
			"tenant_id": tenant,
			"metadata": deepcopy(metadata or {}),
			"status": "pending",
			"created_at": self._now(),
			"confirmed_at": None,
		}
		self.revenues[record["id"]] = record
		self._emit(tenant, "revenue_collected", record["id"],
			{"payer_id": payer_id, "revenue_type": revenue_type, "amount_kes": amount_kes})
		return deepcopy(record)

	async def confirm_revenue(self, revenue_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Confirm a pending revenue payment."""
		tenant = self._tenant(tenant_id)
		record = self.revenues.get(revenue_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"revenue record {revenue_id!r} not found")
		if record["status"] != "pending":
			raise PermissionError("revenue_not_pending")
		record["status"] = "confirmed"
		record["confirmed_at"] = self._now()
		self._emit(tenant, "revenue_confirmed", revenue_id, {"amount_kes": record["amount_kes"]})
		return deepcopy(record)

	async def get_revenue(self, revenue_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve a revenue record."""
		tenant = self._tenant(tenant_id)
		record = self.revenues.get(revenue_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"revenue record {revenue_id!r} not found")
		return deepcopy(record)

	async def list_revenues(
		self,
		tenant_id: str = "default",
		revenue_type: str | None = None,
		status: str | None = None,
		period: str | None = None,
	) -> list[dict[str, Any]]:
		"""List revenue records with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.revenues.values() if r["tenant_id"] == tenant]
		if revenue_type:
			items = [r for r in items if r["revenue_type"] == revenue_type]
		if status:
			items = [r for r in items if r["status"] == status]
		if period:
			items = [r for r in items if r["period"] == period]
		return items

	async def revenue_summary(self, tenant_id: str = "default", period: str | None = None) -> dict[str, Any]:
		"""Summarise revenue collection by type for a period."""
		tenant = self._tenant(tenant_id)
		items = [r for r in self.revenues.values() if r["tenant_id"] == tenant and r["status"] == "confirmed"]
		if period:
			items = [r for r in items if r["period"] == period]
		by_type: dict[str, float] = {}
		for r in items:
			rt = r["revenue_type"]
			by_type[rt] = by_type.get(rt, 0.0) + r["amount_kes"]
		return {
			"tenant_id": tenant,
			"period": period or "all",
			"total_kes": sum(by_type.values()),
			"by_revenue_type": by_type,
			"transaction_count": len(items),
			"generated_at": self._now(),
		}

	# ── County permit issuance ────────────────────────────────────────────────

	async def apply_permit(
		self,
		applicant_id: str,
		applicant_name: str,
		business_name: str,
		permit_type: str,
		location: str,
		sub_county: str,
		fee_paid_kes: float,
		tenant_id: str = "default",
		supporting_documents: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Submit a county permit application."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(applicant_id, "applicant_id")
		if permit_type not in SUPPORTED_PERMIT_TYPES:
			raise ValueError(f"unsupported permit type: {permit_type!r}")
		if fee_paid_kes <= 0:
			raise ValueError("fee_paid_kes must be positive")
		record = {
			"id": self._record_id("permit"),
			"type": "county_permit",
			"applicant_id": applicant_id,
			"applicant_name": applicant_name,
			"business_name": business_name,
			"permit_type": permit_type,
			"location": location,
			"sub_county": sub_county,
			"fee_paid_kes": fee_paid_kes,
			"permit_number": None,
			"issue_date": None,
			"expiry_date": None,
			"issued_by": None,
			"rejection_reason": None,
			"supporting_documents": deepcopy(supporting_documents or []),
			"tenant_id": tenant,
			"metadata": deepcopy(metadata or {}),
			"status": "submitted",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.permits[record["id"]] = record
		self._emit(tenant, "permit_applied", record["id"],
			{"applicant_id": applicant_id, "permit_type": permit_type, "business_name": business_name})
		return deepcopy(record)

	async def issue_permit(
		self,
		permit_id: str,
		permit_number: str,
		issue_date: str,
		expiry_date: str,
		issued_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Issue an approved county permit."""
		tenant = self._tenant(tenant_id)
		record = self.permits.get(permit_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"permit {permit_id!r} not found")
		if record["status"] not in {"submitted", "under_review"}:
			raise PermissionError("permit_cannot_be_issued_in_current_status")
		record["permit_number"] = permit_number
		record["issue_date"] = issue_date
		record["expiry_date"] = expiry_date
		record["issued_by"] = issued_by
		record["status"] = "issued"
		record["updated_at"] = self._now()
		self._emit(tenant, "permit_issued", permit_id,
			{"permit_number": permit_number, "issued_by": issued_by})
		return deepcopy(record)

	async def reject_permit(
		self,
		permit_id: str,
		rejection_reason: str,
		rejected_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Reject a permit application with reason."""
		tenant = self._tenant(tenant_id)
		record = self.permits.get(permit_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"permit {permit_id!r} not found")
		record["rejection_reason"] = rejection_reason
		record["rejected_by"] = rejected_by
		record["status"] = "rejected"
		record["updated_at"] = self._now()
		self._emit(tenant, "permit_rejected", permit_id, {"reason": rejection_reason})
		return deepcopy(record)

	async def get_permit(self, permit_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve a permit record."""
		tenant = self._tenant(tenant_id)
		record = self.permits.get(permit_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"permit {permit_id!r} not found")
		return deepcopy(record)

	async def update_permit(
		self,
		permit_id: str,
		tenant_id: str = "default",
		status: str | None = None,
		issued_by: str | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Update a permit record."""
		tenant = self._tenant(tenant_id)
		record = self.permits.get(permit_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"permit {permit_id!r} not found")
		if status:
			record["status"] = status
		if issued_by:
			record["issued_by"] = issued_by
		if notes:
			record.setdefault("notes", []).append({"note": notes, "at": self._now()})
		record["updated_at"] = self._now()
		self._emit(tenant, "permit_updated", permit_id, {"status": status})
		return deepcopy(record)

	async def list_permits(
		self,
		tenant_id: str = "default",
		permit_type: str | None = None,
		status: str | None = None,
		sub_county: str | None = None,
	) -> list[dict[str, Any]]:
		"""List permits with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.permits.values() if r["tenant_id"] == tenant]
		if permit_type:
			items = [r for r in items if r["permit_type"] == permit_type]
		if status:
			items = [r for r in items if r["status"] == status]
		if sub_county:
			items = [r for r in items if r["sub_county"] == sub_county]
		return items

	async def delete_permit(self, permit_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Delete a permit application (draft/rejected only)."""
		tenant = self._tenant(tenant_id)
		record = self.permits.get(permit_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"permit {permit_id!r} not found")
		if record["status"] == "issued":
			raise PermissionError("cannot_delete_issued_permit")
		del self.permits[permit_id]
		self._emit(tenant, "permit_deleted", permit_id, {})
		return {"id": permit_id, "deleted": True}

	# ── Social welfare ────────────────────────────────────────────────────────

	async def apply_welfare(
		self,
		applicant_id: str,
		applicant_name: str,
		id_number: str,
		programme_type: str,
		sub_county: str,
		ward: str,
		household_size: int,
		tenant_id: str = "default",
		monthly_income_kes: float = 0.0,
		needs_assessment: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Submit a social welfare programme application."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(id_number, "id_number")
		if programme_type not in SUPPORTED_WELFARE_PROGRAMMES:
			raise ValueError(f"unsupported welfare programme: {programme_type!r}")
		if household_size < 1:
			raise ValueError("household_size must be >= 1")
		record = {
			"id": self._record_id("welf"),
			"type": "welfare_application",
			"applicant_id": applicant_id,
			"applicant_name": applicant_name,
			"id_number": id_number,
			"programme_type": programme_type,
			"sub_county": sub_county,
			"ward": ward,
			"household_size": household_size,
			"monthly_income_kes": monthly_income_kes,
			"needs_assessment": needs_assessment,
			"approved_amount_kes": None,
			"payment_frequency": None,
			"case_worker_id": None,
			"tenant_id": tenant,
			"metadata": deepcopy(metadata or {}),
			"status": "submitted",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.welfare_applications[record["id"]] = record
		self._emit(tenant, "welfare_application_submitted", record["id"],
			{"applicant_id": applicant_id, "programme_type": programme_type})
		return deepcopy(record)

	async def approve_welfare(
		self,
		application_id: str,
		approved_amount_kes: float,
		payment_frequency: str,
		case_worker_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Approve a welfare application and set payment terms."""
		tenant = self._tenant(tenant_id)
		record = self.welfare_applications.get(application_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"welfare application {application_id!r} not found")
		if approved_amount_kes <= 0:
			raise ValueError("approved_amount_kes must be positive")
		record["approved_amount_kes"] = approved_amount_kes
		record["payment_frequency"] = payment_frequency
		record["case_worker_id"] = case_worker_id
		record["status"] = "approved"
		record["updated_at"] = self._now()
		self._emit(tenant, "welfare_approved", application_id,
			{"approved_amount_kes": approved_amount_kes, "case_worker_id": case_worker_id})
		return deepcopy(record)

	async def get_welfare_application(self, application_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve a welfare application."""
		tenant = self._tenant(tenant_id)
		record = self.welfare_applications.get(application_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"welfare application {application_id!r} not found")
		return deepcopy(record)

	async def update_welfare_application(
		self,
		application_id: str,
		tenant_id: str = "default",
		status: str | None = None,
		case_worker_id: str | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Update welfare application status or assignment."""
		tenant = self._tenant(tenant_id)
		record = self.welfare_applications.get(application_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"welfare application {application_id!r} not found")
		if status:
			record["status"] = status
		if case_worker_id:
			record["case_worker_id"] = case_worker_id
		if notes:
			record.setdefault("notes", []).append({"note": notes, "at": self._now()})
		record["updated_at"] = self._now()
		self._emit(tenant, "welfare_application_updated", application_id, {"status": status})
		return deepcopy(record)

	async def list_welfare_applications(
		self,
		tenant_id: str = "default",
		programme_type: str | None = None,
		status: str | None = None,
		sub_county: str | None = None,
	) -> list[dict[str, Any]]:
		"""List welfare applications with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.welfare_applications.values() if r["tenant_id"] == tenant]
		if programme_type:
			items = [r for r in items if r["programme_type"] == programme_type]
		if status:
			items = [r for r in items if r["status"] == status]
		if sub_county:
			items = [r for r in items if r["sub_county"] == sub_county]
		return items

	# ── Devolved health services ──────────────────────────────────────────────

	async def register_health_facility(
		self,
		facility_code: str,
		facility_name: str,
		facility_type: str,
		sub_county: str,
		ward: str,
		tenant_id: str = "default",
		beds: int = 0,
		services: list[str] | None = None,
	) -> dict[str, Any]:
		"""Register a devolved health facility."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(facility_code, "facility_code")
		if facility_type not in SUPPORTED_HEALTH_FACILITY_TYPES:
			raise ValueError(f"unsupported facility type: {facility_type!r}")
		for f in self.health_facilities.values():
			if f["tenant_id"] == tenant and f["facility_code"] == facility_code:
				raise ValueError(f"facility {facility_code!r} already registered")
		record = {
			"id": self._record_id("fac"),
			"type": "health_facility",
			"facility_code": facility_code,
			"facility_name": facility_name,
			"facility_type": facility_type,
			"sub_county": sub_county,
			"ward": ward,
			"beds": beds,
			"services": deepcopy(services or []),
			"tenant_id": tenant,
			"status": "active",
			"created_at": self._now(),
		}
		self.health_facilities[record["id"]] = record
		self._emit(tenant, "health_facility_registered", record["id"],
			{"facility_code": facility_code, "facility_type": facility_type})
		return deepcopy(record)

	async def get_health_facility(self, facility_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve a health facility record."""
		tenant = self._tenant(tenant_id)
		record = self.health_facilities.get(facility_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"facility {facility_id!r} not found")
		return deepcopy(record)

	async def list_health_facilities(
		self,
		tenant_id: str = "default",
		facility_type: str | None = None,
		sub_county: str | None = None,
	) -> list[dict[str, Any]]:
		"""List health facilities."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.health_facilities.values() if r["tenant_id"] == tenant]
		if facility_type:
			items = [r for r in items if r["facility_type"] == facility_type]
		if sub_county:
			items = [r for r in items if r["sub_county"] == sub_county]
		return items

	async def register_patient(
		self,
		facility_id: str,
		patient_name: str,
		id_number: str,
		date_of_birth: str,
		gender: str,
		sub_county: str,
		tenant_id: str = "default",
		phone: str | None = None,
	) -> dict[str, Any]:
		"""Register a patient at a health facility."""
		tenant = self._tenant(tenant_id)
		facility = self.health_facilities.get(facility_id)
		if not facility or facility["tenant_id"] != tenant:
			raise KeyError(f"facility {facility_id!r} not found")
		patient_number = f"PAT-{uuid4().hex[:8].upper()}"
		record = {
			"id": self._record_id("pat"),
			"type": "patient_registration",
			"facility_id": facility_id,
			"patient_name": patient_name,
			"id_number": id_number,
			"date_of_birth": date_of_birth,
			"gender": gender,
			"sub_county": sub_county,
			"phone": phone,
			"patient_number": patient_number,
			"tenant_id": tenant,
			"status": "active",
			"created_at": self._now(),
		}
		self.patients[record["id"]] = record
		self._emit(tenant, "patient_registered", record["id"],
			{"facility_id": facility_id, "patient_number": patient_number})
		return deepcopy(record)

	async def get_patient(self, patient_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve a patient record."""
		tenant = self._tenant(tenant_id)
		record = self.patients.get(patient_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"patient {patient_id!r} not found")
		return deepcopy(record)

	async def list_patients(
		self,
		tenant_id: str = "default",
		facility_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List patients with optional facility filter."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.patients.values() if r["tenant_id"] == tenant]
		if facility_id:
			items = [r for r in items if r["facility_id"] == facility_id]
		return items

	# ── Public works ticketing ────────────────────────────────────────────────

	async def create_ticket(
		self,
		reporter_id: str,
		reporter_name: str,
		ticket_type: str,
		description: str,
		location: str,
		sub_county: str,
		ward: str,
		tenant_id: str = "default",
		priority: str = "normal",
		reporter_phone: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Create a public works service ticket."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(description, "description")
		if ticket_type not in SUPPORTED_TICKET_TYPES:
			raise ValueError(f"unsupported ticket type: {ticket_type!r}")
		if priority not in SUPPORTED_PRIORITIES:
			raise ValueError(f"unsupported priority: {priority!r}")
		record = {
			"id": self._record_id("ticket"),
			"type": "public_works_ticket",
			"reporter_id": reporter_id,
			"reporter_name": reporter_name,
			"reporter_phone": reporter_phone,
			"ticket_type": ticket_type,
			"description": description,
			"location": location,
			"sub_county": sub_county,
			"ward": ward,
			"priority": priority,
			"assigned_to": None,
			"resolution_notes": None,
			"estimated_completion": None,
			"tenant_id": tenant,
			"metadata": deepcopy(metadata or {}),
			"status": "open",
			"created_at": self._now(),
			"updated_at": None,
			"resolved_at": None,
		}
		self.tickets[record["id"]] = record
		self._emit(tenant, "public_works_ticket_created", record["id"],
			{"ticket_type": ticket_type, "priority": priority, "sub_county": sub_county})
		return deepcopy(record)

	async def get_ticket(self, ticket_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve a public works ticket."""
		tenant = self._tenant(tenant_id)
		record = self.tickets.get(ticket_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"ticket {ticket_id!r} not found")
		return deepcopy(record)

	async def update_ticket(
		self,
		ticket_id: str,
		tenant_id: str = "default",
		status: str | None = None,
		assigned_to: str | None = None,
		resolution_notes: str | None = None,
		estimated_completion: str | None = None,
	) -> dict[str, Any]:
		"""Update a public works ticket."""
		tenant = self._tenant(tenant_id)
		record = self.tickets.get(ticket_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"ticket {ticket_id!r} not found")
		if status:
			record["status"] = status
			if status == "resolved":
				record["resolved_at"] = self._now()
		if assigned_to:
			record["assigned_to"] = assigned_to
		if resolution_notes:
			record["resolution_notes"] = resolution_notes
		if estimated_completion:
			record["estimated_completion"] = estimated_completion
		record["updated_at"] = self._now()
		self._emit(tenant, "ticket_updated", ticket_id,
			{"status": status, "assigned_to": assigned_to})
		return deepcopy(record)

	async def list_tickets(
		self,
		tenant_id: str = "default",
		ticket_type: str | None = None,
		status: str | None = None,
		priority: str | None = None,
		sub_county: str | None = None,
	) -> list[dict[str, Any]]:
		"""List public works tickets with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.tickets.values() if r["tenant_id"] == tenant]
		if ticket_type:
			items = [r for r in items if r["ticket_type"] == ticket_type]
		if status:
			items = [r for r in items if r["status"] == status]
		if priority:
			items = [r for r in items if r["priority"] == priority]
		if sub_county:
			items = [r for r in items if r["sub_county"] == sub_county]
		return items

	async def delete_ticket(self, ticket_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Delete a public works ticket."""
		tenant = self._tenant(tenant_id)
		record = self.tickets.get(ticket_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"ticket {ticket_id!r} not found")
		del self.tickets[ticket_id]
		self._emit(tenant, "ticket_deleted", ticket_id, {})
		return {"id": ticket_id, "deleted": True}

	# ── Budget management ─────────────────────────────────────────────────────

	async def create_budget(
		self,
		budget_year: int,
		department: str,
		total_budget_kes: float,
		approved_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Create a departmental budget allocation."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(department, "department")
		if total_budget_kes <= 0:
			raise ValueError("total_budget_kes must be positive")
		record = {
			"id": self._record_id("budget"),
			"type": "county_budget",
			"budget_year": budget_year,
			"department": department,
			"total_budget_kes": total_budget_kes,
			"allocated_kes": 0.0,
			"spent_kes": 0.0,
			"approved_by": approved_by,
			"tenant_id": tenant,
			"status": "approved",
			"created_at": self._now(),
		}
		self.budgets[record["id"]] = record
		self._emit(tenant, "budget_created", record["id"],
			{"department": department, "total_budget_kes": total_budget_kes})
		return deepcopy(record)

	async def list_budgets(self, tenant_id: str = "default", budget_year: int | None = None) -> list[dict[str, Any]]:
		"""List budget allocations."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.budgets.values() if r["tenant_id"] == tenant]
		if budget_year:
			items = [r for r in items if r["budget_year"] == budget_year]
		return items

	# ── Market stalls ─────────────────────────────────────────────────────────

	async def allocate_market_stall(
		self,
		market_name: str,
		stall_number: str,
		trader_id: str,
		trader_name: str,
		stall_type: str,
		monthly_fee_kes: float,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Allocate a market stall to a trader."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(stall_number, "stall_number")
		if monthly_fee_kes <= 0:
			raise ValueError("monthly_fee_kes must be positive")
		# Check stall not already occupied
		for s in self.market_stalls.values():
			if (s["tenant_id"] == tenant and s["market_name"] == market_name
					and s["stall_number"] == stall_number and s["status"] == "occupied"):
				raise PermissionError("stall_already_occupied")
		record = {
			"id": self._record_id("stall"),
			"type": "market_stall",
			"market_name": market_name,
			"stall_number": stall_number,
			"trader_id": trader_id,
			"trader_name": trader_name,
			"stall_type": stall_type,
			"monthly_fee_kes": monthly_fee_kes,
			"tenant_id": tenant,
			"status": "occupied",
			"created_at": self._now(),
		}
		self.market_stalls[record["id"]] = record
		self._emit(tenant, "market_stall_allocated", record["id"],
			{"market_name": market_name, "stall_number": stall_number, "trader_id": trader_id})
		return deepcopy(record)

	async def list_market_stalls(self, tenant_id: str = "default", market_name: str | None = None) -> list[dict[str, Any]]:
		"""List market stalls."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.market_stalls.values() if r["tenant_id"] == tenant]
		if market_name:
			items = [r for r in items if r["market_name"] == market_name]
		return items

	# ── Contractor management ─────────────────────────────────────────────────

	async def register_contractor(
		self,
		contractor_id: str,
		contractor_name: str,
		registration_number: str,
		category: str,
		contact_phone: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Register a public works contractor."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(registration_number, "registration_number")
		record = {
			"id": self._record_id("cont"),
			"type": "contractor",
			"contractor_id": contractor_id,
			"contractor_name": contractor_name,
			"registration_number": registration_number,
			"category": category,
			"contact_phone": contact_phone,
			"tenant_id": tenant,
			"status": "active",
			"created_at": self._now(),
		}
		self.contractors[record["id"]] = record
		self._emit(tenant, "contractor_registered", record["id"],
			{"contractor_id": contractor_id, "category": category})
		return deepcopy(record)

	async def list_contractors(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		"""List registered contractors."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.contractors.values() if r["tenant_id"] == tenant]

	# ── Dashboard ─────────────────────────────────────────────────────────────

	async def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return county services dashboard metrics."""
		tenant = self._tenant(tenant_id)
		revenues = [r for r in self.revenues.values() if r["tenant_id"] == tenant]
		tickets = [t for t in self.tickets.values() if t["tenant_id"] == tenant]
		return {
			"tenant_id": tenant,
			"revenue_total_kes": sum(r["amount_kes"] for r in revenues if r["status"] == "confirmed"),
			"revenue_pending_kes": sum(r["amount_kes"] for r in revenues if r["status"] == "pending"),
			"total_permits": sum(1 for r in self.permits.values() if r["tenant_id"] == tenant),
			"issued_permits": sum(1 for r in self.permits.values() if r["tenant_id"] == tenant and r["status"] == "issued"),
			"welfare_applications": sum(1 for r in self.welfare_applications.values() if r["tenant_id"] == tenant),
			"welfare_approved": sum(1 for r in self.welfare_applications.values() if r["tenant_id"] == tenant and r["status"] == "approved"),
			"health_facilities": sum(1 for r in self.health_facilities.values() if r["tenant_id"] == tenant),
			"registered_patients": sum(1 for r in self.patients.values() if r["tenant_id"] == tenant),
			"open_tickets": sum(1 for t in tickets if t["status"] == "open"),
			"resolved_tickets": sum(1 for t in tickets if t["status"] == "resolved"),
			"critical_tickets": sum(1 for t in tickets if t["priority"] == "critical" and t["status"] == "open"),
			"market_stalls": sum(1 for r in self.market_stalls.values() if r["tenant_id"] == tenant),
			"contractors": sum(1 for r in self.contractors.values() if r["tenant_id"] == tenant),
			"generated_at": self._now(),
		}
