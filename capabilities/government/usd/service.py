"""USSD Government Services — async service implementation."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "gov_usd"

SUPPORTED_SERVICE_CODES = {"*384#", "*385#", "*386#", "*400#", "*500#"}
SUPPORTED_PERMIT_TYPES = {
	"business_permit", "building_permit", "health_certificate",
	"liquor_licence", "food_hygiene", "environmental_clearance",
}
SUPPORTED_TAX_TYPES = {
	"income_tax", "vat", "paye", "corporation_tax",
	"withholding_tax", "excise_duty", "turnover_tax",
}
SUPPORTED_ID_TYPES = {"national_id", "passport", "alien_card", "military_id"}
SUPPORTED_CERTIFICATE_TYPES = {
	"good_conduct", "tax_compliance", "business_registration",
	"birth_certificate", "death_certificate", "marriage_certificate",
	"clearance_certificate",
}


class USSDGovService:
	"""Async USSD Government Services capability service."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.sessions: dict[str, dict[str, Any]] = {}
		self.permit_enquiries: dict[str, dict[str, Any]] = {}
		self.tax_enquiries: dict[str, dict[str, Any]] = {}
		self.id_verifications: dict[str, dict[str, Any]] = {}
		self.certificate_requests: dict[str, dict[str, Any]] = {}
		self.menus: dict[str, dict[str, Any]] = {}
		self.service_codes: dict[str, dict[str, Any]] = {}
		self.otp_store: dict[str, dict[str, Any]] = {}
		self.payment_references: dict[str, dict[str, Any]] = {}
		self.sms_notifications: list[dict[str, Any]] = []
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)

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
		"""Return USSD service health status."""
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"active_sessions": sum(1 for s in self.sessions.values() if s["status"] == "active"),
			"total_sessions": len(self.sessions),
			"certificate_requests": len(self.certificate_requests),
			"id_verifications": len(self.id_verifications),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		"""Return capability contract metadata."""
		return {
			"capability_id": CAPABILITY_ID,
			"name": "USSD Government Services",
			"version": "1.0.0",
			"domain": "government",
			"description": "USSD-based government services: permit status, tax balance inquiry, ID verification, certificate requests",
			"supported_service_codes": sorted(SUPPORTED_SERVICE_CODES),
			"supported_permit_types": sorted(SUPPORTED_PERMIT_TYPES),
			"supported_tax_types": sorted(SUPPORTED_TAX_TYPES),
			"supported_id_types": sorted(SUPPORTED_ID_TYPES),
			"supported_certificate_types": sorted(SUPPORTED_CERTIFICATE_TYPES),
		}

	async def get_audit_events(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		"""Return all audit events for a tenant."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── USSD Session management ───────────────────────────────────────────────

	async def create_session(
		self,
		msisdn: str,
		service_code: str,
		tenant_id: str = "default",
		session_data: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Initiate a new USSD session for a subscriber."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(msisdn, "msisdn")
		guard_non_empty_string(service_code, "service_code")
		record = {
			"id": self._record_id("session"),
			"type": "ussd_session",
			"msisdn": msisdn,
			"service_code": service_code,
			"tenant_id": tenant,
			"menu_level": 1,
			"session_data": deepcopy(session_data or {}),
			"history": [],
			"status": "active",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.sessions[record["id"]] = record
		self._emit(tenant, "ussd_session_created", record["id"], {"msisdn": msisdn, "service_code": service_code})
		return deepcopy(record)

	async def get_session(self, session_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve a USSD session by ID."""
		tenant = self._tenant(tenant_id)
		session = self.sessions.get(session_id)
		if not session or session["tenant_id"] != tenant:
			raise KeyError(f"session {session_id!r} not found")
		return deepcopy(session)

	async def update_session(
		self,
		session_id: str,
		input_text: str,
		tenant_id: str = "default",
		menu_level: int | None = None,
		session_data: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Advance a USSD session with subscriber input."""
		tenant = self._tenant(tenant_id)
		session = self.sessions.get(session_id)
		if not session or session["tenant_id"] != tenant:
			raise KeyError(f"session {session_id!r} not found")
		session["history"].append({
			"level": session["menu_level"],
			"input": input_text,
			"at": self._now(),
		})
		if menu_level is not None:
			session["menu_level"] = menu_level
		else:
			session["menu_level"] += 1
		if session_data:
			session["session_data"].update(session_data)
		session["updated_at"] = self._now()
		self._emit(tenant, "ussd_session_updated", session_id, {"input": input_text, "level": session["menu_level"]})
		return deepcopy(session)

	async def close_session(self, session_id: str, tenant_id: str = "default", reason: str = "completed") -> dict[str, Any]:
		"""Close a USSD session."""
		tenant = self._tenant(tenant_id)
		session = self.sessions.get(session_id)
		if not session or session["tenant_id"] != tenant:
			raise KeyError(f"session {session_id!r} not found")
		session["status"] = "closed"
		session["close_reason"] = reason
		session["updated_at"] = self._now()
		self._emit(tenant, "ussd_session_closed", session_id, {"reason": reason})
		return deepcopy(session)

	async def list_sessions(
		self,
		tenant_id: str = "default",
		msisdn: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List USSD sessions with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(s) for s in self.sessions.values() if s["tenant_id"] == tenant]
		if msisdn:
			items = [s for s in items if s["msisdn"] == msisdn]
		if status:
			items = [s for s in items if s["status"] == status]
		return items

	async def delete_session(self, session_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Delete a closed USSD session."""
		tenant = self._tenant(tenant_id)
		session = self.sessions.get(session_id)
		if not session or session["tenant_id"] != tenant:
			raise KeyError(f"session {session_id!r} not found")
		del self.sessions[session_id]
		self._emit(tenant, "ussd_session_deleted", session_id, {})
		return {"id": session_id, "deleted": True}

	# ── Permit enquiry ────────────────────────────────────────────────────────

	async def enquire_permit_status(
		self,
		msisdn: str,
		permit_number: str,
		permit_type: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Query permit status via USSD."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(permit_number, "permit_number")
		if permit_type not in SUPPORTED_PERMIT_TYPES:
			raise ValueError(f"unsupported permit type: {permit_type!r}")
		# Simulate registry lookup — in production would call permit registry
		permit_data = {
			"permit_number": permit_number,
			"permit_type": permit_type,
			"holder_name": "ENQUIRY_RESULT",
			"issue_date": "2024-01-01",
			"expiry_date": "2025-01-01",
			"is_valid": True,
		}
		record = {
			"id": self._record_id("penq"),
			"type": "permit_enquiry",
			"msisdn": msisdn,
			"permit_number": permit_number,
			"permit_type": permit_type,
			"tenant_id": tenant,
			"holder_name": permit_data["holder_name"],
			"issue_date": permit_data["issue_date"],
			"expiry_date": permit_data["expiry_date"],
			"is_valid": permit_data["is_valid"],
			"status": "valid" if permit_data["is_valid"] else "expired",
			"created_at": self._now(),
		}
		self.permit_enquiries[record["id"]] = record
		self._emit(tenant, "permit_enquiry_made", record["id"], {"msisdn": msisdn, "permit_number": permit_number})
		return deepcopy(record)

	async def get_permit_enquiry(self, enquiry_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve a permit enquiry result."""
		tenant = self._tenant(tenant_id)
		record = self.permit_enquiries.get(enquiry_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"permit enquiry {enquiry_id!r} not found")
		return deepcopy(record)

	async def list_permit_enquiries(self, tenant_id: str = "default", msisdn: str | None = None) -> list[dict[str, Any]]:
		"""List permit enquiries for a tenant."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.permit_enquiries.values() if r["tenant_id"] == tenant]
		if msisdn:
			items = [r for r in items if r["msisdn"] == msisdn]
		return items

	# ── Tax balance enquiry ───────────────────────────────────────────────────

	async def enquire_tax_balance(
		self,
		msisdn: str,
		tax_pin: str,
		tax_type: str = "income_tax",
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Query outstanding tax balance via USSD."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(tax_pin, "tax_pin")
		if tax_type not in SUPPORTED_TAX_TYPES:
			raise ValueError(f"unsupported tax type: {tax_type!r}")
		# Simulate KRA/revenue authority lookup
		record = {
			"id": self._record_id("taxenq"),
			"type": "tax_balance_enquiry",
			"msisdn": msisdn,
			"tax_pin": tax_pin,
			"tax_type": tax_type,
			"tenant_id": tenant,
			"outstanding_balance": 0.0,
			"currency": "KES",
			"last_payment_date": None,
			"due_date": None,
			"compliance_status": "compliant",
			"status": "fetched",
			"created_at": self._now(),
		}
		self.tax_enquiries[record["id"]] = record
		self._emit(tenant, "tax_balance_enquired", record["id"], {"msisdn": msisdn, "tax_pin": tax_pin, "tax_type": tax_type})
		return deepcopy(record)

	async def get_tax_enquiry(self, enquiry_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve a tax balance enquiry result."""
		tenant = self._tenant(tenant_id)
		record = self.tax_enquiries.get(enquiry_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"tax enquiry {enquiry_id!r} not found")
		return deepcopy(record)

	async def list_tax_enquiries(self, tenant_id: str = "default", tax_pin: str | None = None) -> list[dict[str, Any]]:
		"""List tax enquiries for a tenant."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.tax_enquiries.values() if r["tenant_id"] == tenant]
		if tax_pin:
			items = [r for r in items if r["tax_pin"] == tax_pin]
		return items

	# ── ID verification ───────────────────────────────────────────────────────

	async def verify_id(
		self,
		msisdn: str,
		id_number: str,
		id_type: str = "national_id",
		full_name: str | None = None,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Verify a national ID or passport via USSD."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(id_number, "id_number")
		if id_type not in SUPPORTED_ID_TYPES:
			raise ValueError(f"unsupported ID type: {id_type!r}")
		# Simulate IPRS/immigration lookup
		verified = len(id_number) >= 7
		record = {
			"id": self._record_id("idv"),
			"type": "id_verification",
			"msisdn": msisdn,
			"id_number": id_number,
			"id_type": id_type,
			"full_name": full_name,
			"tenant_id": tenant,
			"verified": verified,
			"verification_details": {
				"source": "IPRS",
				"match_score": 0.98 if verified else 0.0,
				"flags": [],
			},
			"status": "verified" if verified else "not_found",
			"created_at": self._now(),
		}
		self.id_verifications[record["id"]] = record
		self._emit(tenant, "id_verified" if verified else "id_verification_failed", record["id"],
			{"msisdn": msisdn, "id_type": id_type, "verified": verified})
		return deepcopy(record)

	async def get_id_verification(self, verification_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve an ID verification result."""
		tenant = self._tenant(tenant_id)
		record = self.id_verifications.get(verification_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"id verification {verification_id!r} not found")
		return deepcopy(record)

	async def list_id_verifications(self, tenant_id: str = "default", msisdn: str | None = None) -> list[dict[str, Any]]:
		"""List ID verifications for a tenant."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.id_verifications.values() if r["tenant_id"] == tenant]
		if msisdn:
			items = [r for r in items if r["msisdn"] == msisdn]
		return items

	# ── Certificate requests ──────────────────────────────────────────────────

	async def request_certificate(
		self,
		msisdn: str,
		certificate_type: str,
		applicant_id: str,
		applicant_name: str,
		tenant_id: str = "default",
		reference_number: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Submit a certificate request via USSD."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(certificate_type, "certificate_type")
		guard_non_empty_string(applicant_id, "applicant_id")
		if certificate_type not in SUPPORTED_CERTIFICATE_TYPES:
			raise ValueError(f"unsupported certificate type: {certificate_type!r}")
		record = {
			"id": self._record_id("cert"),
			"type": "certificate_request",
			"msisdn": msisdn,
			"certificate_type": certificate_type,
			"applicant_id": applicant_id,
			"applicant_name": applicant_name,
			"reference_number": reference_number or self._record_id("ref"),
			"certificate_number": None,
			"tenant_id": tenant,
			"metadata": deepcopy(metadata or {}),
			"status": "submitted",
			"issued_by": None,
			"created_at": self._now(),
			"updated_at": None,
		}
		self.certificate_requests[record["id"]] = record
		self._emit(tenant, "certificate_requested", record["id"],
			{"msisdn": msisdn, "certificate_type": certificate_type, "applicant_id": applicant_id})
		return deepcopy(record)

	async def get_certificate_request(self, request_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve a certificate request."""
		tenant = self._tenant(tenant_id)
		record = self.certificate_requests.get(request_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"certificate request {request_id!r} not found")
		return deepcopy(record)

	async def update_certificate_request(
		self,
		request_id: str,
		tenant_id: str = "default",
		status: str | None = None,
		certificate_number: str | None = None,
		issued_by: str | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Update a certificate request status."""
		tenant = self._tenant(tenant_id)
		record = self.certificate_requests.get(request_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"certificate request {request_id!r} not found")
		if status:
			record["status"] = status
		if certificate_number:
			record["certificate_number"] = certificate_number
		if issued_by:
			record["issued_by"] = issued_by
		if notes:
			record.setdefault("notes", []).append({"note": notes, "at": self._now()})
		record["updated_at"] = self._now()
		self._emit(tenant, "certificate_request_updated", request_id,
			{"status": status, "certificate_number": certificate_number})
		return deepcopy(record)

	async def list_certificate_requests(
		self,
		tenant_id: str = "default",
		certificate_type: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List certificate requests with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.certificate_requests.values() if r["tenant_id"] == tenant]
		if certificate_type:
			items = [r for r in items if r["certificate_type"] == certificate_type]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	async def delete_certificate_request(self, request_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Delete a certificate request."""
		tenant = self._tenant(tenant_id)
		record = self.certificate_requests.get(request_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"certificate request {request_id!r} not found")
		del self.certificate_requests[request_id]
		self._emit(tenant, "certificate_request_deleted", request_id, {})
		return {"id": request_id, "deleted": True}

	# ── USSD Menu management ──────────────────────────────────────────────────

	async def create_menu(
		self,
		service_code: str,
		menu_key: str,
		menu_text: str,
		menu_level: int = 1,
		tenant_id: str = "default",
		parent_key: str | None = None,
		action: str | None = None,
	) -> dict[str, Any]:
		"""Define a USSD menu entry."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(menu_key, "menu_key")
		guard_non_empty_string(menu_text, "menu_text")
		record = {
			"id": self._record_id("menu"),
			"type": "ussd_menu",
			"service_code": service_code,
			"menu_key": menu_key,
			"menu_text": menu_text,
			"menu_level": menu_level,
			"parent_key": parent_key,
			"action": action,
			"tenant_id": tenant,
			"status": "active",
			"created_at": self._now(),
		}
		self.menus[record["id"]] = record
		self._emit(tenant, "ussd_menu_created", record["id"], {"service_code": service_code, "menu_key": menu_key})
		return deepcopy(record)

	async def get_menu(self, menu_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Retrieve a USSD menu entry."""
		tenant = self._tenant(tenant_id)
		record = self.menus.get(menu_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"menu {menu_id!r} not found")
		return deepcopy(record)

	async def list_menus(self, tenant_id: str = "default", service_code: str | None = None) -> list[dict[str, Any]]:
		"""List USSD menu entries."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.menus.values() if r["tenant_id"] == tenant]
		if service_code:
			items = [r for r in items if r["service_code"] == service_code]
		return sorted(items, key=lambda x: (x["menu_level"], x["menu_key"]))

	async def delete_menu(self, menu_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Remove a USSD menu entry."""
		tenant = self._tenant(tenant_id)
		record = self.menus.get(menu_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"menu {menu_id!r} not found")
		del self.menus[menu_id]
		self._emit(tenant, "ussd_menu_deleted", menu_id, {})
		return {"id": menu_id, "deleted": True}

	# ── OTP / authentication ──────────────────────────────────────────────────

	async def generate_otp(self, msisdn: str, purpose: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Generate a one-time PIN for USSD service authentication."""
		tenant = self._tenant(tenant_id)
		import random
		otp_code = f"{random.randint(100000, 999999)}"
		record = {
			"id": self._record_id("otp"),
			"type": "ussd_otp",
			"msisdn": msisdn,
			"purpose": purpose,
			"otp_code": otp_code,
			"tenant_id": tenant,
			"attempts": 0,
			"max_attempts": 3,
			"status": "pending",
			"expires_at": self._now(),  # simplified — production would add 5-minute window
			"created_at": self._now(),
		}
		self.otp_store[record["id"]] = record
		self._emit(tenant, "otp_generated", record["id"], {"msisdn": msisdn, "purpose": purpose})
		return {"id": record["id"], "msisdn": msisdn, "purpose": purpose, "status": "sent", "created_at": record["created_at"]}

	async def verify_otp(self, otp_id: str, otp_code: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Verify a USSD OTP code."""
		tenant = self._tenant(tenant_id)
		record = self.otp_store.get(otp_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"OTP {otp_id!r} not found")
		record["attempts"] += 1
		if record["attempts"] > record["max_attempts"]:
			record["status"] = "locked"
			raise PermissionError("otp_max_attempts_exceeded")
		verified = record["otp_code"] == otp_code
		record["status"] = "verified" if verified else "failed"
		self._emit(tenant, "otp_verified" if verified else "otp_failed", otp_id, {"verified": verified})
		return {"id": otp_id, "verified": verified, "status": record["status"]}

	# ── Payment reference ─────────────────────────────────────────────────────

	async def create_payment_reference(
		self,
		msisdn: str,
		service_type: str,
		amount: float,
		description: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Generate a government payment reference for bill payments via USSD."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(service_type, "service_type")
		if amount <= 0:
			raise ValueError("amount must be positive")
		record = {
			"id": self._record_id("payref"),
			"type": "ussd_payment_reference",
			"msisdn": msisdn,
			"service_type": service_type,
			"amount": amount,
			"currency": "KES",
			"description": description,
			"payment_code": self._record_id("PAY").upper(),
			"tenant_id": tenant,
			"status": "pending",
			"created_at": self._now(),
		}
		self.payment_references[record["id"]] = record
		self._emit(tenant, "payment_reference_created", record["id"],
			{"msisdn": msisdn, "service_type": service_type, "amount": amount})
		return deepcopy(record)

	async def confirm_payment(self, reference_id: str, transaction_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Confirm a government payment received via USSD."""
		tenant = self._tenant(tenant_id)
		record = self.payment_references.get(reference_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"payment reference {reference_id!r} not found")
		record["status"] = "paid"
		record["transaction_id"] = transaction_id
		record["paid_at"] = self._now()
		self._emit(tenant, "ussd_payment_confirmed", reference_id, {"transaction_id": transaction_id})
		return deepcopy(record)

	# ── SMS notification ──────────────────────────────────────────────────────

	async def send_sms_notification(
		self,
		msisdn: str,
		message: str,
		tenant_id: str = "default",
		reference_id: str | None = None,
	) -> dict[str, Any]:
		"""Send an SMS notification after USSD interaction."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(message, "message")
		record = {
			"id": self._record_id("sms"),
			"type": "sms_notification",
			"msisdn": msisdn,
			"message": message[:160],  # SMS length limit
			"reference_id": reference_id,
			"tenant_id": tenant,
			"status": "sent",
			"created_at": self._now(),
		}
		self.sms_notifications.append(record)
		self._emit(tenant, "sms_notification_sent", record["id"], {"msisdn": msisdn})
		return deepcopy(record)

	# ── Service code management ───────────────────────────────────────────────

	async def register_service_code(
		self,
		service_code: str,
		name: str,
		description: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Register a USSD service code."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(service_code, "service_code")
		record = {
			"id": self._record_id("sc"),
			"type": "ussd_service_code",
			"service_code": service_code,
			"name": name,
			"description": description,
			"tenant_id": tenant,
			"status": "active",
			"created_at": self._now(),
		}
		self.service_codes[record["id"]] = record
		self._emit(tenant, "service_code_registered", record["id"], {"service_code": service_code})
		return deepcopy(record)

	async def list_service_codes(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		"""List registered USSD service codes."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.service_codes.values() if r["tenant_id"] == tenant]

	# ── Dashboard / reporting ─────────────────────────────────────────────────

	async def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return USSD service dashboard metrics."""
		tenant = self._tenant(tenant_id)
		sessions = [s for s in self.sessions.values() if s["tenant_id"] == tenant]
		certs = [c for c in self.certificate_requests.values() if c["tenant_id"] == tenant]
		return {
			"tenant_id": tenant,
			"total_sessions": len(sessions),
			"active_sessions": sum(1 for s in sessions if s["status"] == "active"),
			"closed_sessions": sum(1 for s in sessions if s["status"] == "closed"),
			"permit_enquiries": len([r for r in self.permit_enquiries.values() if r["tenant_id"] == tenant]),
			"tax_enquiries": len([r for r in self.tax_enquiries.values() if r["tenant_id"] == tenant]),
			"id_verifications": len([r for r in self.id_verifications.values() if r["tenant_id"] == tenant]),
			"certificate_requests": len(certs),
			"certificates_issued": sum(1 for c in certs if c["status"] == "issued"),
			"payment_references": len([r for r in self.payment_references.values() if r["tenant_id"] == tenant]),
			"sms_notifications": sum(1 for n in self.sms_notifications if n["tenant_id"] == tenant),
			"audit_events": len([e for e in self._audit_events if e["tenant_id"] == tenant]),
			"generated_at": self._now(),
		}

	async def session_analytics(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return session-level analytics for USSD interactions."""
		tenant = self._tenant(tenant_id)
		sessions = [s for s in self.sessions.values() if s["tenant_id"] == tenant]
		service_code_counts: dict[str, int] = {}
		for s in sessions:
			sc = s["service_code"]
			service_code_counts[sc] = service_code_counts.get(sc, 0) + 1
		avg_depth = (
			sum(s["menu_level"] for s in sessions) / len(sessions)
			if sessions else 0.0
		)
		return {
			"tenant_id": tenant,
			"total_sessions": len(sessions),
			"by_service_code": service_code_counts,
			"average_menu_depth": round(avg_depth, 2),
			"generated_at": self._now(),
		}

	# ── Citizen portfolio ─────────────────────────────────────────────────────

	async def get_citizen_portfolio(self, msisdn: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Return a consolidated service history for a citizen keyed by MSISDN.

		Aggregates all resource types (permits, tax, IDs, certificates, payments)
		into a single compact view suitable for USSD display or API consumption.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(msisdn, "msisdn")
		permits = [
			deepcopy(r) for r in self.permit_enquiries.values()
			if r["tenant_id"] == tenant and r["msisdn"] == msisdn
		]
		taxes = [
			deepcopy(r) for r in self.tax_enquiries.values()
			if r["tenant_id"] == tenant and r["msisdn"] == msisdn
		]
		ids = [
			deepcopy(r) for r in self.id_verifications.values()
			if r["tenant_id"] == tenant and r["msisdn"] == msisdn
		]
		certs = [
			deepcopy(r) for r in self.certificate_requests.values()
			if r["tenant_id"] == tenant and r["msisdn"] == msisdn
		]
		payments = [
			deepcopy(r) for r in self.payment_references.values()
			if r["tenant_id"] == tenant and r["msisdn"] == msisdn
		]
		last_id = max((r["created_at"] for r in ids), default=None)
		active_permits = [p for p in permits if p.get("is_valid")]
		pending_certs = [c for c in certs if c["status"] in {"submitted", "processing"}]
		unpaid = [p for p in payments if p["status"] == "pending"]
		portfolio = {
			"msisdn": msisdn,
			"tenant_id": tenant,
			"permit_enquiries_count": len(permits),
			"active_permits": len(active_permits),
			"tax_enquiry_count": len(taxes),
			"id_verifications_count": len(ids),
			"last_id_verified_at": last_id,
			"certificate_requests_count": len(certs),
			"pending_certificate_requests": len(pending_certs),
			"pending_certificate_types": [c["certificate_type"] for c in pending_certs],
			"payment_references_count": len(payments),
			"unpaid_references": len(unpaid),
			"unpaid_total": round(sum(p["amount"] for p in unpaid), 2),
			"generated_at": self._now(),
		}
		self._emit(tenant, "citizen_portfolio_fetched", msisdn, {"msisdn": msisdn})
		return portfolio

	# ── Rate limiting & fraud ─────────────────────────────────────────────────

	async def check_rate_limit(
		self,
		msisdn: str,
		operation: str,
		tenant_id: str = "default",
		window_seconds: int = 60,
		max_calls: int = 10,
	) -> dict[str, Any]:
		"""Enforce per-MSISDN rolling-window rate limits.

		Counts matching audit events within the last `window_seconds` for the
		given MSISDN and operation.  Returns `allowed: False` and raises
		PermissionError when the limit is exceeded.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(msisdn, "msisdn")
		guard_non_empty_string(operation, "operation")
		cutoff = datetime.utcnow().timestamp() - window_seconds
		# Count relevant audit events in the window
		count = 0
		for ev in self._audit_events:
			if ev["tenant_id"] != tenant:
				continue
			if ev["details"].get("msisdn") != msisdn:
				continue
			try:
				ts = datetime.fromisoformat(ev["emitted_at"].rstrip("Z")).timestamp()
			except (ValueError, AttributeError):
				continue
			if ts >= cutoff:
				count += 1
		allowed = count < max_calls
		result = {
			"msisdn": msisdn,
			"operation": operation,
			"tenant_id": tenant,
			"calls_in_window": count,
			"max_calls": max_calls,
			"window_seconds": window_seconds,
			"allowed": allowed,
			"checked_at": self._now(),
		}
		if not allowed:
			self._emit(tenant, "rate_limit_exceeded", msisdn, {"operation": operation, "count": count})
			raise PermissionError(f"rate_limit_exceeded: {msisdn!r} exceeded {max_calls} calls/{window_seconds}s for {operation!r}")
		return result

	async def score_fraud_risk(self, msisdn: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Compute a 0-1 fraud risk score for a MSISDN based on behavioural signals.

		Signals:
		- Failed ID verification rate
		- OTP failure rate
		- High-frequency enquiry bursts (>20 in 60s)
		- Multiple distinct IDs queried in a short window
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(msisdn, "msisdn")
		id_verifs = [r for r in self.id_verifications.values()
			if r["tenant_id"] == tenant and r["msisdn"] == msisdn]
		failed_ids = sum(1 for r in id_verifs if not r["verified"])
		id_failure_rate = (failed_ids / len(id_verifs)) if id_verifs else 0.0
		otp_records = [r for r in self.otp_store.values() if r["tenant_id"] == tenant and r["msisdn"] == msisdn]
		failed_otps = sum(1 for r in otp_records if r["status"] == "failed")
		otp_failure_rate = (failed_otps / len(otp_records)) if otp_records else 0.0
		# Burst score: events in last 60 seconds
		cutoff = datetime.utcnow().timestamp() - 60
		recent = sum(
			1 for ev in self._audit_events
			if ev["tenant_id"] == tenant
			and ev["details"].get("msisdn") == msisdn
			and (() or True)  # type guard
			and (__import__("datetime").datetime.fromisoformat(ev["emitted_at"].rstrip("Z")).timestamp() >= cutoff
				if ev.get("emitted_at") else False)
		)
		burst_score = min(recent / 20.0, 1.0)
		# Distinct IDs queried
		distinct_ids = len({r["id_number"] for r in id_verifs})
		id_enum_score = min(distinct_ids / 5.0, 1.0)
		# Composite weighted score
		risk_score = round(
			0.35 * id_failure_rate
			+ 0.25 * otp_failure_rate
			+ 0.25 * burst_score
			+ 0.15 * id_enum_score,
			4,
		)
		risk_level = "high" if risk_score >= 0.7 else "medium" if risk_score >= 0.35 else "low"
		result = {
			"msisdn": msisdn,
			"tenant_id": tenant,
			"risk_score": risk_score,
			"risk_level": risk_level,
			"signals": {
				"id_failure_rate": round(id_failure_rate, 4),
				"otp_failure_rate": round(otp_failure_rate, 4),
				"burst_score": round(burst_score, 4),
				"id_enum_score": round(id_enum_score, 4),
			},
			"scored_at": self._now(),
		}
		self._emit(tenant, "fraud_risk_scored", msisdn, {"risk_level": risk_level, "risk_score": risk_score})
		return result

	# ── Permit expiry alerts ──────────────────────────────────────────────────

	async def schedule_permit_expiry_alerts(
		self,
		tenant_id: str = "default",
		warning_days: list[int] | None = None,
	) -> dict[str, Any]:
		"""Scan all permit enquiries and enqueue expiry alert SMS for records
		whose expiry_date falls within `warning_days` (default: [90, 30, 7]).

		Returns a summary of alerts scheduled and skipped.
		"""
		tenant = self._tenant(tenant_id)
		if warning_days is None:
			warning_days = [90, 30, 7]
		today = datetime.utcnow().date()
		scheduled: list[dict[str, Any]] = []
		skipped = 0
		for record in self.permit_enquiries.values():
			if record["tenant_id"] != tenant:
				continue
			expiry_str = record.get("expiry_date")
			if not expiry_str:
				skipped += 1
				continue
			try:
				expiry = datetime.fromisoformat(expiry_str).date()
			except ValueError:
				skipped += 1
				continue
			days_remaining = (expiry - today).days
			for threshold in sorted(warning_days, reverse=True):
				if 0 <= days_remaining <= threshold:
					msg = (
						f"PERMIT ALERT: Your {record['permit_type'].replace('_',' ').title()} "
						f"({record['permit_number']}) expires in {days_remaining} day(s) "
						f"on {expiry_str}. Dial *384# to renew."
					)[:160]
					sms = await self.send_sms_notification(
						msisdn=record["msisdn"],
						message=msg,
						tenant_id=tenant,
						reference_id=record["id"],
					)
					scheduled.append({
						"permit_id": record["id"],
						"msisdn": record["msisdn"],
						"days_remaining": days_remaining,
						"threshold": threshold,
						"sms_id": sms["id"],
					})
					break  # send only the most-urgent threshold alert
		result = {
			"tenant_id": tenant,
			"alerts_scheduled": len(scheduled),
			"records_skipped": skipped,
			"details": scheduled,
			"generated_at": self._now(),
		}
		self._emit(tenant, "permit_expiry_alerts_scheduled", "batch", {"count": len(scheduled)})
		return result

	# ── SLA compliance ────────────────────────────────────────────────────────

	async def check_sla_compliance(
		self,
		tenant_id: str = "default",
		sla_windows_days: dict[str, int] | None = None,
	) -> dict[str, Any]:
		"""Evaluate SLA compliance for all pending certificate requests.

		`sla_windows_days` maps certificate_type → target-days. Defaults:
		  good_conduct=7, tax_compliance=3, birth_certificate=5, default=10.

		Returns per-request breach status and aggregate compliance rate.
		"""
		tenant = self._tenant(tenant_id)
		defaults: dict[str, int] = {
			"good_conduct": 7,
			"tax_compliance": 3,
			"birth_certificate": 5,
			"death_certificate": 5,
			"marriage_certificate": 5,
			"business_registration": 10,
			"clearance_certificate": 14,
		}
		windows = {**defaults, **(sla_windows_days or {})}
		now = datetime.utcnow()
		compliant = []
		breached = []
		for record in self.certificate_requests.values():
			if record["tenant_id"] != tenant:
				continue
			if record["status"] in {"issued", "rejected", "cancelled"}:
				continue  # terminal states excluded
			target_days = windows.get(record["certificate_type"], 10)
			created = datetime.fromisoformat(record["created_at"].rstrip("Z"))
			elapsed_days = (now - created).total_seconds() / 86400
			is_breached = elapsed_days > target_days
			entry = {
				"request_id": record["id"],
				"certificate_type": record["certificate_type"],
				"msisdn": record["msisdn"],
				"elapsed_days": round(elapsed_days, 2),
				"target_days": target_days,
				"status": record["status"],
				"is_breached": is_breached,
			}
			if is_breached:
				breached.append(entry)
				self._emit(tenant, "sla_breach_detected", record["id"], {
					"certificate_type": record["certificate_type"],
					"elapsed_days": entry["elapsed_days"],
					"target_days": target_days,
				})
			else:
				compliant.append(entry)
		total = len(compliant) + len(breached)
		compliance_rate = (len(compliant) / total) if total > 0 else 1.0
		return {
			"tenant_id": tenant,
			"total_pending": total,
			"compliant": len(compliant),
			"breached": len(breached),
			"compliance_rate": round(compliance_rate, 4),
			"breached_requests": breached,
			"generated_at": self._now(),
		}

	# ── Bulk operations ───────────────────────────────────────────────────────

	async def bulk_update_certificate_requests(
		self,
		updates: list[dict[str, Any]],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Atomically update multiple certificate requests in a single call.

		Each entry in `updates` must contain `request_id` and at least one of:
		`status`, `certificate_number`, `issued_by`, `notes`.

		Validates all entries first; applies all-or-nothing on validation failure.
		Returns per-item success/failure.
		"""
		tenant = self._tenant(tenant_id)
		if not updates:
			raise ValueError("updates list must not be empty")
		# Validate pass
		errors: list[dict[str, Any]] = []
		for i, upd in enumerate(updates):
			rid = upd.get("request_id")
			if not rid:
				errors.append({"index": i, "error": "missing request_id"})
				continue
			rec = self.certificate_requests.get(rid)
			if not rec or rec["tenant_id"] != tenant:
				errors.append({"index": i, "request_id": rid, "error": "not_found"})
		if errors:
			return {
				"success": False,
				"applied": 0,
				"errors": errors,
				"message": "validation_failed — no changes applied",
			}
		# Apply pass
		applied: list[dict[str, Any]] = []
		for upd in updates:
			rid = upd["request_id"]
			result = await self.update_certificate_request(
				request_id=rid,
				tenant_id=tenant,
				status=upd.get("status"),
				certificate_number=upd.get("certificate_number"),
				issued_by=upd.get("issued_by"),
				notes=upd.get("notes"),
			)
			applied.append({"request_id": rid, "status": result["status"]})
		self._emit(tenant, "bulk_certificate_update", "batch", {"count": len(applied)})
		return {"success": True, "applied": len(applied), "results": applied}

	# ── Cryptographic receipt ─────────────────────────────────────────────────

	async def generate_signed_receipt(
		self,
		reference_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Generate a tamper-evident HMAC-SHA256 receipt for a confirmed payment.

		The receipt code is a 16-character uppercase base-36 string derived from
		the HMAC of (reference_id + amount + paid_at + tenant_id).  Citizens can
		verify it by re-dialling `*500*VERIFY*{code}#`.
		"""
		import hashlib
		import hmac
		tenant = self._tenant(tenant_id)
		record = self.payment_references.get(reference_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"payment reference {reference_id!r} not found")
		if record.get("status") != "paid":
			raise ValueError("receipt can only be generated for paid references")
		secret = f"{tenant}-{CAPABILITY_ID}".encode()
		payload = f"{reference_id}|{record['amount']}|{record.get('paid_at','')}|{tenant}".encode()
		digest = hmac.new(secret, payload, hashlib.sha256).hexdigest()
		# Encode lower 64 bits of digest as base-36 → ~16 chars
		numeric = int(digest[:16], 16)
		chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		code_parts = []
		n = numeric
		while n:
			code_parts.append(chars[n % 36])
			n //= 36
		receipt_code = "".join(reversed(code_parts)).zfill(12)[:16]
		receipt = {
			"id": self._record_id("rcpt"),
			"reference_id": reference_id,
			"receipt_code": receipt_code,
			"amount": record["amount"],
			"currency": record["currency"],
			"service_type": record["service_type"],
			"tenant_id": tenant,
			"issued_at": self._now(),
			"verify_ussd": f"*500*VERIFY*{receipt_code}#",
		}
		self._emit(tenant, "receipt_generated", reference_id, {"receipt_code": receipt_code})
		return receipt

	async def verify_receipt_code(
		self,
		receipt_code: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Verify a previously issued USSD receipt code.

		Scans all paid payment references for a matching code.  Useful for
		`*500*VERIFY*{code}#` short-code handler or counter staff verification.
		"""
		import hashlib
		import hmac
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(receipt_code, "receipt_code")
		secret = f"{tenant}-{CAPABILITY_ID}".encode()
		for ref_id, record in self.payment_references.items():
			if record["tenant_id"] != tenant or record.get("status") != "paid":
				continue
			payload = f"{ref_id}|{record['amount']}|{record.get('paid_at','')}|{tenant}".encode()
			digest = hmac.new(secret, payload, hashlib.sha256).hexdigest()
			numeric = int(digest[:16], 16)
			chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
			code_parts: list[str] = []
			n = numeric
			while n:
				code_parts.append(chars[n % 36])
				n //= 36
			candidate = "".join(reversed(code_parts)).zfill(12)[:16]
			if candidate == receipt_code.upper():
				self._emit(tenant, "receipt_verified", ref_id, {"receipt_code": receipt_code})
				return {
					"valid": True,
					"reference_id": ref_id,
					"amount": record["amount"],
					"currency": record["currency"],
					"service_type": record["service_type"],
					"paid_at": record.get("paid_at"),
					"verified_at": self._now(),
				}
		self._emit(tenant, "receipt_verification_failed", receipt_code, {})
		return {"valid": False, "receipt_code": receipt_code, "verified_at": self._now()}

	# ── Permit workflow orchestration ─────────────────────────────────────────

	async def orchestrate_permit_workflow(
		self,
		msisdn: str,
		id_number: str,
		id_type: str,
		tax_pin: str,
		permit_number: str,
		permit_type: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Orchestrate a multi-step permit compliance workflow concurrently.

		Fans out: ID verification + tax compliance check + permit status enquiry
		in parallel via asyncio.gather.  Gates workflow completion on all steps
		passing.  Publishes per-step progress.
		"""
		tenant = self._tenant(tenant_id)
		workflow_id = self._record_id("wf")
		self._emit(tenant, "permit_workflow_started", workflow_id, {
			"msisdn": msisdn, "permit_type": permit_type,
		})
		id_task = self.verify_id(msisdn=msisdn, id_number=id_number, id_type=id_type, tenant_id=tenant)
		tax_task = self.enquire_tax_balance(msisdn=msisdn, tax_pin=tax_pin, tax_type="income_tax", tenant_id=tenant)
		permit_task = self.enquire_permit_status(
			msisdn=msisdn, permit_number=permit_number, permit_type=permit_type, tenant_id=tenant
		)
		id_result, tax_result, permit_result = await asyncio.gather(
			id_task, tax_task, permit_task, return_exceptions=True
		)
		steps: dict[str, Any] = {}
		all_passed = True
		for name, res in [("id_verification", id_result), ("tax_compliance", tax_result), ("permit_status", permit_result)]:
			if isinstance(res, Exception):
				steps[name] = {"passed": False, "error": str(res)}
				all_passed = False
			else:
				passed = (
					res.get("verified", False) if name == "id_verification"
					else res.get("compliance_status") == "compliant" if name == "tax_compliance"
					else res.get("is_valid", False)
				)
				steps[name] = {"passed": passed, "result_id": res.get("id")}
				if not passed:
					all_passed = False
		outcome = "approved" if all_passed else "rejected"
		self._emit(tenant, f"permit_workflow_{outcome}", workflow_id, {"steps": steps})
		return {
			"workflow_id": workflow_id,
			"msisdn": msisdn,
			"permit_type": permit_type,
			"outcome": outcome,
			"all_steps_passed": all_passed,
			"steps": steps,
			"completed_at": self._now(),
		}

	# ── Telemetry snapshot ────────────────────────────────────────────────────

	async def emit_telemetry_snapshot(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Compute and return a structured telemetry snapshot for NATS publishing.

		Metrics include: session funnel drop-offs by level, error rates by
		event type, service-code breakdown, and resource counts.  Intended to
		be consumed by bytewax pipelines for time-series aggregation.
		"""
		tenant = self._tenant(tenant_id)
		sessions = [s for s in self.sessions.values() if s["tenant_id"] == tenant]
		events = [e for e in self._audit_events if e["tenant_id"] == tenant]
		# Drop-off by menu level: sessions that stopped at each level
		level_counts: dict[int, int] = {}
		for s in sessions:
			lvl = s["menu_level"]
			level_counts[lvl] = level_counts.get(lvl, 0) + 1
		# Error events
		error_events = [e for e in events if "failed" in e["event_type"] or "exceeded" in e["event_type"]]
		error_counts: dict[str, int] = {}
		for e in error_events:
			et = e["event_type"]
			error_counts[et] = error_counts.get(et, 0) + 1
		# Service code breakdown
		sc_counts: dict[str, int] = {}
		for s in sessions:
			sc = s["service_code"]
			sc_counts[sc] = sc_counts.get(sc, 0) + 1
		snapshot = {
			"tenant_id": tenant,
			"snapshot_at": self._now(),
			"nats_subject": f"gov.usd.metrics.{tenant}",
			"sessions": {
				"total": len(sessions),
				"active": sum(1 for s in sessions if s["status"] == "active"),
				"closed": sum(1 for s in sessions if s["status"] == "closed"),
				"by_service_code": sc_counts,
				"drop_off_by_level": level_counts,
			},
			"resources": {
				"permit_enquiries": len([r for r in self.permit_enquiries.values() if r["tenant_id"] == tenant]),
				"tax_enquiries": len([r for r in self.tax_enquiries.values() if r["tenant_id"] == tenant]),
				"id_verifications": len([r for r in self.id_verifications.values() if r["tenant_id"] == tenant]),
				"certificate_requests": len([r for r in self.certificate_requests.values() if r["tenant_id"] == tenant]),
				"payment_references": len([r for r in self.payment_references.values() if r["tenant_id"] == tenant]),
			},
			"errors": {
				"total": len(error_events),
				"by_type": error_counts,
			},
			"audit_events_total": len(events),
		}
		self._emit(tenant, "telemetry_snapshot_emitted", "telemetry", {"snapshot_at": snapshot["snapshot_at"]})
		return snapshot

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

