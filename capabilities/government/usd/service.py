"""USSD Government Services — async service implementation."""
from __future__ import annotations

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

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
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
