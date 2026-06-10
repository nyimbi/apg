"""SACCO Member Registry — full async service."""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

_log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CAPABILITY_ID = "fintech_sacco_mem"
MEMBERSHIP_TYPES = {"ordinary", "associate", "institutional"}
KYC_DOCUMENT_TYPES = {"national_id", "passport", "alien_id", "driving_licence"}
EXIT_REASONS = {"resignation", "death", "expulsion", "transfer"}
MEMBER_STATUSES = {"pending", "active", "suspended", "exited"}
KYC_STATUSES = {"pending", "verified", "rejected"}
PAYMENT_METHODS = {"cash", "mpesa", "bank_transfer", "cheque"}


class SaccoMemberRegistryService:
	"""In-memory async service for SACCO member lifecycle management."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.members: dict[str, dict[str, Any]] = {}
		self.kyc_records: dict[str, dict[str, Any]] = {}
		self.share_transactions: dict[str, dict[str, Any]] = {}
		self.guarantor_relationships: dict[str, dict[str, Any]] = {}
		self.exit_records: dict[str, dict[str, Any]] = {}
		self.membership_fees: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []
		self._member_counter: int = 0

	# ── Internal helpers ──────────────────────────────────────────────────────

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _next_member_number(self, tenant_id: str) -> str:
		self._member_counter += 1
		return f"MEM-{tenant_id[:4].upper()}-{self._member_counter:06d}"

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record.get("id", ""),
			"record_type": record.get("type", "member"),
			"status": record.get("status", ""),
			"emitted_at": self._now(),
		})

	def _get_member(self, member_id: str, tenant_id: str) -> dict[str, Any]:
		member = self.members.get(member_id)
		if not member or member["tenant_id"] != tenant_id:
			raise KeyError(f"member_not_found: {member_id}")
		return member

	# ── Health & Describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"member_count": len(self.members),
			"kyc_pending": sum(1 for m in self.members.values() if m.get("kyc_status") == "pending"),
			"active_members": sum(1 for m in self.members.values() if m.get("status") == "active"),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		"""Return capability contract description."""
		return {
			"capability_id": CAPABILITY_ID,
			"version": "1.0.0",
			"domain": "fintech",
			"description": "SACCO member onboarding, KYC, share capital, guarantors, exit processing",
			"membership_types": list(MEMBERSHIP_TYPES),
			"kyc_document_types": list(KYC_DOCUMENT_TYPES),
			"exit_reasons": list(EXIT_REASONS),
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return audit event log for tenant."""
		t = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == t]

	# ── Member CRUD ───────────────────────────────────────────────────────────

	async def list_members(
		self,
		tenant_id: str | None = None,
		status: str | None = None,
		membership_type: str | None = None,
		kyc_status: str | None = None,
		county: str | None = None,
	) -> list[dict[str, Any]]:
		"""List all members with optional filters."""
		t = self._tenant(tenant_id)
		items = [deepcopy(m) for m in self.members.values() if m["tenant_id"] == t]
		if status:
			items = [m for m in items if m.get("status") == status]
		if membership_type:
			items = [m for m in items if m.get("membership_type") == membership_type]
		if kyc_status:
			items = [m for m in items if m.get("kyc_status") == kyc_status]
		if county:
			items = [m for m in items if m.get("county") == county]
		return items

	async def get_member(self, member_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Fetch a single member record."""
		t = self._tenant(tenant_id)
		return deepcopy(self._get_member(member_id, t))

	async def create_member(
		self,
		full_name: str,
		national_id: str,
		phone: str,
		date_of_birth: str,
		gender: str,
		county: str,
		tenant_id: str | None = None,
		email: str | None = None,
		membership_type: str = "ordinary",
		sub_county: str | None = None,
		postal_address: str | None = None,
		occupation: str | None = None,
		employer: str | None = None,
		monthly_income: float | None = None,
		entry_fee: float = 0.0,
		minimum_shares: int = 1,
		next_of_kin_name: str | None = None,
		next_of_kin_phone: str | None = None,
		next_of_kin_relationship: str | None = None,
		referred_by: str | None = None,
	) -> dict[str, Any]:
		"""Onboard a new SACCO member."""
		t = self._tenant(tenant_id)
		if not full_name.strip():
			raise ValueError("full_name_required")
		if not national_id.strip():
			raise ValueError("national_id_required")
		if membership_type not in MEMBERSHIP_TYPES:
			raise ValueError(f"invalid_membership_type: {membership_type}")
		# Check for duplicate national_id within tenant
		for m in self.members.values():
			if m["tenant_id"] == t and m["national_id"] == national_id:
				raise ValueError(f"member_already_exists: {national_id}")
		member_number = self._next_member_number(t)
		record: dict[str, Any] = {
			"id": self._record_id("mem"),
			"type": "sacco_member",
			"tenant_id": t,
			"member_number": member_number,
			"full_name": full_name,
			"national_id": national_id,
			"phone": phone,
			"email": email,
			"date_of_birth": date_of_birth,
			"gender": gender,
			"county": county,
			"sub_county": sub_county,
			"postal_address": postal_address,
			"occupation": occupation,
			"employer": employer,
			"monthly_income": Decimal(str(monthly_income)) if monthly_income else None,
			"membership_type": membership_type,
			"status": "pending",
			"kyc_status": "pending",
			"share_capital": Decimal("0"),
			"total_shares": 0,
			"entry_fee_paid": entry_fee <= 0,
			"entry_fee_amount": Decimal(str(entry_fee)),
			"minimum_shares": minimum_shares,
			"next_of_kin_name": next_of_kin_name,
			"next_of_kin_phone": next_of_kin_phone,
			"next_of_kin_relationship": next_of_kin_relationship,
			"referred_by": referred_by,
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self.members[record["id"]] = record
		self._emit(t, "member_created", record)
		_log.info("Member created: %s tenant=%s", member_number, t)
		return deepcopy(record)

	async def update_member(
		self,
		member_id: str,
		tenant_id: str | None = None,
		phone: str | None = None,
		email: str | None = None,
		county: str | None = None,
		sub_county: str | None = None,
		postal_address: str | None = None,
		occupation: str | None = None,
		employer: str | None = None,
		monthly_income: float | None = None,
		next_of_kin_name: str | None = None,
		next_of_kin_phone: str | None = None,
		next_of_kin_relationship: str | None = None,
	) -> dict[str, Any]:
		"""Update mutable member fields."""
		t = self._tenant(tenant_id)
		member = self._get_member(member_id, t)
		if phone is not None:
			member["phone"] = phone
		if email is not None:
			member["email"] = email
		if county is not None:
			member["county"] = county
		if sub_county is not None:
			member["sub_county"] = sub_county
		if postal_address is not None:
			member["postal_address"] = postal_address
		if occupation is not None:
			member["occupation"] = occupation
		if employer is not None:
			member["employer"] = employer
		if monthly_income is not None:
			member["monthly_income"] = Decimal(str(monthly_income))
		if next_of_kin_name is not None:
			member["next_of_kin_name"] = next_of_kin_name
		if next_of_kin_phone is not None:
			member["next_of_kin_phone"] = next_of_kin_phone
		if next_of_kin_relationship is not None:
			member["next_of_kin_relationship"] = next_of_kin_relationship
		member["updated_at"] = self._now()
		self._emit(t, "member_updated", member)
		return deepcopy(member)

	async def delete_member(self, member_id: str, tenant_id: str | None = None, reason: str = "admin_delete") -> dict[str, Any]:
		"""Soft-delete (mark exited) a member record."""
		t = self._tenant(tenant_id)
		member = self._get_member(member_id, t)
		if member["status"] == "exited":
			raise ValueError("member_already_exited")
		member["status"] = "exited"
		member["exit_reason"] = reason
		member["exited_at"] = self._now()
		member["updated_at"] = self._now()
		self._emit(t, "member_exited", member)
		return deepcopy(member)

	# ── KYC ───────────────────────────────────────────────────────────────────

	async def submit_kyc(
		self,
		member_id: str,
		document_type: str,
		document_number: str,
		document_front_ref: str,
		submitted_by: str,
		tenant_id: str | None = None,
		document_back_ref: str | None = None,
		selfie_ref: str | None = None,
	) -> dict[str, Any]:
		"""Submit KYC documents for a member."""
		t = self._tenant(tenant_id)
		member = self._get_member(member_id, t)
		if document_type not in KYC_DOCUMENT_TYPES:
			raise ValueError(f"invalid_document_type: {document_type}")
		kyc_id = self._record_id("kyc")
		record: dict[str, Any] = {
			"id": kyc_id,
			"type": "sacco_kyc",
			"tenant_id": t,
			"member_id": member_id,
			"document_type": document_type,
			"document_number": document_number,
			"document_front_ref": document_front_ref,
			"document_back_ref": document_back_ref,
			"selfie_ref": selfie_ref,
			"submitted_by": submitted_by,
			"status": "pending",
			"created_at": self._now(),
		}
		self.kyc_records[kyc_id] = record
		member["kyc_status"] = "pending"
		member["latest_kyc_id"] = kyc_id
		self._emit(t, "kyc_submitted", record)
		return deepcopy(record)

	async def approve_kyc(
		self,
		kyc_id: str,
		verified_by: str,
		tenant_id: str | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Approve a KYC submission and activate the member."""
		t = self._tenant(tenant_id)
		kyc = self.kyc_records.get(kyc_id)
		if not kyc or kyc["tenant_id"] != t:
			raise KeyError(f"kyc_not_found: {kyc_id}")
		kyc["status"] = "approved"
		kyc["verified_by"] = verified_by
		kyc["notes"] = notes
		kyc["verified_at"] = self._now()
		member = self._get_member(kyc["member_id"], t)
		member["kyc_status"] = "verified"
		if member["status"] == "pending" and member.get("entry_fee_paid"):
			member["status"] = "active"
		member["updated_at"] = self._now()
		self._emit(t, "kyc_approved", kyc)
		return deepcopy(kyc)

	async def reject_kyc(
		self,
		kyc_id: str,
		verified_by: str,
		rejection_reason: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Reject a KYC submission."""
		t = self._tenant(tenant_id)
		kyc = self.kyc_records.get(kyc_id)
		if not kyc or kyc["tenant_id"] != t:
			raise KeyError(f"kyc_not_found: {kyc_id}")
		kyc["status"] = "rejected"
		kyc["verified_by"] = verified_by
		kyc["rejection_reason"] = rejection_reason
		kyc["verified_at"] = self._now()
		member = self._get_member(kyc["member_id"], t)
		member["kyc_status"] = "rejected"
		member["updated_at"] = self._now()
		self._emit(t, "kyc_rejected", kyc)
		return deepcopy(kyc)

	async def list_kyc_records(self, tenant_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		"""List KYC records, optionally filtered by status."""
		t = self._tenant(tenant_id)
		items = [deepcopy(k) for k in self.kyc_records.values() if k["tenant_id"] == t]
		if status:
			items = [k for k in items if k.get("status") == status]
		return items

	# ── Share Capital ─────────────────────────────────────────────────────────

	async def purchase_shares(
		self,
		member_id: str,
		shares: int,
		share_value: float,
		payment_reference: str,
		recorded_by: str,
		tenant_id: str | None = None,
		payment_method: str = "cash",
	) -> dict[str, Any]:
		"""Record share capital purchase by a member."""
		t = self._tenant(tenant_id)
		member = self._get_member(member_id, t)
		if shares <= 0:
			raise ValueError("shares_must_be_positive")
		if share_value <= 0:
			raise ValueError("share_value_must_be_positive")
		if payment_method not in PAYMENT_METHODS:
			raise ValueError(f"invalid_payment_method: {payment_method}")
		amount = Decimal(str(shares)) * Decimal(str(share_value))
		txn_id = self._record_id("shr")
		record: dict[str, Any] = {
			"id": txn_id,
			"type": "sacco_share_purchase",
			"tenant_id": t,
			"member_id": member_id,
			"member_number": member.get("member_number"),
			"shares": shares,
			"share_value": Decimal(str(share_value)),
			"amount": amount,
			"payment_reference": payment_reference,
			"payment_method": payment_method,
			"recorded_by": recorded_by,
			"status": "completed",
			"created_at": self._now(),
		}
		self.share_transactions[txn_id] = record
		member["share_capital"] = member.get("share_capital", Decimal("0")) + amount
		member["total_shares"] = member.get("total_shares", 0) + shares
		member["updated_at"] = self._now()
		self._emit(t, "shares_purchased", record)
		_log.info("Share purchase: member=%s shares=%d amount=%s", member_id, shares, amount)
		return deepcopy(record)

	async def transfer_shares(
		self,
		from_member_id: str,
		to_member_id: str,
		shares: int,
		transfer_reason: str,
		approved_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Transfer shares from one member to another."""
		t = self._tenant(tenant_id)
		from_member = self._get_member(from_member_id, t)
		to_member = self._get_member(to_member_id, t)
		if from_member.get("total_shares", 0) < shares:
			raise ValueError("insufficient_shares")
		avg_value = (from_member.get("share_capital", Decimal("0")) / Decimal(str(from_member["total_shares"]))) if from_member["total_shares"] > 0 else Decimal("0")
		transfer_amount = avg_value * Decimal(str(shares))
		txn_id = self._record_id("stx")
		record: dict[str, Any] = {
			"id": txn_id,
			"type": "sacco_share_transfer",
			"tenant_id": t,
			"from_member_id": from_member_id,
			"to_member_id": to_member_id,
			"shares": shares,
			"transfer_amount": transfer_amount,
			"transfer_reason": transfer_reason,
			"approved_by": approved_by,
			"status": "completed",
			"created_at": self._now(),
		}
		self.share_transactions[txn_id] = record
		from_member["total_shares"] -= shares
		from_member["share_capital"] -= transfer_amount
		to_member["total_shares"] = to_member.get("total_shares", 0) + shares
		to_member["share_capital"] = to_member.get("share_capital", Decimal("0")) + transfer_amount
		self._emit(t, "shares_transferred", record)
		return deepcopy(record)

	async def list_share_transactions(self, member_id: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List share transactions, optionally filtered by member."""
		t = self._tenant(tenant_id)
		items = [deepcopy(s) for s in self.share_transactions.values() if s["tenant_id"] == t]
		if member_id:
			items = [s for s in items if s.get("member_id") == member_id or s.get("from_member_id") == member_id or s.get("to_member_id") == member_id]
		return items

	# ── Membership Fees ───────────────────────────────────────────────────────

	async def record_membership_fee(
		self,
		member_id: str,
		amount: float,
		payment_reference: str,
		fee_type: str,
		recorded_by: str,
		tenant_id: str | None = None,
		payment_method: str = "cash",
	) -> dict[str, Any]:
		"""Record a membership fee payment (entry, annual, etc.)."""
		t = self._tenant(tenant_id)
		member = self._get_member(member_id, t)
		fee_id = self._record_id("fee")
		record: dict[str, Any] = {
			"id": fee_id,
			"type": "sacco_membership_fee",
			"tenant_id": t,
			"member_id": member_id,
			"member_number": member.get("member_number"),
			"amount": Decimal(str(amount)),
			"fee_type": fee_type,  # entry | annual | registration
			"payment_reference": payment_reference,
			"payment_method": payment_method,
			"recorded_by": recorded_by,
			"status": "paid",
			"created_at": self._now(),
		}
		self.membership_fees[fee_id] = record
		if fee_type == "entry":
			member["entry_fee_paid"] = True
			if member.get("kyc_status") == "verified" and member.get("status") == "pending":
				member["status"] = "active"
		member["updated_at"] = self._now()
		self._emit(t, "membership_fee_paid", record)
		return deepcopy(record)

	async def list_membership_fees(self, member_id: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List membership fee records."""
		t = self._tenant(tenant_id)
		items = [deepcopy(f) for f in self.membership_fees.values() if f["tenant_id"] == t]
		if member_id:
			items = [f for f in items if f.get("member_id") == member_id]
		return items

	# ── Guarantor Relationships ───────────────────────────────────────────────

	async def create_guarantor_relationship(
		self,
		guarantor_member_id: str,
		beneficiary_member_id: str,
		relationship_type: str,
		max_guarantee_amount: float,
		tenant_id: str | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Establish a guarantor relationship between two members."""
		t = self._tenant(tenant_id)
		guarantor = self._get_member(guarantor_member_id, t)
		beneficiary = self._get_member(beneficiary_member_id, t)
		if guarantor.get("status") != "active":
			raise ValueError("guarantor_must_be_active")
		if guarantor_member_id == beneficiary_member_id:
			raise ValueError("self_guarantee_not_allowed")
		rel_id = self._record_id("guar")
		record: dict[str, Any] = {
			"id": rel_id,
			"type": "sacco_guarantor_relationship",
			"tenant_id": t,
			"guarantor_member_id": guarantor_member_id,
			"guarantor_name": guarantor.get("full_name"),
			"beneficiary_member_id": beneficiary_member_id,
			"beneficiary_name": beneficiary.get("full_name"),
			"relationship_type": relationship_type,
			"max_guarantee_amount": Decimal(str(max_guarantee_amount)),
			"utilized_amount": Decimal("0"),
			"notes": notes,
			"status": "active",
			"created_at": self._now(),
		}
		self.guarantor_relationships[rel_id] = record
		self._emit(t, "guarantor_relationship_created", record)
		return deepcopy(record)

	async def list_guarantor_relationships(
		self,
		member_id: str | None = None,
		role: str = "any",
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List guarantor relationships (role: guarantor | beneficiary | any)."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.guarantor_relationships.values() if r["tenant_id"] == t]
		if member_id:
			if role == "guarantor":
				items = [r for r in items if r["guarantor_member_id"] == member_id]
			elif role == "beneficiary":
				items = [r for r in items if r["beneficiary_member_id"] == member_id]
			else:
				items = [r for r in items if r["guarantor_member_id"] == member_id or r["beneficiary_member_id"] == member_id]
		return items

	async def deactivate_guarantor_relationship(
		self,
		relationship_id: str,
		reason: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Deactivate a guarantor relationship."""
		t = self._tenant(tenant_id)
		rel = self.guarantor_relationships.get(relationship_id)
		if not rel or rel["tenant_id"] != t:
			raise KeyError(f"guarantor_relationship_not_found: {relationship_id}")
		rel["status"] = "inactive"
		rel["deactivation_reason"] = reason
		rel["deactivated_at"] = self._now()
		self._emit(t, "guarantor_relationship_deactivated", rel)
		return deepcopy(rel)

	# ── Member Activation ─────────────────────────────────────────────────────

	async def activate_member(self, member_id: str, activated_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Manually activate a pending member after KYC and fee verification."""
		t = self._tenant(tenant_id)
		member = self._get_member(member_id, t)
		if member["status"] not in {"pending"}:
			raise ValueError(f"cannot_activate_member_in_status: {member['status']}")
		member["status"] = "active"
		member["activated_by"] = activated_by
		member["activated_at"] = self._now()
		member["updated_at"] = self._now()
		self._emit(t, "member_activated", member)
		return deepcopy(member)

	async def suspend_member(self, member_id: str, reason: str, suspended_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Suspend an active member."""
		t = self._tenant(tenant_id)
		member = self._get_member(member_id, t)
		if member["status"] != "active":
			raise ValueError("only_active_members_can_be_suspended")
		member["status"] = "suspended"
		member["suspension_reason"] = reason
		member["suspended_by"] = suspended_by
		member["suspended_at"] = self._now()
		member["updated_at"] = self._now()
		self._emit(t, "member_suspended", member)
		return deepcopy(member)

	async def reinstate_member(self, member_id: str, reinstated_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Reinstate a suspended member."""
		t = self._tenant(tenant_id)
		member = self._get_member(member_id, t)
		if member["status"] != "suspended":
			raise ValueError("only_suspended_members_can_be_reinstated")
		member["status"] = "active"
		member["reinstated_by"] = reinstated_by
		member["reinstated_at"] = self._now()
		member["updated_at"] = self._now()
		self._emit(t, "member_reinstated", member)
		return deepcopy(member)

	# ── Exit Processing ───────────────────────────────────────────────────────

	async def initiate_exit(
		self,
		member_id: str,
		exit_reason: str,
		exit_date: str,
		processed_by: str,
		tenant_id: str | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Initiate the exit process for a member."""
		t = self._tenant(tenant_id)
		member = self._get_member(member_id, t)
		if exit_reason not in EXIT_REASONS:
			raise ValueError(f"invalid_exit_reason: {exit_reason}")
		if member["status"] == "exited":
			raise ValueError("member_already_exited")
		exit_id = self._record_id("exit")
		record: dict[str, Any] = {
			"id": exit_id,
			"type": "sacco_member_exit",
			"tenant_id": t,
			"member_id": member_id,
			"member_number": member.get("member_number"),
			"exit_reason": exit_reason,
			"exit_date": exit_date,
			"share_capital_refundable": member.get("share_capital", Decimal("0")),
			"total_shares": member.get("total_shares", 0),
			"processed_by": processed_by,
			"notes": notes,
			"status": "pending",
			"created_at": self._now(),
		}
		self.exit_records[exit_id] = record
		member["exit_initiated"] = True
		member["exit_id"] = exit_id
		member["updated_at"] = self._now()
		self._emit(t, "exit_initiated", record)
		return deepcopy(record)

	async def complete_exit(
		self,
		exit_id: str,
		approved_by: str,
		settlement_reference: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Complete the exit process after share capital refund."""
		t = self._tenant(tenant_id)
		exit_rec = self.exit_records.get(exit_id)
		if not exit_rec or exit_rec["tenant_id"] != t:
			raise KeyError(f"exit_record_not_found: {exit_id}")
		if exit_rec["status"] != "pending":
			raise ValueError("exit_already_processed")
		exit_rec["status"] = "completed"
		exit_rec["approved_by"] = approved_by
		exit_rec["settlement_reference"] = settlement_reference
		exit_rec["completed_at"] = self._now()
		member = self._get_member(exit_rec["member_id"], t)
		member["status"] = "exited"
		member["share_capital"] = Decimal("0")
		member["total_shares"] = 0
		member["updated_at"] = self._now()
		self._emit(t, "exit_completed", exit_rec)
		return deepcopy(exit_rec)

	async def list_exit_records(self, tenant_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		"""List exit records."""
		t = self._tenant(tenant_id)
		items = [deepcopy(e) for e in self.exit_records.values() if e["tenant_id"] == t]
		if status:
			items = [e for e in items if e.get("status") == status]
		return items

	# ── Analytics & Reporting ─────────────────────────────────────────────────

	async def membership_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return membership statistics for the tenant."""
		t = self._tenant(tenant_id)
		members = [m for m in self.members.values() if m["tenant_id"] == t]
		by_status: dict[str, int] = {}
		by_type: dict[str, int] = {}
		by_county: dict[str, int] = {}
		total_share_capital = Decimal("0")
		for m in members:
			by_status[m.get("status", "unknown")] = by_status.get(m.get("status", "unknown"), 0) + 1
			by_type[m.get("membership_type", "unknown")] = by_type.get(m.get("membership_type", "unknown"), 0) + 1
			county = m.get("county", "unknown")
			by_county[county] = by_county.get(county, 0) + 1
			total_share_capital += m.get("share_capital", Decimal("0"))
		return {
			"tenant_id": t,
			"total_members": len(members),
			"by_status": by_status,
			"by_membership_type": by_type,
			"by_county": by_county,
			"total_share_capital": str(total_share_capital),
			"kyc_pending": sum(1 for m in members if m.get("kyc_status") == "pending"),
			"kyc_verified": sum(1 for m in members if m.get("kyc_status") == "verified"),
			"generated_at": self._now(),
		}

	async def get_member_statement(self, member_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Generate a member statement with share and fee history."""
		t = self._tenant(tenant_id)
		member = self._get_member(member_id, t)
		share_txns = [s for s in self.share_transactions.values() if s["tenant_id"] == t and (s.get("member_id") == member_id or s.get("from_member_id") == member_id or s.get("to_member_id") == member_id)]
		fee_txns = [f for f in self.membership_fees.values() if f["tenant_id"] == t and f.get("member_id") == member_id]
		guarantor_rels = [r for r in self.guarantor_relationships.values() if r["tenant_id"] == t and r.get("guarantor_member_id") == member_id]
		return {
			"member_id": member_id,
			"member_number": member.get("member_number"),
			"full_name": member.get("full_name"),
			"status": member.get("status"),
			"kyc_status": member.get("kyc_status"),
			"share_capital": str(member.get("share_capital", Decimal("0"))),
			"total_shares": member.get("total_shares", 0),
			"share_transactions": share_txns,
			"membership_fees": fee_txns,
			"active_guarantees": [r for r in guarantor_rels if r.get("status") == "active"],
			"generated_at": self._now(),
		}

	async def search_members(self, query: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Full-text search on member name, national ID, phone, or member number."""
		t = self._tenant(tenant_id)
		q = query.lower()
		return [
			deepcopy(m) for m in self.members.values()
			if m["tenant_id"] == t and any(
				q in str(m.get(field, "")).lower()
				for field in ("full_name", "national_id", "phone", "member_number", "email")
			)
		]

	async def bulk_activate_members(
		self,
		member_ids: list[str],
		activated_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Bulk activate multiple pending members."""
		t = self._tenant(tenant_id)
		results, errors = [], []
		for mid in member_ids:
			try:
				rec = await self.activate_member(mid, activated_by, tenant_id=t)
				results.append(rec)
			except Exception as exc:
				_log.error("bulk_activate error member=%s: %s", mid, exc)
				errors.append({"member_id": mid, "error": str(exc)})
		return {"activated": len(results), "failed": len(errors), "results": results, "errors": errors}

	async def export_register(self, tenant_id: str | None = None, fmt: str = "json") -> dict[str, Any]:
		"""Export the full member register."""
		t = self._tenant(tenant_id)
		assert fmt in {"json", "csv", "excel"}, "fmt must be json|csv|excel"
		members = [m for m in self.members.values() if m["tenant_id"] == t]
		return {
			"tenant_id": t,
			"format": fmt,
			"record_count": len(members),
			"export_reference": f"register-{t}-{self._now()[:10]}.{fmt}",
			"generated_at": self._now(),
		}


# ── Alias ─────────────────────────────────────────────────────────────────────
MemberRegistryService = SaccoMemberRegistryService
