"""Pydantic v2 models for SACCO Member Registry."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid_extensions import uuid7str
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())


# ── Member models ─────────────────────────────────────────────────────────────

class MemberCreateModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	full_name: str
	national_id: str
	phone: str
	email: str | None = None
	date_of_birth: date
	gender: str  # M | F | O
	county: str
	sub_county: str | None = None
	postal_address: str | None = None
	occupation: str | None = None
	employer: str | None = None
	monthly_income: Decimal | None = None
	membership_type: str = "ordinary"  # ordinary | associate | institutional
	entry_fee: Decimal = Decimal("0")
	minimum_shares: int = 1
	next_of_kin_name: str | None = None
	next_of_kin_phone: str | None = None
	next_of_kin_relationship: str | None = None
	referred_by: str | None = None


class MemberUpdateModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	phone: str | None = None
	email: str | None = None
	county: str | None = None
	sub_county: str | None = None
	postal_address: str | None = None
	occupation: str | None = None
	employer: str | None = None
	monthly_income: Decimal | None = None
	next_of_kin_name: str | None = None
	next_of_kin_phone: str | None = None
	next_of_kin_relationship: str | None = None


class MemberResponseModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	member_number: str
	tenant_id: str
	full_name: str
	national_id: str
	phone: str
	email: str | None = None
	date_of_birth: date | None = None
	gender: str | None = None
	county: str | None = None
	membership_type: str
	status: str  # pending | active | suspended | exited
	kyc_status: str  # pending | verified | rejected
	share_capital: Decimal = Decimal("0")
	total_shares: int = 0
	entry_fee_paid: bool = False
	created_at: str
	updated_at: str | None = None


class MemberListModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	items: list[dict[str, Any]] = Field(default_factory=list)
	total: int = 0
	page: int = 1
	page_size: int = 50


class MemberFilterModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	status: str | None = None
	kyc_status: str | None = None
	membership_type: str | None = None
	county: str | None = None
	from_date: str | None = None
	to_date: str | None = None
	search: str | None = None


# ── KYC models ────────────────────────────────────────────────────────────────

class KYCSubmissionModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	document_type: str  # national_id | passport | alien_id | driving_licence
	document_number: str
	document_front_ref: str
	document_back_ref: str | None = None
	selfie_ref: str | None = None
	submitted_by: str


class KYCVerificationModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	kyc_id: str
	decision: str  # approved | rejected
	verified_by: str
	notes: str | None = None
	rejection_reason: str | None = None


# ── Share capital models ──────────────────────────────────────────────────────

class ShareCapitalModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	shares: int
	share_value: Decimal
	payment_reference: str
	payment_method: str = "cash"  # cash | mpesa | bank_transfer
	recorded_by: str


class ShareTransferModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	from_member_id: str
	to_member_id: str
	shares: int
	transfer_reason: str
	approved_by: str


# ── Guarantor models ──────────────────────────────────────────────────────────

class GuarantorRelationshipModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	guarantor_member_id: str
	beneficiary_member_id: str
	relationship_type: str  # personal | employment | business
	max_guarantee_amount: Decimal
	notes: str | None = None


# ── Exit models ───────────────────────────────────────────────────────────────

class MemberExitModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	exit_reason: str  # resignation | death | expulsion | transfer
	exit_date: date
	processed_by: str
	notes: str | None = None


# ── Audit model ───────────────────────────────────────────────────────────────

class MemberAuditModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	member_id: str
	actor_id: str | None = None
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: str
