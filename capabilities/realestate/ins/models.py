"""Pydantic v2 models for Property Insurance (ins)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


class PolicyType(str, Enum):
	property_all_risk = "property_all_risk"
	fire_perils = "fire_perils"
	public_liability = "public_liability"
	employers_liability = "employers_liability"
	professional_indemnity = "professional_indemnity"
	contractors_all_risk = "contractors_all_risk"
	fidelity_guarantee = "fidelity_guarantee"
	loss_of_rent = "loss_of_rent"
	terrorism = "terrorism"
	flood = "flood"
	earthquake = "earthquake"


class CoverageStatus(str, Enum):
	active = "active"
	lapsed = "lapsed"
	expiring_soon = "expiring_soon"
	expired = "expired"
	cancelled = "cancelled"
	endorsed = "endorsed"


class ClaimStatus(str, Enum):
	lodged = "lodged"
	under_investigation = "under_investigation"
	awaiting_assessment = "awaiting_assessment"
	approved = "approved"
	partially_approved = "partially_approved"
	rejected = "rejected"
	appealed = "appealed"
	settled = "settled"
	closed = "closed"


class ClaimType(str, Enum):
	partial_loss = "partial_loss"
	total_loss = "total_loss"
	business_interruption = "business_interruption"
	liability = "liability"
	third_party = "third_party"
	ad_hoc = "ad_hoc"


class AssetType(str, Enum):
	building = "building"
	plant_equipment = "plant_equipment"
	fit_out = "fit_out"
	fixtures = "fixtures"
	stock = "stock"
	electronic_equipment = "electronic_equipment"
	vehicles = "vehicles"
	art_valuables = "art_valuables"


class ValuationBasis(str, Enum):
	reinstatement_cost = "reinstatement_cost"
	market_value = "market_value"
	agreed_value = "agreed_value"
	indemnity_value = "indemnity_value"
	replacement_cost = "replacement_cost"


class EndorsementType(str, Enum):
	addition_of_property = "addition_of_property"
	deletion_of_property = "deletion_of_property"
	sum_insured_change = "sum_insured_change"
	premium_adjustment = "premium_adjustment"
	clause_amendment = "clause_amendment"
	extension = "extension"
	reinstatement = "reinstatement"


class DeductibleType(str, Enum):
	fixed = "fixed"
	percentage = "percentage"
	franchise = "franchise"
	excess = "excess"


class RenewalStatus(str, Enum):
	pending = "pending"
	in_negotiation = "in_negotiation"
	quoted = "quoted"
	accepted = "accepted"
	bound = "bound"
	lapsed = "lapsed"


class InsurerGrade(str, Enum):
	preferred = "preferred"
	approved = "approved"
	conditional = "conditional"
	suspended = "suspended"


# ── Insurer ───────────────────────────────────────────────────────────────────

class InsurerCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	name: str
	registration_number: str | None = None
	grade: InsurerGrade = InsurerGrade.conditional
	email: str
	phone: str | None = None
	created_by: str


class InsurerResponse(InsurerCreate):
	id: str = Field(default_factory=uuid7str)
	active_policies: int = 0
	claims_handled: int = 0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Insurance Policy ──────────────────────────────────────────────────────────

class PolicyDeductible(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	deductible_type: DeductibleType
	amount: Decimal | None = None
	percentage: Decimal | None = None
	peril: str | None = None


class PolicyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	policy_number: str
	policy_type: PolicyType
	insurer_id: str
	broker_id: str | None = None
	property_ids: list[str] = Field(default_factory=list)
	commencement_date: date
	expiry_date: date
	sum_insured: Decimal
	annual_premium: Decimal
	currency: str = "KES"
	valuation_basis: ValuationBasis
	perils_covered: list[str] = Field(default_factory=list)
	deductibles: list[PolicyDeductible] = Field(default_factory=list)
	created_by: str

	@field_validator("sum_insured", "annual_premium")
	@classmethod
	def _positive(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("must be positive")
		return v


class PolicyResponse(PolicyCreate):
	id: str = Field(default_factory=uuid7str)
	status: CoverageStatus = CoverageStatus.active
	renewal_status: RenewalStatus | None = None
	claims_count: int = 0
	total_claims_value: Decimal = Decimal("0")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class PolicyUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: CoverageStatus | None = None
	sum_insured: Decimal | None = None
	annual_premium: Decimal | None = None
	expiry_date: date | None = None


# ── Asset Schedule ────────────────────────────────────────────────────────────

class InsuredAssetCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	policy_id: str
	property_id: str
	asset_type: AssetType
	description: str
	insured_value: Decimal
	valuation_basis: ValuationBasis
	valuation_date: date | None = None
	currency: str = "KES"
	created_by: str

	@field_validator("insured_value")
	@classmethod
	def _positive_value(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("insured_value must be positive")
		return v


class InsuredAssetResponse(InsuredAssetCreate):
	id: str = Field(default_factory=uuid7str)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Claim ─────────────────────────────────────────────────────────────────────

class ClaimCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	policy_id: str
	claim_type: ClaimType
	peril: str
	incident_date: date
	description: str
	estimated_loss: Decimal
	currency: str = "KES"
	property_id: str
	evidence_ids: list[str] = Field(default_factory=list)
	created_by: str

	@field_validator("estimated_loss")
	@classmethod
	def _positive_loss(cls, v: Decimal) -> Decimal:
		if v < 0:
			raise ValueError("estimated_loss must be non-negative")
		return v


class ClaimResponse(ClaimCreate):
	id: str = Field(default_factory=uuid7str)
	claim_ref: str = ""
	status: ClaimStatus = ClaimStatus.lodged
	assessed_value: Decimal | None = None
	approved_value: Decimal | None = None
	settlement_amount: Decimal | None = None
	senior_approved: bool = False
	settled_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Endorsement ───────────────────────────────────────────────────────────────

class EndorsementCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	policy_id: str
	endorsement_type: EndorsementType
	effective_date: date
	description: str
	premium_adjustment: Decimal = Decimal("0")
	sum_insured_change: Decimal = Decimal("0")
	currency: str = "KES"
	created_by: str


class EndorsementResponse(EndorsementCreate):
	id: str = Field(default_factory=uuid7str)
	ref: str = ""
	approved_by: str | None = None
	issued_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Premium Allocation ────────────────────────────────────────────────────────

class PremiumAllocationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	policy_id: str
	allocation_method: str
	period: str  # YYYY-MM
	allocations: list[dict[str, Any]] = Field(default_factory=list)
	created_by: str


class PremiumAllocationResponse(PremiumAllocationCreate):
	id: str = Field(default_factory=uuid7str)
	total_allocated: Decimal = Decimal("0")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Coverage Gap ──────────────────────────────────────────────────────────────

class CoverageGapCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str
	gap_description: str
	severity: str  # from SUPPORTED_GAP_SEVERITIES
	estimated_exposure: Decimal | None = None
	currency: str = "KES"
	detected_by: str


class CoverageGapResponse(CoverageGapCreate):
	id: str = Field(default_factory=uuid7str)
	alert_sent: bool = False
	remediation_action: str | None = None
	resolved: bool = False
	resolved_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
