"""Pydantic v2 models for Lease Management (lea) — IFRS 16 / ASC 842.

Entities
--------
Lease · LeaseAsset · LeasePaymentSchedule · LeaseModification
RightOfUseAsset · LeaseLiability · EscalationClause · LeaseOption
LeaseAmendment · SubLease · LeaseExpiry · LeaseAbstraction
RentReview · LeaseAssignment · Ifrs16Schedule

Report models: IFRS16DisclosureNotes · PortfolioLeaseAnalytics
               PeriodJournalEntry · LeaseExpiryPipelineItem
               CpiRemeasurementResult · ExtensionOptionAssessment
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


# ── Enumerations ──────────────────────────────────────────────────────────────

class LeaseType(str, Enum):
	commercial = "commercial"
	retail = "retail"
	industrial = "industrial"
	residential = "residential"
	ground_lease = "ground_lease"
	sublease = "sublease"
	licence_to_occupy = "licence_to_occupy"
	peppercorn = "peppercorn"
	assured_shorthold = "assured_shorthold"
	regulated = "regulated"
	office = "office"


class LeaseStatus(str, Enum):
	draft = "draft"
	heads_of_terms = "heads_of_terms"
	negotiating = "negotiating"
	signed = "signed"
	active = "active"
	holding_over = "holding_over"
	notice_served = "notice_served"
	expired = "expired"
	surrendered = "surrendered"
	forfeited = "forfeited"
	assigned = "assigned"
	terminated = "terminated"
	renewed = "renewed"


class EscalationType(str, Enum):
	fixed_percentage = "fixed_percentage"
	cpi_linked = "cpi_linked"
	open_market_review = "open_market_review"
	ratchet = "ratchet"
	turnover_linked = "turnover_linked"
	base_plus_variable = "base_plus_variable"
	stepped = "stepped"


class OptionType(str, Enum):
	break_option_tenant = "break_option_tenant"
	break_option_landlord = "break_option_landlord"
	renewal_option = "renewal_option"
	purchase_option = "purchase_option"
	expansion_option = "expansion_option"
	contraction_option = "contraction_option"
	extension_option = "extension_option"


class OptionStatus(str, Enum):
	open = "open"
	exercised = "exercised"
	lapsed = "lapsed"
	waived = "waived"


class RentReviewType(str, Enum):
	upward_only = "upward_only"
	upward_downward = "upward_downward"
	fixed = "fixed"
	open_market = "open_market"
	indexed = "indexed"


class RentReviewStatus(str, Enum):
	pending = "pending"
	in_negotiation = "in_negotiation"
	agreed = "agreed"
	disputed = "disputed"
	withdrawn = "withdrawn"


class Ifrs16Category(str, Enum):
	finance_lease = "finance_lease"
	operating_lease = "operating_lease"
	short_term_exemption = "short_term_exemption"
	low_value_exemption = "low_value_exemption"
	# Alias used internally
	finance = "finance"
	operating = "operating"


class AbstractionStatus(str, Enum):
	pending = "pending"
	in_progress = "in_progress"
	complete = "complete"
	verified = "verified"
	exception = "exception"


class ModificationTrigger(str, Enum):
	scope_increase = "scope_increase"
	scope_decrease = "scope_decrease"
	term_extension = "term_extension"
	term_shortening = "term_shortening"
	payment_change = "payment_change"
	rate_change = "rate_change"
	combination = "combination"


class ModificationStatus(str, Enum):
	pending = "pending"
	approved = "approved"
	applied = "applied"
	rejected = "rejected"


class AmortisationMethod(str, Enum):
	straight_line = "straight_line"
	declining_balance = "declining_balance"
	units_of_production = "units_of_production"


class PaymentFrequency(str, Enum):
	monthly = "monthly"
	quarterly = "quarterly"
	semi_annual = "semi_annual"
	annual = "annual"
	in_advance = "in_advance"
	in_arrears = "in_arrears"


class SubleaseClassification(str, Enum):
	operating = "operating"
	finance = "finance"


class SubleaseStatus(str, Enum):
	active = "active"
	expired = "expired"
	terminated = "terminated"
	suspended = "suspended"


class ExpiryAction(str, Enum):
	renew = "renew"
	surrender = "surrender"
	holdover = "holdover"
	negotiate = "negotiate"
	vacate = "vacate"


class AssignmentStatus(str, Enum):
	pending = "pending"
	completed = "completed"
	rejected = "rejected"
	withdrawn = "withdrawn"


class AccountingStandard(str, Enum):
	ifrs16 = "ifrs16"
	asc842 = "asc842"


class PaymentStatus(str, Enum):
	scheduled = "scheduled"
	paid = "paid"
	overdue = "overdue"
	partially_paid = "partially_paid"
	waived = "waived"
	disputed = "disputed"


# ── Base Model ────────────────────────────────────────────────────────────────

class LeaBase(BaseModel):
	"""Common audit fields for all LEA entities."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	is_deleted: bool = False


# ── Lease (master entity) ─────────────────────────────────────────────────────

class LeaseCreate(BaseModel):
	"""Input model for creating a new lease."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str
	unit_id: str
	tenant_entity_id: str
	lease_type: LeaseType
	lease_ref: str
	commencement_date: date
	expiry_date: date
	initial_rent: Decimal
	rent_frequency: PaymentFrequency = PaymentFrequency.monthly
	area: Decimal | None = None
	area_unit: str = "sqm"
	currency: str = "KES"
	security_deposit: Decimal = Decimal("0")
	accounting_standard: AccountingStandard = AccountingStandard.ifrs16
	# IFRS 16 fields
	incremental_borrowing_rate: Decimal | None = None   # IBR % p.a. (e.g. 8.5)
	implicit_rate: Decimal | None = None
	initial_direct_costs: Decimal = Decimal("0")
	lease_incentives: Decimal = Decimal("0")            # rent-free periods, fit-out contributions
	dismantling_costs: Decimal = Decimal("0")           # make-good / ARO
	residual_value_guarantee: Decimal = Decimal("0")
	variable_payment_indexed_to_cpi: bool = False
	cpi_base_index: Decimal | None = None
	is_sublease: bool = False
	parent_lease_id: str | None = None
	created_by: str
	notes: str | None = None

	@model_validator(mode="after")
	def _expiry_after_commencement(self) -> "LeaseCreate":
		if self.expiry_date <= self.commencement_date:
			raise ValueError("expiry_date must be after commencement_date")
		return self

	@field_validator("initial_rent")
	@classmethod
	def _positive_rent(cls, v: Decimal) -> Decimal:
		if v < 0:
			raise ValueError("initial_rent must be non-negative")
		return v

	@field_validator("security_deposit")
	@classmethod
	def _non_negative_deposit(cls, v: Decimal) -> Decimal:
		if v < 0:
			raise ValueError("security_deposit must be non-negative")
		return v


class LeaseUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: LeaseStatus | None = None
	current_rent: Decimal | None = None
	expiry_date: date | None = None
	ifrs16_category: Ifrs16Category | None = None
	incremental_borrowing_rate: Decimal | None = None
	notes: str | None = None


class LeaseResponse(LeaseCreate):
	id: str = Field(default_factory=uuid7str)
	status: LeaseStatus = LeaseStatus.heads_of_terms
	current_rent: Decimal = Decimal("0")
	abstraction_status: AbstractionStatus = AbstractionStatus.pending
	abstraction_verified: bool = False
	ifrs16_category: Ifrs16Category | None = None
	rou_asset: Decimal | None = None
	lease_liability: Decimal | None = None
	total_payments_made: Decimal = Decimal("0")
	days_to_expiry: int | None = None
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Lease Asset ───────────────────────────────────────────────────────────────

class LeaseAssetCreate(BaseModel):
	"""Underlying physical asset being leased (IFRS 16 — underlying asset)."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	asset_description: str
	asset_class: str                    # e.g. "building", "vehicle", "equipment"
	fair_value_when_new: Decimal | None = None
	useful_economic_life_months: int | None = None
	location: str | None = None
	asset_ref: str | None = None
	currency: str = "KES"
	created_by: str


class LeaseAssetUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	asset_description: str | None = None
	fair_value_when_new: Decimal | None = None
	useful_economic_life_months: int | None = None
	location: str | None = None


class LeaseAssetResponse(LeaseAssetCreate):
	id: str = Field(default_factory=uuid7str)
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Right-of-Use Asset ────────────────────────────────────────────────────────

class RightOfUseAssetCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	initial_measurement: Decimal
	useful_life_months: int
	amortisation_method: AmortisationMethod = AmortisationMethod.straight_line
	currency: str = "KES"
	created_by: str

	@field_validator("initial_measurement")
	@classmethod
	def _positive(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("initial_measurement must be positive")
		return v


class RightOfUseAssetUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	useful_life_months: int | None = None
	impairment_loss: Decimal | None = None


class RightOfUseAssetResponse(RightOfUseAssetCreate):
	id: str = Field(default_factory=uuid7str)
	accumulated_depreciation: Decimal = Decimal("0")
	impairment_loss: Decimal = Decimal("0")
	carrying_amount: Decimal = Decimal("0")
	periods_amortised: int = 0
	fully_amortised: bool = False
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Lease Liability ───────────────────────────────────────────────────────────

class LeaseLiabilityCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	opening_balance: Decimal
	interest_rate: Decimal     # % p.a.
	currency: str = "KES"
	created_by: str

	@field_validator("opening_balance")
	@classmethod
	def _positive(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("opening_balance must be positive")
		return v


class LeaseLiabilityUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	interest_rate: Decimal | None = None
	current_balance: Decimal | None = None


class LeaseLiabilityResponse(LeaseLiabilityCreate):
	id: str = Field(default_factory=uuid7str)
	current_balance: Decimal = Decimal("0")
	cumulative_interest: Decimal = Decimal("0")
	cumulative_principal: Decimal = Decimal("0")
	current_portion: Decimal = Decimal("0")
	non_current_portion: Decimal = Decimal("0")
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Lease Payment Schedule ────────────────────────────────────────────────────

class LeasePaymentScheduleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	period_number: int
	due_date: date
	opening_balance: Decimal
	payment_amount: Decimal
	interest_portion: Decimal
	principal_portion: Decimal
	closing_balance: Decimal
	cumulative_interest: Decimal
	is_variable: bool = False
	variable_index: str | None = None
	escalation_applied: Decimal = Decimal("0")
	currency: str = "KES"
	created_by: str


class LeasePaymentScheduleUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	paid: bool | None = None
	paid_date: date | None = None
	paid_amount: Decimal | None = None
	payment_status: PaymentStatus | None = None


class LeasePaymentScheduleResponse(LeasePaymentScheduleCreate):
	id: str = Field(default_factory=uuid7str)
	paid: bool = False
	paid_date: date | None = None
	paid_amount: Decimal | None = None
	variance: Decimal = Decimal("0")
	payment_status: PaymentStatus = PaymentStatus.scheduled
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Escalation Clause ─────────────────────────────────────────────────────────

class EscalationClauseCreate(BaseModel):
	"""Persistent escalation clause attached to a lease."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	escalation_type: EscalationType
	fixed_rate: Decimal | None = None
	cpi_base_index: Decimal | None = None
	review_frequency_months: int = 12
	first_review_date: date | None = None
	cap_rate: Decimal | None = None
	floor_rate: Decimal | None = None
	created_by: str

	@model_validator(mode="after")
	def _validate_escalation_fields(self) -> "EscalationClauseCreate":
		if self.escalation_type == EscalationType.fixed_percentage and self.fixed_rate is None:
			raise ValueError("fixed_rate required for fixed_percentage escalation")
		if self.escalation_type == EscalationType.cpi_linked and self.cpi_base_index is None:
			raise ValueError("cpi_base_index required for CPI-linked escalation")
		return self


class EscalationClauseUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	fixed_rate: Decimal | None = None
	cap_rate: Decimal | None = None
	floor_rate: Decimal | None = None
	review_frequency_months: int | None = None


class EscalationClauseResponse(EscalationClauseCreate):
	id: str = Field(default_factory=uuid7str)
	last_applied_date: date | None = None
	last_applied_index: Decimal | None = None
	applied_count: int = 0
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Rent Escalation (event) ───────────────────────────────────────────────────

class RentEscalationCreate(BaseModel):
	"""A single scheduled escalation event derived from an EscalationClause."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	escalation_type: EscalationType
	effective_date: date
	escalation_rate: Decimal | None = None
	new_rent: Decimal | None = None
	index_source: str | None = None
	cpi_current_index: Decimal | None = None
	cpi_base_index: Decimal | None = None
	notes: str | None = None
	created_by: str


class RentEscalationResponse(RentEscalationCreate):
	id: str = Field(default_factory=uuid7str)
	old_rent: Decimal = Decimal("0")
	computed_new_rent: Decimal | None = None
	applied: bool = False
	applied_at: datetime | None = None
	applied_by: str | None = None
	remeasurement_required: bool = False
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Lease Option ──────────────────────────────────────────────────────────────

class LeaseOptionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	option_type: OptionType
	exercise_from: date
	exercise_to: date
	effective_date: date
	notice_required_days: int = 0
	new_expiry: date | None = None
	extension_months: int | None = None
	purchase_price: Decimal | None = None
	reasonably_certain: bool = False
	economic_incentive: bool = False
	notes: str | None = None
	created_by: str


class LeaseOptionUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	reasonably_certain: bool | None = None
	economic_incentive: bool | None = None
	notes: str | None = None


class LeaseOptionResponse(LeaseOptionCreate):
	id: str = Field(default_factory=uuid7str)
	status: OptionStatus = OptionStatus.open
	exercised_at: datetime | None = None
	notice_served_at: datetime | None = None
	last_assessed_date: date | None = None
	assessment_changed: bool = False
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Lease Modification ────────────────────────────────────────────────────────

class LeaseModificationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	modification_date: date
	trigger: ModificationTrigger
	reason: str
	new_lease_term_months: int | None = None
	new_base_payment: Decimal | None = None
	new_rate: Decimal | None = None
	surrendered_proportion: Decimal | None = None
	new_commencement_date: date | None = None
	creates_new_lease: bool = False
	approved_by: str | None = None
	created_by: str

	@model_validator(mode="after")
	def _validate_partial_surrender(self) -> "LeaseModificationCreate":
		if self.trigger == ModificationTrigger.scope_decrease and self.surrendered_proportion is not None:
			if not (Decimal("0") < self.surrendered_proportion < Decimal("1")):
				raise ValueError("surrendered_proportion must be strictly between 0 and 1")
		return self


class LeaseModificationUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: ModificationStatus | None = None
	approved_by: str | None = None
	reason: str | None = None


class LeaseModificationResponse(LeaseModificationCreate):
	id: str = Field(default_factory=uuid7str)
	status: ModificationStatus = ModificationStatus.pending
	remeasured_liability: Decimal | None = None
	remeasured_rou: Decimal | None = None
	gain_loss_on_modification: Decimal = Decimal("0")
	applied: bool = False
	applied_at: datetime | None = None
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Lease Amendment ───────────────────────────────────────────────────────────

class LeaseAmendmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	amendment_date: date
	description: str
	amended_clauses: list[str] = Field(default_factory=list)
	document_ids: list[str] = Field(default_factory=list)
	created_by: str


class LeaseAmendmentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	description: str | None = None
	amended_clauses: list[str] | None = None
	document_ids: list[str] | None = None


class LeaseAmendmentResponse(LeaseAmendmentCreate):
	id: str = Field(default_factory=uuid7str)
	approved_by: str | None = None
	approved_at: datetime | None = None
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── SubLease ──────────────────────────────────────────────────────────────────

class SubleaseCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	head_lease_id: str
	sublessee_entity_id: str
	commencement_date: date
	end_date: date
	payment_amount: Decimal
	payment_frequency: PaymentFrequency = PaymentFrequency.monthly
	sublease_classification: SubleaseClassification = SubleaseClassification.operating
	portion_sqm: Decimal | None = None
	implicit_rate: Decimal | None = None
	currency: str = "KES"
	created_by: str

	@field_validator("payment_amount")
	@classmethod
	def _positive(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("payment_amount must be positive")
		return v


class SubleaseUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: SubleaseStatus | None = None
	payment_amount: Decimal | None = None
	end_date: date | None = None


class SubleaseResponse(SubleaseCreate):
	id: str = Field(default_factory=uuid7str)
	status: SubleaseStatus = SubleaseStatus.active
	total_sublease_income: Decimal = Decimal("0")
	net_investment_in_sublease: Decimal | None = None
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Lease Expiry ──────────────────────────────────────────────────────────────

class LeaseExpiryCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	expiry_date: date
	action_required: ExpiryAction
	assigned_to: str | None = None
	days_ahead_flag: int = 180
	notes: str | None = None
	created_by: str


class LeaseExpiryUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	action_required: ExpiryAction | None = None
	assigned_to: str | None = None
	notes: str | None = None
	resolved: bool | None = None


class LeaseExpiryResponse(LeaseExpiryCreate):
	id: str = Field(default_factory=uuid7str)
	days_to_expiry: int = 0
	action_taken: str | None = None
	resolved: bool = False
	resolved_at: datetime | None = None
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Rent Review ───────────────────────────────────────────────────────────────

class RentReviewCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	review_type: RentReviewType
	review_date: date
	proposed_rent: Decimal | None = None
	created_by: str


class RentReviewUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: RentReviewStatus | None = None
	proposed_rent: Decimal | None = None


class RentReviewResponse(RentReviewCreate):
	id: str = Field(default_factory=uuid7str)
	status: RentReviewStatus = RentReviewStatus.pending
	agreed_rent: Decimal | None = None
	agreed_at: datetime | None = None
	backdating_authorised_by: str | None = None
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Lease Abstraction ─────────────────────────────────────────────────────────

class LeaseAbstractionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	source_document_id: str
	abstracted_by: str
	notes: str | None = None


class LeaseAbstractionUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: AbstractionStatus | None = None
	extracted_fields: dict[str, Any] | None = None
	exceptions: list[str] | None = None


class LeaseAbstractionResponse(LeaseAbstractionCreate):
	id: str = Field(default_factory=uuid7str)
	status: AbstractionStatus = AbstractionStatus.pending
	extracted_fields: dict[str, Any] = Field(default_factory=dict)
	exceptions: list[str] = Field(default_factory=list)
	verified_by: str | None = None
	verified_at: datetime | None = None
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── IFRS 16 Schedule ──────────────────────────────────────────────────────────

class Ifrs16ScheduleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	category: Ifrs16Category
	commencement_date: date
	expiry_date: date
	annual_payment: Decimal
	discount_rate: Decimal             # decimal fraction, e.g. 0.065 for 6.5 %

	@field_validator("discount_rate")
	@classmethod
	def _rate_range(cls, v: Decimal) -> Decimal:
		if not (Decimal("0") < v < Decimal("1")):
			raise ValueError("discount_rate must be between 0 and 1 (e.g. 0.065 for 6.5%)")
		return v


class Ifrs16ScheduleResponse(Ifrs16ScheduleCreate):
	id: str = Field(default_factory=uuid7str)
	rou_asset: Decimal = Decimal("0")
	lease_liability: Decimal = Decimal("0")
	amortisation_schedule: list[dict[str, Any]] = Field(default_factory=list)
	auditor_approved: bool = False
	auditor_approved_by: str | None = None
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Lease Assignment ──────────────────────────────────────────────────────────

class LeaseAssignmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	assignment_type: str
	assignee_id: str
	effective_date: date
	landlord_consent_ref: str | None = None
	created_by: str


class LeaseAssignmentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: AssignmentStatus | None = None
	landlord_consent_ref: str | None = None


class LeaseAssignmentResponse(LeaseAssignmentCreate):
	id: str = Field(default_factory=uuid7str)
	status: AssignmentStatus = AssignmentStatus.pending
	completed_at: datetime | None = None
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Lease Payment (actual payment recording) ──────────────────────────────────

class LeasePaymentCreate(BaseModel):
	"""Record an actual lease payment against a schedule line."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	lease_id: str
	schedule_id: str
	payment_date: date
	amount_paid: Decimal
	payment_reference: str | None = None
	notes: str | None = None
	created_by: str

	@field_validator("amount_paid")
	@classmethod
	def _positive(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("amount_paid must be positive")
		return v


class LeasePaymentResponse(LeasePaymentCreate):
	id: str = Field(default_factory=uuid7str)
	variance: Decimal = Decimal("0")
	overpayment: bool = False
	underpayment: bool = False
	is_deleted: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Report / Disclosure Models ────────────────────────────────────────────────

class IFRS16DisclosureNotes(BaseModel):
	"""IFRS 16.53–58 disclosure notes for a reporting period."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	period_end: date
	accounting_standard: AccountingStandard = AccountingStandard.ifrs16
	total_rou_assets: Decimal
	total_lease_liabilities: Decimal
	current_lease_liabilities: Decimal
	non_current_lease_liabilities: Decimal
	depreciation_charge: Decimal
	interest_expense: Decimal
	short_term_lease_expense: Decimal
	low_value_lease_expense: Decimal
	variable_lease_expense: Decimal
	total_cash_outflow: Decimal
	maturity_analysis: dict[str, Decimal]
	weighted_average_ibr: Decimal
	lease_count: int
	generated_at: datetime = Field(default_factory=datetime.utcnow)


class PortfolioLeaseAnalytics(BaseModel):
	"""Portfolio-level summary analytics."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	as_at: date
	total_leases: int
	active_leases: int
	expiring_within_90_days: int
	expiring_within_180_days: int
	total_rou_assets: Decimal
	total_lease_liabilities: Decimal
	annual_lease_cost: Decimal
	weighted_average_remaining_term_months: Decimal
	leases_by_type: dict[str, int]
	leases_by_status: dict[str, int]
	top_leases_by_liability: list[dict[str, Any]]
	subleases_active: int
	sublease_income_annual: Decimal
	exemptions_short_term: int
	exemptions_low_value: int
	modifications_ytd: int
	total_security_deposits: Decimal
	generated_at: datetime = Field(default_factory=datetime.utcnow)


class PeriodJournalEntry(BaseModel):
	"""Single accounting journal entry generated for a lease period."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	lease_id: str
	period: str          # YYYY-MM
	debit_account: str
	credit_account: str
	amount: Decimal
	description: str
	currency: str = "KES"


class LeaseExpiryPipelineItem(BaseModel):
	"""Single item in the expiry pipeline report."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	lease_id: str
	lease_ref: str | None
	tenant_entity_id: str | None
	property_id: str | None
	unit_id: str | None
	expiry_date: str
	days_remaining: int
	current_rent: str | None
	status: str | None
	action_required: str | None
	options_open: int = 0


class CpiRemeasurementResult(BaseModel):
	"""Result of a CPI-triggered lease liability remeasurement."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	lease_id: str
	old_liability: Decimal
	new_liability: Decimal
	old_rou: Decimal
	new_rou: Decimal
	adjustment: Decimal
	new_payment: Decimal
	current_cpi: Decimal
	base_cpi: Decimal
	remeasured_at: datetime = Field(default_factory=datetime.utcnow)


class ExtensionOptionAssessment(BaseModel):
	"""Result of assessing a lease extension option under IFRS 16.19."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	lease_id: str
	option_id: str
	option_type: str
	reasonably_certain: bool
	economic_incentive: bool
	significant_leasehold_improvements: bool = False
	importance_to_operations: bool = False
	cost_of_relocation: bool = False
	prior_assessment_changed: bool = False
	remeasurement_triggered: bool = False
	assessed_at: datetime = Field(default_factory=datetime.utcnow)
	assessed_by: str
	notes: str | None = None


class LeaseModificationRequest(BaseModel):
	"""Input for handling a lease modification with full remeasurement."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	reason: str
	trigger: ModificationTrigger
	modification_date: date
	new_lease_term_months: int | None = None
	new_base_payment: Decimal | None = None
	new_rate: Decimal | None = None
	surrendered_proportion: Decimal | None = None
	creates_new_lease: bool = False
	approved_by: str | None = None


class SubleaseCreate2(BaseModel):
	"""Sublease management input (simplified for service-level use)."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	head_lease_id: str
	sublessee_entity_id: str
	commencement_date: date
	end_date: date
	monthly_payment: Decimal
	sublease_classification: SubleaseClassification = SubleaseClassification.operating
	implicit_rate: Decimal | None = None
	currency: str = "KES"
