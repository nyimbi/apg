"""Pydantic v2 models for APG Tax Administration.

All entities use UUID7 IDs, tenant isolation, soft-delete, and full audit columns.
Status/type enums cover every valid lifecycle state.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


def _non_empty(v: str) -> str:
	assert v and v.strip(), "must be non-empty"
	return v.strip()


NonEmpty = Annotated[str, AfterValidator(_non_empty)]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TaxType(str, Enum):
	INCOME_TAX = "income_tax"
	VAT = "vat"
	CORPORATE_TAX = "corporate_tax"
	WITHHOLDING_TAX = "withholding_tax"
	CAPITAL_GAINS_TAX = "capital_gains_tax"
	EXCISE_DUTY = "excise_duty"
	CUSTOMS_DUTY = "customs_duty"
	STAMP_DUTY = "stamp_duty"
	RENTAL_INCOME_TAX = "rental_income_tax"
	TURNOVER_TAX = "turnover_tax"
	DIGITAL_SERVICES_TAX = "digital_services_tax"
	PRESUMPTIVE_TAX = "presumptive_tax"


class TaxpayerType(str, Enum):
	INDIVIDUAL = "individual"
	COMPANY = "company"
	PARTNERSHIP = "partnership"
	TRUST = "trust"
	GOVERNMENT_ENTITY = "government_entity"
	NGO = "ngo"
	FOREIGN_ENTITY = "foreign_entity"


class TaxpayerStatus(str, Enum):
	PENDING = "pending"
	ACTIVE = "active"
	SUSPENDED = "suspended"
	DEREGISTERED = "deregistered"
	UNDER_INVESTIGATION = "under_investigation"
	BLOCKED = "blocked"


class ReturnType(str, Enum):
	MONTHLY_VAT = "monthly_vat"
	ANNUAL_INCOME = "annual_income"
	QUARTERLY_ADVANCE = "quarterly_advance"
	WITHHOLDING_TAX_RETURN = "withholding_tax_return"
	CORPORATE_ANNUAL = "corporate_annual"
	CUSTOMS_ENTRY = "customs_entry"
	TURNOVER_TAX_MONTHLY = "turnover_tax_monthly"
	CAPITAL_GAINS = "capital_gains"


class ReturnStatus(str, Enum):
	DRAFT = "draft"
	FILED = "filed"
	AMENDED = "amended"
	UNDER_REVIEW = "under_review"
	ASSESSED = "assessed"
	DISPUTED = "disputed"
	FINALISED = "finalised"
	REJECTED = "rejected"


class AssessmentType(str, Enum):
	SELF_ASSESSMENT = "self_assessment"
	AMENDED_ASSESSMENT = "amended_assessment"
	BEST_JUDGEMENT = "best_judgement"
	AUDIT_ASSESSMENT = "audit_assessment"
	ESTIMATED_ASSESSMENT = "estimated_assessment"
	AGENCY_ASSESSMENT = "agency_assessment"


class AssessmentStatus(str, Enum):
	DRAFT = "draft"
	ISSUED = "issued"
	OBJECTED = "objected"
	UPHELD = "upheld"
	REDUCED = "reduced"
	WITHDRAWN = "withdrawn"
	FINALISED = "finalised"
	APPEALED = "appealed"


class PaymentStatus(str, Enum):
	PENDING = "pending"
	PROCESSING = "processing"
	CONFIRMED = "confirmed"
	FAILED = "failed"
	REVERSED = "reversed"
	PARTIALLY_APPLIED = "partially_applied"
	FULLY_APPLIED = "fully_applied"


class PaymentMethod(str, Enum):
	BANK_TRANSFER = "bank_transfer"
	MOBILE_MONEY = "mobile_money"
	CHEQUE = "cheque"
	CASH = "cash"
	CREDIT_CARD = "credit_card"
	DIRECT_DEBIT = "direct_debit"
	RTGS = "rtgs"
	PAYMENT_PLAN = "payment_plan"


class DebtStatus(str, Enum):
	OUTSTANDING = "outstanding"
	PARTIALLY_PAID = "partially_paid"
	PAID = "paid"
	WRITTEN_OFF = "written_off"
	UNDER_ARRANGEMENT = "under_arrangement"
	IN_LITIGATION = "in_litigation"
	DISPUTED = "disputed"


class CollectionMethod(str, Enum):
	PAYMENT_PLAN = "payment_plan"
	GARNISHMENT = "garnishment"
	ASSET_SEIZURE = "asset_seizure"
	THIRD_PARTY_DEMAND = "third_party_demand"
	LEGAL_PROCEEDINGS = "legal_proceedings"
	WRITE_OFF = "write_off"
	SALARY_ATTACHMENT = "salary_attachment"
	BANK_LEVY = "bank_levy"


class AuditType(str, Enum):
	DESK_AUDIT = "desk_audit"
	FIELD_AUDIT = "field_audit"
	IT_AUDIT = "it_audit"
	TRANSFER_PRICING = "transfer_pricing"
	VAT_REFUND_AUDIT = "vat_refund_audit"
	FORENSIC_AUDIT = "forensic_audit"
	COMPLIANCE_AUDIT = "compliance_audit"
	SECTOR_AUDIT = "sector_audit"


class AuditStatus(str, Enum):
	PLANNED = "planned"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	REPORT_ISSUED = "report_issued"
	OBJECTION_FILED = "objection_filed"
	FINALISED = "finalised"
	WITHDRAWN = "withdrawn"


class FindingType(str, Enum):
	UNDERPAYMENT = "underpayment"
	OVERPAYMENT = "overpayment"
	NON_COMPLIANCE = "non_compliance"
	EVASION = "evasion"
	AVOIDANCE = "avoidance"
	FRAUD = "fraud"
	PROCEDURAL = "procedural"
	INFORMATIONAL = "informational"


class ObjectionStatus(str, Enum):
	SUBMITTED = "submitted"
	UNDER_REVIEW = "under_review"
	UPHELD = "upheld"
	PARTIALLY_UPHELD = "partially_upheld"
	DISMISSED = "dismissed"
	APPEALED = "appealed"
	WITHDRAWN = "withdrawn"


class AppealStatus(str, Enum):
	SUBMITTED = "submitted"
	REGISTERED = "registered"
	HEARING_SCHEDULED = "hearing_scheduled"
	HEARD = "heard"
	DECIDED = "decided"
	FURTHER_APPEALED = "further_appealed"
	WITHDRAWN = "withdrawn"
	CLOSED = "closed"


class RefundStatus(str, Enum):
	CLAIMED = "claimed"
	UNDER_REVIEW = "under_review"
	APPROVED = "approved"
	REJECTED = "rejected"
	PROCESSING = "processing"
	PAID = "paid"
	OFFSET = "offset"
	WITHHELD = "withheld"


class PenaltyType(str, Enum):
	LATE_FILING = "late_filing"
	LATE_PAYMENT = "late_payment"
	UNDERSTATEMENT = "understatement"
	FRAUD = "fraud"
	NON_FILING = "non_filing"
	INCORRECT_RETURN = "incorrect_return"
	WITHHOLDING_DEFAULT = "withholding_default"


class PenaltyStatus(str, Enum):
	ASSESSED = "assessed"
	CONFIRMED = "confirmed"
	REDUCED = "reduced"
	WAIVED = "waived"
	PAID = "paid"
	OUTSTANDING = "outstanding"
	DISPUTED = "disputed"


class InterestType(str, Enum):
	LATE_PAYMENT = "late_payment"
	LATE_FILING = "late_filing"
	REFUND_INTEREST = "refund_interest"
	PENALTY_INTEREST = "penalty_interest"


class ClearanceCertificateStatus(str, Enum):
	APPLIED = "applied"
	UNDER_REVIEW = "under_review"
	ISSUED = "issued"
	REJECTED = "rejected"
	EXPIRED = "expired"
	REVOKED = "revoked"


class ObligationStatus(str, Enum):
	ACTIVE = "active"
	DORMANT = "dormant"
	CANCELLED = "cancelled"
	FULFILLED = "fulfilled"


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class TaxBase(BaseModel):
	"""All tax entities share these audit fields."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmpty
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"
	is_deleted: bool = False
	metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Taxpayer
# ---------------------------------------------------------------------------

class TaxpayerCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	taxpayer_type: TaxpayerType
	tax_pin: NonEmpty
	national_id: str | None = None
	business_registration_number: str | None = None
	taxpayer_name: NonEmpty
	trade_name: str | None = None
	email: str | None = None
	phone: str | None = None
	physical_address: str | None = None
	postal_address: str | None = None
	tax_types: list[TaxType] = Field(default_factory=list)
	sector_code: str | None = None
	country_of_incorporation: str = "KE"
	is_resident: bool = True
	evidence_reference: NonEmpty
	created_by: str = "system"


class TaxpayerUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	taxpayer_name: str | None = None
	trade_name: str | None = None
	email: str | None = None
	phone: str | None = None
	physical_address: str | None = None
	postal_address: str | None = None
	tax_types: list[TaxType] | None = None
	sector_code: str | None = None
	status: TaxpayerStatus | None = None
	evidence_reference: str | None = None
	updated_by: str = "system"


class TaxpayerResponse(TaxBase):
	taxpayer_type: TaxpayerType
	tax_pin: str
	national_id: str | None = None
	business_registration_number: str | None = None
	taxpayer_name: str
	trade_name: str | None = None
	email: str | None = None
	phone: str | None = None
	physical_address: str | None = None
	postal_address: str | None = None
	tax_types: list[TaxType]
	sector_code: str | None = None
	country_of_incorporation: str
	is_resident: bool
	status: TaxpayerStatus
	evidence_reference: str
	compliance_score: Decimal | None = None
	risk_rating: str | None = None


# ---------------------------------------------------------------------------
# TaxObligation
# ---------------------------------------------------------------------------

class TaxObligationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	taxpayer_id: NonEmpty
	tax_type: TaxType
	filing_frequency: str  # monthly, quarterly, annually
	due_day: int = 20
	effective_from: date
	effective_to: date | None = None
	created_by: str = "system"


class TaxObligationResponse(TaxBase):
	taxpayer_id: str
	tax_type: TaxType
	filing_frequency: str
	due_day: int
	effective_from: date
	effective_to: date | None = None
	status: ObligationStatus = ObligationStatus.ACTIVE


# ---------------------------------------------------------------------------
# TaxReturn
# ---------------------------------------------------------------------------

class TaxReturnCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	taxpayer_id: NonEmpty
	tax_pin: NonEmpty
	return_type: ReturnType
	tax_period_start: date
	tax_period_end: date
	gross_income: Decimal = Decimal("0")
	allowable_deductions: Decimal = Decimal("0")
	taxable_income: Decimal = Decimal("0")
	tax_liability: Decimal = Decimal("0")
	tax_credits: Decimal = Decimal("0")
	tax_paid: Decimal = Decimal("0")
	net_tax_payable: Decimal = Decimal("0")
	filing_date: datetime | None = None
	evidence_reference: NonEmpty
	is_amended: bool = False
	original_return_id: str | None = None
	created_by: str = "system"


class TaxReturnUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	gross_income: Decimal | None = None
	allowable_deductions: Decimal | None = None
	taxable_income: Decimal | None = None
	tax_liability: Decimal | None = None
	tax_credits: Decimal | None = None
	tax_paid: Decimal | None = None
	net_tax_payable: Decimal | None = None
	status: ReturnStatus | None = None
	evidence_reference: str | None = None
	updated_by: str = "system"


class TaxReturnResponse(TaxBase):
	taxpayer_id: str
	tax_pin: str
	return_type: ReturnType
	tax_period_start: date
	tax_period_end: date
	gross_income: Decimal
	allowable_deductions: Decimal
	taxable_income: Decimal
	tax_liability: Decimal
	tax_credits: Decimal
	tax_paid: Decimal
	net_tax_payable: Decimal
	filing_date: datetime | None = None
	status: ReturnStatus
	evidence_reference: str
	is_amended: bool
	original_return_id: str | None = None
	late_filing_days: int = 0


# ---------------------------------------------------------------------------
# TaxAssessment
# ---------------------------------------------------------------------------

class TaxAssessmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	return_id: NonEmpty
	taxpayer_id: NonEmpty
	assessment_type: AssessmentType
	assessed_amount: Decimal
	tax_liability_per_return: Decimal = Decimal("0")
	additional_tax: Decimal = Decimal("0")
	assessor_id: NonEmpty
	assessment_date: date
	due_date: date | None = None
	evidence_reference: NonEmpty
	notes: str | None = None
	created_by: str = "system"


class TaxAssessmentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	assessed_amount: Decimal | None = None
	additional_tax: Decimal | None = None
	status: AssessmentStatus | None = None
	due_date: date | None = None
	notes: str | None = None
	evidence_reference: str | None = None
	updated_by: str = "system"


class TaxAssessmentResponse(TaxBase):
	return_id: str
	taxpayer_id: str
	assessment_type: AssessmentType
	assessed_amount: Decimal
	tax_liability_per_return: Decimal
	additional_tax: Decimal
	assessor_id: str
	assessment_date: date
	due_date: date | None = None
	evidence_reference: str
	notes: str | None = None
	status: AssessmentStatus
	penalty_ids: list[str] = Field(default_factory=list)
	interest_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# TaxPayment
# ---------------------------------------------------------------------------

class TaxPaymentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	taxpayer_id: NonEmpty
	assessment_id: str | None = None
	return_id: str | None = None
	payment_reference: NonEmpty
	payment_method: PaymentMethod
	amount: Decimal
	payment_date: date
	bank_reference: str | None = None
	evidence_reference: NonEmpty
	created_by: str = "system"


class TaxPaymentResponse(TaxBase):
	taxpayer_id: str
	assessment_id: str | None = None
	return_id: str | None = None
	payment_reference: str
	payment_method: PaymentMethod
	amount: Decimal
	payment_date: date
	bank_reference: str | None = None
	evidence_reference: str
	status: PaymentStatus
	applied_to: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# TaxDebt
# ---------------------------------------------------------------------------

class TaxDebtCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	taxpayer_id: NonEmpty
	assessment_id: NonEmpty
	principal_amount: Decimal
	penalty_amount: Decimal = Decimal("0")
	interest_amount: Decimal = Decimal("0")
	due_date: date
	created_by: str = "system"


class TaxDebtResponse(TaxBase):
	taxpayer_id: str
	assessment_id: str
	principal_amount: Decimal
	penalty_amount: Decimal
	interest_amount: Decimal
	total_amount: Decimal
	amount_paid: Decimal = Decimal("0")
	balance: Decimal
	due_date: date
	status: DebtStatus
	collection_case_id: str | None = None
	demand_notices_issued: int = 0


# ---------------------------------------------------------------------------
# TaxAudit
# ---------------------------------------------------------------------------

class TaxAuditCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	taxpayer_id: NonEmpty
	tax_pin: NonEmpty
	audit_type: AuditType
	auditor_id: NonEmpty
	audit_team: list[str] = Field(default_factory=list)
	tax_period_start: date
	tax_period_end: date
	scope_description: str | None = None
	risk_score: Decimal | None = None
	evidence_reference: NonEmpty
	created_by: str = "system"


class TaxAuditUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: AuditStatus | None = None
	auditor_id: str | None = None
	audit_team: list[str] | None = None
	scope_description: str | None = None
	evidence_reference: str | None = None
	updated_by: str = "system"


class TaxAuditResponse(TaxBase):
	taxpayer_id: str
	tax_pin: str
	audit_type: AuditType
	auditor_id: str
	audit_team: list[str]
	tax_period_start: date
	tax_period_end: date
	scope_description: str | None = None
	risk_score: Decimal | None = None
	evidence_reference: str
	status: AuditStatus
	finding_ids: list[str] = Field(default_factory=list)
	total_additional_tax: Decimal = Decimal("0")


# ---------------------------------------------------------------------------
# AuditFinding
# ---------------------------------------------------------------------------

class AuditFindingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	audit_id: NonEmpty
	taxpayer_id: NonEmpty
	finding_type: FindingType
	description: NonEmpty
	additional_tax: Decimal = Decimal("0")
	penalty_amount: Decimal = Decimal("0")
	interest_amount: Decimal = Decimal("0")
	period_affected: str | None = None
	evidence_reference: NonEmpty
	created_by: str = "system"


class AuditFindingResponse(TaxBase):
	audit_id: str
	taxpayer_id: str
	finding_type: FindingType
	description: str
	additional_tax: Decimal
	penalty_amount: Decimal
	interest_amount: Decimal
	total_amount: Decimal
	period_affected: str | None = None
	evidence_reference: str
	is_accepted: bool = False
	response_received: str | None = None


# ---------------------------------------------------------------------------
# Objection
# ---------------------------------------------------------------------------

class ObjectionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	assessment_id: NonEmpty
	taxpayer_id: NonEmpty
	tax_pin: NonEmpty
	grounds: NonEmpty
	amount_disputed: Decimal
	supporting_documents: list[str] = Field(default_factory=list)
	evidence_reference: NonEmpty
	filed_date: date | None = None
	created_by: str = "system"


class ObjectionUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: ObjectionStatus | None = None
	determination_notes: str | None = None
	amount_upheld: Decimal | None = None
	reviewing_officer_id: str | None = None
	updated_by: str = "system"


class ObjectionResponse(TaxBase):
	assessment_id: str
	taxpayer_id: str
	tax_pin: str
	grounds: str
	amount_disputed: Decimal
	amount_upheld: Decimal | None = None
	supporting_documents: list[str]
	evidence_reference: str
	filed_date: date
	determination_date: date | None = None
	determination_notes: str | None = None
	reviewing_officer_id: str | None = None
	status: ObjectionStatus
	days_to_determination: int | None = None


# ---------------------------------------------------------------------------
# Appeal
# ---------------------------------------------------------------------------

class AppealCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	objection_id: NonEmpty
	taxpayer_id: NonEmpty
	grounds: NonEmpty
	amount_in_dispute: Decimal
	tribunal: str = "Tax Appeals Tribunal"
	evidence_reference: NonEmpty
	created_by: str = "system"


class AppealResponse(TaxBase):
	objection_id: str
	taxpayer_id: str
	grounds: str
	amount_in_dispute: Decimal
	tribunal: str
	evidence_reference: str
	status: AppealStatus
	hearing_date: date | None = None
	decision_date: date | None = None
	decision_notes: str | None = None


# ---------------------------------------------------------------------------
# TaxRefund
# ---------------------------------------------------------------------------

class TaxRefundCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	taxpayer_id: NonEmpty
	tax_pin: NonEmpty
	return_id: NonEmpty
	refund_type: str  # overpayment, input_vat_credit, withholding_tax_credit
	claimed_amount: Decimal
	supporting_documents: list[str] = Field(default_factory=list)
	bank_account_number: str | None = None
	bank_name: str | None = None
	evidence_reference: NonEmpty
	created_by: str = "system"


class TaxRefundResponse(TaxBase):
	taxpayer_id: str
	tax_pin: str
	return_id: str
	refund_type: str
	claimed_amount: Decimal
	approved_amount: Decimal | None = None
	bank_account_number: str | None = None
	bank_name: str | None = None
	evidence_reference: str
	status: RefundStatus
	reviewer_id: str | None = None
	review_notes: str | None = None
	processed_date: date | None = None


# ---------------------------------------------------------------------------
# Penalty
# ---------------------------------------------------------------------------

class PenaltyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	taxpayer_id: NonEmpty
	assessment_id: str | None = None
	return_id: str | None = None
	penalty_type: PenaltyType
	base_amount: Decimal
	rate: Decimal  # as decimal e.g. 0.05 for 5%
	calculated_amount: Decimal
	period_days: int | None = None
	created_by: str = "system"


class PenaltyResponse(TaxBase):
	taxpayer_id: str
	assessment_id: str | None = None
	return_id: str | None = None
	penalty_type: PenaltyType
	base_amount: Decimal
	rate: Decimal
	calculated_amount: Decimal
	period_days: int | None = None
	status: PenaltyStatus
	waiver_reason: str | None = None
	waived_by: str | None = None


# ---------------------------------------------------------------------------
# Interest
# ---------------------------------------------------------------------------

class InterestCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	taxpayer_id: NonEmpty
	assessment_id: str | None = None
	return_id: str | None = None
	interest_type: InterestType
	principal_amount: Decimal
	annual_rate: Decimal  # e.g. Decimal("0.02") for 2%
	from_date: date
	to_date: date
	calculated_amount: Decimal
	created_by: str = "system"


class InterestResponse(TaxBase):
	taxpayer_id: str
	assessment_id: str | None = None
	return_id: str | None = None
	interest_type: InterestType
	principal_amount: Decimal
	annual_rate: Decimal
	from_date: date
	to_date: date
	days: int
	calculated_amount: Decimal
	status: str = "assessed"


# ---------------------------------------------------------------------------
# TaxClearanceCertificate
# ---------------------------------------------------------------------------

class TaxClearanceCertificateCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	taxpayer_id: NonEmpty
	tax_pin: NonEmpty
	purpose: NonEmpty  # e.g. "government_tender", "business_license"
	validity_months: int = 6
	evidence_reference: NonEmpty
	created_by: str = "system"


class TaxClearanceCertificateResponse(TaxBase):
	taxpayer_id: str
	tax_pin: str
	purpose: str
	certificate_number: str
	issue_date: date | None = None
	expiry_date: date | None = None
	validity_months: int
	evidence_reference: str
	status: ClearanceCertificateStatus
	reviewer_id: str | None = None
	denial_reason: str | None = None


# ---------------------------------------------------------------------------
# Report / aggregation models
# ---------------------------------------------------------------------------

class TaxDashboardKPI(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	as_of: datetime
	registered_taxpayers: int
	active_taxpayers: int
	returns_filed_ytd: int
	returns_overdue: int
	assessments_pending: int
	total_tax_assessed: Decimal
	total_tax_collected: Decimal
	total_outstanding_debt: Decimal
	open_objections: int
	open_audits: int
	pending_refunds: int
	pending_clearance_certs: int
	compliance_rate: Decimal  # 0.0 – 1.0
	collection_rate: Decimal  # 0.0 – 1.0


class ComplianceRiskProfile(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	taxpayer_id: str
	tax_pin: str
	risk_score: Decimal  # 0.0 – 100.0
	risk_category: str  # low, medium, high, critical
	factors: dict[str, Any]
	recommended_action: str
	generated_at: datetime = Field(default_factory=datetime.utcnow)


class DebtAgingBucket(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	bucket_label: str  # "0-30", "31-90", "91-180", "180+"
	taxpayer_count: int
	total_amount: Decimal
	principal: Decimal
	penalty: Decimal
	interest: Decimal


class RevenueReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	period_start: date
	period_end: date
	generated_at: datetime = Field(default_factory=datetime.utcnow)
	by_tax_type: dict[str, Decimal]
	total_assessed: Decimal
	total_collected: Decimal
	total_refunded: Decimal
	net_revenue: Decimal
	target: Decimal | None = None
	variance: Decimal | None = None


class DemandNotice(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	debt_id: NonEmpty
	taxpayer_id: NonEmpty
	tax_pin: NonEmpty
	amount_demanded: Decimal
	due_date: date
	notice_number: str
	issued_date: date = Field(default_factory=date.today)
	notice_text: str | None = None
	issued_by: str = "system"


class EOIRequest(BaseModel):
	"""Exchange of Information request to a treaty partner."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	treaty_partner: NonEmpty  # ISO country code
	subject_taxpayer_id: NonEmpty
	subject_name: NonEmpty
	information_requested: NonEmpty
	legal_basis: str = "double_tax_agreement"
	urgency: str = "routine"  # routine, urgent, spontaneous
	submitted_at: datetime = Field(default_factory=datetime.utcnow)
	response_deadline: date | None = None
	status: str = "submitted"
	response_received: str | None = None
