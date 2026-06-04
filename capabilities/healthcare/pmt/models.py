"""Pydantic v2 models for APG Patient Management.

Covers all lifecycle entities: Patient, Appointment, WaitingList,
BedManagement, AdmissionRecord, DischargeRecord, PatientBill,
InsuranceClaim, CoPay, Deposit — plus their Create/Update/Response
variants, status enums, and report/aggregation models.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import (
	AfterValidator,
	BaseModel,
	ConfigDict,
	Field,
	field_validator,
	model_validator,
)
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


# ── enums ─────────────────────────────────────────────────────────────────────

class PatientStatus(str, Enum):
	active = "active"
	inactive = "inactive"
	deceased = "deceased"
	merged = "merged"


class GenderCode(str, Enum):
	male = "male"
	female = "female"
	other = "other"
	unknown = "unknown"


class AdmissionType(str, Enum):
	emergency = "emergency"
	elective = "elective"
	urgent = "urgent"
	newborn = "newborn"
	trauma = "trauma"
	observation = "observation"
	day_surgery = "day_surgery"
	psychiatric = "psychiatric"


class BedStatus(str, Enum):
	available = "available"
	occupied = "occupied"
	cleaning = "cleaning"
	maintenance = "maintenance"
	blocked = "blocked"
	isolation = "isolation"


class BedType(str, Enum):
	medical_surgical = "medical_surgical"
	icu = "icu"
	paediatric = "paediatric"
	neonatal = "neonatal"
	psychiatric = "psychiatric"
	isolation = "isolation"
	vip_private = "vip_private"
	day_surgery = "day_surgery"
	labour_delivery = "labour_delivery"


class AppointmentType(str, Enum):
	new_patient = "new_patient"
	follow_up = "follow_up"
	annual_wellness = "annual_wellness"
	urgent = "urgent"
	procedure = "procedure"
	telehealth = "telehealth"
	consultation = "consultation"
	preventive = "preventive"


class AppointmentStatus(str, Enum):
	scheduled = "scheduled"
	confirmed = "confirmed"
	checked_in = "checked_in"
	in_progress = "in_progress"
	completed = "completed"
	cancelled = "cancelled"
	no_show = "no_show"
	rescheduled = "rescheduled"


class WaitlistStatus(str, Enum):
	waiting = "waiting"
	offered = "offered"
	accepted = "accepted"
	declined = "declined"
	expired = "expired"
	admitted = "admitted"


class WaitlistPriority(str, Enum):
	emergency = "emergency"
	urgent = "urgent"
	semi_urgent = "semi_urgent"
	routine = "routine"


class DischargeDisposition(str, Enum):
	home = "home"
	home_with_services = "home_with_services"
	snf = "snf"
	rehab = "rehab"
	ltac = "ltac"
	hospice = "hospice"
	ama = "ama"
	expired = "expired"
	transfer = "transfer"
	left_without_treatment = "left_without_treatment"


class InsuranceType(str, Enum):
	commercial = "commercial"
	medicare = "medicare"
	medicaid = "medicaid"
	self_pay = "self_pay"
	workers_comp = "workers_comp"
	tricare = "tricare"
	va = "va"
	other_government = "other_government"


class InsuranceVerificationStatus(str, Enum):
	pending = "pending"
	verified = "verified"
	failed = "failed"
	expired = "expired"


class ClaimStatus(str, Enum):
	draft = "draft"
	pre_auth_pending = "pre_auth_pending"
	pre_auth_approved = "pre_auth_approved"
	pre_auth_denied = "pre_auth_denied"
	submitted = "submitted"
	acknowledged = "acknowledged"
	adjudicated = "adjudicated"
	paid = "paid"
	denied = "denied"
	appealed = "appealed"
	closed = "closed"


class BillingStatus(str, Enum):
	draft = "draft"
	pending = "pending"
	submitted = "submitted"
	partial_paid = "partial_paid"
	paid = "paid"
	denied = "denied"
	appealed = "appealed"
	written_off = "written_off"
	payment_plan = "payment_plan"


class PaymentPlanStatus(str, Enum):
	active = "active"
	completed = "completed"
	defaulted = "defaulted"
	cancelled = "cancelled"


class DepositType(str, Enum):
	admission = "admission"
	surgery = "surgery"
	procedure = "procedure"
	general = "general"


class CoPayStatus(str, Enum):
	pending = "pending"
	collected = "collected"
	waived = "waived"
	refunded = "refunded"


class IsolationReason(str, Enum):
	infectious = "infectious"
	immunocompromised = "immunocompromised"
	respiratory = "respiratory"
	contact = "contact"
	droplet = "droplet"
	airborne = "airborne"


class TriageLevel(str, Enum):
	"""ESI (Emergency Severity Index) triage levels."""
	level_1_resuscitation = "level_1_resuscitation"
	level_2_emergent = "level_2_emergent"
	level_3_urgent = "level_3_urgent"
	level_4_less_urgent = "level_4_less_urgent"
	level_5_non_urgent = "level_5_non_urgent"


class PortalEventType(str, Enum):
	registered = "registered"
	activated = "activated"
	login = "login"
	password_reset = "password_reset"
	deactivated = "deactivated"


# ── validators ────────────────────────────────────────────────────────────────

def _nonempty(v: str) -> str:
	if not v or not v.strip():
		raise ValueError("field must not be empty")
	return v.strip()


def _positive_decimal(v: Decimal) -> Decimal:
	if v < Decimal("0"):
		raise ValueError("monetary amount must be >= 0")
	return v


NonEmptyStr = Annotated[str, AfterValidator(_nonempty)]
PositiveAmount = Annotated[Decimal, AfterValidator(_positive_decimal)]


# ── base ──────────────────────────────────────────────────────────────────────

class PmtBase(BaseModel):
	"""Common audit columns on every persisted entity."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr
	is_deleted: bool = False


# ── patient ───────────────────────────────────────────────────────────────────

class PatientCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	first_name: NonEmptyStr
	last_name: NonEmptyStr
	date_of_birth: datetime
	gender_code: GenderCode
	ssn_last4: str | None = None
	national_id: str | None = None
	address: dict[str, str] = Field(default_factory=dict)
	phone: str | None = None
	email: str | None = None
	emergency_contact: dict[str, str] = Field(default_factory=dict)
	vip: bool = False
	paediatric_guardian_id: str | None = None
	language_preference: str = "en"
	preferred_pronouns: str | None = None
	allergies: list[str] = Field(default_factory=list)
	blood_type: str | None = None
	primary_provider_id: str | None = None
	created_by: NonEmptyStr


class PatientUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	first_name: str | None = None
	last_name: str | None = None
	address: dict[str, str] | None = None
	phone: str | None = None
	email: str | None = None
	emergency_contact: dict[str, str] | None = None
	language_preference: str | None = None
	preferred_pronouns: str | None = None
	vip: bool | None = None
	allergies: list[str] | None = None
	blood_type: str | None = None
	primary_provider_id: str | None = None
	status: PatientStatus | None = None


class PatientResponse(PmtBase):
	mrn: str = ""
	first_name: str
	last_name: str
	date_of_birth: datetime
	gender_code: GenderCode
	ssn_last4: str | None = None
	national_id: str | None = None
	address: dict[str, str] = Field(default_factory=dict)
	phone: str | None = None
	email: str | None = None
	emergency_contact: dict[str, str] = Field(default_factory=dict)
	status: PatientStatus = PatientStatus.active
	merged_into: str | None = None
	vip: bool = False
	paediatric_guardian_id: str | None = None
	language_preference: str = "en"
	preferred_pronouns: str | None = None
	allergies: list[str] = Field(default_factory=list)
	blood_type: str | None = None
	primary_provider_id: str | None = None
	portal_registered: bool = False

	@property
	def full_name(self) -> str:
		return f"{self.first_name} {self.last_name}"

	@property
	def age_years(self) -> int:
		today = datetime.utcnow()
		dob = self.date_of_birth
		return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

	@property
	def is_paediatric(self) -> bool:
		return self.age_years < 18


# ── triage ────────────────────────────────────────────────────────────────────

class TriageCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	patient_id: NonEmptyStr
	encounter_id: str | None = None
	triage_level: TriageLevel
	chief_complaint: NonEmptyStr
	vital_signs: dict[str, Any] = Field(default_factory=dict)
	pain_score: int | None = None  # 0–10
	allergies_confirmed: bool = False
	isolation_required: bool = False
	isolation_reason: IsolationReason | None = None
	triaged_by: NonEmptyStr
	created_by: NonEmptyStr

	@field_validator("pain_score")
	@classmethod
	def pain_score_range(cls, v: int | None) -> int | None:
		if v is not None and not (0 <= v <= 10):
			raise ValueError("pain_score must be 0–10")
		return v


class TriageResponse(PmtBase):
	patient_id: str
	encounter_id: str | None = None
	triage_level: TriageLevel
	chief_complaint: str
	vital_signs: dict[str, Any] = Field(default_factory=dict)
	pain_score: int | None = None
	allergies_confirmed: bool = False
	isolation_required: bool = False
	isolation_reason: IsolationReason | None = None
	triaged_by: str
	triaged_at: datetime = Field(default_factory=datetime.utcnow)
	disposition: str | None = None  # "admit", "discharge", "observation", "transfer"


# ── bed management ────────────────────────────────────────────────────────────

class BedCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	unit_id: NonEmptyStr
	bed_number: NonEmptyStr
	bed_type: BedType
	location: NonEmptyStr
	floor: str | None = None
	wing: str | None = None
	isolation_capable: bool = False
	paediatric_only: bool = False
	max_age_months: int | None = None
	ventilator_capable: bool = False
	telemetry_capable: bool = False
	created_by: NonEmptyStr


class BedUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: BedStatus | None = None
	isolation_reason: IsolationReason | None = None
	notes: str | None = None


class BedResponse(PmtBase):
	unit_id: str
	bed_number: str
	bed_type: BedType
	location: str
	floor: str | None = None
	wing: str | None = None
	status: BedStatus = BedStatus.available
	patient_id: str | None = None
	admission_id: str | None = None
	isolation_capable: bool = False
	isolation_reason: IsolationReason | None = None
	paediatric_only: bool = False
	max_age_months: int | None = None
	ventilator_capable: bool = False
	telemetry_capable: bool = False
	notes: str | None = None


# ── waiting list ──────────────────────────────────────────────────────────────

class WaitlistCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	patient_id: NonEmptyStr
	unit_id: NonEmptyStr
	requested_bed_type: BedType | None = None
	priority: WaitlistPriority = WaitlistPriority.routine
	clinical_notes: str = ""
	isolation_required: bool = False
	isolation_reason: IsolationReason | None = None
	paediatric: bool = False
	created_by: NonEmptyStr


class WaitlistUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	priority: WaitlistPriority | None = None
	status: WaitlistStatus | None = None
	offered_bed_id: str | None = None
	clinical_notes: str | None = None


class WaitlistResponse(PmtBase):
	patient_id: str
	unit_id: str
	requested_bed_type: BedType | None = None
	priority: WaitlistPriority
	status: WaitlistStatus = WaitlistStatus.waiting
	clinical_notes: str = ""
	isolation_required: bool = False
	isolation_reason: IsolationReason | None = None
	paediatric: bool = False
	offered_bed_id: str | None = None
	offered_at: datetime | None = None
	admitted_at: datetime | None = None
	expires_at: datetime | None = None
	position: int = 0
	priority_score: float = 0.0


# ── admission record ──────────────────────────────────────────────────────────

class AdmissionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	patient_id: NonEmptyStr
	admission_type: AdmissionType
	admitting_provider_id: NonEmptyStr
	attending_provider_id: NonEmptyStr
	unit_id: NonEmptyStr
	bed_id: NonEmptyStr
	chief_complaint: NonEmptyStr
	insurance_id: str | None = None
	physician_order_present: bool = True
	isolation_required: bool = False
	isolation_reason: IsolationReason | None = None
	emergency_bypass_registration: bool = False
	triage_id: str | None = None
	icd10_primary: str | None = None
	created_by: NonEmptyStr

	@model_validator(mode="after")
	def emergency_bypass_only_for_emergency(self) -> "AdmissionCreate":
		if self.emergency_bypass_registration and self.admission_type not in (
			AdmissionType.emergency, AdmissionType.trauma
		):
			raise ValueError("emergency_bypass_registration only valid for emergency/trauma admissions")
		return self


class AdmissionUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	attending_provider_id: str | None = None
	unit_id: str | None = None
	bed_id: str | None = None
	insurance_id: str | None = None
	isolation_required: bool | None = None
	isolation_reason: IsolationReason | None = None
	icd10_primary: str | None = None
	notes: str | None = None


class AdmissionResponse(PmtBase):
	patient_id: str
	admission_type: AdmissionType
	admitting_provider_id: str
	attending_provider_id: str
	unit_id: str
	bed_id: str
	chief_complaint: str
	insurance_id: str | None = None
	status: str = "admitted"
	admit_time: datetime = Field(default_factory=datetime.utcnow)
	discharge_time: datetime | None = None
	discharge_disposition: DischargeDisposition | None = None
	isolation_required: bool = False
	isolation_reason: IsolationReason | None = None
	emergency_bypass_registration: bool = False
	triage_id: str | None = None
	icd10_primary: str | None = None
	notes: str | None = None
	los_hours: float | None = None


# ── discharge record ──────────────────────────────────────────────────────────

class DischargeCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	admission_id: NonEmptyStr
	patient_id: NonEmptyStr
	disposition: DischargeDisposition
	discharge_summary: str
	follow_up_instructions: str = ""
	physician_order_present: bool
	discharge_medications: list[str] = Field(default_factory=list)
	follow_up_provider_id: str | None = None
	follow_up_date: datetime | None = None
	created_by: NonEmptyStr


class DischargeResponse(PmtBase):
	admission_id: str
	patient_id: str
	disposition: DischargeDisposition
	discharge_summary: str
	follow_up_instructions: str = ""
	physician_order_present: bool
	discharge_medications: list[str] = Field(default_factory=list)
	follow_up_provider_id: str | None = None
	follow_up_date: datetime | None = None
	discharged_at: datetime = Field(default_factory=datetime.utcnow)
	los_hours: float = 0.0


# ── appointment ───────────────────────────────────────────────────────────────

class AppointmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	patient_id: NonEmptyStr
	provider_id: NonEmptyStr
	appointment_type: AppointmentType
	scheduled_at: datetime
	duration_minutes: int = 30
	location_id: NonEmptyStr
	reason: NonEmptyStr
	slot_available: bool = True
	telemedicine: bool = False
	telemedicine_platform: str | None = None
	telemedicine_link: str | None = None
	reminder_sent: bool = False
	portal_booking: bool = False
	created_by: NonEmptyStr

	@field_validator("duration_minutes")
	@classmethod
	def duration_positive(cls, v: int) -> int:
		if v <= 0:
			raise ValueError("duration_minutes must be positive")
		return v


class AppointmentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	scheduled_at: datetime | None = None
	provider_id: str | None = None
	duration_minutes: int | None = None
	location_id: str | None = None
	reason: str | None = None
	telemedicine_link: str | None = None
	status: AppointmentStatus | None = None
	reminder_sent: bool | None = None


class AppointmentResponse(PmtBase):
	patient_id: str
	provider_id: str
	appointment_type: AppointmentType
	scheduled_at: datetime
	duration_minutes: int
	location_id: str
	reason: str
	status: AppointmentStatus = AppointmentStatus.scheduled
	cancellation_reason: str | None = None
	checked_in_at: datetime | None = None
	completed_at: datetime | None = None
	telemedicine: bool = False
	telemedicine_platform: str | None = None
	telemedicine_link: str | None = None
	portal_booking: bool = False
	reminder_sent: bool = False
	no_show_risk_score: float | None = None


# ── insurance ──────────────────────────────────────────────────────────────────

class InsuranceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	patient_id: NonEmptyStr
	insurance_type: InsuranceType
	payer_name: NonEmptyStr
	member_id: NonEmptyStr
	group_number: str | None = None
	effective_date: datetime
	termination_date: datetime | None = None
	primary: bool = True
	copay_amount: Decimal | None = None
	deductible: Decimal | None = None
	out_of_pocket_max: Decimal | None = None
	created_by: NonEmptyStr


class InsuranceUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	payer_name: str | None = None
	member_id: str | None = None
	group_number: str | None = None
	termination_date: datetime | None = None
	primary: bool | None = None
	verification_status: InsuranceVerificationStatus | None = None
	pre_auth_reference: str | None = None
	copay_amount: Decimal | None = None
	deductible: Decimal | None = None


class InsuranceResponse(PmtBase):
	patient_id: str
	insurance_type: InsuranceType
	payer_name: str
	member_id: str
	group_number: str | None = None
	effective_date: datetime
	termination_date: datetime | None = None
	primary: bool = True
	verification_status: InsuranceVerificationStatus = InsuranceVerificationStatus.pending
	pre_auth_reference: str | None = None
	copay_amount: Decimal | None = None
	deductible: Decimal | None = None
	out_of_pocket_max: Decimal | None = None


# ── insurance claim ────────────────────────────────────────────────────────────

class InsuranceClaimCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	patient_id: NonEmptyStr
	admission_id: NonEmptyStr
	insurance_id: NonEmptyStr
	bill_id: NonEmptyStr
	total_billed: PositiveAmount
	icd10_codes: list[str] = Field(default_factory=list)
	cpt_codes: list[str] = Field(default_factory=list)
	pre_auth_required: bool = False
	pre_auth_reference: str | None = None
	created_by: NonEmptyStr


class InsuranceClaimUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: ClaimStatus | None = None
	pre_auth_reference: str | None = None
	adjudicated_amount: PositiveAmount | None = None
	denial_reason: str | None = None
	appeal_notes: str | None = None


class InsuranceClaimResponse(PmtBase):
	patient_id: str
	admission_id: str
	insurance_id: str
	bill_id: str
	total_billed: Decimal
	adjudicated_amount: Decimal | None = None
	patient_responsibility: Decimal | None = None
	status: ClaimStatus = ClaimStatus.draft
	icd10_codes: list[str] = Field(default_factory=list)
	cpt_codes: list[str] = Field(default_factory=list)
	pre_auth_required: bool = False
	pre_auth_reference: str | None = None
	denial_reason: str | None = None
	appeal_notes: str | None = None
	submitted_at: datetime | None = None
	adjudicated_at: datetime | None = None


# ── patient bill ───────────────────────────────────────────────────────────────

class BillLineItem(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	description: NonEmptyStr
	cpt_code: str | None = None
	icd10_code: str | None = None
	quantity: int = 1
	unit_price: PositiveAmount
	total: PositiveAmount

	@field_validator("quantity")
	@classmethod
	def qty_positive(cls, v: int) -> int:
		if v < 1:
			raise ValueError("quantity must be >= 1")
		return v


class PatientBillCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	patient_id: NonEmptyStr
	admission_id: str | None = None
	appointment_id: str | None = None
	line_items: list[BillLineItem] = Field(default_factory=list)
	insurance_id: str | None = None
	payment_plan_eligible: bool = False
	created_by: NonEmptyStr


class PatientBillUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: BillingStatus | None = None
	insurance_adjustment: PositiveAmount | None = None
	amount_paid: PositiveAmount | None = None
	write_off_amount: PositiveAmount | None = None
	write_off_reason: str | None = None
	payment_plan_eligible: bool | None = None


class PatientBillResponse(PmtBase):
	patient_id: str
	admission_id: str | None = None
	appointment_id: str | None = None
	line_items: list[BillLineItem] = Field(default_factory=list)
	subtotal: Decimal = Decimal("0")
	insurance_adjustment: Decimal = Decimal("0")
	write_off_amount: Decimal = Decimal("0")
	amount_paid: Decimal = Decimal("0")
	balance_due: Decimal = Decimal("0")
	status: BillingStatus = BillingStatus.draft
	insurance_id: str | None = None
	payment_plan_eligible: bool = False
	payment_plan_id: str | None = None
	write_off_reason: str | None = None

	@property
	def is_uninsured(self) -> bool:
		return self.insurance_id is None


# ── copay ─────────────────────────────────────────────────────────────────────

class CoPayCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	patient_id: NonEmptyStr
	bill_id: NonEmptyStr
	insurance_id: NonEmptyStr
	amount: PositiveAmount
	created_by: NonEmptyStr


class CoPayUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: CoPayStatus | None = None
	waiver_reason: str | None = None
	collected_at: datetime | None = None
	collected_by: str | None = None


class CoPayResponse(PmtBase):
	patient_id: str
	bill_id: str
	insurance_id: str
	amount: Decimal
	status: CoPayStatus = CoPayStatus.pending
	waiver_reason: str | None = None
	collected_at: datetime | None = None
	collected_by: str | None = None


# ── deposit ───────────────────────────────────────────────────────────────────

class DepositCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	patient_id: NonEmptyStr
	admission_id: str | None = None
	deposit_type: DepositType = DepositType.admission
	amount: PositiveAmount
	payment_method: str = "cash"
	receipt_reference: str | None = None
	created_by: NonEmptyStr


class DepositUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	refunded: bool | None = None
	refund_reason: str | None = None
	applied_to_bill_id: str | None = None


class DepositResponse(PmtBase):
	patient_id: str
	admission_id: str | None = None
	deposit_type: DepositType
	amount: Decimal
	payment_method: str
	receipt_reference: str | None = None
	refunded: bool = False
	refund_reason: str | None = None
	applied_to_bill_id: str | None = None


# ── payment plan ───────────────────────────────────────────────────────────────

class PaymentPlanCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	patient_id: NonEmptyStr
	bill_id: NonEmptyStr
	total_amount: PositiveAmount
	installments: int
	installment_amount: PositiveAmount
	start_date: datetime
	created_by: NonEmptyStr

	@field_validator("installments")
	@classmethod
	def installments_positive(cls, v: int) -> int:
		if v < 2:
			raise ValueError("payment plan must have at least 2 installments")
		return v


class PaymentPlanResponse(PmtBase):
	patient_id: str
	bill_id: str
	total_amount: Decimal
	installments: int
	installment_amount: Decimal
	amount_paid: Decimal = Decimal("0")
	start_date: datetime
	status: PaymentPlanStatus = PaymentPlanStatus.active
	missed_payments: int = 0
	next_due_date: datetime | None = None


# ── portal registration ────────────────────────────────────────────────────────

class PatientPortalCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	patient_id: NonEmptyStr
	email: NonEmptyStr
	phone_for_mfa: str | None = None
	preferred_language: str = "en"
	created_by: NonEmptyStr


class PatientPortalUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	email: str | None = None
	phone_for_mfa: str | None = None
	preferred_language: str | None = None
	activated: bool | None = None


class PatientPortalResponse(PmtBase):
	patient_id: str
	email: str
	phone_for_mfa: str | None = None
	preferred_language: str = "en"
	activated: bool = False
	activated_at: datetime | None = None
	last_login_at: datetime | None = None
	login_count: int = 0


# ── telemedicine booking ───────────────────────────────────────────────────────

class TelemedicineBookingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	patient_id: NonEmptyStr
	provider_id: NonEmptyStr
	scheduled_at: datetime
	duration_minutes: int = 20
	platform: str = "web"
	chief_complaint: NonEmptyStr
	consent_obtained: bool = False
	created_by: NonEmptyStr

	@field_validator("duration_minutes")
	@classmethod
	def dur_pos(cls, v: int) -> int:
		if v < 5:
			raise ValueError("duration_minutes must be >= 5")
		return v


class TelemedicineBookingResponse(PmtBase):
	patient_id: str
	provider_id: str
	scheduled_at: datetime
	duration_minutes: int
	platform: str
	chief_complaint: str
	consent_obtained: bool = False
	status: str = "scheduled"
	join_url: str = ""
	appointment_id: str | None = None


# ── report / aggregation models ────────────────────────────────────────────────

class BedOccupancyReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	as_of: datetime
	unit_id: str | None
	total_beds: int
	available: int
	occupied: int
	cleaning: int
	maintenance: int
	blocked: int
	isolation: int
	occupancy_rate_pct: float
	overflow_risk: bool


class AdmissionSummaryReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	as_of: datetime
	admitted: int
	discharged_today: int
	avg_los_hours: float
	emergency_admissions: int
	elective_admissions: int
	waitlist_count: int


class BillingCollectionReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	as_of: datetime
	total_billed: Decimal
	total_collected: Decimal
	total_insurance_adjustments: Decimal
	total_write_offs: Decimal
	outstanding_balance: Decimal
	uninsured_balance: Decimal
	claims_pending: int
	claims_denied: int
	collection_rate_pct: float


class WaitlistSummaryReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	as_of: datetime
	total_waiting: int
	emergency_count: int
	urgent_count: int
	semi_urgent_count: int
	routine_count: int
	avg_wait_hours: float
	longest_wait_hours: float


class TriageSummaryReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	as_of: datetime
	total_triaged: int
	level_1_count: int
	level_2_count: int
	level_3_count: int
	level_4_count: int
	level_5_count: int
	avg_triage_to_bed_minutes: float


class DashboardSummary(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	as_of: datetime
	patients: dict[str, Any]
	admissions: dict[str, Any]
	beds: dict[str, Any]
	appointments: dict[str, Any]
	waitlist: dict[str, Any]
	billing: dict[str, Any]
	triage: dict[str, Any]
