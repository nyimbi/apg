"""
Employee Data Management — Pydantic v2 domain models.

Covers the full HCM employee lifecycle: hire → onboard → transfer/promote
→ performance → discipline/grievance → termination, plus supporting
entities (JobGrade, Qualification, Training, Contract, Benefit, Dependant,
EmergencyContact, WorkPermit, BackgroundCheck).

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from pydantic.functional_validators import AfterValidator
from uuid6 import uuid7


# ---------------------------------------------------------------------------
# UUID helper
# ---------------------------------------------------------------------------

def uuid7str() -> str:
	return str(uuid7())


# ---------------------------------------------------------------------------
# Shared validators
# ---------------------------------------------------------------------------

def _positive(v: Decimal) -> Decimal:
	assert v >= 0, "must be non-negative"
	return v


def _probability(v: float) -> float:
	assert 0.0 <= v <= 1.0, "must be in [0, 1]"
	return v


PositiveDecimal = Annotated[Decimal, AfterValidator(_positive)]
Probability = Annotated[float, AfterValidator(_probability)]


# ---------------------------------------------------------------------------
# Status / Type enumerations
# ---------------------------------------------------------------------------

class EmploymentStatus(str, Enum):
	ACTIVE = "active"
	PROBATION = "probation"
	NOTICE = "notice"
	SUSPENDED = "suspended"
	ON_LEAVE = "on_leave"
	TERMINATED = "terminated"
	RETIRED = "retired"
	DECEASED = "deceased"


class EmploymentType(str, Enum):
	FULL_TIME = "full_time"
	PART_TIME = "part_time"
	CONTRACT = "contract"
	INTERN = "intern"
	CASUAL = "casual"
	CONSULTANT = "consultant"


class WorkMode(str, Enum):
	OFFICE = "office"
	REMOTE = "remote"
	HYBRID = "hybrid"
	FIELD = "field"


class Gender(str, Enum):
	MALE = "male"
	FEMALE = "female"
	NON_BINARY = "non_binary"
	UNDISCLOSED = "undisclosed"


class MaritalStatus(str, Enum):
	SINGLE = "single"
	MARRIED = "married"
	DIVORCED = "divorced"
	WIDOWED = "widowed"
	SEPARATED = "separated"
	OTHER = "other"


class JobGradeLevel(str, Enum):
	"""Broad-banding grade levels."""
	GRADE_1 = "G1"
	GRADE_2 = "G2"
	GRADE_3 = "G3"
	GRADE_4 = "G4"
	GRADE_5 = "G5"
	GRADE_6 = "G6"
	GRADE_7 = "G7"
	GRADE_8 = "G8"
	GRADE_9 = "G9"
	GRADE_10 = "G10"


class QualificationLevel(str, Enum):
	CERTIFICATE = "certificate"
	DIPLOMA = "diploma"
	BACHELORS = "bachelors"
	HONOURS = "honours"
	MASTERS = "masters"
	DOCTORATE = "doctorate"
	PROFESSIONAL = "professional"
	VOCATIONAL = "vocational"


class TrainingStatus(str, Enum):
	PLANNED = "planned"
	ENROLLED = "enrolled"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	CANCELLED = "cancelled"
	FAILED = "failed"


class PerformanceRating(str, Enum):
	EXCEPTIONAL = "exceptional"        # 5 — top ~5 %
	EXCEEDS = "exceeds"                # 4
	MEETS = "meets"                    # 3
	NEEDS_IMPROVEMENT = "needs_improvement"  # 2
	UNSATISFACTORY = "unsatisfactory"  # 1


PERFORMANCE_RATING_SCORE: dict[PerformanceRating, int] = {
	PerformanceRating.EXCEPTIONAL: 5,
	PerformanceRating.EXCEEDS: 4,
	PerformanceRating.MEETS: 3,
	PerformanceRating.NEEDS_IMPROVEMENT: 2,
	PerformanceRating.UNSATISFACTORY: 1,
}


class ReviewStatus(str, Enum):
	DRAFT = "draft"
	SELF_ASSESSMENT = "self_assessment"
	MANAGER_REVIEW = "manager_review"
	CALIBRATION = "calibration"
	APPROVED = "approved"
	ACKNOWLEDGED = "acknowledged"


class DisciplinaryType(str, Enum):
	VERBAL_WARNING = "verbal_warning"
	WRITTEN_WARNING = "written_warning"
	FINAL_WARNING = "final_warning"
	SUSPENSION = "suspension"
	DEMOTION = "demotion"
	DISMISSAL = "dismissal"


class DisciplinaryStatus(str, Enum):
	INITIATED = "initiated"
	INVESTIGATION = "investigation"
	HEARING_SCHEDULED = "hearing_scheduled"
	OUTCOME_ISSUED = "outcome_issued"
	APPEALED = "appealed"
	CLOSED = "closed"
	OVERTURNED = "overturned"


class GrievanceStatus(str, Enum):
	SUBMITTED = "submitted"
	ACKNOWLEDGED = "acknowledged"
	INVESTIGATION = "investigation"
	MEDIATION = "mediation"
	RESOLVED = "resolved"
	ESCALATED = "escalated"
	CLOSED = "closed"
	WITHDRAWN = "withdrawn"


class ContractType(str, Enum):
	PERMANENT = "permanent"
	FIXED_TERM = "fixed_term"
	PROBATIONARY = "probationary"
	INTERNSHIP = "internship"
	PART_TIME = "part_time"
	CASUAL = "casual"
	ZERO_HOURS = "zero_hours"


class ContractStatus(str, Enum):
	DRAFT = "draft"
	PENDING_SIGNATURE = "pending_signature"
	ACTIVE = "active"
	EXPIRED = "expired"
	TERMINATED = "terminated"
	RENEWED = "renewed"


class BenefitType(str, Enum):
	HEALTH_INSURANCE = "health_insurance"
	LIFE_INSURANCE = "life_insurance"
	PENSION = "pension"
	PROVIDENT_FUND = "provident_fund"
	HOUSING_ALLOWANCE = "housing_allowance"
	TRANSPORT_ALLOWANCE = "transport_allowance"
	MEAL_ALLOWANCE = "meal_allowance"
	EDUCATION = "education"
	WELLNESS = "wellness"
	OTHER = "other"


class BenefitStatus(str, Enum):
	ELIGIBLE = "eligible"
	ENROLLED = "enrolled"
	ACTIVE = "active"
	SUSPENDED = "suspended"
	TERMINATED = "terminated"
	WAIVED = "waived"


class WorkPermitStatus(str, Enum):
	APPLIED = "applied"
	PENDING = "pending"
	APPROVED = "approved"
	ACTIVE = "active"
	RENEWAL_DUE = "renewal_due"
	EXPIRED = "expired"
	REJECTED = "rejected"
	CANCELLED = "cancelled"


class BackgroundCheckStatus(str, Enum):
	INITIATED = "initiated"
	IN_PROGRESS = "in_progress"
	CLEAR = "clear"
	FLAG = "flag"
	ADVERSE = "adverse"
	CANCELLED = "cancelled"
	EXPIRED = "expired"


class TerminationType(str, Enum):
	RESIGNATION = "resignation"
	REDUNDANCY = "redundancy"
	DISMISSAL = "dismissal"
	RETIREMENT = "retirement"
	CONTRACT_END = "contract_end"
	MUTUAL_AGREEMENT = "mutual_agreement"
	DECEASED = "deceased"
	ABANDONMENT = "abandonment"


class HistoryEventType(str, Enum):
	HIRE = "hire"
	ONBOARD = "onboard"
	TRANSFER = "transfer"
	PROMOTION = "promotion"
	DEMOTION = "demotion"
	COMPENSATION_CHANGE = "compensation_change"
	STATUS_CHANGE = "status_change"
	TERMINATION = "termination"
	REHIRE = "rehire"
	LEAVE_START = "leave_start"
	LEAVE_END = "leave_end"
	PROBATION_PASS = "probation_pass"
	PROBATION_FAIL = "probation_fail"
	CONTRACT_RENEWAL = "contract_renewal"


class OnboardingItemStatus(str, Enum):
	PENDING = "pending"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	SKIPPED = "skipped"
	BLOCKED = "blocked"


class SuccessionReadiness(str, Enum):
	READY_NOW = "ready_now"
	ONE_YEAR = "1_year"
	TWO_YEARS = "2_years"
	DEVELOPMENT = "development"


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class EDMBase(BaseModel):
	"""Base for all Employee Data Management Pydantic models."""
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
	)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"
	is_deleted: bool = False


# ---------------------------------------------------------------------------
# Department
# ---------------------------------------------------------------------------

class DepartmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	code: str = Field(min_length=2, max_length=20)
	name: str = Field(min_length=2, max_length=200)
	description: str | None = None
	parent_id: str | None = None
	manager_id: str | None = None
	cost_center: str | None = None
	location: str | None = None
	created_by: str = "system"


class DepartmentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	name: str | None = None
	description: str | None = None
	parent_id: str | None = None
	manager_id: str | None = None
	cost_center: str | None = None
	location: str | None = None
	is_active: bool | None = None


class Department(EDMBase):
	code: str
	name: str
	description: str | None = None
	parent_id: str | None = None
	manager_id: str | None = None
	cost_center: str | None = None
	location: str | None = None
	is_active: bool = True
	headcount: int = 0


# ---------------------------------------------------------------------------
# Job Grade
# ---------------------------------------------------------------------------

class JobGradeCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	code: str = Field(min_length=2, max_length=20)
	name: str = Field(min_length=2, max_length=100)
	level: JobGradeLevel
	min_salary: PositiveDecimal
	max_salary: PositiveDecimal
	currency: str = Field(default="KES", max_length=3)
	description: str | None = None
	created_by: str = "system"

	@field_validator("max_salary")
	@classmethod
	def max_exceeds_min(cls, v: Decimal, info: Any) -> Decimal:
		if "min_salary" in (info.data or {}) and v < info.data["min_salary"]:
			raise ValueError("max_salary must be >= min_salary")
		return v


class JobGradeUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	name: str | None = None
	min_salary: PositiveDecimal | None = None
	max_salary: PositiveDecimal | None = None
	description: str | None = None
	is_active: bool | None = None


class JobGrade(EDMBase):
	code: str
	name: str
	level: JobGradeLevel
	min_salary: PositiveDecimal
	max_salary: PositiveDecimal
	currency: str = "KES"
	description: str | None = None
	is_active: bool = True

	@property
	def midpoint(self) -> Decimal:
		return (self.min_salary + self.max_salary) / 2


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class PositionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	code: str = Field(min_length=2, max_length=20)
	title: str = Field(min_length=2, max_length=200)
	department_id: str
	job_grade_id: str
	employment_type: EmploymentType = EmploymentType.FULL_TIME
	authorized_headcount: int = Field(default=1, ge=1)
	reports_to_position_id: str | None = None
	description: str | None = None
	responsibilities: str | None = None
	requirements: str | None = None
	is_exempt: bool = True
	created_by: str = "system"


class PositionUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	title: str | None = None
	department_id: str | None = None
	job_grade_id: str | None = None
	authorized_headcount: int | None = None
	reports_to_position_id: str | None = None
	description: str | None = None
	is_active: bool | None = None


class Position(EDMBase):
	code: str
	title: str
	department_id: str
	job_grade_id: str
	employment_type: EmploymentType = EmploymentType.FULL_TIME
	authorized_headcount: int = 1
	current_headcount: int = 0
	reports_to_position_id: str | None = None
	description: str | None = None
	responsibilities: str | None = None
	requirements: str | None = None
	is_exempt: bool = True
	is_active: bool = True


# ---------------------------------------------------------------------------
# Employee (core record)
# ---------------------------------------------------------------------------

class EmployeeCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_number: str = Field(min_length=3, max_length=20)
	first_name: str = Field(min_length=1, max_length=100)
	middle_name: str | None = None
	last_name: str = Field(min_length=1, max_length=100)
	preferred_name: str | None = None
	work_email: EmailStr
	personal_email: EmailStr | None = None
	phone_mobile: str | None = None
	department_id: str
	position_id: str
	job_grade_id: str
	manager_id: str | None = None
	hire_date: date
	start_date: date | None = None
	employment_type: EmploymentType = EmploymentType.FULL_TIME
	employment_status: EmploymentStatus = EmploymentStatus.PROBATION
	work_mode: WorkMode = WorkMode.HYBRID
	nationality: str | None = None
	country_of_work: str = "KE"
	base_salary: PositiveDecimal | None = None
	currency: str = "KES"
	created_by: str = "system"


class EmployeeUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	first_name: str | None = None
	middle_name: str | None = None
	last_name: str | None = None
	preferred_name: str | None = None
	personal_email: EmailStr | None = None
	phone_mobile: str | None = None
	phone_home: str | None = None
	work_mode: WorkMode | None = None
	photo_url: str | None = None
	gender: Gender | None = None
	date_of_birth: date | None = None
	marital_status: MaritalStatus | None = None
	address_line1: str | None = None
	address_line2: str | None = None
	city: str | None = None
	country: str | None = None
	national_id: str | None = None


class Employee(EDMBase):
	employee_number: str
	first_name: str
	middle_name: str | None = None
	last_name: str
	preferred_name: str | None = None
	full_name: str = ""
	work_email: str
	personal_email: str | None = None
	phone_mobile: str | None = None
	phone_home: str | None = None
	phone_work: str | None = None
	gender: Gender | None = None
	date_of_birth: date | None = None
	marital_status: MaritalStatus | None = None
	nationality: str | None = None
	country_of_work: str = "KE"
	national_id: str | None = None
	address_line1: str | None = None
	address_line2: str | None = None
	city: str | None = None
	country: str | None = None
	department_id: str
	position_id: str
	job_grade_id: str
	manager_id: str | None = None
	hire_date: date
	start_date: date | None = None
	probation_end_date: date | None = None
	termination_date: date | None = None
	employment_type: EmploymentType = EmploymentType.FULL_TIME
	employment_status: EmploymentStatus = EmploymentStatus.PROBATION
	work_mode: WorkMode = WorkMode.HYBRID
	base_salary: PositiveDecimal | None = None
	currency: str = "KES"
	pay_frequency: str = "monthly"
	photo_url: str | None = None
	badge_id: str | None = None
	is_active: bool = True

	def model_post_init(self, __context: Any) -> None:
		if not self.full_name:
			parts = [self.first_name, self.middle_name, self.last_name]
			self.full_name = " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Qualification
# ---------------------------------------------------------------------------

class QualificationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_id: str
	institution: str = Field(min_length=2, max_length=300)
	qualification_name: str = Field(min_length=2, max_length=300)
	field_of_study: str | None = None
	level: QualificationLevel
	start_year: int = Field(ge=1950, le=2100)
	end_year: int | None = Field(default=None, ge=1950, le=2100)
	is_completed: bool = True
	grade: str | None = None
	country: str = "KE"
	document_ref: str | None = None
	created_by: str = "system"


class QualificationUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	end_year: int | None = None
	is_completed: bool | None = None
	grade: str | None = None
	document_ref: str | None = None
	verified: bool | None = None
	verified_by: str | None = None


class Qualification(EDMBase):
	employee_id: str
	institution: str
	qualification_name: str
	field_of_study: str | None = None
	level: QualificationLevel
	start_year: int
	end_year: int | None = None
	is_completed: bool = True
	grade: str | None = None
	country: str = "KE"
	document_ref: str | None = None
	verified: bool = False
	verified_by: str | None = None
	verified_at: datetime | None = None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_id: str
	title: str = Field(min_length=2, max_length=300)
	provider: str | None = None
	training_type: str = "internal"  # internal / external / e-learning
	start_date: date
	end_date: date | None = None
	duration_hours: float | None = None
	cost: PositiveDecimal | None = None
	currency: str = "KES"
	location: str | None = None
	objectives: str | None = None
	created_by: str = "system"


class TrainingUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: TrainingStatus | None = None
	end_date: date | None = None
	score: float | None = None
	certificate_ref: str | None = None
	facilitator_notes: str | None = None
	passed: bool | None = None


class Training(EDMBase):
	employee_id: str
	title: str
	provider: str | None = None
	training_type: str = "internal"
	status: TrainingStatus = TrainingStatus.PLANNED
	start_date: date
	end_date: date | None = None
	duration_hours: float | None = None
	cost: PositiveDecimal | None = None
	currency: str = "KES"
	location: str | None = None
	objectives: str | None = None
	score: float | None = None
	passed: bool | None = None
	certificate_ref: str | None = None
	facilitator_notes: str | None = None


# ---------------------------------------------------------------------------
# Performance Review
# ---------------------------------------------------------------------------

class PerformanceReviewCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_id: str
	reviewer_id: str
	review_period_start: date
	review_period_end: date
	review_type: str = "annual"  # annual / mid_year / probation / pip
	goals: list[dict[str, Any]] = Field(default_factory=list)
	created_by: str = "system"


class PerformanceReviewUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: ReviewStatus | None = None
	self_rating: PerformanceRating | None = None
	manager_rating: PerformanceRating | None = None
	overall_rating: PerformanceRating | None = None
	strengths: str | None = None
	development_areas: str | None = None
	goals_next_period: list[dict[str, Any]] | None = None
	calibrated_rating: PerformanceRating | None = None
	approved_by: str | None = None
	acknowledged_at: datetime | None = None


class PerformanceReview(EDMBase):
	employee_id: str
	reviewer_id: str
	review_period_start: date
	review_period_end: date
	review_type: str = "annual"
	status: ReviewStatus = ReviewStatus.DRAFT
	goals: list[dict[str, Any]] = Field(default_factory=list)
	self_rating: PerformanceRating | None = None
	manager_rating: PerformanceRating | None = None
	calibrated_rating: PerformanceRating | None = None
	overall_rating: PerformanceRating | None = None
	strengths: str | None = None
	development_areas: str | None = None
	goals_next_period: list[dict[str, Any]] = Field(default_factory=list)
	approved_by: str | None = None
	acknowledged_at: datetime | None = None


# ---------------------------------------------------------------------------
# Disciplinary
# ---------------------------------------------------------------------------

class DisciplinaryCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_id: str
	initiated_by: str
	disciplinary_type: DisciplinaryType
	incident_date: date
	incident_description: str = Field(min_length=10)
	created_by: str = "system"


class DisciplinaryUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: DisciplinaryStatus | None = None
	hearing_date: date | None = None
	outcome: str | None = None
	outcome_date: date | None = None
	appeal_date: date | None = None
	appeal_outcome: str | None = None
	closed_by: str | None = None
	closed_at: datetime | None = None


class Disciplinary(EDMBase):
	employee_id: str
	initiated_by: str
	disciplinary_type: DisciplinaryType
	status: DisciplinaryStatus = DisciplinaryStatus.INITIATED
	incident_date: date
	incident_description: str
	hearing_date: date | None = None
	outcome: str | None = None
	outcome_date: date | None = None
	appeal_date: date | None = None
	appeal_outcome: str | None = None
	closed_by: str | None = None
	closed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Grievance
# ---------------------------------------------------------------------------

class GrievanceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_id: str
	category: str  # harassment / discrimination / pay / working_conditions / other
	description: str = Field(min_length=10)
	is_anonymous: bool = False
	against_employee_id: str | None = None
	created_by: str = "system"


class GrievanceUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: GrievanceStatus | None = None
	assigned_to: str | None = None
	investigation_notes: str | None = None
	resolution: str | None = None
	resolved_at: datetime | None = None
	withdrawn_reason: str | None = None


class Grievance(EDMBase):
	employee_id: str
	category: str
	description: str
	status: GrievanceStatus = GrievanceStatus.SUBMITTED
	is_anonymous: bool = False
	against_employee_id: str | None = None
	assigned_to: str | None = None
	investigation_notes: str | None = None
	resolution: str | None = None
	resolved_at: datetime | None = None
	withdrawn_reason: str | None = None


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

class ContractCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_id: str
	contract_type: ContractType
	start_date: date
	end_date: date | None = None
	probation_end_date: date | None = None
	notice_period_days: int = 30
	base_salary: PositiveDecimal
	currency: str = "KES"
	pay_frequency: str = "monthly"
	position_id: str
	job_grade_id: str
	document_ref: str | None = None
	created_by: str = "system"


class ContractUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: ContractStatus | None = None
	end_date: date | None = None
	signed_by_employee_at: datetime | None = None
	signed_by_employer_at: datetime | None = None
	terminated_at: datetime | None = None
	termination_reason: str | None = None


class Contract(EDMBase):
	employee_id: str
	contract_type: ContractType
	status: ContractStatus = ContractStatus.DRAFT
	start_date: date
	end_date: date | None = None
	probation_end_date: date | None = None
	notice_period_days: int = 30
	base_salary: PositiveDecimal
	currency: str = "KES"
	pay_frequency: str = "monthly"
	position_id: str
	job_grade_id: str
	document_ref: str | None = None
	signed_by_employee_at: datetime | None = None
	signed_by_employer_at: datetime | None = None
	terminated_at: datetime | None = None
	termination_reason: str | None = None


# ---------------------------------------------------------------------------
# Benefit Enrollment
# ---------------------------------------------------------------------------

class BenefitEnrollmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_id: str
	benefit_type: BenefitType
	plan_name: str
	provider: str | None = None
	coverage_start: date
	coverage_end: date | None = None
	employee_contribution: PositiveDecimal = Decimal("0")
	employer_contribution: PositiveDecimal = Decimal("0")
	currency: str = "KES"
	created_by: str = "system"


class BenefitEnrollmentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: BenefitStatus | None = None
	coverage_end: date | None = None
	employee_contribution: PositiveDecimal | None = None
	employer_contribution: PositiveDecimal | None = None
	policy_number: str | None = None


class BenefitEnrollment(EDMBase):
	employee_id: str
	benefit_type: BenefitType
	plan_name: str
	provider: str | None = None
	status: BenefitStatus = BenefitStatus.ELIGIBLE
	coverage_start: date
	coverage_end: date | None = None
	employee_contribution: PositiveDecimal = Decimal("0")
	employer_contribution: PositiveDecimal = Decimal("0")
	currency: str = "KES"
	policy_number: str | None = None


# ---------------------------------------------------------------------------
# Dependant
# ---------------------------------------------------------------------------

class DependantCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_id: str
	first_name: str
	last_name: str
	relationship: str  # spouse / child / parent / sibling / other
	date_of_birth: date | None = None
	gender: Gender | None = None
	national_id: str | None = None
	is_beneficiary: bool = False
	created_by: str = "system"


class DependantUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	first_name: str | None = None
	last_name: str | None = None
	date_of_birth: date | None = None
	gender: Gender | None = None
	national_id: str | None = None
	is_beneficiary: bool | None = None
	is_active: bool | None = None


class Dependant(EDMBase):
	employee_id: str
	first_name: str
	last_name: str
	relationship: str
	date_of_birth: date | None = None
	gender: Gender | None = None
	national_id: str | None = None
	is_beneficiary: bool = False
	is_active: bool = True


# ---------------------------------------------------------------------------
# Emergency Contact
# ---------------------------------------------------------------------------

class EmergencyContactCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_id: str
	name: str = Field(min_length=2, max_length=200)
	relationship: str
	phone_primary: str = Field(min_length=7, max_length=25)
	phone_secondary: str | None = None
	email: str | None = None
	address: str | None = None
	is_primary: bool = False
	created_by: str = "system"


class EmergencyContactUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	name: str | None = None
	phone_primary: str | None = None
	phone_secondary: str | None = None
	email: str | None = None
	address: str | None = None
	is_primary: bool | None = None
	is_active: bool | None = None


class EmergencyContact(EDMBase):
	employee_id: str
	name: str
	relationship: str
	phone_primary: str
	phone_secondary: str | None = None
	email: str | None = None
	address: str | None = None
	is_primary: bool = False
	is_active: bool = True


# ---------------------------------------------------------------------------
# Work Permit
# ---------------------------------------------------------------------------

class WorkPermitCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_id: str
	nationality: str
	permit_type: str  # work_permit / residence / critical_skills / exemption
	permit_number: str | None = None
	country_of_work: str = "KE"
	issue_date: date | None = None
	expiry_date: date | None = None
	issuing_authority: str | None = None
	document_ref: str | None = None
	created_by: str = "system"


class WorkPermitUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: WorkPermitStatus | None = None
	permit_number: str | None = None
	issue_date: date | None = None
	expiry_date: date | None = None
	renewal_submitted_at: date | None = None
	rejection_reason: str | None = None


class WorkPermit(EDMBase):
	employee_id: str
	nationality: str
	permit_type: str
	status: WorkPermitStatus = WorkPermitStatus.APPLIED
	permit_number: str | None = None
	country_of_work: str = "KE"
	issue_date: date | None = None
	expiry_date: date | None = None
	renewal_submitted_at: date | None = None
	issuing_authority: str | None = None
	document_ref: str | None = None
	rejection_reason: str | None = None


# ---------------------------------------------------------------------------
# Background Check
# ---------------------------------------------------------------------------

class BackgroundCheckCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_id: str
	check_type: str  # criminal / credit / identity / education / employment
	provider: str | None = None
	initiated_by: str
	consent_given: bool
	consent_date: date
	created_by: str = "system"


class BackgroundCheckUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: BackgroundCheckStatus | None = None
	result_summary: str | None = None
	flags: list[str] | None = None
	completed_at: datetime | None = None
	expires_at: date | None = None
	report_ref: str | None = None


class BackgroundCheck(EDMBase):
	employee_id: str
	check_type: str
	provider: str | None = None
	initiated_by: str
	status: BackgroundCheckStatus = BackgroundCheckStatus.INITIATED
	consent_given: bool = False
	consent_date: date
	result_summary: str | None = None
	flags: list[str] = Field(default_factory=list)
	completed_at: datetime | None = None
	expires_at: date | None = None
	report_ref: str | None = None


# ---------------------------------------------------------------------------
# Employment History (audit trail entry)
# ---------------------------------------------------------------------------

class EmploymentHistoryEntry(EDMBase):
	employee_id: str
	event_type: HistoryEventType
	effective_date: date
	reason: str | None = None
	prev_department_id: str | None = None
	new_department_id: str | None = None
	prev_position_id: str | None = None
	new_position_id: str | None = None
	prev_job_grade_id: str | None = None
	new_job_grade_id: str | None = None
	prev_manager_id: str | None = None
	new_manager_id: str | None = None
	prev_salary: PositiveDecimal | None = None
	new_salary: PositiveDecimal | None = None
	prev_status: EmploymentStatus | None = None
	new_status: EmploymentStatus | None = None
	approved_by: str | None = None
	approved_at: datetime | None = None
	notes: str | None = None


# ---------------------------------------------------------------------------
# Onboarding
# ---------------------------------------------------------------------------

class OnboardingItem(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	task: str
	owner: str
	due_date: date | None = None
	status: OnboardingItemStatus = OnboardingItemStatus.PENDING
	completed_at: datetime | None = None
	notes: str | None = None


class OnboardingChecklist(EDMBase):
	employee_id: str
	items: list[OnboardingItem] = Field(default_factory=list)
	completed_at: datetime | None = None
	completion_pct: float = 0.0

	def model_post_init(self, __context: Any) -> None:
		if self.items:
			done = sum(1 for i in self.items if i.status == OnboardingItemStatus.COMPLETED)
			self.completion_pct = round(done / len(self.items) * 100, 1)


class OnboardingChecklistCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_id: str
	items: list[OnboardingItem]
	created_by: str = "system"


# ---------------------------------------------------------------------------
# Action request models (service input)
# ---------------------------------------------------------------------------

class TerminationRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_id: str
	termination_type: TerminationType
	effective_date: date
	reason: str = Field(min_length=5)
	last_working_day: date | None = None
	notice_date: date | None = None
	exit_interview_done: bool = False
	final_settlement_amount: PositiveDecimal | None = None
	initiated_by: str
	approved_by: str | None = None


class TransferRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_id: str
	new_department_id: str
	new_position_id: str
	new_manager_id: str | None = None
	effective_date: date
	reason: str
	approved_by: str | None = None


class PromotionRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_id: str
	new_position_id: str
	new_job_grade_id: str
	new_salary: PositiveDecimal
	effective_date: date
	reason: str
	approved_by: str | None = None


class CompensationChangeRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	employee_id: str
	new_salary: PositiveDecimal
	new_job_grade_id: str | None = None
	effective_date: date
	reason: str
	approved_by: str | None = None


# ---------------------------------------------------------------------------
# Aggregate / report models
# ---------------------------------------------------------------------------

class HeadcountSummary(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	as_at: date
	total_headcount: int
	active: int
	on_probation: int
	on_leave: int
	on_notice: int
	by_department: dict[str, int] = Field(default_factory=dict)
	by_employment_type: dict[str, int] = Field(default_factory=dict)
	by_gender: dict[str, int] = Field(default_factory=dict)
	by_nationality: dict[str, int] = Field(default_factory=dict)
	by_work_mode: dict[str, int] = Field(default_factory=dict)


class AttritionReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	period_start: date
	period_end: date
	opening_headcount: int
	closing_headcount: int
	new_hires: int
	terminations: int
	attrition_rate: float
	voluntary_turnover: int
	involuntary_turnover: int
	top_termination_reasons: list[dict[str, Any]] = Field(default_factory=list)
	attrition_by_department: dict[str, float] = Field(default_factory=dict)


class SuccessionCandidate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	employee_id: str
	full_name: str
	current_position_id: str
	target_position_id: str
	readiness: SuccessionReadiness
	performance_rating: PerformanceRating | None = None
	readiness_score: float = 0.0
	retention_risk: Probability = 0.0
	gap_areas: list[str] = Field(default_factory=list)


class ProbationReviewResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	employee_id: str
	outcome: str  # pass / fail / extend
	effective_date: date
	new_probation_end: date | None = None
	notes: str | None = None
	decided_by: str


class ListParams(BaseModel):
	"""Common pagination / filter parameters for list endpoints."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	page: int = Field(default=1, ge=1)
	page_size: int = Field(default=50, ge=1, le=500)
	search: str | None = None
	sort_by: str = "created_at"
	sort_dir: str = "desc"


class PagedResponse(BaseModel):
	"""Generic paged response wrapper."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	items: list[Any]
	total: int
	page: int
	page_size: int
	pages: int
