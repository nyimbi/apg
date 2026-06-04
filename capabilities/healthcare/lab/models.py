"""Pydantic v2 models for APG Laboratory Information System.

Entities: LabTest, LabOrder, Specimen, LabResult, ReferenceRange, QCResult,
          AnalyserInterface, CriticalValue, ExternalReferral

All models enforce tenant isolation, UUID7 IDs, and full audit columns.
© 2025 Datacraft — nyimbi@gmail.com
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator
from uuid6 import uuid7


# ── ID factory ─────────────────────────────────────────────────────────────────

def uuid7str() -> str:
	"""Return a time-sortable UUID7 string."""
	return str(uuid7())


# ── Enumerations ───────────────────────────────────────────────────────────────

class OrderStatus(str, Enum):
	PENDING		= "pending"
	COLLECTED	= "collected"
	IN_TRANSIT	= "in_transit"
	RECEIVED	= "received"
	PROCESSING	= "processing"
	RESULTED	= "resulted"
	VERIFIED	= "verified"
	REPORTED	= "reported"
	CANCELLED	= "cancelled"
	ON_HOLD		= "on_hold"


class SpecimenStatus(str, Enum):
	COLLECTED	= "collected"
	IN_TRANSIT	= "in_transit"
	RECEIVED	= "received"
	PROCESSING	= "processing"
	STORED		= "stored"
	REJECTED	= "rejected"
	DISPOSED	= "disposed"


class SpecimenType(str, Enum):
	BLOOD_VENOUS	= "blood_venous"
	BLOOD_ARTERIAL	= "blood_arterial"
	BLOOD_CAPILLARY	= "blood_capillary"
	URINE_RANDOM	= "urine_random"
	URINE_24H		= "urine_24h"
	CSF				= "csf"
	STOOL			= "stool"
	SPUTUM			= "sputum"
	SWAB_THROAT		= "swab_throat"
	SWAB_WOUND		= "swab_wound"
	BIOPSY_TISSUE	= "biopsy_tissue"
	PLEURAL_FLUID	= "pleural_fluid"
	SYNOVIAL_FLUID	= "synovial_fluid"
	BONE_MARROW		= "bone_marrow"
	SERUM			= "serum"
	PLASMA_EDTA		= "plasma_edta"
	PLASMA_CITRATE	= "plasma_citrate"


class ResultStatus(str, Enum):
	PRELIMINARY		= "preliminary"
	FINAL			= "final"
	CORRECTED		= "corrected"
	CANCELLED		= "cancelled"
	ENTERED_IN_ERROR	= "entered_in_error"


class AbnormalFlag(str, Enum):
	HIGH			= "H"
	VERY_HIGH		= "HH"
	LOW				= "L"
	VERY_LOW		= "LL"
	ABNORMAL		= "A"
	CRITICAL_HIGH	= "CH"
	CRITICAL_LOW	= "CL"


class CollectionPriority(str, Enum):
	ROUTINE	= "routine"
	STAT	= "stat"
	ASAP	= "asap"
	TIMED	= "timed"


class TestCategory(str, Enum):
	HEMATOLOGY		= "hematology"
	CHEMISTRY		= "chemistry"
	MICROBIOLOGY	= "microbiology"
	IMMUNOLOGY		= "immunology"
	URINALYSIS		= "urinalysis"
	COAGULATION		= "coagulation"
	TOXICOLOGY		= "toxicology"
	SEROLOGY		= "serology"
	MOLECULAR		= "molecular_diagnostics"
	PATHOLOGY		= "pathology"
	BLOOD_BANK		= "blood_bank"
	ENDOCRINOLOGY	= "endocrinology"


class QCStatus(str, Enum):
	PASSED			= "passed"
	FAILED			= "failed"
	PENDING_REVIEW	= "pending_review"
	REPEATED		= "repeated"
	ACCEPTED		= "accepted"
	REJECTED		= "rejected"


class InstrumentStatus(str, Enum):
	ONLINE			= "online"
	OFFLINE			= "offline"
	MAINTENANCE		= "maintenance"
	CALIBRATING		= "calibrating"
	QC_HOLD			= "qc_hold"
	DECOMMISSIONED	= "decommissioned"


class CriticalSeverity(str, Enum):
	CRITICAL_HIGH	= "critical_high"
	CRITICAL_LOW	= "critical_low"
	PANIC_VALUE		= "panic_value"


class RejectionReason(str, Enum):
	HEMOLYZED				= "hemolyzed"
	LIPEMIC					= "lipemic"
	INSUFFICIENT_VOLUME		= "insufficient_volume"
	WRONG_TUBE				= "wrong_tube"
	CLOTTED					= "clotted"
	INCORRECT_PATIENT_ID	= "incorrect_patient_id"
	TEMPERATURE_EXCURSION	= "temperature_excursion"
	UNLABELED				= "unlabeled"
	EXPIRED					= "expired"
	CONTAMINATED			= "contaminated"


class ReferralStatus(str, Enum):
	PENDING		= "pending"
	DISPATCHED	= "dispatched"
	RECEIVED	= "received"
	RESULTED	= "resulted"
	CANCELLED	= "cancelled"


class InterfaceProtocol(str, Enum):
	HL7_V2		= "hl7_v2"
	HL7_FHIR	= "hl7_fhir"
	ASTM_E1381	= "astm_e1381"
	POCT1_A		= "poct1_a"
	LIS_BRIDGE	= "lis_bridge"
	REST_JSON	= "rest_json"


class PatientSex(str, Enum):
	MALE	= "M"
	FEMALE	= "F"
	OTHER	= "O"
	UNKNOWN	= "U"


class CustodyEventType(str, Enum):
	"""Events recorded on a specimen chain-of-custody log."""
	COLLECTED	= "collected"
	TRANSFERRED	= "transferred"
	RECEIVED	= "received"
	PROCESSED	= "processed"
	STORED		= "stored"
	ALIQUOTED	= "aliquoted"
	DISPOSED	= "disposed"


class ReportType(str, Enum):
	PATIENT_RESULTS	= "patient_results"
	TAT_ANALYSIS	= "tat_analysis"
	QC_SUMMARY		= "qc_summary"
	CRITICAL_VALUES	= "critical_values"
	WORKLOAD		= "workload"
	REJECTION_RATE	= "rejection_rate"
	INSTRUMENT_PERF	= "instrument_performance"


class SortDirection(str, Enum):
	ASC		= "asc"
	DESC	= "desc"


# ── Shared validators ──────────────────────────────────────────────────────────

def _non_empty(v: str) -> str:
	if not v or not v.strip():
		raise ValueError("field must not be empty")
	return v.strip()


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]


# ── Base model ─────────────────────────────────────────────────────────────────

class AuditBase(BaseModel):
	"""Shared audit columns present on every entity."""
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
	)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr
	is_deleted: bool = False


# ── LabTest ────────────────────────────────────────────────────────────────────

class LabTestCreate(BaseModel):
	"""Master catalogue entry for a single diagnostic test."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	test_code: NonEmptyStr
	test_name: NonEmptyStr
	category: TestCategory
	specimen_types: list[SpecimenType] = Field(min_length=1)
	loinc_code: str | None = None
	cpt_code: str | None = None
	snomed_code: str | None = None
	turnaround_minutes: int = Field(default=120, gt=0)
	stat_turnaround_minutes: int = Field(default=60, gt=0)
	active: bool = True
	requires_fasting: bool = False
	requires_consent: bool = False
	price: float | None = Field(default=None, ge=0)
	department: str | None = None
	instructions: str | None = None
	sample_volume_ml: float | None = Field(default=None, gt=0)
	container_type: str | None = None
	storage_temperature: str | None = None
	created_by: NonEmptyStr


class LabTestUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	test_name: str | None = None
	turnaround_minutes: int | None = Field(default=None, gt=0)
	stat_turnaround_minutes: int | None = Field(default=None, gt=0)
	active: bool | None = None
	requires_fasting: bool | None = None
	requires_consent: bool | None = None
	price: float | None = Field(default=None, ge=0)
	department: str | None = None
	instructions: str | None = None
	sample_volume_ml: float | None = Field(default=None, gt=0)
	container_type: str | None = None


class LabTestResponse(AuditBase):
	"""Full lab test catalogue entry."""
	test_code: NonEmptyStr
	test_name: NonEmptyStr
	category: TestCategory
	specimen_types: list[SpecimenType]
	loinc_code: str | None = None
	cpt_code: str | None = None
	snomed_code: str | None = None
	turnaround_minutes: int = 120
	stat_turnaround_minutes: int = 60
	active: bool = True
	requires_fasting: bool = False
	requires_consent: bool = False
	price: float | None = None
	department: str | None = None
	instructions: str | None = None
	sample_volume_ml: float | None = None
	container_type: str | None = None
	storage_temperature: str | None = None


# ── LabOrder ───────────────────────────────────────────────────────────────────

class LabOrderCreate(BaseModel):
	"""Create a new lab order linked to a patient encounter."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	patient_id: NonEmptyStr
	encounter_id: NonEmptyStr
	test_code: NonEmptyStr
	test_name: NonEmptyStr
	test_category: TestCategory
	collection_priority: CollectionPriority = CollectionPriority.ROUTINE
	ordered_by: NonEmptyStr
	clinical_indication: NonEmptyStr
	specimen_type: SpecimenType
	patient_age_years: float | None = Field(default=None, ge=0)
	patient_sex: PatientSex | None = None
	fasting: bool = False
	notes: str | None = None
	created_by: NonEmptyStr


class LabOrderUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	collection_priority: CollectionPriority | None = None
	clinical_indication: str | None = None
	notes: str | None = None
	status: OrderStatus | None = None
	on_hold_reason: str | None = None


class LabOrderResponse(AuditBase):
	"""Full lab order with lifecycle status."""
	patient_id: NonEmptyStr
	encounter_id: NonEmptyStr
	test_code: NonEmptyStr
	test_name: NonEmptyStr
	test_category: TestCategory
	collection_priority: CollectionPriority = CollectionPriority.ROUTINE
	ordered_by: NonEmptyStr
	clinical_indication: NonEmptyStr
	specimen_type: SpecimenType
	patient_age_years: float | None = None
	patient_sex: PatientSex | None = None
	fasting: bool = False
	notes: str | None = None
	status: OrderStatus = OrderStatus.PENDING
	specimen_id: str | None = None
	result_id: str | None = None
	report_url: str | None = None
	ordered_at: datetime = Field(default_factory=datetime.utcnow)
	tat_due_at: datetime | None = None
	completed_at: datetime | None = None
	cancelled_reason: str | None = None
	on_hold_reason: str | None = None
	referral_id: str | None = None


# ── Specimen ───────────────────────────────────────────────────────────────────

class CustodyEvent(BaseModel):
	"""A single chain-of-custody event on a specimen."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	event_type: CustodyEventType
	actor_id: NonEmptyStr
	location: str | None = None
	notes: str | None = None
	timestamp: datetime = Field(default_factory=datetime.utcnow)


class SpecimenCreate(BaseModel):
	"""Collect a specimen for an existing order."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	order_id: NonEmptyStr
	patient_id: NonEmptyStr
	specimen_type: SpecimenType
	collected_by: NonEmptyStr
	collection_site: NonEmptyStr
	collection_volume_ml: float | None = Field(default=None, gt=0)
	tube_type: str | None = None
	barcode: str | None = None
	notes: str | None = None
	created_by: NonEmptyStr


class SpecimenUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	collection_site: str | None = None
	collection_volume_ml: float | None = Field(default=None, gt=0)
	tube_type: str | None = None
	notes: str | None = None
	storage_location: str | None = None
	status: SpecimenStatus | None = None


class SpecimenResponse(AuditBase):
	"""Full specimen with chain-of-custody tracking."""
	order_id: NonEmptyStr
	patient_id: NonEmptyStr
	specimen_type: SpecimenType
	collected_by: NonEmptyStr
	collection_site: NonEmptyStr
	collection_volume_ml: float | None = None
	tube_type: str | None = None
	barcode: str = Field(default_factory=lambda: f"LAB{uuid7str()[:8].upper()}")
	status: SpecimenStatus = SpecimenStatus.COLLECTED
	rejection_reason: RejectionReason | None = None
	rejection_notes: str | None = None
	collected_at: datetime = Field(default_factory=datetime.utcnow)
	received_at: datetime | None = None
	received_by: str | None = None
	processing_started_at: datetime | None = None
	stored_at: datetime | None = None
	storage_location: str | None = None
	aliquot_of: str | None = None  # parent specimen ID if aliquoted
	notes: str | None = None
	chain_of_custody: list[CustodyEvent] = Field(default_factory=list)


class SpecimenTrackRequest(BaseModel):
	"""Append a custody event to a specimen's chain-of-custody log."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	event_type: CustodyEventType
	actor_id: NonEmptyStr
	location: str | None = None
	notes: str | None = None


# ── ReferenceRange ─────────────────────────────────────────────────────────────

class ReferenceRangeCreate(BaseModel):
	"""Define normal ranges for an analyte, optionally stratified by demographics."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	test_code: NonEmptyStr
	analyte: NonEmptyStr
	unit: NonEmptyStr
	low: float | None = None
	high: float | None = None
	critical_low: float | None = None
	critical_high: float | None = None
	age_min_years: float | None = Field(default=None, ge=0)
	age_max_years: float | None = Field(default=None, ge=0)
	sex: PatientSex | None = None
	condition: str | None = None
	effective_date: datetime = Field(default_factory=datetime.utcnow)
	expiry_date: datetime | None = None
	source: str | None = None
	created_by: NonEmptyStr

	@field_validator("high")
	@classmethod
	def high_above_low(cls, v: float | None, info: Any) -> float | None:
		low = info.data.get("low")
		if v is not None and low is not None and v <= low:
			raise ValueError("reference high must be greater than low")
		return v

	@model_validator(mode="after")
	def critical_limits_outside_normal(self) -> "ReferenceRangeCreate":
		if self.critical_low is not None and self.low is not None and self.critical_low >= self.low:
			raise ValueError("critical_low must be below normal low")
		if self.critical_high is not None and self.high is not None and self.critical_high <= self.high:
			raise ValueError("critical_high must be above normal high")
		return self


class ReferenceRangeUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	low: float | None = None
	high: float | None = None
	critical_low: float | None = None
	critical_high: float | None = None
	active: bool | None = None
	expiry_date: datetime | None = None


class ReferenceRangeResponse(AuditBase):
	"""Persisted reference range with demographic stratification."""
	test_code: NonEmptyStr
	analyte: NonEmptyStr
	unit: NonEmptyStr
	low: float | None = None
	high: float | None = None
	critical_low: float | None = None
	critical_high: float | None = None
	age_min_years: float | None = None
	age_max_years: float | None = None
	sex: PatientSex | None = None
	condition: str | None = None
	effective_date: datetime = Field(default_factory=datetime.utcnow)
	expiry_date: datetime | None = None
	source: str | None = None
	active: bool = True


# ── LabResult ──────────────────────────────────────────────────────────────────

class LabResultCreate(BaseModel):
	"""Enter a test result for a specimen."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	order_id: NonEmptyStr
	specimen_id: NonEmptyStr
	analyte: NonEmptyStr
	value: float | str
	unit: NonEmptyStr
	reference_low: float | None = None
	reference_high: float | None = None
	critical_low: float | None = None
	critical_high: float | None = None
	result_status: ResultStatus = ResultStatus.PRELIMINARY
	instrument_id: str | None = None
	method: str | None = None
	dilution_factor: float | None = Field(default=None, gt=0)
	previous_value: float | str | None = None
	notes: str | None = None
	performed_by: NonEmptyStr
	created_by: NonEmptyStr


class LabResultUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	value: float | str | None = None
	unit: str | None = None
	notes: str | None = None
	result_status: ResultStatus | None = None
	method: str | None = None


class LabResultResponse(AuditBase):
	"""Full result with flags, audit trail, and amendment linkage."""
	order_id: NonEmptyStr
	specimen_id: NonEmptyStr
	analyte: NonEmptyStr
	value: float | str
	unit: NonEmptyStr
	reference_low: float | None = None
	reference_high: float | None = None
	critical_low: float | None = None
	critical_high: float | None = None
	result_status: ResultStatus = ResultStatus.PRELIMINARY
	abnormal_flag: AbnormalFlag | None = None
	critical_value: bool = False
	delta_check_flag: bool = False
	previous_value: float | str | None = None
	amendment_of: str | None = None
	instrument_id: str | None = None
	method: str | None = None
	dilution_factor: float | None = None
	notes: str | None = None
	performed_by: NonEmptyStr
	verified_by: str | None = None
	verified_at: datetime | None = None
	released_at: datetime | None = None


# ── QCResult ───────────────────────────────────────────────────────────────────

class QCRunCreate(BaseModel):
	"""Record a quality control run on an analyser."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	instrument_id: NonEmptyStr
	test_code: NonEmptyStr
	lot_number: NonEmptyStr
	expiry_date: datetime | None = None
	level: NonEmptyStr  # e.g. "low", "normal", "high"
	measured_value: float
	target_value: float
	sd: float = Field(gt=0)
	performed_by: NonEmptyStr
	created_by: NonEmptyStr

	@field_validator("sd")
	@classmethod
	def sd_positive(cls, v: float) -> float:
		if v <= 0:
			raise ValueError("sd must be positive")
		return v


class QCRunUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: QCStatus | None = None
	notes: str | None = None
	reviewed_by: str | None = None


class QCRunResponse(AuditBase):
	"""QC run with full Westgard rule evaluation."""
	instrument_id: NonEmptyStr
	test_code: NonEmptyStr
	lot_number: NonEmptyStr
	expiry_date: datetime | None = None
	level: NonEmptyStr
	measured_value: float
	target_value: float
	sd: float
	z_score: float = 0.0
	cv_percent: float = 0.0
	status: QCStatus = QCStatus.PENDING_REVIEW
	westgard_violations: list[str] = Field(default_factory=list)
	reviewed_by: str | None = None
	reviewed_at: datetime | None = None
	performed_by: NonEmptyStr
	notes: str | None = None


# ── AnalyserInterface ──────────────────────────────────────────────────────────

class AnalyserInterfaceCreate(BaseModel):
	"""Register an analyser instrument interface."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	name: NonEmptyStr
	model: NonEmptyStr
	serial_number: NonEmptyStr
	manufacturer: NonEmptyStr
	protocol: InterfaceProtocol = InterfaceProtocol.HL7_V2
	test_categories: list[TestCategory] = Field(default_factory=list)
	location: NonEmptyStr
	ip_address: str | None = None
	port: int | None = Field(default=None, gt=0, lt=65536)
	connection_string: str | None = None
	calibration_interval_days: int = Field(default=90, gt=0)
	created_by: NonEmptyStr


class AnalyserInterfaceUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: InstrumentStatus | None = None
	location: str | None = None
	ip_address: str | None = None
	port: int | None = Field(default=None, gt=0, lt=65536)
	connection_string: str | None = None
	calibration_interval_days: int | None = Field(default=None, gt=0)


class AnalyserInterfaceResponse(AuditBase):
	"""Full analyser interface record."""
	name: NonEmptyStr
	model: NonEmptyStr
	serial_number: NonEmptyStr
	manufacturer: NonEmptyStr
	protocol: InterfaceProtocol = InterfaceProtocol.HL7_V2
	test_categories: list[TestCategory] = Field(default_factory=list)
	location: NonEmptyStr
	ip_address: str | None = None
	port: int | None = None
	connection_string: str | None = None
	status: InstrumentStatus = InstrumentStatus.ONLINE
	calibration_interval_days: int = 90
	last_calibrated_at: datetime | None = None
	calibration_due_at: datetime | None = None
	last_qc_at: datetime | None = None
	last_message_at: datetime | None = None
	message_count: int = 0


# Aliases for backward-compat with service.py
InstrumentCreate = AnalyserInterfaceCreate
InstrumentResponse = AnalyserInterfaceResponse


# ── CriticalValue ──────────────────────────────────────────────────────────────

class CriticalValueCreate(BaseModel):
	"""Flag and log a critical value notification."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	result_id: NonEmptyStr
	patient_id: NonEmptyStr
	analyte: NonEmptyStr
	value: float | str
	unit: NonEmptyStr
	severity: CriticalSeverity = CriticalSeverity.CRITICAL_HIGH
	notified_to: NonEmptyStr
	notified_by: NonEmptyStr
	notification_method: str = "phone"
	read_back_confirmed: bool = False
	created_by: NonEmptyStr


class CriticalValueResponse(AuditBase):
	"""Critical value notification with acknowledgement tracking."""
	result_id: NonEmptyStr
	patient_id: NonEmptyStr
	analyte: NonEmptyStr
	value: float | str
	unit: NonEmptyStr
	severity: CriticalSeverity
	notified_to: NonEmptyStr
	notified_by: NonEmptyStr
	notification_method: str = "phone"
	read_back_confirmed: bool = False
	acknowledged_by: str | None = None
	acknowledged_at: datetime | None = None
	escalated: bool = False
	escalated_to: str | None = None
	escalated_at: datetime | None = None
	notes: str | None = None


# Alias for backward-compat
CriticalValueNotification = CriticalValueResponse


# ── ExternalReferral ───────────────────────────────────────────────────────────

class ExternalReferralCreate(BaseModel):
	"""Send a specimen/test to an external reference laboratory."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	order_id: NonEmptyStr
	specimen_id: NonEmptyStr
	patient_id: NonEmptyStr
	reference_lab_name: NonEmptyStr
	reference_lab_code: NonEmptyStr
	test_code: NonEmptyStr
	test_name: NonEmptyStr
	clinical_notes: str | None = None
	expected_tat_hours: int | None = Field(default=None, gt=0)
	dispatched_by: NonEmptyStr
	created_by: NonEmptyStr


class ExternalReferralUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: ReferralStatus | None = None
	tracking_number: str | None = None
	external_result_id: str | None = None
	notes: str | None = None
	received_at: datetime | None = None


class ExternalReferralResponse(AuditBase):
	"""External referral with dispatch and result tracking."""
	order_id: NonEmptyStr
	specimen_id: NonEmptyStr
	patient_id: NonEmptyStr
	reference_lab_name: NonEmptyStr
	reference_lab_code: NonEmptyStr
	test_code: NonEmptyStr
	test_name: NonEmptyStr
	clinical_notes: str | None = None
	expected_tat_hours: int | None = None
	dispatched_by: NonEmptyStr
	dispatched_at: datetime | None = None
	tracking_number: str | None = None
	status: ReferralStatus = ReferralStatus.PENDING
	received_at: datetime | None = None
	external_result_id: str | None = None
	result_received_at: datetime | None = None
	notes: str | None = None


# ── Pagination ─────────────────────────────────────────────────────────────────

class PageParams(BaseModel):
	"""Query parameters for paginated list endpoints."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	page: int = Field(default=1, ge=1)
	page_size: int = Field(default=50, ge=1, le=500)
	sort_by: str | None = None
	sort_dir: SortDirection = SortDirection.DESC


class PaginatedResponse(BaseModel):
	"""Generic paginated list wrapper."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	items: list[Any]
	total: int
	page: int
	page_size: int
	pages: int


# ── Report / aggregation models ────────────────────────────────────────────────

class LabReportRequest(BaseModel):
	"""Parameters for generating a lab report."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	report_type: ReportType
	date_from: datetime | None = None
	date_to: datetime | None = None
	patient_id: str | None = None
	order_id: str | None = None
	instrument_id: str | None = None
	test_category: TestCategory | None = None
	format: str = "json"  # json | csv | pdf
	requested_by: NonEmptyStr


class DashboardSummary(BaseModel):
	"""Dashboard KPIs for the LIS home screen."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	orders: dict[str, Any]
	specimens: dict[str, Any]
	results: dict[str, Any]
	critical_values: dict[str, Any]
	qc: dict[str, Any]
	instruments: dict[str, Any]
	referrals: dict[str, Any]
	tat_metrics: dict[str, Any]
	generated_at: datetime = Field(default_factory=datetime.utcnow)


class TATMetrics(BaseModel):
	"""Turnaround time metrics for a cohort of orders."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	total_orders: int
	completed_orders: int
	median_tat_minutes: float | None
	p90_tat_minutes: float | None
	stat_median_tat_minutes: float | None
	overdue_count: int
	on_time_rate_pct: float | None


class WestgardSummary(BaseModel):
	"""Aggregate Westgard violation summary per instrument/test."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	instrument_id: str
	test_code: str
	period_runs: int
	failed_runs: int
	pass_rate_pct: float
	violations: dict[str, int]  # rule -> count
	current_status: InstrumentStatus


class InstrumentMessage(BaseModel):
	"""A raw message received from an analyser interface."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	instrument_id: NonEmptyStr
	tenant_id: NonEmptyStr
	protocol: InterfaceProtocol
	message_type: str
	raw_payload: str
	parsed_results: list[dict[str, Any]] = Field(default_factory=list)
	received_at: datetime = Field(default_factory=datetime.utcnow)
	processed: bool = False
	error: str | None = None


class CalibrationRecord(BaseModel):
	"""Record of an instrument calibration event."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	instrument_id: NonEmptyStr
	tenant_id: NonEmptyStr
	calibrated_by: NonEmptyStr
	calibration_date: datetime = Field(default_factory=datetime.utcnow)
	next_due_date: datetime
	notes: str | None = None
	pass_fail: bool = True
	created_by: NonEmptyStr
