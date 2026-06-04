"""Pydantic v2 models for APG Electronic Medical Records — FHIR R4 compliant."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


# ── enums ──────────────────────────────────────────────────────────────────────

class Gender(str, Enum):
	male = "male"
	female = "female"
	other = "other"
	unknown = "unknown"

class MaritalStatus(str, Enum):
	single = "single"
	married = "married"
	divorced = "divorced"
	widowed = "widowed"
	separated = "separated"
	unknown = "unknown"

class PatientStatus(str, Enum):
	active = "active"
	inactive = "inactive"
	deceased = "deceased"
	merged = "merged"

class EncounterStatus(str, Enum):
	planned = "planned"
	arrived = "arrived"
	triaged = "triaged"
	in_progress = "in_progress"
	on_leave = "on_leave"
	finished = "finished"
	cancelled = "cancelled"

class EncounterType(str, Enum):
	inpatient = "inpatient"
	outpatient = "outpatient"
	emergency = "emergency"
	observation = "observation"
	ambulatory = "ambulatory"
	home_health = "home_health"
	telehealth = "telehealth"

class NoteType(str, Enum):
	soap_note = "soap_note"
	progress_note = "progress_note"
	discharge_summary = "discharge_summary"
	operative_note = "operative_note"
	consultation_note = "consultation_note"
	nursing_note = "nursing_note"
	procedure_note = "procedure_note"
	psychiatric_note = "psychiatric_note"
	history_physical = "history_physical"
	emergency_note = "emergency_note"
	addendum = "addendum"

class NoteStatus(str, Enum):
	draft = "draft"
	final = "final"
	amended = "amended"
	entered_in_error = "entered_in_error"

class ProblemStatus(str, Enum):
	active = "active"
	inactive = "inactive"
	resolved = "resolved"
	chronic = "chronic"
	episodic = "episodic"

class DiagnosisCertainty(str, Enum):
	confirmed = "confirmed"
	differential = "differential"
	provisional = "provisional"
	refuted = "refuted"

class AllergyType(str, Enum):
	drug = "drug"
	food = "food"
	environmental = "environmental"
	contrast = "contrast"
	latex = "latex"
	other = "other"

class AllergySeverity(str, Enum):
	mild = "mild"
	moderate = "moderate"
	severe = "severe"
	life_threatening = "life_threatening"

class AllergyStatus(str, Enum):
	active = "active"
	inactive = "inactive"
	resolved = "resolved"
	entered_in_error = "entered_in_error"

class MedicationStatus(str, Enum):
	active = "active"
	discontinued = "discontinued"
	on_hold = "on_hold"
	completed = "completed"
	entered_in_error = "entered_in_error"

class PrescriptionStatus(str, Enum):
	draft = "draft"
	active = "active"
	on_hold = "on_hold"
	cancelled = "cancelled"
	completed = "completed"
	stopped = "stopped"
	entered_in_error = "entered_in_error"

class LabOrderStatus(str, Enum):
	draft = "draft"
	requested = "requested"
	received = "received"
	accepted = "accepted"
	in_progress = "in_progress"
	completed = "completed"
	cancelled = "cancelled"

class LabResultStatus(str, Enum):
	pending = "pending"
	preliminary = "preliminary"
	final = "final"
	amended = "amended"
	corrected = "corrected"
	cancelled = "cancelled"

class LabResultFlag(str, Enum):
	normal = "normal"
	low = "low"
	high = "high"
	critical_low = "critical_low"
	critical_high = "critical_high"
	abnormal = "abnormal"

class ImagingStatus(str, Enum):
	requested = "requested"
	scheduled = "scheduled"
	in_progress = "in_progress"
	completed = "completed"
	cancelled = "cancelled"

class CarePlanStatus(str, Enum):
	draft = "draft"
	active = "active"
	on_hold = "on_hold"
	completed = "completed"
	cancelled = "cancelled"
	revoked = "revoked"

class ReferralStatus(str, Enum):
	draft = "draft"
	active = "active"
	completed = "completed"
	cancelled = "cancelled"
	declined = "declined"

class ConsentStatus(str, Enum):
	active = "active"
	inactive = "inactive"
	entered_in_error = "entered_in_error"
	proposed = "proposed"
	rejected = "rejected"

class ImmunisationStatus(str, Enum):
	completed = "completed"
	entered_in_error = "entered_in_error"
	not_done = "not_done"

class VitalType(str, Enum):
	blood_pressure = "blood_pressure"
	heart_rate = "heart_rate"
	respiratory_rate = "respiratory_rate"
	temperature = "temperature"
	oxygen_saturation = "oxygen_saturation"
	weight = "weight"
	height = "height"
	bmi = "bmi"
	pain_scale = "pain_scale"
	blood_glucose = "blood_glucose"
	head_circumference = "head_circumference"
	waist_circumference = "waist_circumference"

class DrugInteractionSeverity(str, Enum):
	contraindicated = "contraindicated"
	major = "major"
	moderate = "moderate"
	minor = "minor"

class AlertType(str, Enum):
	drug_allergy = "drug_allergy"
	drug_interaction = "drug_interaction"
	drug_pregnancy = "drug_pregnancy"
	drug_renal = "drug_renal"
	drug_hepatic = "drug_hepatic"
	drug_paediatric = "drug_paediatric"
	critical_lab = "critical_lab"
	overdue_immunisation = "overdue_immunisation"
	care_gap = "care_gap"
	controlled_substance = "controlled_substance"


# ── base ───────────────────────────────────────────────────────────────────────

class _Base(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	is_deleted: bool = False


# ── patient ────────────────────────────────────────────────────────────────────

class PatientIdentifier(BaseModel):
	"""FHIR R4 Identifier."""
	model_config = ConfigDict(extra="forbid")
	system: str
	value: str
	use: str = "official"

class PatientName(BaseModel):
	"""FHIR R4 HumanName."""
	model_config = ConfigDict(extra="forbid")
	use: str = "official"
	family: str
	given: list[str] = Field(default_factory=list)
	prefix: list[str] = Field(default_factory=list)
	suffix: list[str] = Field(default_factory=list)

class PatientAddress(BaseModel):
	"""FHIR R4 Address."""
	model_config = ConfigDict(extra="forbid")
	use: str = "home"
	line: list[str] = Field(default_factory=list)
	city: str = ""
	district: str = ""
	state: str = ""
	postal_code: str = ""
	country: str = ""

class PatientTelecom(BaseModel):
	"""FHIR R4 ContactPoint."""
	model_config = ConfigDict(extra="forbid")
	system: str  # phone | fax | email | pager | url | sms | other
	value: str
	use: str = "home"

class NextOfKin(BaseModel):
	model_config = ConfigDict(extra="forbid")
	name: str
	relationship: str
	phone: str = ""
	email: str = ""

class PatientCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	identifiers: list[PatientIdentifier] = Field(default_factory=list)
	name: PatientName
	birth_date: date
	gender: Gender
	marital_status: MaritalStatus = MaritalStatus.unknown
	deceased_date: date | None = None
	is_deceased: bool = False
	address: list[PatientAddress] = Field(default_factory=list)
	telecom: list[PatientTelecom] = Field(default_factory=list)
	language: str = "en"
	nationality: str = ""
	religion: str = ""
	race: str = ""
	ethnicity: str = ""
	blood_type: str | None = None
	next_of_kin: list[NextOfKin] = Field(default_factory=list)
	emergency_contact: NextOfKin | None = None
	# mental health flag — triggers enhanced confidentiality
	mental_health_record: bool = False
	# biometric fingerprint hash for dedup
	biometric_hash: str | None = None
	created_by: str

class PatientUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	name: PatientName | None = None
	gender: Gender | None = None
	marital_status: MaritalStatus | None = None
	deceased_date: date | None = None
	is_deceased: bool | None = None
	address: list[PatientAddress] | None = None
	telecom: list[PatientTelecom] | None = None
	blood_type: str | None = None
	next_of_kin: list[NextOfKin] | None = None
	emergency_contact: NextOfKin | None = None
	mental_health_record: bool | None = None

class PatientResponse(_Base):
	identifiers: list[PatientIdentifier] = Field(default_factory=list)
	name: PatientName
	birth_date: date
	gender: Gender
	marital_status: MaritalStatus = MaritalStatus.unknown
	deceased_date: date | None = None
	is_deceased: bool = False
	address: list[PatientAddress] = Field(default_factory=list)
	telecom: list[PatientTelecom] = Field(default_factory=list)
	language: str = "en"
	nationality: str = ""
	religion: str = ""
	race: str = ""
	ethnicity: str = ""
	blood_type: str | None = None
	next_of_kin: list[NextOfKin] = Field(default_factory=list)
	emergency_contact: NextOfKin | None = None
	status: PatientStatus = PatientStatus.active
	mental_health_record: bool = False
	merged_into: str | None = None
	biometric_hash: str | None = None

	def age_years(self) -> int:
		"""Compute age in whole years as of today."""
		today = date.today()
		return today.year - self.birth_date.year - (
			(today.month, today.day) < (self.birth_date.month, self.birth_date.day)
		)

	def is_paediatric(self) -> bool:
		return self.age_years() < 18

	def is_neonate(self) -> bool:
		return self.age_years() == 0 and (date.today() - self.birth_date).days < 28

	def weight_kg(self) -> float | None:
		"""Placeholder — real impl fetches latest weight vital."""
		return None


# ── encounter ──────────────────────────────────────────────────────────────────

class EncounterCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	encounter_type: EncounterType
	provider_id: str
	location_id: str
	chief_complaint: str
	reason_codes: list[str] = Field(default_factory=list)
	created_by: str

class EncounterUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: EncounterStatus | None = None
	provider_id: str | None = None
	location_id: str | None = None
	icd10_codes: list[str] | None = None
	discharge_summary_id: str | None = None

class EncounterResponse(_Base):
	patient_id: str
	encounter_type: EncounterType
	provider_id: str
	location_id: str
	chief_complaint: str
	reason_codes: list[str] = Field(default_factory=list)
	status: EncounterStatus = EncounterStatus.in_progress
	admit_time: datetime = Field(default_factory=datetime.utcnow)
	discharge_time: datetime | None = None
	discharge_summary_id: str | None = None
	icd10_codes: list[str] = Field(default_factory=list)
	care_team: list[str] = Field(default_factory=list)


# ── diagnosis ──────────────────────────────────────────────────────────────────

class DiagnosisCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	encounter_id: str
	icd10_code: str
	description: str
	certainty: DiagnosisCertainty = DiagnosisCertainty.confirmed
	is_primary: bool = False
	onset_date: date | None = None
	body_site: str | None = None
	laterality: str | None = None
	created_by: str

	@field_validator("icd10_code")
	@classmethod
	def normalise_icd10(cls, v: str) -> str:
		v = v.strip().upper()
		if not v:
			raise ValueError("icd10_code must not be empty")
		return v

class DiagnosisUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	certainty: DiagnosisCertainty | None = None
	description: str | None = None
	is_primary: bool | None = None
	onset_date: date | None = None

class DiagnosisResponse(_Base):
	patient_id: str
	encounter_id: str
	icd10_code: str
	description: str
	certainty: DiagnosisCertainty = DiagnosisCertainty.confirmed
	is_primary: bool = False
	onset_date: date | None = None
	body_site: str | None = None
	laterality: str | None = None
	status: ProblemStatus = ProblemStatus.active


# ── problem list ───────────────────────────────────────────────────────────────

class ProblemCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	icd10_code: str
	description: str
	status: ProblemStatus = ProblemStatus.active
	onset_date: datetime | None = None
	created_by: str

	@field_validator("icd10_code")
	@classmethod
	def icd10_not_empty(cls, v: str) -> str:
		v = v.strip().upper()
		if not v:
			raise ValueError("icd10_code must not be empty")
		return v

class ProblemResponse(_Base):
	patient_id: str
	icd10_code: str
	description: str
	status: ProblemStatus = ProblemStatus.active
	onset_date: datetime | None = None
	resolved_date: datetime | None = None


# ── allergy / intolerance ──────────────────────────────────────────────────────

class AllergyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	allergen: str
	allergy_type: AllergyType
	severity: AllergySeverity
	reaction: str
	onset_date: date | None = None
	notes: str = ""
	created_by: str

class AllergyUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	severity: AllergySeverity | None = None
	reaction: str | None = None
	status: AllergyStatus | None = None
	notes: str | None = None

class AllergyResponse(_Base):
	patient_id: str
	allergen: str
	allergy_type: AllergyType
	severity: AllergySeverity
	reaction: str
	onset_date: date | None = None
	notes: str = ""
	status: AllergyStatus = AllergyStatus.active


# ── medication ─────────────────────────────────────────────────────────────────

class MedicationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	drug_name: str
	ndc_code: str | None = None
	rxnorm_code: str | None = None
	dose: str
	route: str
	frequency: str
	prescriber_id: str
	indication_icd10: str | None = None
	allergy_check_performed: bool = False
	interaction_check_performed: bool = False
	created_by: str

class MedicationResponse(_Base):
	patient_id: str
	drug_name: str
	ndc_code: str | None = None
	rxnorm_code: str | None = None
	dose: str
	route: str
	frequency: str
	prescriber_id: str
	indication_icd10: str | None = None
	status: MedicationStatus = MedicationStatus.active
	allergy_check_performed: bool = False
	interaction_check_performed: bool = False
	start_date: datetime = Field(default_factory=datetime.utcnow)
	end_date: datetime | None = None


# ── prescription ───────────────────────────────────────────────────────────────

class PrescriptionCreate(BaseModel):
	"""Structured prescription with pharmacy workflow fields."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	encounter_id: str
	prescriber_id: str
	drug_name: str
	ndc_code: str | None = None
	rxnorm_code: str | None = None
	# dosing
	dose_quantity: float
	dose_unit: str
	route: str
	frequency: str
	duration_days: int | None = None
	refills_allowed: int = 0
	# paediatric
	dose_per_kg: float | None = None
	# controlled substance
	is_controlled: bool = False
	dea_schedule: str | None = None  # II, III, IV, V
	quantity_dispensed: float | None = None
	# checks
	allergy_check_performed: bool = False
	interaction_check_performed: bool = False
	pregnancy_check_performed: bool = False
	renal_dose_adjusted: bool = False
	hepatic_dose_adjusted: bool = False
	# indication
	indication_icd10: str | None = None
	patient_instructions: str = ""
	pharmacist_notes: str = ""
	created_by: str

	@model_validator(mode="after")
	def controlled_needs_dea_schedule(self) -> "PrescriptionCreate":
		if self.is_controlled and not self.dea_schedule:
			raise ValueError("dea_schedule required for controlled substances")
		return self

class PrescriptionUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: PrescriptionStatus | None = None
	refills_allowed: int | None = None
	patient_instructions: str | None = None
	pharmacist_notes: str | None = None

class PrescriptionResponse(_Base):
	patient_id: str
	encounter_id: str
	prescriber_id: str
	drug_name: str
	ndc_code: str | None = None
	rxnorm_code: str | None = None
	dose_quantity: float
	dose_unit: str
	route: str
	frequency: str
	duration_days: int | None = None
	refills_allowed: int = 0
	refills_used: int = 0
	dose_per_kg: float | None = None
	is_controlled: bool = False
	dea_schedule: str | None = None
	quantity_dispensed: float | None = None
	allergy_check_performed: bool = False
	interaction_check_performed: bool = False
	pregnancy_check_performed: bool = False
	renal_dose_adjusted: bool = False
	hepatic_dose_adjusted: bool = False
	indication_icd10: str | None = None
	patient_instructions: str = ""
	pharmacist_notes: str = ""
	status: PrescriptionStatus = PrescriptionStatus.draft
	dispensed_at: datetime | None = None
	dispensed_by: str | None = None


# ── lab order / result ─────────────────────────────────────────────────────────

class LabOrderCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	encounter_id: str
	ordering_provider_id: str
	test_code: str          # LOINC code
	test_name: str
	specimen_type: str = ""
	priority: str = "routine"   # routine | urgent | stat
	clinical_indication: str = ""
	created_by: str

class LabOrderResponse(_Base):
	patient_id: str
	encounter_id: str
	ordering_provider_id: str
	test_code: str
	test_name: str
	specimen_type: str = ""
	priority: str = "routine"
	clinical_indication: str = ""
	status: LabOrderStatus = LabOrderStatus.requested
	accession_number: str | None = None
	collection_time: datetime | None = None
	received_time: datetime | None = None

class LabResultCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	order_id: str
	patient_id: str
	test_code: str
	test_name: str
	value: str
	value_numeric: float | None = None
	unit: str = ""
	reference_range: str = ""
	flag: LabResultFlag = LabResultFlag.normal
	result_status: LabResultStatus = LabResultStatus.final
	performing_lab: str = ""
	result_time: datetime = Field(default_factory=datetime.utcnow)
	verified_by: str = ""
	created_by: str

class LabResultResponse(_Base):
	order_id: str
	patient_id: str
	test_code: str
	test_name: str
	value: str
	value_numeric: float | None = None
	unit: str = ""
	reference_range: str = ""
	flag: LabResultFlag = LabResultFlag.normal
	result_status: LabResultStatus = LabResultStatus.final
	performing_lab: str = ""
	result_time: datetime
	verified_by: str = ""
	critical_notified: bool = False
	critical_notified_at: datetime | None = None
	critical_notified_to: str | None = None


# ── imaging ────────────────────────────────────────────────────────────────────

class ImagingOrderCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	encounter_id: str
	ordering_provider_id: str
	modality: str       # XR | CT | MRI | US | NM | PET | DEXA | FLUORO
	body_part: str
	laterality: str = ""
	cpt_code: str = ""
	clinical_indication: str = ""
	priority: str = "routine"
	contrast_required: bool = False
	patient_instructions: str = ""
	created_by: str

class ImagingOrderResponse(_Base):
	patient_id: str
	encounter_id: str
	ordering_provider_id: str
	modality: str
	body_part: str
	laterality: str = ""
	cpt_code: str = ""
	clinical_indication: str = ""
	priority: str = "routine"
	contrast_required: bool = False
	patient_instructions: str = ""
	status: ImagingStatus = ImagingStatus.requested
	accession_number: str | None = None
	report_id: str | None = None
	reported_at: datetime | None = None
	radiologist_id: str | None = None
	impression: str | None = None


# ── vital signs ────────────────────────────────────────────────────────────────

class VitalSignCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	encounter_id: str
	vital_type: VitalType
	value: float
	value2: float | None = None   # diastolic for BP
	unit: str
	recorded_by: str
	recorded_at: datetime = Field(default_factory=datetime.utcnow)
	method: str = ""
	position: str = ""   # sitting | standing | supine

	@field_validator("value")
	@classmethod
	def value_non_negative(cls, v: float) -> float:
		if v < 0:
			raise ValueError("vital value must be non-negative")
		return v

class VitalSignResponse(_Base):
	patient_id: str
	encounter_id: str
	vital_type: VitalType
	value: float
	value2: float | None = None
	unit: str
	recorded_by: str
	recorded_at: datetime
	method: str = ""
	position: str = ""


# ── clinical note ──────────────────────────────────────────────────────────────

class ClinicalNoteCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	encounter_id: str
	note_type: NoteType
	author_id: str
	content: str
	subjective: str | None = None
	objective: str | None = None
	assessment: str | None = None
	plan: str | None = None
	icd10_codes: list[str] = Field(default_factory=list)
	# mental health notes require enhanced access controls
	is_sensitive: bool = False

	@field_validator("content")
	@classmethod
	def content_not_empty(cls, v: str) -> str:
		if not v.strip():
			raise ValueError("note content must not be empty")
		return v

class ClinicalNoteUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	content: str | None = None
	subjective: str | None = None
	objective: str | None = None
	assessment: str | None = None
	plan: str | None = None
	icd10_codes: list[str] | None = None

class ClinicalNoteResponse(_Base):
	patient_id: str
	encounter_id: str
	note_type: NoteType
	author_id: str
	content: str
	subjective: str | None = None
	objective: str | None = None
	assessment: str | None = None
	plan: str | None = None
	icd10_codes: list[str] = Field(default_factory=list)
	status: NoteStatus = NoteStatus.draft
	is_sensitive: bool = False
	amendment_of: str | None = None
	cosigned_by: str | None = None
	finalized_at: datetime | None = None


# ── care plan ──────────────────────────────────────────────────────────────────

class CarePlanActivity(BaseModel):
	model_config = ConfigDict(extra="forbid")
	description: str
	kind: str = ""   # medication | procedure | lab | referral | education
	scheduled_date: date | None = None
	status: str = "not_started"
	performed_by: str = ""

class CarePlanCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	encounter_id: str | None = None
	title: str
	description: str = ""
	goal: str = ""
	icd10_codes: list[str] = Field(default_factory=list)
	activities: list[CarePlanActivity] = Field(default_factory=list)
	start_date: date | None = None
	end_date: date | None = None
	care_team: list[str] = Field(default_factory=list)
	created_by: str

class CarePlanUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: CarePlanStatus | None = None
	description: str | None = None
	goal: str | None = None
	activities: list[CarePlanActivity] | None = None
	care_team: list[str] | None = None
	end_date: date | None = None

class CarePlanResponse(_Base):
	patient_id: str
	encounter_id: str | None = None
	title: str
	description: str = ""
	goal: str = ""
	icd10_codes: list[str] = Field(default_factory=list)
	activities: list[CarePlanActivity] = Field(default_factory=list)
	start_date: date | None = None
	end_date: date | None = None
	care_team: list[str] = Field(default_factory=list)
	status: CarePlanStatus = CarePlanStatus.draft


# ── referral ───────────────────────────────────────────────────────────────────

class ReferralCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	encounter_id: str
	referring_provider_id: str
	referred_to_provider_id: str | None = None
	referred_to_specialty: str
	reason: str
	urgency: str = "routine"   # routine | urgent | emergent
	icd10_code: str = ""
	notes: str = ""
	created_by: str

class ReferralUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: ReferralStatus | None = None
	referred_to_provider_id: str | None = None
	appointment_date: date | None = None
	outcome_notes: str | None = None

class ReferralResponse(_Base):
	patient_id: str
	encounter_id: str
	referring_provider_id: str
	referred_to_provider_id: str | None = None
	referred_to_specialty: str
	reason: str
	urgency: str = "routine"
	icd10_code: str = ""
	notes: str = ""
	status: ReferralStatus = ReferralStatus.draft
	appointment_date: date | None = None
	outcome_notes: str = ""


# ── consent ────────────────────────────────────────────────────────────────────

class ConsentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	encounter_id: str | None = None
	scope: str          # treatment | research | privacy | adr
	category: str = ""
	policy_rule: str = ""
	granted_to: list[str] = Field(default_factory=list)
	exceptions: list[str] = Field(default_factory=list)
	start_date: date | None = None
	end_date: date | None = None
	signed_by: str        # patient or guardian ID
	witness_id: str = ""
	notes: str = ""
	created_by: str

class ConsentResponse(_Base):
	patient_id: str
	encounter_id: str | None = None
	scope: str
	category: str = ""
	policy_rule: str = ""
	granted_to: list[str] = Field(default_factory=list)
	exceptions: list[str] = Field(default_factory=list)
	start_date: date | None = None
	end_date: date | None = None
	signed_by: str
	witness_id: str = ""
	notes: str = ""
	status: ConsentStatus = ConsentStatus.proposed
	verified_at: datetime | None = None
	revoked_at: datetime | None = None


# ── immunisation ───────────────────────────────────────────────────────────────

class ImmunisationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	encounter_id: str | None = None
	vaccine_code: str      # CVX code
	vaccine_name: str
	dose_quantity: float | None = None
	dose_unit: str = "mL"
	route: str = ""
	site: str = ""
	lot_number: str = ""
	manufacturer: str = ""
	expiration_date: date | None = None
	administered_date: date
	administered_by: str
	notes: str = ""
	created_by: str

class ImmunisationResponse(_Base):
	patient_id: str
	encounter_id: str | None = None
	vaccine_code: str
	vaccine_name: str
	dose_quantity: float | None = None
	dose_unit: str = "mL"
	route: str = ""
	site: str = ""
	lot_number: str = ""
	manufacturer: str = ""
	expiration_date: date | None = None
	administered_date: date
	administered_by: str
	notes: str = ""
	status: ImmunisationStatus = ImmunisationStatus.completed


# ── family history ─────────────────────────────────────────────────────────────

class FamilyHistoryCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	relationship: str    # father | mother | sibling | child | grandparent | other
	deceased: bool = False
	age_at_death: int | None = None
	conditions: list[str] = Field(default_factory=list)   # ICD-10 codes
	notes: str = ""
	created_by: str

class FamilyHistoryResponse(_Base):
	patient_id: str
	relationship: str
	deceased: bool = False
	age_at_death: int | None = None
	conditions: list[str] = Field(default_factory=list)
	notes: str = ""


# ── clinical decision support alert ───────────────────────────────────────────

class ClinicalAlert(BaseModel):
	"""Non-persistent CDS alert returned by clinical_decision_support()."""
	model_config = ConfigDict(extra="forbid")

	alert_type: AlertType
	severity: str   # info | warning | critical
	title: str
	message: str
	affected_entity_id: str | None = None
	suggested_action: str = ""
	references: list[str] = Field(default_factory=list)
	overridable: bool = True
	override_reason_required: bool = False


# ── drug interaction ───────────────────────────────────────────────────────────

class DrugInteraction(BaseModel):
	model_config = ConfigDict(extra="forbid")

	drug_a: str
	drug_b: str
	severity: DrugInteractionSeverity
	mechanism: str = ""
	clinical_effect: str = ""
	management: str = ""


# ── dedup match ────────────────────────────────────────────────────────────────

class PatientMatchCandidate(BaseModel):
	"""Probabilistic dedup candidate record."""
	model_config = ConfigDict(extra="forbid")

	candidate_patient_id: str
	match_score: float       # 0.0–1.0
	matching_fields: list[str] = Field(default_factory=list)
	is_certain_duplicate: bool = False


# ── reports ────────────────────────────────────────────────────────────────────

class PatientSummaryReport(BaseModel):
	model_config = ConfigDict(extra="forbid")

	patient_id: str
	tenant_id: str
	generated_at: datetime = Field(default_factory=datetime.utcnow)
	active_problems: int
	active_medications: int
	known_allergies: int
	open_encounters: int
	pending_lab_orders: int
	pending_imaging_orders: int
	overdue_immunisations: list[str] = Field(default_factory=list)
	unresolved_alerts: list[ClinicalAlert] = Field(default_factory=list)

class TenantDashboardReport(BaseModel):
	model_config = ConfigDict(extra="forbid")

	tenant_id: str
	generated_at: datetime = Field(default_factory=datetime.utcnow)
	total_patients: int
	active_encounters: int
	notes_today: int
	critical_lab_results_unnotified: int
	controlled_substance_prescriptions_today: int
	pending_referrals: int
	total_problems: int
	total_medications: int
