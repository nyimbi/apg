"""Pydantic v2 models for APG Pharmacy Management.

All models enforce strict extra='forbid', support alias-based binding,
and use uuid7str for IDs. Enums capture every valid lifecycle state.
Entities: Drug, Prescription, DispensedMedication, DrugInteraction,
          DrugInventory, NarcoticsRegister, ColdChainRecord, ExpiryTracking,
          ReturnedMedication, PriorAuthorization, CounsellingChecklist,
          ReorderRequest, ControlledSubstanceLog.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


# ── Enums ─────────────────────────────────────────────────────────────────────

class DrugType(str, Enum):
	BRAND = "brand"
	GENERIC = "generic"
	BIOSIMILAR = "biosimilar"
	OTC = "otc"
	COMPOUNDED = "compounded"
	INVESTIGATIONAL = "investigational"
	VACCINE = "vaccine"
	BLOOD_PRODUCT = "blood_product"


class DrugSchedule(str, Enum):
	SCHEDULE_I = "schedule_i"
	SCHEDULE_II = "schedule_ii"
	SCHEDULE_III = "schedule_iii"
	SCHEDULE_IV = "schedule_iv"
	SCHEDULE_V = "schedule_v"
	NON_CONTROLLED = "non_controlled"


class DosageForm(str, Enum):
	TABLET = "tablet"
	CAPSULE = "capsule"
	LIQUID = "liquid"
	INJECTION = "injection"
	PATCH = "patch"
	INHALER = "inhaler"
	SUPPOSITORY = "suppository"
	CREAM = "cream"
	OINTMENT = "ointment"
	DROPS = "drops"
	INFUSION = "infusion"
	POWDER = "powder"
	GEL = "gel"
	SPRAY = "spray"
	LOZENGE = "lozenge"


class FormularyStatus(str, Enum):
	PREFERRED = "preferred"
	NON_PREFERRED = "non_preferred"
	NON_FORMULARY = "non_formulary"
	PRIOR_AUTH_REQUIRED = "prior_auth_required"
	STEP_THERAPY = "step_therapy"


class DispenseStatus(str, Enum):
	PENDING = "pending"
	VERIFIED = "verified"
	DISPENSED = "dispensed"
	PICKED_UP = "picked_up"
	RETURNED = "returned"
	CANCELLED = "cancelled"
	ON_HOLD = "on_hold"


class InteractionSeverity(str, Enum):
	CONTRAINDICATED = "contraindicated"
	MAJOR = "major"
	MODERATE = "moderate"
	MINOR = "minor"
	INFORMATIONAL = "informational"


class ControlledSubstanceAction(str, Enum):
	DISPENSE = "dispense"
	WASTE = "waste"
	DESTROY = "destroy"
	COUNT = "count"
	TRANSFER = "transfer"
	RECEIVE = "receive"


class InventoryStatus(str, Enum):
	IN_STOCK = "in_stock"
	LOW_STOCK = "low_stock"
	OUT_OF_STOCK = "out_of_stock"
	ON_ORDER = "on_order"
	RECALLED = "recalled"
	EXPIRED = "expired"
	QUARANTINED = "quarantined"


class LasaAlertType(str, Enum):
	LOOK_ALIKE = "look_alike"
	SOUND_ALIKE = "sound_alike"
	LOOK_AND_SOUND_ALIKE = "look_and_sound_alike"


class PriorAuthStatus(str, Enum):
	PENDING = "pending"
	APPROVED = "approved"
	DENIED = "denied"
	EXPIRED = "expired"
	WITHDRAWN = "withdrawn"


class ReturnReason(str, Enum):
	ADVERSE_REACTION = "adverse_reaction"
	PATIENT_REFUSED = "patient_refused"
	WRONG_MEDICATION = "wrong_medication"
	EXPIRED = "expired"
	DISPENSING_ERROR = "dispensing_error"
	THERAPY_CHANGE = "therapy_change"
	PATIENT_DECEASED = "patient_deceased"


class ReturnDisposition(str, Enum):
	DESTROY = "destroy"
	RESTOCK = "restock"
	QUARANTINE = "quarantine"
	RETURN_TO_MANUFACTURER = "return_to_manufacturer"


class ColdChainStatus(str, Enum):
	COMPLIANT = "compliant"
	EXCURSION = "excursion"
	CRITICAL = "critical"
	QUARANTINED = "quarantined"


class PrescriptionStatus(str, Enum):
	RECEIVED = "received"
	VERIFIED = "verified"
	IN_PROGRESS = "in_progress"
	READY = "ready"
	DISPENSED = "dispensed"
	EXPIRED = "expired"
	CANCELLED = "cancelled"
	ON_HOLD = "on_hold"


class NarcoticsRegisterAction(str, Enum):
	RECEIPT = "receipt"
	DISPENSE = "dispense"
	WASTE = "waste"
	DESTROY = "destroy"
	TRANSFER = "transfer"
	AUDIT = "audit"
	DISCREPANCY = "discrepancy"


class ReorderStatus(str, Enum):
	PENDING = "pending"
	SUBMITTED = "submitted"
	ACKNOWLEDGED = "acknowledged"
	RECEIVED = "received"
	CANCELLED = "cancelled"
	PARTIAL = "partial"


class ExpiryAlertLevel(str, Enum):
	EXPIRED = "expired"
	CRITICAL = "critical"   # < 7 days
	WARNING = "warning"     # < 30 days
	NOTICE = "notice"       # < 90 days
	OK = "ok"


class Urgency(str, Enum):
	ROUTINE = "routine"
	URGENT = "urgent"
	STAT = "stat"


class ReorderTrigger(str, Enum):
	MANUAL = "manual"
	AUTO_REORDER = "auto_reorder"


# ── Validators ─────────────────────────────────────────────────────────────────

def _non_empty(v: str) -> str:
	if not v or not v.strip():
		raise ValueError("field must not be empty")
	return v.strip()


NonEmpty = Annotated[str, AfterValidator(_non_empty)]


# ── Base ───────────────────────────────────────────────────────────────────────

class PhaBase(BaseModel):
	"""Shared audit fields for all pharmacy entities."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmpty
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmpty
	is_deleted: bool = False


# ── Drug / Formulary ───────────────────────────────────────────────────────────

class DrugCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	drug_name: NonEmpty
	generic_name: NonEmpty
	ndc_code: NonEmpty
	rxnorm_code: str | None = None
	atc_code: str | None = None
	drug_type: DrugType
	drug_schedule: DrugSchedule
	dosage_form: DosageForm
	strength: NonEmpty
	unit: NonEmpty
	route_of_administration: str = "oral"
	manufacturer: NonEmpty
	formulary_status: FormularyStatus = FormularyStatus.PREFERRED
	requires_refrigeration: bool = False
	is_hazardous: bool = False
	tall_man_name: str | None = None
	therapeutic_class: str | None = None
	sub_therapeutic_class: str | None = None
	black_box_warning: bool = False
	black_box_text: str | None = None
	storage_conditions: str | None = None
	created_by: NonEmpty


class DrugUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	drug_name: str | None = None
	formulary_status: FormularyStatus | None = None
	requires_refrigeration: bool | None = None
	is_hazardous: bool | None = None
	tall_man_name: str | None = None
	black_box_warning: bool | None = None
	black_box_text: str | None = None
	storage_conditions: str | None = None
	updated_by: NonEmpty


class DrugResponse(PhaBase):
	drug_name: str
	generic_name: str
	ndc_code: str
	rxnorm_code: str | None = None
	atc_code: str | None = None
	drug_type: DrugType
	drug_schedule: DrugSchedule
	dosage_form: DosageForm
	strength: str
	unit: str
	route_of_administration: str = "oral"
	manufacturer: str
	formulary_status: FormularyStatus = FormularyStatus.PREFERRED
	requires_refrigeration: bool = False
	is_hazardous: bool = False
	is_lasa: bool = False
	lasa_pair: str | None = None
	lasa_alert_type: LasaAlertType | None = None
	tall_man_name: str | None = None
	therapeutic_class: str | None = None
	sub_therapeutic_class: str | None = None
	black_box_warning: bool = False
	black_box_text: str | None = None
	storage_conditions: str | None = None


# ── Prescription ───────────────────────────────────────────────────────────────

class PrescriptionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	patient_id: NonEmpty
	prescriber_id: NonEmpty
	prescriber_npi: NonEmpty
	drug_id: NonEmpty
	drug_name: NonEmpty
	dosage_form: DosageForm
	strength: NonEmpty
	quantity: float
	unit: NonEmpty
	days_supply: int
	sig: NonEmpty
	refills_authorized: int = 0
	refills_remaining: int = 0
	diagnosis_icd10: str | None = None
	dea_number: str | None = None
	is_controlled: bool = False
	formulary_override_reason: str | None = None
	electronic: bool = True
	prescriber_signature_ref: str | None = None
	created_by: NonEmpty

	@field_validator("quantity")
	@classmethod
	def qty_positive(cls, v: float) -> float:
		if v <= 0:
			raise ValueError("quantity must be positive")
		return v

	@field_validator("days_supply")
	@classmethod
	def days_positive(cls, v: int) -> int:
		if v <= 0:
			raise ValueError("days_supply must be positive")
		return v

	@field_validator("refills_authorized", "refills_remaining")
	@classmethod
	def refills_non_negative(cls, v: int) -> int:
		if v < 0:
			raise ValueError("refills cannot be negative")
		return v


class PrescriptionUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: PrescriptionStatus | None = None
	refills_remaining: int | None = None
	formulary_override_reason: str | None = None
	updated_by: NonEmpty


class PrescriptionResponse(PhaBase):
	patient_id: str
	prescriber_id: str
	prescriber_npi: str
	drug_id: str
	drug_name: str
	dosage_form: DosageForm
	strength: str
	quantity: float
	unit: str
	days_supply: int
	sig: str
	refills_authorized: int = 0
	refills_remaining: int = 0
	diagnosis_icd10: str | None = None
	dea_number: str | None = None
	is_controlled: bool = False
	status: PrescriptionStatus = PrescriptionStatus.RECEIVED
	formulary_override_reason: str | None = None
	electronic: bool = True
	prescriber_signature_ref: str | None = None
	dispensed_at: datetime | None = None
	expires_at: datetime | None = None
	verified_by: str | None = None
	verified_at: datetime | None = None


# ── DispensedMedication ────────────────────────────────────────────────────────

class DispenseOrderCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	patient_id: NonEmpty
	drug_id: NonEmpty
	prescription_id: NonEmpty
	inventory_item_id: str | None = None
	quantity: float
	unit: NonEmpty
	pharmacist_verified: bool = False
	interaction_severity: InteractionSeverity | None = None
	drug_inventory_status: InventoryStatus = InventoryStatus.IN_STOCK
	formulary_status: FormularyStatus = FormularyStatus.PREFERRED
	prior_auth_approved: bool = True
	formulary_override_present: bool = False
	step_therapy_completed: bool = True
	counselling_completed: bool = False
	label_printed: bool = False
	barcode_scanned: bool = False
	override_reason: str | None = None
	created_by: NonEmpty

	@field_validator("quantity")
	@classmethod
	def quantity_positive(cls, v: float) -> float:
		if v <= 0:
			raise ValueError("quantity must be positive")
		return v


class DispenseOrderUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: DispenseStatus | None = None
	counselling_completed: bool | None = None
	label_printed: bool | None = None
	barcode_scanned: bool | None = None
	override_reason: str | None = None
	updated_by: NonEmpty


class DispenseOrderResponse(PhaBase):
	patient_id: str
	drug_id: str
	prescription_id: str
	inventory_item_id: str | None = None
	quantity: float
	unit: str
	status: DispenseStatus = DispenseStatus.PENDING
	pharmacist_verified: bool = False
	pharmacist_id: str | None = None
	verified_at: datetime | None = None
	dispensed_at: datetime | None = None
	picked_up_at: datetime | None = None
	counselling_completed: bool = False
	label_printed: bool = False
	barcode_scanned: bool = False
	interaction_severity: InteractionSeverity | None = None
	formulary_status: FormularyStatus = FormularyStatus.PREFERRED
	override_reason: str | None = None


# ── DrugInteraction ────────────────────────────────────────────────────────────

class DrugInteractionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	drug_a_id: NonEmpty
	drug_b_id: NonEmpty
	severity: InteractionSeverity
	mechanism: NonEmpty
	clinical_effect: NonEmpty
	management: NonEmpty
	evidence_source: NonEmpty
	onset: str | None = None
	documentation_level: str | None = None
	created_by: NonEmpty

	@model_validator(mode="after")
	def drugs_distinct(self) -> "DrugInteractionCreate":
		if self.drug_a_id == self.drug_b_id:
			raise ValueError("drug_a_id and drug_b_id must be different")
		return self


class DrugInteractionUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	severity: InteractionSeverity | None = None
	management: str | None = None
	documentation_level: str | None = None
	updated_by: NonEmpty


class DrugInteractionResponse(PhaBase):
	drug_a_id: str
	drug_b_id: str
	severity: InteractionSeverity
	mechanism: str
	clinical_effect: str
	management: str
	evidence_source: str
	onset: str | None = None
	documentation_level: str | None = None


# ── DrugInventory ──────────────────────────────────────────────────────────────

class InventoryItemCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	drug_id: NonEmpty
	lot_number: NonEmpty
	quantity_on_hand: float
	reorder_point: float = 0.0
	reorder_quantity: float = 0.0
	unit: NonEmpty
	expiry_date: datetime
	location: NonEmpty
	storage_temperature_min_c: float | None = None
	storage_temperature_max_c: float | None = None
	supplier_id: str | None = None
	purchase_price: float | None = None
	created_by: NonEmpty

	@field_validator("quantity_on_hand")
	@classmethod
	def qty_non_negative(cls, v: float) -> float:
		if v < 0:
			raise ValueError("quantity_on_hand cannot be negative")
		return v

	@field_validator("purchase_price")
	@classmethod
	def price_non_negative(cls, v: float | None) -> float | None:
		if v is not None and v < 0:
			raise ValueError("purchase_price cannot be negative")
		return v


class InventoryItemUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	quantity_on_hand: float | None = None
	status: InventoryStatus | None = None
	location: str | None = None
	reorder_point: float | None = None
	reorder_quantity: float | None = None
	updated_by: NonEmpty


class InventoryItemResponse(PhaBase):
	drug_id: str
	lot_number: str
	quantity_on_hand: float
	reorder_point: float = 0.0
	reorder_quantity: float = 0.0
	unit: str
	expiry_date: datetime
	location: str
	status: InventoryStatus = InventoryStatus.IN_STOCK
	days_remaining: int = 0
	storage_temperature_min_c: float | None = None
	storage_temperature_max_c: float | None = None
	supplier_id: str | None = None
	purchase_price: float | None = None
	is_below_reorder_point: bool = False


# ── NarcoticsRegister ──────────────────────────────────────────────────────────

class NarcoticsRegisterEntryCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	drug_id: NonEmpty
	drug_name: NonEmpty
	drug_schedule: DrugSchedule
	action: NarcoticsRegisterAction
	quantity: float
	unit: NonEmpty
	balance_before: float
	balance_after: float
	patient_id: str | None = None
	prescription_id: str | None = None
	dispense_order_id: str | None = None
	performed_by: NonEmpty
	witness_id: str | None = None
	witness_signature_ref: str | None = None
	notes: str = ""
	discrepancy_amount: float | None = None
	discrepancy_reason: str | None = None
	created_by: NonEmpty

	@field_validator("quantity")
	@classmethod
	def qty_positive(cls, v: float) -> float:
		if v <= 0:
			raise ValueError("quantity must be positive")
		return v

	@field_validator("balance_after")
	@classmethod
	def balance_non_negative(cls, v: float) -> float:
		if v < 0:
			raise ValueError("balance_after cannot be negative")
		return v

	@model_validator(mode="after")
	def waste_requires_witness(self) -> "NarcoticsRegisterEntryCreate":
		if self.action == NarcoticsRegisterAction.WASTE and not self.witness_id:
			raise ValueError("witness_id required for waste actions")
		return self


class NarcoticsRegisterEntryResponse(PhaBase):
	drug_id: str
	drug_name: str
	drug_schedule: DrugSchedule
	action: NarcoticsRegisterAction
	quantity: float
	unit: str
	balance_before: float
	balance_after: float
	patient_id: str | None = None
	prescription_id: str | None = None
	dispense_order_id: str | None = None
	performed_by: str
	witness_id: str | None = None
	witness_signature_ref: str | None = None
	notes: str = ""
	discrepancy_amount: float | None = None
	discrepancy_reason: str | None = None


# ── ColdChainRecord ────────────────────────────────────────────────────────────

class ColdChainRecordCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	inventory_item_id: NonEmpty
	drug_id: NonEmpty
	recorded_temperature_c: float
	min_acceptable_c: float
	max_acceptable_c: float
	location: NonEmpty
	sensor_id: str | None = None
	excursion_duration_minutes: int | None = None
	corrective_action: str | None = None
	created_by: NonEmpty

	@model_validator(mode="after")
	def temp_range_valid(self) -> "ColdChainRecordCreate":
		if self.min_acceptable_c >= self.max_acceptable_c:
			raise ValueError("min_acceptable_c must be less than max_acceptable_c")
		return self


class ColdChainRecordResponse(PhaBase):
	inventory_item_id: str
	drug_id: str
	recorded_temperature_c: float
	min_acceptable_c: float
	max_acceptable_c: float
	location: str
	sensor_id: str | None = None
	status: ColdChainStatus = ColdChainStatus.COMPLIANT
	excursion_duration_minutes: int | None = None
	corrective_action: str | None = None
	deviation_c: float = 0.0  # signed deviation from nearest bound


# ── ExpiryTracking ─────────────────────────────────────────────────────────────

class ExpiryAlertResponse(BaseModel):
	"""Aggregated expiry alert — not a persisted entity."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	inventory_item_id: str
	drug_id: str
	drug_name: str
	lot_number: str
	quantity_on_hand: float
	unit: str
	expiry_date: datetime
	days_remaining: int
	alert_level: ExpiryAlertLevel
	location: str
	storage_temperature_min_c: float | None = None
	storage_temperature_max_c: float | None = None


class ExpiryCheckResult(BaseModel):
	"""Summary result of a bulk expiry check."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	checked_at: datetime = Field(default_factory=datetime.utcnow)
	total_items_checked: int
	expired_count: int
	critical_count: int
	warning_count: int
	notice_count: int
	ok_count: int
	alerts: list[ExpiryAlertResponse] = Field(default_factory=list)


# ── ReturnedMedication ─────────────────────────────────────────────────────────

class ReturnedMedicationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	patient_id: NonEmpty
	drug_id: NonEmpty
	dispense_order_id: NonEmpty
	prescription_id: NonEmpty
	quantity_returned: float
	unit: NonEmpty
	return_reason: ReturnReason
	condition: str = "intact"
	return_disposition: ReturnDisposition = ReturnDisposition.DESTROY
	returned_by: NonEmpty
	received_by: NonEmpty
	notes: str = ""
	created_by: NonEmpty

	@field_validator("quantity_returned")
	@classmethod
	def qty_positive(cls, v: float) -> float:
		if v <= 0:
			raise ValueError("quantity_returned must be positive")
		return v

	@field_validator("condition")
	@classmethod
	def condition_valid(cls, v: str) -> str:
		if v not in ("intact", "damaged", "partial", "unknown"):
			raise ValueError("condition must be intact/damaged/partial/unknown")
		return v


class ReturnedMedicationUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	processed: bool | None = None
	processed_by: str | None = None
	return_disposition: ReturnDisposition | None = None
	notes: str | None = None
	updated_by: NonEmpty


class ReturnedMedicationResponse(PhaBase):
	patient_id: str
	drug_id: str
	dispense_order_id: str
	prescription_id: str
	quantity_returned: float
	unit: str
	return_reason: ReturnReason
	condition: str = "intact"
	return_disposition: ReturnDisposition = ReturnDisposition.DESTROY
	returned_by: str
	received_by: str
	notes: str = ""
	processed: bool = False
	processed_at: datetime | None = None
	processed_by: str | None = None


# ── ControlledSubstanceLog ─────────────────────────────────────────────────────

class ControlledSubstanceLogCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	drug_id: NonEmpty
	drug_schedule: DrugSchedule
	action: ControlledSubstanceAction
	quantity: float
	unit: NonEmpty
	patient_id: str | None = None
	performed_by: NonEmpty
	witness_id: str | None = None
	waste_amount: float | None = None
	notes: str = ""
	created_by: NonEmpty

	@field_validator("quantity")
	@classmethod
	def qty_positive(cls, v: float) -> float:
		if v <= 0:
			raise ValueError("quantity must be positive")
		return v


class ControlledSubstanceLogResponse(PhaBase):
	drug_id: str
	drug_schedule: DrugSchedule
	action: ControlledSubstanceAction
	quantity: float
	unit: str
	patient_id: str | None = None
	performed_by: str
	witness_id: str | None = None
	waste_amount: float | None = None
	notes: str = ""


# ── Prior Authorization ────────────────────────────────────────────────────────

class PriorAuthCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	patient_id: NonEmpty
	drug_id: NonEmpty
	prescription_id: NonEmpty
	insurance_id: NonEmpty
	diagnosis_icd10: NonEmpty
	requested_by: NonEmpty
	clinical_justification: NonEmpty
	supporting_documents: list[str] = Field(default_factory=list)
	created_by: NonEmpty


class PriorAuthUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: PriorAuthStatus | None = None
	denial_reason: str | None = None
	updated_by: NonEmpty


class PriorAuthResponse(PhaBase):
	patient_id: str
	drug_id: str
	prescription_id: str
	insurance_id: str
	diagnosis_icd10: str
	requested_by: str
	clinical_justification: str
	supporting_documents: list[str] = Field(default_factory=list)
	status: PriorAuthStatus = PriorAuthStatus.PENDING
	decision_by: str | None = None
	decision_at: datetime | None = None
	denial_reason: str | None = None
	expires_at: datetime | None = None


# ── Automated Reorder ──────────────────────────────────────────────────────────

class ReorderRequestCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	drug_id: NonEmpty
	inventory_item_id: NonEmpty
	quantity_requested: float
	unit: NonEmpty
	supplier_id: str | None = None
	urgency: Urgency = Urgency.ROUTINE
	triggered_by: ReorderTrigger = ReorderTrigger.MANUAL
	created_by: NonEmpty

	@field_validator("quantity_requested")
	@classmethod
	def qty_positive(cls, v: float) -> float:
		if v <= 0:
			raise ValueError("quantity_requested must be positive")
		return v


class ReorderRequestUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: ReorderStatus | None = None
	quantity_received: float | None = None
	updated_by: NonEmpty


class ReorderRequestResponse(PhaBase):
	drug_id: str
	inventory_item_id: str
	quantity_requested: float
	unit: str
	supplier_id: str | None = None
	urgency: Urgency = Urgency.ROUTINE
	triggered_by: ReorderTrigger = ReorderTrigger.MANUAL
	status: ReorderStatus = ReorderStatus.PENDING
	submitted_at: datetime | None = None
	acknowledged_at: datetime | None = None
	received_at: datetime | None = None
	quantity_received: float | None = None


# ── Counselling Checklist ──────────────────────────────────────────────────────

class CounsellingChecklistCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmpty
	patient_id: NonEmpty
	dispense_order_id: NonEmpty
	drug_id: NonEmpty
	indication_explained: bool = False
	dosage_explained: bool = False
	administration_explained: bool = False
	side_effects_explained: bool = False
	interactions_explained: bool = False
	storage_explained: bool = False
	missed_dose_explained: bool = False
	patient_questions_addressed: bool = False
	patient_understood: bool = False
	interpreter_used: bool = False
	language: str = "en"
	pharmacist_id: NonEmpty
	created_by: NonEmpty


class CounsellingChecklistUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	indication_explained: bool | None = None
	dosage_explained: bool | None = None
	administration_explained: bool | None = None
	side_effects_explained: bool | None = None
	interactions_explained: bool | None = None
	storage_explained: bool | None = None
	missed_dose_explained: bool | None = None
	patient_questions_addressed: bool | None = None
	patient_understood: bool | None = None
	interpreter_used: bool | None = None
	updated_by: NonEmpty


class CounsellingChecklistResponse(PhaBase):
	patient_id: str
	dispense_order_id: str
	drug_id: str
	indication_explained: bool = False
	dosage_explained: bool = False
	administration_explained: bool = False
	side_effects_explained: bool = False
	interactions_explained: bool = False
	storage_explained: bool = False
	missed_dose_explained: bool = False
	patient_questions_addressed: bool = False
	patient_understood: bool = False
	interpreter_used: bool = False
	language: str = "en"
	pharmacist_id: str
	completion_score: float = 0.0


# ── Drug Substitution Result ───────────────────────────────────────────────────

class DrugSubstituteResult(BaseModel):
	"""Result of a generic substitution search."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	original_drug_id: str
	substitute_found: bool
	substitute: DrugResponse | None = None
	reason: str = ""
	savings_estimate: float | None = None


# ── Report Models ──────────────────────────────────────────────────────────────

class DispensingSummaryReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	period_start: datetime
	period_end: datetime
	total_dispenses: int
	pending: int
	verified: int
	dispensed: int
	picked_up: int
	returned: int
	cancelled: int
	top_drugs: list[dict[str, Any]] = Field(default_factory=list)
	avg_verification_time_minutes: float | None = None
	counselling_completion_rate: float = 0.0


class InventoryValuationReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	as_of: datetime
	total_items: int
	total_value: float
	in_stock_value: float
	low_stock_count: int
	expired_count: int
	recalled_count: int
	quarantined_count: int
	expiring_within_30_days: int
	below_reorder_point: int


class NarcoticsAuditReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	period_start: datetime
	period_end: datetime
	total_entries: int
	discrepancies_found: int
	drugs_audited: list[str]
	entries_by_action: dict[str, int]
	witness_compliance_rate: float


class ColdChainReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	period_start: datetime
	period_end: datetime
	total_readings: int
	compliant: int
	excursions: int
	critical: int
	quarantined: int
	compliance_rate: float
	affected_drugs: list[str] = Field(default_factory=list)


class PharmacyDashboard(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	as_of: datetime
	formulary: dict[str, Any]
	dispensing: dict[str, Any]
	inventory: dict[str, Any]
	prior_auth: dict[str, Any]
	controlled_substances: dict[str, Any]
	cold_chain: dict[str, Any]
	counselling: dict[str, Any]
	narcotics: dict[str, Any]
	returns: dict[str, Any]
	reorders: dict[str, Any]
	alerts: list[str]
