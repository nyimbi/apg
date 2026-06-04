"""Pydantic v2 models for APG Pharma Clinical Trials Management.

Entities:
	ClinicalTrial, TrialSite, TrialPatient (Subject), RandomisationRecord,
	TrialProtocol, ProtocolAmendment, AdverseEvent, SeriousAdverseEvent,
	CRFForm, DataEntry, DataQuery, MonitoringVisit, Inspection, TMFDocument,
	RegulatorySubmission, IRBApproval

All IDs are UUID7 strings. All timestamps are UTC datetimes.
Tenant isolation is encoded in every record via tenant_id.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from uuid6 import uuid7


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _uuid7str() -> str:
	return str(uuid7())


def _utcnow() -> datetime:
	return datetime.utcnow()


# ─────────────────────────────────────────────────────────────────────────────
# Enums — status lifecycles
# ─────────────────────────────────────────────────────────────────────────────

class TrialPhase(str, Enum):
	PHASE_1          = "phase_1"
	PHASE_1B         = "phase_1b"
	PHASE_2          = "phase_2"
	PHASE_2B         = "phase_2b"
	PHASE_3          = "phase_3"
	PHASE_3B         = "phase_3b"
	PHASE_4          = "phase_4"
	EXPANDED_ACCESS  = "expanded_access"
	OBSERVATIONAL    = "observational"


class TrialType(str, Enum):
	INTERVENTIONAL  = "interventional"
	OBSERVATIONAL   = "observational"
	EXPANDED_ACCESS = "expanded_access"
	REGISTRY        = "registry"
	BIOEQUIVALENCE  = "bioequivalence"
	FIRST_IN_HUMAN  = "first_in_human"
	BASKET          = "basket"
	UMBRELLA        = "umbrella"


class TrialStatus(str, Enum):
	PLANNED             = "planned"
	ACTIVE              = "active"
	ENROLLING           = "enrolling"
	ENROLLMENT_COMPLETE = "enrollment_complete"
	TREATMENT_ONGOING   = "treatment_ongoing"
	FOLLOW_UP           = "follow_up"
	COMPLETED           = "completed"
	TERMINATED          = "terminated"
	SUSPENDED           = "suspended"
	WITHDRAWN           = "withdrawn"


class BlindingType(str, Enum):
	OPEN_LABEL   = "open_label"
	SINGLE_BLIND = "single_blind"
	DOUBLE_BLIND = "double_blind"
	TRIPLE_BLIND = "triple_blind"


class SiteStatus(str, Enum):
	PRE_SELECTED        = "pre_selected"
	SELECTED            = "selected"
	INITIATED           = "initiated"
	ENROLLING           = "enrolling"
	ENROLLMENT_COMPLETE = "enrollment_complete"
	CLOSED              = "closed"
	TERMINATED          = "terminated"
	WITHDRAWN           = "withdrawn"


class SubjectStatus(str, Enum):
	SCREENED          = "screened"
	ENROLLED          = "enrolled"
	RANDOMISED        = "randomised"
	ON_TREATMENT      = "on_treatment"
	COMPLETED         = "completed"
	WITHDRAWN         = "withdrawn"
	LOST_TO_FOLLOW_UP = "lost_to_follow_up"
	SCREEN_FAILURE    = "screen_failure"


class RandomisationMethod(str, Enum):
	SIMPLE      = "simple"
	STRATIFIED  = "stratified"
	BLOCK       = "block"
	ADAPTIVE    = "adaptive"
	MINIMISATION = "minimisation"
	DYNAMIC     = "dynamic"


class ProtocolStatus(str, Enum):
	DRAFT         = "draft"
	UNDER_REVIEW  = "under_review"
	APPROVED      = "approved"
	AMENDED       = "amended"
	SUPERSEDED    = "superseded"
	WITHDRAWN     = "withdrawn"


class AmendmentType(str, Enum):
	SUBSTANTIAL     = "substantial"
	NON_SUBSTANTIAL = "non_substantial"
	URGENT_SAFETY   = "urgent_safety"
	ADMINISTRATIVE  = "administrative"


class AmendmentStatus(str, Enum):
	DRAFT    = "draft"
	PENDING  = "pending_irb_review"
	APPROVED = "approved"
	REJECTED = "rejected"
	WITHDRAWN = "withdrawn"


class AEType(str, Enum):
	ADVERSE_EVENT   = "adverse_event"
	SAE             = "serious_adverse_event"
	SUSAR           = "suspected_unexpected_serious_adverse_reaction"
	DISEASE_RELATED = "disease_related_event"
	PROTOCOL_DEV    = "protocol_deviation"


class AESeverity(str, Enum):
	GRADE_1 = "grade_1"
	GRADE_2 = "grade_2"
	GRADE_3 = "grade_3"
	GRADE_4 = "grade_4"
	GRADE_5 = "grade_5"  # Fatal


class AEOutcome(str, Enum):
	RECOVERED  = "recovered"
	RECOVERING = "recovering"
	NOT_RECOVERED = "not_recovered"
	SEQUELAE   = "recovered_with_sequelae"
	FATAL      = "fatal"
	UNKNOWN    = "unknown"


class AECausality(str, Enum):
	UNRELATED     = "unrelated"
	UNLIKELY      = "unlikely"
	POSSIBLE      = "possible"
	PROBABLE      = "probable"
	DEFINITE      = "definite"
	NOT_ASSESSABLE = "not_assessable"


class CRFStatus(str, Enum):
	DRAFT          = "draft"
	PENDING_QUERY  = "pending_query"
	QUERY_RESOLVED = "query_resolved"
	SIGNED_OFF     = "signed_off"
	LOCKED         = "locked"
	UNLOCKED       = "unlocked"


class QueryStatus(str, Enum):
	OPEN      = "open"
	ANSWERED  = "answered"
	CLOSED    = "closed"
	CANCELLED = "cancelled"


class MonitoringVisitType(str, Enum):
	QUALIFICATION = "qualification"
	INITIATION    = "initiation"
	ROUTINE       = "routine"
	CLOSE_OUT     = "close_out"
	FOR_CAUSE     = "for_cause"
	RISK_BASED    = "risk_based"


class InspectionType(str, Enum):
	GCP     = "gcp"
	GLP     = "glp"
	SPONSOR = "sponsor_audit"
	CRO     = "cro_audit"
	SITE    = "site_audit"


class SubmissionType(str, Enum):
	IND                = "ind"
	CTA                = "cta"
	PROTOCOL_AMENDMENT = "protocol_amendment"
	ANNUAL_REPORT      = "annual_report"
	SAFETY_REPORT      = "safety_report"
	FINAL_REPORT       = "final_report"
	EUDRACT            = "eudract"
	CTIS               = "ctis"


class SubmissionStatus(str, Enum):
	NOT_SUBMITTED = "not_submitted"
	SUBMITTED     = "submitted"
	ACKNOWLEDGED  = "acknowledged"
	UNDER_REVIEW  = "under_review"
	APPROVED      = "approved"
	REJECTED      = "rejected"
	WITHDRAWN     = "withdrawn"


class RegulatoryAuthority(str, Enum):
	FDA          = "fda"
	EMA          = "ema"
	MHRA         = "mhra"
	PMDA         = "pmda"
	HEALTH_CANADA = "health_canada"
	TGA          = "tga"
	ANVISA       = "anvisa"
	CDSCO        = "cdsco"
	NMPA         = "nmpa"
	NMRA         = "nmra"


class IRBDecision(str, Enum):
	APPROVED                 = "approved"
	APPROVED_WITH_CONDITIONS = "approved_with_conditions"
	REQUIRES_MODIFICATION    = "requires_modification"
	REJECTED                 = "rejected"
	WITHDRAWN                = "withdrawn"


class TMFDocumentStatus(str, Enum):
	EXPECTED       = "expected"
	FILED          = "filed"
	OVERDUE        = "overdue"
	NOT_APPLICABLE = "not_applicable"
	ARCHIVED       = "archived"


class InspectionOutcome(str, Enum):
	NO_FINDINGS       = "no_findings"
	MINOR_FINDINGS    = "minor_findings"
	MAJOR_FINDINGS    = "major_findings"
	CRITICAL_FINDINGS = "critical_findings"
	WARNING_LETTER    = "warning_letter"


class ProtocolDeviationType(str, Enum):
	IMPORTANT     = "important"
	NON_IMPORTANT = "non_important"
	MAJOR         = "major"
	MINOR         = "minor"


class ProtocolDeviationImpact(str, Enum):
	SAFETY_IMPACT          = "safety_impact"
	DATA_INTEGRITY_IMPACT  = "data_integrity_impact"
	NO_IMPACT              = "no_impact"


# ─────────────────────────────────────────────────────────────────────────────
# Base models
# ─────────────────────────────────────────────────────────────────────────────

class CtrBase(BaseModel):
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
		use_enum_values=True,
	)


class CtrRecord(CtrBase):
	"""Base for all persistent records — provides the standard audit columns."""
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str = Field(min_length=1)
	created_at: datetime = Field(default_factory=_utcnow)
	updated_at: datetime = Field(default_factory=_utcnow)
	created_by: str = Field(min_length=1)
	is_deleted: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# ClinicalTrial
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalTrial(CtrRecord):
	"""Master record for a clinical trial (IND/CTA holder perspective)."""
	trial_number: str = Field(min_length=1)
	eudract_number: str | None = None        # EU Clinical Trials Register
	nct_number: str | None = None            # ClinicalTrials.gov NCT identifier
	ctis_number: str | None = None           # EU CTIS post-2023
	phase: TrialPhase
	trial_type: TrialType
	title: str = Field(min_length=1)
	short_title: str | None = None
	sponsor_id: str = Field(min_length=1)
	cro_id: str | None = None
	medical_monitor_id: str | None = None
	blinding: BlindingType
	status: TrialStatus = TrialStatus.PLANNED
	indication: str = Field(min_length=1)
	therapeutic_area: str | None = None
	icd10_code: str | None = None
	investigational_product: str | None = None
	comparator: str | None = None
	target_enrollment: int = Field(default=0, ge=0)
	planned_start_date: datetime | None = None
	planned_end_date: datetime | None = None
	actual_start_date: datetime | None = None
	actual_end_date: datetime | None = None
	irb_approval_reference: str | None = None
	protocol_version: str | None = None
	primary_endpoint: str | None = None
	secondary_endpoints: list[str] = Field(default_factory=list)
	termination_reason: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class ClinicalTrialCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_number: str = Field(min_length=1)
	phase: TrialPhase
	trial_type: TrialType
	title: str = Field(min_length=1)
	short_title: str | None = None
	sponsor_id: str = Field(min_length=1)
	cro_id: str | None = None
	blinding: BlindingType
	indication: str = Field(min_length=1)
	therapeutic_area: str | None = None
	icd10_code: str | None = None
	investigational_product: str | None = None
	target_enrollment: int = Field(default=0, ge=0)
	planned_start_date: datetime | None = None
	planned_end_date: datetime | None = None
	primary_endpoint: str | None = None
	secondary_endpoints: list[str] = Field(default_factory=list)
	created_by: str = Field(min_length=1)


class ClinicalTrialUpdate(CtrBase):
	title: str | None = None
	short_title: str | None = None
	cro_id: str | None = None
	medical_monitor_id: str | None = None
	target_enrollment: int | None = Field(default=None, ge=0)
	investigational_product: str | None = None
	comparator: str | None = None
	primary_endpoint: str | None = None
	secondary_endpoints: list[str] | None = None
	planned_start_date: datetime | None = None
	planned_end_date: datetime | None = None
	therapeutic_area: str | None = None
	icd10_code: str | None = None
	metadata: dict[str, Any] | None = None


class ClinicalTrialResponse(ClinicalTrial):
	"""Extended response with computed fields."""
	days_active: int | None = None
	enrollment_completion_pct: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# TrialSite
# ─────────────────────────────────────────────────────────────────────────────

class TrialSite(CtrRecord):
	"""A participating clinical site within a trial."""
	trial_id: str
	site_number: str = Field(min_length=1)
	site_name: str = Field(min_length=1)
	country: str = Field(min_length=2, max_length=3)     # ISO 3166-1 alpha-2/3
	city: str | None = None
	address: str | None = None
	principal_investigator_id: str = Field(min_length=1)
	sub_investigators: list[str] = Field(default_factory=list)
	coordinator_id: str | None = None
	status: SiteStatus = SiteStatus.PRE_SELECTED
	qualification_visit_date: datetime | None = None
	initiation_visit_date: datetime | None = None
	close_out_date: datetime | None = None
	enrolled_count: int = Field(default=0, ge=0)
	randomised_count: int = Field(default=0, ge=0)
	target_enrollment: int = Field(ge=1)
	ethics_approval_reference: str | None = None
	regulatory_clearance_reference: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class TrialSiteCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	site_number: str = Field(min_length=1)
	site_name: str = Field(min_length=1)
	country: str = Field(min_length=2, max_length=3)
	city: str | None = None
	address: str | None = None
	principal_investigator_id: str = Field(min_length=1)
	target_enrollment: int = Field(ge=1)
	coordinator_id: str | None = None
	created_by: str = Field(min_length=1)


class TrialSiteUpdate(CtrBase):
	site_name: str | None = None
	coordinator_id: str | None = None
	target_enrollment: int | None = Field(default=None, ge=1)
	address: str | None = None
	ethics_approval_reference: str | None = None
	regulatory_clearance_reference: str | None = None
	metadata: dict[str, Any] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# TrialProtocol / ProtocolAmendment
# ─────────────────────────────────────────────────────────────────────────────

class TrialProtocol(CtrRecord):
	"""Versioned protocol document for a trial."""
	trial_id: str
	version: str = Field(min_length=1)
	status: ProtocolStatus = ProtocolStatus.DRAFT
	synopsis: str | None = None
	irb_submission_reference: str | None = None
	irb_approval_reference: str | None = None
	amendment_reason: str | None = None
	parent_version: str | None = None          # Version this was derived from
	effective_date: datetime | None = None
	superseded_date: datetime | None = None
	document_reference: str | None = None


class TrialProtocolCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	version: str = Field(min_length=1)
	synopsis: str | None = None
	parent_version: str | None = None
	document_reference: str | None = None
	created_by: str = Field(min_length=1)


class TrialProtocolUpdate(CtrBase):
	synopsis: str | None = None
	document_reference: str | None = None
	irb_submission_reference: str | None = None


class ProtocolAmendment(CtrRecord):
	"""Amendment to an approved protocol, tracked separately for audit."""
	trial_id: str
	protocol_id: str
	amendment_number: str = Field(min_length=1)
	amendment_type: AmendmentType
	status: AmendmentStatus = AmendmentStatus.DRAFT
	rationale: str = Field(min_length=1)
	summary_of_changes: str = Field(min_length=1)
	irb_submission_reference: str | None = None
	irb_approval_reference: str | None = None
	regulatory_notification_required: bool = False
	regulatory_submitted_at: datetime | None = None
	effective_date: datetime | None = None
	approved_by: str | None = None
	approved_at: datetime | None = None


class ProtocolAmendmentCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	protocol_id: str
	amendment_number: str = Field(min_length=1)
	amendment_type: AmendmentType
	rationale: str = Field(min_length=10)
	summary_of_changes: str = Field(min_length=10)
	regulatory_notification_required: bool = False
	created_by: str = Field(min_length=1)


# ─────────────────────────────────────────────────────────────────────────────
# IRBApproval
# ─────────────────────────────────────────────────────────────────────────────

class IRBApproval(CtrRecord):
	"""Institutional Review Board / Ethics Committee approval record."""
	trial_id: str
	protocol_id: str | None = None
	amendment_id: str | None = None
	irb_name: str = Field(min_length=1)
	irb_reference: str = Field(min_length=1)
	submission_date: datetime
	decision: IRBDecision | None = None
	decision_date: datetime | None = None
	expiry_date: datetime | None = None
	conditions: list[str] = Field(default_factory=list)
	document_reference: str | None = None
	is_initial: bool = True     # False for continuing reviews / amendments


class IRBApprovalCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	protocol_id: str | None = None
	amendment_id: str | None = None
	irb_name: str = Field(min_length=1)
	irb_reference: str = Field(min_length=1)
	submission_date: datetime
	is_initial: bool = True
	created_by: str = Field(min_length=1)


class IRBApprovalUpdate(CtrBase):
	decision: IRBDecision | None = None
	decision_date: datetime | None = None
	expiry_date: datetime | None = None
	conditions: list[str] | None = None
	document_reference: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# TrialPatient (Subject)
# ─────────────────────────────────────────────────────────────────────────────

class TrialPatient(CtrRecord):
	"""Subject enrolled in a clinical trial — de-identified per GCP."""
	trial_id: str
	site_id: str
	patient_code: str = Field(min_length=1)   # De-identified subject ID
	status: SubjectStatus = SubjectStatus.SCREENED
	informed_consent_date: datetime | None = None
	informed_consent_version: str | None = None
	randomisation_date: datetime | None = None
	randomisation_code: str | None = None
	treatment_arm: str | None = None
	screen_failure_reason: str | None = None
	withdrawal_date: datetime | None = None
	withdrawal_reason: str | None = None
	completion_date: datetime | None = None
	date_of_birth_year: int | None = None     # Year only — de-identification
	sex: str | None = None                     # M / F / U
	eligibility_criteria_met: bool = False
	re_consent_required: bool = False
	metadata: dict[str, Any] = Field(default_factory=dict)


class TrialPatientCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	site_id: str
	patient_code: str = Field(min_length=1)
	date_of_birth_year: int | None = None
	sex: str | None = None
	created_by: str = Field(min_length=1)


class TrialPatientUpdate(CtrBase):
	withdrawal_reason: str | None = None
	screen_failure_reason: str | None = None
	re_consent_required: bool | None = None
	metadata: dict[str, Any] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# RandomisationRecord
# ─────────────────────────────────────────────────────────────────────────────

class RandomisationRecord(CtrRecord):
	"""Randomisation allocation — blinded arm stored separately under RTSM."""
	trial_id: str
	patient_id: str
	site_id: str
	randomisation_method: RandomisationMethod
	randomisation_code: str = Field(min_length=1)
	treatment_arm: str = Field(min_length=1)      # Unblinded (RTSM-access-controlled)
	blinded_arm_label: str | None = None           # e.g. "A" / "B"
	stratification_factors: dict[str, str] = Field(default_factory=dict)
	block_size: int | None = None
	ivrs_reference: str | None = None              # Interactive Voice/Web Response System
	randomised_at: datetime = Field(default_factory=_utcnow)
	unblinded: bool = False
	unblinded_at: datetime | None = None
	unblinded_by: str | None = None
	unblinding_reason: str | None = None


class RandomisationCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	patient_id: str
	site_id: str
	randomisation_method: RandomisationMethod
	randomisation_code: str = Field(min_length=1)
	treatment_arm: str = Field(min_length=1)
	stratification_factors: dict[str, str] = Field(default_factory=dict)
	block_size: int | None = None
	ivrs_reference: str | None = None
	created_by: str = Field(min_length=1)


# ─────────────────────────────────────────────────────────────────────────────
# CRFForm / DataEntry / DataQuery
# ─────────────────────────────────────────────────────────────────────────────

class CRFFieldDef(CtrBase):
	"""Definition of a single CRF field."""
	field_name: str = Field(min_length=1)
	field_type: str = "text"    # text | number | date | choice | boolean
	label: str
	required: bool = False
	choices: list[str] = Field(default_factory=list)
	min_value: float | None = None
	max_value: float | None = None
	validation_regex: str | None = None
	help_text: str | None = None


class CRFForm(CtrRecord):
	"""Case Report Form definition attached to a trial visit."""
	trial_id: str
	form_name: str = Field(min_length=1)
	form_version: str = Field(min_length=1)
	visit_name: str | None = None
	fields: list[CRFFieldDef] = Field(default_factory=list)
	is_active: bool = True


class CRFFormCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	form_name: str = Field(min_length=1)
	form_version: str = Field(min_length=1)
	visit_name: str | None = None
	fields: list[CRFFieldDef] = Field(default_factory=list)
	created_by: str = Field(min_length=1)


class DataEntry(CtrRecord):
	"""A single CRF data entry for a patient visit (21 CFR Part 11 compliant)."""
	trial_id: str
	site_id: str
	patient_id: str
	crf_form_id: str
	visit_name: str = Field(min_length=1)
	visit_date: datetime
	data: dict[str, Any] = Field(default_factory=dict)
	status: CRFStatus = CRFStatus.DRAFT
	open_query_count: int = Field(default=0, ge=0)
	signed_off_by: str | None = None
	signed_off_at: datetime | None = None
	locked_by: str | None = None
	locked_at: datetime | None = None
	data_entry_operator: str | None = None
	double_data_entry_operator: str | None = None
	discrepancy_resolved: bool = True
	validation_status: str = "pending"   # pending | passed | warnings | failed
	ecrf_version: str = "1.0"


class DataEntryCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	site_id: str
	patient_id: str
	crf_form_id: str
	visit_name: str = Field(min_length=1)
	visit_date: datetime
	data: dict[str, Any] = Field(default_factory=dict)
	data_entry_operator: str | None = None
	created_by: str = Field(min_length=1)


class DataEntryUpdate(CtrBase):
	data: dict[str, Any] | None = None
	visit_date: datetime | None = None


class DataQuery(CtrRecord):
	"""Data clarification query raised against a CRF field."""
	data_entry_id: str
	field_name: str = Field(min_length=1)
	query_type: str = "missing_data"    # missing_data | out_of_range | inconsistency | other
	query_text: str = Field(min_length=1)
	raised_by: str = Field(min_length=1)
	raised_at: datetime = Field(default_factory=_utcnow)
	sla_due_date: datetime | None = None
	status: QueryStatus = QueryStatus.OPEN
	response: str | None = None
	responded_by: str | None = None
	responded_at: datetime | None = None
	closed_at: datetime | None = None
	closed_by: str | None = None


class DataQueryCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	data_entry_id: str
	field_name: str = Field(min_length=1)
	query_type: str = "missing_data"
	query_text: str = Field(min_length=1)
	raised_by: str = Field(min_length=1)
	created_by: str = Field(min_length=1)


# ─────────────────────────────────────────────────────────────────────────────
# AdverseEvent / SeriousAdverseEvent
# ─────────────────────────────────────────────────────────────────────────────

class AdverseEvent(CtrRecord):
	"""Non-serious adverse event report (ICH E2A compliant)."""
	trial_id: str
	patient_id: str
	site_id: str
	ae_type: AEType = AEType.ADVERSE_EVENT
	severity_grade: AESeverity
	meddra_pt: str | None = None    # MedDRA Preferred Term
	meddra_soc: str | None = None   # System Organ Class
	meddra_llt: str | None = None   # Lowest Level Term
	onset_date: datetime
	resolution_date: datetime | None = None
	causality: AECausality | None = None
	causality_sponsor: AECausality | None = None
	outcome: AEOutcome | None = None
	reported_at: datetime = Field(default_factory=_utcnow)
	reported_to_authority_at: datetime | None = None
	narrative: str = Field(min_length=1)
	is_serious: bool = False
	action_taken: str | None = None
	concomitant_medications: list[str] = Field(default_factory=list)
	reporter_id: str | None = None
	sar_filed: bool = False
	sar_id: str | None = None


class AdverseEventCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	patient_id: str
	site_id: str
	ae_type: AEType = AEType.ADVERSE_EVENT
	severity_grade: AESeverity
	onset_date: datetime
	narrative: str = Field(min_length=1)
	meddra_pt: str | None = None
	meddra_soc: str | None = None
	causality: AECausality | None = None
	action_taken: str | None = None
	concomitant_medications: list[str] = Field(default_factory=list)
	created_by: str = Field(min_length=1)


class AdverseEventUpdate(CtrBase):
	causality: AECausality | None = None
	causality_sponsor: AECausality | None = None
	outcome: AEOutcome | None = None
	resolution_date: datetime | None = None
	meddra_pt: str | None = None
	meddra_soc: str | None = None
	meddra_llt: str | None = None
	narrative: str | None = None
	action_taken: str | None = None


class SeriousAdverseEvent(CtrRecord):
	"""Serious Adverse Event (SAE) with expedited regulatory reporting obligations."""
	trial_id: str
	patient_id: str
	site_id: str
	linked_ae_id: str | None = None
	ae_type: AEType = AEType.SAE
	severity_grade: AESeverity
	meddra_pt: str | None = None
	meddra_soc: str | None = None
	meddra_llt: str | None = None
	onset_date: datetime
	resolution_date: datetime | None = None
	causality: AECausality | None = None
	causality_sponsor: AECausality | None = None
	outcome: AEOutcome | None = None
	narrative: str = Field(min_length=1)
	seriousness_criteria: list[str] = Field(default_factory=list)
	is_unexpected: bool = False
	susar_criteria_met: bool = False
	reported_to_sponsor_at: datetime | None = None
	reported_to_irb_at: datetime | None = None
	reported_to_authority_at: datetime | None = None
	reporting_deadline: datetime | None = None
	timeline_met: bool = True
	follow_up_required: bool = False
	follow_up_count: int = Field(default=0, ge=0)
	sar_filed: bool = False
	sar_reference: str | None = None


class SeriousAdverseEventCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	patient_id: str
	site_id: str
	linked_ae_id: str | None = None
	severity_grade: AESeverity
	onset_date: datetime
	narrative: str = Field(min_length=10)
	seriousness_criteria: list[str] = Field(min_length=1)
	is_unexpected: bool = False
	meddra_pt: str | None = None
	meddra_soc: str | None = None
	causality: AECausality | None = None
	created_by: str = Field(min_length=1)


# ─────────────────────────────────────────────────────────────────────────────
# MonitoringVisit
# ─────────────────────────────────────────────────────────────────────────────

class ActionItem(CtrBase):
	"""An action item arising from a monitoring visit."""
	id: str = Field(default_factory=_uuid7str)
	description: str = Field(min_length=1)
	assigned_to: str
	due_date: datetime | None = None
	closed: bool = False
	closed_at: datetime | None = None


class MonitoringVisit(CtrRecord):
	"""Site monitoring visit record (risk-based or 100% SDV)."""
	trial_id: str
	site_id: str
	visit_type: MonitoringVisitType
	monitor_id: str = Field(min_length=1)
	planned_date: datetime
	actual_date: datetime | None = None
	completed: bool = False
	sdv_rate: float | None = Field(default=None, ge=0.0, le=1.0)
	protocol_deviations_identified: int = Field(default=0, ge=0)
	critical_findings: int = Field(default=0, ge=0)
	action_items: list[ActionItem] = Field(default_factory=list)
	follow_up_required: bool = False
	follow_up_deadline: datetime | None = None
	report_reference: str | None = None
	sponsor_reviewed_at: datetime | None = None


class MonitoringVisitCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	site_id: str
	visit_type: MonitoringVisitType
	monitor_id: str = Field(min_length=1)
	planned_date: datetime
	created_by: str = Field(min_length=1)


class MonitoringVisitUpdate(CtrBase):
	actual_date: datetime | None = None
	completed: bool | None = None
	sdv_rate: float | None = Field(default=None, ge=0.0, le=1.0)
	protocol_deviations_identified: int | None = Field(default=None, ge=0)
	critical_findings: int | None = Field(default=None, ge=0)
	action_items: list[ActionItem] | None = None
	follow_up_required: bool | None = None
	follow_up_deadline: datetime | None = None
	report_reference: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Inspection
# ─────────────────────────────────────────────────────────────────────────────

class InspectionFinding(CtrBase):
	"""A single finding from a regulatory inspection."""
	id: str = Field(default_factory=_uuid7str)
	finding_type: str = "observation"   # observation | major | critical
	description: str = Field(min_length=1)
	reference_standard: str | None = None
	capa_required: bool = False
	capa_description: str | None = None
	capa_due_date: datetime | None = None
	closed: bool = False


class Inspection(CtrRecord):
	"""Regulatory authority inspection of sponsor, site, or CRO."""
	trial_id: str
	site_id: str | None = None
	authority: RegulatoryAuthority
	inspection_type: InspectionType
	inspector_ids: list[str] = Field(default_factory=list)
	announced: bool = True
	planned_start_date: datetime
	planned_end_date: datetime
	actual_start_date: datetime | None = None
	actual_end_date: datetime | None = None
	outcome: InspectionOutcome | None = None
	findings: list[InspectionFinding] = Field(default_factory=list)
	response_due_date: datetime | None = None
	response_submitted_at: datetime | None = None
	closed_at: datetime | None = None
	document_reference: str | None = None


class InspectionCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	site_id: str | None = None
	authority: RegulatoryAuthority
	inspection_type: InspectionType
	announced: bool = True
	planned_start_date: datetime
	planned_end_date: datetime
	created_by: str = Field(min_length=1)


# ─────────────────────────────────────────────────────────────────────────────
# TMFDocument (Trial Master File)
# ─────────────────────────────────────────────────────────────────────────────

class TMFDocument(CtrRecord):
	"""A document within the Trial Master File (ICH E6 R2/R3 / TMF Reference Model)."""
	trial_id: str
	site_id: str | None = None
	tmf_section: str = Field(min_length=1)       # e.g. "01.01", "03.02.01"
	tmf_artifact_name: str = Field(min_length=1) # TMF Reference Model artifact name
	document_title: str = Field(min_length=1)
	document_reference: str | None = None
	version: str | None = None
	status: TMFDocumentStatus = TMFDocumentStatus.EXPECTED
	expected_date: datetime | None = None
	filed_date: datetime | None = None
	overdue_since: datetime | None = None
	archive_location: str | None = None
	file_hash_sha256: str | None = None          # Integrity proof per ICH E6(R3)
	file_size_bytes: int | None = None
	mime_type: str | None = None
	is_essential: bool = True
	superseded_by: str | None = None             # ID of superseding document
	metadata: dict[str, Any] = Field(default_factory=dict)


class TMFDocumentCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	site_id: str | None = None
	tmf_section: str = Field(min_length=1)
	tmf_artifact_name: str = Field(min_length=1)
	document_title: str = Field(min_length=1)
	document_reference: str | None = None
	version: str | None = None
	expected_date: datetime | None = None
	is_essential: bool = True
	file_hash_sha256: str | None = None
	file_size_bytes: int | None = None
	mime_type: str | None = None
	created_by: str = Field(min_length=1)


class TMFDocumentUpdate(CtrBase):
	status: TMFDocumentStatus | None = None
	document_reference: str | None = None
	version: str | None = None
	filed_date: datetime | None = None
	archive_location: str | None = None
	file_hash_sha256: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# RegulatorySubmission (IND / CTA / etc.)
# ─────────────────────────────────────────────────────────────────────────────

class RegulatorySubmission(CtrRecord):
	"""Regulatory submission to a national authority."""
	trial_id: str
	submission_type: SubmissionType
	authority: RegulatoryAuthority
	submission_date: datetime | None = None
	reference_number: str | None = None
	cover_letter_reference: str = Field(min_length=1)
	dossier_reference: str = Field(min_length=1)
	status: SubmissionStatus = SubmissionStatus.NOT_SUBMITTED
	response_due_date: datetime | None = None
	authority_response: str | None = None
	authority_response_date: datetime | None = None
	approval_date: datetime | None = None
	approved: bool = False
	rejection_reason: str | None = None
	amendment_number: str | None = None
	ectd_compliant: bool = False
	package_items: list[str] = Field(default_factory=list)


class RegulatorySubmissionCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	submission_type: SubmissionType
	authority: RegulatoryAuthority
	cover_letter_reference: str = Field(min_length=1)
	dossier_reference: str = Field(min_length=1)
	amendment_number: str | None = None
	package_items: list[str] = Field(default_factory=list)
	created_by: str = Field(min_length=1)


class RegulatorySubmissionUpdate(CtrBase):
	status: SubmissionStatus | None = None
	reference_number: str | None = None
	authority_response: str | None = None
	authority_response_date: datetime | None = None
	approval_date: datetime | None = None
	rejection_reason: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Protocol deviation
# ─────────────────────────────────────────────────────────────────────────────

class ProtocolDeviation(CtrRecord):
	"""Protocol deviation record with IRB reportability classification."""
	trial_id: str
	site_id: str | None = None
	patient_id: str | None = None
	deviation_type: ProtocolDeviationType
	impact: ProtocolDeviationImpact
	description: str = Field(min_length=1)
	corrective_action: str = Field(min_length=1)
	reported_by: str = Field(min_length=1)
	irb_reportable: bool = False
	irb_reported: bool = False
	irb_reported_at: datetime | None = None
	status: str = "open"   # open | closed


class ProtocolDeviationCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	site_id: str | None = None
	patient_id: str | None = None
	deviation_type: ProtocolDeviationType
	impact: ProtocolDeviationImpact
	description: str = Field(min_length=1)
	corrective_action: str = Field(min_length=1)
	reported_by: str = Field(min_length=1)
	created_by: str = Field(min_length=1)


# ─────────────────────────────────────────────────────────────────────────────
# Informed consent tracking
# ─────────────────────────────────────────────────────────────────────────────

class InformedConsentRecord(CtrRecord):
	"""Audit record for informed consent version and re-consent tracking."""
	trial_id: str
	patient_id: str
	site_id: str
	icf_version: str = Field(min_length=1)
	consent_date: datetime
	consented_by: str = Field(min_length=1)   # Patient signature witness / investigator
	witness_id: str | None = None
	re_consent_required: bool = False
	re_consent_reason: str | None = None
	status: str = "consented"   # consented | re_consent_pending | withdrawn


class InformedConsentCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	patient_id: str
	site_id: str
	icf_version: str = Field(min_length=1)
	consent_date: datetime
	consented_by: str = Field(min_length=1)
	witness_id: str | None = None
	re_consent_required: bool = False
	re_consent_reason: str | None = None
	created_by: str = Field(min_length=1)


# ─────────────────────────────────────────────────────────────────────────────
# Interim analysis request
# ─────────────────────────────────────────────────────────────────────────────

class InterimAnalysisRequest(CtrRecord):
	"""Request to generate a pre-specified interim analysis."""
	trial_id: str
	analysis_number: int = Field(ge=1)
	analysis_type: str = "efficacy"   # efficacy | futility | safety | combined
	triggered_by: str = Field(min_length=1)
	data_cut_date: datetime
	statistical_method: str = Field(min_length=1)
	spending_function: str = "obrien_fleming"
	alpha_spending: float = Field(default=0.025, gt=0.0, le=0.5)
	stopping_rule: str | None = None
	blinded: bool = True
	requested_at: datetime = Field(default_factory=_utcnow)
	completed_at: datetime | None = None
	efficacy_boundary_z: float | None = None
	futility_boundary_z: float | None = None
	observed_z_statistic: float | None = None
	recommendation: str = "pending"
	report_reference: str | None = None


class InterimAnalysisCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	analysis_number: int = Field(ge=1)
	analysis_type: str = "efficacy"
	data_cut_date: datetime
	statistical_method: str = Field(min_length=1)
	spending_function: str = "obrien_fleming"
	alpha_spending: float = Field(default=0.025, gt=0.0, le=0.5)
	stopping_rule: str | None = None
	blinded: bool = True
	triggered_by: str = Field(min_length=1)
	created_by: str = Field(min_length=1)


# ─────────────────────────────────────────────────────────────────────────────
# Trial closeout
# ─────────────────────────────────────────────────────────────────────────────

class TrialCloseoutRecord(CtrRecord):
	"""Trial close-out checklist and completion record."""
	trial_id: str
	initiated_by: str = Field(min_length=1)
	initiated_at: datetime = Field(default_factory=_utcnow)
	checklist: dict[str, bool] = Field(default_factory=dict)
	all_data_locked: bool = False
	tmf_complete: bool = False
	all_sites_closed: bool = False
	final_report_submitted: bool = False
	all_aes_resolved: bool = False
	all_submissions_filed: bool = False
	completed: bool = False
	completed_at: datetime | None = None


class TrialCloseoutCreate(CtrBase):
	tenant_id: str = Field(min_length=1)
	trial_id: str
	initiated_by: str = Field(min_length=1)
	created_by: str = Field(min_length=1)


# ─────────────────────────────────────────────────────────────────────────────
# Report / Aggregation models
# ─────────────────────────────────────────────────────────────────────────────

class TrialSummaryReport(CtrBase):
	"""High-level trial status report (KPI dashboard card)."""
	trial_id: str
	trial_number: str
	phase: str
	status: str
	enrolled: int
	randomised: int
	completed: int
	withdrawn: int
	screen_failures: int
	total_ae_count: int
	total_sae_count: int
	total_susar_count: int
	open_queries: int
	overdue_tmf_docs: int
	site_count: int
	active_site_count: int
	submissions_filed: int
	monitoring_visits_completed: int
	enrollment_completion_pct: float
	generated_at: datetime = Field(default_factory=_utcnow)


class SafetyReport(CtrBase):
	"""Aggregated safety report for a trial (DSMB / SMC periodic)."""
	trial_id: str
	period_start: datetime
	period_end: datetime
	total_ae: int
	grade_1: int
	grade_2: int
	grade_3: int
	grade_4: int
	grade_5: int
	total_sae: int
	susar_count: int
	ae_by_system_organ_class: dict[str, int] = Field(default_factory=dict)
	ae_by_causality: dict[str, int] = Field(default_factory=dict)
	ae_timeline_violations: int
	smc_recommendation: str = "continue"
	generated_at: datetime = Field(default_factory=_utcnow)


class EnrollmentReport(CtrBase):
	"""Enrollment status across all sites."""
	trial_id: str
	target_enrollment: int
	total_enrolled: int
	total_randomised: int
	total_screen_failures: int
	screen_failure_rate: float
	enrollment_rate_per_site_per_month: float
	projected_completion_date: datetime | None = None
	by_site: list[dict[str, Any]] = Field(default_factory=list)
	by_country: dict[str, int] = Field(default_factory=dict)
	generated_at: datetime = Field(default_factory=_utcnow)


class DataQualityReport(CtrBase):
	"""CRF / EDC data quality metrics."""
	trial_id: str
	total_crf_forms: int
	completed_forms: int
	forms_with_queries: int
	total_open_queries: int
	total_resolved_queries: int
	query_resolution_rate: float
	overdue_forms: int
	sdv_rate: float
	missing_data_rate: float
	data_quality_score: float
	generated_at: datetime = Field(default_factory=_utcnow)


class TMFCompletenessReport(CtrBase):
	"""Trial Master File completeness metrics per ICH E6(R3)."""
	trial_id: str
	total_expected_documents: int
	total_filed_documents: int
	overdue_documents: int
	completeness_rate: float
	overdue_rate: float
	health: str   # green | amber | red
	by_section: dict[str, dict[str, Any]] = Field(default_factory=dict)
	generated_at: datetime = Field(default_factory=_utcnow)
