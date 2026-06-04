"""Pydantic v2 models for APG Pharma Clinical Trials Management.

Entities:
	ClinicalTrial, TrialSite, Subject (TrialPatient), Randomisation,
	Protocol, ProtocolAmendment, AdverseEvent, SeriousAdverseEvent,
	CRF, DataEntry, MonitoringVisit, Inspection, TMF,
	INDSubmission / CTASubmission, IRBApproval

All IDs are UUID7 strings.  All timestamps are UTC datetimes.
Tenant isolation is encoded in every record.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _uuid7str() -> str:
	return str(uuid7())


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class TrialPhase(str, Enum):
	PHASE_1 = "phase_1"
	PHASE_1B = "phase_1b"
	PHASE_2 = "phase_2"
	PHASE_2B = "phase_2b"
	PHASE_3 = "phase_3"
	PHASE_3B = "phase_3b"
	PHASE_4 = "phase_4"
	EXPANDED_ACCESS = "expanded_access"
	OBSERVATIONAL = "observational"


class TrialType(str, Enum):
	INTERVENTIONAL = "interventional"
	OBSERVATIONAL = "observational"
	EXPANDED_ACCESS = "expanded_access"
	REGISTRY = "registry"
	BIOEQUIVALENCE = "bioequivalence"
	FIRST_IN_HUMAN = "first_in_human"
	BASKET = "basket"
	UMBRELLA = "umbrella"


class TrialStatus(str, Enum):
	PLANNED = "planned"
	ACTIVE = "active"
	ENROLLING = "enrolling"
	ENROLLMENT_COMPLETE = "enrollment_complete"
	TREATMENT_ONGOING = "treatment_ongoing"
	FOLLOW_UP = "follow_up"
	COMPLETED = "completed"
	TERMINATED = "terminated"
	SUSPENDED = "suspended"
	WITHDRAWN = "withdrawn"


class BlindingType(str, Enum):
	OPEN_LABEL = "open_label"
	SINGLE_BLIND = "single_blind"
	DOUBLE_BLIND = "double_blind"
	TRIPLE_BLIND = "triple_blind"


class SiteStatus(str, Enum):
	PRE_SELECTED = "pre_selected"
	SELECTED = "selected"
	INITIATED = "initiated"
	ENROLLING = "enrolling"
	ENROLLMENT_COMPLETE = "enrollment_complete"
	CLOSED = "closed"
	TERMINATED = "terminated"
	WITHDRAWN = "withdrawn"


class SubjectStatus(str, Enum):
	SCREENED = "screened"
	ENROLLED = "enrolled"
	RANDOMISED = "randomised"
	ON_TREATMENT = "on_treatment"
	COMPLETED = "completed"
	WITHDRAWN = "withdrawn"
	LOST_TO_FOLLOW_UP = "lost_to_follow_up"
	SCREEN_FAILURE = "screen_failure"


class RandomisationMethod(str, Enum):
	SIMPLE = "simple"
	STRATIFIED = "stratified"
	BLOCK = "block"
	ADAPTIVE = "adaptive"
	MINIMISATION = "minimisation"
	DYNAMIC = "dynamic"


class ProtocolStatus(str, Enum):
	DRAFT = "draft"
	UNDER_REVIEW = "under_review"
	APPROVED = "approved"
	AMENDED = "amended"
	SUPERSEDED = "superseded"
	WITHDRAWN = "withdrawn"


class AmendmentType(str, Enum):
	SUBSTANTIAL = "substantial"
	NON_SUBSTANTIAL = "non_substantial"
	URGENT_SAFETY = "urgent_safety"
	ADMINISTRATIVE = "administrative"


class AEType(str, Enum):
	ADVERSE_EVENT = "adverse_event"
	SERIOUS_ADVERSE_EVENT = "serious_adverse_event"
	SUSAR = "suspected_unexpected_serious_adverse_reaction"
	DISEASE_RELATED = "disease_related_event"
	PROTOCOL_DEVIATION = "protocol_deviation"


class AESeverity(str, Enum):
	GRADE_1 = "grade_1"
	GRADE_2 = "grade_2"
	GRADE_3 = "grade_3"
	GRADE_4 = "grade_4"
	GRADE_5 = "grade_5"  # Fatal


class AEOutcome(str, Enum):
	RECOVERED = "recovered"
	RECOVERING = "recovering"
	NOT_RECOVERED = "not_recovered"
	SEQUELAE = "recovered_with_sequelae"
	FATAL = "fatal"
	UNKNOWN = "unknown"


class AECausality(str, Enum):
	UNRELATED = "unrelated"
	UNLIKELY = "unlikely"
	POSSIBLE = "possible"
	PROBABLE = "probable"
	DEFINITE = "definite"
	NOT_ASSESSABLE = "not_assessable"


class CRFStatus(str, Enum):
	DRAFT = "draft"
	PENDING_QUERY = "pending_query"
	QUERY_RESOLVED = "query_resolved"
	SIGNED_OFF = "signed_off"
	LOCKED = "locked"
	UNLOCKED = "unlocked"


class MonitoringVisitType(str, Enum):
	QUALIFICATION = "qualification"
	INITIATION = "initiation"
	ROUTINE = "routine"
	CLOSE_OUT = "close_out"
	FOR_CAUSE = "for_cause"
	RISK_BASED = "risk_based"


class SubmissionType(str, Enum):
	IND = "ind"
	CTA = "cta"
	PROTOCOL_AMENDMENT = "protocol_amendment"
	ANNUAL_REPORT = "annual_report"
	SAFETY_REPORT = "safety_report"
	FINAL_REPORT = "final_report"
	EUDRACT = "eudract"
	CTIS = "ctis"


class RegulatoryAuthority(str, Enum):
	FDA = "fda"
	EMA = "ema"
	MHRA = "mhra"
	PMDA = "pmda"
	HEALTH_CANADA = "health_canada"
	TGA = "tga"
	ANVISA = "anvisa"
	CDSCO = "cdsco"
	NMPA = "nmpa"
	NMRA = "nmra"


class IRBDecision(str, Enum):
	APPROVED = "approved"
	APPROVED_WITH_CONDITIONS = "approved_with_conditions"
	REQUIRES_MODIFICATION = "requires_modification"
	REJECTED = "rejected"
	WITHDRAWN = "withdrawn"


class TMFDocumentStatus(str, Enum):
	EXPECTED = "expected"
	FILED = "filed"
	OVERDUE = "overdue"
	NOT_APPLICABLE = "not_applicable"
	ARCHIVED = "archived"


class InspectionOutcome(str, Enum):
	NO_FINDINGS = "no_findings"
	MINOR_FINDINGS = "minor_findings"
	MAJOR_FINDINGS = "major_findings"
	CRITICAL_FINDINGS = "critical_findings"
	WARNING_LETTER = "warning_letter"


# ─────────────────────────────────────────────────────────────────────────────
# Base model
# ─────────────────────────────────────────────────────────────────────────────

class CtrBase(BaseModel):
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
		use_enum_values=True,
	)


class CtrRecord(CtrBase):
	"""Base for all persistent records — provides the audit columns."""
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	is_deleted: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# ClinicalTrial
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalTrial(CtrRecord):
	"""Master record for a clinical trial."""
	trial_number: str
	eudract_number: str | None = None
	nct_number: str | None = None  # ClinicalTrials.gov identifier
	phase: TrialPhase
	trial_type: TrialType
	title: str
	short_title: str | None = None
	sponsor_id: str
	cro_id: str | None = None
	medical_monitor_id: str | None = None
	blinding: BlindingType
	status: TrialStatus = TrialStatus.PLANNED
	indication: str
	therapeutic_area: str | None = None
	investigational_product: str | None = None
	comparator: str | None = None
	target_enrollment: int = 0
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
	tenant_id: str
	trial_number: str
	phase: TrialPhase
	trial_type: TrialType
	title: str
	short_title: str | None = None
	sponsor_id: str
	cro_id: str | None = None
	blinding: BlindingType
	indication: str
	therapeutic_area: str | None = None
	investigational_product: str | None = None
	target_enrollment: int = 0
	planned_start_date: datetime | None = None
	planned_end_date: datetime | None = None
	primary_endpoint: str | None = None
	created_by: str


class ClinicalTrialUpdate(CtrBase):
	title: str | None = None
	short_title: str | None = None
	cro_id: str | None = None
	medical_monitor_id: str | None = None
	target_enrollment: int | None = None
	investigational_product: str | None = None
	primary_endpoint: str | None = None
	secondary_endpoints: list[str] | None = None
	metadata: dict[str, Any] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# TrialSite
# ─────────────────────────────────────────────────────────────────────────────

class TrialSite(CtrRecord):
	"""A participating clinical site within a trial."""
	trial_id: str
	site_number: str
	site_name: str
	country: str
	city: str | None = None
	address: str | None = None
	principal_investigator_id: str
	sub_investigators: list[str] = Field(default_factory=list)
	coordinator_id: str | None = None
	status: SiteStatus = SiteStatus.PRE_SELECTED
	qualification_visit_date: datetime | None = None
	initiation_visit_date: datetime | None = None
	close_out_date: datetime | None = None
	enrolled_count: int = 0
	randomised_count: int = 0
	target_enrollment: int
	ethics_approval_reference: str | None = None
	regulatory_clearance_reference: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class TrialSiteCreate(CtrBase):
	tenant_id: str
	trial_id: str
	site_number: str
	site_name: str
	country: str
	city: str | None = None
	principal_investigator_id: str
	target_enrollment: int
	created_by: str


class TrialSiteUpdate(CtrBase):
	site_name: str | None = None
	coordinator_id: str | None = None
	target_enrollment: int | None = None
	metadata: dict[str, Any] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Protocol / Amendment
# ─────────────────────────────────────────────────────────────────────────────

class TrialProtocol(CtrRecord):
	"""Versioned protocol document for a trial."""
	trial_id: str
	version: str
	status: ProtocolStatus = ProtocolStatus.DRAFT
	synopsis: str | None = None
	irb_submission_reference: str | None = None
	irb_approval_reference: str | None = None
	amendment_reason: str | None = None
	parent_version: str | None = None
	effective_date: datetime | None = None
	superseded_date: datetime | None = None
	document_reference: str | None = None


class TrialProtocolCreate(CtrBase):
	tenant_id: str
	trial_id: str
	version: str
	synopsis: str | None = None
	parent_version: str | None = None
	document_reference: str | None = None
	created_by: str


class ProtocolAmendment(CtrRecord):
	"""Amendment to an approved protocol."""
	trial_id: str
	protocol_id: str
	amendment_number: str
	amendment_type: AmendmentType
	rationale: str
	summary_of_changes: str
	irb_submission_reference: str | None = None
	irb_approval_reference: str | None = None
	regulatory_notification_required: bool = False
	regulatory_submitted_at: datetime | None = None
	effective_date: datetime | None = None
	status: ProtocolStatus = ProtocolStatus.DRAFT


class ProtocolAmendmentCreate(CtrBase):
	tenant_id: str
	trial_id: str
	protocol_id: str
	amendment_number: str
	amendment_type: AmendmentType
	rationale: str
	summary_of_changes: str
	regulatory_notification_required: bool = False
	created_by: str


# ─────────────────────────────────────────────────────────────────────────────
# IRBApproval
# ─────────────────────────────────────────────────────────────────────────────

class IRBApproval(CtrRecord):
	"""Institutional Review Board / Ethics Committee approval record."""
	trial_id: str
	protocol_id: str | None = None
	amendment_id: str | None = None
	irb_name: str
	irb_reference: str
	submission_date: datetime
	decision: IRBDecision | None = None
	decision_date: datetime | None = None
	expiry_date: datetime | None = None
	conditions: list[str] = Field(default_factory=list)
	document_reference: str | None = None
	is_initial: bool = True


class IRBApprovalCreate(CtrBase):
	tenant_id: str
	trial_id: str
	protocol_id: str | None = None
	amendment_id: str | None = None
	irb_name: str
	irb_reference: str
	submission_date: datetime
	is_initial: bool = True
	created_by: str


# ─────────────────────────────────────────────────────────────────────────────
# TrialPatient (Subject)
# ─────────────────────────────────────────────────────────────────────────────

class TrialPatient(CtrRecord):
	"""Subject enrolled in a clinical trial."""
	trial_id: str
	site_id: str
	patient_code: str  # De-identified subject ID
	status: SubjectStatus = SubjectStatus.SCREENED
	informed_consent_date: datetime | None = None
	randomisation_date: datetime | None = None
	randomisation_code: str | None = None
	treatment_arm: str | None = None
	screen_failure_reason: str | None = None
	withdrawal_date: datetime | None = None
	withdrawal_reason: str | None = None
	completion_date: datetime | None = None
	date_of_birth_year: int | None = None  # Only year — de-identification
	sex: str | None = None
	eligibility_criteria_met: bool = False
	metadata: dict[str, Any] = Field(default_factory=dict)


class TrialPatientCreate(CtrBase):
	tenant_id: str
	trial_id: str
	site_id: str
	patient_code: str
	date_of_birth_year: int | None = None
	sex: str | None = None
	created_by: str


class TrialPatientUpdate(CtrBase):
	withdrawal_reason: str | None = None
	metadata: dict[str, Any] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Randomisation
# ─────────────────────────────────────────────────────────────────────────────

class RandomisationRecord(CtrRecord):
	"""Randomisation allocation record (blinded until unblinding)."""
	trial_id: str
	patient_id: str
	site_id: str
	randomisation_method: RandomisationMethod
	randomisation_code: str
	treatment_arm: str
	stratification_factors: dict[str, str] = Field(default_factory=dict)
	block_size: int | None = None
	ivrs_reference: str | None = None
	randomised_at: datetime = Field(default_factory=datetime.utcnow)
	unblinded: bool = False
	unblinded_at: datetime | None = None
	unblinded_by: str | None = None
	unblinding_reason: str | None = None


class RandomisationCreate(CtrBase):
	tenant_id: str
	trial_id: str
	patient_id: str
	site_id: str
	randomisation_method: RandomisationMethod
	randomisation_code: str
	treatment_arm: str
	stratification_factors: dict[str, str] = Field(default_factory=dict)
	block_size: int | None = None
	ivrs_reference: str | None = None
	created_by: str


# ─────────────────────────────────────────────────────────────────────────────
# CRF / DataEntry
# ─────────────────────────────────────────────────────────────────────────────

class CRFForm(CtrRecord):
	"""Case Report Form definition attached to a trial."""
	trial_id: str
	form_name: str
	form_version: str
	visit_name: str | None = None
	fields: list[dict[str, Any]] = Field(default_factory=list)
	is_active: bool = True


class CRFFormCreate(CtrBase):
	tenant_id: str
	trial_id: str
	form_name: str
	form_version: str
	visit_name: str | None = None
	fields: list[dict[str, Any]] = Field(default_factory=list)
	created_by: str


class DataEntry(CtrRecord):
	"""A single CRF data entry for a patient visit."""
	trial_id: str
	site_id: str
	patient_id: str
	crf_form_id: str
	visit_name: str
	visit_date: datetime
	data: dict[str, Any] = Field(default_factory=dict)
	status: CRFStatus = CRFStatus.DRAFT
	queries: list[dict[str, Any]] = Field(default_factory=list)
	signed_off_by: str | None = None
	signed_off_at: datetime | None = None
	locked_by: str | None = None
	locked_at: datetime | None = None
	data_entry_operator: str | None = None
	double_data_entry_operator: str | None = None
	discrepancy_resolved: bool = True


class DataEntryCreate(CtrBase):
	tenant_id: str
	trial_id: str
	site_id: str
	patient_id: str
	crf_form_id: str
	visit_name: str
	visit_date: datetime
	data: dict[str, Any] = Field(default_factory=dict)
	data_entry_operator: str | None = None
	created_by: str


class DataQuery(CtrBase):
	"""A data query raised against a specific data entry field."""
	id: str = Field(default_factory=_uuid7str)
	data_entry_id: str
	field_name: str
	query_text: str
	raised_by: str
	raised_at: datetime = Field(default_factory=datetime.utcnow)
	response: str | None = None
	responded_by: str | None = None
	responded_at: datetime | None = None
	resolved: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# AdverseEvent / SeriousAdverseEvent
# ─────────────────────────────────────────────────────────────────────────────

class AdverseEvent(CtrRecord):
	"""Non-serious adverse event report."""
	trial_id: str
	patient_id: str
	site_id: str
	ae_type: AEType = AEType.ADVERSE_EVENT
	severity_grade: AESeverity
	meddra_pt: str | None = None   # MedDRA Preferred Term
	meddra_soc: str | None = None  # System Organ Class
	meddra_llt: str | None = None  # Lowest Level Term
	onset_date: datetime
	resolution_date: datetime | None = None
	causality: AECausality | None = None
	outcome: AEOutcome | None = None
	reported_at: datetime = Field(default_factory=datetime.utcnow)
	reported_to_authority_at: datetime | None = None
	narrative: str
	is_serious: bool = False
	action_taken: str | None = None
	concomitant_medications: list[str] = Field(default_factory=list)
	reporter_id: str | None = None


class AdverseEventCreate(CtrBase):
	tenant_id: str
	trial_id: str
	patient_id: str
	site_id: str
	ae_type: AEType = AEType.ADVERSE_EVENT
	severity_grade: AESeverity
	onset_date: datetime
	narrative: str
	meddra_pt: str | None = None
	meddra_soc: str | None = None
	causality: AECausality | None = None
	action_taken: str | None = None
	created_by: str


class SeriousAdverseEvent(CtrRecord):
	"""Serious Adverse Event (SAE) with expedited reporting obligations."""
	trial_id: str
	patient_id: str
	site_id: str
	linked_ae_id: str | None = None
	ae_type: AEType = AEType.SERIOUS_ADVERSE_EVENT
	severity_grade: AESeverity
	meddra_pt: str | None = None
	meddra_soc: str | None = None
	onset_date: datetime
	resolution_date: datetime | None = None
	causality: AECausality | None = None
	outcome: AEOutcome | None = None
	narrative: str
	seriousness_criteria: list[str] = Field(default_factory=list)  # e.g., ["death", "hospitalisation"]
	is_unexpected: bool = False
	susar_criteria_met: bool = False
	reported_to_sponsor_at: datetime | None = None
	reported_to_irb_at: datetime | None = None
	reported_to_authority_at: datetime | None = None
	# 24h for SAE, 15 days for SUSAR
	reporting_deadline: datetime | None = None
	timeline_met: bool = True
	follow_up_required: bool = False
	follow_up_count: int = 0
	causality_sponsor: AECausality | None = None


class SeriousAdverseEventCreate(CtrBase):
	tenant_id: str
	trial_id: str
	patient_id: str
	site_id: str
	linked_ae_id: str | None = None
	severity_grade: AESeverity
	onset_date: datetime
	narrative: str
	seriousness_criteria: list[str]
	is_unexpected: bool = False
	meddra_pt: str | None = None
	meddra_soc: str | None = None
	causality: AECausality | None = None
	created_by: str


# ─────────────────────────────────────────────────────────────────────────────
# MonitoringVisit
# ─────────────────────────────────────────────────────────────────────────────

class MonitoringVisit(CtrRecord):
	"""Site monitoring visit record (risk-based or 100% SDV)."""
	trial_id: str
	site_id: str
	visit_type: MonitoringVisitType
	monitor_id: str
	planned_date: datetime
	actual_date: datetime | None = None
	completed: bool = False
	sdv_rate: float | None = None  # Source Data Verification rate 0-1
	protocol_deviations_identified: int = 0
	critical_findings: int = 0
	action_items: list[dict[str, Any]] = Field(default_factory=list)
	follow_up_required: bool = False
	follow_up_deadline: datetime | None = None
	report_reference: str | None = None
	sponsor_reviewed_at: datetime | None = None


class MonitoringVisitCreate(CtrBase):
	tenant_id: str
	trial_id: str
	site_id: str
	visit_type: MonitoringVisitType
	monitor_id: str
	planned_date: datetime
	created_by: str


# ─────────────────────────────────────────────────────────────────────────────
# Inspection
# ─────────────────────────────────────────────────────────────────────────────

class Inspection(CtrRecord):
	"""Regulatory authority inspection record."""
	trial_id: str
	site_id: str | None = None
	authority: RegulatoryAuthority
	inspection_type: str  # GCP, GLP, sponsor audit, etc.
	inspector_ids: list[str] = Field(default_factory=list)
	announced: bool = True
	planned_start_date: datetime
	planned_end_date: datetime
	actual_start_date: datetime | None = None
	actual_end_date: datetime | None = None
	outcome: InspectionOutcome | None = None
	findings: list[dict[str, Any]] = Field(default_factory=list)
	response_due_date: datetime | None = None
	response_submitted_at: datetime | None = None
	closed_at: datetime | None = None
	document_reference: str | None = None


class InspectionCreate(CtrBase):
	tenant_id: str
	trial_id: str
	site_id: str | None = None
	authority: RegulatoryAuthority
	inspection_type: str
	announced: bool = True
	planned_start_date: datetime
	planned_end_date: datetime
	created_by: str


# ─────────────────────────────────────────────────────────────────────────────
# TMF (Trial Master File)
# ─────────────────────────────────────────────────────────────────────────────

class TMFDocument(CtrRecord):
	"""A single document within the Trial Master File (ICH E6 R2 / TMF Reference Model)."""
	trial_id: str
	site_id: str | None = None
	tmf_section: str  # e.g., "1.1", "2.4.3"
	tmf_artifact_name: str
	document_title: str
	document_reference: str | None = None
	version: str | None = None
	status: TMFDocumentStatus = TMFDocumentStatus.EXPECTED
	expected_date: datetime | None = None
	filed_date: datetime | None = None
	overdue_since: datetime | None = None
	archive_location: str | None = None
	is_essential: bool = True
	metadata: dict[str, Any] = Field(default_factory=dict)


class TMFDocumentCreate(CtrBase):
	tenant_id: str
	trial_id: str
	site_id: str | None = None
	tmf_section: str
	tmf_artifact_name: str
	document_title: str
	document_reference: str | None = None
	version: str | None = None
	expected_date: datetime | None = None
	is_essential: bool = True
	created_by: str


# ─────────────────────────────────────────────────────────────────────────────
# Regulatory Submission (IND / CTA / etc.)
# ─────────────────────────────────────────────────────────────────────────────

class RegulatorySubmission(CtrRecord):
	"""Regulatory submission to a national authority."""
	trial_id: str
	submission_type: SubmissionType
	authority: RegulatoryAuthority
	submission_date: datetime | None = None
	reference_number: str | None = None
	cover_letter_reference: str
	dossier_reference: str
	status: str = "not_submitted"
	response_due_date: datetime | None = None
	authority_response: str | None = None
	authority_response_date: datetime | None = None
	approval_date: datetime | None = None
	approved: bool = False
	rejection_reason: str | None = None
	amendment_number: str | None = None  # For amendment submissions


class RegulatorySubmissionCreate(CtrBase):
	tenant_id: str
	trial_id: str
	submission_type: SubmissionType
	authority: RegulatoryAuthority
	cover_letter_reference: str
	dossier_reference: str
	amendment_number: str | None = None
	created_by: str


# ─────────────────────────────────────────────────────────────────────────────
# Report / Aggregation models
# ─────────────────────────────────────────────────────────────────────────────

class TrialSummaryReport(CtrBase):
	"""High-level trial status report."""
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
	generated_at: datetime = Field(default_factory=datetime.utcnow)


class SafetyReport(CtrBase):
	"""Aggregated safety report for a trial."""
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
	generated_at: datetime = Field(default_factory=datetime.utcnow)


class EnrollmentReport(CtrBase):
	"""Enrollment status across sites."""
	trial_id: str
	target_enrollment: int
	total_enrolled: int
	total_randomised: int
	total_screen_failures: int
	enrollment_rate: float  # Per site per month
	projected_completion_date: datetime | None = None
	by_site: list[dict[str, Any]] = Field(default_factory=list)
	by_country: dict[str, int] = Field(default_factory=dict)
	generated_at: datetime = Field(default_factory=datetime.utcnow)


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
	generated_at: datetime = Field(default_factory=datetime.utcnow)


class InterimAnalysisRequest(CtrBase):
	"""Request to generate an interim analysis."""
	id: str = Field(default_factory=_uuid7str)
	trial_id: str
	tenant_id: str
	analysis_number: int
	triggered_by: str
	data_cut_date: datetime
	statistical_method: str
	stopping_rule: str | None = None
	blinded: bool = True
	requested_at: datetime = Field(default_factory=datetime.utcnow)
	completed_at: datetime | None = None
	report_reference: str | None = None


class TrialCloseoutRecord(CtrBase):
	"""Trial close-out checklist and completion record."""
	id: str = Field(default_factory=_uuid7str)
	trial_id: str
	tenant_id: str
	initiated_by: str
	initiated_at: datetime = Field(default_factory=datetime.utcnow)
	checklist: dict[str, bool] = Field(default_factory=dict)
	all_data_locked: bool = False
	tmf_complete: bool = False
	all_sites_closed: bool = False
	final_report_submitted: bool = False
	completed: bool = False
	completed_at: datetime | None = None
