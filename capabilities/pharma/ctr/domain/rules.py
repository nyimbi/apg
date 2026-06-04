"""Deterministic domain rules for Clinical Trials Management.

Single source of truth for all governance decisions within this capability.
All rules are pure functions — no side effects, no I/O.

GCP (ICH E6 R2/R3), ICH E2A, ICH E3, 21 CFR Part 11 compliance enforced.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Exception
# ─────────────────────────────────────────────────────────────────────────────

class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"[{rule_name}] {reason}")


# ─────────────────────────────────────────────────────────────────────────────
# Tenant / access control
# ─────────────────────────────────────────────────────────────────────────────

def assert_tenant_context(context: dict[str, Any]) -> None:
	"""All operations require a tenant context."""
	if not context.get("tenant_id"):
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all operations",
			"attach_tenant_context",
		)


def assert_write_policy(context: dict[str, Any]) -> None:
	"""Write operations require an attached policy."""
	if context.get("operation_type") == "write" and not context.get("policy_attached"):
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached policy",
			"attach_policy",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' may not access resources of tenant '{resource_tenant}'",
			"use_own_tenant_resources",
		)


def assert_gcp_compliance(context: dict[str, Any]) -> None:
	"""All write operations must be GCP-compliant."""
	if context.get("operation_type") == "write" and context.get("gcp_compliant") is False:
		raise RuleViolation(
			"gcp_compliance_required",
			"operation does not meet GCP (ICH E6) requirements",
			"ensure_gcp_compliance",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Trial lifecycle
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_TRIAL_PHASES = frozenset({
	"phase_1", "phase_1b", "phase_2", "phase_2b",
	"phase_3", "phase_3b", "phase_4", "expanded_access", "observational",
})

SUPPORTED_TRIAL_TYPES = frozenset({
	"interventional", "observational", "expanded_access", "registry",
	"bioequivalence", "first_in_human", "basket", "umbrella",
})

ALLOWED_TRIAL_TRANSITIONS: dict[str, frozenset[str]] = {
	"planned":              frozenset({"active", "withdrawn"}),
	"active":               frozenset({"enrolling", "suspended", "terminated"}),
	"enrolling":            frozenset({"enrollment_complete", "suspended", "terminated"}),
	"enrollment_complete":  frozenset({"treatment_ongoing"}),
	"treatment_ongoing":    frozenset({"follow_up", "terminated"}),
	"follow_up":            frozenset({"completed", "terminated"}),
	"suspended":            frozenset({"active", "terminated"}),
	"completed":            frozenset(),
	"terminated":           frozenset(),
	"withdrawn":            frozenset(),
}


def assert_trial_phase_supported(phase: str) -> None:
	if phase not in SUPPORTED_TRIAL_PHASES:
		raise RuleViolation(
			"trial_phase_not_supported",
			f"phase '{phase}' is not in the supported set",
			"select_supported_phase",
		)


def assert_trial_type_supported(trial_type: str) -> None:
	if trial_type not in SUPPORTED_TRIAL_TYPES:
		raise RuleViolation(
			"trial_type_not_supported",
			f"trial type '{trial_type}' is not in the supported set",
			"select_supported_type",
		)


def assert_trial_sponsor_present(sponsor_id: str) -> None:
	if not sponsor_id or not sponsor_id.strip():
		raise RuleViolation(
			"sponsor_required",
			"a sponsor must be identified for every trial",
			"identify_sponsor",
		)


def assert_irb_approval_for_activation(irb_approval_reference: str) -> None:
	"""Trial activation requires a valid IRB approval reference."""
	if not irb_approval_reference or not irb_approval_reference.strip():
		raise RuleViolation(
			"irb_approval_required",
			"trial cannot be activated without IRB/Ethics Committee approval",
			"obtain_irb_approval",
		)


def assert_trial_transition_valid(current_status: str, new_status: str) -> None:
	allowed = ALLOWED_TRIAL_TRANSITIONS.get(current_status, frozenset())
	if new_status not in allowed:
		raise RuleViolation(
			"invalid_trial_status_transition",
			f"cannot transition trial from '{current_status}' to '{new_status}'",
			f"allowed_transitions: {sorted(allowed)}",
		)


def assert_trial_not_terminated(status: str, operation: str) -> None:
	if status in {"terminated", "withdrawn", "completed"}:
		raise RuleViolation(
			"trial_closed",
			f"operation '{operation}' is not permitted on a {status} trial",
			"reopen_trial_or_create_new",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Protocol
# ─────────────────────────────────────────────────────────────────────────────

def assert_protocol_version_present(version: str) -> None:
	if not version or not version.strip():
		raise RuleViolation(
			"protocol_version_required",
			"a protocol version identifier is required",
			"version_protocol",
		)


def assert_protocol_irb_reviewed(irb_approval_reference: str) -> None:
	if not irb_approval_reference or not irb_approval_reference.strip():
		raise RuleViolation(
			"irb_review_required",
			"protocol cannot be approved without IRB review reference",
			"submit_to_irb",
		)


def assert_protocol_not_superseded(status: str) -> None:
	if status == "superseded":
		raise RuleViolation(
			"protocol_superseded",
			"the target protocol version is superseded — use the current version",
			"use_current_protocol_version",
		)


def assert_amendment_rationale_present(rationale: str) -> None:
	if not rationale or not rationale.strip():
		raise RuleViolation(
			"amendment_rationale_required",
			"a rationale must be provided for protocol amendments",
			"provide_amendment_rationale",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Site
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_SITE_STATUSES = frozenset({
	"pre_selected", "selected", "initiated", "enrolling",
	"enrollment_complete", "closed", "terminated", "withdrawn",
})

ALLOWED_SITE_TRANSITIONS: dict[str, frozenset[str]] = {
	"pre_selected":         frozenset({"selected", "withdrawn"}),
	"selected":             frozenset({"initiated", "withdrawn"}),
	"initiated":            frozenset({"enrolling", "closed", "terminated"}),
	"enrolling":            frozenset({"enrollment_complete", "terminated"}),
	"enrollment_complete":  frozenset({"closed"}),
	"closed":               frozenset(),
	"terminated":           frozenset(),
	"withdrawn":            frozenset(),
}


def assert_site_qualification_visit_completed(qualification_visit_date: datetime | None) -> None:
	if qualification_visit_date is None:
		raise RuleViolation(
			"qualification_visit_required",
			"site must complete a qualification visit before initiation",
			"complete_qualification_visit",
		)


def assert_site_initiated_before_enrollment(site_status: str) -> None:
	if site_status not in {"initiated", "enrolling"}:
		raise RuleViolation(
			"site_not_initiated",
			f"site status is '{site_status}' — must be 'initiated' or 'enrolling' to enrol patients",
			"complete_site_initiation",
		)


def assert_site_transition_valid(current_status: str, new_status: str) -> None:
	allowed = ALLOWED_SITE_TRANSITIONS.get(current_status, frozenset())
	if new_status not in allowed:
		raise RuleViolation(
			"invalid_site_status_transition",
			f"cannot transition site from '{current_status}' to '{new_status}'",
			f"allowed_transitions: {sorted(allowed)}",
		)


def assert_site_target_enrollment_positive(target: int) -> None:
	if target <= 0:
		raise RuleViolation(
			"site_target_enrollment_invalid",
			"site target enrollment must be a positive integer",
			"set_positive_target_enrollment",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Subject / Patient
# ─────────────────────────────────────────────────────────────────────────────

def assert_informed_consent_obtained(consent_date: datetime | None) -> None:
	if consent_date is None:
		raise RuleViolation(
			"informed_consent_required",
			"informed consent must be obtained before subject enrolment",
			"obtain_informed_consent",
		)


def assert_consent_date_not_future(consent_date: datetime) -> None:
	if consent_date > datetime.utcnow() + timedelta(minutes=5):  # 5-min grace for clock skew
		raise RuleViolation(
			"consent_date_future",
			"informed consent date cannot be in the future",
			"correct_consent_date",
		)


def assert_eligibility_confirmed(eligibility_met: bool) -> None:
	if not eligibility_met:
		raise RuleViolation(
			"eligibility_confirmation_required",
			"subject has not met all inclusion/exclusion criteria",
			"confirm_eligibility_criteria",
		)


def assert_patient_not_already_randomised(randomisation_code: str | None) -> None:
	if randomisation_code is not None:
		raise RuleViolation(
			"subject_already_randomised",
			"subject already has a randomisation allocation — cannot re-randomise",
			"contact_unblinding_committee",
		)


def assert_patient_enrolled_status(status: str) -> None:
	if status not in {"enrolled", "screened"}:
		raise RuleViolation(
			"subject_not_eligible_for_randomisation",
			f"subject status is '{status}' — must be 'enrolled' for randomisation",
			"enrol_subject_first",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Randomisation
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_RANDOMISATION_METHODS = frozenset({
	"simple", "stratified", "block", "adaptive", "minimisation", "dynamic",
})


def assert_randomisation_method_supported(method: str) -> None:
	if method not in SUPPORTED_RANDOMISATION_METHODS:
		raise RuleViolation(
			"randomisation_method_not_supported",
			f"randomisation method '{method}' is not in the supported set",
			"select_supported_randomisation_method",
		)


def assert_randomisation_code_unique(code: str, existing_codes: set[str]) -> None:
	if code in existing_codes:
		raise RuleViolation(
			"randomisation_code_duplicate",
			f"randomisation code '{code}' is already assigned to another subject",
			"generate_unique_randomisation_code",
		)


def assert_stratification_factors_valid(
	factors: dict[str, str],
	allowed_factors: set[str] | None = None,
) -> None:
	if not factors:
		return  # stratification is optional
	if allowed_factors and not set(factors.keys()).issubset(allowed_factors):
		invalid = set(factors.keys()) - allowed_factors
		raise RuleViolation(
			"invalid_stratification_factor",
			f"unrecognised stratification factors: {invalid}",
			"use_protocol_defined_stratification_factors",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Adverse events
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_AE_SEVERITIES = frozenset({"grade_1", "grade_2", "grade_3", "grade_4", "grade_5"})
SUPPORTED_AE_TYPES = frozenset({
	"adverse_event", "serious_adverse_event",
	"suspected_unexpected_serious_adverse_reaction",
	"disease_related_event", "protocol_deviation",
})
SUSAR_REPORTING_DAYS = 15
SAE_REPORTING_DAYS = 7  # fatal/life-threatening
SAE_NON_FATAL_REPORTING_DAYS = 15


def assert_ae_severity_supported(severity_grade: str) -> None:
	if severity_grade not in SUPPORTED_AE_SEVERITIES:
		raise RuleViolation(
			"ae_severity_not_supported",
			f"AE severity '{severity_grade}' is not in the supported set",
			"select_supported_ae_severity",
		)


def assert_ae_type_supported(ae_type: str) -> None:
	if ae_type not in SUPPORTED_AE_TYPES:
		raise RuleViolation(
			"ae_type_not_supported",
			f"AE type '{ae_type}' is not in the supported set",
			"select_supported_ae_type",
		)


def assert_ae_narrative_present(narrative: str) -> None:
	"""ICH E2A requires a narrative description for all SAEs."""
	if not narrative or not narrative.strip():
		raise RuleViolation(
			"ae_narrative_required",
			"a narrative is required for all adverse event reports (ICH E2A)",
			"provide_ae_narrative",
		)


def assert_sae_reporting_timeline(onset_date: datetime, report_date: datetime, ae_type: str) -> None:
	"""Enforce ICH E2A reporting timelines:
	- SAE (fatal/life-threatening): 7 days
	- SUSAR: 15 days
	- Other SAE: 15 days
	"""
	elapsed_days = (report_date - onset_date).total_seconds() / 86400
	if ae_type == "suspected_unexpected_serious_adverse_reaction":
		if elapsed_days > SUSAR_REPORTING_DAYS:
			raise RuleViolation(
				"susar_reporting_deadline_exceeded",
				f"SUSAR must be reported within {SUSAR_REPORTING_DAYS} days of onset "
				f"(elapsed: {elapsed_days:.1f} days)",
				"expedite_susar_report",
			)
	elif ae_type == "serious_adverse_event":
		deadline = SAE_REPORTING_DAYS if elapsed_days > 7 else SAE_NON_FATAL_REPORTING_DAYS
		if elapsed_days > deadline:
			raise RuleViolation(
				"sae_reporting_deadline_exceeded",
				f"SAE must be reported within {deadline} days of onset "
				f"(elapsed: {elapsed_days:.1f} days)",
				"expedite_sae_report",
			)


def assert_seriousness_criteria_documented(seriousness_criteria: list[str]) -> None:
	"""SAE reports must document the specific seriousness criteria met (ICH E2A)."""
	valid_criteria = {
		"death", "life_threatening", "hospitalisation", "prolonged_hospitalisation",
		"persistent_disability", "congenital_anomaly", "medically_important",
	}
	if not seriousness_criteria:
		raise RuleViolation(
			"seriousness_criteria_required",
			"at least one seriousness criterion must be documented for an SAE",
			"document_seriousness_criteria",
		)
	invalid = set(seriousness_criteria) - valid_criteria
	if invalid:
		raise RuleViolation(
			"invalid_seriousness_criteria",
			f"unrecognised seriousness criteria: {invalid}",
			"use_ich_e2a_seriousness_criteria",
		)


def assert_meddra_coding_present(meddra_pt: str | None) -> None:
	"""MedDRA preferred term coding is required for all SAEs."""
	if not meddra_pt or not meddra_pt.strip():
		raise RuleViolation(
			"meddra_coding_required",
			"MedDRA preferred term (PT) coding is required for adverse events",
			"apply_meddra_coding",
		)


# ─────────────────────────────────────────────────────────────────────────────
# CRF / Data management
# ─────────────────────────────────────────────────────────────────────────────

def assert_data_entry_operator_present(operator: str | None) -> None:
	if not operator or not operator.strip():
		raise RuleViolation(
			"data_entry_operator_required",
			"a data entry operator must be identified for CRF submissions",
			"identify_data_entry_operator",
		)


def assert_no_open_queries_before_lock(open_query_count: int) -> None:
	if open_query_count > 0:
		raise RuleViolation(
			"open_queries_must_be_resolved",
			f"{open_query_count} open data queries must be resolved before database lock",
			"resolve_all_open_queries",
		)


def assert_all_crfs_validated_before_lock(unvalidated_count: int) -> None:
	if unvalidated_count > 0:
		raise RuleViolation(
			"unvalidated_crfs_present",
			f"{unvalidated_count} CRF forms are pending validation — lock not permitted",
			"validate_all_crf_forms",
		)


def assert_database_locked_for_csr(is_locked: bool) -> None:
	if not is_locked:
		raise RuleViolation(
			"database_lock_required_for_csr",
			"the clinical database must be locked before generating a Clinical Study Report",
			"lock_database_first",
		)


def assert_crf_status_allows_edit(status: str) -> None:
	if status in {"locked", "signed_off"}:
		raise RuleViolation(
			"crf_locked_or_signed_off",
			f"CRF with status '{status}' cannot be edited without a formal unlock/re-open",
			"raise_formal_data_change_request",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Monitoring visits
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_MONITORING_VISIT_TYPES = frozenset({
	"qualification", "initiation", "routine", "close_out", "for_cause", "risk_based",
})


def assert_monitoring_visit_type_supported(visit_type: str) -> None:
	if visit_type not in SUPPORTED_MONITORING_VISIT_TYPES:
		raise RuleViolation(
			"monitoring_visit_type_not_supported",
			f"monitoring visit type '{visit_type}' is not in the supported set",
			"select_supported_visit_type",
		)


def assert_monitoring_report_present(report_reference: str | None) -> None:
	if not report_reference or not report_reference.strip():
		raise RuleViolation(
			"monitoring_report_required",
			"a monitoring visit report reference is required to close the visit",
			"submit_monitoring_visit_report",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Regulatory submissions
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_REGULATORY_AUTHORITIES = frozenset({
	"fda", "ema", "mhra", "pmda", "health_canada",
	"tga", "anvisa", "cdsco", "nmpa", "nmra",
})

SUPPORTED_SUBMISSION_TYPES = frozenset({
	"ind", "cta", "protocol_amendment", "annual_report",
	"safety_report", "final_report", "eudract", "ctis",
})


def assert_regulatory_authority_supported(authority: str) -> None:
	if authority not in SUPPORTED_REGULATORY_AUTHORITIES:
		raise RuleViolation(
			"regulatory_authority_not_supported",
			f"regulatory authority '{authority}' is not in the supported set",
			"select_supported_authority",
		)


def assert_submission_type_supported(submission_type: str) -> None:
	if submission_type not in SUPPORTED_SUBMISSION_TYPES:
		raise RuleViolation(
			"submission_type_not_supported",
			f"submission type '{submission_type}' is not in the supported set",
			"select_supported_submission_type",
		)


def assert_submission_cover_letter_present(cover_letter_reference: str) -> None:
	if not cover_letter_reference or not cover_letter_reference.strip():
		raise RuleViolation(
			"cover_letter_required",
			"a cover letter reference is required for regulatory submissions",
			"attach_cover_letter",
		)


def assert_submission_dossier_present(dossier_reference: str) -> None:
	if not dossier_reference or not dossier_reference.strip():
		raise RuleViolation(
			"dossier_reference_required",
			"a dossier reference is required for regulatory submissions",
			"attach_dossier",
		)


# ─────────────────────────────────────────────────────────────────────────────
# TMF
# ─────────────────────────────────────────────────────────────────────────────

def assert_tmf_file_hash_present(file_metadata: dict[str, Any]) -> None:
	"""ICH E6(R3) requires cryptographic integrity proof for all eTMF documents."""
	if "file_hash_sha256" not in file_metadata or not file_metadata["file_hash_sha256"]:
		raise RuleViolation(
			"tmf_file_integrity_hash_required",
			"a SHA-256 file hash is required for TMF document integrity per ICH E6(R3)",
			"compute_and_attach_file_hash",
		)


def assert_tmf_section_valid(section: str) -> None:
	"""TMF section must follow reference model zone numbering."""
	if not section or not section.strip():
		raise RuleViolation(
			"tmf_section_required",
			"a TMF section reference (e.g. '01.01', '03.02.01') is required",
			"specify_tmf_section",
		)


# ─────────────────────────────────────────────────────────────────────────────
# IRB approval
# ─────────────────────────────────────────────────────────────────────────────

def assert_irb_decision_not_rejected(decision: str | None) -> None:
	if decision == "rejected":
		raise RuleViolation(
			"irb_application_rejected",
			"the IRB/Ethics Committee rejected this application — trial cannot proceed",
			"address_irb_rejection_and_resubmit",
		)


def assert_irb_not_expired(expiry_date: datetime | None) -> None:
	if expiry_date is not None and expiry_date < datetime.utcnow():
		raise RuleViolation(
			"irb_approval_expired",
			f"IRB approval expired on {expiry_date.date()} — renewal required",
			"renew_irb_approval",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Trial closeout
# ─────────────────────────────────────────────────────────────────────────────

def assert_all_sites_closed_before_trial_closure(open_site_count: int) -> None:
	if open_site_count > 0:
		raise RuleViolation(
			"open_sites_prevent_closure",
			f"{open_site_count} site(s) are still open — all sites must be closed before trial closeout",
			"close_all_sites_first",
		)


def assert_tmf_complete_before_closure(tmf_completeness_rate: float) -> None:
	if tmf_completeness_rate < 0.95:
		raise RuleViolation(
			"tmf_incomplete",
			f"TMF completeness is {tmf_completeness_rate:.1%} — must reach ≥95% before trial closure",
			"file_missing_tmf_documents",
		)


def assert_final_report_filed_before_closure(final_report_filed: bool) -> None:
	if not final_report_filed:
		raise RuleViolation(
			"final_report_not_filed",
			"the Clinical Study Report (CSR) must be filed before trial closeout",
			"submit_clinical_study_report",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Composite assertion helpers
# ─────────────────────────────────────────────────────────────────────────────

def assert_trial_creation_preconditions(
	phase: str,
	trial_type: str,
	sponsor_id: str,
	target_enrollment: int,
) -> None:
	"""Run all precondition checks for trial creation atomically."""
	assert_trial_phase_supported(phase)
	assert_trial_type_supported(trial_type)
	assert_trial_sponsor_present(sponsor_id)
	if target_enrollment < 0:
		raise RuleViolation(
			"target_enrollment_negative",
			"target enrollment cannot be negative",
			"set_non_negative_target_enrollment",
		)


def assert_patient_enrolment_preconditions(
	site_status: str,
	consent_date: datetime | None,
	eligibility_met: bool,
) -> None:
	"""Run all checks required before enrolling a patient."""
	assert_site_initiated_before_enrollment(site_status)
	assert_informed_consent_obtained(consent_date)
	if consent_date is not None:
		assert_consent_date_not_future(consent_date)
	assert_eligibility_confirmed(eligibility_met)


def assert_sae_report_preconditions(
	severity_grade: str,
	narrative: str,
	seriousness_criteria: list[str],
	onset_date: datetime,
	report_date: datetime,
) -> None:
	"""Run all precondition checks for SAE reporting."""
	assert_ae_severity_supported(severity_grade)
	assert_ae_narrative_present(narrative)
	assert_seriousness_criteria_documented(seriousness_criteria)
	assert_sae_reporting_timeline(onset_date, report_date, "serious_adverse_event")


def calculate_sae_reporting_deadline(
	onset_date: datetime,
	severity_grade: str,
) -> datetime:
	"""Return the regulatory reporting deadline for an SAE.

	- Grade 5 (fatal) or life-threatening: 7 calendar days from onset
	- All other serious: 15 calendar days from onset
	"""
	if severity_grade in {"grade_4", "grade_5"}:
		return onset_date + timedelta(days=7)
	return onset_date + timedelta(days=15)
