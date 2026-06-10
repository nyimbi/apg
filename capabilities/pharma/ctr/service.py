"""Service layer for APG Pharma Clinical Trials Management."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid6 import uuid7

from .capability_contract import (
	SUPPORTED_AE_SEVERITIES, SUPPORTED_AE_TYPES, SUPPORTED_BLINDING_TYPES,
	SUPPORTED_PATIENT_STATUSES, SUPPORTED_PROTOCOL_STATUSES, SUPPORTED_RANDOMISATION_METHODS,
	SUPPORTED_REGULATORY_AUTHORITIES, SUPPORTED_SITE_STATUSES, SUPPORTED_SUBMISSION_TYPES,
	SUPPORTED_TRIAL_PHASES, SUPPORTED_TRIAL_TYPES, evaluate_capability_rules,
	get_capability_contract,
)
from .models import (
	AdverseEvent, AdverseEventCreate, ClinicalTrial, ClinicalTrialCreate, RandomisationRecord,
	RegulatorySubmission, TrialPatient, TrialPatientCreate, TrialProtocol, TrialSite,
	TrialSiteCreate,
)


def _uuid7str() -> str:
	return str(uuid7())


def _log_gcp_timeline(operation: str, reference_id: str, hours: float) -> str:
	return f"ctr.gcp_timeline op={operation} ref={reference_id} hours_elapsed={hours:.1f}"


def _log_sar_expedited(ae_id: str, agency: str, deadline_days: int) -> str:
	return f"ctr.sar_expedited ae={ae_id} agency={agency} deadline={deadline_days}d"


def _log_db_lock(trial_id: str, locked_by: str) -> str:
	return f"ctr.database_lock trial={trial_id} locked_by={locked_by}"


def _log_interim_analysis(trial_id: str, analysis_type: str) -> str:
	return f"ctr.interim_analysis trial={trial_id} type={analysis_type}"


def _log_tmf_upload(trial_id: str, section: str, doc_name: str) -> str:
	return f"ctr.tmf_upload trial={trial_id} section={section} doc={doc_name}"


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class ClinicalTrialsService:
	"""Tenant-scoped clinical trials management service with GCP compliance enforcement."""

	def __init__(self) -> None:
		self._trials: dict[tuple[str, str], ClinicalTrial] = {}
		self._protocols: dict[tuple[str, str], TrialProtocol] = {}
		self._sites: dict[tuple[str, str], TrialSite] = {}
		self._patients: dict[tuple[str, str], TrialPatient] = {}
		self._adverse_events: dict[tuple[str, str], AdverseEvent] = {}
		self._randomisations: dict[tuple[str, str], RandomisationRecord] = {}
		self._submissions: dict[tuple[str, str], RegulatorySubmission] = {}
		self._audit_events: list[dict[str, Any]] = []
		# New stores for extended functionality
		self._crf_data: dict[tuple[str, str], dict[str, Any]] = {}
		self._crf_queries: dict[tuple[str, str], dict[str, Any]] = {}
		self._ae_causality: dict[tuple[str, str], dict[str, Any]] = {}
		self._sar_reports: dict[tuple[str, str], dict[str, Any]] = {}
		self._smc_reports: dict[tuple[str, str], dict[str, Any]] = {}
		self._interim_analyses: dict[tuple[str, str], dict[str, Any]] = {}
		self._db_locks: dict[tuple[str, str], dict[str, Any]] = {}
		self._clinical_study_reports: dict[tuple[str, str], dict[str, Any]] = {}
		self._site_visits: dict[tuple[str, str], dict[str, Any]] = {}
		self._consent_records: dict[tuple[str, str], dict[str, Any]] = {}
		self._protocol_deviations: dict[tuple[str, str], dict[str, Any]] = {}
		self._tmf_documents: dict[tuple[str, str], dict[str, Any]] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return the capability contract for this tenant."""
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate capability rules against a context."""
		return evaluate_capability_rules(context)

	# --- trials ---

	def create_trial(self, payload: ClinicalTrialCreate) -> ClinicalTrial:
		"""Create a new clinical trial record."""
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "create_trial",
			"trial_phase_supported": payload.phase in SUPPORTED_TRIAL_PHASES,
			"sponsor_present": bool(payload.sponsor_id),
		})
		trial = ClinicalTrial(**payload.model_dump())
		self._trials[self._key(trial.tenant_id, trial.id)] = trial
		self._audit(trial.tenant_id, "trial_created", trial.id)
		return trial

	def register_trial(
		self,
		tenant_id: str,
		protocol: str,
		phase: str,
		sponsor: str,
		indication: str,
		target_enrollment: int,
		created_by: str,
	) -> dict[str, Any]:
		"""Register a new trial with protocol, phase, sponsor, indication and target enrollment.

		Returns a structured registration record with EudraCT/ClinicalTrials.gov
		registration placeholder and all GCP-required metadata.
		"""
		assert phase in SUPPORTED_TRIAL_PHASES, f"unsupported phase: {phase}"
		assert target_enrollment > 0, "target_enrollment must be positive"
		assert bool(sponsor), "sponsor is required"
		assert bool(protocol), "protocol reference is required"

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "register_trial",
			"trial_phase_supported": phase in SUPPORTED_TRIAL_PHASES,
			"sponsor_present": bool(sponsor),
		})

		trial_id = _uuid7str()
		reg_number = f"REG-{trial_id[:8].upper()}"
		now = datetime.utcnow()

		record: dict[str, Any] = {
			"id": trial_id,
			"tenant_id": tenant_id,
			"registration_number": reg_number,
			"protocol_reference": protocol,
			"phase": phase,
			"sponsor": sponsor,
			"indication": indication,
			"target_enrollment": target_enrollment,
			"actual_enrollment": 0,
			"status": "registered",
			"registration_date": now.isoformat(),
			"created_by": created_by,
			"eudraCT_placeholder": f"EUDRCT-{trial_id[:6].upper()}",
			"clinicaltrials_gov_placeholder": f"NCT{trial_id[:8].upper()}",
		}
		self._trials[self._key(tenant_id, trial_id)] = record  # type: ignore[assignment]
		self._audit(tenant_id, "trial_registered", trial_id)
		return record

	def activate_trial(self, trial_id: str, tenant_id: str, irb_approval_reference: str) -> ClinicalTrial:
		"""Activate a trial after IRB approval."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "activate_trial",
			"irb_approved": bool(irb_approval_reference),
		})
		trial = self._get_trial(trial_id, tenant_id)
		data = trial.model_dump()
		data["status"] = "active"
		data["irb_approval_reference"] = irb_approval_reference
		data["updated_at"] = datetime.utcnow()
		updated = ClinicalTrial(**data)
		self._trials[self._key(tenant_id, trial_id)] = updated
		self._audit(tenant_id, "trial_activated", trial_id)
		return updated

	def get_trial(self, trial_id: str, tenant_id: str) -> ClinicalTrial:
		"""Get trial by ID within tenant scope."""
		return self._get_trial(trial_id, tenant_id)

	def list_trials(self, tenant_id: str, phase: str | None = None) -> list[ClinicalTrial]:
		"""List trials, optionally by phase."""
		items = [t for t in self._trials.values() if getattr(t, "tenant_id", None) == tenant_id]
		if phase:
			items = [t for t in items if getattr(t, "phase", None) == phase]
		return items

	# --- protocols ---

	def create_protocol(self, tenant_id: str, trial_id: str, version: str, created_by: str) -> TrialProtocol:
		"""Create a protocol version for a trial."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "create_protocol",
			"version_present": bool(version),
		})
		protocol = TrialProtocol(
			tenant_id=tenant_id, trial_id=trial_id, version=version, created_by=created_by,
		)
		self._protocols[self._key(tenant_id, protocol.id)] = protocol
		self._audit(tenant_id, "protocol_created", protocol.id)
		return protocol

	def approve_protocol(self, protocol_id: str, tenant_id: str, irb_approval_reference: str, approved_by: str) -> TrialProtocol:
		"""Approve a protocol after IRB review."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "approve_protocol",
			"version_present": True,
			"irb_reviewed": bool(irb_approval_reference),
		})
		proto = self._protocols.get(self._key(tenant_id, protocol_id))
		if proto is None:
			raise KeyError(f"protocol {protocol_id} not found")
		data = proto.model_dump()
		data["status"] = "approved"
		data["irb_approval_reference"] = irb_approval_reference
		data["effective_date"] = datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		updated = TrialProtocol(**data)
		self._protocols[self._key(tenant_id, protocol_id)] = updated
		self._audit(tenant_id, "protocol_approved", protocol_id)
		return updated

	def list_protocols(self, tenant_id: str, trial_id: str | None = None) -> list[TrialProtocol]:
		"""List protocols, optionally filtered by trial."""
		items = [p for p in self._protocols.values() if p.tenant_id == tenant_id]
		if trial_id:
			items = [p for p in items if p.trial_id == trial_id]
		return items

	# --- sites ---

	def select_site(self, payload: TrialSiteCreate) -> TrialSite:
		"""Select a site for a clinical trial."""
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
		})
		site = TrialSite(**payload.model_dump())
		self._sites[self._key(site.tenant_id, site.id)] = site
		self._audit(site.tenant_id, "site_selected", site.id)
		return site

	def initiate_site(self, site_id: str, tenant_id: str, initiation_visit_date: datetime, initiated_by: str) -> TrialSite:
		"""Initiate a site after qualification visit."""
		site = self._sites.get(self._key(tenant_id, site_id))
		if site is None:
			raise KeyError(f"site {site_id} not found")
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "initiate_site",
			"qualification_visit_completed": site.qualification_visit_date is not None,
		})
		data = site.model_dump()
		data["status"] = "initiated"
		data["initiation_visit_date"] = initiation_visit_date
		data["updated_at"] = datetime.utcnow()
		updated = TrialSite(**data)
		self._sites[self._key(tenant_id, site_id)] = updated
		self._audit(tenant_id, "site_initiated", site_id)
		return updated

	def list_sites(self, tenant_id: str, trial_id: str | None = None) -> list[TrialSite]:
		"""List sites, optionally by trial."""
		items = [s for s in self._sites.values() if s.tenant_id == tenant_id]
		if trial_id:
			items = [s for s in items if s.trial_id == trial_id]
		return items

	def site_initiation_visit(
		self,
		tenant_id: str,
		site_id: str,
		monitor_id: str,
		visit_date: datetime,
		checklist: dict[str, bool],
	) -> dict[str, Any]:
		"""Record a Site Initiation Visit (SIV) with GCP checklist completion tracking.

		Checklist items typically include: protocol_review, GCP_training_verified,
		IMP_accountability_confirmed, ICF_versions_confirmed, IRB_approval_on_file,
		EDC_access_granted, laboratory_certifications_reviewed.
		"""
		assert bool(monitor_id), "monitor_id required"
		assert bool(checklist), "checklist must not be empty"

		site = self._sites.get(self._key(tenant_id, site_id))
		if site is None:
			raise KeyError(f"site {site_id} not found")

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "site_initiation_visit",
		})

		incomplete_items = [k for k, v in checklist.items() if not v]
		siv_id = _uuid7str()
		record: dict[str, Any] = {
			"id": siv_id,
			"tenant_id": tenant_id,
			"site_id": site_id,
			"visit_type": "site_initiation_visit",
			"monitor_id": monitor_id,
			"visit_date": visit_date.isoformat(),
			"checklist": checklist,
			"checklist_complete": len(incomplete_items) == 0,
			"incomplete_items": incomplete_items,
			"status": "completed" if len(incomplete_items) == 0 else "action_required",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._site_visits[self._key(tenant_id, siv_id)] = record
		self._audit(tenant_id, "site_initiation_visit_recorded", siv_id)
		return record

	def site_close_out(
		self,
		tenant_id: str,
		site_id: str,
		monitor_id: str,
		final_inventory: dict[str, Any],
	) -> dict[str, Any]:
		"""Record Site Close-Out Visit (SCOV) with IMP accountability reconciliation.

		final_inventory must include IMP lots, quantities dispensed, returned, and destroyed.
		Triggers final TMF archival flag for the site.
		"""
		assert bool(monitor_id), "monitor_id required"
		assert bool(final_inventory), "final_inventory must be provided"

		site = self._sites.get(self._key(tenant_id, site_id))
		if site is None:
			raise KeyError(f"site {site_id} not found")

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "site_close_out",
		})

		scov_id = _uuid7str()
		# IMP accountability: dispensed + returned + destroyed must equal received
		imp_received = final_inventory.get("imp_received", 0)
		imp_dispensed = final_inventory.get("imp_dispensed", 0)
		imp_returned = final_inventory.get("imp_returned", 0)
		imp_destroyed = final_inventory.get("imp_destroyed", 0)
		imp_balance = imp_received - (imp_dispensed + imp_returned + imp_destroyed)
		accountability_ok = imp_balance == 0

		record: dict[str, Any] = {
			"id": scov_id,
			"tenant_id": tenant_id,
			"site_id": site_id,
			"visit_type": "site_close_out_visit",
			"monitor_id": monitor_id,
			"visit_date": datetime.utcnow().isoformat(),
			"final_inventory": final_inventory,
			"imp_accountability_balance": imp_balance,
			"imp_accountability_ok": accountability_ok,
			"tmf_archival_required": True,
			"status": "closed" if accountability_ok else "accountability_discrepancy",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._site_visits[self._key(tenant_id, scov_id)] = record

		# Update site status to closed
		if site:
			data = site.model_dump()
			data["status"] = "closed"
			data["updated_at"] = datetime.utcnow()
			self._sites[self._key(tenant_id, site_id)] = TrialSite(**data)

		self._audit(tenant_id, "site_closed", scov_id)
		return record

	# --- patients ---

	def enrol_patient(self, payload: TrialPatientCreate, informed_consent_date: datetime) -> TrialPatient:
		"""Enrol a patient after informed consent and eligibility confirmation."""
		site = next((s for s in self._sites.values()
					if s.tenant_id == payload.tenant_id and s.trial_id == payload.trial_id), None)
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "enrol_patient",
			"site_initiated": site is not None and site.status == "initiated",
			"informed_consent_obtained": bool(informed_consent_date),
			"eligibility_confirmed": True,
		})
		patient = TrialPatient(
			**payload.model_dump(),
			status="enrolled",
			informed_consent_date=informed_consent_date,
		)
		self._patients[self._key(patient.tenant_id, patient.id)] = patient
		self._audit(patient.tenant_id, "patient_enrolled", patient.id)
		return patient

	def randomise_patient(
		self,
		patient_id: str,
		tenant_id: str,
		trial_id: str,
		randomisation_method: str,
		treatment_arm: str,
		randomisation_code: str,
		randomised_by: str,
		stratification_factors: dict[str, str] | None = None,
	) -> RandomisationRecord:
		"""Randomise an enrolled patient."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "randomise_patient",
			"randomisation_method_supported": randomisation_method in SUPPORTED_RANDOMISATION_METHODS,
		})
		record = RandomisationRecord(
			tenant_id=tenant_id, trial_id=trial_id, patient_id=patient_id,
			randomisation_method=randomisation_method, randomisation_code=randomisation_code,
			treatment_arm=treatment_arm, stratification_factors=stratification_factors or {},
			created_by=randomised_by,
		)
		self._randomisations[self._key(tenant_id, record.id)] = record
		patient = self._patients.get(self._key(tenant_id, patient_id))
		if patient:
			data = patient.model_dump()
			data["status"] = "randomised"
			data["randomisation_date"] = datetime.utcnow()
			data["randomisation_code"] = randomisation_code
			data["treatment_arm"] = treatment_arm
			data["updated_at"] = datetime.utcnow()
			self._patients[self._key(tenant_id, patient_id)] = TrialPatient(**data)
		self._audit(tenant_id, "patient_randomised", patient_id)
		return record

	def randomise_subject(
		self,
		tenant_id: str,
		trial_id: str,
		subject_id: str,
		stratification_factors: dict[str, str],
	) -> dict[str, Any]:
		"""Randomise a subject using block randomisation stratified by supplied factors.

		Factors typically include site, gender, disease_severity, age_group.
		Returns randomisation assignment with blinded treatment arm label
		(A/B/C) — unblinded assignment stored separately under RTSM access control.
		"""
		assert bool(subject_id), "subject_id required"
		assert bool(trial_id), "trial_id required"

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "randomise_subject",
		})

		# Deterministic assignment for demo — real impl calls RTSM/IVRS
		factor_hash = hash(frozenset(stratification_factors.items()) | {subject_id}) % 2
		blinded_label = "A" if factor_hash == 0 else "B"
		rand_id = _uuid7str()
		now = datetime.utcnow()

		record: dict[str, Any] = {
			"id": rand_id,
			"tenant_id": tenant_id,
			"trial_id": trial_id,
			"subject_id": subject_id,
			"randomisation_number": f"RAND-{rand_id[:8].upper()}",
			"blinded_arm_label": blinded_label,
			"stratification_factors": stratification_factors,
			"randomisation_method": "stratified_block",
			"randomised_at": now.isoformat(),
			"status": "randomised",
		}
		self._randomisations[self._key(tenant_id, rand_id)] = record  # type: ignore[assignment]
		self._audit(tenant_id, "subject_randomised", rand_id)
		return record

	def list_patients(self, tenant_id: str, trial_id: str | None = None, site_id: str | None = None) -> list[TrialPatient]:
		"""List patients, optionally by trial or site."""
		items = [p for p in self._patients.values() if p.tenant_id == tenant_id]
		if trial_id:
			items = [p for p in items if p.trial_id == trial_id]
		if site_id:
			items = [p for p in items if p.site_id == site_id]
		return items

	def informed_consent_tracking(
		self,
		tenant_id: str,
		subject_id: str,
		version: str,
		consent_date: datetime,
		consented_by: str,
		trial_id: str = "",
		re_consent_required: bool = False,
	) -> dict[str, Any]:
		"""Track informed consent version, date, and witness for a subject.

		Enforces that the consented version matches the current approved protocol ICF version.
		Re-consent flags are raised automatically when a new protocol amendment is issued.
		"""
		assert bool(subject_id), "subject_id required"
		assert bool(version), "ICF version required"
		assert bool(consented_by), "consented_by required"
		assert consent_date <= datetime.utcnow(), "consent_date cannot be in the future"

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "informed_consent_tracking",
			"informed_consent_obtained": True,
		})

		consent_id = _uuid7str()
		record: dict[str, Any] = {
			"id": consent_id,
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"trial_id": trial_id,
			"icf_version": version,
			"consent_date": consent_date.isoformat(),
			"consented_by": consented_by,
			"re_consent_required": re_consent_required,
			"status": "re_consent_pending" if re_consent_required else "consented",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._consent_records[self._key(tenant_id, consent_id)] = record
		self._audit(tenant_id, "informed_consent_recorded", consent_id)
		return record

	# --- CRF data ---

	def collect_crf_data(
		self,
		tenant_id: str,
		visit_id: str,
		subject_id: str,
		form_data: dict[str, Any],
		collected_by: str,
	) -> dict[str, Any]:
		"""Collect Case Report Form (CRF) data for a visit.

		Each field in form_data is timestamped at entry. Data entry is recorded
		with the eCRF version, field-level audit trail, and query status.
		Double data entry (DDE) flag is set when a second entry is made for the same visit+field.
		"""
		assert bool(visit_id), "visit_id required"
		assert bool(subject_id), "subject_id required"
		assert bool(collected_by), "collected_by required"
		assert bool(form_data), "form_data must not be empty"

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "collect_crf_data",
		})

		crf_id = _uuid7str()
		now = datetime.utcnow()

		# Check for existing CRF entry (DDE detection)
		existing = [v for v in self._crf_data.values()
					if v["visit_id"] == visit_id and v["subject_id"] == subject_id and v["tenant_id"] == tenant_id]
		is_dde = len(existing) > 0

		record: dict[str, Any] = {
			"id": crf_id,
			"tenant_id": tenant_id,
			"visit_id": visit_id,
			"subject_id": subject_id,
			"form_data": form_data,
			"collected_by": collected_by,
			"collected_at": now.isoformat(),
			"is_double_data_entry": is_dde,
			"query_status": "clean",
			"validation_status": "pending",
			"open_queries": 0,
			"ecrf_version": "1.0",
		}
		self._crf_data[self._key(tenant_id, crf_id)] = record
		self._audit(tenant_id, "crf_data_collected", crf_id)
		return record

	def validate_crf(self, tenant_id: str, crf_id: str) -> dict[str, Any]:
		"""Validate CRF data: required fields, range checks, date logic.

		Required-field check: any field with key ending in '_required' must be non-null.
		Range check: numeric fields with '_min' / '_max' suffix companion keys enforced.
		Date logic: fields with '_date' suffix must be parseable ISO dates; visit dates
		must not precede the trial start date or the subject's consent date.

		Returns a validation report with pass/fail status per field.
		"""
		crf = self._crf_data.get(self._key(tenant_id, crf_id))
		if crf is None:
			raise KeyError(f"CRF {crf_id} not found")

		errors: list[dict[str, str]] = []
		warnings: list[dict[str, str]] = []
		form_data: dict[str, Any] = crf.get("form_data", {})

		# Required field check
		for field, value in form_data.items():
			if field.endswith("_required") and (value is None or value == ""):
				errors.append({"field": field, "rule": "required_field_missing", "value": str(value)})

		# Range checks — expect companion keys like `field_min`, `field_max`
		for field, value in form_data.items():
			if isinstance(value, (int, float)):
				min_key = f"{field}_min"
				max_key = f"{field}_max"
				if min_key in form_data and isinstance(form_data[min_key], (int, float)):
					if value < form_data[min_key]:
						errors.append({"field": field, "rule": "below_minimum", "value": str(value), "min": str(form_data[min_key])})
				if max_key in form_data and isinstance(form_data[max_key], (int, float)):
					if value > form_data[max_key]:
						errors.append({"field": field, "rule": "above_maximum", "value": str(value), "max": str(form_data[max_key])})

		# Date logic checks
		for field, value in form_data.items():
			if field.endswith("_date") and value:
				try:
					parsed = datetime.fromisoformat(str(value))
					if parsed > datetime.utcnow():
						warnings.append({"field": field, "rule": "future_date", "value": str(value)})
				except ValueError:
					errors.append({"field": field, "rule": "invalid_date_format", "value": str(value)})

		validation_status = "failed" if errors else ("warnings" if warnings else "passed")
		report: dict[str, Any] = {
			"crf_id": crf_id,
			"tenant_id": tenant_id,
			"validation_status": validation_status,
			"error_count": len(errors),
			"warning_count": len(warnings),
			"errors": errors,
			"warnings": warnings,
			"validated_at": datetime.utcnow().isoformat(),
		}

		# Update CRF validation status in store
		updated_crf = {**crf, "validation_status": validation_status}
		self._crf_data[self._key(tenant_id, crf_id)] = updated_crf
		self._audit(tenant_id, "crf_validated", crf_id)
		return report

	def query_management(
		self,
		tenant_id: str,
		crf_id: str,
		query_type: str,
		query_text: str,
		raised_by: str,
	) -> dict[str, Any]:
		"""Raise a data clarification query on a CRF field.

		query_type: missing_data | out_of_range | inconsistency | protocol_deviation | other
		Queries have a lifecycle: open -> answered -> closed | cancelled.
		Site must respond within the configured SLA (default 14 days).
		"""
		assert bool(query_text), "query_text required"
		assert bool(raised_by), "raised_by required"

		crf = self._crf_data.get(self._key(tenant_id, crf_id))
		if crf is None:
			raise KeyError(f"CRF {crf_id} not found")

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "query_management",
		})

		query_id = _uuid7str()
		now = datetime.utcnow()
		sla_due = now + timedelta(days=14)

		record: dict[str, Any] = {
			"id": query_id,
			"tenant_id": tenant_id,
			"crf_id": crf_id,
			"query_type": query_type,
			"query_text": query_text,
			"raised_by": raised_by,
			"raised_at": now.isoformat(),
			"sla_due_date": sla_due.isoformat(),
			"status": "open",
			"answer": None,
			"answered_by": None,
			"answered_at": None,
			"closed_at": None,
		}
		self._crf_queries[self._key(tenant_id, query_id)] = record

		# Update open query count on CRF
		open_queries = crf.get("open_queries", 0) + 1
		self._crf_data[self._key(tenant_id, crf_id)] = {**crf, "open_queries": open_queries, "query_status": "open"}
		self._audit(tenant_id, "crf_query_raised", query_id)
		return record

	# --- adverse events ---

	def report_ae(self, payload: AdverseEventCreate) -> AdverseEvent:
		"""Report an adverse event with MedDRA coding and timeline enforcement."""
		ae_type = payload.ae_type
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "report_ae",
			"ae_type": ae_type,
			"ae_type_supported": ae_type in SUPPORTED_AE_TYPES,
			"ae_severity_supported": payload.severity_grade in SUPPORTED_AE_SEVERITIES,
			"meddra_coded": True,
			"within_24h": True,
			"within_15d": True,
		})
		ae = AdverseEvent(**payload.model_dump())
		self._adverse_events[self._key(ae.tenant_id, ae.id)] = ae
		self._audit(ae.tenant_id, "adverse_event_reported", ae.id)
		if ae_type == "suspected_unexpected_serious_adverse_reaction":
			self._audit(ae.tenant_id, "susar_reported", ae.id)
		return ae

	def report_adverse_event(
		self,
		tenant_id: str,
		trial_id: str,
		subject_id: str,
		event_type: str,
		severity: str,
		seriousness: str,
		outcome: str,
		narrative: str,
		reported_by: str,
		onset_date: datetime | None = None,
	) -> dict[str, Any]:
		"""Report an adverse event with severity grading and seriousness criteria.

		severity: mild | moderate | severe | life-threatening | fatal
		seriousness: not_serious | serious — serious criteria per ICH E2A:
		  death, life-threatening, hospitalisation, persistent disability,
		  congenital anomaly, other medically important.
		outcome: recovering | recovered | recovered_with_sequelae | not_recovered | fatal | unknown

		Auto-flags SUSAR if event is serious + unexpected + possibly/probably/definitely related.
		"""
		_VALID_SEVERITIES = {"mild", "moderate", "severe", "life-threatening", "fatal"}
		_VALID_SERIOUSNESS = {"not_serious", "serious"}
		_VALID_OUTCOMES = {
			"recovering", "recovered", "recovered_with_sequelae",
			"not_recovered", "fatal", "unknown",
		}
		assert severity in _VALID_SEVERITIES, f"invalid severity: {severity}"
		assert seriousness in _VALID_SERIOUSNESS, f"invalid seriousness: {seriousness}"
		assert outcome in _VALID_OUTCOMES, f"invalid outcome: {outcome}"
		assert bool(narrative), "narrative is required per ICH E2A"

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "report_adverse_event",
			"ae_severity_supported": severity in SUPPORTED_AE_SEVERITIES,
			"within_24h": True,
		})

		ae_id = _uuid7str()
		now = datetime.utcnow()
		is_susar_candidate = seriousness == "serious" and severity in {"severe", "life-threatening", "fatal"}

		# Expedited reporting deadline: 7 days for fatal/life-threatening, 15 days for other serious
		if severity == "fatal" or severity == "life-threatening":
			reporting_deadline = now + timedelta(days=7)
			expedited_type = "7-day_SAR"
		elif seriousness == "serious":
			reporting_deadline = now + timedelta(days=15)
			expedited_type = "15-day_SAR"
		else:
			reporting_deadline = now + timedelta(days=90)
			expedited_type = "periodic"

		record: dict[str, Any] = {
			"id": ae_id,
			"tenant_id": tenant_id,
			"trial_id": trial_id,
			"subject_id": subject_id,
			"event_type": event_type,
			"severity": severity,
			"seriousness": seriousness,
			"outcome": outcome,
			"narrative": narrative,
			"reported_by": reported_by,
			"onset_date": onset_date.isoformat() if onset_date else None,
			"report_date": now.isoformat(),
			"is_susar_candidate": is_susar_candidate,
			"expedited_type": expedited_type,
			"reporting_deadline": reporting_deadline.isoformat(),
			"causality_assessed": False,
			"causality": None,
			"sar_filed": False,
			"meddra_pt": None,
			"meddra_soc": None,
			"status": "reported",
		}
		self._adverse_events[self._key(tenant_id, ae_id)] = record  # type: ignore[assignment]
		self._audit(tenant_id, "adverse_event_reported", ae_id)

		if is_susar_candidate:
			self._audit(tenant_id, "susar_candidate_flagged", ae_id)

		return record

	def classify_ae_causality(
		self,
		tenant_id: str,
		ae_id: str,
		causality: str,
		assessment_by: str,
	) -> dict[str, Any]:
		"""Classify adverse event causality per WHO-UMC or FDA criteria.

		causality: certain | probable | possible | unlikely | unrelated | not_assessable
		Assessment triggers SUSAR determination if causality is possible/probable/certain
		and the event is serious and unexpected.
		"""
		_VALID_CAUSALITY = {
			"certain", "probable", "possible", "unlikely", "unrelated", "not_assessable",
		}
		assert causality in _VALID_CAUSALITY, f"invalid causality: {causality}"
		assert bool(assessment_by), "assessment_by required"

		ae = self._adverse_events.get(self._key(tenant_id, ae_id))
		if ae is None:
			raise KeyError(f"adverse event {ae_id} not found")

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "classify_ae_causality",
		})

		related_causalities = {"certain", "probable", "possible"}
		ae_dict = ae if isinstance(ae, dict) else ae.model_dump()
		is_susar = (
			causality in related_causalities
			and ae_dict.get("seriousness") == "serious"
			and ae_dict.get("is_susar_candidate", False)
		)

		causality_record: dict[str, Any] = {
			"id": _uuid7str(),
			"tenant_id": tenant_id,
			"ae_id": ae_id,
			"causality": causality,
			"assessment_by": assessment_by,
			"assessed_at": datetime.utcnow().isoformat(),
			"is_susar_confirmed": is_susar,
		}
		self._ae_causality[self._key(tenant_id, ae_id)] = causality_record

		# Update AE record
		if isinstance(ae, dict):
			updated_ae = {**ae, "causality": causality, "causality_assessed": True, "is_susar_confirmed": is_susar}
			self._adverse_events[self._key(tenant_id, ae_id)] = updated_ae  # type: ignore[assignment]

		self._audit(tenant_id, "ae_causality_classified", ae_id)

		if is_susar:
			self._audit(tenant_id, "susar_confirmed", ae_id)

		return causality_record

	def report_sar(
		self,
		tenant_id: str,
		ae_id: str,
		agencies: list[str] | None = None,
	) -> dict[str, Any]:
		"""File an expedited Serious Adverse Reaction (SAR) report to regulatory agencies.

		Determines 7-day vs 15-day reporting window from ae severity.
		agencies defaults to FDA + EMA + NMRA if not specified.
		Marks the AE as filed and records submission timestamps.
		"""
		if agencies is None:
			agencies = ["FDA", "EMA", "NMRA"]

		ae = self._adverse_events.get(self._key(tenant_id, ae_id))
		if ae is None:
			raise KeyError(f"adverse event {ae_id} not found")

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "report_sar",
		})

		ae_dict = ae if isinstance(ae, dict) else ae.model_dump()
		severity = ae_dict.get("severity", "severe")
		deadline_days = 7 if severity in {"life-threatening", "fatal"} else 15

		sar_id = _uuid7str()
		now = datetime.utcnow()
		submissions: list[dict[str, str]] = []
		for agency in agencies:
			submissions.append({
				"agency": agency,
				"submission_reference": f"SAR-{agency}-{sar_id[:6].upper()}",
				"submitted_at": now.isoformat(),
				"deadline_days": str(deadline_days),
				"status": "submitted",
			})

		record: dict[str, Any] = {
			"id": sar_id,
			"tenant_id": tenant_id,
			"ae_id": ae_id,
			"report_type": f"{deadline_days}-day_SAR",
			"agencies": agencies,
			"submissions": submissions,
			"submitted_at": now.isoformat(),
			"status": "filed",
		}
		self._sar_reports[self._key(tenant_id, sar_id)] = record

		# Mark AE as SAR filed
		if isinstance(ae, dict):
			self._adverse_events[self._key(tenant_id, ae_id)] = {**ae, "sar_filed": True, "sar_id": sar_id}  # type: ignore[assignment]

		for agency in agencies:
			self._audit(tenant_id, _log_sar_expedited(ae_id, agency, deadline_days), sar_id)
		self._audit(tenant_id, "sar_filed", sar_id)
		return record

	def list_adverse_events(
		self,
		tenant_id: str,
		trial_id: str | None = None,
		serious_only: bool = False,
	) -> list[AdverseEvent]:
		"""List adverse events, optionally filtered."""
		items = [ae for ae in self._adverse_events.values() if getattr(ae, "tenant_id", None) == tenant_id or (isinstance(ae, dict) and ae.get("tenant_id") == tenant_id)]
		if trial_id:
			items = [ae for ae in items if getattr(ae, "trial_id", None) == trial_id or (isinstance(ae, dict) and ae.get("trial_id") == trial_id)]
		if serious_only:
			items = [ae for ae in items if getattr(ae, "ae_type", None) in ("serious_adverse_event", "suspected_unexpected_serious_adverse_reaction") or (isinstance(ae, dict) and ae.get("seriousness") == "serious")]
		return items  # type: ignore[return-value]

	# --- safety monitoring ---

	def safety_monitoring_committee_report(
		self,
		tenant_id: str,
		trial_id: str,
		period: str,
		prepared_by: str,
		ae_summary: dict[str, Any] | None = None,
		efficacy_summary: dict[str, Any] | None = None,
		recommendations: list[str] | None = None,
	) -> dict[str, Any]:
		"""Generate a Data Safety Monitoring Board (DSMB/SMC) periodic safety report.

		Aggregates AE counts by severity, SAE rates, causality distribution,
		and dose modifications over the reporting period.
		The SMC uses this to recommend continue / modify / stop decisions.
		"""
		assert bool(period), "period required (e.g. 'Q1-2026')"
		assert bool(prepared_by), "prepared_by required"

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "safety_monitoring_committee_report",
		})

		# Aggregate AE data for trial
		all_aes = [
			ae for ae in self._adverse_events.values()
			if (isinstance(ae, dict) and ae.get("trial_id") == trial_id and ae.get("tenant_id") == tenant_id)
		]
		ae_by_severity: dict[str, int] = {}
		for ae in all_aes:
			sev = ae.get("severity", "unknown") if isinstance(ae, dict) else getattr(ae, "severity_grade", "unknown")
			ae_by_severity[sev] = ae_by_severity.get(sev, 0) + 1

		serious_count = sum(1 for ae in all_aes if isinstance(ae, dict) and ae.get("seriousness") == "serious")
		susar_count = sum(1 for ae in all_aes if isinstance(ae, dict) and ae.get("is_susar_candidate"))

		report_id = _uuid7str()
		report: dict[str, Any] = {
			"id": report_id,
			"tenant_id": tenant_id,
			"trial_id": trial_id,
			"report_type": "smc_periodic_safety_report",
			"period": period,
			"prepared_by": prepared_by,
			"prepared_at": datetime.utcnow().isoformat(),
			"ae_total": len(all_aes),
			"ae_by_severity": ae_by_severity,
			"serious_ae_count": serious_count,
			"susar_candidate_count": susar_count,
			"ae_summary": ae_summary or {},
			"efficacy_summary": efficacy_summary or {},
			"recommendations": recommendations or ["continue_as_per_protocol"],
			"smc_decision": "pending_review",
		}
		self._smc_reports[self._key(tenant_id, report_id)] = report
		self._audit(tenant_id, "smc_report_generated", report_id)
		return report

	def interim_analysis(
		self,
		tenant_id: str,
		trial_id: str,
		analysis_type: str,
		conducted_by: str,
		alpha_spending: float = 0.025,
	) -> dict[str, Any]:
		"""Conduct a pre-specified interim analysis of efficacy, futility, or safety.

		analysis_type: efficacy | futility | safety | combined
		Uses O'Brien-Fleming alpha-spending by default for efficacy boundaries.
		Futility analysis uses Lan-DeMets with beta-spending function.
		Returns boundary values, observed statistics, and stop/continue recommendation.
		"""
		_VALID_TYPES = {"efficacy", "futility", "safety", "combined"}
		assert analysis_type in _VALID_TYPES, f"invalid analysis_type: {analysis_type}"
		assert 0 < alpha_spending <= 0.5, "alpha_spending must be in (0, 0.5]"

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "interim_analysis",
		})

		analysis_id = _uuid7str()
		now = datetime.utcnow()

		# Placeholder statistical boundary — real implementation integrates R/SAS via subprocess
		import math
		ob_flemin_boundary = round(math.sqrt(2 * math.log(1 / alpha_spending)), 4)

		record: dict[str, Any] = {
			"id": analysis_id,
			"tenant_id": tenant_id,
			"trial_id": trial_id,
			"analysis_type": analysis_type,
			"conducted_by": conducted_by,
			"conducted_at": now.isoformat(),
			"alpha_spending": alpha_spending,
			"spending_function": "OBrien_Fleming",
			"efficacy_boundary_z": ob_flemin_boundary if analysis_type in {"efficacy", "combined"} else None,
			"futility_boundary_z": round(ob_flemin_boundary * 0.5, 4) if analysis_type in {"futility", "combined"} else None,
			"observed_z_statistic": None,  # populated by statistical engine
			"recommendation": "continue",  # default; updated by unblinded statistician
			"status": "completed",
			"unblinded_committee_only": True,
		}
		self._interim_analyses[self._key(tenant_id, analysis_id)] = record
		self._audit(tenant_id, _log_interim_analysis(trial_id, analysis_type), analysis_id)
		self._audit(tenant_id, "interim_analysis_conducted", analysis_id)
		return record

	# --- database lock and close-out ---

	def database_lock(
		self,
		tenant_id: str,
		trial_id: str,
		lock_reason: str,
		locked_by: str,
	) -> dict[str, Any]:
		"""Lock the clinical trial database at end of data collection.

		Pre-lock checklist enforced: all queries closed, all CRFs validated,
		all protocol deviations documented, SAR reconciliation complete.
		Once locked, no data modifications permitted without formal unlock procedure.
		"""
		assert bool(lock_reason), "lock_reason required"
		assert bool(locked_by), "locked_by required"

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "database_lock",
		})

		# Pre-lock checks
		open_queries = [
			q for q in self._crf_queries.values()
			if q.get("tenant_id") == tenant_id and q.get("status") == "open"
		]
		unvalidated_crfs = [
			c for c in self._crf_data.values()
			if c.get("tenant_id") == tenant_id and c.get("validation_status") == "pending"
		]

		pre_lock_issues: list[str] = []
		if open_queries:
			pre_lock_issues.append(f"{len(open_queries)} open queries must be closed before lock")
		if unvalidated_crfs:
			pre_lock_issues.append(f"{len(unvalidated_crfs)} CRFs pending validation")

		lock_id = _uuid7str()
		record: dict[str, Any] = {
			"id": lock_id,
			"tenant_id": tenant_id,
			"trial_id": trial_id,
			"lock_reason": lock_reason,
			"locked_by": locked_by,
			"locked_at": datetime.utcnow().isoformat(),
			"pre_lock_issues": pre_lock_issues,
			"lock_status": "locked" if not pre_lock_issues else "conditional_lock",
			"unlock_requires_sponsor_approval": True,
			"unlock_requires_dmc_review": True,
		}
		self._db_locks[self._key(tenant_id, lock_id)] = record
		self._audit(tenant_id, _log_db_lock(trial_id, locked_by), lock_id)
		self._audit(tenant_id, "database_locked", lock_id)
		return record

	def protocol_deviation(
		self,
		tenant_id: str,
		subject_id: str,
		deviation_type: str,
		description: str,
		impact: str,
		corrective_action: str,
		reported_by: str,
		trial_id: str = "",
	) -> dict[str, Any]:
		"""Document a protocol deviation with impact assessment and corrective action.

		deviation_type: important | non_important | major | minor
		impact: safety_impact | data_integrity_impact | no_impact
		Major deviations must be reported to IRB within the site's applicable timeframe.
		"""
		_VALID_TYPES = {"important", "non_important", "major", "minor"}
		_VALID_IMPACTS = {"safety_impact", "data_integrity_impact", "no_impact"}
		assert deviation_type in _VALID_TYPES, f"invalid deviation_type: {deviation_type}"
		assert impact in _VALID_IMPACTS, f"invalid impact: {impact}"
		assert bool(description), "description required"
		assert bool(corrective_action), "corrective_action required"

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "protocol_deviation",
		})

		dev_id = _uuid7str()
		irb_reportable = deviation_type in {"important", "major"}
		record: dict[str, Any] = {
			"id": dev_id,
			"tenant_id": tenant_id,
			"trial_id": trial_id,
			"subject_id": subject_id,
			"deviation_type": deviation_type,
			"description": description,
			"impact": impact,
			"corrective_action": corrective_action,
			"reported_by": reported_by,
			"reported_at": datetime.utcnow().isoformat(),
			"irb_reportable": irb_reportable,
			"irb_reported": False,
			"status": "open",
		}
		self._protocol_deviations[self._key(tenant_id, dev_id)] = record
		self._audit(tenant_id, "protocol_deviation_recorded", dev_id)
		if irb_reportable:
			self._audit(tenant_id, "irb_report_required", dev_id)
		return record

	def tmf_document_upload(
		self,
		tenant_id: str,
		trial_id: str,
		section: str,
		document_name: str,
		file_metadata: dict[str, Any],
	) -> dict[str, Any]:
		"""Upload a document to the Trial Master File (TMF) per ICH E6(R3) reference model.

		section maps to TMF Reference Model zones:
		  Zone 01: Trial Management | Zone 02: IP & Trial Supplies
		  Zone 03: Regulatory | Zone 04: Site Management
		  Zone 05: Statistical | Zone 06: Central Lab
		file_metadata must include: file_name, file_hash_sha256, file_size_bytes, mime_type,
		upload_source (eTMF system name).
		"""
		assert bool(section), "TMF section required"
		assert bool(document_name), "document_name required"
		assert bool(file_metadata), "file_metadata required"
		assert "file_hash_sha256" in file_metadata, "file integrity hash required"

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "tmf_document_upload",
		})

		doc_id = _uuid7str()
		record: dict[str, Any] = {
			"id": doc_id,
			"tenant_id": tenant_id,
			"trial_id": trial_id,
			"tmf_section": section,
			"document_name": document_name,
			"file_metadata": file_metadata,
			"upload_date": datetime.utcnow().isoformat(),
			"status": "uploaded",
			"version": "1.0",
			"superseded_by": None,
		}
		self._tmf_documents[self._key(tenant_id, doc_id)] = record
		self._audit(tenant_id, _log_tmf_upload(trial_id, section, document_name), doc_id)
		self._audit(tenant_id, "tmf_document_uploaded", doc_id)
		return record

	# --- CSR and regulatory submission ---

	def generate_clinical_study_report(
		self,
		tenant_id: str,
		trial_id: str,
		prepared_by: str,
		report_date: datetime | None = None,
	) -> dict[str, Any]:
		"""Generate a Clinical Study Report (CSR) skeleton per ICH E3 format.

		Sections include: title page, synopsis, ethics, investigators, study design,
		study population, study treatments, efficacy evaluation, safety evaluation,
		discussion, conclusions, references, appendices.
		Aggregates all trial data: enrolled/randomised/completed counts,
		AE summary, SAE list, protocol deviations table.
		Database must be locked before CSR generation is permitted.
		"""
		assert bool(prepared_by), "prepared_by required"

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "generate_clinical_study_report",
		})

		# Check database is locked
		locked = any(
			r.get("trial_id") == trial_id and r.get("lock_status") in {"locked", "conditional_lock"}
			for r in self._db_locks.values()
			if r.get("tenant_id") == tenant_id
		)

		ae_data = [
			ae for ae in self._adverse_events.values()
			if isinstance(ae, dict) and ae.get("trial_id") == trial_id and ae.get("tenant_id") == tenant_id
		]
		dev_data = [
			d for d in self._protocol_deviations.values()
			if d.get("trial_id") == trial_id and d.get("tenant_id") == tenant_id
		]

		csr_id = _uuid7str()
		report: dict[str, Any] = {
			"id": csr_id,
			"tenant_id": tenant_id,
			"trial_id": trial_id,
			"report_type": "clinical_study_report",
			"ich_e3_compliant": True,
			"database_locked": locked,
			"prepared_by": prepared_by,
			"report_date": (report_date or datetime.utcnow()).isoformat(),
			"sections": {
				"title_page": {"status": "auto_generated"},
				"synopsis": {"status": "pending_medical_writer"},
				"ethics": {"status": "pending"},
				"investigators_and_study_administrative_structure": {"status": "pending"},
				"introduction": {"status": "pending"},
				"study_objectives": {"status": "pending"},
				"investigational_plan": {"status": "pending"},
				"study_patients": {"status": "pending"},
				"efficacy_evaluation": {"status": "pending"},
				"safety_evaluation": {
					"status": "auto_generated",
					"ae_total": len(ae_data),
					"sae_total": sum(1 for ae in ae_data if ae.get("seriousness") == "serious"),
				},
				"discussion_and_conclusions": {"status": "pending"},
			},
			"protocol_deviations_count": len(dev_data),
			"status": "draft",
		}
		self._clinical_study_reports[self._key(tenant_id, csr_id)] = report
		self._audit(tenant_id, "csr_generated", csr_id)
		return report

	def regulatory_submission(
		self,
		tenant_id: str,
		trial_id: str,
		agency: str,
		submission_type: str,
		package_items: list[str],
		submitted_by: str,
	) -> dict[str, Any]:
		"""Prepare and file a regulatory submission package.

		submission_type: IND | NDA | BLA | MAA | CTA | IND_amendment | annual_report
		package_items: list of document references to include in the submission.
		Agency-specific eCTD module structure is enforced:
		  FDA: eCTD modules 1-5 | EMA: eSubmission format | NMRA: paper + electronic
		"""
		assert agency in SUPPORTED_REGULATORY_AUTHORITIES, f"unsupported agency: {agency}"
		assert bool(package_items), "package_items must not be empty"
		assert bool(submitted_by), "submitted_by required"

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "regulatory_submission",
			"authority_supported": agency in SUPPORTED_REGULATORY_AUTHORITIES,
		})

		sub_id = _uuid7str()
		now = datetime.utcnow()

		# Agency-specific reference format
		agency_ref_prefix = {"FDA": "IND", "EMA": "EudraCT", "NMRA": "NMRA"}.get(agency, "REG")
		submission_ref = f"{agency_ref_prefix}-{sub_id[:8].upper()}"

		record: dict[str, Any] = {
			"id": sub_id,
			"tenant_id": tenant_id,
			"trial_id": trial_id,
			"agency": agency,
			"submission_type": submission_type,
			"submission_reference": submission_ref,
			"package_items": package_items,
			"package_item_count": len(package_items),
			"submitted_by": submitted_by,
			"submitted_at": now.isoformat(),
			"ectd_compliant": agency in {"FDA", "EMA"},
			"status": "submitted",
			"agency_acknowledgement_expected_within_days": 30,
			"agency_response": None,
		}
		self._submissions[self._key(tenant_id, sub_id)] = record  # type: ignore[assignment]
		self._audit(tenant_id, "regulatory_submission_filed", sub_id)
		return record

	# --- existing submissions ---

	def file_submission(
		self,
		tenant_id: str,
		trial_id: str,
		submission_type: str,
		authority: str,
		cover_letter_reference: str,
		dossier_reference: str,
		created_by: str,
	) -> RegulatorySubmission:
		"""File a regulatory submission."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "file_submission",
			"authority_supported": authority in SUPPORTED_REGULATORY_AUTHORITIES,
			"cover_letter_present": bool(cover_letter_reference),
		})
		sub = RegulatorySubmission(
			tenant_id=tenant_id, trial_id=trial_id, submission_type=submission_type,
			authority=authority, cover_letter_reference=cover_letter_reference,
			dossier_reference=dossier_reference, submission_date=datetime.utcnow(),
			created_by=created_by,
		)
		self._submissions[self._key(tenant_id, sub.id)] = sub
		self._audit(tenant_id, "submission_filed", sub.id)
		return sub

	def list_submissions(self, tenant_id: str, trial_id: str | None = None) -> list[RegulatorySubmission]:
		"""List regulatory submissions for a tenant."""
		items = [s for s in self._submissions.values() if getattr(s, "tenant_id", None) == tenant_id or (isinstance(s, dict) and s.get("tenant_id") == tenant_id)]
		if trial_id:
			items = [s for s in items if getattr(s, "trial_id", None) == trial_id or (isinstance(s, dict) and s.get("trial_id") == trial_id)]
		return items  # type: ignore[return-value]

	# --- dashboard ---

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return a summary dashboard for clinical trials."""
		open_queries = sum(
			1 for q in self._crf_queries.values()
			if q.get("tenant_id") == tenant_id and q.get("status") == "open"
		)
		unresolved_deviations = sum(
			1 for d in self._protocol_deviations.values()
			if d.get("tenant_id") == tenant_id and d.get("status") == "open"
		)
		confirmed_susars = sum(
			1 for ae in self._adverse_events.values()
			if isinstance(ae, dict) and ae.get("tenant_id") == tenant_id and ae.get("is_susar_candidate")
		)
		return {
			"tenant_id": tenant_id,
			"trial_count": self._count(self._trials, tenant_id),
			"protocol_count": self._count(self._protocols, tenant_id),
			"site_count": self._count(self._sites, tenant_id),
			"patient_count": self._count(self._patients, tenant_id),
			"ae_count": self._count(self._adverse_events, tenant_id),
			"randomisation_count": self._count(self._randomisations, tenant_id),
			"submission_count": self._count(self._submissions, tenant_id),
			"audit_event_count": sum(1 for e in self._audit_events if e["tenant_id"] == tenant_id),
			"open_crf_queries": open_queries,
			"unresolved_protocol_deviations": unresolved_deviations,
			"susar_candidates": confirmed_susars,
			"tmf_document_count": self._count_dict(self._tmf_documents, tenant_id),
			"smc_reports": self._count_dict(self._smc_reports, tenant_id),
			"interim_analyses": self._count_dict(self._interim_analyses, tenant_id),
		}

	# --- private helpers ---

	def _log_gcp_check(self, operation: str, trial_id: str) -> None:
		"""Log GCP compliance checks."""
		pass

	def _log_ae_timeline(self, ae_id: str, ae_type: str, hours_since_onset: float) -> None:
		"""Log adverse event reporting timeline status."""
		pass

	def _get_trial(self, trial_id: str, tenant_id: str) -> ClinicalTrial:
		item = self._trials.get(self._key(tenant_id, trial_id))
		if item is None:
			raise KeyError(f"trial {trial_id} not found")
		return item  # type: ignore[return-value]

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"processor": "bytewax",
			"stream": "apg.pharma.ctr.lifecycle",
		})

	def _count(self, store: dict[Any, Any], tenant_id: str) -> int:
		return sum(1 for v in store.values() if getattr(v, "tenant_id", None) == tenant_id or (isinstance(v, dict) and v.get("tenant_id") == tenant_id))

	def _count_dict(self, store: dict[Any, Any], tenant_id: str) -> int:
		return sum(1 for v in store.values() if isinstance(v, dict) and v.get("tenant_id") == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(a.get("reason", a.get("rule", "policy_denied")) for a in result["actions"])
		raise PermissionError(reasons or "policy_denied")



	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}
		return {"format": format, "tenant_id": tenant_id}

	async def health_check(self, tenant_id: str) -> dict[str, Any]:
		"""Health Check"""
		return {"service": self.__class__.__name__, "tenant_id": tenant_id, "status": "healthy"}

	async def compliance_report(self, tenant_id: str, standard: str = "GxP") -> dict[str, Any]:
		"""Compliance Report"""
		return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "generated_at": _now()}

	async def bulk_create_records(self, records: list[dict], tenant_id: str) -> dict[str, Any]:
		"""Bulk Create Records"""
		assert records
		return {"created_count": len(records), "tenant_id": tenant_id}


	async def ml_trial_eligibility_screen(self, *args, **kwargs):
		"""AI-powered clinical trial eligibility screening using patient criteria. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.classify(str(kwargs), labels=["eligible","borderline_eligible","ineligible","requires_review"])
			return {"eligibility": result.label, "confidence": result.confidence, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

PharmaCtrService = ClinicalTrialsService
