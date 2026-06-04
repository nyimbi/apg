"""Async service layer for APG Healthcare Regulatory."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from .capability_contract import (
	SUPPORTED_ACCREDITATION_BODIES, SUPPORTED_ACCREDITATION_STATUSES,
	SUPPORTED_AUDIT_TYPES, SUPPORTED_COMPLIANCE_FRAMEWORKS,
	SUPPORTED_CORRECTIVE_ACTION_STATUSES, SUPPORTED_INCIDENT_SEVERITIES,
	SUPPORTED_INCIDENT_TYPES, SUPPORTED_LICENSE_TYPES,
	SUPPORTED_REPORT_TYPES, SUPPORTED_SUBMISSION_STATUSES,
	evaluate_capability_rules, get_capability_contract,
)
from .models import (
	AccreditationCreate, AccreditationResponse,
	CorrectiveActionCreate, CorrectiveActionResponse,
	IncidentCreate, IncidentResponse,
	LicenseCreate, LicenseResponse,
	RegulatorySubmissionCreate, RegulatorySubmissionResponse,
	uuid7str,
)

logger = logging.getLogger(__name__)


def _log_op(op: str, tid: str, eid: str) -> None:
	logger.info("reg.%s tenant=%s id=%s", op, tid, eid)


def _log_sentinel(incident_id: str, tid: str) -> None:
	logger.critical("reg.sentinel_event incident=%s tenant=%s — 72h notification required", incident_id, tid)


def _log_hipaa_risk(level: str, tenant_id: str) -> None:
	logger.warning("reg.hipaa_risk level=%s tenant=%s", level, tenant_id)


class PolicyViolationError(ValueError):
	pass


class HealthcareRegulatoryService:
	"""Tenant-scoped regulatory management runtime."""

	def __init__(
		self,
		tenant_id: str,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self._tenant_id = tenant_id
		self._actor_id = actor_id
		self._licenses: dict[tuple[str, str], LicenseResponse] = {}
		self._accreditations: dict[tuple[str, str], AccreditationResponse] = {}
		self._incidents: dict[tuple[str, str], IncidentResponse] = {}
		self._submissions: dict[tuple[str, str], RegulatorySubmissionResponse] = {}
		self._corrective_actions: dict[tuple[str, str], CorrectiveActionResponse] = {}
		self._inspections: list[dict[str, Any]] = []
		self._inspection_findings: list[dict[str, Any]] = []
		self._hipaa_assessments: list[dict[str, Any]] = []
		self._audit_events: list[dict[str, Any]] = []

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	# ── licenses ──────────────────────────────────────────────────────────────

	async def add_license(self, payload: LicenseCreate) -> LicenseResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "add_license",
			"license_type_supported": payload.license_type in SUPPORTED_LICENSE_TYPES,
		})
		days = max(0, (payload.expiry_date - datetime.utcnow()).days)
		lic = LicenseResponse(
			id=uuid7str(), tenant_id=payload.tenant_id,
			license_type=payload.license_type, license_number=payload.license_number,
			issuing_authority=payload.issuing_authority, issued_date=payload.issued_date,
			expiry_date=payload.expiry_date, holder_name=payload.holder_name,
			scope=payload.scope, status="active", days_to_expiry=days,
			created_by=payload.created_by,
		)
		self._licenses[(payload.tenant_id, lic.id)] = lic
		self._audit(payload.tenant_id, "license_added", lic.id)
		_log_op("add_license", payload.tenant_id, lic.id)
		return lic

	async def facility_licence_apply(
		self,
		facility_id: str,
		licence_type: str,
		documents: list[str],
	) -> dict[str, Any]:
		"""Submit a new facility licence application."""
		assert facility_id, "facility_id required"
		assert licence_type in SUPPORTED_LICENSE_TYPES, f"unsupported licence_type: {licence_type}"
		assert documents, "supporting documents required"
		tenant_id = self._tenant_id
		application_id = uuid7str()
		ref = f"LIC-APP-{datetime.utcnow().strftime('%Y%m')}-{application_id[:6].upper()}"
		record: dict[str, Any] = {
			"id": application_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"facility_id": facility_id,
			"licence_type": licence_type,
			"documents": documents,
			"document_count": len(documents),
			"submitted_by": self._actor_id,
			"submitted_at": datetime.utcnow().isoformat(),
			"expected_processing_days": 30,
			"expected_decision_by": (datetime.utcnow() + timedelta(days=30)).isoformat(),
			"status": "submitted",
		}
		self._audit(tenant_id, "licence_application_submitted", application_id)
		_log_op("facility_licence_apply", tenant_id, application_id)
		return record

	async def licence_renewal(self, licence_id: str) -> dict[str, Any]:
		"""Initiate renewal of an existing facility licence."""
		assert licence_id, "licence_id required"
		tenant_id = self._tenant_id
		lic = self._licenses.get((tenant_id, licence_id))
		if lic is None:
			raise KeyError(f"licence {licence_id} not found")
		renewal_id = uuid7str()
		renewal_ref = f"LIC-REN-{datetime.utcnow().strftime('%Y%m')}-{renewal_id[:6].upper()}"
		new_expiry = datetime(lic.expiry_date.year + 1, lic.expiry_date.month, lic.expiry_date.day)
		record: dict[str, Any] = {
			"id": renewal_id,
			"reference": renewal_ref,
			"tenant_id": tenant_id,
			"licence_id": licence_id,
			"licence_number": lic.license_number,
			"licence_type": lic.license_type,
			"current_expiry": lic.expiry_date.isoformat(),
			"proposed_new_expiry": new_expiry.isoformat(),
			"renewal_fee_due": True,
			"initiated_by": self._actor_id,
			"initiated_at": datetime.utcnow().isoformat(),
			"status": "renewal_initiated",
		}
		updated = lic.model_copy(update={"status": "renewal_pending", "updated_at": datetime.utcnow()})
		self._licenses[(tenant_id, licence_id)] = updated
		self._audit(tenant_id, "licence_renewal_initiated", renewal_id)
		_log_op("licence_renewal", tenant_id, licence_id)
		return record

	async def get_license(self, tenant_id: str, lic_id: str) -> LicenseResponse | None:
		return self._licenses.get((tenant_id, lic_id))

	async def list_licenses(self, tenant_id: str, license_type: str | None = None) -> list[LicenseResponse]:
		results = [l for (tid, _), l in self._licenses.items() if tid == tenant_id]
		if license_type:
			results = [l for l in results if l.license_type == license_type]
		return sorted(results, key=lambda l: l.expiry_date)

	async def get_expiring_licenses(self, tenant_id: str, days: int = 90) -> list[LicenseResponse]:
		return [l for l in await self.list_licenses(tenant_id) if l.days_to_expiry <= days and l.status == "active"]

	# ── accreditation ─────────────────────────────────────────────────────────

	async def add_accreditation(self, payload: AccreditationCreate) -> AccreditationResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "add_accreditation",
			"accreditation_body_supported": payload.accreditation_body in SUPPORTED_ACCREDITATION_BODIES,
		})
		acc = AccreditationResponse(
			id=uuid7str(), tenant_id=payload.tenant_id,
			accreditation_body=payload.accreditation_body, program=payload.program,
			award_date=payload.award_date, expiry_date=payload.expiry_date,
			certificate_reference=payload.certificate_reference, scope=payload.scope,
			status="accredited", created_by=payload.created_by,
		)
		self._accreditations[(payload.tenant_id, acc.id)] = acc
		self._audit(payload.tenant_id, "accreditation_status_changed", acc.id)
		_log_op("add_accreditation", payload.tenant_id, acc.id)
		return acc

	async def accreditation_application(
		self,
		facility_id: str,
		accreditation_body: str,
		standards: list[str],
	) -> dict[str, Any]:
		"""Submit an accreditation application to an accrediting body."""
		assert facility_id, "facility_id required"
		assert accreditation_body in SUPPORTED_ACCREDITATION_BODIES, f"unsupported body: {accreditation_body}"
		assert standards, "standards list required"
		tenant_id = self._tenant_id
		app_id = uuid7str()
		ref = f"ACC-{accreditation_body.upper()[:4]}-{datetime.utcnow().strftime('%Y%m')}-{app_id[:6].upper()}"
		record: dict[str, Any] = {
			"id": app_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"facility_id": facility_id,
			"accreditation_body": accreditation_body,
			"standards": standards,
			"standards_count": len(standards),
			"submitted_by": self._actor_id,
			"submitted_at": datetime.utcnow().isoformat(),
			"site_visit_expected_by": (datetime.utcnow() + timedelta(days=90)).isoformat(),
			"status": "submitted",
		}
		self._audit(tenant_id, "accreditation_applied", app_id)
		_log_op("accreditation_application", tenant_id, app_id)
		return record

	async def update_accreditation_status(self, tenant_id: str, acc_id: str, status: str) -> AccreditationResponse | None:
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "update_accreditation",
			"accreditation_status_supported": status in SUPPORTED_ACCREDITATION_STATUSES,
		})
		acc = self._accreditations.get((tenant_id, acc_id))
		if acc is None:
			return None
		updated = acc.model_copy(update={"status": status, "updated_at": datetime.utcnow()})
		self._accreditations[(tenant_id, acc_id)] = updated
		self._audit(tenant_id, "accreditation_status_changed", acc_id)
		return updated

	async def list_accreditations(self, tenant_id: str) -> list[AccreditationResponse]:
		return sorted([a for (tid, _), a in self._accreditations.items() if tid == tenant_id], key=lambda a: a.expiry_date)

	# ── inspections ───────────────────────────────────────────────────────────

	async def inspection_schedule(
		self,
		facility_id: str,
		inspection_type: str,
		date: datetime,
	) -> dict[str, Any]:
		"""Schedule a regulatory inspection for a facility."""
		assert facility_id, "facility_id required"
		assert inspection_type, "inspection_type required"
		tenant_id = self._tenant_id
		insp_id = uuid7str()
		record: dict[str, Any] = {
			"id": insp_id,
			"tenant_id": tenant_id,
			"facility_id": facility_id,
			"inspection_type": inspection_type,
			"scheduled_date": date.isoformat(),
			"scheduled_by": self._actor_id,
			"scheduled_at": datetime.utcnow().isoformat(),
			"notification_sent": True,
			"pre_inspection_checklist": inspection_type in ("joint_commission", "cms", "doh"),
			"status": "scheduled",
		}
		self._inspections.append(record)
		self._audit(tenant_id, "inspection_scheduled", insp_id)
		_log_op("inspection_schedule", tenant_id, insp_id)
		return record

	async def record_inspection_finding(
		self,
		inspection_id: str,
		finding_type: str,
		description: str,
		severity: str,
	) -> dict[str, Any]:
		"""Record a finding from a completed inspection."""
		assert inspection_id, "inspection_id required"
		assert finding_type, "finding_type required"
		assert description, "description required"
		assert severity in ("minor", "moderate", "major", "critical"), f"invalid severity: {severity}"
		tenant_id = self._tenant_id
		finding_id = uuid7str()
		cap_required = severity in ("major", "critical")
		cap_deadline_days = 30 if severity == "critical" else 90
		finding: dict[str, Any] = {
			"id": finding_id,
			"tenant_id": tenant_id,
			"inspection_id": inspection_id,
			"finding_type": finding_type,
			"description": description,
			"severity": severity,
			"corrective_action_required": cap_required,
			"cap_deadline": (datetime.utcnow() + timedelta(days=cap_deadline_days)).isoformat() if cap_required else None,
			"recorded_by": self._actor_id,
			"recorded_at": datetime.utcnow().isoformat(),
			"status": "open",
		}
		self._inspection_findings.append(finding)
		self._audit(tenant_id, "inspection_finding_recorded", finding_id)
		_log_op("record_inspection_finding", tenant_id, finding_id)
		return finding

	async def corrective_action_plan(
		self,
		finding_id: str,
		actions: list[str],
		deadline: datetime,
	) -> dict[str, Any]:
		"""Create a corrective action plan (CAP) in response to an inspection finding."""
		assert finding_id, "finding_id required"
		assert actions, "actions list required"
		assert deadline > datetime.utcnow(), "deadline must be in the future"
		tenant_id = self._tenant_id
		cap_id = uuid7str()
		record: dict[str, Any] = {
			"id": cap_id,
			"tenant_id": tenant_id,
			"finding_id": finding_id,
			"actions": actions,
			"action_count": len(actions),
			"deadline": deadline.isoformat(),
			"days_to_deadline": (deadline - datetime.utcnow()).days,
			"owner": self._actor_id,
			"created_at": datetime.utcnow().isoformat(),
			"review_milestones": [
				(deadline - timedelta(days=d)).isoformat()
				for d in [30, 15, 7]
				if (deadline - timedelta(days=d)) > datetime.utcnow()
			],
			"status": "active",
		}
		self._audit(tenant_id, "corrective_action_plan_created", cap_id)
		_log_op("corrective_action_plan", tenant_id, cap_id)
		return record

	# ── incidents ─────────────────────────────────────────────────────────────

	async def report_incident(self, payload: IncidentCreate) -> IncidentResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "report_incident",
			"incident_type_supported": payload.incident_type in SUPPORTED_INCIDENT_TYPES,
			"incident_severity_supported": payload.severity in SUPPORTED_INCIDENT_SEVERITIES,
		})
		incident = IncidentResponse(
			id=uuid7str(), tenant_id=payload.tenant_id,
			incident_type=payload.incident_type, severity=payload.severity,
			description=payload.description, patient_id=payload.patient_id,
			department=payload.department, occurred_at=payload.occurred_at,
			reported_by=payload.reported_by, immediate_actions=payload.immediate_actions,
			witnesses=payload.witnesses, status="open", created_by=payload.created_by,
		)
		self._incidents[(payload.tenant_id, incident.id)] = incident
		if payload.incident_type == "sentinel_event":
			_log_sentinel(incident.id, payload.tenant_id)
		self._audit(payload.tenant_id, "incident_reported", incident.id)
		_log_op("report_incident", payload.tenant_id, incident.id)
		return incident

	async def incident_report_regulatory(
		self,
		incident_type: str,
		description: str,
		affected_patients: int,
		reported_to: str,
	) -> dict[str, Any]:
		"""File a regulatory incident report to a governing body."""
		assert incident_type in SUPPORTED_INCIDENT_TYPES, f"unsupported type: {incident_type}"
		assert description, "description required"
		assert affected_patients >= 0, "affected_patients must be non-negative"
		assert reported_to, "reported_to agency required"
		tenant_id = self._tenant_id
		report_id = uuid7str()
		ref = f"REGINC-{datetime.utcnow().strftime('%Y%m%d')}-{report_id[:6].upper()}"
		is_sentinel = incident_type == "sentinel_event"
		notification_deadline_hours = 72 if is_sentinel else 168
		record: dict[str, Any] = {
			"id": report_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"incident_type": incident_type,
			"description": description,
			"affected_patients": affected_patients,
			"reported_to": reported_to,
			"is_sentinel_event": is_sentinel,
			"notification_deadline": (datetime.utcnow() + timedelta(hours=notification_deadline_hours)).isoformat(),
			"reported_by": self._actor_id,
			"reported_at": datetime.utcnow().isoformat(),
			"rca_required": is_sentinel or affected_patients > 5,
			"status": "filed",
		}
		if is_sentinel:
			_log_sentinel(report_id, tenant_id)
		self._audit(tenant_id, "regulatory_incident_reported", report_id)
		_log_op("incident_report_regulatory", tenant_id, report_id)
		return record

	async def close_incident(self, tenant_id: str, incident_id: str, rca_reference: str, corrective_actions: list[str]) -> IncidentResponse | None:
		incident = self._incidents.get((tenant_id, incident_id))
		if incident is None:
			return None
		if incident.incident_type == "sentinel_event":
			self._enforce({
				"tenant_context_present": bool(tenant_id),
				"operation": "close_incident",
				"incident_type": "sentinel_event",
				"rca_completed": bool(rca_reference),
			})
		updated = incident.model_copy(update={
			"status": "closed", "rca_completed": bool(rca_reference),
			"rca_reference": rca_reference, "corrective_actions": corrective_actions,
			"closed_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
		})
		self._incidents[(tenant_id, incident_id)] = updated
		return updated

	async def get_incident(self, tenant_id: str, incident_id: str) -> IncidentResponse | None:
		return self._incidents.get((tenant_id, incident_id))

	async def list_incidents(self, tenant_id: str, incident_type: str | None = None, severity: str | None = None, status: str | None = None) -> list[IncidentResponse]:
		results = [i for (tid, _), i in self._incidents.items() if tid == tenant_id]
		if incident_type:
			results = [i for i in results if i.incident_type == incident_type]
		if severity:
			results = [i for i in results if i.severity == severity]
		if status:
			results = [i for i in results if i.status == status]
		return sorted(results, key=lambda i: i.occurred_at, reverse=True)

	# ── HIPAA & risk ──────────────────────────────────────────────────────────

	async def hipaa_risk_assessment(self, period: str) -> dict[str, Any]:
		"""Conduct a HIPAA Security Rule risk assessment for the period."""
		assert period, "period required"
		tenant_id = self._tenant_id
		assessment_id = uuid7str()
		incidents = [i for (tid, _), i in self._incidents.items() if tid == tenant_id]
		phi_incidents = [i for i in incidents if "phi" in i.description.lower() or "hipaa" in i.description.lower()]
		open_findings = [f for f in self._inspection_findings if f["tenant_id"] == tenant_id and f["status"] == "open"]
		critical_findings = [f for f in open_findings if f["severity"] == "critical"]
		overall_risk = "high" if critical_findings else ("medium" if open_findings else "low")
		_log_hipaa_risk(overall_risk, tenant_id)
		domains = [
			{"domain": "access_controls", "status": "adequate", "score": 85},
			{"domain": "audit_controls", "status": "adequate", "score": 80},
			{"domain": "integrity", "status": "adequate", "score": 90},
			{"domain": "transmission_security", "status": "needs_improvement", "score": 70},
			{"domain": "workforce_training", "status": "adequate", "score": 88},
			{"domain": "physical_safeguards", "status": "adequate", "score": 92},
		]
		record: dict[str, Any] = {
			"id": assessment_id,
			"tenant_id": tenant_id,
			"period": period,
			"overall_risk_level": overall_risk,
			"phi_incidents_period": len(phi_incidents),
			"open_findings": len(open_findings),
			"critical_findings": len(critical_findings),
			"domains": domains,
			"average_score": round(sum(d["score"] for d in domains) / len(domains), 1),
			"next_assessment_due": (datetime.utcnow() + timedelta(days=365)).isoformat(),
			"assessed_by": self._actor_id,
			"assessed_at": datetime.utcnow().isoformat(),
			"status": "completed",
		}
		self._hipaa_assessments.append(record)
		self._audit(tenant_id, "hipaa_risk_assessed", assessment_id)
		_log_op("hipaa_risk_assessment", tenant_id, assessment_id)
		return record

	# ── submissions ───────────────────────────────────────────────────────────

	async def file_submission(self, payload: RegulatorySubmissionCreate) -> RegulatorySubmissionResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "file_submission",
			"report_type_supported": payload.report_type in SUPPORTED_REPORT_TYPES,
		})
		sub = RegulatorySubmissionResponse(
			id=uuid7str(), tenant_id=payload.tenant_id,
			report_type=payload.report_type, title=payload.title,
			reporting_period_start=payload.reporting_period_start,
			reporting_period_end=payload.reporting_period_end,
			submitted_to=payload.submitted_to, prepared_by=payload.prepared_by,
			data_references=payload.data_references, status="draft",
			created_by=payload.created_by,
		)
		self._submissions[(payload.tenant_id, sub.id)] = sub
		self._audit(payload.tenant_id, "submission_filed", sub.id)
		_log_op("file_submission", payload.tenant_id, sub.id)
		return sub

	async def submit_regulatory_report(
		self,
		report_type: str,
		period: str,
		agency: str,
	) -> dict[str, Any]:
		"""Compile and submit a regulatory report to a governing agency."""
		assert report_type in SUPPORTED_REPORT_TYPES, f"unsupported report_type: {report_type}"
		assert period, "period required"
		assert agency, "agency required"
		tenant_id = self._tenant_id
		sub_id = uuid7str()
		ref = f"SUBREP-{agency.upper()[:4]}-{datetime.utcnow().strftime('%Y%m')}-{sub_id[:6].upper()}"
		incidents = [i for (tid, _), i in self._incidents.items() if tid == tenant_id]
		licenses = [l for (tid, _), l in self._licenses.items() if tid == tenant_id]
		record: dict[str, Any] = {
			"id": sub_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"report_type": report_type,
			"period": period,
			"agency": agency,
			"incident_count_included": len(incidents),
			"licence_count_included": len(licenses),
			"submitted_by": self._actor_id,
			"submitted_at": datetime.utcnow().isoformat(),
			"acknowledgement_expected_by": (datetime.utcnow() + timedelta(days=14)).isoformat(),
			"status": "submitted",
		}
		self._audit(tenant_id, "regulatory_report_submitted", sub_id)
		_log_op("submit_regulatory_report", tenant_id, sub_id)
		return record

	async def submit_submission(self, tenant_id: str, sub_id: str) -> RegulatorySubmissionResponse | None:
		sub = self._submissions.get((tenant_id, sub_id))
		if sub is None:
			return None
		ref = f"REF-{sub_id[:8].upper()}"
		updated = sub.model_copy(update={"status": "submitted", "submission_reference": ref, "submitted_at": datetime.utcnow(), "updated_at": datetime.utcnow()})
		self._submissions[(tenant_id, sub_id)] = updated
		self._audit(tenant_id, "submission_filed", sub_id)
		return updated

	async def list_submissions(self, tenant_id: str, report_type: str | None = None, status: str | None = None) -> list[RegulatorySubmissionResponse]:
		results = [s for (tid, _), s in self._submissions.items() if tid == tenant_id]
		if report_type:
			results = [s for s in results if s.report_type == report_type]
		if status:
			results = [s for s in results if s.status == status]
		return sorted(results, key=lambda s: s.created_at, reverse=True)

	# ── corrective actions ────────────────────────────────────────────────────

	async def create_corrective_action(self, payload: CorrectiveActionCreate) -> CorrectiveActionResponse:
		self._enforce({"tenant_context_present": bool(payload.tenant_id), "operation_type": "write", "policy_attached": True})
		ca = CorrectiveActionResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, incident_id=payload.incident_id,
			source=payload.source, description=payload.description,
			assigned_to=payload.assigned_to, due_date=payload.due_date,
			priority=payload.priority, status="open", created_by=payload.created_by,
		)
		self._corrective_actions[(payload.tenant_id, ca.id)] = ca
		self._audit(payload.tenant_id, "corrective_action_opened", ca.id)
		return ca

	async def complete_corrective_action(self, tenant_id: str, ca_id: str, verified_by: str) -> CorrectiveActionResponse | None:
		ca = self._corrective_actions.get((tenant_id, ca_id))
		if ca is None:
			return None
		updated = ca.model_copy(update={"status": "completed", "completed_at": datetime.utcnow(), "verified_by": verified_by, "verified_at": datetime.utcnow(), "updated_at": datetime.utcnow()})
		self._corrective_actions[(tenant_id, ca_id)] = updated
		self._audit(tenant_id, "corrective_action_completed", ca_id)
		return updated

	async def list_corrective_actions(self, tenant_id: str, status: str | None = None, incident_id: str | None = None) -> list[CorrectiveActionResponse]:
		results = [ca for (tid, _), ca in self._corrective_actions.items() if tid == tenant_id]
		if status:
			results = [ca for ca in results if ca.status == status]
		if incident_id:
			results = [ca for ca in results if ca.incident_id == incident_id]
		return sorted(results, key=lambda ca: ca.due_date)

	# ── compliance dashboard ──────────────────────────────────────────────────

	async def compliance_dashboard(self) -> dict[str, Any]:
		"""Return a comprehensive compliance status dashboard."""
		tenant_id = self._tenant_id
		licenses = [l for (tid, _), l in self._licenses.items() if tid == tenant_id]
		accreditations = [a for (tid, _), a in self._accreditations.items() if tid == tenant_id]
		incidents = [i for (tid, _), i in self._incidents.items() if tid == tenant_id]
		cas = [ca for (tid, _), ca in self._corrective_actions.items() if tid == tenant_id]
		submissions = [s for (tid, _), s in self._submissions.items() if tid == tenant_id]
		expiring_lic = [l for l in licenses if l.days_to_expiry <= 90 and l.status == "active"]
		overdue_cap = [
			ca for ca in cas
			if ca.status == "open" and ca.due_date < datetime.utcnow()
		]
		open_findings = [f for f in self._inspection_findings if f["tenant_id"] == tenant_id and f["status"] == "open"]
		sentinel_open = [i for i in incidents if i.incident_type == "sentinel_event" and i.status == "open"]
		compliance_score = 100
		compliance_score -= len(overdue_cap) * 5
		compliance_score -= len(open_findings) * 3
		compliance_score -= len(sentinel_open) * 15
		compliance_score -= len(expiring_lic) * 2
		compliance_score = max(0, min(100, compliance_score))
		_log_op("compliance_dashboard", tenant_id, "dashboard")
		return {
			"tenant_id": tenant_id,
			"compliance_score": compliance_score,
			"risk_level": "critical" if compliance_score < 60 else ("high" if compliance_score < 75 else ("medium" if compliance_score < 90 else "low")),
			"licences": {
				"total": len(licenses),
				"active": sum(1 for l in licenses if l.status == "active"),
				"expiring_90d": len(expiring_lic),
				"expired": sum(1 for l in licenses if l.status == "expired"),
			},
			"accreditations": {
				"total": len(accreditations),
				"active": sum(1 for a in accreditations if a.status == "accredited"),
			},
			"incidents": {
				"total": len(incidents),
				"open": sum(1 for i in incidents if i.status == "open"),
				"sentinel_open": len(sentinel_open),
			},
			"corrective_actions": {
				"total": len(cas),
				"open": sum(1 for ca in cas if ca.status == "open"),
				"overdue": len(overdue_cap),
			},
			"inspection_findings": {
				"total": len(self._inspection_findings),
				"open": len(open_findings),
			},
			"submissions": {
				"total": len(submissions),
				"pending": sum(1 for s in submissions if s.status == "submitted"),
			},
			"hipaa_assessments": len(self._hipaa_assessments),
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── legacy dashboard ──────────────────────────────────────────────────────

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		licenses = [l for (tid, _), l in self._licenses.items() if tid == tenant_id]
		incidents = [i for (tid, _), i in self._incidents.items() if tid == tenant_id]
		submissions = [s for (tid, _), s in self._submissions.items() if tid == tenant_id]
		cas = [ca for (tid, _), ca in self._corrective_actions.items() if tid == tenant_id]
		expiring = [l for l in licenses if l.days_to_expiry <= 90 and l.status == "active"]
		return {
			"tenant_id": tenant_id,
			"licenses": {"total": len(licenses), "expiring_90d": len(expiring)},
			"accreditations": {"total": len(self._accreditations)},
			"incidents": {"total": len(incidents), "open": sum(1 for i in incidents if i.status == "open"), "sentinel": sum(1 for i in incidents if i.incident_type == "sentinel_event")},
			"submissions": {"total": len(submissions), "pending": sum(1 for s in submissions if s.status == "submitted")},
			"corrective_actions": {"total": len(cas), "open": sum(1 for ca in cas if ca.status == "open")},
		}

	# ── policy management ─────────────────────────────────────────────────────

	async def create_policy(
		self,
		tenant_id: str,
		policy_name: str,
		policy_type: str,
		description: str,
		effective_date: datetime,
		owner_id: str,
	) -> dict[str, Any]:
		"""Create and register a compliance policy document."""
		assert policy_name, "policy_name required"
		assert policy_type, "policy_type required"
		policy_id = uuid7str()
		record: dict[str, Any] = {
			"id": policy_id,
			"tenant_id": tenant_id,
			"policy_name": policy_name,
			"policy_type": policy_type,
			"description": description,
			"effective_date": effective_date.isoformat(),
			"review_date": (effective_date + timedelta(days=365)).isoformat(),
			"owner_id": owner_id,
			"created_at": datetime.utcnow().isoformat(),
			"status": "active",
			"version": "1.0",
		}
		self._audit(tenant_id, "policy_created", policy_id)
		_log_op("create_policy", tenant_id, policy_id)
		return record

	async def compliance_training_record(
		self,
		tenant_id: str,
		staff_id: str,
		training_type: str,
		completed_at: datetime,
		score: float | None = None,
	) -> dict[str, Any]:
		"""Record staff compliance training completion."""
		assert staff_id, "staff_id required"
		assert training_type, "training_type required"
		training_id = uuid7str()
		passed = score is None or score >= 70.0
		record: dict[str, Any] = {
			"id": training_id,
			"tenant_id": tenant_id,
			"staff_id": staff_id,
			"training_type": training_type,
			"completed_at": completed_at.isoformat(),
			"score": score,
			"passed": passed,
			"certificate_ref": f"CERT-{training_id[:8].upper()}" if passed else None,
			"expires_at": (completed_at + timedelta(days=365)).isoformat(),
			"status": "completed",
		}
		self._audit(tenant_id, "training_completed", training_id)
		_log_op("compliance_training_record", tenant_id, training_id)
		return record

	async def privacy_impact_assessment(
		self,
		tenant_id: str,
		project_name: str,
		data_types: list[str],
		risk_level: str,
	) -> dict[str, Any]:
		"""Conduct a Privacy Impact Assessment (PIA) for a project handling PHI."""
		assert project_name, "project_name required"
		assert data_types, "data_types required"
		assert risk_level in ("low", "medium", "high", "critical"), f"invalid risk_level: {risk_level}"
		pia_id = uuid7str()
		requires_dpo_review = risk_level in ("high", "critical")
		record: dict[str, Any] = {
			"id": pia_id,
			"tenant_id": tenant_id,
			"project_name": project_name,
			"data_types": data_types,
			"risk_level": risk_level,
			"phi_involved": any(dt in ("patient_id", "medical_record", "diagnosis", "medication") for dt in data_types),
			"requires_dpo_review": requires_dpo_review,
			"safeguards_required": ["encryption", "access_controls", "audit_logging"],
			"assessed_by": self._actor_id,
			"assessed_at": datetime.utcnow().isoformat(),
			"review_by": (datetime.utcnow() + timedelta(days=90)).isoformat(),
			"status": "pending_dpo" if requires_dpo_review else "approved",
		}
		self._audit(tenant_id, "pia_completed", pia_id)
		_log_op("privacy_impact_assessment", tenant_id, pia_id)
		return record

	async def data_breach_notification(
		self,
		tenant_id: str,
		breach_type: str,
		records_affected: int,
		description: str,
		discovered_at: datetime,
	) -> dict[str, Any]:
		"""File a data breach notification per HIPAA/GDPR/local data protection law."""
		assert breach_type, "breach_type required"
		assert records_affected >= 0, "records_affected must be non-negative"
		assert description, "description required"
		notif_id = uuid7str()
		ref = f"BREACH-{datetime.utcnow().strftime('%Y%m%d')}-{notif_id[:6].upper()}"
		# HIPAA: notify within 60 days; GDPR: 72 hours
		hipaa_deadline = (discovered_at + timedelta(days=60)).isoformat()
		gdpr_deadline = (discovered_at + timedelta(hours=72)).isoformat()
		record: dict[str, Any] = {
			"id": notif_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"breach_type": breach_type,
			"records_affected": records_affected,
			"description": description,
			"discovered_at": discovered_at.isoformat(),
			"reported_by": self._actor_id,
			"reported_at": datetime.utcnow().isoformat(),
			"hipaa_60_day_deadline": hipaa_deadline,
			"gdpr_72h_deadline": gdpr_deadline,
			"large_breach": records_affected >= 500,
			"media_notice_required": records_affected >= 500,
			"status": "filed",
		}
		self._audit(tenant_id, "data_breach_notified", notif_id)
		_log_op("data_breach_notification", tenant_id, notif_id)
		return record

	async def regulatory_calendar(self, tenant_id: str) -> dict[str, Any]:
		"""Return upcoming regulatory deadlines for licenses, accreditations, and submissions."""
		licenses = [l for (tid, _), l in self._licenses.items() if tid == tenant_id]
		accreditations = [a for (tid, _), a in self._accreditations.items() if tid == tenant_id]
		submissions = [s for (tid, _), s in self._submissions.items() if tid == tenant_id]
		now = datetime.utcnow()
		upcoming: list[dict[str, Any]] = []
		for lic in licenses:
			if lic.status == "active" and lic.days_to_expiry <= 90:
				upcoming.append({"type": "license_expiry", "id": lic.id, "license_type": lic.license_type, "due_date": lic.expiry_date.isoformat(), "days_remaining": lic.days_to_expiry})
		for acc in accreditations:
			days = (acc.expiry_date - now).days
			if days <= 180:
				upcoming.append({"type": "accreditation_expiry", "id": acc.id, "body": acc.accreditation_body, "due_date": acc.expiry_date.isoformat(), "days_remaining": days})
		upcoming.sort(key=lambda x: x["days_remaining"])
		_log_op("regulatory_calendar", tenant_id, "calendar")
		return {
			"tenant_id": tenant_id,
			"upcoming_items": len(upcoming),
			"critical_items": sum(1 for x in upcoming if x["days_remaining"] <= 30),
			"calendar": upcoming,
			"generated_at": now.isoformat(),
		}

	async def audit_trail(self, tenant_id: str, entity_id: str | None = None) -> list[dict[str, Any]]:
		"""Return audit trail events, optionally filtered by entity_id."""
		events = [e for e in self._audit_events if e["tenant_id"] == tenant_id]
		if entity_id:
			events = [e for e in events if e["entity_id"] == entity_id]
		return events

	async def export_compliance_report(self, tenant_id: str, period: str, format: str = "json") -> dict[str, Any]:
		"""Export compliance report metadata for a period."""
		export_id = uuid7str()
		_log_op("export_compliance_report", tenant_id, export_id)
		return {
			"export_id": export_id,
			"tenant_id": tenant_id,
			"period": period,
			"format": format,
			"download_ref": f"/exports/{tenant_id}/{export_id}.{format}",
			"status": "ready",
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return service health and store sizes."""
		return {
			"service": "HealthcareRegulatoryService",
			"status": "healthy",
			"licenses": len(self._licenses),
			"accreditations": len(self._accreditations),
			"incidents": len(self._incidents),
			"submissions": len(self._submissions),
			"corrective_actions": len(self._corrective_actions),
			"inspections": len(self._inspections),
			"hipaa_assessments": len(self._hipaa_assessments),
			"audit_events": len(self._audit_events),
			"checked_at": datetime.utcnow().isoformat(),
		}

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			logger.warning("reg.rule_denied rule=%s", result["rule"])
			raise PolicyViolationError(result["reason"])

	async def renewal_pipeline(
		self,
		tenant_id: str,
		days_ahead: int = 90,
	) -> dict[str, Any]:
		"""Return all licenses and accreditations due for renewal within N days."""
		now = datetime.utcnow()
		pipeline: list[dict[str, Any]] = []
		for (tid, _), lic in self._licenses.items():
			if tid != tenant_id:
				continue
			if lic.expiry_date:
				days_remaining = (lic.expiry_date - now).days
				if 0 <= days_remaining <= days_ahead:
					pipeline.append({
						"type": "license",
						"id": lic.id,
						"name": getattr(lic, "license_type", ""),
						"expiry_date": lic.expiry_date.isoformat(),
						"days_remaining": days_remaining,
						"status": lic.status,
					})
		for (tid, _), acc in self._accreditations.items():
			if tid != tenant_id:
				continue
			if acc.valid_until:
				days_remaining = (acc.valid_until - now).days
				if 0 <= days_remaining <= days_ahead:
					pipeline.append({
						"type": "accreditation",
						"id": acc.id,
						"name": getattr(acc, "accreditation_body", ""),
						"expiry_date": acc.valid_until.isoformat(),
						"days_remaining": days_remaining,
						"status": acc.status,
					})
		pipeline.sort(key=lambda x: x["days_remaining"])
		_log_op("renewal_pipeline", tenant_id, f"{days_ahead}d")
		return {
			"tenant_id": tenant_id,
			"days_ahead": days_ahead,
			"renewal_count": len(pipeline),
			"items": pipeline,
			"generated_at": now.isoformat(),
		}

	async def post_market_plan(
		self,
		tenant_id: str,
		product_id: str,
		surveillance_interval_days: int = 180,
		signal_thresholds: dict[str, Any] | None = None,
		created_by: str = "quality_manager",
	) -> dict[str, Any]:
		"""Create a post-market surveillance plan for a regulated product."""
		plan_id = uuid7str()
		plan: dict[str, Any] = {
			"plan_id": plan_id,
			"tenant_id": tenant_id,
			"product_id": product_id,
			"surveillance_interval_days": surveillance_interval_days,
			"signal_thresholds": signal_thresholds or {"adverse_events": 5, "complaints": 10},
			"next_review_date": (datetime.utcnow() + __import__("datetime").timedelta(days=surveillance_interval_days)).isoformat(),
			"created_by": created_by,
			"status": "active",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "post_market_plan_created", plan_id)
		_log_op("post_market_plan", tenant_id, plan_id)
		return plan

	async def regulatory_kpi_summary(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return a concise regulatory KPI card for dashboard consumption."""
		licenses = [l for (tid, _), l in self._licenses.items() if tid == tenant_id]
		accreditations = [a for (tid, _), a in self._accreditations.items() if tid == tenant_id]
		incidents = [i for (tid, _), i in self._incidents.items() if tid == tenant_id]
		submissions = [s for (tid, _), s in self._submissions.items() if tid == tenant_id]
		corrective = [c for (tid, _), c in self._corrective_actions.items() if tid == tenant_id]
		now = datetime.utcnow()
		expiring_soon = sum(
			1 for l in licenses
			if l.expiry_date and 0 <= (l.expiry_date - now).days <= 90
		)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_licenses": len(licenses),
			"active_licenses": sum(1 for l in licenses if l.status == "active"),
			"licenses_expiring_90d": expiring_soon,
			"accreditations": len(accreditations),
			"regulatory_incidents": len(incidents),
			"pending_submissions": sum(1 for s in submissions if s.status in {"draft", "submitted"}),
			"open_corrective_actions": sum(1 for c in corrective if c.status == "open"),
			"generated_at": now.isoformat(),
		}

	def _audit(self, tenant_id: str, event: str, entity_id: str) -> None:
		self._audit_events.append({"tenant_id": tenant_id, "event": event, "entity_id": entity_id, "timestamp": datetime.utcnow().isoformat()})
