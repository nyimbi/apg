"""Compliance management service for the APG COMP capability."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .capability_contract import DEFAULT_CONFIGURATION, evaluate_capability_rules, get_capability_contract
from .compliance_engine import assessment_result, evidence_age_days, finding_age_days, framework_coverage, stable_digest
from .models import (
	AttestationRecord,
	ComplianceAuditEvent,
	ComplianceControl,
	ComplianceFinding,
	ComplianceFramework,
	ComplianceReport,
	ControlAssessment,
	EvidenceRecord,
	utc_now,
)


class CompService:
	"""Tenant-scoped compliance framework, control, evidence, and reporting service."""

	def __init__(self) -> None:
		self._frameworks: dict[str, ComplianceFramework] = {}
		self._controls: dict[str, ComplianceControl] = {}
		self._evidence: dict[str, EvidenceRecord] = {}
		self._assessments: dict[str, ControlAssessment] = {}
		self._findings: dict[str, ComplianceFinding] = {}
		self._reports: dict[str, ComplianceReport] = {}
		self._attestations: dict[str, AttestationRecord] = {}
		self._audit_events: list[ComplianceAuditEvent] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_framework(
		self,
		framework_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		obligations: list[str],
		policy_version: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			raise PermissionError("framework_owner_required")
		if DEFAULT_CONFIGURATION["frameworks"]["obligation_mapping_required"] and not obligations:
			raise ValueError("obligation_mapping_required")
		framework = ComplianceFramework(
			id=framework_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			obligations=list(obligations),
			policy_version=policy_version,
		)
		self._frameworks[framework_id] = framework
		self._record_audit(tenant_id, "framework_registered", framework_id, owner, framework.to_dict())
		return framework.to_dict()

	def create_control(
		self,
		control_id: str,
		tenant_id: str,
		framework_id: str,
		name: str,
		owner: str,
		control_type: str = "preventive",
		regulated_data_scope: bool = False,
		dlp_policy_linked: bool = False,
		testing_frequency_days: int | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_framework(framework_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_control",
			"control_owner_assigned": bool(owner),
			"regulated_data_scope": regulated_data_scope,
			"dlp_policy_linked": dlp_policy_linked,
		})
		self._raise_if_denied(result)
		control = ComplianceControl(
			id=control_id,
			tenant_id=tenant_id,
			framework_id=framework_id,
			name=name,
			owner=owner,
			control_type=control_type,
			regulated_data_scope=regulated_data_scope,
			dlp_policy_linked=dlp_policy_linked,
			testing_frequency_days=testing_frequency_days or DEFAULT_CONFIGURATION["controls"]["testing_frequency_days"],
		)
		self._controls[control_id] = control
		self._record_audit(tenant_id, "control_created", control_id, owner, control.to_dict())
		return control.to_dict()

	def record_evidence(
		self,
		evidence_id: str,
		tenant_id: str,
		control_id: str,
		source: str,
		collected_by: str,
		encrypted: bool,
		immutable_reference: str | None = None,
		collected_at: datetime | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_control(control_id, tenant_id)
		if DEFAULT_CONFIGURATION["evidence"]["encrypted_evidence_required"] and not encrypted:
			raise PermissionError("encrypted_evidence_required")
		if DEFAULT_CONFIGURATION["evidence"]["immutable_audit_required"] and not immutable_reference:
			raise PermissionError("immutable_evidence_reference_required")
		evidence = EvidenceRecord(
			id=evidence_id,
			tenant_id=tenant_id,
			control_id=control_id,
			source=source,
			collected_by=collected_by,
			encrypted=encrypted,
			immutable_reference=immutable_reference or stable_digest({"evidence_id": evidence_id, "source": source}),
			collected_at=collected_at or utc_now(),
			metadata=dict(metadata or {}),
		)
		self._evidence[evidence_id] = evidence
		self._record_audit(tenant_id, "evidence_recorded", evidence_id, collected_by, evidence.to_dict())
		return evidence.to_dict()

	def assess_control(
		self,
		assessment_id: str,
		tenant_id: str,
		control_id: str,
		evidence_id: str,
		tested_by: str,
		now: datetime | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_control(control_id, tenant_id)
		evidence = self._require_evidence(evidence_id, tenant_id, control_id)
		age = evidence_age_days(evidence.collected_at, now)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"evidence_age_days": age,
			"evidence_refresh_completed": False,
		})
		self._raise_if_denied(result)
		open_finding_ids = [
			finding.id
			for finding in self._findings.values()
			if finding.tenant_id == tenant_id and finding.control_id == control_id and finding.status == "open"
		]
		assessment = ControlAssessment(
			id=assessment_id,
			tenant_id=tenant_id,
			control_id=control_id,
			evidence_id=evidence_id,
			result=assessment_result(age, DEFAULT_CONFIGURATION["evidence"]["evidence_freshness_days"], bool(open_finding_ids)),
			tested_by=tested_by,
			evidence_age_days=age,
			findings=open_finding_ids,
		)
		self._assessments[assessment_id] = assessment
		self._record_audit(tenant_id, "control_assessed", assessment_id, tested_by, assessment.to_dict())
		return assessment.to_dict()

	def open_finding(
		self,
		finding_id: str,
		tenant_id: str,
		control_id: str,
		severity: str,
		description: str,
		owner: str,
		created_at: datetime | None = None,
		remediation_plan: str = "",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_control(control_id, tenant_id)
		due_at = (created_at or utc_now()) + timedelta(days=DEFAULT_CONFIGURATION["reporting"]["finding_remediation_sla_days"])
		finding = ComplianceFinding(
			id=finding_id,
			tenant_id=tenant_id,
			control_id=control_id,
			severity=severity,
			description=description,
			owner=owner,
			due_at=due_at,
			created_at=created_at or utc_now(),
			remediation_plan=remediation_plan,
		)
		self._findings[finding_id] = finding
		self._record_audit(tenant_id, "finding_opened", finding_id, owner, finding.to_dict())
		return finding.to_dict()

	def escalate_overdue_findings(self, tenant_id: str, now: datetime | None = None) -> list[dict[str, Any]]:
		self._require_tenant(tenant_id)
		escalated: list[dict[str, Any]] = []
		for finding in self._findings.values():
			if finding.tenant_id != tenant_id or finding.status != "open":
				continue
			age = finding_age_days(finding.created_at, now)
			result = self.evaluate({
				"tenant_context_present": True,
				"finding_age_days": age,
				"escalation_recorded": finding.escalated,
			})
			if result["decision"] == "require_review":
				finding.escalated = True
				escalated.append(finding.to_dict())
				self._record_audit(tenant_id, "finding_escalated", finding.id, finding.owner, finding.to_dict())
		return escalated

	def prepare_report(
		self,
		report_id: str,
		tenant_id: str,
		framework_id: str,
		period: str,
		prepared_by: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_framework(framework_id, tenant_id)
		control_count = len([control for control in self._controls.values() if control.tenant_id == tenant_id and control.framework_id == framework_id])
		finding_count = len([finding for finding in self._findings.values() if finding.tenant_id == tenant_id and finding.status == "open"])
		report = ComplianceReport(
			id=report_id,
			tenant_id=tenant_id,
			framework_id=framework_id,
			period=period,
			prepared_by=prepared_by,
			control_count=control_count,
			finding_count=finding_count,
		)
		self._reports[report_id] = report
		self._record_audit(tenant_id, "report_prepared", report_id, prepared_by, report.to_dict())
		return report.to_dict()

	def approve_report(self, report_id: str, tenant_id: str, approved_by: str) -> dict[str, Any]:
		report = self._require_report(report_id, tenant_id)
		report.status = "approved"
		report.approved_by = approved_by
		report.approved_at = utc_now()
		self._record_audit(tenant_id, "report_approved", report_id, approved_by, report.to_dict())
		return report.to_dict()

	def attest_report(self, attestation_id: str, report_id: str, tenant_id: str, attested_by: str, statement: str) -> dict[str, Any]:
		report = self._require_report(report_id, tenant_id)
		if report.status != "approved":
			raise PermissionError("report_approval_required")
		attestation = AttestationRecord(
			id=attestation_id,
			tenant_id=tenant_id,
			report_id=report_id,
			attested_by=attested_by,
			statement=statement,
		)
		self._attestations[attestation_id] = attestation
		self._record_audit(tenant_id, "report_attested", attestation_id, attested_by, attestation.to_dict())
		return attestation.to_dict()

	def publish_report(self, report_id: str, tenant_id: str) -> dict[str, Any]:
		report = self._require_report(report_id, tenant_id)
		has_attestation = any(attestation.tenant_id == tenant_id and attestation.report_id == report_id for attestation in self._attestations.values())
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "publish_report",
			"approval_recorded": report.status == "approved" and has_attestation,
		})
		self._raise_if_denied(result)
		report.status = "published"
		report.published_at = utc_now()
		self._record_audit(tenant_id, "report_published", report_id, report.approved_by or "system", report.to_dict())
		return report.to_dict()

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		controls = [control for control in self._controls.values() if control.tenant_id == tenant_id]
		assessments = [assessment for assessment in self._assessments.values() if assessment.tenant_id == tenant_id]
		findings = [finding for finding in self._findings.values() if finding.tenant_id == tenant_id and finding.status == "open"]
		failing_count = len([assessment for assessment in assessments if assessment.result != "effective"]) + len(findings)
		coverage = framework_coverage(len(controls), len({assessment.control_id for assessment in assessments}), failing_count)
		return {
			"tenant_id": tenant_id,
			"framework_count": len([framework for framework in self._frameworks.values() if framework.tenant_id == tenant_id]),
			"control_count": len(controls),
			"evidence_count": len([evidence for evidence in self._evidence.values() if evidence.tenant_id == tenant_id]),
			"assessment_count": len(assessments),
			"open_finding_count": len(findings),
			"escalated_finding_count": len([finding for finding in findings if finding.escalated]),
			"report_count": len([report for report in self._reports.values() if report.tenant_id == tenant_id]),
			"attestation_count": len([attestation for attestation in self._attestations.values() if attestation.tenant_id == tenant_id]),
			"coverage": coverage,
		}

	def list_frameworks(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._frameworks.values(), tenant_id)

	def list_controls(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._controls.values(), tenant_id)

	def list_evidence(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._evidence.values(), tenant_id)

	def list_assessments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._assessments.values(), tenant_id)

	def list_findings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._findings.values(), tenant_id)

	def list_reports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._reports.values(), tenant_id)

	def list_attestations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._attestations.values(), tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if tenant_id is not None:
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.to_dict() for event in sorted(events, key=lambda item: item.id)]

	def _record_audit(self, tenant_id: str, event_type: str, subject_id: str, actor: str, payload: dict[str, Any]) -> None:
		event = ComplianceAuditEvent(
			id=f"audit-{len(self._audit_events) + 1:06d}",
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			actor=actor,
			payload_hash=stable_digest(payload),
		)
		self._audit_events.append(event)

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_framework(self, framework_id: str, tenant_id: str) -> ComplianceFramework:
		framework = self._frameworks.get(framework_id)
		if framework is None or framework.tenant_id != tenant_id:
			raise KeyError("framework_not_found")
		return framework

	def _require_control(self, control_id: str, tenant_id: str) -> ComplianceControl:
		control = self._controls.get(control_id)
		if control is None or control.tenant_id != tenant_id:
			raise KeyError("control_not_found")
		return control

	def _require_evidence(self, evidence_id: str, tenant_id: str, control_id: str) -> EvidenceRecord:
		evidence = self._evidence.get(evidence_id)
		if evidence is None or evidence.tenant_id != tenant_id or evidence.control_id != control_id:
			raise KeyError("evidence_not_found")
		return evidence

	def _require_report(self, report_id: str, tenant_id: str) -> ComplianceReport:
		report = self._reports.get(report_id)
		if report is None or report.tenant_id != tenant_id:
			raise KeyError("report_not_found")
		return report

	def _tenant_sorted(self, values: Any, tenant_id: str | None) -> list[dict[str, Any]]:
		items = list(values)
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "compliance_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "compliance_policy_blocked")
