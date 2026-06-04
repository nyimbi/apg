"""Compliance management service for the APG COMP capability."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .capability_contract import DEFAULT_CONFIGURATION, evaluate_capability_rules, get_capability_contract
from .compliance_engine import assessment_result, evidence_age_days, finding_age_days, framework_coverage, stable_digest
from .models import (
	AttestationRecord,
	CompLifecycleBatchRecord,
	ComplianceAgentRecord,
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
		self._compliance_agents: dict[str, ComplianceAgentRecord] = {}
		self._lifecycle_batches: dict[str, CompLifecycleBatchRecord] = {}
		self._audit_events: list[ComplianceAuditEvent] = []
		self._agent_runtimes = {_normalize_token(runtime) for runtime in DEFAULT_CONFIGURATION["agents"]["supported_runtimes"]}
		self._agent_roles = {_normalize_token(role) for role in DEFAULT_CONFIGURATION["agents"]["supported_roles"]}
		self._privileged_agent_roles = {_normalize_token(role) for role in DEFAULT_CONFIGURATION["agents"]["privileged_roles"]}
		self._lifecycle_operations = {_normalize_token(operation) for operation in DEFAULT_CONFIGURATION["streaming"]["required_operations"]}

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
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_framework",
			"framework_owner_assigned": bool(owner),
			"obligations_mapped": bool(obligations),
			"policy_version_present": bool(policy_version),
			"duplicate_framework": self._key(tenant_id, framework_id) in self._frameworks,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		framework = ComplianceFramework(
			id=framework_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			obligations=list(obligations),
			policy_version=policy_version,
		)
		self._frameworks[self._key(tenant_id, framework_id)] = framework
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
		framework = self._require_framework(framework_id, tenant_id)
		frequency = testing_frequency_days or DEFAULT_CONFIGURATION["controls"]["testing_frequency_days"]
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_control",
			"framework_present": bool(framework),
			"control_name_present": bool(name),
			"control_owner_assigned": bool(owner),
			"testing_frequency_days": frequency,
			"regulated_data_scope": regulated_data_scope,
			"dlp_policy_linked": dlp_policy_linked,
			"state_change_requested": True,
			"audit_event_recorded": True,
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
			testing_frequency_days=frequency,
		)
		self._controls[self._key(tenant_id, control_id)] = control
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
		control = self._require_control(control_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "record_evidence",
			"control_present": bool(control),
			"evidence_source_present": bool(source),
			"evidence_collector_present": bool(collected_by),
			"evidence_encrypted": bool(encrypted),
			"immutable_reference_present": bool(immutable_reference),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
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
		self._evidence[self._key(tenant_id, evidence_id)] = evidence
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
		control = self._require_control(control_id, tenant_id)
		evidence = self._require_evidence(evidence_id, tenant_id, control_id)
		age = evidence_age_days(evidence.collected_at, now)
		open_finding_ids = [
			finding.id
			for finding in self._findings.values()
			if finding.tenant_id == tenant_id and finding.control_id == control_id and finding.status == "open"
		]
		preliminary_result = assessment_result(age, DEFAULT_CONFIGURATION["evidence"]["evidence_freshness_days"], bool(open_finding_ids))
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "assess_control",
			"tester_present": bool(tested_by),
			"tester_is_control_owner": tested_by == control.owner,
			"evidence_age_days": age,
			"evidence_refresh_completed": False,
			"assessment_failed": preliminary_result != "effective",
			"finding_linked": bool(open_finding_ids),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		required_actions = [action.get("required_action") for action in result["actions"] if action.get("decision") == "require_review"]
		assessment = ControlAssessment(
			id=assessment_id,
			tenant_id=tenant_id,
			control_id=control_id,
			evidence_id=evidence_id,
			result="review_required" if required_actions else preliminary_result,
			tested_by=tested_by,
			evidence_age_days=age,
			findings=open_finding_ids,
		)
		self._assessments[self._key(tenant_id, assessment_id)] = assessment
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
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "open_finding",
			"finding_owner_assigned": bool(owner),
			"finding_description_present": bool(description),
			"high_severity_finding": severity in {"high", "critical"},
			"remediation_plan_present": bool(remediation_plan),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
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
		self._findings[self._key(tenant_id, finding_id)] = finding
		self._record_audit(tenant_id, "finding_opened", finding_id, owner, finding.to_dict())
		return finding.to_dict()

	def resolve_finding(
		self,
		finding_id: str,
		tenant_id: str,
		resolved_by: str,
		resolution: str,
		evidence_id: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		finding = self._require_finding(finding_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "resolve_finding",
			"resolution_evidence_present": bool(evidence_id),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		finding.status = "resolved"
		finding.remediation_plan = resolution or finding.remediation_plan
		self._record_audit(tenant_id, "finding_resolved", finding_id, resolved_by, {"resolution": resolution, "evidence_id": evidence_id})
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
		framework = self._require_framework(framework_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "prepare_report",
			"framework_present": bool(framework),
			"report_period_present": bool(period),
			"report_preparer_present": bool(prepared_by),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
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
		self._reports[self._key(tenant_id, report_id)] = report
		self._record_audit(tenant_id, "report_prepared", report_id, prepared_by, report.to_dict())
		return report.to_dict()

	def approve_report(self, report_id: str, tenant_id: str, approved_by: str) -> dict[str, Any]:
		report = self._require_report(report_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "approve_report",
			"approver_is_preparer": approved_by == report.prepared_by,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		report.status = "approved"
		report.approved_by = approved_by
		report.approved_at = utc_now()
		self._record_audit(tenant_id, "report_approved", report_id, approved_by, report.to_dict())
		return report.to_dict()

	def attest_report(self, attestation_id: str, report_id: str, tenant_id: str, attested_by: str, statement: str) -> dict[str, Any]:
		report = self._require_report(report_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "attest_report",
			"report_approved": report.status == "approved",
			"attestation_statement_present": bool(statement),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		attestation = AttestationRecord(
			id=attestation_id,
			tenant_id=tenant_id,
			report_id=report_id,
			attested_by=attested_by,
			statement=statement,
		)
		self._attestations[self._key(tenant_id, attestation_id)] = attestation
		self._record_audit(tenant_id, "report_attested", attestation_id, attested_by, attestation.to_dict())
		return attestation.to_dict()

	def publish_report(self, report_id: str, tenant_id: str) -> dict[str, Any]:
		report = self._require_report(report_id, tenant_id)
		has_attestation = any(attestation.tenant_id == tenant_id and attestation.report_id == report_id for attestation in self._attestations.values())
		open_critical_findings = any(finding.tenant_id == tenant_id and finding.status == "open" and finding.severity == "critical" for finding in self._findings.values())
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "publish_report",
			"approval_recorded": report.status == "approved",
			"attestation_recorded": has_attestation,
			"open_critical_findings": open_critical_findings,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		report.status = "published"
		report.published_at = utc_now()
		self._record_audit(tenant_id, "report_published", report_id, report.approved_by or "system", report.to_dict())
		return report.to_dict()

	def register_compliance_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		record_key = self._key(tenant_id, agent_id)
		if record_key in self._compliance_agents:
			raise ValueError(f"compliance_agent_already_exists:{agent_id}")
		runtime_value = _normalize_token(runtime)
		role_value = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_compliance_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"scope_present": bool(str(scope or "").strip()),
			"owner_present": bool(str(owner or "").strip()),
			"purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		if not str(name or "").strip():
			raise ValueError("compliance_agent_name_required")
		agent = ComplianceAgentRecord(
			id=agent_id,
			tenant_id=tenant_id,
			name=str(name).strip(),
			runtime=runtime_value,
			role=role_value,
			scope=str(scope).strip(),
			owner=str(owner).strip(),
			purpose=str(purpose).strip(),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
			status="pending_review" if result["decision"] == "require_review" else "active",
		)
		self._compliance_agents[record_key] = agent
		self._record_audit(tenant_id, "compliance_agent_registered", agent_id, owner, {**agent.to_dict(), "rule_decision": result["decision"]})
		return agent.to_dict()

	def validate_comp_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "compliance_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("comp_lifecycle_batch_empty")
		stream_value = _normalize_token(event_stream)
		operation_value = _normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_comp_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_comp_lifecycle_batch",
			"event_stream": stream_value,
			"mutation_count": mutation_count,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		accepted = result["decision"] == "allow"
		record_id = batch_id or f"comp-batch-{len(self._lifecycle_batches) + 1:06d}"
		record = CompLifecycleBatchRecord(
			id=record_id,
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			operation=operation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			status="accepted" if accepted else "denied",
		)
		self._lifecycle_batches[self._key(tenant_id, record_id)] = record
		self._record_audit(tenant_id, f"comp_lifecycle_batch_{record.status}", record_id, "bytewax", record.to_dict())
		self._raise_if_denied(result)
		return record.to_dict()

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
			"compliance_agent_count": len(self.list_compliance_agents(tenant_id)),
			"pending_agent_review_count": sum(1 for item in self.list_compliance_agents(tenant_id) if item["status"] == "pending_review"),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": sum(1 for item in self.list_lifecycle_batches(tenant_id) if item["status"] == "denied"),
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

	def list_compliance_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._compliance_agents.values(), tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._lifecycle_batches.values(), tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if tenant_id is not None:
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.to_dict() for event in sorted(events, key=lambda item: item.id)]

	# ── New methods ────────────────────────────────────────────────────────────

	def obligation_register(
		self,
		obligation_id: str,
		tenant_id: str,
		framework_id: str,
		title: str,
		owner: str,
		regulation: str,
		due_date: datetime | None = None,
	) -> dict[str, Any]:
		"""Register a regulatory obligation against a framework."""
		self._require_tenant(tenant_id)
		self._require_framework(framework_id, tenant_id)
		record: dict[str, Any] = {
			"id": obligation_id,
			"tenant_id": tenant_id,
			"framework_id": framework_id,
			"title": title,
			"owner": owner,
			"regulation": regulation,
			"due_date": due_date.isoformat() if due_date else None,
			"status": "open",
			"created_at": utc_now().isoformat(),
		}
		self._frameworks[self._key(tenant_id, framework_id)].obligations.append(obligation_id)
		self._record_audit(tenant_id, "obligation_registered", obligation_id, owner, record)
		return record

	def gap_assess(
		self,
		tenant_id: str,
		framework_id: str,
	) -> dict[str, Any]:
		"""Return a gap assessment: controls without recent effective assessments."""
		self._require_tenant(tenant_id)
		self._require_framework(framework_id, tenant_id)
		controls = [c for c in self._controls.values() if c.tenant_id == tenant_id and c.framework_id == framework_id]
		assessed_ids = {a.control_id for a in self._assessments.values() if a.tenant_id == tenant_id and a.result == "effective"}
		gaps = [c.to_dict() for c in controls if c.id not in assessed_ids]
		return {
			"framework_id": framework_id,
			"tenant_id": tenant_id,
			"total_controls": len(controls),
			"assessed_controls": len(controls) - len(gaps),
			"gap_count": len(gaps),
			"gaps": gaps,
		}

	def evidence_upload(
		self,
		evidence_id: str,
		tenant_id: str,
		control_id: str,
		source: str,
		collected_by: str,
		file_ref: str,
		encrypted: bool = True,
	) -> dict[str, Any]:
		"""Upload evidence file reference for a control."""
		return self.record_evidence(
			evidence_id=evidence_id,
			tenant_id=tenant_id,
			control_id=control_id,
			source=source,
			collected_by=collected_by,
			encrypted=encrypted,
			immutable_reference=file_ref,
		)

	def control_map(self, tenant_id: str, framework_id: str) -> list[dict[str, Any]]:
		"""Return control-to-assessment mapping for a framework."""
		self._require_tenant(tenant_id)
		self._require_framework(framework_id, tenant_id)
		controls = [c for c in self._controls.values() if c.tenant_id == tenant_id and c.framework_id == framework_id]
		result: list[dict[str, Any]] = []
		for ctrl in controls:
			assessments = [a.to_dict() for a in self._assessments.values() if a.tenant_id == tenant_id and a.control_id == ctrl.id]
			result.append({"control": ctrl.to_dict(), "assessments": assessments})
		return result

	def policy_enforce(
		self,
		tenant_id: str,
		policy_id: str,
		policy_name: str,
		rules: list[str],
		enforced_by: str,
	) -> dict[str, Any]:
		"""Record a policy enforcement action."""
		self._require_tenant(tenant_id)
		record: dict[str, Any] = {
			"id": policy_id,
			"tenant_id": tenant_id,
			"policy_name": policy_name,
			"rules": rules,
			"enforced_by": enforced_by,
			"enforced_at": utc_now().isoformat(),
			"status": "enforced",
		}
		self._record_audit(tenant_id, "policy_enforced", policy_id, enforced_by, record)
		return record

	def training_assign(
		self,
		assignment_id: str,
		tenant_id: str,
		control_id: str,
		assignee: str,
		training_ref: str,
		due_days: int = 30,
	) -> dict[str, Any]:
		"""Assign compliance training to a user for a control."""
		self._require_tenant(tenant_id)
		self._require_control(control_id, tenant_id)
		due_at = utc_now() + timedelta(days=due_days)
		record: dict[str, Any] = {
			"id": assignment_id,
			"tenant_id": tenant_id,
			"control_id": control_id,
			"assignee": assignee,
			"training_ref": training_ref,
			"due_at": due_at.isoformat(),
			"status": "assigned",
			"assigned_at": utc_now().isoformat(),
		}
		self._record_audit(tenant_id, "training_assigned", assignment_id, assignee, record)
		return record

	def audit_schedule(
		self,
		schedule_id: str,
		tenant_id: str,
		framework_id: str,
		audit_date: datetime,
		auditor: str,
		scope: str,
	) -> dict[str, Any]:
		"""Schedule an internal or external audit for a framework."""
		self._require_tenant(tenant_id)
		self._require_framework(framework_id, tenant_id)
		record: dict[str, Any] = {
			"id": schedule_id,
			"tenant_id": tenant_id,
			"framework_id": framework_id,
			"audit_date": audit_date.isoformat(),
			"auditor": auditor,
			"scope": scope,
			"status": "scheduled",
			"created_at": utc_now().isoformat(),
		}
		self._record_audit(tenant_id, "audit_scheduled", schedule_id, auditor, record)
		return record

	def finding_create(
		self,
		finding_id: str,
		tenant_id: str,
		control_id: str,
		severity: str,
		description: str,
		owner: str,
		remediation_plan: str = "",
	) -> dict[str, Any]:
		"""Alias for open_finding with explicit naming."""
		return self.open_finding(
			finding_id=finding_id,
			tenant_id=tenant_id,
			control_id=control_id,
			severity=severity,
			description=description,
			owner=owner,
			remediation_plan=remediation_plan,
		)

	def remediation_track(self, tenant_id: str, finding_id: str, progress_note: str, updated_by: str) -> dict[str, Any]:
		"""Add a progress note to an open finding's remediation."""
		self._require_tenant(tenant_id)
		finding = self._require_finding(finding_id, tenant_id)
		finding.remediation_plan = f"{finding.remediation_plan}\n[{utc_now().isoformat()}] {progress_note}".strip()
		self._record_audit(tenant_id, "remediation_tracked", finding_id, updated_by, {"note": progress_note})
		return finding.to_dict()

	def regulatory_alert(
		self,
		alert_id: str,
		tenant_id: str,
		regulation: str,
		summary: str,
		severity: str,
		effective_date: datetime | None = None,
	) -> dict[str, Any]:
		"""Record an incoming regulatory change alert."""
		self._require_tenant(tenant_id)
		record: dict[str, Any] = {
			"id": alert_id,
			"tenant_id": tenant_id,
			"regulation": regulation,
			"summary": summary,
			"severity": severity,
			"effective_date": effective_date.isoformat() if effective_date else None,
			"status": "new",
			"created_at": utc_now().isoformat(),
		}
		self._record_audit(tenant_id, "regulatory_alert_created", alert_id, "system", record)
		return record

	def iso27001_checklist(self, tenant_id: str) -> dict[str, Any]:
		"""Return ISO 27001 control coverage status for the tenant."""
		self._require_tenant(tenant_id)
		iso_frameworks = [f for f in self._frameworks.values() if f.tenant_id == tenant_id and "27001" in f.name]
		controls = [c for c in self._controls.values() if c.tenant_id == tenant_id]
		assessed = [a for a in self._assessments.values() if a.tenant_id == tenant_id and a.result == "effective"]
		return {
			"tenant_id": tenant_id,
			"iso27001_frameworks": [f.to_dict() for f in iso_frameworks],
			"total_controls": len(controls),
			"effective_controls": len(assessed),
			"coverage_pct": round(len(assessed) / max(len(controls), 1) * 100, 1),
		}

	def gdpr_dpia(
		self,
		dpia_id: str,
		tenant_id: str,
		processing_activity: str,
		data_types: list[str],
		risk_level: str,
		owner: str,
	) -> dict[str, Any]:
		"""Record a GDPR Data Protection Impact Assessment."""
		self._require_tenant(tenant_id)
		record: dict[str, Any] = {
			"id": dpia_id,
			"tenant_id": tenant_id,
			"processing_activity": processing_activity,
			"data_types": data_types,
			"risk_level": risk_level,
			"owner": owner,
			"status": "draft",
			"created_at": utc_now().isoformat(),
		}
		self._record_audit(tenant_id, "gdpr_dpia_created", dpia_id, owner, record)
		return record

	def soc2_evidence(self, tenant_id: str) -> dict[str, Any]:
		"""Aggregate SOC 2 evidence items for the tenant."""
		self._require_tenant(tenant_id)
		evidence_items = [e.to_dict() for e in self._evidence.values() if e.tenant_id == tenant_id and e.encrypted]
		attestations = [a.to_dict() for a in self._attestations.values() if a.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"encrypted_evidence_count": len(evidence_items),
			"attestation_count": len(attestations),
			"evidence": evidence_items,
			"attestations": attestations,
		}

	def compliance_dashboard(self, tenant_id: str) -> dict[str, Any]:
		"""Alias for dashboard_summary with extended breakdown."""
		summary = self.dashboard_summary(tenant_id)
		summary["gap_assessment"] = self.gap_assess(tenant_id, next(
			(f.id for f in self._frameworks.values() if f.tenant_id == tenant_id),
			"__none__",
		)) if any(f.tenant_id == tenant_id for f in self._frameworks.values()) else {}
		return summary

	def risk_integrate(
		self,
		tenant_id: str,
		risk_id: str,
		control_id: str,
		risk_score: float,
		risk_owner: str,
	) -> dict[str, Any]:
		"""Link an external risk item to a compliance control."""
		self._require_tenant(tenant_id)
		self._require_control(control_id, tenant_id)
		record: dict[str, Any] = {
			"id": risk_id,
			"tenant_id": tenant_id,
			"control_id": control_id,
			"risk_score": max(0.0, min(1.0, risk_score)),
			"risk_owner": risk_owner,
			"integrated_at": utc_now().isoformat(),
		}
		self._record_audit(tenant_id, "risk_integrated", risk_id, risk_owner, record)
		return record

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
		framework = self._frameworks.get(self._key(tenant_id, framework_id))
		if framework is None or framework.tenant_id != tenant_id:
			raise KeyError("framework_not_found")
		return framework

	def _require_control(self, control_id: str, tenant_id: str) -> ComplianceControl:
		control = self._controls.get(self._key(tenant_id, control_id))
		if control is None or control.tenant_id != tenant_id:
			raise KeyError("control_not_found")
		return control

	def _require_evidence(self, evidence_id: str, tenant_id: str, control_id: str) -> EvidenceRecord:
		evidence = self._evidence.get(self._key(tenant_id, evidence_id))
		if evidence is None or evidence.tenant_id != tenant_id or evidence.control_id != control_id:
			raise KeyError("evidence_not_found")
		return evidence

	def _require_finding(self, finding_id: str, tenant_id: str) -> ComplianceFinding:
		finding = self._findings.get(self._key(tenant_id, finding_id))
		if finding is None or finding.tenant_id != tenant_id:
			raise KeyError("finding_not_found")
		return finding

	def _require_report(self, report_id: str, tenant_id: str) -> ComplianceReport:
		report = self._reports.get(self._key(tenant_id, report_id))
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

	@staticmethod
	def _key(tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"


def _normalize_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
