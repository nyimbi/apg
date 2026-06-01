"""Executable service layer for APG FinTech Compliance Automation."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CHECK_TYPES, SUPPORTED_CONTROL_TYPES, SUPPORTED_EVIDENCE_TYPES, SUPPORTED_OBLIGATION_TYPES, SUPPORTED_REGULATORY_FRAMEWORKS, SUPPORTED_REPORT_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SEVERITIES, SUPPORTED_STATUSES, evaluate_capability_rules, get_capability_contract
	from .compliance_runtime import check_failed, normalize_code, retention_present
	from .models import ComplianceAgent, ComplianceAttestation, ComplianceCheck, ComplianceControl, ComplianceEvidence, ComplianceIssue, ComplianceObligation, ComplianceRemediation, ComplianceReport, ComplianceReview
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CHECK_TYPES, SUPPORTED_CONTROL_TYPES, SUPPORTED_EVIDENCE_TYPES, SUPPORTED_OBLIGATION_TYPES, SUPPORTED_REGULATORY_FRAMEWORKS, SUPPORTED_REPORT_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SEVERITIES, SUPPORTED_STATUSES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from compliance_runtime import check_failed, normalize_code, retention_present  # type: ignore
	from models import ComplianceAgent, ComplianceAttestation, ComplianceCheck, ComplianceControl, ComplianceEvidence, ComplianceIssue, ComplianceObligation, ComplianceRemediation, ComplianceReport, ComplianceReview  # type: ignore


class ComplianceAutomationService:
	"""Dependency-light compliance runtime for generated APG applications."""

	def __init__(self) -> None:
		self.obligations: dict[str, ComplianceObligation] = {}
		self.controls: dict[str, ComplianceControl] = {}
		self.checks: dict[str, ComplianceCheck] = {}
		self.evidence: dict[str, ComplianceEvidence] = {}
		self.attestations: dict[str, ComplianceAttestation] = {}
		self.issues: dict[str, ComplianceIssue] = {}
		self.remediations: dict[str, ComplianceRemediation] = {}
		self.reports: dict[str, ComplianceReport] = {}
		self.reviews: dict[str, ComplianceReview] = {}
		self.agents: dict[str, ComplianceAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_obligation(self, obligation_id: str, tenant_id: str, framework: str, obligation_type: str, title: str, owner_id: str, evidence_reference: str, effective_date: str, policy_attached: bool = True) -> dict[str, Any]:
		framework = normalize_code(framework)
		obligation_type = normalize_code(obligation_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_obligation", "framework_supported": framework in SUPPORTED_REGULATORY_FRAMEWORKS, "obligation_type_supported": obligation_type in SUPPORTED_OBLIGATION_TYPES, "owner_present": bool(owner_id), "evidence_present": bool(evidence_reference), "effective_date_present": bool(effective_date)})
		item = ComplianceObligation(obligation_id, tenant_id, framework, obligation_type, title, owner_id, evidence_reference, effective_date, "active")
		self.obligations[obligation_id] = item
		self._audit(tenant_id, "compliance_obligation_registered", obligation_id)
		return item.to_dict()

	def map_control(self, control_id: str, tenant_id: str, obligation_id: str, control_type: str, owner_id: str, evidence_reference: str, frequency: str) -> dict[str, Any]:
		obligation = self._tenant_obligation_or_none(obligation_id, tenant_id)
		control_type = normalize_code(control_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "map_control", "obligation_present": obligation is not None, "control_type_supported": control_type in SUPPORTED_CONTROL_TYPES, "owner_present": bool(owner_id), "evidence_present": bool(evidence_reference), "frequency_present": bool(frequency)})
		item = ComplianceControl(control_id, tenant_id, obligation_id, control_type, owner_id, evidence_reference, frequency)
		self.controls[control_id] = item
		self._audit(tenant_id, "compliance_control_mapped", control_id)
		return item.to_dict()

	def record_check(self, check_id: str, tenant_id: str, obligation_id: str, control_id: str, check_type: str, subject_reference: str, result: str, evidence_reference: str = "") -> dict[str, Any]:
		obligation = self._tenant_obligation_or_none(obligation_id, tenant_id)
		control = self._tenant_control_or_none(control_id, tenant_id)
		check_type = normalize_code(check_type)
		result = normalize_code(result)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_check", "obligation_present": obligation is not None, "control_present": control is not None, "check_type_supported": check_type in SUPPORTED_CHECK_TYPES, "subject_present": bool(subject_reference), "result_present": bool(result), "failed_check": check_failed(result), "evidence_present": bool(evidence_reference)})
		item = ComplianceCheck(check_id, tenant_id, obligation_id, control_id, check_type, subject_reference, result, evidence_reference)
		self.checks[check_id] = item
		self._audit(tenant_id, "compliance_check_recorded", check_id)
		return item.to_dict()

	def attach_evidence(self, evidence_id: str, tenant_id: str, reference_id: str, evidence_type: str, source_reference: str, retention_days: int) -> dict[str, Any]:
		evidence_type = normalize_code(evidence_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "attach_evidence", "reference_present": bool(reference_id), "evidence_type_supported": evidence_type in SUPPORTED_EVIDENCE_TYPES, "source_present": bool(source_reference), "retention_present": retention_present(retention_days)})
		item = ComplianceEvidence(evidence_id, tenant_id, reference_id, evidence_type, source_reference, int(retention_days))
		self.evidence[evidence_id] = item
		self._audit(tenant_id, "compliance_evidence_attached", evidence_id)
		return item.to_dict()

	def record_attestation(self, attestation_id: str, tenant_id: str, obligation_id: str, attestor_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		obligation = self._tenant_obligation_or_none(obligation_id, tenant_id)
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_attestation", "obligation_present": obligation is not None, "attestor_present": bool(attestor_id), "status_supported": status in SUPPORTED_STATUSES, "evidence_present": bool(evidence_reference)})
		item = ComplianceAttestation(attestation_id, tenant_id, obligation_id, attestor_id, status, evidence_reference)
		self.attestations[attestation_id] = item
		self._audit(tenant_id, "compliance_attestation_recorded", attestation_id)
		return item.to_dict()

	def open_issue(self, issue_id: str, tenant_id: str, obligation_id: str, severity: str, owner_id: str, evidence_reference: str, due_date: str) -> dict[str, Any]:
		obligation = self._tenant_obligation_or_none(obligation_id, tenant_id)
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "open_issue", "obligation_present": obligation is not None, "severity_supported": severity in SUPPORTED_SEVERITIES, "owner_present": bool(owner_id), "evidence_present": bool(evidence_reference), "due_date_present": bool(due_date)})
		item = ComplianceIssue(issue_id, tenant_id, obligation_id, severity, owner_id, evidence_reference, due_date, "active")
		self.issues[issue_id] = item
		self._audit(tenant_id, "compliance_issue_opened", issue_id)
		return item.to_dict()

	def record_remediation(self, remediation_id: str, tenant_id: str, issue_id: str, owner_id: str, plan_reference: str, high_impact: bool = False, approval_reference: str = "") -> dict[str, Any]:
		issue = self._tenant_issue_or_none(issue_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_remediation", "issue_present": issue is not None, "owner_present": bool(owner_id), "plan_present": bool(plan_reference), "high_impact": high_impact, "approval_present": bool(approval_reference)})
		item = ComplianceRemediation(remediation_id, tenant_id, issue_id, owner_id, plan_reference, approval_reference, "active")
		self.remediations[remediation_id] = item
		if issue is not None:
			issue.status = "remediated"
		self._audit(tenant_id, "compliance_remediation_recorded", remediation_id)
		return item.to_dict()

	def publish_report(self, report_id: str, tenant_id: str, report_type: str, framework: str, period: str, evidence_reference: str, approver_id: str) -> dict[str, Any]:
		report_type = normalize_code(report_type)
		framework = normalize_code(framework)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "publish_report", "report_type_supported": report_type in SUPPORTED_REPORT_TYPES, "framework_supported": framework in SUPPORTED_REGULATORY_FRAMEWORKS, "period_present": bool(period), "evidence_present": bool(evidence_reference), "approver_present": bool(approver_id)})
		item = ComplianceReport(report_id, tenant_id, report_type, framework, period, evidence_reference, approver_id)
		self.reports[report_id] = item
		self._audit(tenant_id, "compliance_report_published", report_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": bool(reviewer_id), "evidence_present": bool(evidence_reference)})
		item = ComplianceReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = item
		self._audit(tenant_id, "compliance_review_recorded", review_id)
		return item.to_dict()

	def register_compliance_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_compliance_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = ComplianceAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[agent_id] = item
		self._audit(tenant_id, "compliance_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "compliance_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "compliance_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.compliance.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "obligation_count": self._count(self.obligations, tenant_id), "control_count": self._count(self.controls, tenant_id), "check_count": self._count(self.checks, tenant_id), "failed_check_count": sum(1 for item in self.checks.values() if item.tenant_id == tenant_id and check_failed(item.result)), "evidence_count": self._count(self.evidence, tenant_id), "attestation_count": self._count(self.attestations, tenant_id), "issue_count": self._count(self.issues, tenant_id), "open_issue_count": sum(1 for item in self.issues.values() if item.tenant_id == tenant_id and item.status != "closed"), "report_count": self._count(self.reports, tenant_id), "review_count": self._count(self.reviews, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_obligation_or_none(self, item_id: str, tenant_id: str) -> ComplianceObligation | None:
		item = self.obligations.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_control_or_none(self, item_id: str, tenant_id: str) -> ComplianceControl | None:
		item = self.controls.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_issue_or_none(self, item_id: str, tenant_id: str) -> ComplianceIssue | None:
		item = self.issues.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "compliance_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "compliance_policy_denied")


FintechComplianceService = ComplianceAutomationService
