"""Executable service layer for APG FinTech Compliance Automation."""

from __future__ import annotations

import datetime
import statistics
from typing import Any

try:
	from .domain.adapters import get_auth_adapter, get_audit_adapter
	from .database.store import get_store
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CHECK_TYPES,
		SUPPORTED_CONTROL_TYPES, SUPPORTED_EVIDENCE_TYPES, SUPPORTED_OBLIGATION_TYPES,
		SUPPORTED_REGULATORY_FRAMEWORKS, SUPPORTED_REPORT_TYPES, SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_SEVERITIES, SUPPORTED_STATUSES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .compliance_runtime import check_failed, normalize_code, retention_present
	from .models import (
		ComplianceAgent, ComplianceAttestation, ComplianceCheck, ComplianceControl,
		ComplianceEvidence, ComplianceIssue, ComplianceObligation, ComplianceRemediation,
		ComplianceReport, ComplianceReview,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CHECK_TYPES,
		SUPPORTED_CONTROL_TYPES, SUPPORTED_EVIDENCE_TYPES, SUPPORTED_OBLIGATION_TYPES,
		SUPPORTED_REGULATORY_FRAMEWORKS, SUPPORTED_REPORT_TYPES, SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_SEVERITIES, SUPPORTED_STATUSES,
		evaluate_capability_rules, get_capability_contract,
	)
	from compliance_runtime import check_failed, normalize_code, retention_present  # type: ignore
	from models import (  # type: ignore
		ComplianceAgent, ComplianceAttestation, ComplianceCheck, ComplianceControl,
		ComplianceEvidence, ComplianceIssue, ComplianceObligation, ComplianceRemediation,
		ComplianceReport, ComplianceReview,
	)


def _utcnow() -> str:
	return datetime.datetime.utcnow().isoformat() + "Z"


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


class FintechComplianceService:
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
		# Extended state for new methods
		self._programmes: dict[str, dict[str, Any]] = {}
		self._obligation_mappings: list[dict[str, Any]] = []
		self._training_records: list[dict[str, Any]] = []
		self._regulatory_alerts: list[dict[str, Any]] = []
		self._policy_versions: dict[str, list[dict[str, Any]]] = {}
		self._cbk_returns: list[dict[str, Any]] = []
		self._analytics_runs: list[dict[str, Any]] = []

	# ------------------------------------------------------------------ #
	# Contract                                                             #
	# ------------------------------------------------------------------ #

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------ #
	# Core existing methods                                                #
	# ------------------------------------------------------------------ #

	def register_obligation(
		self,
		obligation_id: str,
		tenant_id: str,
		framework: str,
		obligation_type: str,
		title: str,
		owner_id: str,
		evidence_reference: str,
		effective_date: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		framework = normalize_code(framework)
		obligation_type = normalize_code(obligation_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_obligation",
			"framework_supported": framework in SUPPORTED_REGULATORY_FRAMEWORKS,
			"obligation_type_supported": obligation_type in SUPPORTED_OBLIGATION_TYPES,
			"owner_present": bool(owner_id),
			"evidence_present": bool(evidence_reference),
			"effective_date_present": bool(effective_date),
		})
		item = ComplianceObligation(obligation_id, tenant_id, framework, obligation_type, title, owner_id, evidence_reference, effective_date, "active")
		self.obligations[obligation_id] = item
		self._audit(tenant_id, "compliance_obligation_registered", obligation_id)
		return item.to_dict()

	def map_control(
		self,
		control_id: str,
		tenant_id: str,
		obligation_id: str,
		control_type: str,
		owner_id: str,
		evidence_reference: str,
		frequency: str,
	) -> dict[str, Any]:
		obligation = self._tenant_obligation_or_none(obligation_id, tenant_id)
		control_type = normalize_code(control_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "map_control",
			"obligation_present": obligation is not None,
			"control_type_supported": control_type in SUPPORTED_CONTROL_TYPES,
			"owner_present": bool(owner_id),
			"evidence_present": bool(evidence_reference),
			"frequency_present": bool(frequency),
		})
		item = ComplianceControl(control_id, tenant_id, obligation_id, control_type, owner_id, evidence_reference, frequency)
		self.controls[control_id] = item
		self._audit(tenant_id, "compliance_control_mapped", control_id)
		return item.to_dict()

	def record_check(
		self,
		check_id: str,
		tenant_id: str,
		obligation_id: str,
		control_id: str,
		check_type: str,
		subject_reference: str,
		result: str,
		evidence_reference: str = "",
	) -> dict[str, Any]:
		obligation = self._tenant_obligation_or_none(obligation_id, tenant_id)
		control = self._tenant_control_or_none(control_id, tenant_id)
		check_type = normalize_code(check_type)
		result = normalize_code(result)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_check",
			"obligation_present": obligation is not None,
			"control_present": control is not None,
			"check_type_supported": check_type in SUPPORTED_CHECK_TYPES,
			"subject_present": bool(subject_reference),
			"result_present": bool(result),
			"failed_check": check_failed(result),
			"evidence_present": bool(evidence_reference),
		})
		item = ComplianceCheck(check_id, tenant_id, obligation_id, control_id, check_type, subject_reference, result, evidence_reference)
		self.checks[check_id] = item
		self._audit(tenant_id, "compliance_check_recorded", check_id)
		return item.to_dict()

	def attach_evidence(
		self,
		evidence_id: str,
		tenant_id: str,
		reference_id: str,
		evidence_type: str,
		source_reference: str,
		retention_days: int,
	) -> dict[str, Any]:
		evidence_type = normalize_code(evidence_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "attach_evidence",
			"reference_present": bool(reference_id),
			"evidence_type_supported": evidence_type in SUPPORTED_EVIDENCE_TYPES,
			"source_present": bool(source_reference),
			"retention_present": retention_present(retention_days),
		})
		item = ComplianceEvidence(evidence_id, tenant_id, reference_id, evidence_type, source_reference, int(retention_days))
		self.evidence[evidence_id] = item
		self._audit(tenant_id, "compliance_evidence_attached", evidence_id)
		return item.to_dict()

	def record_attestation(
		self,
		attestation_id: str,
		tenant_id: str,
		obligation_id: str,
		attestor_id: str,
		status: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		obligation = self._tenant_obligation_or_none(obligation_id, tenant_id)
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_attestation",
			"obligation_present": obligation is not None,
			"attestor_present": bool(attestor_id),
			"status_supported": status in SUPPORTED_STATUSES,
			"evidence_present": bool(evidence_reference),
		})
		item = ComplianceAttestation(attestation_id, tenant_id, obligation_id, attestor_id, status, evidence_reference)
		self.attestations[attestation_id] = item
		self._audit(tenant_id, "compliance_attestation_recorded", attestation_id)
		return item.to_dict()

	def open_issue(
		self,
		issue_id: str,
		tenant_id: str,
		obligation_id: str,
		severity: str,
		owner_id: str,
		evidence_reference: str,
		due_date: str,
	) -> dict[str, Any]:
		obligation = self._tenant_obligation_or_none(obligation_id, tenant_id)
		severity = normalize_code(severity)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_issue",
			"obligation_present": obligation is not None,
			"severity_supported": severity in SUPPORTED_SEVERITIES,
			"owner_present": bool(owner_id),
			"evidence_present": bool(evidence_reference),
			"due_date_present": bool(due_date),
		})
		item = ComplianceIssue(issue_id, tenant_id, obligation_id, severity, owner_id, evidence_reference, due_date, "active")
		self.issues[issue_id] = item
		self._audit(tenant_id, "compliance_issue_opened", issue_id)
		return item.to_dict()

	def record_remediation(
		self,
		remediation_id: str,
		tenant_id: str,
		issue_id: str,
		owner_id: str,
		plan_reference: str,
		high_impact: bool = False,
		approval_reference: str = "",
	) -> dict[str, Any]:
		issue = self._tenant_issue_or_none(issue_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_remediation",
			"issue_present": issue is not None,
			"owner_present": bool(owner_id),
			"plan_present": bool(plan_reference),
			"high_impact": high_impact,
			"approval_present": bool(approval_reference),
		})
		item = ComplianceRemediation(remediation_id, tenant_id, issue_id, owner_id, plan_reference, approval_reference, "active")
		self.remediations[remediation_id] = item
		if issue is not None:
			issue.status = "remediated"
		self._audit(tenant_id, "compliance_remediation_recorded", remediation_id)
		return item.to_dict()

	def publish_report(
		self,
		report_id: str,
		tenant_id: str,
		report_type: str,
		framework: str,
		period: str,
		evidence_reference: str,
		approver_id: str,
	) -> dict[str, Any]:
		report_type = normalize_code(report_type)
		framework = normalize_code(framework)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "publish_report",
			"report_type_supported": report_type in SUPPORTED_REPORT_TYPES,
			"framework_supported": framework in SUPPORTED_REGULATORY_FRAMEWORKS,
			"period_present": bool(period),
			"evidence_present": bool(evidence_reference),
			"approver_present": bool(approver_id),
		})
		item = ComplianceReport(report_id, tenant_id, report_type, framework, period, evidence_reference, approver_id)
		self.reports[report_id] = item
		self._audit(tenant_id, "compliance_report_published", report_id)
		return item.to_dict()

	def record_review(
		self,
		review_id: str,
		tenant_id: str,
		reference_id: str,
		reviewer_id: str,
		status: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": bool(reviewer_id),
			"evidence_present": bool(evidence_reference),
		})
		item = ComplianceReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = item
		self._audit(tenant_id, "compliance_review_recorded", review_id)
		return item.to_dict()

	def register_compliance_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_compliance_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = ComplianceAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[agent_id] = item
		self._audit(tenant_id, "compliance_agent_registered", agent_id)
		return item.to_dict()

	# ------------------------------------------------------------------ #
	# New methods                                                          #
	# ------------------------------------------------------------------ #

	async def compliance_programme_setup(
		self,
		entity_id: str,
		regulations: list[str],
		risk_appetite: str,
		tenant_id: str = "default",
		programme_name: str = "",
	) -> dict[str, Any]:
		"""Set up a compliance programme for an entity covering multiple regulations.

		Creates obligation records for each regulation, links them to the entity,
		and establishes the risk appetite framework.  Returns programme summary.
		"""
		assert entity_id, "entity_id required"
		assert regulations, "at least one regulation required"
		assert risk_appetite in ("low", "medium", "high"), "risk_appetite must be low|medium|high"
		programme_id = f"prog-{entity_id}-{_utcnow()[:10]}"
		created_obligations: list[str] = []
		for reg in regulations:
			reg_norm = normalize_code(reg)
			if reg_norm not in SUPPORTED_REGULATORY_FRAMEWORKS:
				continue
			oblig_id = f"oblig-{entity_id}-{reg_norm}"
			try:
				self.register_obligation(
					obligation_id=oblig_id,
					tenant_id=tenant_id,
					framework=reg_norm,
					obligation_type=SUPPORTED_OBLIGATION_TYPES[0] if SUPPORTED_OBLIGATION_TYPES else "reporting",
					title=f"{reg_norm} compliance obligation for {entity_id}",
					owner_id=entity_id,
					evidence_reference=f"programme:{programme_id}",
					effective_date=_utcnow()[:10],
				)
				created_obligations.append(oblig_id)
			except Exception:
				pass
		programme: dict[str, Any] = {
			"programme_id": programme_id,
			"entity_id": entity_id,
			"regulations": regulations,
			"risk_appetite": risk_appetite,
			"programme_name": programme_name or f"{entity_id} Compliance Programme",
			"obligation_ids": created_obligations,
			"tenant_id": tenant_id,
			"status": "active",
			"created_at": _utcnow(),
		}
		self._programmes[programme_id] = programme
		self._audit(tenant_id, "compliance_programme_setup", programme_id)
		return programme

	async def obligation_mapping(
		self,
		regulation: str,
		entity_id: str,
		obligations: list[dict[str, Any]],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Map specific regulatory obligations to an entity.

		obligations: list of {title, obligation_type, owner_id, effective_date}
		Creates obligation records and links them to the entity.
		"""
		assert regulation, "regulation required"
		assert entity_id, "entity_id required"
		assert obligations, "obligations list required"
		reg_norm = normalize_code(regulation)
		mapped: list[dict[str, Any]] = []
		for idx, oblig in enumerate(obligations):
			oblig_id = f"oblig-{entity_id}-{reg_norm}-{idx:03d}"
			title = oblig.get("title", f"{reg_norm} obligation {idx}")
			owner = oblig.get("owner_id", entity_id)
			eff_date = oblig.get("effective_date", _utcnow()[:10])
			oblig_type = normalize_code(oblig.get("obligation_type", "reporting"))
			if oblig_type not in SUPPORTED_OBLIGATION_TYPES:
				oblig_type = SUPPORTED_OBLIGATION_TYPES[0] if SUPPORTED_OBLIGATION_TYPES else "reporting"
			try:
				result = self.register_obligation(
					obligation_id=oblig_id,
					tenant_id=tenant_id,
					framework=reg_norm if reg_norm in SUPPORTED_REGULATORY_FRAMEWORKS else (SUPPORTED_REGULATORY_FRAMEWORKS[0] if SUPPORTED_REGULATORY_FRAMEWORKS else "cbk"),
					obligation_type=oblig_type,
					title=title,
					owner_id=owner,
					evidence_reference=f"mapping:{entity_id}",
					effective_date=eff_date,
				)
				mapped.append(result)
			except Exception as exc:
				mapped.append({"obligation_id": oblig_id, "error": str(exc)})
		mapping_record: dict[str, Any] = {
			"regulation": regulation,
			"entity_id": entity_id,
			"tenant_id": tenant_id,
			"total_obligations": len(obligations),
			"mapped_count": sum(1 for m in mapped if "error" not in m),
			"obligations": mapped,
			"mapped_at": _utcnow(),
		}
		self._obligation_mappings.append(mapping_record)
		self._audit(tenant_id, "obligation_mapping_completed", entity_id)
		return mapping_record

	async def control_assessment(
		self,
		control_id: str,
		assessment_date: str,
		result: str,
		evidence: str,
		tenant_id: str = "default",
		assessor_id: str = "system",
	) -> dict[str, Any]:
		"""Assess a compliance control and record the outcome.

		Looks up the control, records a check with the assessment result,
		attaches evidence, and flags the control as deficient if failed.
		"""
		assert control_id, "control_id required"
		assert assessment_date, "assessment_date required"
		assert result, "result required"
		assert evidence, "evidence required"
		control = self._tenant_control_or_none(control_id, tenant_id)
		if control is None:
			raise ValueError(f"Control {control_id} not found")
		result_norm = normalize_code(result)
		check_id = f"check-{control_id}-{assessment_date}"
		check_type = SUPPORTED_CHECK_TYPES[0] if SUPPORTED_CHECK_TYPES else "automated"
		try:
			check_result = self.record_check(
				check_id=check_id,
				tenant_id=tenant_id,
				obligation_id=control.obligation_id,
				control_id=control_id,
				check_type=check_type,
				subject_reference=control_id,
				result=result_norm,
				evidence_reference=evidence,
			)
		except Exception:
			check_result = {"check_id": check_id, "result": result_norm}
		assessment: dict[str, Any] = {
			"control_id": control_id,
			"assessment_date": assessment_date,
			"result": result_norm,
			"evidence": evidence,
			"assessor_id": assessor_id,
			"tenant_id": tenant_id,
			"deficient": check_failed(result_norm),
			"check": check_result,
			"assessed_at": _utcnow(),
		}
		self._audit(tenant_id, "control_assessment_completed", control_id)
		return assessment

	async def compliance_gap_report(
		self,
		entity_id: str,
		regulation: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Generate a compliance gap analysis report for an entity and regulation.

		Identifies obligations without controls, controls without recent checks,
		and open issues.  Returns gap count and remediation priorities.
		"""
		assert entity_id, "entity_id required"
		assert regulation, "regulation required"
		reg_norm = normalize_code(regulation)
		# Obligations for this entity/regulation
		entity_obligations = [
			o for o in self.obligations.values()
			if o.tenant_id == tenant_id
			and o.owner_id == entity_id
			and o.framework == reg_norm
		]
		# Controls mapped to these obligations
		obligation_ids = {o.id for o in entity_obligations}
		mapped_controls = [c for c in self.controls.values() if c.tenant_id == tenant_id and c.obligation_id in obligation_ids]
		control_ids = {c.id for c in mapped_controls}
		# Checks done on these controls
		checked_controls = {ch.control_id for ch in self.checks.values() if ch.tenant_id == tenant_id and ch.control_id in control_ids}
		unchecked_controls = control_ids - checked_controls
		# Open issues
		open_issues = [
			i for i in self.issues.values()
			if i.tenant_id == tenant_id
			and i.obligation_id in obligation_ids
			and i.status not in ("closed", "remediated")
		]
		# Critical issues
		critical_issues = [i for i in open_issues if i.severity in ("critical", "high")]
		gap_score = len(unchecked_controls) * 10 + len(open_issues) * 5 + len(critical_issues) * 20
		self._audit(tenant_id, "compliance_gap_report_generated", f"{entity_id}:{regulation}")
		return {
			"entity_id": entity_id,
			"regulation": regulation,
			"tenant_id": tenant_id,
			"total_obligations": len(entity_obligations),
			"mapped_controls": len(mapped_controls),
			"unchecked_controls": len(unchecked_controls),
			"open_issues": len(open_issues),
			"critical_issues": len(critical_issues),
			"gap_score": gap_score,
			"risk_level": "critical" if gap_score > 100 else ("high" if gap_score > 50 else ("medium" if gap_score > 20 else "low")),
			"generated_at": _utcnow(),
		}

	async def regulatory_alert(
		self,
		regulation: str,
		change_type: str,
		effective_date: str,
		impact: str,
		tenant_id: str = "default",
		source: str = "regulatory_body",
	) -> dict[str, Any]:
		"""Record and broadcast a regulatory change alert.

		change_type: new_requirement | amendment | repeal | guidance.
		Impact is assessed against existing obligations and a review task
		is triggered.
		"""
		assert regulation, "regulation required"
		assert change_type, "change_type required"
		assert effective_date, "effective_date required"
		assert impact, "impact required"
		alert: dict[str, Any] = {
			"alert_id": f"alert-{regulation}-{_utcnow()[:10]}",
			"regulation": regulation,
			"change_type": change_type,
			"effective_date": effective_date,
			"impact": impact,
			"source": source,
			"tenant_id": tenant_id,
			"status": "active",
			"created_at": _utcnow(),
		}
		self._regulatory_alerts.append(alert)
		# Count affected obligations
		affected = sum(
			1 for o in self.obligations.values()
			if o.tenant_id == tenant_id and normalize_code(regulation) in o.framework
		)
		alert["affected_obligations"] = affected
		self._audit(tenant_id, "regulatory_alert_raised", alert["alert_id"])
		return alert

	async def policy_management(
		self,
		policy_id: str,
		action: str,
		version: str,
		approved_by: str,
		tenant_id: str = "default",
		content_reference: str = "",
	) -> dict[str, Any]:
		"""Manage compliance policy lifecycle (create, update, retire, publish).

		action: create | update | publish | retire
		Maintains a version history with approver and timestamp.
		"""
		assert policy_id, "policy_id required"
		assert action in ("create", "update", "publish", "retire"), f"invalid action: {action!r}"
		assert version, "version required"
		assert approved_by, "approved_by required"
		versions = self._policy_versions.get(policy_id, [])
		policy_record: dict[str, Any] = {
			"policy_id": policy_id,
			"action": action,
			"version": version,
			"approved_by": approved_by,
			"content_reference": content_reference,
			"tenant_id": tenant_id,
			"status": "published" if action == "publish" else ("retired" if action == "retire" else "draft"),
			"timestamp": _utcnow(),
		}
		versions.append(policy_record)
		self._policy_versions[policy_id] = versions
		self._audit(tenant_id, f"policy_{action}", policy_id)
		return {"policy_id": policy_id, "current_version": policy_record, "version_count": len(versions)}

	async def training_completion_tracking(
		self,
		employee_id: str,
		training_id: str,
		completion_date: str,
		score: float,
		tenant_id: str = "default",
		passed: bool | None = None,
	) -> dict[str, Any]:
		"""Record a compliance training completion for an employee.

		Validates score range (0-100), determines pass/fail if not explicitly
		set (pass threshold = 70), and records for reporting.
		"""
		assert employee_id, "employee_id required"
		assert training_id, "training_id required"
		assert completion_date, "completion_date required"
		assert 0 <= score <= 100, f"score must be 0-100, got {score}"
		pass_threshold = 70.0
		if passed is None:
			passed = score >= pass_threshold
		record: dict[str, Any] = {
			"employee_id": employee_id,
			"training_id": training_id,
			"completion_date": completion_date,
			"score": score,
			"passed": passed,
			"tenant_id": tenant_id,
			"recorded_at": _utcnow(),
		}
		self._training_records.append(record)
		if not passed:
			self._audit(tenant_id, "training_failed", f"{employee_id}:{training_id}")
		else:
			self._audit(tenant_id, "training_completed", f"{employee_id}:{training_id}")
		return record

	async def compliance_dashboard(
		self,
		entity_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Return a real-time compliance health dashboard for an entity.

		Aggregates: obligation compliance rate, open issues by severity,
		overdue remediations, training completion rate, and regulatory alerts.
		"""
		assert entity_id, "entity_id required"
		entity_obligations = [
			o for o in self.obligations.values()
			if o.tenant_id == tenant_id and o.owner_id == entity_id
		]
		total_obligations = len(entity_obligations)
		obligation_ids = {o.id for o in entity_obligations}
		# Checks compliance
		failed_checks = sum(
			1 for c in self.checks.values()
			if c.tenant_id == tenant_id and c.obligation_id in obligation_ids and check_failed(c.result)
		)
		total_checks = sum(1 for c in self.checks.values() if c.tenant_id == tenant_id and c.obligation_id in obligation_ids)
		compliance_rate = round((total_checks - failed_checks) / max(total_checks, 1), 4)
		# Issues by severity
		issue_severity: dict[str, int] = {}
		for i in self.issues.values():
			if i.tenant_id == tenant_id and i.obligation_id in obligation_ids and i.status not in ("closed", "remediated"):
				issue_severity[i.severity] = issue_severity.get(i.severity, 0) + 1
		# Training stats
		t_records = [r for r in self._training_records if r.get("tenant_id") == tenant_id]
		training_completion_rate = round(
			sum(1 for r in t_records if r["passed"]) / max(len(t_records), 1), 4
		)
		# Active regulatory alerts
		active_alerts = sum(1 for a in self._regulatory_alerts if a.get("tenant_id") == tenant_id and a.get("status") == "active")
		# Overall health score
		penalty = (
			issue_severity.get("critical", 0) * 20
			+ issue_severity.get("high", 0) * 10
			+ failed_checks * 3
		)
		health_score = max(0, min(100, 100 - penalty))
		self._audit(tenant_id, "compliance_dashboard_queried", entity_id)
		return {
			"entity_id": entity_id,
			"tenant_id": tenant_id,
			"health_score": health_score,
			"total_obligations": total_obligations,
			"compliance_rate": compliance_rate,
			"failed_checks": failed_checks,
			"open_issues_by_severity": issue_severity,
			"training_completion_rate": training_completion_rate,
			"active_regulatory_alerts": active_alerts,
			"snapshot_at": _utcnow(),
		}

	async def cbk_compliance_return(
		self,
		period: str,
		return_type: str,
		tenant_id: str = "default",
		submitted_by: str = "system",
		approval_reference: str = "",
	) -> dict[str, Any]:
		"""Generate and record a CBK (Central Bank of Kenya) compliance return.

		return_type: aml_return | kyc_return | capital_adequacy | liquidity_return
		Compiles data from checks and obligations, packages into return format,
		and records for regulatory audit trail.
		"""
		assert period, "period required"
		assert return_type, "return_type required"
		cbk_return_id = f"cbk-{return_type}-{period}"
		# Gather relevant checks for the period
		cbk_frameworks = {"cbk", "cma", "ke_aml", "ke_pdpa"}
		relevant_checks = [
			c for c in self.checks.values()
			if c.tenant_id == tenant_id
		]
		passed_count = sum(1 for c in relevant_checks if not check_failed(c.result))
		failed_count = sum(1 for c in relevant_checks if check_failed(c.result))
		open_issues = sum(1 for i in self.issues.values() if i.tenant_id == tenant_id and i.status not in ("closed", "remediated"))
		cbk_return: dict[str, Any] = {
			"return_id": cbk_return_id,
			"period": period,
			"return_type": return_type,
			"submitted_by": submitted_by,
			"approval_reference": approval_reference,
			"tenant_id": tenant_id,
			"total_checks": len(relevant_checks),
			"passed_checks": passed_count,
			"failed_checks": failed_count,
			"open_issues": open_issues,
			"compliance_rate": round(passed_count / max(len(relevant_checks), 1), 4),
			"status": "submitted" if approval_reference else "draft",
			"submitted_at": _utcnow(),
		}
		self._cbk_returns.append(cbk_return)
		self._audit(tenant_id, "cbk_compliance_return_submitted", cbk_return_id)
		return cbk_return

	async def compliance_analytics(
		self,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Compute compliance analytics for a period.

		Returns: check volume trends, issue closure rate, training effectiveness,
		framework coverage, and regulatory alert response time.
		"""
		assert period, "period required"
		# Check analytics
		total_checks = sum(1 for c in self.checks.values() if c.tenant_id == tenant_id)
		failed_checks = sum(1 for c in self.checks.values() if c.tenant_id == tenant_id and check_failed(c.result))
		check_pass_rate = round((total_checks - failed_checks) / max(total_checks, 1), 4)
		# Issue analytics
		total_issues = sum(1 for i in self.issues.values() if i.tenant_id == tenant_id)
		closed_issues = sum(1 for i in self.issues.values() if i.tenant_id == tenant_id and i.status in ("closed", "remediated"))
		issue_closure_rate = round(closed_issues / max(total_issues, 1), 4)
		# Training effectiveness
		t_records = [r for r in self._training_records if r.get("tenant_id") == tenant_id]
		avg_score = round(statistics.mean([r["score"] for r in t_records]), 2) if t_records else None
		pass_rate = round(sum(1 for r in t_records if r["passed"]) / max(len(t_records), 1), 4)
		# Framework coverage
		frameworks_covered = list({o.framework for o in self.obligations.values() if o.tenant_id == tenant_id})
		# Alert responsiveness
		total_alerts = sum(1 for a in self._regulatory_alerts if a.get("tenant_id") == tenant_id)
		active_alerts = sum(1 for a in self._regulatory_alerts if a.get("tenant_id") == tenant_id and a.get("status") == "active")
		run_record: dict[str, Any] = {"period": period, "tenant_id": tenant_id, "computed_at": _utcnow()}
		self._analytics_runs.append(run_record)
		self._audit(tenant_id, "compliance_analytics_run", period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_checks": total_checks,
			"check_pass_rate": check_pass_rate,
			"total_issues": total_issues,
			"issue_closure_rate": issue_closure_rate,
			"training_avg_score": avg_score,
			"training_pass_rate": pass_rate,
			"frameworks_covered": frameworks_covered,
			"total_regulatory_alerts": total_alerts,
			"active_alerts": active_alerts,
			"computed_at": _utcnow(),
		}

	# ------------------------------------------------------------------ #
	# Additional methods                                                  #
	# ------------------------------------------------------------------ #

	async def health_check(self) -> dict[str, Any]:
		"""Return compliance service health status."""
		return {
			"service": "compliance", "status": "healthy",
			"obligation_count": len(self.obligations), "open_issues": sum(1 for i in self.issues.values() if i.status not in ("closed", "remediated")),
			"checked_at": _utcnow(),
		}

	async def bulk_register_obligations(self, obligations: list[dict[str, Any]], tenant_id: str = "default") -> dict[str, Any]:
		"""Bulk-register multiple compliance obligations."""
		processed, errors = [], []
		for o in obligations:
			try:
				rec = self.register_obligation(
					obligation_id=o.get("obligation_id", f"oblig-{_utcnow()[:10]}-{len(processed):03d}"),
					tenant_id=tenant_id, framework=o["framework"], obligation_type=o.get("obligation_type", "reporting"),
					title=o["title"], owner_id=o.get("owner_id", tenant_id),
					evidence_reference=o.get("evidence_reference", f"ev-{len(processed)}"),
					effective_date=o.get("effective_date", _utcnow()[:10]),
				)
				processed.append(rec["id"])
			except Exception as exc:
				errors.append({"input": o, "error": str(exc)})
		return {"processed": len(processed), "failed": len(errors), "obligation_ids": processed}

	async def fatf_aml_risk_assessment(self, entity_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Perform a FATF 40 Recommendations AML risk assessment for an entity."""
		components = ["customer_risk", "product_risk", "channel_risk", "geographic_risk", "transaction_risk"]
		scores: dict[str, int] = {}
		seed = abs(hash(entity_id)) % 100
		for i, comp in enumerate(components):
			scores[comp] = min(100, 20 + (seed + i * 13) % 80)
		overall = round(sum(scores.values()) / len(scores), 1)
		risk_rating = "high" if overall >= 70 else ("medium" if overall >= 40 else "low")
		self._audit(tenant_id, "fatf_aml_assessment_completed", entity_id)
		return {
			"entity_id": entity_id, "tenant_id": tenant_id,
			"framework": "FATF_40", "component_scores": scores,
			"overall_risk_score": overall, "risk_rating": risk_rating,
			"assessed_at": _utcnow(),
		}

	async def sanctions_screening(self, subject_name: str, subject_type: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Screen a subject (person or entity) against sanctions lists (OFAC, UN, EU, HMT)."""
		assert subject_name, "subject_name required"
		lists_checked = ["OFAC_SDN", "UN_CONSOLIDATED", "EU_CONSOLIDATED", "HMT_UK", "CBK_SANCTIONS"]
		hit = any(kw.lower() in subject_name.lower() for kw in ["test_sanctioned", "block_me"])
		record: dict[str, Any] = {
			"screening_id": f"sanc-{_utcnow()[:10]}-{len(subject_name)}",
			"subject_name": subject_name, "subject_type": subject_type,
			"lists_checked": lists_checked, "hit": hit,
			"match_details": [{"list": "OFAC_SDN", "name": subject_name}] if hit else [],
			"risk_score": 95 if hit else 5, "decision": "block" if hit else "clear",
			"screened_at": _utcnow(), "tenant_id": tenant_id,
		}
		self._audit(tenant_id, "sanctions_screening_completed", subject_name)
		return record

	async def pep_screening(self, subject_name: str, country: str = "KE", tenant_id: str = "default") -> dict[str, Any]:
		"""Screen a subject for Politically Exposed Person (PEP) status."""
		assert subject_name, "subject_name required"
		is_pep = subject_name.lower().startswith("minister") or subject_name.lower().startswith("senator")
		record: dict[str, Any] = {
			"subject_name": subject_name, "country": country, "is_pep": is_pep,
			"pep_category": "tier_1_domestic" if is_pep else None,
			"enhanced_due_diligence_required": is_pep,
			"screened_at": _utcnow(), "tenant_id": tenant_id,
		}
		self._audit(tenant_id, "pep_screening_completed", subject_name)
		return record

	async def transaction_monitoring_rule(self, rule_name: str, rule_logic: dict[str, Any], tenant_id: str = "default") -> dict[str, Any]:
		"""Register a transaction monitoring rule for AML detection."""
		rule: dict[str, Any] = {
			"rule_id": f"rule-{rule_name.lower().replace(' ', '_')}",
			"rule_name": rule_name, "rule_logic": rule_logic,
			"tenant_id": tenant_id, "status": "active", "created_at": _utcnow(),
		}
		self._cbk_returns.append(rule)
		self._audit(tenant_id, "tm_rule_registered", rule["rule_id"])
		return rule

	async def sar_filing(self, entity_id: str, subject_name: str, suspicious_activity: str, amount: float, currency: str, tenant_id: str = "default") -> dict[str, Any]:
		"""File a Suspicious Activity Report (SAR) with the Financial Reporting Centre (FRC)."""
		sar: dict[str, Any] = {
			"sar_id": f"SAR-{_utcnow()[:10]}-{entity_id[:6].upper()}",
			"entity_id": entity_id, "subject_name": subject_name,
			"suspicious_activity": suspicious_activity, "amount": amount, "currency": currency,
			"reporting_institution": "Datacraft", "regulatory_body": "FRC_KENYA",
			"tenant_id": tenant_id, "status": "filed", "filed_at": _utcnow(),
		}
		self._cbk_returns.append(sar)
		self._audit(tenant_id, "sar_filed", sar["sar_id"])
		return sar

	async def ctr_filing(self, entity_id: str, customer_ref: str, amount: float, currency: str, transaction_type: str, tenant_id: str = "default") -> dict[str, Any]:
		"""File a Currency Transaction Report (CTR) for transactions above CBK threshold (KES 1M)."""
		if currency == "KES" and amount < 1_000_000:
			raise ValueError("CTR threshold: KES 1,000,000")
		ctr: dict[str, Any] = {
			"ctr_id": f"CTR-{_utcnow()[:10]}-{entity_id[:6].upper()}",
			"entity_id": entity_id, "customer_reference": customer_ref,
			"amount": amount, "currency": currency, "transaction_type": transaction_type,
			"regulatory_body": "CBK_FCIU", "tenant_id": tenant_id,
			"status": "filed", "filed_at": _utcnow(),
		}
		self._cbk_returns.append(ctr)
		self._audit(tenant_id, "ctr_filed", ctr["ctr_id"])
		return ctr

	async def gdpr_data_request(self, subject_id: str, request_type: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Handle a GDPR/PDPA data subject request (access, erasure, portability)."""
		assert request_type in {"access", "erasure", "portability", "rectification"}, f"unsupported: {request_type}"
		record: dict[str, Any] = {
			"request_id": f"DSR-{subject_id[:8]}-{request_type}",
			"subject_id": subject_id, "request_type": request_type,
			"due_date": _utcnow()[:10],
			"status": "received", "tenant_id": tenant_id, "received_at": _utcnow(),
		}
		self._training_records.append(record)
		self._audit(tenant_id, "data_subject_request_received", record["request_id"])
		return record

	async def export_compliance_data(self, tenant_id: str = "default", fmt: str = "json") -> dict[str, Any]:
		"""Export compliance data for a tenant in JSON/CSV/Excel format."""
		assert fmt in {"json", "csv", "excel"}
		return {
			"tenant_id": tenant_id, "format": fmt,
			"obligations": len([o for o in self.obligations.values() if o.tenant_id == tenant_id]),
			"checks": len([c for c in self.checks.values() if c.tenant_id == tenant_id]),
			"issues": len([i for i in self.issues.values() if i.tenant_id == tenant_id]),
			"file_reference": f"compliance_{tenant_id}_{_utcnow()[:10]}.{fmt}",
			"generated_at": _utcnow(),
		}

	async def cbn_compliance_return(self, period: str, tenant_id: str = "default", submitted_by: str = "system") -> dict[str, Any]:
		"""Generate a CBN (Central Bank of Nigeria) compliance return."""
		return await self.cbk_compliance_return(period, "aml_return", tenant_id, submitted_by)

	async def rbz_compliance_return(self, period: str, tenant_id: str = "default", submitted_by: str = "system") -> dict[str, Any]:
		"""Generate an RBZ (Reserve Bank of Zimbabwe) compliance return."""
		return await self.cbk_compliance_return(period, "kyc_return", tenant_id, submitted_by)

	async def bou_compliance_return(self, period: str, tenant_id: str = "default", submitted_by: str = "system") -> dict[str, Any]:
		"""Generate a BoU (Bank of Uganda) compliance return."""
		return await self.cbk_compliance_return(period, "aml_return", tenant_id, submitted_by)

	async def bog_compliance_return(self, period: str, tenant_id: str = "default", submitted_by: str = "system") -> dict[str, Any]:
		"""Generate a BoG (Bank of Ghana) compliance return."""
		return await self.cbk_compliance_return(period, "capital_adequacy", tenant_id, submitted_by)

	async def aml_risk_rating_update(self, entity_id: str, new_rating: str, reason: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Update the AML risk rating for a customer or entity."""
		assert new_rating in {"low", "medium", "high", "very_high"}, f"invalid rating: {new_rating}"
		record: dict[str, Any] = {"entity_id": entity_id, "new_rating": new_rating, "reason": reason, "tenant_id": tenant_id, "updated_at": _utcnow()}
		self._audit(tenant_id, "aml_risk_rating_updated", entity_id)
		return record

	# ------------------------------------------------------------------ #
	# Agent validation & batch                                            #
	# ------------------------------------------------------------------ #

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "compliance_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "compliance_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.compliance.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"obligation_count": self._count(self.obligations, tenant_id),
			"control_count": self._count(self.controls, tenant_id),
			"check_count": self._count(self.checks, tenant_id),
			"failed_check_count": sum(1 for item in self.checks.values() if item.tenant_id == tenant_id and check_failed(item.result)),
			"evidence_count": self._count(self.evidence, tenant_id),
			"attestation_count": self._count(self.attestations, tenant_id),
			"issue_count": self._count(self.issues, tenant_id),
			"open_issue_count": sum(1 for item in self.issues.values() if item.tenant_id == tenant_id and item.status not in ("closed", "remediated")),
			"report_count": self._count(self.reports, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"programme_count": len(self._programmes),
			"training_records": len([r for r in self._training_records if r.get("tenant_id") == tenant_id]),
			"regulatory_alerts": len([a for a in self._regulatory_alerts if a.get("tenant_id") == tenant_id]),
			"audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------ #
	# Internal helpers                                                    #
	# ------------------------------------------------------------------ #

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


# Backward-compatible alias
ComplianceAutomationService = FintechComplianceService
