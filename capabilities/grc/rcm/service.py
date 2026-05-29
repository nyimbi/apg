"""Executable RCM service facade for APG capability composition."""

from __future__ import annotations

from itertools import count
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	GRCComplianceStatus,
	GRCControlType,
	GRCGovernanceDecisionType,
	GRCRiskStatus,
	RCMAuditEvent,
	RCMComplianceObligation,
	RCMControl,
	RCMControlAssessment,
	RCMEvidence,
	RCMGovernanceDecision,
	RCMRisk,
)


class GrcRcmService:
	"""Tenant-aware governance, risk, compliance, and control runtime."""

	def __init__(self) -> None:
		self._risks: dict[str, RCMRisk] = {}
		self._controls: dict[str, RCMControl] = {}
		self._obligations: dict[str, RCMComplianceObligation] = {}
		self._assessments: dict[str, RCMControlAssessment] = {}
		self._decisions: dict[str, RCMGovernanceDecision] = {}
		self._evidence: dict[str, RCMEvidence] = {}
		self._audit_events: dict[str, RCMAuditEvent] = {}
		self._audit_counter = count(1)

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_risk(
		self,
		risk_id: str,
		tenant_id: str,
		title: str,
		category: str,
		owner_id: str,
		probability: float,
		impact: float,
		control_effectiveness: float = 0.0,
		tags: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
		review_recorded: bool = True,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		self._require_text(owner_id, "risk_owner_required")
		self._require_unit_interval(probability, "risk_probability_out_of_range")
		self._require_unit_interval(impact, "risk_impact_out_of_range")
		self._require_unit_interval(control_effectiveness, "control_effectiveness_out_of_range")

		preview = RCMRisk(
			id=risk_id,
			tenant_id=tenant_id,
			title=title,
			category=category,
			owner_id=owner_id,
			probability=probability,
			impact=impact,
			control_effectiveness=control_effectiveness,
			tags=list(tags or []),
			metadata=dict(metadata or {}),
		)
		if preview.level.value in {"critical", "high"} and not review_recorded:
			raise PermissionError("high_risk_review_required")
		self._enforce_write_policy(
			tenant_id,
			risk_level=preview.level.value,
			review_recorded=review_recorded,
			policy_attached=policy_attached,
		)
		self._risks[risk_id] = preview
		self._audit(tenant_id, "risk_registered", risk_id, {"risk_level": preview.level.value})
		return preview.to_dict()

	def register_control(
		self,
		control_id: str,
		tenant_id: str,
		name: str,
		owner_id: str,
		control_type: str = GRCControlType.PREVENTIVE.value,
		mapped_risk_ids: list[str] | None = None,
		effectiveness: float = 0.0,
		test_frequency_days: int = 90,
		metadata: dict[str, Any] | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		self._enforce_write_policy(tenant_id, policy_attached=policy_attached)
		self._require_text(owner_id, "control_owner_required")
		self._require_unit_interval(effectiveness, "control_effectiveness_out_of_range")
		if test_frequency_days <= 0:
			raise ValueError("control_test_frequency_required")
		risk_ids = list(mapped_risk_ids or [])
		self._require_same_tenant(self._risks, risk_ids, tenant_id, "mapped_risk_missing")

		control = RCMControl(
			id=control_id,
			tenant_id=tenant_id,
			name=name,
			owner_id=owner_id,
			control_type=GRCControlType(control_type),
			mapped_risk_ids=risk_ids,
			effectiveness=effectiveness,
			test_frequency_days=test_frequency_days,
			metadata=dict(metadata or {}),
		)
		self._controls[control_id] = control
		self._audit(tenant_id, "control_registered", control_id, {"mapped_risk_count": len(risk_ids)})
		return control.to_dict()

	def add_compliance_obligation(
		self,
		obligation_id: str,
		tenant_id: str,
		framework: str,
		requirement: str,
		owner_id: str,
		jurisdiction: str,
		due_date: str,
		mapped_control_ids: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		self._enforce_write_policy(tenant_id, policy_attached=policy_attached)
		self._require_text(owner_id, "obligation_owner_required")
		self._require_text(framework, "compliance_framework_required")
		control_ids = list(mapped_control_ids or [])
		self._require_same_tenant(self._controls, control_ids, tenant_id, "mapped_control_missing")

		obligation = RCMComplianceObligation(
			id=obligation_id,
			tenant_id=tenant_id,
			framework=framework,
			requirement=requirement,
			owner_id=owner_id,
			jurisdiction=jurisdiction,
			due_date=due_date,
			mapped_control_ids=control_ids,
			metadata=dict(metadata or {}),
		)
		self._obligations[obligation_id] = obligation
		self._audit(tenant_id, "obligation_added", obligation_id, {"framework": framework})
		return obligation.to_dict()

	def assess_control(
		self,
		assessment_id: str,
		tenant_id: str,
		control_id: str,
		assessor_id: str,
		design_effective: bool,
		operating_effective: bool,
		evidence_refs: list[str] | None = None,
		findings: list[str] | None = None,
		review_recorded: bool = True,
	) -> dict[str, Any]:
		self._enforce_write_policy(tenant_id, risk_level="high" if findings else "low", review_recorded=review_recorded)
		self._require_text(assessor_id, "assessor_required")
		self._require_same_tenant(self._controls, [control_id], tenant_id, "control_missing")
		if (not design_effective or not operating_effective) and not evidence_refs:
			raise PermissionError("failed_control_requires_evidence")

		assessment = RCMControlAssessment(
			id=assessment_id,
			tenant_id=tenant_id,
			control_id=control_id,
			assessor_id=assessor_id,
			design_effective=design_effective,
			operating_effective=operating_effective,
			evidence_refs=list(evidence_refs or []),
			findings=list(findings or []),
		)
		self._assessments[assessment_id] = assessment
		self._controls[control_id].last_test_status = assessment.status
		self._audit(tenant_id, "control_assessed", assessment_id, {"status": assessment.status.value})
		return assessment.to_dict()

	def collect_evidence(
		self,
		evidence_id: str,
		tenant_id: str,
		source: str,
		linked_control_id: str | None = None,
		linked_obligation_id: str | None = None,
		encrypted: bool = True,
		retention_days: int = 2555,
		metadata: dict[str, Any] | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		self._enforce_write_policy(tenant_id, policy_attached=policy_attached)
		self._require_text(source, "evidence_source_required")
		if linked_control_id:
			self._require_same_tenant(self._controls, [linked_control_id], tenant_id, "linked_control_missing")
		if linked_obligation_id:
			self._require_same_tenant(self._obligations, [linked_obligation_id], tenant_id, "linked_obligation_missing")
		if not encrypted:
			raise PermissionError("evidence_encryption_required")
		if retention_days < 365:
			raise PermissionError("evidence_retention_too_short")

		evidence = RCMEvidence(
			id=evidence_id,
			tenant_id=tenant_id,
			source=source,
			linked_control_id=linked_control_id,
			linked_obligation_id=linked_obligation_id,
			encrypted=encrypted,
			retention_days=retention_days,
			metadata=dict(metadata or {}),
		)
		self._evidence[evidence_id] = evidence
		self._audit(tenant_id, "evidence_collected", evidence_id, {"source": source})
		return evidence.to_dict()

	def record_governance_decision(
		self,
		decision_id: str,
		tenant_id: str,
		title: str,
		decision_type: str,
		approver_id: str,
		related_risk_ids: list[str] | None = None,
		rationale: str = "",
		approved: bool = True,
		review_recorded: bool = True,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		self._enforce_write_policy(tenant_id, policy_attached=policy_attached)
		self._require_text(approver_id, "approver_required")
		risk_ids = list(related_risk_ids or [])
		self._require_same_tenant(self._risks, risk_ids, tenant_id, "decision_risk_missing")
		highest_risk = self._highest_risk_level(risk_ids)
		if highest_risk in {"critical", "high"} and (not review_recorded or not rationale):
			raise PermissionError("high_risk_governance_decision_requires_rationale")

		decision = RCMGovernanceDecision(
			id=decision_id,
			tenant_id=tenant_id,
			title=title,
			decision_type=GRCGovernanceDecisionType(decision_type),
			approver_id=approver_id,
			related_risk_ids=risk_ids,
			rationale=rationale,
			approved=approved,
		)
		self._decisions[decision_id] = decision
		self._audit(tenant_id, "governance_decision_recorded", decision_id, {"approved": approved})
		return decision.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper used by generated package tests and API shims."""
		data = dict(metadata or {})
		return self.register_risk(
			risk_id=record_id,
			tenant_id=tenant_id,
			title=str(data.get("title") or data.get("name") or record_id),
			category=str(data.get("category") or "operational"),
			owner_id=str(data.get("owner_id") or "system"),
			probability=float(data.get("probability", 0.2)),
			impact=float(data.get("impact", 0.2)),
			control_effectiveness=float(data.get("control_effectiveness", 0.0)),
			metadata={"compatibility_status": status, **data},
			review_recorded=bool(data.get("review_recorded", True)),
			policy_attached=bool(data.get("policy_attached", True)),
		)

	def list_risks(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._risks, tenant_id)

	def list_controls(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._controls, tenant_id)

	def list_obligations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._obligations, tenant_id)

	def list_assessments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._assessments, tenant_id)

	def list_decisions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._decisions, tenant_id)

	def list_evidence(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._evidence, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		records: list[dict[str, Any]] = []
		for store in (
			self._risks,
			self._controls,
			self._obligations,
			self._assessments,
			self._decisions,
			self._evidence,
		):
			records.extend(self._list(store, tenant_id))
		return sorted(records, key=lambda item: (item["kind"], item["id"]))

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		risks = self.list_risks(tenant_id)
		controls = self.list_controls(tenant_id)
		obligations = self.list_obligations(tenant_id)
		assessments = self.list_assessments(tenant_id)
		decisions = self.list_decisions(tenant_id)
		evidence = self.list_evidence(tenant_id)
		high_risks = [risk for risk in risks if risk["risk_level"] in {"high", "critical"}]
		failed_assessments = [
			item for item in assessments
			if item["status"] in {GRCComplianceStatus.NON_COMPLIANT.value, GRCComplianceStatus.PARTIALLY_COMPLIANT.value}
		]
		return {
			"tenant_id": tenant_id,
			"risk_count": len(risks),
			"high_risk_count": len(high_risks),
			"control_count": len(controls),
			"obligation_count": len(obligations),
			"assessment_count": len(assessments),
			"failed_assessment_count": len(failed_assessments),
			"governance_decision_count": len(decisions),
			"evidence_count": len(evidence),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"overall_status": "attention_required" if high_risks or failed_assessments else "operating",
		}

	def _enforce_write_policy(
		self,
		tenant_id: str,
		risk_level: str = "low",
		review_recorded: bool = True,
		policy_attached: bool = True,
	) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"risk_level": risk_level,
			"review_recorded": review_recorded,
		})
		if result["decision"] != "allow":
			reasons = ", ".join(action.get("reason", "capability_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "capability_policy_blocked")

	def _audit(self, tenant_id: str, action: str, subject_id: str, details: dict[str, Any] | None = None) -> None:
		event_id = f"audit-{next(self._audit_counter):06d}"
		self._audit_events[event_id] = RCMAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			action=action,
			subject_id=subject_id,
			details=dict(details or {}),
		)

	def _list(self, store: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(store.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _require_same_tenant(self, store: dict[str, Any], ids: list[str], tenant_id: str, reason: str) -> None:
		for item_id in ids:
			item = store.get(item_id)
			if item is None or item.tenant_id != tenant_id:
				raise PermissionError(reason)

	def _highest_risk_level(self, risk_ids: list[str]) -> str:
		order = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
		levels = [self._risks[risk_id].level.value for risk_id in risk_ids if risk_id in self._risks]
		return max(levels, key=lambda item: order[item]) if levels else "low"

	@staticmethod
	def _require_text(value: str, reason: str) -> None:
		if not value:
			raise PermissionError(reason)

	@staticmethod
	def _require_unit_interval(value: float, reason: str) -> None:
		if value < 0 or value > 1:
			raise ValueError(reason)
