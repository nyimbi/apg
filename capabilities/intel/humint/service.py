"""Executable service layer for APG Human Intelligence."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_CONTACT_METHODS, SUPPORTED_HANDLING_STATUSES, SUPPORTED_LEAD_TYPES, SUPPORTED_PRIORITIES, SUPPORTED_RELIABILITY_GRADES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SOURCE_TYPES, evaluate_capability_rules, get_capability_contract
	from .humint_runtime import bounded_score, normalize_code, positive_int, present
	from .models import ContactPlan, ContactReport, Debriefing, HUMINTAgent, HUMINTDissemination, HUMINTLead, HUMINTReview, HumanSource, ReliabilityAssessment, SourceAuthority
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_CONTACT_METHODS, SUPPORTED_HANDLING_STATUSES, SUPPORTED_LEAD_TYPES, SUPPORTED_PRIORITIES, SUPPORTED_RELIABILITY_GRADES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SOURCE_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from humint_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore
	from models import ContactPlan, ContactReport, Debriefing, HUMINTAgent, HUMINTDissemination, HUMINTLead, HUMINTReview, HumanSource, ReliabilityAssessment, SourceAuthority  # type: ignore


class HumanIntelligenceService:
	"""Tenant-scoped HUMINT coordination runtime for generated APG applications."""

	def __init__(self) -> None:
		self.authorities: dict[tuple[str, str], SourceAuthority] = {}
		self.sources: dict[tuple[str, str], HumanSource] = {}
		self.contact_plans: dict[tuple[str, str], ContactPlan] = {}
		self.contact_reports: dict[tuple[str, str], ContactReport] = {}
		self.debriefings: dict[tuple[str, str], Debriefing] = {}
		self.reliability_assessments: dict[tuple[str, str], ReliabilityAssessment] = {}
		self.leads: dict[tuple[str, str], HUMINTLead] = {}
		self.disseminations: dict[tuple[str, str], HUMINTDissemination] = {}
		self.reviews: dict[tuple[str, str], HUMINTReview] = {}
		self.agents: dict[tuple[str, str], HUMINTAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = SourceAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "humint_authority_recorded", authority_id)
		return item.to_dict()

	def register_source(self, source_id: str, tenant_id: str, source_type: str, handling_status: str, risk_level: str, owner_id: str, authority_id: str, protection_reference: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source_type = normalize_code(source_type)
		handling_status = normalize_code(handling_status)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_source", "source_type_supported": source_type in SUPPORTED_SOURCE_TYPES, "handling_status_supported": handling_status in SUPPORTED_HANDLING_STATUSES, "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "owner_present": present(owner_id), "authority_present": authority is not None, "protection_present": present(protection_reference), "evidence_present": present(evidence_reference)})
		item = HumanSource(source_id, tenant_id, source_type, handling_status, risk_level, owner_id, authority_id, protection_reference, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "humint_source_registered", source_id)
		return item.to_dict()

	def record_contact_plan(self, plan_id: str, tenant_id: str, authority_id: str, source_id: str, contact_method: str, objective_reference: str, safety_plan_reference: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source = self._tenant_source_or_none(source_id, tenant_id)
		contact_method = normalize_code(contact_method)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_contact_plan", "authority_present": authority is not None, "source_present": source is not None, "source_authority_match": source is not None and source.authority_id == authority_id, "contact_method_supported": contact_method in SUPPORTED_CONTACT_METHODS, "objective_present": present(objective_reference), "safety_plan_present": present(safety_plan_reference), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = ContactPlan(plan_id, tenant_id, authority_id, source_id, contact_method, objective_reference, safety_plan_reference, approval_reference, evidence_reference)
		self.contact_plans[self._tenant_key(tenant_id, plan_id)] = item
		self._audit(tenant_id, "humint_contact_plan_recorded", plan_id)
		return item.to_dict()

	def record_contact_report(self, report_id: str, tenant_id: str, plan_id: str, report_reference: str, handler_id: str, source_welfare_score: float, evidence_reference: str) -> dict[str, Any]:
		plan = self._tenant_plan_or_none(plan_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_contact_report", "plan_present": plan is not None, "report_reference_present": present(report_reference), "handler_present": present(handler_id), "source_welfare_valid": bounded_score(source_welfare_score), "evidence_present": present(evidence_reference)})
		item = ContactReport(report_id, tenant_id, plan_id, report_reference, handler_id, float(source_welfare_score), evidence_reference)
		self.contact_reports[self._tenant_key(tenant_id, report_id)] = item
		self._audit(tenant_id, "humint_contact_report_recorded", report_id)
		return item.to_dict()

	def record_debriefing(self, debriefing_id: str, tenant_id: str, report_id: str, topic: str, classification: str, credibility_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		report = self._tenant_report_or_none(report_id, tenant_id)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_debriefing", "report_present": report is not None, "topic_present": present(topic), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "credibility_valid": bounded_score(credibility_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = Debriefing(debriefing_id, tenant_id, report_id, topic, classification, float(credibility_score), analyst_id, evidence_reference)
		self.debriefings[self._tenant_key(tenant_id, debriefing_id)] = item
		self._audit(tenant_id, "humint_debriefing_recorded", debriefing_id)
		return item.to_dict()

	def record_reliability(self, assessment_id: str, tenant_id: str, source_id: str, reliability_grade: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		source = self._tenant_source_or_none(source_id, tenant_id)
		reliability_grade = normalize_code(reliability_grade)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_reliability", "source_present": source is not None, "reliability_grade_supported": reliability_grade in SUPPORTED_RELIABILITY_GRADES, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = ReliabilityAssessment(assessment_id, tenant_id, source_id, reliability_grade, float(confidence_score), analyst_id, evidence_reference)
		self.reliability_assessments[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "humint_reliability_recorded", assessment_id)
		return item.to_dict()

	def record_lead(self, lead_id: str, tenant_id: str, debriefing_id: str, lead_type: str, priority: str, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		debriefing = self._tenant_debriefing_or_none(debriefing_id, tenant_id)
		lead_type = normalize_code(lead_type)
		priority = normalize_code(priority)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_lead", "debriefing_present": debriefing is not None, "lead_type_supported": lead_type in SUPPORTED_LEAD_TYPES, "priority_supported": priority in SUPPORTED_PRIORITIES, "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = HUMINTLead(lead_id, tenant_id, debriefing_id, lead_type, priority, analyst_id, evidence_reference)
		self.leads[self._tenant_key(tenant_id, lead_id)] = item
		self._audit(tenant_id, "humint_lead_recorded", lead_id)
		return item.to_dict()

	def record_dissemination(self, dissemination_id: str, tenant_id: str, lead_id: str, audience: str, release_marking: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		lead = self._tenant_lead_or_none(lead_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_dissemination", "lead_present": lead is not None, "audience_present": present(audience), "release_marking_present": present(release_marking), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = HUMINTDissemination(dissemination_id, tenant_id, lead_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "humint_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = HUMINTReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "humint_review_recorded", review_id)
		return item.to_dict()

	def register_humint_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_humint_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = HUMINTAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "humint_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, coercive_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "humint_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "coercive_scope": coercive_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope, "coercive_scope": coercive_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "humint_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.humint.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "source_count": self._count(self.sources, tenant_id), "contact_plan_count": self._count(self.contact_plans, tenant_id), "contact_report_count": self._count(self.contact_reports, tenant_id), "debriefing_count": self._count(self.debriefings, tenant_id), "reliability_count": self._count(self.reliability_assessments, tenant_id), "lead_count": self._count(self.leads, tenant_id), "dissemination_count": self._count(self.disseminations, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> SourceAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> HumanSource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_plan_or_none(self, item_id: str, tenant_id: str) -> ContactPlan | None:
		return self.contact_plans.get(self._tenant_key(tenant_id, item_id))

	def _tenant_report_or_none(self, item_id: str, tenant_id: str) -> ContactReport | None:
		return self.contact_reports.get(self._tenant_key(tenant_id, item_id))

	def _tenant_debriefing_or_none(self, item_id: str, tenant_id: str) -> Debriefing | None:
		return self.debriefings.get(self._tenant_key(tenant_id, item_id))

	def _tenant_lead_or_none(self, item_id: str, tenant_id: str) -> HUMINTLead | None:
		return self.leads.get(self._tenant_key(tenant_id, item_id))

	def _tenant_key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "humint_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "humint_policy_denied")


IntelHUMINTService = HumanIntelligenceService
