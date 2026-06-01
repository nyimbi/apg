"""Executable service layer for APG Open Source Intelligence."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_COLLECTION_METHODS, SUPPORTED_PRIORITIES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_TIERS, SUPPORTED_SOURCE_TYPES, SUPPORTED_TRIAGE_DECISIONS, evaluate_capability_rules, get_capability_contract
	from .models import CollectionPlan, CollectionRequirement, DisseminationPackage, EvidenceRecord, IntelligenceAssessment, OSINTAgent, OSINTReview, SourceRegistryEntry, TriageDecision
	from .osint_runtime import bounded_score, normalize_code, positive_int, present
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_COLLECTION_METHODS, SUPPORTED_PRIORITIES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_TIERS, SUPPORTED_SOURCE_TYPES, SUPPORTED_TRIAGE_DECISIONS, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import CollectionPlan, CollectionRequirement, DisseminationPackage, EvidenceRecord, IntelligenceAssessment, OSINTAgent, OSINTReview, SourceRegistryEntry, TriageDecision  # type: ignore
	from osint_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore


class OpenSourceIntelligenceService:
	"""Tenant-scoped OSINT runtime for generated APG applications."""

	def __init__(self) -> None:
		self.requirements: dict[str, CollectionRequirement] = {}
		self.sources: dict[str, SourceRegistryEntry] = {}
		self.plans: dict[str, CollectionPlan] = {}
		self.evidence: dict[str, EvidenceRecord] = {}
		self.triage: dict[str, TriageDecision] = {}
		self.assessments: dict[str, IntelligenceAssessment] = {}
		self.dissemination: dict[str, DisseminationPackage] = {}
		self.reviews: dict[str, OSINTReview] = {}
		self.agents: dict[str, OSINTAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_requirement(self, requirement_id: str, tenant_id: str, topic: str, priority: str, requester_id: str, classification: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		priority = normalize_code(priority)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_requirement", "topic_present": present(topic), "priority_supported": priority in SUPPORTED_PRIORITIES, "requester_present": present(requester_id), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "evidence_present": present(evidence_reference)})
		item = CollectionRequirement(requirement_id, tenant_id, topic, priority, requester_id, classification, evidence_reference)
		self.requirements[requirement_id] = item
		self._audit(tenant_id, "osint_requirement_registered", requirement_id)
		return item.to_dict()

	def register_source(self, source_id: str, tenant_id: str, source_type: str, source_reference: str, owner_id: str, terms_review_reference: str, risk_tier: str, evidence_reference: str) -> dict[str, Any]:
		source_type = normalize_code(source_type)
		risk_tier = normalize_code(risk_tier)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_source", "source_type_supported": source_type in SUPPORTED_SOURCE_TYPES, "source_reference_present": present(source_reference), "owner_present": present(owner_id), "terms_review_present": present(terms_review_reference), "risk_tier_supported": risk_tier in SUPPORTED_RISK_TIERS, "evidence_present": present(evidence_reference)})
		item = SourceRegistryEntry(source_id, tenant_id, source_type, source_reference, owner_id, terms_review_reference, risk_tier, evidence_reference)
		self.sources[source_id] = item
		self._audit(tenant_id, "osint_source_registered", source_id)
		return item.to_dict()

	def record_collection_plan(self, plan_id: str, tenant_id: str, requirement_id: str, source_id: str, method: str, cadence: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		requirement = self._tenant_requirement_or_none(requirement_id, tenant_id)
		source = self._tenant_source_or_none(source_id, tenant_id)
		method = normalize_code(method)
		high_risk_source = source is not None and source.risk_tier in {"high", "critical"}
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_collection_plan", "requirement_present": requirement is not None, "source_present": source is not None, "method_supported": method in SUPPORTED_COLLECTION_METHODS, "cadence_present": present(cadence), "high_risk_source": high_risk_source, "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = CollectionPlan(plan_id, tenant_id, requirement_id, source_id, method, cadence, approval_reference, evidence_reference)
		self.plans[plan_id] = item
		self._audit(tenant_id, "osint_collection_plan_recorded", plan_id)
		return item.to_dict()

	def record_evidence(self, evidence_id: str, tenant_id: str, plan_id: str, content_reference: str, fingerprint: str, confidence_score: float, evidence_reference: str) -> dict[str, Any]:
		plan = self._tenant_plan_or_none(plan_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_evidence", "plan_present": plan is not None, "content_present": present(content_reference), "fingerprint_present": present(fingerprint), "confidence_valid": bounded_score(confidence_score), "evidence_reference_present": present(evidence_reference)})
		item = EvidenceRecord(evidence_id, tenant_id, plan_id, content_reference, fingerprint, float(confidence_score), evidence_reference)
		self.evidence[evidence_id] = item
		self._audit(tenant_id, "osint_evidence_recorded", evidence_id)
		return item.to_dict()

	def record_triage(self, triage_id: str, tenant_id: str, evidence_id: str, decision: str, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		evidence = self._tenant_evidence_or_none(evidence_id, tenant_id)
		decision = normalize_code(decision)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_triage", "evidence_present": evidence is not None, "decision_supported": decision in SUPPORTED_TRIAGE_DECISIONS, "analyst_present": present(analyst_id), "evidence_reference_present": present(evidence_reference)})
		item = TriageDecision(triage_id, tenant_id, evidence_id, decision, analyst_id, evidence_reference)
		self.triage[triage_id] = item
		self._audit(tenant_id, "osint_triage_recorded", triage_id)
		return item.to_dict()

	def record_assessment(self, assessment_id: str, tenant_id: str, requirement_id: str, assessment_type: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		requirement = self._tenant_requirement_or_none(requirement_id, tenant_id)
		assessment_type = normalize_code(assessment_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_assessment", "requirement_present": requirement is not None, "assessment_type_supported": assessment_type in SUPPORTED_ASSESSMENT_TYPES, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = IntelligenceAssessment(assessment_id, tenant_id, requirement_id, assessment_type, float(confidence_score), analyst_id, evidence_reference)
		self.assessments[assessment_id] = item
		self._audit(tenant_id, "osint_assessment_recorded", assessment_id)
		return item.to_dict()

	def record_dissemination(self, package_id: str, tenant_id: str, assessment_id: str, audience: str, release_marking: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._tenant_assessment_or_none(assessment_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_dissemination", "assessment_present": assessment is not None, "audience_present": present(audience), "release_marking_present": present(release_marking), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = DisseminationPackage(package_id, tenant_id, assessment_id, audience, release_marking, approval_reference, evidence_reference)
		self.dissemination[package_id] = item
		self._audit(tenant_id, "osint_dissemination_recorded", package_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = OSINTReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = item
		self._audit(tenant_id, "osint_review_recorded", review_id)
		return item.to_dict()

	def register_osint_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_osint_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = OSINTAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[agent_id] = item
		self._audit(tenant_id, "osint_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "osint_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "osint_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.osint.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "requirement_count": self._count(self.requirements, tenant_id), "source_count": self._count(self.sources, tenant_id), "high_risk_source_count": sum(1 for item in self.sources.values() if item.tenant_id == tenant_id and item.risk_tier in {"high", "critical"}), "plan_count": self._count(self.plans, tenant_id), "evidence_count": self._count(self.evidence, tenant_id), "triage_count": self._count(self.triage, tenant_id), "assessment_count": self._count(self.assessments, tenant_id), "dissemination_count": self._count(self.dissemination, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_requirement_or_none(self, item_id: str, tenant_id: str) -> CollectionRequirement | None:
		item = self.requirements.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> SourceRegistryEntry | None:
		item = self.sources.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_plan_or_none(self, item_id: str, tenant_id: str) -> CollectionPlan | None:
		item = self.plans.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_evidence_or_none(self, item_id: str, tenant_id: str) -> EvidenceRecord | None:
		item = self.evidence.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_assessment_or_none(self, item_id: str, tenant_id: str) -> IntelligenceAssessment | None:
		item = self.assessments.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "osint_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "osint_policy_denied")


IntelOSINTService = OpenSourceIntelligenceService
