"""Executable service layer for APG Regulatory Technology."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CHANGE_TYPES, SUPPORTED_FILING_TYPES, SUPPORTED_JURISDICTIONS, SUPPORTED_REGULATORS, SUPPORTED_REGULATORY_FRAMEWORKS, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_RATINGS, SUPPORTED_SUBMISSION_CHANNELS, evaluate_capability_rules, get_capability_contract
	from .models import ImpactAssessment, ObligationMapping, RegulatoryChange, RegulatoryFiling, RegulatoryInquiry, RegulatoryResponse, RegulatorySource, RegulatorySubmission, RegTechAgent, RegTechReview
	from .regtech_runtime import normalize_code, normalize_jurisdiction, present
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CHANGE_TYPES, SUPPORTED_FILING_TYPES, SUPPORTED_JURISDICTIONS, SUPPORTED_REGULATORS, SUPPORTED_REGULATORY_FRAMEWORKS, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_RATINGS, SUPPORTED_SUBMISSION_CHANNELS, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import ImpactAssessment, ObligationMapping, RegulatoryChange, RegulatoryFiling, RegulatoryInquiry, RegulatoryResponse, RegulatorySource, RegulatorySubmission, RegTechAgent, RegTechReview  # type: ignore
	from regtech_runtime import normalize_code, normalize_jurisdiction, present  # type: ignore


class RegTechService:
	"""Dependency-light regulatory technology runtime for generated APG applications."""

	def __init__(self) -> None:
		self.sources: dict[str, RegulatorySource] = {}
		self.changes: dict[str, RegulatoryChange] = {}
		self.obligations: dict[str, ObligationMapping] = {}
		self.impacts: dict[str, ImpactAssessment] = {}
		self.filings: dict[str, RegulatoryFiling] = {}
		self.submissions: dict[str, RegulatorySubmission] = {}
		self.inquiries: dict[str, RegulatoryInquiry] = {}
		self.responses: dict[str, RegulatoryResponse] = {}
		self.reviews: dict[str, RegTechReview] = {}
		self.agents: dict[str, RegTechAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_source(self, source_id: str, tenant_id: str, regulator: str, jurisdiction: str, source_reference: str, owner_id: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		regulator = normalize_code(regulator)
		jurisdiction = normalize_jurisdiction(jurisdiction)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_source", "regulator_supported": regulator in SUPPORTED_REGULATORS, "jurisdiction_supported": jurisdiction in SUPPORTED_JURISDICTIONS, "source_present": present(source_reference), "owner_present": present(owner_id), "evidence_present": present(evidence_reference)})
		item = RegulatorySource(source_id, tenant_id, regulator, jurisdiction, source_reference, owner_id, evidence_reference)
		self.sources[source_id] = item
		self._audit(tenant_id, "regulatory_source_registered", source_id)
		return item.to_dict()

	def record_change(self, change_id: str, tenant_id: str, source_id: str, framework: str, change_type: str, title: str, effective_date: str, severity: str, evidence_reference: str) -> dict[str, Any]:
		source = self._tenant_source_or_none(source_id, tenant_id)
		framework = normalize_code(framework)
		change_type = normalize_code(change_type)
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_change", "source_present": source is not None, "framework_supported": framework in SUPPORTED_REGULATORY_FRAMEWORKS, "change_type_supported": change_type in SUPPORTED_CHANGE_TYPES, "effective_date_present": present(effective_date), "severity_supported": severity in SUPPORTED_RISK_RATINGS, "evidence_present": present(evidence_reference)})
		item = RegulatoryChange(change_id, tenant_id, source_id, framework, change_type, title, effective_date, severity, evidence_reference, "active")
		self.changes[change_id] = item
		self._audit(tenant_id, "regulatory_change_recorded", change_id)
		return item.to_dict()

	def map_obligation(self, mapping_id: str, tenant_id: str, change_id: str, obligation_reference: str, policy_reference: str, owner_id: str, due_date: str) -> dict[str, Any]:
		change = self._tenant_change_or_none(change_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "map_obligation", "change_present": change is not None, "obligation_present": present(obligation_reference), "policy_present": present(policy_reference), "owner_present": present(owner_id), "due_date_present": present(due_date)})
		item = ObligationMapping(mapping_id, tenant_id, change_id, obligation_reference, policy_reference, owner_id, due_date)
		self.obligations[mapping_id] = item
		if change is not None:
			change.status = "mapped"
		self._audit(tenant_id, "regulatory_obligation_mapped", mapping_id)
		return item.to_dict()

	def assess_impact(self, assessment_id: str, tenant_id: str, change_id: str, impacted_capability: str, risk_rating: str, evidence_reference: str, reviewer_id: str) -> dict[str, Any]:
		change = self._tenant_change_or_none(change_id, tenant_id)
		risk_rating = normalize_code(risk_rating)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "assess_impact", "change_present": change is not None, "impacted_capability_present": present(impacted_capability), "risk_rating_supported": risk_rating in SUPPORTED_RISK_RATINGS, "evidence_present": present(evidence_reference), "reviewer_present": present(reviewer_id)})
		item = ImpactAssessment(assessment_id, tenant_id, change_id, impacted_capability, risk_rating, evidence_reference, reviewer_id)
		self.impacts[assessment_id] = item
		self._audit(tenant_id, "regulatory_impact_assessed", assessment_id)
		return item.to_dict()

	def prepare_filing(self, filing_id: str, tenant_id: str, framework: str, filing_type: str, period: str, evidence_reference: str, owner_id: str) -> dict[str, Any]:
		framework = normalize_code(framework)
		filing_type = normalize_code(filing_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "prepare_filing", "framework_supported": framework in SUPPORTED_REGULATORY_FRAMEWORKS, "filing_type_supported": filing_type in SUPPORTED_FILING_TYPES, "period_present": present(period), "evidence_present": present(evidence_reference), "owner_present": present(owner_id)})
		item = RegulatoryFiling(filing_id, tenant_id, framework, filing_type, period, evidence_reference, owner_id, "draft")
		self.filings[filing_id] = item
		self._audit(tenant_id, "regulatory_filing_prepared", filing_id)
		return item.to_dict()

	def record_submission(self, submission_id: str, tenant_id: str, filing_id: str, channel: str, submitted_by: str, submitted_at: str, acknowledgment_reference: str) -> dict[str, Any]:
		filing = self._tenant_filing_or_none(filing_id, tenant_id)
		channel = normalize_code(channel)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_submission", "filing_present": filing is not None, "channel_supported": channel in SUPPORTED_SUBMISSION_CHANNELS, "submitted_by_present": present(submitted_by), "submitted_at_present": present(submitted_at), "acknowledgment_present": present(acknowledgment_reference)})
		item = RegulatorySubmission(submission_id, tenant_id, filing_id, channel, submitted_by, submitted_at, acknowledgment_reference)
		self.submissions[submission_id] = item
		if filing is not None:
			filing.status = "submitted"
		self._audit(tenant_id, "regulatory_submission_recorded", submission_id)
		return item.to_dict()

	def open_inquiry(self, inquiry_id: str, tenant_id: str, regulator: str, reference_id: str, severity: str, due_date: str, evidence_reference: str) -> dict[str, Any]:
		regulator = normalize_code(regulator)
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "open_inquiry", "regulator_supported": regulator in SUPPORTED_REGULATORS, "reference_present": present(reference_id), "severity_supported": severity in SUPPORTED_RISK_RATINGS, "due_date_present": present(due_date), "evidence_present": present(evidence_reference)})
		item = RegulatoryInquiry(inquiry_id, tenant_id, regulator, reference_id, severity, due_date, evidence_reference, "open")
		self.inquiries[inquiry_id] = item
		self._audit(tenant_id, "regulatory_inquiry_opened", inquiry_id)
		return item.to_dict()

	def record_response(self, response_id: str, tenant_id: str, inquiry_id: str, responder_id: str, response_reference: str, approval_reference: str) -> dict[str, Any]:
		inquiry = self._tenant_inquiry_or_none(inquiry_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_response", "inquiry_present": inquiry is not None, "responder_present": present(responder_id), "response_present": present(response_reference), "approval_present": present(approval_reference)})
		item = RegulatoryResponse(response_id, tenant_id, inquiry_id, responder_id, response_reference, approval_reference)
		self.responses[response_id] = item
		if inquiry is not None:
			inquiry.status = "responded"
		self._audit(tenant_id, "regulatory_response_recorded", response_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = RegTechReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = item
		self._audit(tenant_id, "regulatory_review_recorded", review_id)
		return item.to_dict()

	def register_regtech_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_regtech_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = RegTechAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[agent_id] = item
		self._audit(tenant_id, "regulatory_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "regtech_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "regtech_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.regtech.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "source_count": self._count(self.sources, tenant_id), "change_count": self._count(self.changes, tenant_id), "obligation_count": self._count(self.obligations, tenant_id), "impact_count": self._count(self.impacts, tenant_id), "filing_count": self._count(self.filings, tenant_id), "submission_count": self._count(self.submissions, tenant_id), "open_inquiry_count": sum(1 for item in self.inquiries.values() if item.tenant_id == tenant_id and item.status == "open"), "response_count": self._count(self.responses, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> RegulatorySource | None:
		item = self.sources.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_change_or_none(self, item_id: str, tenant_id: str) -> RegulatoryChange | None:
		item = self.changes.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_filing_or_none(self, item_id: str, tenant_id: str) -> RegulatoryFiling | None:
		item = self.filings.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_inquiry_or_none(self, item_id: str, tenant_id: str) -> RegulatoryInquiry | None:
		item = self.inquiries.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "regtech_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "regtech_policy_denied")


RegulatoryTechnologyService = RegTechService
FintechRegTechService = RegTechService
