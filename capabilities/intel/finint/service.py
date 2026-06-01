"""Executable service layer for APG Financial Intelligence."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_PATTERN_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_TIERS, SUPPORTED_RISK_TYPES, SUPPORTED_SOURCE_TYPES, SUPPORTED_SUBJECT_TYPES, SUPPORTED_TRANSACTION_TYPES, evaluate_capability_rules, get_capability_contract
	from .finint_runtime import bounded_score, normalize_code, positive_amount, positive_int, present
	from .models import FININTAgent, FININTDissemination, FININTReview, FinancialAuthority, FinancialPattern, FinancialReferral, FinancialRiskAssessment, FinancialSource, FinancialSubject, FinancialTransaction
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_PATTERN_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_TIERS, SUPPORTED_RISK_TYPES, SUPPORTED_SOURCE_TYPES, SUPPORTED_SUBJECT_TYPES, SUPPORTED_TRANSACTION_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from finint_runtime import bounded_score, normalize_code, positive_amount, positive_int, present  # type: ignore
	from models import FININTAgent, FININTDissemination, FININTReview, FinancialAuthority, FinancialPattern, FinancialReferral, FinancialRiskAssessment, FinancialSource, FinancialSubject, FinancialTransaction  # type: ignore


class FinancialIntelligenceService:
	"""Tenant-scoped FININT coordination runtime for generated APG applications."""

	def __init__(self) -> None:
		self.authorities: dict[tuple[str, str], FinancialAuthority] = {}
		self.sources: dict[tuple[str, str], FinancialSource] = {}
		self.subjects: dict[tuple[str, str], FinancialSubject] = {}
		self.transactions: dict[tuple[str, str], FinancialTransaction] = {}
		self.patterns: dict[tuple[str, str], FinancialPattern] = {}
		self.risks: dict[tuple[str, str], FinancialRiskAssessment] = {}
		self.referrals: dict[tuple[str, str], FinancialReferral] = {}
		self.disseminations: dict[tuple[str, str], FININTDissemination] = {}
		self.reviews: dict[tuple[str, str], FININTReview] = {}
		self.agents: dict[tuple[str, str], FININTAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = FinancialAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "finint_authority_recorded", authority_id)
		return item.to_dict()

	def register_source(self, source_id: str, tenant_id: str, source_type: str, jurisdiction: str, owner_id: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source_type = normalize_code(source_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_source", "source_type_supported": source_type in SUPPORTED_SOURCE_TYPES, "jurisdiction_present": present(jurisdiction), "owner_present": present(owner_id), "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = FinancialSource(source_id, tenant_id, source_type, jurisdiction, owner_id, authority_id, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "finint_source_registered", source_id)
		return item.to_dict()

	def record_subject(self, subject_id: str, tenant_id: str, subject_type: str, subject_reference: str, risk_tier: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		subject_type = normalize_code(subject_type)
		risk_tier = normalize_code(risk_tier)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_subject", "subject_type_supported": subject_type in SUPPORTED_SUBJECT_TYPES, "subject_reference_present": present(subject_reference), "risk_tier_supported": risk_tier in SUPPORTED_RISK_TIERS, "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = FinancialSubject(subject_id, tenant_id, subject_type, subject_reference, risk_tier, authority_id, evidence_reference)
		self.subjects[self._tenant_key(tenant_id, subject_id)] = item
		self._audit(tenant_id, "finint_subject_recorded", subject_id)
		return item.to_dict()

	def record_transaction(self, transaction_id: str, tenant_id: str, source_id: str, subject_id: str, transaction_reference: str, amount: float, currency: str, transaction_type: str, occurred_at: str, evidence_reference: str) -> dict[str, Any]:
		source = self._tenant_source_or_none(source_id, tenant_id)
		subject = self._tenant_subject_or_none(subject_id, tenant_id)
		transaction_type = normalize_code(transaction_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_transaction", "source_present": source is not None, "subject_present": subject is not None, "source_subject_authority_match": source is not None and subject is not None and source.authority_id == subject.authority_id, "transaction_reference_present": present(transaction_reference), "amount_positive": positive_amount(amount), "currency_present": present(currency), "transaction_type_supported": transaction_type in SUPPORTED_TRANSACTION_TYPES, "occurred_at_present": present(occurred_at), "evidence_present": present(evidence_reference)})
		item = FinancialTransaction(transaction_id, tenant_id, source_id, subject_id, transaction_reference, float(amount), currency.upper(), transaction_type, occurred_at, evidence_reference)
		self.transactions[self._tenant_key(tenant_id, transaction_id)] = item
		self._audit(tenant_id, "finint_transaction_recorded", transaction_id)
		return item.to_dict()

	def record_pattern(self, pattern_id: str, tenant_id: str, transaction_id: str, pattern_type: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		transaction = self._tenant_transaction_or_none(transaction_id, tenant_id)
		pattern_type = normalize_code(pattern_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_pattern", "transaction_present": transaction is not None, "pattern_type_supported": pattern_type in SUPPORTED_PATTERN_TYPES, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = FinancialPattern(pattern_id, tenant_id, transaction_id, pattern_type, float(confidence_score), analyst_id, evidence_reference)
		self.patterns[self._tenant_key(tenant_id, pattern_id)] = item
		self._audit(tenant_id, "finint_pattern_recorded", pattern_id)
		return item.to_dict()

	def record_risk(self, assessment_id: str, tenant_id: str, pattern_id: str, risk_type: str, risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		pattern = self._tenant_pattern_or_none(pattern_id, tenant_id)
		risk_type = normalize_code(risk_type)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_risk", "pattern_present": pattern is not None, "risk_type_supported": risk_type in SUPPORTED_RISK_TYPES, "risk_level_supported": risk_level in SUPPORTED_RISK_TIERS, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = FinancialRiskAssessment(assessment_id, tenant_id, pattern_id, risk_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.risks[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "finint_risk_recorded", assessment_id)
		return item.to_dict()

	def record_referral(self, referral_id: str, tenant_id: str, assessment_id: str, referral_type: str, recipient: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._tenant_risk_or_none(assessment_id, tenant_id)
		referral_type = normalize_code(referral_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_referral", "assessment_present": assessment is not None, "referral_type_supported": referral_type in SUPPORTED_REFERRAL_TYPES, "recipient_present": present(recipient), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = FinancialReferral(referral_id, tenant_id, assessment_id, referral_type, recipient, approval_reference, evidence_reference)
		self.referrals[self._tenant_key(tenant_id, referral_id)] = item
		self._audit(tenant_id, "finint_referral_recorded", referral_id)
		return item.to_dict()

	def record_dissemination(self, dissemination_id: str, tenant_id: str, assessment_id: str, audience: str, release_marking: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._tenant_risk_or_none(assessment_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_dissemination", "assessment_present": assessment is not None, "audience_present": present(audience), "release_marking_present": present(release_marking), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = FININTDissemination(dissemination_id, tenant_id, assessment_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "finint_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = FININTReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "finint_review_recorded", review_id)
		return item.to_dict()

	def register_finint_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_finint_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = FININTAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "finint_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, funds_movement_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "finint_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "funds_movement_scope": funds_movement_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope, "funds_movement_scope": funds_movement_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "finint_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.finint.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "source_count": self._count(self.sources, tenant_id), "subject_count": self._count(self.subjects, tenant_id), "transaction_count": self._count(self.transactions, tenant_id), "pattern_count": self._count(self.patterns, tenant_id), "risk_count": self._count(self.risks, tenant_id), "referral_count": self._count(self.referrals, tenant_id), "dissemination_count": self._count(self.disseminations, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> FinancialAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> FinancialSource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_subject_or_none(self, item_id: str, tenant_id: str) -> FinancialSubject | None:
		return self.subjects.get(self._tenant_key(tenant_id, item_id))

	def _tenant_transaction_or_none(self, item_id: str, tenant_id: str) -> FinancialTransaction | None:
		return self.transactions.get(self._tenant_key(tenant_id, item_id))

	def _tenant_pattern_or_none(self, item_id: str, tenant_id: str) -> FinancialPattern | None:
		return self.patterns.get(self._tenant_key(tenant_id, item_id))

	def _tenant_risk_or_none(self, item_id: str, tenant_id: str) -> FinancialRiskAssessment | None:
		return self.risks.get(self._tenant_key(tenant_id, item_id))

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
		reasons = ", ".join(action.get("reason", action.get("rule", "finint_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "finint_policy_denied")


IntelFININTService = FinancialIntelligenceService
