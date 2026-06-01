"""Executable service layer for APG Signals Intelligence."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_BANDS, SUPPORTED_CLASSIFICATIONS, SUPPORTED_COLLECTION_MODES, SUPPORTED_PATTERN_TYPES, SUPPORTED_PROCESSING_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SOURCE_TYPES, evaluate_capability_rules, get_capability_contract
	from .models import CollectionTask, ProcessingBatch, SIGINTAgent, SIGINTReview, SignalAssessment, SignalAuthority, SignalObservation, SignalPattern, SignalSource
	from .sigint_runtime import bounded_score, normalize_code, positive_int, present
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_BANDS, SUPPORTED_CLASSIFICATIONS, SUPPORTED_COLLECTION_MODES, SUPPORTED_PATTERN_TYPES, SUPPORTED_PROCESSING_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SOURCE_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import CollectionTask, ProcessingBatch, SIGINTAgent, SIGINTReview, SignalAssessment, SignalAuthority, SignalObservation, SignalPattern, SignalSource  # type: ignore
	from sigint_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore


class SignalsIntelligenceService:
	"""Tenant-scoped SIGINT coordination runtime for generated APG applications."""

	def __init__(self) -> None:
		self.authorities: dict[tuple[str, str], SignalAuthority] = {}
		self.sources: dict[tuple[str, str], SignalSource] = {}
		self.tasks: dict[tuple[str, str], CollectionTask] = {}
		self.observations: dict[tuple[str, str], SignalObservation] = {}
		self.processing_batches: dict[tuple[str, str], ProcessingBatch] = {}
		self.patterns: dict[tuple[str, str], SignalPattern] = {}
		self.assessments: dict[tuple[str, str], SignalAssessment] = {}
		self.reviews: dict[tuple[str, str], SIGINTReview] = {}
		self.agents: dict[tuple[str, str], SIGINTAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = SignalAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "sigint_authority_recorded", authority_id)
		return item.to_dict()

	def register_source(self, source_id: str, tenant_id: str, source_type: str, band: str, source_reference: str, owner_id: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source_type = normalize_code(source_type)
		band = normalize_code(band)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_source", "source_type_supported": source_type in SUPPORTED_SOURCE_TYPES, "band_supported": band in SUPPORTED_BANDS, "source_reference_present": present(source_reference), "owner_present": present(owner_id), "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = SignalSource(source_id, tenant_id, source_type, band, source_reference, owner_id, authority_id, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "sigint_source_registered", source_id)
		return item.to_dict()

	def record_collection_task(self, task_id: str, tenant_id: str, authority_id: str, source_id: str, collection_mode: str, retention_days: int, minimization_reference: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source = self._tenant_source_or_none(source_id, tenant_id)
		collection_mode = normalize_code(collection_mode)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_collection_task", "authority_present": authority is not None, "source_present": source is not None, "source_authority_match": source is not None and source.authority_id == authority_id, "collection_mode_supported": collection_mode in SUPPORTED_COLLECTION_MODES, "retention_days_positive": positive_int(retention_days), "minimization_present": present(minimization_reference), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = CollectionTask(task_id, tenant_id, authority_id, source_id, collection_mode, int(retention_days), minimization_reference, approval_reference, evidence_reference)
		self.tasks[self._tenant_key(tenant_id, task_id)] = item
		self._audit(tenant_id, "sigint_collection_task_recorded", task_id)
		return item.to_dict()

	def record_observation(self, observation_id: str, tenant_id: str, task_id: str, observation_reference: str, fingerprint: str, confidence_score: float, evidence_reference: str) -> dict[str, Any]:
		task = self._tenant_task_or_none(task_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_observation", "task_present": task is not None, "observation_reference_present": present(observation_reference), "fingerprint_present": present(fingerprint), "confidence_valid": bounded_score(confidence_score), "evidence_present": present(evidence_reference)})
		item = SignalObservation(observation_id, tenant_id, task_id, observation_reference, fingerprint, float(confidence_score), evidence_reference)
		self.observations[self._tenant_key(tenant_id, observation_id)] = item
		self._audit(tenant_id, "sigint_observation_recorded", observation_id)
		return item.to_dict()

	def record_processing_batch(self, batch_id: str, tenant_id: str, observation_id: str, processing_type: str, quality_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		observation = self._tenant_observation_or_none(observation_id, tenant_id)
		processing_type = normalize_code(processing_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_processing_batch", "observation_present": observation is not None, "processing_type_supported": processing_type in SUPPORTED_PROCESSING_TYPES, "quality_valid": bounded_score(quality_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = ProcessingBatch(batch_id, tenant_id, observation_id, processing_type, float(quality_score), analyst_id, evidence_reference)
		self.processing_batches[self._tenant_key(tenant_id, batch_id)] = item
		self._audit(tenant_id, "sigint_processing_batch_recorded", batch_id)
		return item.to_dict()

	def record_pattern(self, pattern_id: str, tenant_id: str, batch_id: str, pattern_type: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		batch = self._tenant_batch_or_none(batch_id, tenant_id)
		pattern_type = normalize_code(pattern_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_pattern", "batch_present": batch is not None, "pattern_type_supported": pattern_type in SUPPORTED_PATTERN_TYPES, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = SignalPattern(pattern_id, tenant_id, batch_id, pattern_type, float(confidence_score), analyst_id, evidence_reference)
		self.patterns[self._tenant_key(tenant_id, pattern_id)] = item
		self._audit(tenant_id, "sigint_pattern_recorded", pattern_id)
		return item.to_dict()

	def record_assessment(self, assessment_id: str, tenant_id: str, pattern_id: str, assessment_type: str, classification: str, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		pattern = self._tenant_pattern_or_none(pattern_id, tenant_id)
		assessment_type = normalize_code(assessment_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_assessment", "pattern_present": pattern is not None, "assessment_type_supported": assessment_type in SUPPORTED_ASSESSMENT_TYPES, "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = SignalAssessment(assessment_id, tenant_id, pattern_id, assessment_type, classification, analyst_id, evidence_reference)
		self.assessments[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "sigint_assessment_recorded", assessment_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = SIGINTReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "sigint_review_recorded", review_id)
		return item.to_dict()

	def register_sigint_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_sigint_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = SIGINTAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "sigint_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "sigint_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "sigint_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.sigint.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "source_count": self._count(self.sources, tenant_id), "task_count": self._count(self.tasks, tenant_id), "observation_count": self._count(self.observations, tenant_id), "processing_batch_count": self._count(self.processing_batches, tenant_id), "pattern_count": self._count(self.patterns, tenant_id), "assessment_count": self._count(self.assessments, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> SignalAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> SignalSource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_task_or_none(self, item_id: str, tenant_id: str) -> CollectionTask | None:
		return self.tasks.get(self._tenant_key(tenant_id, item_id))

	def _tenant_observation_or_none(self, item_id: str, tenant_id: str) -> SignalObservation | None:
		return self.observations.get(self._tenant_key(tenant_id, item_id))

	def _tenant_batch_or_none(self, item_id: str, tenant_id: str) -> ProcessingBatch | None:
		return self.processing_batches.get(self._tenant_key(tenant_id, item_id))

	def _tenant_pattern_or_none(self, item_id: str, tenant_id: str) -> SignalPattern | None:
		return self.patterns.get(self._tenant_key(tenant_id, item_id))

	def _tenant_key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "sigint_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "sigint_policy_denied")


IntelSIGINTService = SignalsIntelligenceService
