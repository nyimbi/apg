"""Executable service layer for APG Intelligence Fusion."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ARTIFACT_TYPES, SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_CORRELATION_TYPES, SUPPORTED_HYPOTHESIS_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SOURCE_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract
	from .fusion_runtime import bounded_score, normalize_code, positive_int, present
	from .models import FusionAgent, FusionArtifact, FusionAssessment, FusionAuthority, FusionCorrelation, FusionDissemination, FusionHypothesis, FusionReferral, FusionReview, FusionSource, FusionWorkspace
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ARTIFACT_TYPES, SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_CORRELATION_TYPES, SUPPORTED_HYPOTHESIS_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SOURCE_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from fusion_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore
	from models import FusionAgent, FusionArtifact, FusionAssessment, FusionAuthority, FusionCorrelation, FusionDissemination, FusionHypothesis, FusionReferral, FusionReview, FusionSource, FusionWorkspace  # type: ignore


class IntelligenceFusionService:
	"""Tenant-scoped intelligence fusion runtime for generated APG applications."""

	def __init__(self) -> None:
		self.authorities: dict[tuple[str, str], FusionAuthority] = {}
		self.workspaces: dict[tuple[str, str], FusionWorkspace] = {}
		self.sources: dict[tuple[str, str], FusionSource] = {}
		self.artifacts: dict[tuple[str, str], FusionArtifact] = {}
		self.correlations: dict[tuple[str, str], FusionCorrelation] = {}
		self.hypotheses: dict[tuple[str, str], FusionHypothesis] = {}
		self.assessments: dict[tuple[str, str], FusionAssessment] = {}
		self.referrals: dict[tuple[str, str], FusionReferral] = {}
		self.disseminations: dict[tuple[str, str], FusionDissemination] = {}
		self.reviews: dict[tuple[str, str], FusionReview] = {}
		self.agents: dict[tuple[str, str], FusionAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = FusionAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "fusion_authority_recorded", authority_id)
		return item.to_dict()

	def record_workspace(self, workspace_id: str, tenant_id: str, workspace_type: str, name: str, classification: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		workspace_type = normalize_code(workspace_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_workspace", "workspace_type_supported": workspace_type in SUPPORTED_WORKSPACE_TYPES, "workspace_name_present": present(name), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = FusionWorkspace(workspace_id, tenant_id, workspace_type, name, classification, authority_id, evidence_reference)
		self.workspaces[self._tenant_key(tenant_id, workspace_id)] = item
		self._audit(tenant_id, "fusion_workspace_recorded", workspace_id)
		return item.to_dict()

	def register_source(self, source_id: str, tenant_id: str, source_type: str, source_reference: str, custodian_id: str, authority_id: str, lineage_reference: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source_type = normalize_code(source_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_source", "source_type_supported": source_type in SUPPORTED_SOURCE_TYPES, "source_reference_present": present(source_reference), "custodian_present": present(custodian_id), "authority_present": authority is not None, "lineage_present": present(lineage_reference), "evidence_present": present(evidence_reference)})
		item = FusionSource(source_id, tenant_id, source_type, source_reference, custodian_id, authority_id, lineage_reference, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "fusion_source_registered", source_id)
		return item.to_dict()

	def record_artifact(self, artifact_id: str, tenant_id: str, workspace_id: str, source_id: str, artifact_type: str, artifact_reference: str, content_fingerprint: str, confidence_score: float, evidence_reference: str) -> dict[str, Any]:
		workspace = self._tenant_workspace_or_none(workspace_id, tenant_id)
		source = self._tenant_source_or_none(source_id, tenant_id)
		artifact_type = normalize_code(artifact_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_artifact", "workspace_present": workspace is not None, "source_present": source is not None, "workspace_source_authority_match": workspace is not None and source is not None and workspace.authority_id == source.authority_id, "artifact_type_supported": artifact_type in SUPPORTED_ARTIFACT_TYPES, "artifact_reference_present": present(artifact_reference), "fingerprint_present": present(content_fingerprint), "confidence_valid": bounded_score(confidence_score), "evidence_present": present(evidence_reference)})
		item = FusionArtifact(artifact_id, tenant_id, workspace_id, source_id, artifact_type, artifact_reference, content_fingerprint, float(confidence_score), evidence_reference)
		self.artifacts[self._tenant_key(tenant_id, artifact_id)] = item
		self._audit(tenant_id, "fusion_artifact_recorded", artifact_id)
		return item.to_dict()

	def record_correlation(self, correlation_id: str, tenant_id: str, artifact_id: str, correlation_type: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		artifact = self._tenant_artifact_or_none(artifact_id, tenant_id)
		correlation_type = normalize_code(correlation_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_correlation", "artifact_present": artifact is not None, "correlation_type_supported": correlation_type in SUPPORTED_CORRELATION_TYPES, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = FusionCorrelation(correlation_id, tenant_id, artifact_id, correlation_type, float(confidence_score), analyst_id, evidence_reference)
		self.correlations[self._tenant_key(tenant_id, correlation_id)] = item
		self._audit(tenant_id, "fusion_correlation_recorded", correlation_id)
		return item.to_dict()

	def record_hypothesis(self, hypothesis_id: str, tenant_id: str, correlation_id: str, hypothesis_type: str, claim_reference: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		correlation = self._tenant_correlation_or_none(correlation_id, tenant_id)
		hypothesis_type = normalize_code(hypothesis_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_hypothesis", "correlation_present": correlation is not None, "hypothesis_type_supported": hypothesis_type in SUPPORTED_HYPOTHESIS_TYPES, "claim_present": present(claim_reference), "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = FusionHypothesis(hypothesis_id, tenant_id, correlation_id, hypothesis_type, claim_reference, float(confidence_score), analyst_id, evidence_reference)
		self.hypotheses[self._tenant_key(tenant_id, hypothesis_id)] = item
		self._audit(tenant_id, "fusion_hypothesis_recorded", hypothesis_id)
		return item.to_dict()

	def record_assessment(self, assessment_id: str, tenant_id: str, hypothesis_id: str, assessment_type: str, risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		hypothesis = self._tenant_hypothesis_or_none(hypothesis_id, tenant_id)
		assessment_type = normalize_code(assessment_type)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_assessment", "hypothesis_present": hypothesis is not None, "assessment_type_supported": assessment_type in SUPPORTED_ASSESSMENT_TYPES, "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = FusionAssessment(assessment_id, tenant_id, hypothesis_id, assessment_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.assessments[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "fusion_assessment_recorded", assessment_id)
		return item.to_dict()

	def record_referral(self, referral_id: str, tenant_id: str, assessment_id: str, referral_type: str, recipient: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._tenant_assessment_or_none(assessment_id, tenant_id)
		referral_type = normalize_code(referral_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_referral", "assessment_present": assessment is not None, "referral_type_supported": referral_type in SUPPORTED_REFERRAL_TYPES, "recipient_present": present(recipient), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = FusionReferral(referral_id, tenant_id, assessment_id, referral_type, recipient, approval_reference, evidence_reference)
		self.referrals[self._tenant_key(tenant_id, referral_id)] = item
		self._audit(tenant_id, "fusion_referral_recorded", referral_id)
		return item.to_dict()

	def record_dissemination(self, dissemination_id: str, tenant_id: str, assessment_id: str, audience: str, release_marking: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._tenant_assessment_or_none(assessment_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_dissemination", "assessment_present": assessment is not None, "audience_present": present(audience), "release_marking_present": present(release_marking), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = FusionDissemination(dissemination_id, tenant_id, assessment_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "fusion_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = FusionReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "fusion_review_recorded", reference_id)
		return item.to_dict()

	def register_fusion_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_fusion_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = FusionAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "fusion_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, evidence_fabrication_scope: bool = False, source_tampering_scope: bool = False, privacy_bypass_scope: bool = False, unsupported_identity_resolution_scope: bool = False, autonomous_dissemination_scope: bool = False, unapproved_attribution_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "fusion_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "evidence_fabrication_scope": evidence_fabrication_scope, "source_tampering_scope": source_tampering_scope, "privacy_bypass_scope": privacy_bypass_scope, "unsupported_identity_resolution_scope": unsupported_identity_resolution_scope, "autonomous_dissemination_scope": autonomous_dissemination_scope, "unapproved_attribution_scope": unapproved_attribution_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "fusion_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.fusion.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "workspace_count": self._count(self.workspaces, tenant_id), "source_count": self._count(self.sources, tenant_id), "artifact_count": self._count(self.artifacts, tenant_id), "correlation_count": self._count(self.correlations, tenant_id), "hypothesis_count": self._count(self.hypotheses, tenant_id), "assessment_count": self._count(self.assessments, tenant_id), "referral_count": self._count(self.referrals, tenant_id), "dissemination_count": self._count(self.disseminations, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> FusionAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_workspace_or_none(self, item_id: str, tenant_id: str) -> FusionWorkspace | None:
		return self.workspaces.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> FusionSource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_artifact_or_none(self, item_id: str, tenant_id: str) -> FusionArtifact | None:
		return self.artifacts.get(self._tenant_key(tenant_id, item_id))

	def _tenant_correlation_or_none(self, item_id: str, tenant_id: str) -> FusionCorrelation | None:
		return self.correlations.get(self._tenant_key(tenant_id, item_id))

	def _tenant_hypothesis_or_none(self, item_id: str, tenant_id: str) -> FusionHypothesis | None:
		return self.hypotheses.get(self._tenant_key(tenant_id, item_id))

	def _tenant_assessment_or_none(self, item_id: str, tenant_id: str) -> FusionAssessment | None:
		return self.assessments.get(self._tenant_key(tenant_id, item_id))

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
		reasons = ", ".join(action.get("reason", action.get("rule", "fusion_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "fusion_policy_denied")


IntelFusionService = IntelligenceFusionService
