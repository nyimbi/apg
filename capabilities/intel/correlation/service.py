"""Executable service layer for APG Data Correlation."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_CLUSTER_TYPES, SUPPORTED_DECISION_TYPES, SUPPORTED_ENTITY_TYPES, SUPPORTED_OBSERVATION_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RULE_TYPES, SUPPORTED_RUN_TYPES, SUPPORTED_SOURCE_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract
	from .correlation_runtime import bounded_score, normalize_code, positive_int, present
	from .models import CorrelationAgent, CorrelationAuthority, CorrelationCluster, CorrelationDecision, CorrelationEntity, CorrelationObservation, CorrelationReferral, CorrelationReview, CorrelationRule, CorrelationRun, CorrelationSource, CorrelationWorkspace
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_CLUSTER_TYPES, SUPPORTED_DECISION_TYPES, SUPPORTED_ENTITY_TYPES, SUPPORTED_OBSERVATION_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RULE_TYPES, SUPPORTED_RUN_TYPES, SUPPORTED_SOURCE_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from correlation_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore
	from models import CorrelationAgent, CorrelationAuthority, CorrelationCluster, CorrelationDecision, CorrelationEntity, CorrelationObservation, CorrelationReferral, CorrelationReview, CorrelationRule, CorrelationRun, CorrelationSource, CorrelationWorkspace  # type: ignore


class DataCorrelationService:
	"""Tenant-scoped data-correlation runtime for generated APG applications."""

	def __init__(self) -> None:
		self.authorities: dict[tuple[str, str], CorrelationAuthority] = {}
		self.workspaces: dict[tuple[str, str], CorrelationWorkspace] = {}
		self.sources: dict[tuple[str, str], CorrelationSource] = {}
		self.entities: dict[tuple[str, str], CorrelationEntity] = {}
		self.observations: dict[tuple[str, str], CorrelationObservation] = {}
		self.rules: dict[tuple[str, str], CorrelationRule] = {}
		self.runs: dict[tuple[str, str], CorrelationRun] = {}
		self.clusters: dict[tuple[str, str], CorrelationCluster] = {}
		self.decisions: dict[tuple[str, str], CorrelationDecision] = {}
		self.referrals: dict[tuple[str, str], CorrelationReferral] = {}
		self.reviews: dict[tuple[str, str], CorrelationReview] = {}
		self.agents: dict[tuple[str, str], CorrelationAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = CorrelationAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "correlation_authority_recorded", authority_id)
		return item.to_dict()

	def record_workspace(self, workspace_id: str, tenant_id: str, workspace_type: str, name: str, classification: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		workspace_type = normalize_code(workspace_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_workspace", "workspace_type_supported": workspace_type in SUPPORTED_WORKSPACE_TYPES, "workspace_name_present": present(name), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = CorrelationWorkspace(workspace_id, tenant_id, workspace_type, name, classification, authority_id, evidence_reference)
		self.workspaces[self._tenant_key(tenant_id, workspace_id)] = item
		self._audit(tenant_id, "correlation_workspace_recorded", workspace_id)
		return item.to_dict()

	def register_source(self, source_id: str, tenant_id: str, workspace_id: str, source_type: str, source_reference: str, custodian_id: str, lineage_reference: str, evidence_reference: str) -> dict[str, Any]:
		workspace = self._tenant_workspace_or_none(workspace_id, tenant_id)
		source_type = normalize_code(source_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_source", "workspace_present": workspace is not None, "source_type_supported": source_type in SUPPORTED_SOURCE_TYPES, "source_reference_present": present(source_reference), "custodian_present": present(custodian_id), "lineage_present": present(lineage_reference), "evidence_present": present(evidence_reference)})
		item = CorrelationSource(source_id, tenant_id, workspace_id, source_type, source_reference, custodian_id, lineage_reference, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "correlation_source_registered", source_id)
		return item.to_dict()

	def record_entity(self, entity_id: str, tenant_id: str, source_id: str, entity_type: str, entity_reference: str, confidence_score: float, evidence_reference: str) -> dict[str, Any]:
		source = self._tenant_source_or_none(source_id, tenant_id)
		entity_type = normalize_code(entity_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_entity", "source_present": source is not None, "entity_type_supported": entity_type in SUPPORTED_ENTITY_TYPES, "entity_reference_present": present(entity_reference), "confidence_valid": bounded_score(confidence_score), "evidence_present": present(evidence_reference)})
		item = CorrelationEntity(entity_id, tenant_id, source_id, entity_type, entity_reference, float(confidence_score), evidence_reference)
		self.entities[self._tenant_key(tenant_id, entity_id)] = item
		self._audit(tenant_id, "correlation_entity_recorded", entity_id)
		return item.to_dict()

	def record_observation(self, observation_id: str, tenant_id: str, entity_id: str, observation_type: str, observation_reference: str, observed_at: str, confidence_score: float, evidence_reference: str) -> dict[str, Any]:
		entity = self._tenant_entity_or_none(entity_id, tenant_id)
		observation_type = normalize_code(observation_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_observation", "entity_present": entity is not None, "observation_type_supported": observation_type in SUPPORTED_OBSERVATION_TYPES, "observation_reference_present": present(observation_reference), "observed_at_present": present(observed_at), "confidence_valid": bounded_score(confidence_score), "evidence_present": present(evidence_reference)})
		item = CorrelationObservation(observation_id, tenant_id, entity_id, observation_type, observation_reference, observed_at, float(confidence_score), evidence_reference)
		self.observations[self._tenant_key(tenant_id, observation_id)] = item
		self._audit(tenant_id, "correlation_observation_recorded", observation_id)
		return item.to_dict()

	def record_rule(self, rule_id: str, tenant_id: str, workspace_id: str, rule_type: str, rule_reference: str, threshold_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		workspace = self._tenant_workspace_or_none(workspace_id, tenant_id)
		rule_type = normalize_code(rule_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_rule", "workspace_present": workspace is not None, "rule_type_supported": rule_type in SUPPORTED_RULE_TYPES, "rule_reference_present": present(rule_reference), "threshold_valid": bounded_score(threshold_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = CorrelationRule(rule_id, tenant_id, workspace_id, rule_type, rule_reference, float(threshold_score), analyst_id, evidence_reference)
		self.rules[self._tenant_key(tenant_id, rule_id)] = item
		self._audit(tenant_id, "correlation_rule_recorded", rule_id)
		return item.to_dict()

	def record_run(self, run_id: str, tenant_id: str, rule_id: str, run_type: str, result_reference: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		rule = self._tenant_rule_or_none(rule_id, tenant_id)
		run_type = normalize_code(run_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_run", "rule_present": rule is not None, "run_type_supported": run_type in SUPPORTED_RUN_TYPES, "result_reference_present": present(result_reference), "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = CorrelationRun(run_id, tenant_id, rule_id, run_type, result_reference, float(confidence_score), analyst_id, evidence_reference)
		self.runs[self._tenant_key(tenant_id, run_id)] = item
		self._audit(tenant_id, "correlation_run_recorded", run_id)
		return item.to_dict()

	def record_cluster(self, cluster_id: str, tenant_id: str, run_id: str, cluster_type: str, cluster_reference: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		run = self._tenant_run_or_none(run_id, tenant_id)
		cluster_type = normalize_code(cluster_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_cluster", "run_present": run is not None, "cluster_type_supported": cluster_type in SUPPORTED_CLUSTER_TYPES, "cluster_reference_present": present(cluster_reference), "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = CorrelationCluster(cluster_id, tenant_id, run_id, cluster_type, cluster_reference, float(confidence_score), analyst_id, evidence_reference)
		self.clusters[self._tenant_key(tenant_id, cluster_id)] = item
		self._audit(tenant_id, "correlation_cluster_recorded", cluster_id)
		return item.to_dict()

	def record_decision(self, decision_id: str, tenant_id: str, cluster_id: str, decision_type: str, rationale_reference: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		cluster = self._tenant_cluster_or_none(cluster_id, tenant_id)
		decision_type = normalize_code(decision_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_decision", "cluster_present": cluster is not None, "decision_type_supported": decision_type in SUPPORTED_DECISION_TYPES, "rationale_present": present(rationale_reference), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = CorrelationDecision(decision_id, tenant_id, cluster_id, decision_type, rationale_reference, approval_reference, evidence_reference)
		self.decisions[self._tenant_key(tenant_id, decision_id)] = item
		self._audit(tenant_id, "correlation_decision_recorded", decision_id)
		return item.to_dict()

	def record_referral(self, referral_id: str, tenant_id: str, decision_id: str, referral_type: str, recipient: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		decision = self._tenant_decision_or_none(decision_id, tenant_id)
		referral_type = normalize_code(referral_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_referral", "decision_present": decision is not None, "referral_type_supported": referral_type in SUPPORTED_REFERRAL_TYPES, "recipient_present": present(recipient), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = CorrelationReferral(referral_id, tenant_id, decision_id, referral_type, recipient, approval_reference, evidence_reference)
		self.referrals[self._tenant_key(tenant_id, referral_id)] = item
		self._audit(tenant_id, "correlation_referral_recorded", referral_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = CorrelationReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "correlation_review_recorded", reference_id)
		return item.to_dict()

	def register_correlation_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_correlation_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = CorrelationAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "correlation_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, unapproved_identity_merge_scope: bool = False, source_tampering_scope: bool = False, privacy_bypass_scope: bool = False, evidence_fabrication_scope: bool = False, autonomous_referral_scope: bool = False, unreviewed_high_impact_match_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "correlation_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "unapproved_identity_merge_scope": unapproved_identity_merge_scope, "source_tampering_scope": source_tampering_scope, "privacy_bypass_scope": privacy_bypass_scope, "evidence_fabrication_scope": evidence_fabrication_scope, "autonomous_referral_scope": autonomous_referral_scope, "unreviewed_high_impact_match_scope": unreviewed_high_impact_match_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "correlation_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.correlation.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "workspace_count": self._count(self.workspaces, tenant_id), "source_count": self._count(self.sources, tenant_id), "entity_count": self._count(self.entities, tenant_id), "observation_count": self._count(self.observations, tenant_id), "rule_count": self._count(self.rules, tenant_id), "run_count": self._count(self.runs, tenant_id), "cluster_count": self._count(self.clusters, tenant_id), "decision_count": self._count(self.decisions, tenant_id), "referral_count": self._count(self.referrals, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> CorrelationAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_workspace_or_none(self, item_id: str, tenant_id: str) -> CorrelationWorkspace | None:
		return self.workspaces.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> CorrelationSource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_entity_or_none(self, item_id: str, tenant_id: str) -> CorrelationEntity | None:
		return self.entities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_rule_or_none(self, item_id: str, tenant_id: str) -> CorrelationRule | None:
		return self.rules.get(self._tenant_key(tenant_id, item_id))

	def _tenant_run_or_none(self, item_id: str, tenant_id: str) -> CorrelationRun | None:
		return self.runs.get(self._tenant_key(tenant_id, item_id))

	def _tenant_cluster_or_none(self, item_id: str, tenant_id: str) -> CorrelationCluster | None:
		return self.clusters.get(self._tenant_key(tenant_id, item_id))

	def _tenant_decision_or_none(self, item_id: str, tenant_id: str) -> CorrelationDecision | None:
		return self.decisions.get(self._tenant_key(tenant_id, item_id))

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
		reasons = ", ".join(action.get("reason", action.get("rule", "correlation_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "correlation_policy_denied")


IntelCorrelationService = DataCorrelationService
