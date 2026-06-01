"""Executable service layer for APG Intelligence Analytics."""

from __future__ import annotations

from typing import Any

try:
	from .analytics_runtime import bounded_score, normalize_code, positive_int, present
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_DATASET_TYPES, SUPPORTED_FEATURE_TYPES, SUPPORTED_INSIGHT_TYPES, SUPPORTED_MODEL_TYPES, SUPPORTED_NARRATIVE_TYPES, SUPPORTED_RECOMMENDATION_TYPES, SUPPORTED_RETENTION_CLASSES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_RUN_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract
	from .models import AnalyticsAgent, AnalyticsAuthority, AnalyticsDashboard, AnalyticsDataset, AnalyticsFeatureSet, AnalyticsInsight, AnalyticsModel, AnalyticsNarrative, AnalyticsRecommendation, AnalyticsReview, AnalyticsRun, AnalyticsWorkspace
except ImportError:  # pragma: no cover
	from analytics_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_DATASET_TYPES, SUPPORTED_FEATURE_TYPES, SUPPORTED_INSIGHT_TYPES, SUPPORTED_MODEL_TYPES, SUPPORTED_NARRATIVE_TYPES, SUPPORTED_RECOMMENDATION_TYPES, SUPPORTED_RETENTION_CLASSES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_RUN_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import AnalyticsAgent, AnalyticsAuthority, AnalyticsDashboard, AnalyticsDataset, AnalyticsFeatureSet, AnalyticsInsight, AnalyticsModel, AnalyticsNarrative, AnalyticsRecommendation, AnalyticsReview, AnalyticsRun, AnalyticsWorkspace  # type: ignore


class IntelligenceAnalyticsService:
	"""Tenant-scoped intelligence analytics runtime for generated APG applications."""

	def __init__(self) -> None:
		self.authorities: dict[tuple[str, str], AnalyticsAuthority] = {}
		self.workspaces: dict[tuple[str, str], AnalyticsWorkspace] = {}
		self.datasets: dict[tuple[str, str], AnalyticsDataset] = {}
		self.feature_sets: dict[tuple[str, str], AnalyticsFeatureSet] = {}
		self.models: dict[tuple[str, str], AnalyticsModel] = {}
		self.runs: dict[tuple[str, str], AnalyticsRun] = {}
		self.insights: dict[tuple[str, str], AnalyticsInsight] = {}
		self.dashboards: dict[tuple[str, str], AnalyticsDashboard] = {}
		self.narratives: dict[tuple[str, str], AnalyticsNarrative] = {}
		self.recommendations: dict[tuple[str, str], AnalyticsRecommendation] = {}
		self.reviews: dict[tuple[str, str], AnalyticsReview] = {}
		self.agents: dict[tuple[str, str], AnalyticsAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = AnalyticsAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "analytics_authority_recorded", authority_id)
		return item.to_dict()

	def record_workspace(self, workspace_id: str, tenant_id: str, workspace_type: str, name: str, classification: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		workspace_type = normalize_code(workspace_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_workspace", "workspace_type_supported": workspace_type in SUPPORTED_WORKSPACE_TYPES, "workspace_name_present": present(name), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = AnalyticsWorkspace(workspace_id, tenant_id, workspace_type, name, classification, authority_id, evidence_reference)
		self.workspaces[self._tenant_key(tenant_id, workspace_id)] = item
		self._audit(tenant_id, "analytics_workspace_recorded", workspace_id)
		return item.to_dict()

	def register_dataset(self, dataset_id: str, tenant_id: str, workspace_id: str, dataset_type: str, source_reference: str, owner_id: str, lineage_reference: str, retention_class: str, evidence_reference: str) -> dict[str, Any]:
		workspace = self._tenant_workspace_or_none(workspace_id, tenant_id)
		dataset_type = normalize_code(dataset_type)
		retention_class = normalize_code(retention_class)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_dataset", "workspace_present": workspace is not None, "dataset_type_supported": dataset_type in SUPPORTED_DATASET_TYPES, "source_reference_present": present(source_reference), "owner_present": present(owner_id), "lineage_present": present(lineage_reference), "retention_supported": retention_class in SUPPORTED_RETENTION_CLASSES, "evidence_present": present(evidence_reference)})
		item = AnalyticsDataset(dataset_id, tenant_id, workspace_id, dataset_type, source_reference, owner_id, lineage_reference, retention_class, evidence_reference)
		self.datasets[self._tenant_key(tenant_id, dataset_id)] = item
		self._audit(tenant_id, "analytics_dataset_registered", dataset_id)
		return item.to_dict()

	def record_feature_set(self, feature_set_id: str, tenant_id: str, dataset_id: str, feature_type: str, feature_reference: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		dataset = self._tenant_dataset_or_none(dataset_id, tenant_id)
		feature_type = normalize_code(feature_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_feature_set", "dataset_present": dataset is not None, "feature_type_supported": feature_type in SUPPORTED_FEATURE_TYPES, "feature_reference_present": present(feature_reference), "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = AnalyticsFeatureSet(feature_set_id, tenant_id, dataset_id, feature_type, feature_reference, float(confidence_score), analyst_id, evidence_reference)
		self.feature_sets[self._tenant_key(tenant_id, feature_set_id)] = item
		self._audit(tenant_id, "analytics_feature_set_recorded", feature_set_id)
		return item.to_dict()

	def record_model(self, model_id: str, tenant_id: str, feature_set_id: str, model_type: str, objective: str, validation_reference: str, risk_level: str, evidence_reference: str) -> dict[str, Any]:
		feature_set = self._tenant_feature_set_or_none(feature_set_id, tenant_id)
		model_type = normalize_code(model_type)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_model", "feature_set_present": feature_set is not None, "model_type_supported": model_type in SUPPORTED_MODEL_TYPES, "objective_present": present(objective), "validation_present": present(validation_reference), "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "evidence_present": present(evidence_reference)})
		item = AnalyticsModel(model_id, tenant_id, feature_set_id, model_type, objective, validation_reference, risk_level, evidence_reference)
		self.models[self._tenant_key(tenant_id, model_id)] = item
		self._audit(tenant_id, "analytics_model_recorded", model_id)
		return item.to_dict()

	def record_run(self, run_id: str, tenant_id: str, model_id: str, run_type: str, result_reference: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		model = self._tenant_model_or_none(model_id, tenant_id)
		run_type = normalize_code(run_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_run", "model_present": model is not None, "run_type_supported": run_type in SUPPORTED_RUN_TYPES, "result_reference_present": present(result_reference), "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = AnalyticsRun(run_id, tenant_id, model_id, run_type, result_reference, float(confidence_score), analyst_id, evidence_reference)
		self.runs[self._tenant_key(tenant_id, run_id)] = item
		self._audit(tenant_id, "analytics_run_recorded", run_id)
		return item.to_dict()

	def record_insight(self, insight_id: str, tenant_id: str, run_id: str, insight_type: str, claim_reference: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		run = self._tenant_run_or_none(run_id, tenant_id)
		insight_type = normalize_code(insight_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_insight", "run_present": run is not None, "insight_type_supported": insight_type in SUPPORTED_INSIGHT_TYPES, "claim_present": present(claim_reference), "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = AnalyticsInsight(insight_id, tenant_id, run_id, insight_type, claim_reference, float(confidence_score), analyst_id, evidence_reference)
		self.insights[self._tenant_key(tenant_id, insight_id)] = item
		self._audit(tenant_id, "analytics_insight_recorded", insight_id)
		return item.to_dict()

	def record_dashboard(self, dashboard_id: str, tenant_id: str, insight_id: str, name: str, audience: str, release_marking: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		insight = self._tenant_insight_or_none(insight_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_dashboard", "insight_present": insight is not None, "dashboard_name_present": present(name), "audience_present": present(audience), "release_marking_present": present(release_marking), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = AnalyticsDashboard(dashboard_id, tenant_id, insight_id, name, audience, release_marking, approval_reference, evidence_reference)
		self.dashboards[self._tenant_key(tenant_id, dashboard_id)] = item
		self._audit(tenant_id, "analytics_dashboard_recorded", dashboard_id)
		return item.to_dict()

	def record_narrative(self, narrative_id: str, tenant_id: str, insight_id: str, narrative_type: str, summary_reference: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		insight = self._tenant_insight_or_none(insight_id, tenant_id)
		narrative_type = normalize_code(narrative_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_narrative", "insight_present": insight is not None, "narrative_type_supported": narrative_type in SUPPORTED_NARRATIVE_TYPES, "summary_present": present(summary_reference), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = AnalyticsNarrative(narrative_id, tenant_id, insight_id, narrative_type, summary_reference, approval_reference, evidence_reference)
		self.narratives[self._tenant_key(tenant_id, narrative_id)] = item
		self._audit(tenant_id, "analytics_narrative_recorded", narrative_id)
		return item.to_dict()

	def record_recommendation(self, recommendation_id: str, tenant_id: str, insight_id: str, recommendation_type: str, action_reference: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		insight = self._tenant_insight_or_none(insight_id, tenant_id)
		recommendation_type = normalize_code(recommendation_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_recommendation", "insight_present": insight is not None, "recommendation_type_supported": recommendation_type in SUPPORTED_RECOMMENDATION_TYPES, "action_present": present(action_reference), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = AnalyticsRecommendation(recommendation_id, tenant_id, insight_id, recommendation_type, action_reference, approval_reference, evidence_reference)
		self.recommendations[self._tenant_key(tenant_id, recommendation_id)] = item
		self._audit(tenant_id, "analytics_recommendation_recorded", recommendation_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = AnalyticsReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "analytics_review_recorded", reference_id)
		return item.to_dict()

	def register_analytics_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_analytics_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = AnalyticsAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "analytics_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, hallucinated_insight_scope: bool = False, training_data_leakage_scope: bool = False, privacy_bypass_scope: bool = False, unsupported_automated_decision_scope: bool = False, unapproved_model_deployment_scope: bool = False, autonomous_dissemination_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "analytics_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "hallucinated_insight_scope": hallucinated_insight_scope, "training_data_leakage_scope": training_data_leakage_scope, "privacy_bypass_scope": privacy_bypass_scope, "unsupported_automated_decision_scope": unsupported_automated_decision_scope, "unapproved_model_deployment_scope": unapproved_model_deployment_scope, "autonomous_dissemination_scope": autonomous_dissemination_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "analytics_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.analytics.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "workspace_count": self._count(self.workspaces, tenant_id), "dataset_count": self._count(self.datasets, tenant_id), "feature_set_count": self._count(self.feature_sets, tenant_id), "model_count": self._count(self.models, tenant_id), "run_count": self._count(self.runs, tenant_id), "insight_count": self._count(self.insights, tenant_id), "dashboard_count": self._count(self.dashboards, tenant_id), "narrative_count": self._count(self.narratives, tenant_id), "recommendation_count": self._count(self.recommendations, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> AnalyticsAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_workspace_or_none(self, item_id: str, tenant_id: str) -> AnalyticsWorkspace | None:
		return self.workspaces.get(self._tenant_key(tenant_id, item_id))

	def _tenant_dataset_or_none(self, item_id: str, tenant_id: str) -> AnalyticsDataset | None:
		return self.datasets.get(self._tenant_key(tenant_id, item_id))

	def _tenant_feature_set_or_none(self, item_id: str, tenant_id: str) -> AnalyticsFeatureSet | None:
		return self.feature_sets.get(self._tenant_key(tenant_id, item_id))

	def _tenant_model_or_none(self, item_id: str, tenant_id: str) -> AnalyticsModel | None:
		return self.models.get(self._tenant_key(tenant_id, item_id))

	def _tenant_run_or_none(self, item_id: str, tenant_id: str) -> AnalyticsRun | None:
		return self.runs.get(self._tenant_key(tenant_id, item_id))

	def _tenant_insight_or_none(self, item_id: str, tenant_id: str) -> AnalyticsInsight | None:
		return self.insights.get(self._tenant_key(tenant_id, item_id))

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
		reasons = ", ".join(action.get("reason", action.get("rule", "analytics_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "analytics_policy_denied")


IntelAnalyticsService = IntelligenceAnalyticsService
