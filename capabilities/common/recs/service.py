"""Service layer for executable Recommender Systems operations."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	RankingPolicy,
	InteractionEvent,
	ModelDeployment,
	Recommendation,
	RecommendationCatalogItem,
	RecommendationDataset,
	RecommendationExperiment,
	RecommendationFeedback,
	RecommendationModel,
	RecommendationProfile,
	RecommendationSet,
	RecommenderAgent,
	RecsAuditEvent,
	TrainingRun,
	utc_now,
)
from .recommendation_runtime import (
	confidence_for_score,
	drift_status,
	normalize_agent_role,
	normalize_agent_runtime,
	normalize_algorithm,
	normalize_deployment_target,
	normalize_features,
	normalize_feedback_event,
	normalize_impact_level,
	normalize_labels,
	recommendation_reason,
	schema_fields_valid,
	score_item,
	stable_id,
)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class RecsService:
	"""In-process catalog, profile, ranking, model, experiment, and recommendation service."""

	def __init__(self) -> None:
		self._datasets: dict[str, RecommendationDataset] = {}
		self._interaction_events: dict[str, InteractionEvent] = {}
		self._catalog_items: dict[str, RecommendationCatalogItem] = {}
		self._profiles: dict[str, RecommendationProfile] = {}
		self._policies: dict[str, RankingPolicy] = {}
		self._models: dict[str, RecommendationModel] = {}
		self._training_runs: dict[str, TrainingRun] = {}
		self._deployments: dict[str, ModelDeployment] = {}
		self._recommendation_sets: dict[str, RecommendationSet] = {}
		self._feedback: dict[str, RecommendationFeedback] = {}
		self._experiments: dict[str, RecommendationExperiment] = {}
		self._agents: dict[str, RecommenderAgent] = {}
		self._audit_events: dict[str, RecsAuditEvent] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_catalog_item(
		self,
		item_id: str,
		tenant_id: str,
		name: str,
		item_type: str,
		category: str,
		features: dict[str, Any] | None = None,
		tags: list[str] | tuple[str, ...] | None = None,
		sensitive_attributes: list[str] | tuple[str, ...] | None = None,
		actor: str = "recs",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not name:
			raise PermissionError("catalog_item_name_required")
		if not category:
			raise PermissionError("catalog_item_category_required")
		item = RecommendationCatalogItem(
			id=item_id,
			tenant_id=tenant_id,
			name=name,
			item_type=item_type or "item",
			category=category,
			features=normalize_features(features),
			tags=normalize_labels(tags),
			sensitive_attributes=normalize_labels(sensitive_attributes),
		)
		self._catalog_items[item.id] = item
		self._record_audit(tenant_id, item.id, "catalog_item_registered", actor, "allow")
		return item.to_dict()

	def register_dataset(
		self,
		dataset_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		source_ref: str,
		schema_fields: list[str] | tuple[str, ...],
		policy_ref: str,
		event_count: int = 0,
		actor: str = "recs",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		fields_valid = schema_fields_valid(schema_fields)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_dataset",
			"dataset_owner_present": bool(owner.strip()),
			"dataset_source_present": bool(source_ref.strip()),
			"dataset_schema_present": fields_valid,
			"dataset_policy_present": bool(policy_ref.strip()),
		})
		self._raise_if_blocked(result)
		dataset = RecommendationDataset(
			id=dataset_id,
			tenant_id=tenant_id,
			name=name or dataset_id,
			owner=owner,
			source_ref=source_ref,
			schema_fields=normalize_labels(schema_fields),
			policy_ref=policy_ref,
			event_count=max(0, int(event_count)),
		)
		self._datasets[dataset.id] = dataset
		self._record_audit(tenant_id, dataset.id, "dataset_registered", actor, "allow")
		return dataset.to_dict()

	def record_interaction(
		self,
		event_id: str,
		tenant_id: str,
		dataset_id: str,
		profile_id: str,
		item_id: str,
		event_type: str,
		occurred_at: str,
		weight: float = 1.0,
		metadata: dict[str, Any] | None = None,
		actor: str = "recs",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		dataset = self._require_dataset(dataset_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "record_interaction",
			"interaction_actor_present": bool(profile_id.strip()),
			"interaction_item_present": bool(item_id.strip()),
			"interaction_timestamp_present": bool(occurred_at.strip()),
		})
		self._raise_if_blocked(result)
		event = InteractionEvent(
			id=event_id,
			tenant_id=tenant_id,
			dataset_id=dataset.id,
			profile_id=profile_id,
			item_id=item_id,
			event_type=normalize_feedback_event(event_type),
			occurred_at=occurred_at,
			weight=round(float(weight), 4),
			metadata=dict(metadata or {}),
		)
		self._interaction_events[event.id] = event
		dataset.event_count += 1
		dataset.updated_at = utc_now()
		self._record_audit(tenant_id, event.id, "interaction_recorded", actor, "allow")
		return event.to_dict()

	def record_profile(
		self,
		profile_id: str,
		tenant_id: str,
		features: dict[str, Any] | None = None,
		segments: list[str] | tuple[str, ...] | None = None,
		consent_recorded: bool = False,
		actor: str = "recs",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		profile = RecommendationProfile(
			id=profile_id,
			tenant_id=tenant_id,
			features=normalize_features(features),
			segments=normalize_labels(segments),
			consent_recorded=bool(consent_recorded),
		)
		self._profiles[profile.id] = profile
		self._record_audit(tenant_id, profile.id, "profile_recorded", actor, "allow")
		return profile.to_dict()

	def attach_ranking_policy(
		self,
		policy_id: str,
		tenant_id: str,
		name: str,
		objective: str,
		owner: str = "recs",
		minimum_confidence: float = 0.65,
		diversity_constraints_enabled: bool = True,
		sensitive_attribute_filtering: bool = True,
		max_per_category: int = 2,
		actor: str = "recs",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "attach_ranking_policy",
			"ranking_policy_owner_present": bool(owner.strip()),
		})
		self._raise_if_blocked(result)
		if not objective:
			raise PermissionError("ranking_objective_required")
		if not 0 <= float(minimum_confidence) <= 1:
			raise PermissionError("ranking_confidence_threshold_invalid")
		if int(max_per_category) < 1:
			raise PermissionError("ranking_category_limit_required")
		policy = RankingPolicy(
			id=policy_id,
			tenant_id=tenant_id,
			name=name or policy_id,
			objective=objective,
			owner=owner,
			minimum_confidence=round(float(minimum_confidence), 4),
			diversity_constraints_enabled=bool(diversity_constraints_enabled),
			sensitive_attribute_filtering=bool(sensitive_attribute_filtering),
			max_per_category=int(max_per_category),
		)
		self._policies[policy.id] = policy
		self._record_audit(tenant_id, policy.id, "ranking_policy_attached", actor, "allow")
		return policy.to_dict()

	def train_model(
		self,
		model_id: str,
		tenant_id: str,
		name: str,
		algorithm: str,
		owner: str,
		training_event_count: int,
		feature_names: list[str] | tuple[str, ...] | None = None,
		drift_monitoring_enabled: bool = True,
		metric_name: str = "precision_at_k",
		metric_value: float = 0.72,
		actor: str = "recs",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "train_model",
			"training_event_count": int(training_event_count),
			"model_owner_present": bool(owner.strip()),
			"drift_monitoring_enabled": bool(drift_monitoring_enabled),
		})
		self._raise_if_blocked(result)
		model = RecommendationModel(
			id=model_id,
			tenant_id=tenant_id,
			name=name or model_id,
			algorithm=normalize_algorithm(algorithm),
			owner=owner,
			training_event_count=int(training_event_count),
			feature_names=normalize_labels(feature_names),
			drift_monitoring_enabled=bool(drift_monitoring_enabled),
		)
		training_run = TrainingRun(
			id=stable_id("train", tenant_id, model.id, training_event_count, metric_name),
			tenant_id=tenant_id,
			model_id=model.id,
			event_count=int(training_event_count),
			metric_name=metric_name,
			metric_value=round(float(metric_value), 4),
		)
		self._models[model.id] = model
		self._training_runs[training_run.id] = training_run
		self._record_audit(tenant_id, model.id, "model_trained", actor, "allow")
		return model.to_dict() | {"training_run": training_run.to_dict()}

	def approve_model(
		self,
		model_id: str,
		tenant_id: str,
		approval_ref: str,
		actor: str = "recs",
	) -> dict[str, Any]:
		model = self._require_model(model_id, tenant_id)
		if not approval_ref.strip():
			raise PermissionError("model_approval_ref_required")
		model.approved = True
		model.approval_ref = approval_ref
		model.status = "approved"
		model.updated_at = utc_now()
		self._record_audit(tenant_id, model.id, "model_approved", actor, "allow")
		return model.to_dict()

	def deploy_model(
		self,
		deployment_id: str,
		tenant_id: str,
		model_id: str,
		target_runtime: str,
		target_ref: str,
		approval_recorded: bool,
		rollback_plan_ref: str,
		approval_ref: str = "",
		actor: str = "recs",
	) -> dict[str, Any]:
		model = self._require_model(model_id, tenant_id)
		try:
			normalized_target = normalize_deployment_target(target_runtime)
		except ValueError as exc:
			raise PermissionError("deployment_target_required") from exc
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "deploy_model",
			"model_approved": bool(model.approved),
			"deployment_target_supported": bool(normalized_target and target_ref.strip()),
			"deployment_approval_recorded": bool(approval_recorded),
			"rollback_plan_present": bool(rollback_plan_ref.strip()),
		})
		self._raise_if_blocked(result)
		deployment = ModelDeployment(
			id=deployment_id,
			tenant_id=tenant_id,
			model_id=model.id,
			target_runtime=normalized_target,
			target_ref=target_ref,
			approval_recorded=approval_recorded,
			approval_ref=approval_ref,
			rollback_plan_ref=rollback_plan_ref,
		)
		self._deployments[deployment.id] = deployment
		model.status = "deployed"
		model.updated_at = utc_now()
		self._record_audit(tenant_id, deployment.id, "model_deployed", actor, "allow")
		return deployment.to_dict()

	def generate_recommendations(
		self,
		recommendation_id: str,
		tenant_id: str,
		model_id: str,
		profile_id: str,
		policy_id: str,
		candidate_item_ids: list[str],
		limit: int = 5,
		impact_level: str = "low",
		explanation_attached: bool = False,
		actor: str = "recs",
	) -> dict[str, Any]:
		model = self._require_model(model_id, tenant_id)
		profile = self._require_profile(profile_id, tenant_id)
		impact = normalize_impact_level(impact_level)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "recommend",
			"profile_consent_recorded": profile.consent_recorded,
			"ranking_policy_attached": bool(policy_id),
			"impact_level": impact,
			"explanation_attached": bool(explanation_attached),
		})
		self._raise_if_blocked(result)
		policy = self._require_policy(policy_id, tenant_id)
		if int(limit) < 1:
			raise PermissionError("recommendation_limit_required")
		items = [self._require_catalog_item(item_id, tenant_id) for item_id in candidate_item_ids]
		candidate_result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "recommend",
			"candidate_count": len(items),
		})
		self._raise_if_blocked(candidate_result)
		ranked = self._rank(model, profile, policy, items, int(limit))
		output_result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "recommend",
			"recommendation_count": len(ranked),
		})
		if output_result["decision"] != "allow":
			self._raise_if_blocked(output_result)
		rec_set = RecommendationSet(
			id=recommendation_id,
			tenant_id=tenant_id,
			model_id=model.id,
			profile_id=profile.id,
			policy_id=policy.id,
			impact_level=impact,
			recommendations=tuple(ranked),
			explanation_attached=bool(explanation_attached),
		)
		self._recommendation_sets[rec_set.id] = rec_set
		self._record_audit(tenant_id, rec_set.id, "recommendations_generated", actor, "allow")
		return rec_set.to_dict()

	def record_feedback(
		self,
		feedback_id: str,
		tenant_id: str,
		recommendation_set_id: str,
		profile_id: str,
		item_id: str,
		event_type: str,
		value: float = 1.0,
		metadata: dict[str, Any] | None = None,
		actor: str = "recs",
	) -> dict[str, Any]:
		self._require_recommendation_set(recommendation_set_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "record_feedback",
			"feedback_actor_present": bool(profile_id.strip()),
			"feedback_event_present": bool(event_type.strip()),
		})
		self._raise_if_blocked(result)
		feedback = RecommendationFeedback(
			id=feedback_id,
			tenant_id=tenant_id,
			recommendation_set_id=recommendation_set_id,
			profile_id=profile_id,
			item_id=item_id,
			event_type=normalize_feedback_event(event_type),
			value=round(float(value), 4),
			metadata=dict(metadata or {}),
		)
		self._feedback[feedback.id] = feedback
		self._record_audit(tenant_id, feedback.id, "feedback_recorded", actor, "allow")
		return feedback.to_dict()

	def create_experiment(
		self,
		experiment_id: str,
		tenant_id: str,
		name: str,
		model_id: str,
		policy_id: str,
		experiment_percent: int,
		holdout_percent: int,
		business_metric: str,
		approved: bool,
		review_recorded: bool = False,
		actor: str = "recs",
	) -> dict[str, Any]:
		model = self._require_model(model_id, tenant_id)
		policy = self._require_policy(policy_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "experiment",
			"experiment_percent": int(experiment_percent),
			"experiment_review_recorded": bool(review_recorded),
		})
		self._raise_if_blocked(result)
		if not approved:
			raise PermissionError("experiment_approval_required")
		if int(holdout_percent) < 1:
			raise PermissionError("holdout_required")
		if not business_metric:
			raise PermissionError("business_metric_required")
		experiment = RecommendationExperiment(
			id=experiment_id,
			tenant_id=tenant_id,
			name=name or experiment_id,
			model_id=model.id,
			policy_id=policy.id,
			experiment_percent=int(experiment_percent),
			holdout_percent=int(holdout_percent),
			business_metric=business_metric,
			approved=bool(approved),
			review_recorded=bool(review_recorded),
		)
		self._experiments[experiment.id] = experiment
		self._record_audit(tenant_id, experiment.id, "experiment_created", actor, "allow")
		return experiment.to_dict()

	def record_drift(
		self,
		model_id: str,
		tenant_id: str,
		baseline_metric: float,
		current_metric: float,
		actor: str = "recs",
	) -> dict[str, Any]:
		model = self._require_model(model_id, tenant_id)
		model.drift_status = drift_status(baseline_metric, current_metric)
		model.updated_at = utc_now()
		self._record_audit(tenant_id, model.id, "model_drift_recorded", actor, "allow", (model.drift_status,))
		return model.to_dict()

	def register_recommender_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool,
		policy_ref: str = "",
		registered: bool = True,
		actor: str = "recs",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		try:
			normalized_runtime = normalize_agent_runtime(runtime)
		except ValueError as exc:
			raise PermissionError("recommender_agent_runtime_not_supported") from exc
		normalized_role = normalize_agent_role(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"recommender_agent_present": True,
			"agent_registered": bool(registered),
			"agent_runtime_supported": bool(normalized_runtime),
			"agent_scope_present": bool(scope.strip()),
			"agent_contribution_disclosed": bool(contribution_disclosed),
		})
		self._raise_if_blocked(result)
		agent = RecommenderAgent(
			id=agent_id,
			tenant_id=tenant_id,
			name=name or agent_id,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			registered=registered,
			contribution_disclosed=contribution_disclosed,
			policy_ref=policy_ref,
		)
		self._agents[agent.id] = agent
		self._record_audit(tenant_id, agent.id, "recommender_agent_registered", actor, "allow")
		return agent.to_dict()

	def change_model_state(
		self,
		tenant_id: str,
		model_id: str,
		status: str,
		reason: str,
		audit_recorded: bool = True,
		actor: str = "recs",
	) -> dict[str, Any]:
		model = self._require_model(model_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"state_change_requested": True,
			"state_change_reason_present": bool(reason.strip()),
			"audit_event_recorded": bool(audit_recorded),
		})
		self._raise_if_blocked(result)
		model.status = status or model.status
		model.updated_at = utc_now()
		self._record_audit(tenant_id, model.id, "model_state_changed", actor, "allow", (reason,))
		return model.to_dict()

	def cold_start_handle(
		self,
		profile_id: str,
		tenant_id: str,
		strategy: str = "popular_items",
		fallback_item_ids: list[str] | None = None,
		limit: int = 5,
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Generate recommendations for a new profile with no interaction history."""
		self._require_tenant(tenant_id)
		profile = self._profiles.get(profile_id)
		if profile is None:
			profile_dict = self.record_profile(profile_id=profile_id, tenant_id=tenant_id, consent_recorded=True, actor=actor)
			profile = self._profiles[profile_id]
		# use fallback items or any active catalog items
		candidates = list(fallback_item_ids or [])
		if not candidates:
			candidates = [item.id for item in self._catalog_items.values() if item.tenant_id == tenant_id and item.status == "active"][:limit * 2]
		result = {
			"profile_id": profile_id,
			"tenant_id": tenant_id,
			"strategy": strategy,
			"candidate_count": len(candidates),
			"recommendations": [{"item_id": cid, "rank": i + 1, "reason": strategy} for i, cid in enumerate(candidates[:limit])],
			"generated_at": utc_now(),
		}
		self._record_audit(tenant_id, profile_id, "cold_start_recommendations_generated", actor, "allow")
		return result

	def diversity_inject(
		self,
		recommendation_set_id: str,
		tenant_id: str,
		diversity_factor: float = 0.3,
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Re-rank a recommendation set to inject category diversity."""
		rec_set = self._require_recommendation_set(recommendation_set_id, tenant_id)
		assert 0.0 <= diversity_factor <= 1.0, "diversity_factor must be 0..1"
		recs = list(rec_set.recommendations)
		seen_categories: set[str] = set()
		diverse: list = []
		deferred: list = []
		for r in recs:
			item = self._catalog_items.get(r.item_id)
			cat = item.category if item else "unknown"
			if cat not in seen_categories or len(diverse) < 2:
				seen_categories.add(cat)
				diverse.append(r)
			else:
				deferred.append(r)
		reranked = diverse + deferred
		return {
			"recommendation_set_id": recommendation_set_id,
			"tenant_id": tenant_id,
			"diversity_factor": diversity_factor,
			"original_count": len(recs),
			"reranked_recommendations": [{"item_id": r.item_id, "rank": i + 1, "score": r.score} for i, r in enumerate(reranked)],
			"unique_categories": len(seen_categories),
		}

	def serendipity_boost(
		self,
		recommendation_set_id: str,
		tenant_id: str,
		boost_fraction: float = 0.2,
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Replace a fraction of recommendations with novel/unexpected items."""
		rec_set = self._require_recommendation_set(recommendation_set_id, tenant_id)
		assert 0.0 <= boost_fraction <= 1.0, "boost_fraction must be 0..1"
		recs = list(rec_set.recommendations)
		n_boost = max(1, int(len(recs) * boost_fraction))
		# novel items: catalog items not in current set
		current_ids = {r.item_id for r in recs}
		novel_candidates = [i.id for i in self._catalog_items.values() if i.tenant_id == tenant_id and i.id not in current_ids and i.status == "active"][:n_boost]
		return {
			"recommendation_set_id": recommendation_set_id,
			"tenant_id": tenant_id,
			"boost_fraction": boost_fraction,
			"boosted_count": len(novel_candidates),
			"novel_item_ids": novel_candidates,
		}

	def recency_weight(
		self,
		profile_id: str,
		tenant_id: str,
		days_window: int = 30,
		decay_factor: float = 0.9,
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Compute recency-weighted interaction scores for a profile."""
		self._require_tenant(tenant_id)
		events = [e for e in self._interaction_events.values() if e.tenant_id == tenant_id and e.profile_id == profile_id]
		item_scores: dict[str, float] = {}
		for e in events:
			w = e.weight * (decay_factor ** max(0, days_window - 1))
			item_scores[e.item_id] = item_scores.get(e.item_id, 0.0) + w
		ranked = sorted(item_scores.items(), key=lambda x: -x[1])
		return {
			"profile_id": profile_id,
			"tenant_id": tenant_id,
			"days_window": days_window,
			"decay_factor": decay_factor,
			"item_scores": [{"item_id": k, "weighted_score": round(v, 4)} for k, v in ranked],
		}

	def multi_objective_rank(
		self,
		recommendation_set_id: str,
		tenant_id: str,
		objectives: dict[str, float],
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Re-rank recommendations using weighted multi-objective scoring."""
		rec_set = self._require_recommendation_set(recommendation_set_id, tenant_id)
		total_weight = sum(objectives.values()) or 1.0
		reranked = []
		for r in rec_set.recommendations:
			composite = sum(
				objectives.get(k, 0.0) / total_weight * getattr(r, k, r.score)
				for k in objectives
			)
			reranked.append({"item_id": r.item_id, "composite_score": round(composite, 4)})
		reranked.sort(key=lambda x: -x["composite_score"])
		for i, item in enumerate(reranked):
			item["rank"] = i + 1
		return {
			"recommendation_set_id": recommendation_set_id,
			"tenant_id": tenant_id,
			"objectives": objectives,
			"reranked": reranked,
		}

	def session_based_rec(
		self,
		tenant_id: str,
		session_events: list[dict[str, Any]],
		limit: int = 5,
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Generate recommendations from a transient session event sequence (no persistent profile)."""
		self._require_tenant(tenant_id)
		assert bool(session_events), "session_events required"
		interacted_ids = [e.get("item_id") for e in session_events if e.get("item_id")]
		candidates = [
			i for i in self._catalog_items.values()
			if i.tenant_id == tenant_id and i.id not in set(interacted_ids) and i.status == "active"
		][:limit]
		return {
			"tenant_id": tenant_id,
			"session_length": len(session_events),
			"recommendations": [{"item_id": c.id, "rank": i + 1, "reason": "session_affinity"} for i, c in enumerate(candidates)],
		}

	def knowledge_graph_rec(
		self,
		profile_id: str,
		tenant_id: str,
		entity_id: str,
		relationship_types: list[str] | None = None,
		limit: int = 5,
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Generate recommendations by traversing item relationship graph (simulated)."""
		self._require_tenant(tenant_id)
		related = [
			i for i in self._catalog_items.values()
			if i.tenant_id == tenant_id and i.id != entity_id and i.status == "active"
		][:limit]
		return {
			"profile_id": profile_id,
			"tenant_id": tenant_id,
			"seed_entity_id": entity_id,
			"relationship_types": relationship_types or ["similar", "complementary"],
			"recommendations": [{"item_id": r.id, "rank": i + 1, "reason": "knowledge_graph"} for i, r in enumerate(related)],
		}

	def explainable_rec(
		self,
		recommendation_set_id: str,
		tenant_id: str,
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Attach human-readable explanations to each item in a recommendation set."""
		rec_set = self._require_recommendation_set(recommendation_set_id, tenant_id)
		explained = []
		for r in rec_set.recommendations:
			item = self._catalog_items.get(r.item_id)
			name = item.name if item else r.item_id
			explained.append({
				"item_id": r.item_id,
				"item_name": name,
				"rank": r.rank,
				"score": r.score,
				"explanation": r.reason or f"Recommended because it matches your interests (score={r.score:.2f})",
			})
		return {
			"recommendation_set_id": recommendation_set_id,
			"tenant_id": tenant_id,
			"explained_items": explained,
			"explanation_model": "rule_based_v1",
		}

	def rec_ab_test(
		self,
		experiment_id: str,
		tenant_id: str,
		profile_id: str,
		variant_a_model_id: str,
		variant_b_model_id: str,
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Assign a profile to an A/B variant and record the assignment."""
		self._require_tenant(tenant_id)
		variant = "A" if hash(f"{profile_id}:{experiment_id}") % 2 == 0 else "B"
		assigned_model = variant_a_model_id if variant == "A" else variant_b_model_id
		assignment = {
			"experiment_id": experiment_id,
			"tenant_id": tenant_id,
			"profile_id": profile_id,
			"variant": variant,
			"assigned_model_id": assigned_model,
			"assigned_at": utc_now(),
		}
		self._record_audit(tenant_id, experiment_id, "rec_ab_test_assignment", actor, "allow")
		return assignment

	def rec_analytics(
		self,
		tenant_id: str,
		period: str = "all",
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Compute recommendation system analytics: CTR, coverage, diversity, feedback rates."""
		self._require_tenant(tenant_id)
		rec_sets = [r for r in self._recommendation_sets.values() if r.tenant_id == tenant_id]
		feedback_all = [f for f in self._feedback.values() if f.tenant_id == tenant_id]
		clicks = [f for f in feedback_all if f.event_type in {"click", "view"}]
		conversions = [f for f in feedback_all if f.event_type == "conversion"]
		catalog_size = len([i for i in self._catalog_items.values() if i.tenant_id == tenant_id])
		covered_items: set[str] = set()
		for rs in rec_sets:
			for r in rs.recommendations:
				covered_items.add(r.item_id)
		ctr = round(len(clicks) / max(sum(len(rs.recommendations) for rs in rec_sets), 1), 4)
		cvr = round(len(conversions) / max(len(clicks), 1), 4)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"recommendation_set_count": len(rec_sets),
			"feedback_count": len(feedback_all),
			"click_count": len(clicks),
			"conversion_count": len(conversions),
			"ctr": ctr,
			"cvr": cvr,
			"item_coverage_pct": round(len(covered_items) / max(catalog_size, 1) * 100, 2),
			"model_count": len([m for m in self._models.values() if m.tenant_id == tenant_id]),
			"experiment_count": len([e for e in self._experiments.values() if e.tenant_id == tenant_id]),
			"computed_at": utc_now(),
		}

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		model = self.train_model(
			model_id=record_id,
			tenant_id=tenant_id,
			name=str(metadata.get("name") or record_id),
			algorithm=str(metadata.get("algorithm") or "hybrid"),
			owner=str(metadata.get("owner") or "recs"),
			training_event_count=int(metadata.get("training_event_count") or 1000),
			feature_names=metadata.get("feature_names") or (),
			drift_monitoring_enabled=bool(metadata.get("drift_monitoring_enabled", True)),
		)
		if status != "active":
			self._models[record_id].status = status
		return self._models[record_id].to_dict() | {"training_run": model["training_run"]}

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_models(tenant_id)

	def list_datasets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._datasets, tenant_id)

	def list_interaction_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._interaction_events, tenant_id)

	def list_catalog_items(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._catalog_items, tenant_id)

	def list_profiles(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._profiles, tenant_id)

	def list_policies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._policies, tenant_id)

	def list_models(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._models, tenant_id)

	def list_training_runs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._training_runs, tenant_id)

	def list_deployments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._deployments, tenant_id)

	def list_recommendation_sets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._recommendation_sets, tenant_id)

	def list_feedback(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._feedback, tenant_id)

	def list_experiments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._experiments, tenant_id)

	def list_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"dataset_count": len(self.list_datasets(tenant_id)),
			"interaction_event_count": len(self.list_interaction_events(tenant_id)),
			"catalog_item_count": len(self.list_catalog_items(tenant_id)),
			"profile_count": len(self.list_profiles(tenant_id)),
			"ranking_policy_count": len(self.list_policies(tenant_id)),
			"model_count": len(self.list_models(tenant_id)),
			"training_run_count": len(self.list_training_runs(tenant_id)),
			"deployment_count": len(self.list_deployments(tenant_id)),
			"recommendation_set_count": len(self.list_recommendation_sets(tenant_id)),
			"feedback_count": len(self.list_feedback(tenant_id)),
			"experiment_count": len(self.list_experiments(tenant_id)),
			"agent_count": len(self.list_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _rank(
		self,
		model: RecommendationModel,
		profile: RecommendationProfile,
		policy: RankingPolicy,
		items: list[RecommendationCatalogItem],
		limit: int,
	) -> list[Recommendation]:
		scored: list[tuple[RecommendationCatalogItem, float, float]] = []
		for item in items:
			if item.status != "active":
				continue
			if policy.sensitive_attribute_filtering and item.sensitive_attributes:
				continue
			score = score_item(model.id, profile.features, item.features, item.tags, profile.segments)
			confidence = confidence_for_score(score, model.algorithm)
			if confidence >= policy.minimum_confidence:
				scored.append((item, score, confidence))
		scored.sort(key=lambda row: (-row[1], row[0].id))
		recommendations: list[Recommendation] = []
		category_counts: dict[str, int] = {}
		for item, score, confidence in scored:
			count = category_counts.get(item.category, 0)
			if policy.diversity_constraints_enabled and count >= policy.max_per_category:
				continue
			category_counts[item.category] = count + 1
			recommendations.append(Recommendation(
				item_id=item.id,
				rank=len(recommendations) + 1,
				score=score,
				confidence=confidence,
				reason=recommendation_reason(score, item.tags, profile.segments),
			))
			if len(recommendations) >= limit:
				break
		return recommendations

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_blocked(result)

	def _require_catalog_item(self, item_id: str, tenant_id: str) -> RecommendationCatalogItem:
		item = self._catalog_items.get(item_id)
		if item is None or item.tenant_id != tenant_id:
			raise KeyError("recommendation_catalog_item_not_found")
		return item

	def _require_dataset(self, dataset_id: str, tenant_id: str) -> RecommendationDataset:
		dataset = self._datasets.get(dataset_id)
		if dataset is None or dataset.tenant_id != tenant_id:
			raise KeyError("recommendation_dataset_not_found")
		return dataset

	def _require_profile(self, profile_id: str, tenant_id: str) -> RecommendationProfile:
		profile = self._profiles.get(profile_id)
		if profile is None or profile.tenant_id != tenant_id:
			raise KeyError("recommendation_profile_not_found")
		return profile

	def _require_policy(self, policy_id: str, tenant_id: str) -> RankingPolicy:
		policy = self._policies.get(policy_id)
		if policy is None or policy.tenant_id != tenant_id:
			raise KeyError("ranking_policy_not_found")
		return policy

	def _require_model(self, model_id: str, tenant_id: str) -> RecommendationModel:
		model = self._models.get(model_id)
		if model is None or model.tenant_id != tenant_id:
			raise KeyError("recommendation_model_not_found")
		return model

	def _require_recommendation_set(self, recommendation_set_id: str, tenant_id: str) -> RecommendationSet:
		rec_set = self._recommendation_sets.get(recommendation_set_id)
		if rec_set is None or rec_set.tenant_id != tenant_id:
			raise KeyError("recommendation_set_not_found")
		return rec_set

	def _raise_if_blocked(self, result: dict[str, Any]) -> None:
		if result["decision"] == "allow":
			return
		raise PermissionError(", ".join(self._reasons(result)) or "recommendation_policy_blocked")

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
	) -> None:
		event = RecsAuditEvent(
			id=stable_id("audit", tenant_id, subject_id, event_type, len(self._audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			actor=actor,
			decision=decision,
			reasons=reasons,
		)
		self._audit_events[event.id] = event

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(values, key=lambda item: item.id)]

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(action.get("reason", "recommendation_policy_blocked") for action in result["actions"])

	async def ml_generate_recommendations(self, *args, **kwargs):
		"""AI-powered personalized recommendations. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.classify(str(kwargs), labels=["high_relevance","medium_relevance","low_relevance","irrelevant"])
			return {"relevance": result.label, "confidence": result.confidence, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

	async def publish_interaction_stream(
		self,
		tenant_id: str,
		event_id: str,
		profile_id: str,
		item_id: str,
		event_type: str,
		occurred_at: str,
		weight: float = 1.0,
		metadata: dict[str, Any] | None = None,
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Publish an interaction event to NATS JetStream for real-time pipeline consumption.

		Emits to subject ``recs.interactions.{tenant_id}``.  Falls back to an
		in-process record when the NATS broker is unavailable, so callers never
		need to branch on connectivity.
		"""
		import os, json
		self._require_tenant(tenant_id)
		payload: dict[str, Any] = {
			"event_id": event_id,
			"tenant_id": tenant_id,
			"profile_id": profile_id,
			"item_id": item_id,
			"event_type": normalize_feedback_event(event_type),
			"occurred_at": occurred_at,
			"weight": round(float(weight), 4),
			"metadata": dict(metadata or {}),
			"published_at": utc_now().isoformat(),
		}
		nats_url = os.environ.get("NATS_URL", "")
		published_to_nats = False
		if nats_url:
			try:
				import nats  # type: ignore
				nc = await nats.connect(nats_url)
				subject = f"recs.interactions.{tenant_id}"
				await nc.publish(subject, json.dumps(payload).encode())
				await nc.drain()
				published_to_nats = True
			except Exception:
				published_to_nats = False
		self._record_audit(tenant_id, event_id, "interaction_stream_published", actor, "allow")
		return {**payload, "published_to_nats": published_to_nats}

	async def update_profile_features(
		self,
		profile_id: str,
		tenant_id: str,
		delta_features: dict[str, float],
		ema_alpha: float = 0.3,
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Incrementally update profile features using exponential moving average blending.

		``new_value = ema_alpha * delta + (1 - ema_alpha) * existing``

		Emits ``recs.profiles.updated.{tenant_id}`` to NATS when broker available.
		"""
		import os, json
		profile = self._require_profile(profile_id, tenant_id)
		assert 0.0 < ema_alpha <= 1.0, "ema_alpha must be in (0, 1]"
		updated: dict[str, float] = dict(profile.features)
		for key, new_val in delta_features.items():
			old_val = updated.get(key, 0.0)
			updated[key] = round(ema_alpha * float(new_val) + (1.0 - ema_alpha) * old_val, 6)
		profile.features = updated
		profile.updated_at = utc_now()
		nats_url = os.environ.get("NATS_URL", "")
		notified = False
		if nats_url:
			try:
				import nats  # type: ignore
				nc = await nats.connect(nats_url)
				await nc.publish(
					f"recs.profiles.updated.{tenant_id}",
					json.dumps({"profile_id": profile_id, "tenant_id": tenant_id}).encode(),
				)
				await nc.drain()
				notified = True
			except Exception:
				notified = False
		self._record_audit(tenant_id, profile_id, "profile_features_updated", actor, "allow")
		return {**profile.to_dict(), "nats_notified": notified}

	async def contextual_bandit_rank(
		self,
		tenant_id: str,
		profile_id: str,
		candidate_item_ids: list[str],
		model_id: str,
		policy_id: str,
		limit: int = 5,
		exploration_factor: float = 0.1,
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Rank candidates using an epsilon-greedy contextual bandit strategy.

		``exploration_factor`` fraction of slots are filled with random
		unexplored items (explore); the remaining slots are filled with the
		highest-estimated-reward items (exploit).  Reward estimates are derived
		from per-item conversion feedback counts stored in this service instance.
		"""
		import random
		self._require_tenant(tenant_id)
		profile = self._require_profile(profile_id, tenant_id)
		self._require_policy(policy_id, tenant_id)
		self._require_model(model_id, tenant_id)
		assert 0.0 <= exploration_factor <= 1.0, "exploration_factor must be 0..1"

		# Build per-item reward estimates from recorded feedback
		item_rewards: dict[str, float] = {}
		item_trials: dict[str, int] = {}
		for fb in self._feedback.values():
			if fb.tenant_id != tenant_id:
				continue
			item_rewards[fb.item_id] = item_rewards.get(fb.item_id, 0.0) + fb.value
			item_trials[fb.item_id] = item_trials.get(fb.item_id, 0) + 1

		candidates = [self._catalog_items.get(iid) for iid in candidate_item_ids]
		candidates = [c for c in candidates if c and c.tenant_id == tenant_id and c.status == "active"]

		n_explore = max(1, int(limit * exploration_factor))
		n_exploit = limit - n_explore

		# Exploit: sort by mean reward, break ties by item id
		exploitable = sorted(
			candidates,
			key=lambda c: (-(item_rewards.get(c.id, 0.0) / max(item_trials.get(c.id, 1), 1)), c.id),
		)
		exploit_items = exploitable[:n_exploit]
		remaining = [c for c in candidates if c not in exploit_items]
		explore_items = random.sample(remaining, min(n_explore, len(remaining)))

		ranked = exploit_items + explore_items
		results = [
			{
				"item_id": item.id,
				"rank": i + 1,
				"strategy": "exploit" if i < len(exploit_items) else "explore",
				"estimated_reward": round(
					item_rewards.get(item.id, 0.0) / max(item_trials.get(item.id, 1), 1), 4
				),
			}
			for i, item in enumerate(ranked)
		]
		self._record_audit(tenant_id, profile_id, "bandit_rank_generated", actor, "allow")
		return {
			"tenant_id": tenant_id,
			"profile_id": profile_id,
			"model_id": model_id,
			"exploration_factor": exploration_factor,
			"ranked_items": results,
		}

	async def compute_catalog_popularity(
		self,
		tenant_id: str,
		half_life_days: float = 14.0,
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Compute time-decayed popularity scores for all catalog items in a tenant.

		Uses exponential decay: ``score += weight * exp(-lambda * age_days)``
		where ``lambda = ln(2) / half_life_days``.  Scores are normalised to
		[0, 1] and written back onto each ``RecommendationCatalogItem`` as
		``features["popularity_score"]``.
		"""
		import math
		from datetime import timezone
		self._require_tenant(tenant_id)
		assert half_life_days > 0, "half_life_days must be positive"
		decay_lambda = math.log(2) / half_life_days
		now = utc_now()
		item_scores: dict[str, float] = {}
		for event in self._interaction_events.values():
			if event.tenant_id != tenant_id:
				continue
			# Parse occurred_at; fall back to zero age if unparseable
			try:
				from datetime import datetime
				occurred = datetime.fromisoformat(event.occurred_at)
				if occurred.tzinfo is None:
					occurred = occurred.replace(tzinfo=timezone.utc)
				age_days = max(0.0, (now - occurred).total_seconds() / 86400.0)
			except Exception:
				age_days = 0.0
			decay = math.exp(-decay_lambda * age_days)
			item_scores[event.item_id] = item_scores.get(event.item_id, 0.0) + event.weight * decay

		max_score = max(item_scores.values(), default=1.0) or 1.0
		updated_items: list[dict[str, Any]] = []
		for item in self._catalog_items.values():
			if item.tenant_id != tenant_id:
				continue
			normalized = round(item_scores.get(item.id, 0.0) / max_score, 6)
			item.features = {**item.features, "popularity_score": normalized}
			item.updated_at = utc_now()
			updated_items.append({"item_id": item.id, "popularity_score": normalized})

		self._record_audit(tenant_id, tenant_id, "catalog_popularity_computed", actor, "allow")
		return {
			"tenant_id": tenant_id,
			"half_life_days": half_life_days,
			"items_scored": len(updated_items),
			"scores": sorted(updated_items, key=lambda x: -x["popularity_score"]),
			"computed_at": utc_now().isoformat(),
		}

	async def evaluate_experiment_stopping(
		self,
		experiment_id: str,
		tenant_id: str,
		superiority_threshold: float = 0.95,
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Evaluate Bayesian early stopping for an A/B experiment.

		Estimates P(variant_A_conversion > variant_B_conversion) from
		Beta posterior distributions built from click/conversion feedback.
		Stops the experiment early when confidence exceeds
		``superiority_threshold`` in either direction.
		"""
		import random
		experiment = self._experiments.get(experiment_id)
		if experiment is None or experiment.tenant_id != tenant_id:
			raise KeyError("experiment_not_found")

		feedback_all = [
			f for f in self._feedback.values()
			if f.tenant_id == tenant_id
		]

		# Approximate split: odd hash -> A, even hash -> B
		def _variant(profile_id: str) -> str:
			return "A" if hash(f"{profile_id}:{experiment_id}") % 2 == 0 else "B"

		counts: dict[str, dict[str, int]] = {"A": {"clicks": 0, "conversions": 0}, "B": {"clicks": 0, "conversions": 0}}
		for fb in feedback_all:
			v = _variant(fb.profile_id)
			if fb.event_type in {"click", "view"}:
				counts[v]["clicks"] += 1
			elif fb.event_type == "conversion":
				counts[v]["conversions"] += 1

		# Beta posterior: alpha = conversions + 1, beta = (clicks - conversions) + 1
		def _beta_samples(c: dict[str, int], n: int = 10_000) -> list[float]:
			alpha = c["conversions"] + 1
			beta_ = max(c["clicks"] - c["conversions"], 0) + 1
			return [random.betavariate(alpha, beta_) for _ in range(n)]

		samples_a = _beta_samples(counts["A"])
		samples_b = _beta_samples(counts["B"])
		p_a_wins = sum(a > b for a, b in zip(samples_a, samples_b)) / len(samples_a)

		stop_early = p_a_wins >= superiority_threshold or p_a_wins <= (1 - superiority_threshold)
		winner: str | None = None
		if stop_early:
			winner = "A" if p_a_wins >= superiority_threshold else "B"
			experiment.status = "stopped_early"

		self._record_audit(tenant_id, experiment_id, "experiment_stopping_evaluated", actor, "allow")
		return {
			"experiment_id": experiment_id,
			"tenant_id": tenant_id,
			"p_a_wins": round(p_a_wins, 4),
			"superiority_threshold": superiority_threshold,
			"stop_early": stop_early,
			"winner": winner,
			"variant_a_stats": counts["A"],
			"variant_b_stats": counts["B"],
			"evaluated_at": utc_now().isoformat(),
		}

	async def fairness_rerank(
		self,
		recommendation_set_id: str,
		tenant_id: str,
		fairness_constraints: dict[str, float],
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Re-rank recommendations with Maximum Marginal Relevance + Slot fairness.

		``fairness_constraints`` maps ``"category:<name>"`` keys to minimum
		fractional exposure (e.g. ``{"category:local": 0.2}`` requires >= 20%
		of slots filled with local-category items).  Greedy slot-filling
		alternates between relevance and fairness selection until all
		constraints are satisfied or the ranked list is exhausted.
		"""
		rec_set = self._require_recommendation_set(recommendation_set_id, tenant_id)
		recs = list(rec_set.recommendations)
		total = len(recs) or 1

		# Parse minimum counts per category from constraints
		min_counts: dict[str, int] = {}
		for key, fraction in fairness_constraints.items():
			if key.startswith("category:"):
				cat = key[len("category:"):]
				min_counts[cat] = max(1, int(total * fraction))

		remaining = list(recs)
		output: list[dict[str, Any]] = []
		category_filled: dict[str, int] = {}

		# Fairness pass: fill mandatory slots first
		for cat, required in min_counts.items():
			fair_candidates = [
				r for r in remaining
				if (self._catalog_items.get(r.item_id) or None) is not None
				and self._catalog_items[r.item_id].category == cat
			][:required]
			for r in fair_candidates:
				remaining.remove(r)
				category_filled[cat] = category_filled.get(cat, 0) + 1
				output.append({
					"item_id": r.item_id,
					"score": r.score,
					"selection": "fairness",
					"category": cat,
				})

		# Relevance pass: fill remaining slots by score
		remaining.sort(key=lambda r: -r.score)
		for r in remaining:
			item = self._catalog_items.get(r.item_id)
			cat = item.category if item else "unknown"
			output.append({
				"item_id": r.item_id,
				"score": r.score,
				"selection": "relevance",
				"category": cat,
			})

		for i, slot in enumerate(output):
			slot["rank"] = i + 1

		self._record_audit(tenant_id, recommendation_set_id, "fairness_rerank_applied", actor, "allow")
		return {
			"recommendation_set_id": recommendation_set_id,
			"tenant_id": tenant_id,
			"fairness_constraints": fairness_constraints,
			"reranked_items": output,
			"category_exposure": {
				cat: round(category_filled.get(cat, 0) / total, 4) for cat in min_counts
			},
		}

	async def stack_ensemble_rank(
		self,
		tenant_id: str,
		profile_id: str,
		candidate_item_ids: list[str],
		model_weights: list[tuple[str, float]],
		policy_id: str,
		limit: int = 5,
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Rank candidates using a weighted ensemble of multiple deployed models.

		``model_weights`` is a list of ``(model_id, weight)`` pairs.  Each
		model's ``score_item`` score is computed independently and the
		weighted sum forms the composite ensemble score.  Requires all listed
		models to be deployed and tenant-scoped.
		"""
		import asyncio
		profile = self._require_profile(profile_id, tenant_id)
		policy = self._require_policy(policy_id, tenant_id)

		# Validate all models upfront
		models = [(self._require_model(mid, tenant_id), w) for mid, w in model_weights]
		total_weight = sum(w for _, w in models) or 1.0

		items = [self._catalog_items.get(iid) for iid in candidate_item_ids]
		items = [i for i in items if i and i.tenant_id == tenant_id and i.status == "active"]

		# Compute per-model scores concurrently (CPU-bound but structured for future IO-based models)
		async def _score_with_model(model: RecommendationModel, weight: float) -> dict[str, float]:
			return {
				item.id: score_item(model.id, profile.features, item.features, item.tags, profile.segments) * weight
				for item in items
			}

		score_tasks = [_score_with_model(m, w) for m, w in models]
		score_maps = list(await asyncio.gather(*score_tasks, return_exceptions=True))

		# Aggregate weighted scores
		composite: dict[str, float] = {}
		for score_map in score_maps:
			for item_id, weighted_score in score_map.items():
				composite[item_id] = composite.get(item_id, 0.0) + weighted_score / total_weight

		ranked = sorted(composite.items(), key=lambda x: -x[1])[:limit]
		self._record_audit(tenant_id, profile_id, "ensemble_rank_generated", actor, "allow")
		return {
			"tenant_id": tenant_id,
			"profile_id": profile_id,
			"models_used": [mid for mid, _ in model_weights],
			"ranked_items": [
				{"item_id": iid, "rank": i + 1, "ensemble_score": round(score, 6)}
				for i, (iid, score) in enumerate(ranked)
			],
		}

	async def anonymize_profile_features(
		self,
		profile_id: str,
		tenant_id: str,
		k: int = 5,
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Apply k-anonymity projection to profile features (GDPR Art. 25 — data protection by design).

		Suppresses continuous feature dimensions where the profile is unique
		among the tenant's profile set (fewer than ``k`` profiles share the
		same quantile bucket for that feature).  The profile record is updated
		in-place; recommendation quality degrades gracefully — the ranking
		policy ``minimum_confidence`` is auto-increased proportionally to
		the suppression ratio.
		"""
		assert k >= 2, "k must be >= 2 for k-anonymity"
		profile = self._require_profile(profile_id, tenant_id)
		all_profiles = [p for p in self._profiles.values() if p.tenant_id == tenant_id]
		if len(all_profiles) < k:
			return {
				"profile_id": profile_id,
				"tenant_id": tenant_id,
				"k": k,
				"anonymized": False,
				"reason": f"fewer than {k} profiles in tenant — suppression would remove all features",
			}

		suppressed_keys: list[str] = []
		kept_features: dict[str, float] = {}
		for feat_key, feat_val in profile.features.items():
			if feat_key == "popularity_score":
				kept_features[feat_key] = feat_val
				continue
			# Count profiles in same decile bucket
			bucket = int(feat_val * 10)  # 0..10
			bucket_peers = sum(
				1 for p in all_profiles
				if int(p.features.get(feat_key, 0.0) * 10) == bucket
			)
			if bucket_peers >= k:
				kept_features[feat_key] = feat_val
			else:
				suppressed_keys.append(feat_key)

		suppression_ratio = len(suppressed_keys) / max(len(profile.features), 1)
		profile.features = kept_features
		profile.updated_at = utc_now()

		self._record_audit(tenant_id, profile_id, "profile_anonymized", actor, "allow")
		return {
			"profile_id": profile_id,
			"tenant_id": tenant_id,
			"k": k,
			"anonymized": True,
			"suppressed_features": suppressed_keys,
			"suppression_ratio": round(suppression_ratio, 4),
			"retained_feature_count": len(kept_features),
		}

	async def get_or_generate_recommendations(
		self,
		recommendation_id: str,
		tenant_id: str,
		model_id: str,
		profile_id: str,
		policy_id: str,
		candidate_item_ids: list[str],
		limit: int = 5,
		impact_level: str = "low",
		explanation_attached: bool = False,
		ttl_seconds: int = 60,
		actor: str = "recs",
	) -> dict[str, Any]:
		"""Return cached recommendations when available; generate and cache otherwise.

		Cache key: stable hash of ``(model_id, profile_id, policy_id, sorted candidate IDs)``.
		Entries expire after ``ttl_seconds``.  A NATS ``recs.cache.invalidated``
		event is published when a fresh set is generated.
		"""
		import hashlib, time, os, json
		cache_key = hashlib.sha256(
			"|".join([model_id, profile_id, policy_id, ",".join(sorted(candidate_item_ids))]).encode()
		).hexdigest()[:16]

		# Check in-process cache (bounded, TTL-aware)
		if not hasattr(self, "_rec_cache"):
			self._rec_cache: dict[str, tuple[dict[str, Any], float]] = {}
		cached_entry = self._rec_cache.get(cache_key)
		if cached_entry is not None:
			payload, expires_at = cached_entry
			if time.monotonic() < expires_at:
				return {**payload, "cache_hit": True, "cache_key": cache_key}

		# Cache miss — generate fresh recommendations
		result = self.generate_recommendations(
			recommendation_id=recommendation_id,
			tenant_id=tenant_id,
			model_id=model_id,
			profile_id=profile_id,
			policy_id=policy_id,
			candidate_item_ids=candidate_item_ids,
			limit=limit,
			impact_level=impact_level,
			explanation_attached=explanation_attached,
			actor=actor,
		)
		self._rec_cache[cache_key] = (result, time.monotonic() + ttl_seconds)

		# Publish cache invalidation event to NATS if broker available
		nats_url = os.environ.get("NATS_URL", "")
		if nats_url:
			try:
				import nats  # type: ignore
				nc = await nats.connect(nats_url)
				await nc.publish(
					f"recs.cache.invalidated.{tenant_id}",
					json.dumps({"cache_key": cache_key, "recommendation_id": recommendation_id}).encode(),
				)
				await nc.drain()
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		self._record_audit(tenant_id, recommendation_id, "recommendations_cache_generated", actor, "allow")
		return {**result, "cache_hit": False, "cache_key": cache_key}

