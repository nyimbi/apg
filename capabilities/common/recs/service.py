"""Service layer for executable Recommender Systems operations."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	RankingPolicy,
	Recommendation,
	RecommendationCatalogItem,
	RecommendationExperiment,
	RecommendationModel,
	RecommendationProfile,
	RecommendationSet,
	RecsAuditEvent,
	TrainingRun,
	utc_now,
)
from .recommendation_runtime import (
	confidence_for_score,
	drift_status,
	normalize_algorithm,
	normalize_features,
	normalize_impact_level,
	normalize_labels,
	recommendation_reason,
	score_item,
	stable_id,
)


class RecsService:
	"""In-process catalog, profile, ranking, model, experiment, and recommendation service."""

	def __init__(self) -> None:
		self._catalog_items: dict[str, RecommendationCatalogItem] = {}
		self._profiles: dict[str, RecommendationProfile] = {}
		self._policies: dict[str, RankingPolicy] = {}
		self._models: dict[str, RecommendationModel] = {}
		self._training_runs: dict[str, TrainingRun] = {}
		self._recommendation_sets: dict[str, RecommendationSet] = {}
		self._experiments: dict[str, RecommendationExperiment] = {}
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
		minimum_confidence: float = 0.65,
		diversity_constraints_enabled: bool = True,
		sensitive_attribute_filtering: bool = True,
		max_per_category: int = 2,
		actor: str = "recs",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
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
		})
		self._raise_if_blocked(result)
		if not owner:
			raise PermissionError("model_owner_required")
		if not drift_monitoring_enabled:
			raise PermissionError("drift_monitoring_required")
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
		ranked = self._rank(model, profile, policy, items, int(limit))
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

	def list_recommendation_sets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._recommendation_sets, tenant_id)

	def list_experiments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._experiments, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"catalog_item_count": len(self.list_catalog_items(tenant_id)),
			"profile_count": len(self.list_profiles(tenant_id)),
			"ranking_policy_count": len(self.list_policies(tenant_id)),
			"model_count": len(self.list_models(tenant_id)),
			"training_run_count": len(self.list_training_runs(tenant_id)),
			"recommendation_set_count": len(self.list_recommendation_sets(tenant_id)),
			"experiment_count": len(self.list_experiments(tenant_id)),
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
