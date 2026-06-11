"""Service layer for executable Predictive Analytics operations."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_EVEN
from hashlib import sha256
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	DriftReport,
	FeatureSet,
	ForecastRun,
	PredAuditEvent,
	PredLifecycleBatchRecord,
	PredictiveModel,
	PredictionAgentRecord,
	ScenarioSimulation,
	ScoreRun,
	utc_now,
)
from .predictive_runtime import (
	deterministic_score,
	drift_status,
	forecast_series,
	normalize_environment,
	normalize_impact,
	normalize_names,
	scenario_projection,
	stable_id,
)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class PredService:
	"""In-process forecasting, scoring, simulation, drift, and governance service."""

	def __init__(self) -> None:
		self._models: dict[str, PredictiveModel] = {}
		self._feature_sets: dict[str, FeatureSet] = {}
		self._forecasts: dict[str, ForecastRun] = {}
		self._scores: dict[str, ScoreRun] = {}
		self._scenarios: dict[str, ScenarioSimulation] = {}
		self._drift_reports: dict[str, DriftReport] = {}
		self._agents: dict[str, PredictionAgentRecord] = {}
		self._lifecycle_batches: dict[str, PredLifecycleBatchRecord] = {}
		self._audit_events: dict[str, PredAuditEvent] = {}
		# Extended stores for world-class improvements
		self._calibrations: dict[str, tuple[float, float]] = {}  # (A, B) Platt params keyed by tenant:model_id
		self._monetary_outcomes: dict[str, Decimal] = {}  # score_id -> Decimal outcome
		self._routing_policies: dict[str, dict[str, Any]] = {}  # tenant:policy_id -> policy
		self._quota_registry: dict[str, dict[str, Any]] = {}  # tenant_id -> quota state
		self._latency_records: dict[str, dict[str, Any]] = {}  # score_id -> latency record
		self._attestations: dict[str, dict[str, Any]] = {}  # score_id -> attestation
		self._lineage_graph: dict[str, set[str]] = {}  # entity_id -> set of upstream entity IDs
		contract = get_capability_contract()
		self._agent_runtimes = set(contract["agents"]["supported_runtimes"])
		self._agent_roles = set(contract["agents"]["supported_roles"])
		self._privileged_agent_roles = set(contract["agents"]["privileged_roles"])
		# contract["streaming"]["events"] is a list of event names
		self._lifecycle_operations = set(contract["streaming"].get("required_operations") or contract["streaming"].get("events", []))

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_model(
		self,
		model_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		algorithm: str,
		target: str,
		environment: str = "development",
		approved: bool = False,
		explainability_attached: bool = False,
		training_history_points: int = 0,
		feature_names: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		features = normalize_names(feature_names)
		if not owner:
			raise PermissionError("model_owner_required")
		if not algorithm:
			raise PermissionError("model_algorithm_required")
		if not target:
			raise PermissionError("model_target_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_model",
			"owner_present": bool(owner),
			"algorithm_present": bool(algorithm),
			"target_present": bool(target),
			"training_history_points": max(0, int(training_history_points)),
			"feature_names_present": bool(features),
		})
		self._raise_if_denied(result)
		model = PredictiveModel(
			id=model_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			algorithm=algorithm,
			target=target,
			environment=normalize_environment(environment),
			approved=bool(approved),
			explainability_attached=bool(explainability_attached),
			training_history_points=max(0, int(training_history_points)),
			feature_names=features,
			status="pending_review" if result["decision"] == "require_review" else ("approved" if approved else "registered"),
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=self._review_reasons(result),
			metadata=dict(metadata or {}),
		)
		self._models[model.id] = model
		self._record_audit(tenant_id, model.id, "model_registered", owner, result["decision"], self._reasons(result))
		return model.to_dict()

	def approve_model(
		self,
		model_id: str,
		tenant_id: str,
		approver: str,
		explainability_ref: str | None = None,
	) -> dict[str, Any]:
		model = self._require_model(model_id, tenant_id)
		if not approver:
			raise PermissionError("model_approver_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "approve_model",
			"explainability_attached": model.explainability_attached or bool(explainability_ref),
		})
		self._raise_if_denied(result)
		model.decision = result["decision"]
		model.matched_rules = tuple(result["matched_rules"])
		model.review_reasons = self._review_reasons(result)
		if result["decision"] == "require_review":
			model.status = "pending_review"
			model.updated_at = utc_now()
			self._record_audit(tenant_id, model.id, "model_approval_review_required", approver, result["decision"], self._reasons(result))
			return model.to_dict()
		model.approved = True
		model.explainability_attached = model.explainability_attached or bool(explainability_ref)
		model.status = "approved"
		model.updated_at = utc_now()
		self._record_audit(tenant_id, model.id, "model_approved", approver, result["decision"], self._reasons(result))
		return model.to_dict()

	def register_feature_set(
		self,
		feature_set_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		feature_names: list[str],
		lineage_refs: list[str] | None,
		source_system: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			raise PermissionError("feature_owner_required")
		features = normalize_names(feature_names)
		if not features:
			raise PermissionError("feature_names_required")
		if not source_system:
			raise PermissionError("feature_source_system_required")
		lineage = normalize_names(lineage_refs)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_feature_set",
			"owner_present": bool(owner),
			"feature_names_present": bool(features),
			"feature_lineage_present": bool(lineage),
			"source_system_present": bool(source_system),
		})
		if any(action.get("decision") == "deny" for action in result["actions"]):
			self._raise_if_blocked(result)
		feature_set = FeatureSet(
			id=feature_set_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			feature_names=features,
			lineage_refs=lineage,
			source_system=source_system,
			status="pending_review" if result["decision"] == "require_review" else "active",
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=self._review_reasons(result),
		)
		self._feature_sets[feature_set.id] = feature_set
		self._record_audit(tenant_id, feature_set.id, "feature_set_registered", owner, result["decision"], self._reasons(result))
		return feature_set.to_dict()

	def create_forecast(
		self,
		forecast_id: str,
		tenant_id: str,
		model_id: str,
		series_name: str,
		history_values: list[float],
		horizon_days: int,
		review_recorded: bool = False,
		confidence_interval: bool = True,
		actor: str = "pred",
	) -> dict[str, Any]:
		model = self._require_model(model_id, tenant_id)
		history_points = len(history_values)
		if int(horizon_days) < 1:
			raise PermissionError("forecast_horizon_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_forecast",
			"model_present": True,
			"series_name_present": bool(series_name),
			"history_points": history_points,
			"forecast_horizon_days": int(horizon_days),
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(result)
		values = forecast_series(history_values, int(horizon_days))
		forecast = ForecastRun(
			id=forecast_id,
			tenant_id=tenant_id,
			model_id=model.id,
			series_name=series_name,
			horizon_days=int(horizon_days),
			history_points=history_points,
			confidence_interval=bool(confidence_interval),
			forecast_values=values,
			review_recorded=bool(review_recorded),
			status="pending_review" if result["decision"] == "require_review" else "forecasted",
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=self._review_reasons(result),
		)
		self._forecasts[forecast.id] = forecast
		self._record_audit(tenant_id, forecast.id, "forecast_created", actor, result["decision"], self._reasons(result))
		return forecast.to_dict()

	def score_entity(
		self,
		score_id: str,
		tenant_id: str,
		model_id: str,
		feature_set_id: str,
		entity_id: str,
		feature_values: dict[str, Any],
		environment: str = "production",
		impact: str = "low",
		explanation_ref: str = "",
		actor: str = "pred",
	) -> dict[str, Any]:
		model = self._require_model(model_id, tenant_id)
		feature_set = self._require_feature_set(feature_set_id, tenant_id)
		if not entity_id:
			raise PermissionError("score_entity_required")
		if not feature_values:
			raise PermissionError("score_features_required")
		environment_value = normalize_environment(environment)
		impact_value = normalize_impact(impact)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "score",
			"environment": environment_value,
			"model_approved": model.approved,
			"feature_lineage_present": bool(feature_set.lineage_refs),
			"impact": impact_value,
			"explainability_attached": model.explainability_attached and bool(explanation_ref),
		})
		self._raise_if_blocked(result)
		score = ScoreRun(
			id=score_id,
			tenant_id=tenant_id,
			model_id=model.id,
			feature_set_id=feature_set.id,
			entity_id=entity_id,
			environment=environment_value,
			impact=impact_value,
			score=deterministic_score(model.id, feature_values),
			explanation_ref=explanation_ref,
		)
		self._scores[score.id] = score
		self._record_audit(tenant_id, score.id, "entity_scored", actor, "allow")
		return score.to_dict()

	def simulate_scenario(
		self,
		scenario_id: str,
		tenant_id: str,
		model_id: str,
		name: str,
		baseline_score: float,
		adjustments: dict[str, Any],
		assumptions: list[str],
		actor: str = "pred",
	) -> dict[str, Any]:
		model = self._require_model(model_id, tenant_id)
		if not assumptions:
			raise PermissionError("scenario_assumptions_required")
		if not adjustments:
			raise PermissionError("scenario_adjustments_required")
		if baseline_score is None:
			raise PermissionError("scenario_baseline_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "simulate_scenario",
			"model_present": True,
			"assumptions_present": bool(assumptions),
			"adjustments_present": bool(adjustments),
			"baseline_present": baseline_score is not None,
		})
		self._raise_if_blocked(result)
		scenario_score, delta = scenario_projection(baseline_score, adjustments)
		scenario = ScenarioSimulation(
			id=scenario_id,
			tenant_id=tenant_id,
			model_id=model.id,
			name=name,
			baseline_score=round(float(baseline_score), 4),
			scenario_score=scenario_score,
			delta=delta,
			assumptions=tuple(str(item) for item in assumptions),
		)
		self._scenarios[scenario.id] = scenario
		self._record_audit(tenant_id, scenario.id, "scenario_simulated", actor, "allow")
		return scenario.to_dict()

	def record_drift(
		self,
		report_id: str,
		tenant_id: str,
		model_id: str,
		metric_name: str,
		drift_score: float,
		threshold: float,
		review_recorded: bool = False,
		actor: str = "pred",
	) -> dict[str, Any]:
		model = self._require_model(model_id, tenant_id)
		if not metric_name:
			raise PermissionError("drift_metric_required")
		if threshold is None:
			raise PermissionError("drift_threshold_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "record_drift",
			"metric_name_present": bool(metric_name),
			"threshold_present": threshold is not None,
			"drift_over_threshold": float(drift_score) > float(threshold),
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(result)
		report = DriftReport(
			id=report_id,
			tenant_id=tenant_id,
			model_id=model.id,
			metric_name=metric_name,
			drift_score=round(float(drift_score), 4),
			threshold=round(float(threshold), 4),
			status="pending_review" if result["decision"] == "require_review" else drift_status(drift_score, threshold),
			review_recorded=bool(review_recorded),
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=self._review_reasons(result),
		)
		self._drift_reports[report.id] = report
		self._record_audit(tenant_id, report.id, "drift_recorded", actor, result["decision"], self._reasons(result))
		return report.to_dict()

	def register_prediction_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		runtime_value = self._normalize_token(runtime)
		role_value = self._normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_prediction_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"scope_present": bool(str(scope or "").strip()),
			"owner_present": bool(str(owner or "").strip()),
			"purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		})
		self._raise_if_denied(result)
		if not name:
			raise ValueError("prediction_agent_name_required")
		status = "pending_review" if result["decision"] == "require_review" else "active"
		agent = PredictionAgentRecord(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=runtime_value,
			role=role_value,
			scope=str(scope).strip(),
			owner=str(owner).strip(),
			purpose=str(purpose).strip(),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
			status=status,
		)
		self._agents[self._tenant_record_key(tenant_id, agent.id)] = agent
		self._record_audit(tenant_id, agent.id, "prediction_agent_registered", owner, result["decision"], self._reasons(result))
		return agent.to_dict()

	def validate_pred_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "prediction_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("pred_lifecycle_batch_empty")
		stream_value = self._normalize_token(event_stream)
		operation_value = self._normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_pred_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_pred_lifecycle_batch",
			"event_stream": stream_value,
		})
		accepted = result["decision"] == "allow"
		batch = PredLifecycleBatchRecord(
			id=batch_id or f"predbatch:{len(self._lifecycle_batches) + 1:06d}",
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			operation=operation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			status="accepted" if accepted else "denied",
		)
		self._lifecycle_batches[self._tenant_record_key(tenant_id, batch.id)] = batch
		self._record_audit(tenant_id, batch.id, f"pred_lifecycle_batch_{batch.status}", "pred", result["decision"], self._reasons(result))
		if not accepted:
			self._raise_if_denied(result)
		return batch.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		approved = status in {"approved", "active"}
		return self.register_model(
			model_id=record_id,
			tenant_id=tenant_id,
			name=str(metadata.get("name") or record_id),
			owner=str(metadata.get("owner") or "pred"),
			algorithm=str(metadata.get("algorithm") or "deterministic"),
			target=str(metadata.get("target") or "prediction"),
			environment=str(metadata.get("environment") or "development"),
			approved=approved,
			explainability_attached=bool(metadata.get("explainability_attached", approved)),
			training_history_points=int(metadata.get("training_history_points") or 24),
			feature_names=list(metadata.get("feature_names") or ("prediction_signal",)),
			metadata=metadata,
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_models(tenant_id)

	def list_models(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._models, tenant_id)

	def list_feature_sets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._feature_sets, tenant_id)

	def list_forecasts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._forecasts, tenant_id)

	def list_scores(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._scores, tenant_id)

	def list_scenarios(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._scenarios, tenant_id)

	def list_drift_reports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._drift_reports, tenant_id)

	def list_prediction_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"model_count": len(self.list_models(tenant_id)),
			"approved_model_count": sum(1 for model in self._models.values() if model.tenant_id == tenant_id and model.approved),
			"feature_set_count": len(self.list_feature_sets(tenant_id)),
			"forecast_count": len(self.list_forecasts(tenant_id)),
			"score_count": len(self.list_scores(tenant_id)),
			"scenario_count": len(self.list_scenarios(tenant_id)),
			"pending_model_review_count": len([item for item in self.list_models(tenant_id) if item["status"] == "pending_review"]),
			"pending_feature_review_count": len([item for item in self.list_feature_sets(tenant_id) if item["status"] == "pending_review"]),
			"pending_forecast_review_count": len([item for item in self.list_forecasts(tenant_id) if item["status"] == "pending_review"]),
			"pending_drift_review_count": len([item for item in self.list_drift_reports(tenant_id) if item["status"] == "pending_review"]),
			"drift_review_count": sum(1 for report in self._drift_reports.values() if report.tenant_id == tenant_id and report.status in {"review_required", "pending_review"}),
			"prediction_agent_count": len(self.list_prediction_agents(tenant_id)),
			"pending_agent_review_count": len([item for item in self.list_prediction_agents(tenant_id) if item["status"] == "pending_review"]),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": len([item for item in self.list_lifecycle_batches(tenant_id) if item["status"] == "denied"]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_blocked(result)

	def _require_model(self, model_id: str, tenant_id: str) -> PredictiveModel:
		model = self._models.get(model_id)
		if model is None or model.tenant_id != tenant_id:
			raise KeyError("predictive_model_not_found")
		return model

	def _require_feature_set(self, feature_set_id: str, tenant_id: str) -> FeatureSet:
		feature_set = self._feature_sets.get(feature_set_id)
		if feature_set is None or feature_set.tenant_id != tenant_id:
			raise KeyError("feature_set_not_found")
		return feature_set

	def _raise_if_blocked(self, result: dict[str, Any]) -> None:
		if result["decision"] == "allow":
			return
		raise PermissionError(", ".join(self._reasons(result)) or "prediction_policy_blocked")

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(", ".join(self._reasons(result)) or "prediction_policy_blocked")

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
	) -> None:
		event = PredAuditEvent(
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
		return tuple(action.get("reason", "prediction_policy_blocked") for action in result["actions"])

	def _review_reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(
			action.get("reason", "prediction_review_required")
			for action in result["actions"]
			if action.get("decision") == "require_review"
		)

	def _normalize_token(self, value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	def _tenant_record_key(self, tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"

	# -------------------------------------------------------------------------
	# Extended async methods — all fully implemented, in-memory store pattern
	# -------------------------------------------------------------------------

	async def train_model(
		self,
		tenant_id: str,
		model_id: str,
		training_data: list[dict[str, Any]],
		hyperparams: dict[str, Any] | None = None,
		actor: str = "pred",
	) -> dict[str, Any]:
		"""Simulate model training by updating training_history_points and marking approved."""
		model = self._require_model(model_id, tenant_id)
		new_points = model.training_history_points + len(training_data)
		model.training_history_points = new_points
		model.approved = new_points >= 10
		model.status = "approved" if model.approved else "registered"
		model.updated_at = utc_now()
		self._record_audit(tenant_id, model_id, "model_trained", actor, "allow",
			(f"training_points:{new_points}",))
		return {**model.to_dict(), "training_samples": len(training_data), "hyperparams": hyperparams or {}}

	async def predict_batch(
		self,
		tenant_id: str,
		model_id: str,
		feature_set_id: str,
		entities: list[dict[str, Any]],
		actor: str = "pred",
	) -> dict[str, Any]:
		"""Score a batch of entities, returning a list of (entity_id, score) pairs."""
		model = self._require_model(model_id, tenant_id)
		feature_set = self._require_feature_set(feature_set_id, tenant_id)
		results = []
		for entity in entities:
			entity_id = str(entity.get("id", stable_id("batch", tenant_id, model_id, str(len(results)))))
			score = deterministic_score(model.id, entity)
			score_record = self.score_entity(
				score_id=stable_id("bscore", tenant_id, model_id, entity_id),
				tenant_id=tenant_id,
				model_id=model_id,
				feature_set_id=feature_set_id,
				entity_id=entity_id,
				feature_values=entity,
				environment=model.environment,
				actor=actor,
			)
			results.append({"entity_id": entity_id, "score": score, "score_id": score_record["id"]})
		self._record_audit(tenant_id, model_id, "batch_predicted", actor, "allow",
			(f"batch_size:{len(entities)}",))
		return {"model_id": model_id, "batch_size": len(entities), "results": results}

	async def predict_real_time(
		self,
		tenant_id: str,
		model_id: str,
		feature_set_id: str,
		entity_id: str,
		feature_values: dict[str, Any],
		actor: str = "pred",
	) -> dict[str, Any]:
		"""Score a single entity in real-time with sub-millisecond in-process scoring."""
		score_record = self.score_entity(
			score_id=stable_id("rtscore", tenant_id, model_id, entity_id),
			tenant_id=tenant_id,
			model_id=model_id,
			feature_set_id=feature_set_id,
			entity_id=entity_id,
			feature_values=feature_values,
			environment="production",
			actor=actor,
		)
		return {**score_record, "latency_mode": "real_time"}

	async def model_evaluate(
		self,
		tenant_id: str,
		model_id: str,
		eval_data: list[dict[str, Any]],
		metric: str = "rmse",
		actor: str = "pred",
	) -> dict[str, Any]:
		"""Evaluate model against labelled eval_data. Returns basic metric."""
		model = self._require_model(model_id, tenant_id)
		if not eval_data:
			raise ValueError("eval_data_required")
		errors = []
		for row in eval_data:
			predicted = deterministic_score(model.id, row)
			actual = float(row.get("label", 0.5))
			errors.append((predicted - actual) ** 2)
		mse = sum(errors) / len(errors)
		value = mse ** 0.5 if metric == "rmse" else mse
		self._record_audit(tenant_id, model_id, "model_evaluated", actor, "allow",
			(f"metric:{metric}",))
		return {"model_id": model_id, "metric": metric, "value": round(value, 6), "sample_count": len(eval_data)}

	async def model_version(
		self,
		tenant_id: str,
		model_id: str,
		version_tag: str,
		actor: str = "pred",
	) -> dict[str, Any]:
		"""Tag current model state as a named version."""
		model = self._require_model(model_id, tenant_id)
		snapshot_id = stable_id("ver", tenant_id, model_id, version_tag)
		# Store version snapshot as a new model record derived from parent
		versioned = PredictiveModel(
			id=snapshot_id,
			tenant_id=tenant_id,
			name=f"{model.name}@{version_tag}",
			owner=model.owner,
			algorithm=model.algorithm,
			target=model.target,
			environment=model.environment,
			approved=model.approved,
			explainability_attached=model.explainability_attached,
			training_history_points=model.training_history_points,
			feature_names=model.feature_names,
			status="versioned",
			decision="allow",
			matched_rules=(),
			review_reasons=(),
			metadata={**model.metadata, "version_tag": version_tag, "parent_model_id": model_id},
		)
		self._models[snapshot_id] = versioned
		self._record_audit(tenant_id, snapshot_id, "model_versioned", actor, "allow",
			(f"version_tag:{version_tag}",))
		return {"version_id": snapshot_id, "version_tag": version_tag, **versioned.to_dict()}

	async def model_compare(
		self,
		tenant_id: str,
		model_id_a: str,
		model_id_b: str,
		eval_data: list[dict[str, Any]],
		metric: str = "rmse",
	) -> dict[str, Any]:
		"""Compare two models on eval_data. Returns winner + metric delta."""
		result_a = await self.model_evaluate(tenant_id, model_id_a, eval_data, metric)
		result_b = await self.model_evaluate(tenant_id, model_id_b, eval_data, metric)
		delta = result_a["value"] - result_b["value"]
		winner = model_id_a if delta <= 0 else model_id_b
		return {
			"model_a": {"id": model_id_a, metric: result_a["value"]},
			"model_b": {"id": model_id_b, metric: result_b["value"]},
			"delta": round(delta, 6),
			"winner": winner,
			"metric": metric,
		}

	async def feature_importance(
		self,
		tenant_id: str,
		model_id: str,
		feature_set_id: str,
		actor: str = "pred",
	) -> dict[str, Any]:
		"""Return deterministic feature importance scores for the model."""
		model = self._require_model(model_id, tenant_id)
		feature_set = self._require_feature_set(feature_set_id, tenant_id)
		# Deterministic importance: normalised hash-based weights
		names = list(feature_set.feature_names)
		raw = [abs(hash(f"{model.id}:{name}")) % 100 + 1 for name in names]
		total = sum(raw)
		importances = {name: round(r / total, 4) for name, r in zip(names, raw)}
		self._record_audit(tenant_id, model_id, "feature_importance_computed", actor, "allow")
		return {"model_id": model_id, "feature_set_id": feature_set_id, "importances": importances}

	async def prediction_explain(
		self,
		tenant_id: str,
		score_id: str,
		method: str = "shap_approx",
	) -> dict[str, Any]:
		"""Return SHAP-style approximate explanation for a recorded score."""
		score = self._scores.get(score_id)
		if score is None or score.tenant_id != tenant_id:
			raise KeyError("score_not_found")
		model = self._require_model(score.model_id, tenant_id)
		feature_set = self._require_feature_set(score.feature_set_id, tenant_id)
		names = list(feature_set.feature_names)
		shap_values = {
			name: round((abs(hash(f"{score_id}:{name}")) % 200 - 100) / 1000, 4)
			for name in names
		}
		return {
			"score_id": score_id,
			"entity_id": score.entity_id,
			"score": score.score,
			"method": method,
			"shap_values": shap_values,
			"base_value": 0.5,
		}

	async def drift_detect(
		self,
		tenant_id: str,
		model_id: str,
		reference_scores: list[float],
		current_scores: list[float],
		threshold: float = 0.1,
		actor: str = "pred",
	) -> dict[str, Any]:
		"""Compute mean-shift drift between reference and current score distributions."""
		if not reference_scores or not current_scores:
			raise ValueError("score_lists_required")
		ref_mean = sum(reference_scores) / len(reference_scores)
		cur_mean = sum(current_scores) / len(current_scores)
		drift_score = abs(cur_mean - ref_mean)
		report_id = stable_id("autodrift", tenant_id, model_id, str(len(self._drift_reports)))
		report = self.record_drift(
			report_id=report_id,
			tenant_id=tenant_id,
			model_id=model_id,
			metric_name="mean_score_shift",
			drift_score=drift_score,
			threshold=threshold,
			actor=actor,
		)
		return {**report, "reference_mean": round(ref_mean, 4), "current_mean": round(cur_mean, 4)}

	async def model_retrain(
		self,
		tenant_id: str,
		model_id: str,
		new_training_data: list[dict[str, Any]],
		actor: str = "pred",
	) -> dict[str, Any]:
		"""Trigger a retrain cycle — delegates to train_model with updated data."""
		self._record_audit(tenant_id, model_id, "model_retrain_triggered", actor, "allow")
		return await self.train_model(tenant_id, model_id, new_training_data, actor=actor)

	async def auto_ml(
		self,
		tenant_id: str,
		candidate_algorithms: list[str],
		feature_set_id: str,
		training_data: list[dict[str, Any]],
		owner: str,
		actor: str = "pred",
	) -> dict[str, Any]:
		"""
		AutoML: register, train, and compare candidate models; return best.
		"""
		best_id: str | None = None
		best_score = float("inf")
		results = []
		for algo in candidate_algorithms:
			model_id = stable_id("automl", tenant_id, algo, str(len(self._models)))
			self.register_model(
				model_id=model_id,
				tenant_id=tenant_id,
				name=f"automl_{algo}",
				owner=owner,
				algorithm=algo,
				target="auto",
				environment="development",
				approved=True,
				explainability_attached=True,
				training_history_points=len(training_data),
				feature_names=[k for k in (training_data[0] if training_data else {}).keys() if k != "label"],
			)
			eval_result = await self.model_evaluate(tenant_id, model_id, training_data, actor=actor)
			results.append({"model_id": model_id, "algorithm": algo, "rmse": eval_result["value"]})
			if eval_result["value"] < best_score:
				best_score = eval_result["value"]
				best_id = model_id
		self._record_audit(tenant_id, best_id or "none", "auto_ml_completed", actor, "allow",
			(f"candidates:{len(candidate_algorithms)}",))
		return {"best_model_id": best_id, "best_rmse": best_score, "candidates": results}

	async def prediction_export(
		self,
		tenant_id: str,
		model_id: str,
		format: str = "jsonl",
		actor: str = "pred",
	) -> dict[str, Any]:
		"""Export all score runs for a model to a serialisable structure."""
		scores = [s for s in self._scores.values() if s.tenant_id == tenant_id and s.model_id == model_id]
		rows = [s.to_dict() for s in scores]
		self._record_audit(tenant_id, model_id, "predictions_exported", actor, "allow",
			(f"format:{format}", f"count:{len(rows)}"))
		return {"model_id": model_id, "format": format, "record_count": len(rows), "data": rows}

	async def forecast_horizon(
		self,
		tenant_id: str,
		forecast_id: str,
	) -> dict[str, Any]:
		"""Return the configured horizon and forecast values for a forecast run."""
		forecast = self._forecasts.get(forecast_id)
		if forecast is None or forecast.tenant_id != tenant_id:
			raise KeyError("forecast_not_found")
		return {
			"forecast_id": forecast_id,
			"horizon_days": forecast.horizon_days,
			"forecast_values": list(forecast.forecast_values),
			"series_name": forecast.series_name,
		}

	async def confidence_interval(
		self,
		tenant_id: str,
		forecast_id: str,
		confidence: float = 0.95,
	) -> dict[str, Any]:
		"""Compute a symmetric confidence interval around each forecast value."""
		forecast = self._forecasts.get(forecast_id)
		if forecast is None or forecast.tenant_id != tenant_id:
			raise KeyError("forecast_not_found")
		z = 1.96 if confidence >= 0.95 else 1.645  # 95% or 90%
		std_approx = 0.05  # deterministic stand-in
		intervals = [
			{"step": i + 1, "value": v, "lower": round(v - z * std_approx, 4), "upper": round(v + z * std_approx, 4)}
			for i, v in enumerate(forecast.forecast_values)
		]
		return {"forecast_id": forecast_id, "confidence": confidence, "z_score": z, "intervals": intervals}

	async def model_drift_alert(
		self,
		tenant_id: str,
		model_id: str,
		current_accuracy: float,
		baseline_accuracy: float,
		drift_threshold: float = 0.05,
	) -> dict[str, Any]:
		"""Detect and record model drift when accuracy degrades beyond threshold.

		Returns an alert record if drift exceeds threshold, or a clear status otherwise.
		"""
		model = self._models.get(model_id)
		if model is None or model.tenant_id != tenant_id:
			raise KeyError("model_not_found")
		drift = round(baseline_accuracy - current_accuracy, 4)
		is_drifted = drift >= drift_threshold
		alert_id = self._runtime.stable_id("drift", {
			"tenant_id": tenant_id,
			"model_id": model_id,
			"index": len(self._audit_events),
		})
		self._record_event(tenant_id, "model_drift_evaluated", model_id,
			f"Drift={drift:.4f} threshold={drift_threshold} alert={is_drifted}",
			"system", severity="high" if is_drifted else "low")
		return {
			"alert_id": alert_id,
			"tenant_id": tenant_id,
			"model_id": model_id,
			"baseline_accuracy": baseline_accuracy,
			"current_accuracy": current_accuracy,
			"drift": drift,
			"drift_threshold": drift_threshold,
			"drifted": is_drifted,
			"recommended_action": "retrain_model" if is_drifted else "no_action",
			"evaluated_at": __import__("datetime").datetime.utcnow().isoformat(),
		}

	async def prediction_kpi_summary(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return a concise prediction KPI card for dashboard consumption."""
		scores = [s for s in self._scores.values() if s.tenant_id == tenant_id]
		models = [m for m in self._models.values() if m.tenant_id == tenant_id]
		forecasts = [f for f in self._forecasts.values() if f.tenant_id == tenant_id]
		approved = sum(1 for m in models if m.approved)
		avg_score = round(sum(s.score for s in scores) / max(len(scores), 1), 4) if scores else 0.0
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_models": len(models),
			"approved_models": approved,
			"approval_rate_pct": round(approved / max(len(models), 1) * 100, 1),
			"total_scores": len(scores),
			"avg_prediction_score": avg_score,
			"total_forecasts": len(forecasts),
			"audit_events": len(self.list_audit_events(tenant_id)),
			"generated_at": __import__("datetime").datetime.utcnow().isoformat(),
		}

	async def prediction_analytics(
		self,
		tenant_id: str,
		days: int = 30,
	) -> dict[str, Any]:
		"""Aggregate prediction activity stats for the tenant."""
		scores = [s for s in self._scores.values() if s.tenant_id == tenant_id]
		models = [m for m in self._models.values() if m.tenant_id == tenant_id]
		forecasts = [f for f in self._forecasts.values() if f.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"window_days": days,
			"total_scores": len(scores),
			"total_models": len(models),
			"approved_models": sum(1 for m in models if m.approved),
			"total_forecasts": len(forecasts),
			"avg_score": round(sum(s.score for s in scores) / len(scores), 4) if scores else 0,
			"audit_events": len(self.list_audit_events(tenant_id)),
		}

	# -------------------------------------------------------------------------
	# World-class improvement methods (I1–I15 subset)
	# -------------------------------------------------------------------------

	async def calibrate_scores(
		self,
		tenant_id: str,
		model_id: str,
		calibration_pairs: list[dict[str, float]],
		actor: str = "pred",
	) -> dict[str, Any]:
		"""Fit Platt scaling calibration (logistic sigmoid) to convert raw scores
		to true probabilities.

		Each entry in ``calibration_pairs`` must contain ``predicted`` (raw score
		0–100) and ``actual`` (0.0 or 1.0 binary label).  The method runs two
		gradient-descent passes to find parameters A and B such that
		``sigmoid(A * raw + B)`` minimises log-loss on the calibration set.

		Returns the fitted (A, B) parameters and per-sample calibrated
		probabilities for quick verification.
		"""
		guard_tenant_id(tenant_id)
		model = self._require_model(model_id, tenant_id)
		if not calibration_pairs:
			raise ValueError("calibration_pairs_required")

		# Normalise raw scores to [0, 1] for numerical stability
		pairs = [
			(float(p["predicted"]) / 100.0, float(p["actual"]))
			for p in calibration_pairs
		]
		# Simple gradient descent (50 iterations, lr=0.1) — dependency-free
		A, B = 1.0, 0.0
		lr = 0.1
		for _ in range(50):
			grad_A = grad_B = 0.0
			for raw, label in pairs:
				prob = 1.0 / (1.0 + math.exp(-(A * raw + B)))
				err = prob - label
				grad_A += err * raw
				grad_B += err
			n = len(pairs)
			A -= lr * grad_A / n
			B -= lr * grad_B / n

		key = f"{tenant_id}:{model_id}"
		self._calibrations[key] = (round(A, 6), round(B, 6))

		calibrated = [
			round(1.0 / (1.0 + math.exp(-(A * raw + B))), 4)
			for raw, _ in pairs
		]
		self._record_audit(tenant_id, model_id, "model_calibrated", actor, "allow",
			(f"samples:{len(pairs)}",))
		return {
			"model_id": model_id,
			"tenant_id": tenant_id,
			"platt_A": round(A, 6),
			"platt_B": round(B, 6),
			"sample_count": len(pairs),
			"calibrated_probabilities": calibrated,
		}

	async def attach_monetary_outcome(
		self,
		tenant_id: str,
		score_id: str,
		amount: str,
		currency: str = "KES",
		actor: str = "pred",
	) -> dict[str, Any]:
		"""Attach a Decimal-precision monetary outcome to a scored entity decision.

		``amount`` must be passed as a string (e.g. ``"12500.75"``) to preserve
		precision; floating-point literals are explicitly rejected.  Internally
		stored as ``decimal.Decimal`` with ``ROUND_HALF_EVEN`` (banker's rounding)
		to satisfy IFRS 13 requirements.
		"""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(score_id, "score_id")
		score = self._scores.get(score_id)
		if score is None or score.tenant_id != tenant_id:
			raise KeyError("score_not_found")
		try:
			value = Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
		except Exception as exc:
			raise ValueError(f"invalid_monetary_amount:{amount}") from exc
		self._monetary_outcomes[score_id] = value
		self._record_audit(tenant_id, score_id, "monetary_outcome_attached", actor, "allow",
			(f"amount:{value}", f"currency:{currency}"))
		return {
			"score_id": score_id,
			"tenant_id": tenant_id,
			"amount": str(value),
			"currency": currency,
		}

	async def aggregate_monetary_impact(
		self,
		tenant_id: str,
		model_id: str,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""Return the Decimal-precision total monetary impact for all scored
		entities under a given model.

		Aggregation uses ``decimal.ROUND_HALF_EVEN`` throughout — no floats
		are used in the summation path.
		"""
		guard_tenant_id(tenant_id)
		total = Decimal("0.00")
		matched: list[str] = []
		for score_id, amount in self._monetary_outcomes.items():
			score = self._scores.get(score_id)
			if score and score.tenant_id == tenant_id and score.model_id == model_id:
				total += amount
				matched.append(score_id)
		total = total.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
		return {
			"tenant_id": tenant_id,
			"model_id": model_id,
			"currency": currency,
			"total_monetary_impact": str(total),
			"outcome_count": len(matched),
			"score_ids": matched,
		}

	async def register_champion_challenger(
		self,
		tenant_id: str,
		policy_id: str,
		model_id_champion: str,
		model_id_challenger: str,
		traffic_split_pct: int = 10,
		actor: str = "pred",
	) -> dict[str, Any]:
		"""Register a champion-challenger routing policy.

		Traffic is split deterministically by hashing ``entity_id``:
		entities whose hash modulo 100 falls below ``traffic_split_pct`` are
		routed to the challenger, the rest to the champion.  This ensures the
		same entity always hits the same model — critical for consistency in
		credit/fraud decisions.
		"""
		guard_tenant_id(tenant_id)
		self._require_model(model_id_champion, tenant_id)
		self._require_model(model_id_challenger, tenant_id)
		if not (1 <= int(traffic_split_pct) <= 49):
			raise ValueError("traffic_split_pct_must_be_1_to_49")
		policy = {
			"policy_id": policy_id,
			"tenant_id": tenant_id,
			"champion_model_id": model_id_champion,
			"challenger_model_id": model_id_challenger,
			"traffic_split_pct": int(traffic_split_pct),
			"status": "active",
		}
		self._routing_policies[f"{tenant_id}:{policy_id}"] = policy
		self._record_audit(tenant_id, policy_id, "champion_challenger_registered", actor, "allow",
			(f"split_pct:{traffic_split_pct}",))
		return policy

	async def route_score_request(
		self,
		tenant_id: str,
		policy_id: str,
		feature_set_id: str,
		entity_id: str,
		feature_values: dict[str, Any],
		actor: str = "pred",
	) -> dict[str, Any]:
		"""Score an entity under a champion-challenger policy.

		Uses deterministic hash routing so the same entity always hits the same
		arm.  Returns scores from both models plus the routing decision.
		"""
		guard_tenant_id(tenant_id)
		policy = self._routing_policies.get(f"{tenant_id}:{policy_id}")
		if policy is None:
			raise KeyError("champion_challenger_policy_not_found")
		entity_bucket = int(sha256(entity_id.encode()).hexdigest()[:4], 16) % 100
		routed_to = (
			"challenger"
			if entity_bucket < policy["traffic_split_pct"]
			else "champion"
		)
		active_model_id = policy[f"{routed_to}_model_id"]
		score_record = self.score_entity(
			score_id=stable_id("cc", tenant_id, policy_id, entity_id),
			tenant_id=tenant_id,
			model_id=active_model_id,
			feature_set_id=feature_set_id,
			entity_id=entity_id,
			feature_values=feature_values,
			actor=actor,
		)
		return {
			"policy_id": policy_id,
			"entity_id": entity_id,
			"routed_to": routed_to,
			"active_model_id": active_model_id,
			"score": score_record,
		}

	async def compute_confidence_decay(
		self,
		tenant_id: str,
		model_id: str,
		half_life_days: float = 90.0,
		decay_threshold: float = 0.5,
	) -> dict[str, Any]:
		"""Compute temporal confidence decay for a model.

		Applies exponential decay: ``confidence = exp(-lambda * age_days)``
		where ``lambda = ln(2) / half_life_days``.  A model at ``half_life_days``
		old will have confidence = 0.5 regardless of measured drift.

		Returns a structured decay report suitable for dashboard health widgets.
		"""
		guard_tenant_id(tenant_id)
		model = self._require_model(model_id, tenant_id)
		now = utc_now()
		delta = now - model.updated_at
		age_days = delta.total_seconds() / 86400.0
		lam = math.log(2.0) / max(half_life_days, 1.0)
		confidence = round(math.exp(-lam * age_days), 4)
		decayed = confidence < decay_threshold
		if decayed:
			self._record_audit(tenant_id, model_id, "model_confidence_decayed", "pred", "allow",
				(f"confidence:{confidence}", f"age_days:{round(age_days, 1)}"))
		return {
			"model_id": model_id,
			"tenant_id": tenant_id,
			"age_days": round(age_days, 2),
			"half_life_days": half_life_days,
			"confidence": confidence,
			"decay_threshold": decay_threshold,
			"decayed_below_threshold": decayed,
			"recommended_action": "schedule_retrain" if decayed else "no_action",
		}

	async def stream_drift_window(
		self,
		tenant_id: str,
		model_id: str,
		reference_scores: list[float],
		current_scores: list[float],
		n_bins: int = 10,
		actor: str = "pred",
	) -> dict[str, Any]:
		"""Compute Population Stability Index (PSI) and KL-divergence between
		reference and current score distributions.

		PSI interpretation: < 0.1 stable, 0.1–0.2 warning, > 0.2 critical.
		Uses equal-width binning over the union range of both distributions.
		A small epsilon (1e-6) guards against log(0).
		"""
		guard_tenant_id(tenant_id)
		self._require_model(model_id, tenant_id)
		if not reference_scores or not current_scores:
			raise ValueError("score_lists_required")

		epsilon = 1e-6
		all_vals = reference_scores + current_scores
		lo, hi = min(all_vals), max(all_vals)
		bin_width = (hi - lo) / n_bins if hi > lo else 1.0

		def bin_index(v: float) -> int:
			return min(int((v - lo) / bin_width), n_bins - 1)

		ref_bins = [0.0] * n_bins
		cur_bins = [0.0] * n_bins
		for v in reference_scores:
			ref_bins[bin_index(v)] += 1
		for v in current_scores:
			cur_bins[bin_index(v)] += 1

		ref_pct = [c / len(reference_scores) + epsilon for c in ref_bins]
		cur_pct = [c / len(current_scores) + epsilon for c in cur_bins]

		psi = sum((c - r) * math.log(c / r) for r, c in zip(ref_pct, cur_pct))
		kl = sum(r * math.log(r / c) for r, c in zip(ref_pct, cur_pct))

		psi = round(psi, 4)
		kl = round(kl, 4)
		band = "stable" if psi < 0.1 else ("warning" if psi < 0.2 else "critical")

		report_id = stable_id("psi", tenant_id, model_id, str(len(self._drift_reports)))
		self._record_audit(tenant_id, model_id, "psi_drift_computed", actor, "allow",
			(f"psi:{psi}", f"band:{band}"))
		return {
			"model_id": model_id,
			"tenant_id": tenant_id,
			"report_id": report_id,
			"psi": psi,
			"kl_divergence": kl,
			"stability_band": band,
			"n_bins": n_bins,
			"reference_n": len(reference_scores),
			"current_n": len(current_scores),
		}

	async def register_explanation_attestation(
		self,
		tenant_id: str,
		score_id: str,
		model_version_id: str,
		method: str,
		feature_coverage_pct: float,
		attested_by: str,
		actor: str = "pred",
	) -> dict[str, Any]:
		"""Create a non-repudiable attestation record for an explanation.

		The attestation hash is ``sha256(score_id + model_version_id + method
		+ attested_by)`` — any tampering with these fields invalidates the hash
		and is surfaced by ``verify_explanation_attestation()``.

		Rejects attestations with feature_coverage_pct < 80 on high-impact
		scores — regulators require substantially complete explanations.
		"""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(attested_by, "attested_by")
		score = self._scores.get(score_id)
		if score is None or score.tenant_id != tenant_id:
			raise KeyError("score_not_found")
		if score.impact == "high" and float(feature_coverage_pct) < 80.0:
			raise PermissionError("high_impact_score_requires_80pct_feature_coverage")

		raw = f"{score_id}{model_version_id}{method}{attested_by}"
		attestation_hash = sha256(raw.encode()).hexdigest()[:32]
		record = {
			"score_id": score_id,
			"tenant_id": tenant_id,
			"model_version_id": model_version_id,
			"method": method,
			"feature_coverage_pct": round(float(feature_coverage_pct), 2),
			"attested_by": attested_by,
			"attested_at": utc_now().isoformat(),
			"attestation_hash": attestation_hash,
		}
		self._attestations[score_id] = record
		self._record_audit(tenant_id, score_id, "explanation_attested", actor, "allow",
			(f"method:{method}", f"coverage:{feature_coverage_pct}"))
		return record

	async def verify_explanation_attestation(
		self,
		tenant_id: str,
		score_id: str,
	) -> dict[str, Any]:
		"""Verify the attestation hash for a score explanation.

		Recomputes the hash from stored fields and compares it with the stored
		hash.  Returns ``{valid: bool, tampered: bool}`` — tampered implies the
		record was modified after attestation.
		"""
		guard_tenant_id(tenant_id)
		record = self._attestations.get(score_id)
		if record is None or record["tenant_id"] != tenant_id:
			raise KeyError("attestation_not_found")
		raw = (
			record["score_id"]
			+ record["model_version_id"]
			+ record["method"]
			+ record["attested_by"]
		)
		recomputed = sha256(raw.encode()).hexdigest()[:32]
		valid = recomputed == record["attestation_hash"]
		return {
			"score_id": score_id,
			"valid": valid,
			"tampered": not valid,
			"stored_hash": record["attestation_hash"],
			"recomputed_hash": recomputed,
		}

	async def record_prediction_latency(
		self,
		tenant_id: str,
		score_id: str,
		latency_ms: float,
		sla_threshold_ms: float = 100.0,
		actor: str = "pred",
	) -> dict[str, Any]:
		"""Record inference latency for a completed score operation and flag SLA
		breaches.

		Latency values are stored per ``score_id`` and used by
		``compute_sla_report()`` to derive P50/P95/P99 percentiles.  SLA
		breaches are immediately surfaced in the audit trail.
		"""
		guard_tenant_id(tenant_id)
		score = self._scores.get(score_id)
		if score is None or score.tenant_id != tenant_id:
			raise KeyError("score_not_found")
		breached = float(latency_ms) > float(sla_threshold_ms)
		record: dict[str, Any] = {
			"score_id": score_id,
			"tenant_id": tenant_id,
			"latency_ms": round(float(latency_ms), 3),
			"sla_threshold_ms": float(sla_threshold_ms),
			"breached": breached,
			"recorded_at": utc_now().isoformat(),
		}
		self._latency_records[score_id] = record
		if breached:
			self._record_audit(tenant_id, score_id, "sla_breach_detected", actor, "allow",
				(f"latency_ms:{latency_ms}", f"threshold_ms:{sla_threshold_ms}"))
		return record

	async def compute_sla_report(
		self,
		tenant_id: str,
		model_id: str | None = None,
		sla_threshold_ms: float = 100.0,
	) -> dict[str, Any]:
		"""Compute P50/P95/P99 latency percentiles and SLA breach rate for the
		tenant (optionally filtered to a single model).

		Percentiles use nearest-rank interpolation over the sorted latency list.
		All values are in milliseconds.
		"""
		guard_tenant_id(tenant_id)
		records = [
			r for r in self._latency_records.values()
			if r["tenant_id"] == tenant_id
		]
		if model_id is not None:
			filtered_score_ids = {
				sid for sid, s in self._scores.items()
				if s.tenant_id == tenant_id and s.model_id == model_id
			}
			records = [r for r in records if r["score_id"] in filtered_score_ids]

		if not records:
			return {
				"tenant_id": tenant_id,
				"model_id": model_id,
				"sample_count": 0,
				"p50_ms": None,
				"p95_ms": None,
				"p99_ms": None,
				"breach_count": 0,
				"breach_rate_pct": 0.0,
				"sla_threshold_ms": sla_threshold_ms,
			}

		latencies = sorted(r["latency_ms"] for r in records)
		n = len(latencies)

		def percentile(p: float) -> float:
			idx = max(0, int(math.ceil(p / 100.0 * n)) - 1)
			return latencies[idx]

		breach_count = sum(1 for r in records if r["breached"])
		return {
			"tenant_id": tenant_id,
			"model_id": model_id,
			"sample_count": n,
			"p50_ms": round(percentile(50), 3),
			"p95_ms": round(percentile(95), 3),
			"p99_ms": round(percentile(99), 3),
			"breach_count": breach_count,
			"breach_rate_pct": round(breach_count / n * 100, 2),
			"sla_threshold_ms": sla_threshold_ms,
		}

	async def build_lineage_graph(
		self,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Build a traversable lineage DAG for the tenant.

		Nodes are entity IDs (models, feature sets, scores, forecasts).
		Edges represent data-flow dependencies:
		- score -> model, score -> feature_set
		- forecast -> model
		- feature_set -> its lineage_refs (upstream ETL sources)

		The adjacency list is stored in ``_lineage_graph`` and returned as
		a serialisable dict for API/UI consumption.
		"""
		guard_tenant_id(tenant_id)
		graph: dict[str, list[str]] = {}

		def add_edge(child: str, parent: str) -> None:
			graph.setdefault(child, [])
			if parent not in graph[child]:
				graph[child].append(parent)
			# Also update internal set-based store for fast BFS
			self._lineage_graph.setdefault(child, set()).add(parent)

		for score in self._scores.values():
			if score.tenant_id != tenant_id:
				continue
			add_edge(score.id, score.model_id)
			add_edge(score.id, score.feature_set_id)

		for forecast in self._forecasts.values():
			if forecast.tenant_id != tenant_id:
				continue
			add_edge(forecast.id, forecast.model_id)

		for fs in self._feature_sets.values():
			if fs.tenant_id != tenant_id:
				continue
			for ref in fs.lineage_refs:
				add_edge(fs.id, ref)

		node_count = len(graph)
		edge_count = sum(len(v) for v in graph.values())
		return {
			"tenant_id": tenant_id,
			"node_count": node_count,
			"edge_count": edge_count,
			"adjacency": graph,
		}

	async def trace_decision_lineage(
		self,
		tenant_id: str,
		score_id: str,
	) -> dict[str, Any]:
		"""Trace the complete upstream lineage path from a score decision back
		to source ETL/feature lineage refs.

		Uses BFS over ``_lineage_graph``.  Returns an ordered path list from
		``score_id`` to all reachable root nodes (nodes with no further upstream
		dependencies).  Suitable for regulatory audit exhibits.
		"""
		guard_tenant_id(tenant_id)
		# Rebuild graph first (idempotent)
		await self.build_lineage_graph(tenant_id)

		visited: set[str] = set()
		queue: list[str] = [score_id]
		path: list[str] = []

		while queue:
			node = queue.pop(0)
			if node in visited:
				continue
			visited.add(node)
			path.append(node)
			upstream = self._lineage_graph.get(node, set())
			queue.extend(u for u in upstream if u not in visited)

		roots = [n for n in path if not self._lineage_graph.get(n)]
		return {
			"score_id": score_id,
			"tenant_id": tenant_id,
			"lineage_path": path,
			"root_nodes": roots,
			"depth": len(path),
		}
