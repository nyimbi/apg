"""Service layer for executable Predictive Analytics operations."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	DriftReport,
	FeatureSet,
	ForecastRun,
	PredAuditEvent,
	PredictiveModel,
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


class PredService:
	"""In-process forecasting, scoring, simulation, drift, and governance service."""

	def __init__(self) -> None:
		self._models: dict[str, PredictiveModel] = {}
		self._feature_sets: dict[str, FeatureSet] = {}
		self._forecasts: dict[str, ForecastRun] = {}
		self._scores: dict[str, ScoreRun] = {}
		self._scenarios: dict[str, ScenarioSimulation] = {}
		self._drift_reports: dict[str, DriftReport] = {}
		self._audit_events: dict[str, PredAuditEvent] = {}

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
		self._raise_if_blocked(result)
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
			status="approved" if approved else "registered",
			metadata=dict(metadata or {}),
		)
		self._models[model.id] = model
		self._record_audit(tenant_id, model.id, "model_registered", owner, "allow")
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
		self._raise_if_blocked(result)
		model.approved = True
		model.explainability_attached = model.explainability_attached or bool(explainability_ref)
		model.status = "approved"
		model.updated_at = utc_now()
		self._record_audit(tenant_id, model.id, "model_approved", approver, "allow")
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
			status="review_required" if result["decision"] == "require_review" else "active",
		)
		self._feature_sets[feature_set.id] = feature_set
		self._record_audit(tenant_id, feature_set.id, "feature_set_registered", owner, "allow")
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
		self._raise_if_blocked(result)
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
		)
		self._forecasts[forecast.id] = forecast
		self._record_audit(tenant_id, forecast.id, "forecast_created", actor, "allow")
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
		self._raise_if_blocked(result)
		report = DriftReport(
			id=report_id,
			tenant_id=tenant_id,
			model_id=model.id,
			metric_name=metric_name,
			drift_score=round(float(drift_score), 4),
			threshold=round(float(threshold), 4),
			status=drift_status(drift_score, threshold),
			review_recorded=bool(review_recorded),
		)
		self._drift_reports[report.id] = report
		self._record_audit(tenant_id, report.id, "drift_recorded", actor, report.status)
		return report.to_dict()

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
			"drift_review_count": sum(1 for report in self._drift_reports.values() if report.tenant_id == tenant_id and report.status == "review_required"),
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
