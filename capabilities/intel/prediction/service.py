"""Executable service layer for APG Predictive Intelligence."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_FORECAST_TYPES, SUPPORTED_HORIZONS, SUPPORTED_INDICATOR_TYPES, SUPPORTED_MODEL_TYPES, SUPPORTED_PROJECTION_TYPES, SUPPORTED_RECOMMENDATION_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SCENARIO_TYPES, SUPPORTED_WARNING_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract
	from .models import PredictionAgent, PredictionAuthority, PredictionForecast, PredictionIndicator, PredictionModel, PredictionProjection, PredictionRecommendation, PredictionReview, PredictionScenario, PredictionWarning, PredictionWorkspace
	from .prediction_runtime import bounded_score, normalize_code, positive_int, present
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_FORECAST_TYPES, SUPPORTED_HORIZONS, SUPPORTED_INDICATOR_TYPES, SUPPORTED_MODEL_TYPES, SUPPORTED_PROJECTION_TYPES, SUPPORTED_RECOMMENDATION_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SCENARIO_TYPES, SUPPORTED_WARNING_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import PredictionAgent, PredictionAuthority, PredictionForecast, PredictionIndicator, PredictionModel, PredictionProjection, PredictionRecommendation, PredictionReview, PredictionScenario, PredictionWarning, PredictionWorkspace  # type: ignore
	from prediction_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore


class PredictiveIntelligenceService:
	"""Tenant-scoped predictive-intelligence runtime for generated APG applications."""

	def __init__(self) -> None:
		self.authorities: dict[tuple[str, str], PredictionAuthority] = {}
		self.workspaces: dict[tuple[str, str], PredictionWorkspace] = {}
		self.scenarios: dict[tuple[str, str], PredictionScenario] = {}
		self.indicators: dict[tuple[str, str], PredictionIndicator] = {}
		self.models: dict[tuple[str, str], PredictionModel] = {}
		self.forecasts: dict[tuple[str, str], PredictionForecast] = {}
		self.projections: dict[tuple[str, str], PredictionProjection] = {}
		self.warnings: dict[tuple[str, str], PredictionWarning] = {}
		self.recommendations: dict[tuple[str, str], PredictionRecommendation] = {}
		self.reviews: dict[tuple[str, str], PredictionReview] = {}
		self.agents: dict[tuple[str, str], PredictionAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = PredictionAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "prediction_authority_recorded", authority_id)
		return item.to_dict()

	def record_workspace(self, workspace_id: str, tenant_id: str, workspace_type: str, name: str, classification: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		workspace_type = normalize_code(workspace_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_workspace", "workspace_type_supported": workspace_type in SUPPORTED_WORKSPACE_TYPES, "workspace_name_present": present(name), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = PredictionWorkspace(workspace_id, tenant_id, workspace_type, name, classification, authority_id, evidence_reference)
		self.workspaces[self._tenant_key(tenant_id, workspace_id)] = item
		self._audit(tenant_id, "prediction_workspace_recorded", workspace_id)
		return item.to_dict()

	def record_scenario(self, scenario_id: str, tenant_id: str, workspace_id: str, scenario_type: str, scenario_reference: str, horizon: str, owner_id: str, evidence_reference: str) -> dict[str, Any]:
		workspace = self._tenant_workspace_or_none(workspace_id, tenant_id)
		scenario_type = normalize_code(scenario_type)
		horizon = normalize_code(horizon)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_scenario", "workspace_present": workspace is not None, "scenario_type_supported": scenario_type in SUPPORTED_SCENARIO_TYPES, "scenario_reference_present": present(scenario_reference), "horizon_supported": horizon in SUPPORTED_HORIZONS, "owner_present": present(owner_id), "evidence_present": present(evidence_reference)})
		item = PredictionScenario(scenario_id, tenant_id, workspace_id, scenario_type, scenario_reference, horizon, owner_id, evidence_reference)
		self.scenarios[self._tenant_key(tenant_id, scenario_id)] = item
		self._audit(tenant_id, "prediction_scenario_recorded", scenario_id)
		return item.to_dict()

	def record_indicator(self, indicator_id: str, tenant_id: str, scenario_id: str, indicator_type: str, indicator_reference: str, confidence_score: float, evidence_reference: str) -> dict[str, Any]:
		scenario = self._tenant_scenario_or_none(scenario_id, tenant_id)
		indicator_type = normalize_code(indicator_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_indicator", "scenario_present": scenario is not None, "indicator_type_supported": indicator_type in SUPPORTED_INDICATOR_TYPES, "indicator_reference_present": present(indicator_reference), "confidence_valid": bounded_score(confidence_score), "evidence_present": present(evidence_reference)})
		item = PredictionIndicator(indicator_id, tenant_id, scenario_id, indicator_type, indicator_reference, float(confidence_score), evidence_reference)
		self.indicators[self._tenant_key(tenant_id, indicator_id)] = item
		self._audit(tenant_id, "prediction_indicator_recorded", indicator_id)
		return item.to_dict()

	def record_model(self, model_id: str, tenant_id: str, scenario_id: str, model_type: str, objective: str, validation_reference: str, risk_level: str, evidence_reference: str) -> dict[str, Any]:
		scenario = self._tenant_scenario_or_none(scenario_id, tenant_id)
		model_type = normalize_code(model_type)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_model", "scenario_present": scenario is not None, "model_type_supported": model_type in SUPPORTED_MODEL_TYPES, "objective_present": present(objective), "validation_present": present(validation_reference), "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "evidence_present": present(evidence_reference)})
		item = PredictionModel(model_id, tenant_id, scenario_id, model_type, objective, validation_reference, risk_level, evidence_reference)
		self.models[self._tenant_key(tenant_id, model_id)] = item
		self._audit(tenant_id, "prediction_model_recorded", model_id)
		return item.to_dict()

	def record_forecast(self, forecast_id: str, tenant_id: str, model_id: str, forecast_type: str, forecast_reference: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		model = self._tenant_model_or_none(model_id, tenant_id)
		forecast_type = normalize_code(forecast_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_forecast", "model_present": model is not None, "forecast_type_supported": forecast_type in SUPPORTED_FORECAST_TYPES, "forecast_reference_present": present(forecast_reference), "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = PredictionForecast(forecast_id, tenant_id, model_id, forecast_type, forecast_reference, float(confidence_score), analyst_id, evidence_reference)
		self.forecasts[self._tenant_key(tenant_id, forecast_id)] = item
		self._audit(tenant_id, "prediction_forecast_recorded", forecast_id)
		return item.to_dict()

	def record_projection(self, projection_id: str, tenant_id: str, forecast_id: str, projection_type: str, risk_level: str, probability_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		forecast = self._tenant_forecast_or_none(forecast_id, tenant_id)
		projection_type = normalize_code(projection_type)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_projection", "forecast_present": forecast is not None, "projection_type_supported": projection_type in SUPPORTED_PROJECTION_TYPES, "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "probability_valid": bounded_score(probability_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = PredictionProjection(projection_id, tenant_id, forecast_id, projection_type, risk_level, float(probability_score), analyst_id, evidence_reference)
		self.projections[self._tenant_key(tenant_id, projection_id)] = item
		self._audit(tenant_id, "prediction_projection_recorded", projection_id)
		return item.to_dict()

	def record_warning(self, warning_id: str, tenant_id: str, projection_id: str, warning_type: str, severity: str, trigger_reference: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		projection = self._tenant_projection_or_none(projection_id, tenant_id)
		warning_type = normalize_code(warning_type)
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_warning", "projection_present": projection is not None, "warning_type_supported": warning_type in SUPPORTED_WARNING_TYPES, "severity_supported": severity in SUPPORTED_RISK_LEVELS, "trigger_present": present(trigger_reference), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = PredictionWarning(warning_id, tenant_id, projection_id, warning_type, severity, trigger_reference, approval_reference, evidence_reference)
		self.warnings[self._tenant_key(tenant_id, warning_id)] = item
		self._audit(tenant_id, "prediction_warning_recorded", warning_id)
		return item.to_dict()

	def record_recommendation(self, recommendation_id: str, tenant_id: str, projection_id: str, recommendation_type: str, action_reference: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		projection = self._tenant_projection_or_none(projection_id, tenant_id)
		recommendation_type = normalize_code(recommendation_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_recommendation", "projection_present": projection is not None, "recommendation_type_supported": recommendation_type in SUPPORTED_RECOMMENDATION_TYPES, "action_present": present(action_reference), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = PredictionRecommendation(recommendation_id, tenant_id, projection_id, recommendation_type, action_reference, approval_reference, evidence_reference)
		self.recommendations[self._tenant_key(tenant_id, recommendation_id)] = item
		self._audit(tenant_id, "prediction_recommendation_recorded", recommendation_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = PredictionReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "prediction_review_recorded", reference_id)
		return item.to_dict()

	def register_prediction_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_prediction_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES, "agent_name_present": present(name), "agent_scope_present": present(scope)})
		item = PredictionAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "prediction_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, unsupported_automated_decision_scope: bool = False, hallucinated_forecast_scope: bool = False, privacy_bypass_scope: bool = False, unapproved_model_deployment_scope: bool = False, autonomous_warning_scope: bool = False, autonomous_recommendation_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "prediction_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "unsupported_automated_decision_scope": unsupported_automated_decision_scope, "hallucinated_forecast_scope": hallucinated_forecast_scope, "privacy_bypass_scope": privacy_bypass_scope, "unapproved_model_deployment_scope": unapproved_model_deployment_scope, "autonomous_warning_scope": autonomous_warning_scope, "autonomous_recommendation_scope": autonomous_recommendation_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "prediction_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.prediction.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "workspace_count": self._count(self.workspaces, tenant_id), "scenario_count": self._count(self.scenarios, tenant_id), "indicator_count": self._count(self.indicators, tenant_id), "model_count": self._count(self.models, tenant_id), "forecast_count": self._count(self.forecasts, tenant_id), "projection_count": self._count(self.projections, tenant_id), "warning_count": self._count(self.warnings, tenant_id), "recommendation_count": self._count(self.recommendations, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> PredictionAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_workspace_or_none(self, item_id: str, tenant_id: str) -> PredictionWorkspace | None:
		return self.workspaces.get(self._tenant_key(tenant_id, item_id))

	def _tenant_scenario_or_none(self, item_id: str, tenant_id: str) -> PredictionScenario | None:
		return self.scenarios.get(self._tenant_key(tenant_id, item_id))

	def _tenant_model_or_none(self, item_id: str, tenant_id: str) -> PredictionModel | None:
		return self.models.get(self._tenant_key(tenant_id, item_id))

	def _tenant_forecast_or_none(self, item_id: str, tenant_id: str) -> PredictionForecast | None:
		return self.forecasts.get(self._tenant_key(tenant_id, item_id))

	def _tenant_projection_or_none(self, item_id: str, tenant_id: str) -> PredictionProjection | None:
		return self.projections.get(self._tenant_key(tenant_id, item_id))

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
		reasons = ", ".join(action.get("reason", action.get("rule", "prediction_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "prediction_policy_denied")


IntelPredictionService = PredictiveIntelligenceService
