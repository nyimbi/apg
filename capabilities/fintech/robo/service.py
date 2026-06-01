"""Executable service layer for APG Robo Advisory."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CADENCES, SUPPORTED_CURRENCIES, SUPPORTED_GOAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_PROFILES, evaluate_capability_rules, get_capability_contract
	from .models import AutomationPlan, DriftRecord, GoalPlan, InvestorProfile, ModelPortfolio, RecommendationPacket, ReviewRecord, RoboEvidence, TaxLossCandidate
	from .robo_runtime import allocation_totals_100, normalize_code, normalize_currency, positive_minor
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CADENCES, SUPPORTED_CURRENCIES, SUPPORTED_GOAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_PROFILES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import AutomationPlan, DriftRecord, GoalPlan, InvestorProfile, ModelPortfolio, RecommendationPacket, ReviewRecord, RoboEvidence, TaxLossCandidate  # type: ignore
	from robo_runtime import allocation_totals_100, normalize_code, normalize_currency, positive_minor  # type: ignore


class RoboAdvisoryService:
	"""In-memory Robo Advisory runtime for generated APG applications."""

	def __init__(self) -> None:
		self.profiles: dict[str, InvestorProfile] = {}
		self.goals: dict[str, GoalPlan] = {}
		self.models: dict[str, ModelPortfolio] = {}
		self.recommendations: dict[str, RecommendationPacket] = {}
		self.automation: dict[str, AutomationPlan] = {}
		self.drift: dict[str, DriftRecord] = {}
		self.tax_loss: dict[str, TaxLossCandidate] = {}
		self.reviews: dict[str, ReviewRecord] = {}
		self.evidence: dict[str, RoboEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_investor_profile(self, profile_id: str, tenant_id: str, client_id: str, kyc_reference: str, suitability_reference: str, risk_profile: str, policy_attached: bool = True) -> dict[str, Any]:
		risk_profile = normalize_code(risk_profile)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "create_investor_profile", "client_present": bool(client_id), "kyc_present": bool(kyc_reference), "suitability_present": bool(suitability_reference), "risk_profile_supported": risk_profile in SUPPORTED_RISK_PROFILES})
		profile = InvestorProfile(profile_id, tenant_id, client_id, kyc_reference, suitability_reference, risk_profile)
		self.profiles[profile_id] = profile
		self._audit(tenant_id, "investor_profile_created", profile_id)
		return profile.to_dict()

	def define_goal_plan(self, goal_id: str, tenant_id: str, profile_id: str, goal_type: str, target_amount_minor: int, currency: str, horizon_date: str) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		goal_type = normalize_code(goal_type)
		currency = normalize_currency(currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "define_goal_plan", "profile_present": profile is not None, "goal_type_supported": goal_type in SUPPORTED_GOAL_TYPES, "positive_target": positive_minor(target_amount_minor), "currency_supported": currency in SUPPORTED_CURRENCIES, "horizon_present": bool(horizon_date)})
		goal = GoalPlan(goal_id, tenant_id, profile_id, goal_type, int(target_amount_minor), currency, horizon_date)
		self.goals[goal_id] = goal
		self._audit(tenant_id, "goal_plan_defined", goal_id)
		return goal.to_dict()

	def publish_model_portfolio(self, model_id: str, tenant_id: str, name: str, risk_profile: str, target_allocation: dict[str, float], policy_reference: str) -> dict[str, Any]:
		risk_profile = normalize_code(risk_profile)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "publish_model_portfolio", "risk_profile_supported": risk_profile in SUPPORTED_RISK_PROFILES, "allocation_totals_100": allocation_totals_100(target_allocation), "policy_present": bool(policy_reference)})
		model = ModelPortfolio(model_id, tenant_id, name, risk_profile, dict(target_allocation), policy_reference)
		self.models[model_id] = model
		self._audit(tenant_id, "model_portfolio_published", model_id)
		return model.to_dict()

	def generate_recommendation(self, recommendation_id: str, tenant_id: str, profile_id: str, goal_id: str, model_id: str, analysis_reference: str) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		goal = self._tenant_goal_or_none(goal_id, tenant_id)
		model = self._tenant_model_or_none(model_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "generate_recommendation", "profile_present": profile is not None, "goal_present": goal is not None, "model_present": model is not None, "analysis_present": bool(analysis_reference)})
		recommendation = RecommendationPacket(recommendation_id, tenant_id, profile_id, goal_id, model_id, analysis_reference)
		self.recommendations[recommendation_id] = recommendation
		self._audit(tenant_id, "recommendation_generated", recommendation_id)
		return recommendation.to_dict()

	def approve_recommendation(self, recommendation_id: str, tenant_id: str, reviewer_id: str) -> dict[str, Any]:
		recommendation = self._tenant_recommendation_or_none(recommendation_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "approve_recommendation", "recommendation_present": recommendation is not None, "reviewer_present": bool(reviewer_id)})
		assert recommendation is not None
		recommendation.status = "approved"
		self._audit(tenant_id, "recommendation_approved", recommendation_id)
		return recommendation.to_dict() | {"reviewer_id": reviewer_id}

	def configure_automation_plan(self, plan_id: str, tenant_id: str, recommendation_id: str, funding_source_reference: str, cadence: str) -> dict[str, Any]:
		recommendation = self._tenant_recommendation_or_none(recommendation_id, tenant_id)
		cadence = normalize_code(cadence)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "configure_automation_plan", "approved_recommendation_present": recommendation is not None and recommendation.status == "approved", "cadence_supported": cadence in SUPPORTED_CADENCES, "funding_source_present": bool(funding_source_reference)})
		plan = AutomationPlan(plan_id, tenant_id, recommendation_id, funding_source_reference, cadence)
		self.automation[plan_id] = plan
		self._audit(tenant_id, "automation_plan_configured", plan_id)
		return plan.to_dict()

	def record_drift(self, drift_id: str, tenant_id: str, profile_id: str, drift_bps: int, analysis_reference: str) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_drift", "profile_present": profile is not None, "analysis_present": bool(analysis_reference)})
		record = DriftRecord(drift_id, tenant_id, profile_id, int(drift_bps), analysis_reference)
		self.drift[drift_id] = record
		self._audit(tenant_id, "drift_recorded", drift_id)
		return record.to_dict()

	def record_tax_loss_candidate(self, candidate_id: str, tenant_id: str, profile_id: str, instrument_id: str, loss_minor: int, tax_lot_reference: str) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_tax_loss_candidate", "profile_present": profile is not None, "tax_lot_present": bool(tax_lot_reference), "positive_loss": positive_minor(loss_minor)})
		candidate = TaxLossCandidate(candidate_id, tenant_id, profile_id, instrument_id, int(loss_minor), tax_lot_reference)
		self.tax_loss[candidate_id] = candidate
		self._audit(tenant_id, "tax_loss_candidate_recorded", candidate_id)
		return candidate.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "evidence_present": bool(evidence_reference)})
		review = ReviewRecord(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = review
		self._audit(tenant_id, "robo_review_recorded", review_id)
		return review.to_dict()

	def register_robo_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_robo_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = RoboEvidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self.evidence[agent_id] = evidence
		self._audit(tenant_id, "robo_agent_registered", agent_id)
		return evidence.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "robo_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.robo.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "profile_count": self._count(self.profiles, tenant_id), "goal_count": self._count(self.goals, tenant_id), "model_count": self._count(self.models, tenant_id), "recommendation_count": self._count(self.recommendations, tenant_id), "automation_count": self._count(self.automation, tenant_id), "drift_count": self._count(self.drift, tenant_id), "tax_loss_count": self._count(self.tax_loss, tenant_id), "review_count": self._count(self.reviews, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_profile_or_none(self, item_id: str, tenant_id: str) -> InvestorProfile | None:
		item = self.profiles.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_goal_or_none(self, item_id: str, tenant_id: str) -> GoalPlan | None:
		item = self.goals.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_model_or_none(self, item_id: str, tenant_id: str) -> ModelPortfolio | None:
		item = self.models.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_recommendation_or_none(self, item_id: str, tenant_id: str) -> RecommendationPacket | None:
		item = self.recommendations.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "robo_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "robo_policy_denied")


RoboService = RoboAdvisoryService
