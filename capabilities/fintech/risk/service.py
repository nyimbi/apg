"""Executable service layer for APG FinTech Risk Management."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CONTROL_TYPES, SUPPORTED_CURRENCIES, SUPPORTED_EVENT_TYPES, SUPPORTED_EXPOSURE_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_DOMAINS, SUPPORTED_SCENARIO_TYPES, SUPPORTED_SEVERITIES, SUPPORTED_SUBJECT_TYPES, evaluate_capability_rules, get_capability_contract
	from .models import LimitBreach, RiskAppetite, RiskControl, RiskEvent, RiskEvidence, RiskExposure, RiskProfile, RiskReview, StressScenario
	from .risk_runtime import normalize_code, normalize_currency, positive_minor, probability_bps_valid, risk_band, score_valid
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CONTROL_TYPES, SUPPORTED_CURRENCIES, SUPPORTED_EVENT_TYPES, SUPPORTED_EXPOSURE_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_DOMAINS, SUPPORTED_SCENARIO_TYPES, SUPPORTED_SEVERITIES, SUPPORTED_SUBJECT_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import LimitBreach, RiskAppetite, RiskControl, RiskEvent, RiskEvidence, RiskExposure, RiskProfile, RiskReview, StressScenario  # type: ignore
	from risk_runtime import normalize_code, normalize_currency, positive_minor, probability_bps_valid, risk_band, score_valid  # type: ignore


class RiskManagementService:
	"""Dependency-light FinTech risk runtime for generated APG applications."""

	def __init__(self) -> None:
		self.appetites: dict[str, RiskAppetite] = {}
		self.profiles: dict[str, RiskProfile] = {}
		self.exposures: dict[str, RiskExposure] = {}
		self.controls: dict[str, RiskControl] = {}
		self.scenarios: dict[str, StressScenario] = {}
		self.breaches: dict[str, LimitBreach] = {}
		self.events: dict[str, RiskEvent] = {}
		self.reviews: dict[str, RiskReview] = {}
		self.evidence: dict[str, RiskEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_appetite(self, appetite_id: str, tenant_id: str, risk_domain: str, threshold_minor: int, currency: str, owner_id: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		risk_domain = normalize_code(risk_domain)
		currency = normalize_currency(currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_appetite", "domain_supported": risk_domain in SUPPORTED_RISK_DOMAINS, "positive_threshold": positive_minor(threshold_minor), "owner_present": bool(owner_id), "evidence_present": bool(evidence_reference)})
		item = RiskAppetite(appetite_id, tenant_id, risk_domain, int(threshold_minor), currency, owner_id, evidence_reference)
		self.appetites[appetite_id] = item
		self._audit(tenant_id, "risk_appetite_registered", appetite_id)
		return item.to_dict()

	def create_profile(self, profile_id: str, tenant_id: str, subject_reference: str, subject_type: str, kyc_reference: str, exposure_minor: int, currency: str, risk_score: float, source_reference: str) -> dict[str, Any]:
		subject_type = normalize_code(subject_type)
		currency = normalize_currency(currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "create_profile", "subject_present": bool(subject_reference), "subject_type_supported": subject_type in SUPPORTED_SUBJECT_TYPES, "kyc_present": bool(kyc_reference), "score_valid": score_valid(risk_score), "currency_supported": currency in SUPPORTED_CURRENCIES, "source_present": bool(source_reference)})
		item = RiskProfile(profile_id, tenant_id, subject_reference, subject_type, kyc_reference, int(exposure_minor), currency, float(risk_score), source_reference)
		self.profiles[profile_id] = item
		self._audit(tenant_id, "risk_profile_created", profile_id)
		return item.to_dict() | {"risk_band": risk_band(float(risk_score))}

	def record_exposure(self, exposure_id: str, tenant_id: str, profile_id: str, exposure_type: str, amount_minor: int, currency: str, limit_minor: int, source_reference: str, human_approval: str = "") -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		exposure_type = normalize_code(exposure_type)
		currency = normalize_currency(currency)
		over_limit = positive_minor(amount_minor) and positive_minor(limit_minor) and int(amount_minor) > int(limit_minor)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_exposure", "profile_present": profile is not None, "exposure_type_supported": exposure_type in SUPPORTED_EXPOSURE_TYPES, "positive_amount": positive_minor(amount_minor), "currency_supported": currency in SUPPORTED_CURRENCIES, "positive_limit": positive_minor(limit_minor), "source_present": bool(source_reference), "over_limit": over_limit, "human_approval_recorded": bool(human_approval)})
		item = RiskExposure(exposure_id, tenant_id, profile_id, exposure_type, int(amount_minor), currency, int(limit_minor), source_reference, "over_limit" if over_limit else "within_limit")
		self.exposures[exposure_id] = item
		self._audit(tenant_id, "risk_exposure_recorded", exposure_id)
		return item.to_dict()

	def evaluate_control(self, control_id: str, tenant_id: str, profile_id: str, control_type: str, owner_id: str, evidence_reference: str, effectiveness_score: float) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		control_type = normalize_code(control_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "evaluate_control", "profile_present": profile is not None, "control_type_supported": control_type in SUPPORTED_CONTROL_TYPES, "owner_present": bool(owner_id), "evidence_present": bool(evidence_reference), "effectiveness_score_valid": score_valid(effectiveness_score)})
		item = RiskControl(control_id, tenant_id, profile_id, control_type, owner_id, evidence_reference, float(effectiveness_score))
		self.controls[control_id] = item
		self._audit(tenant_id, "risk_control_evaluated", control_id)
		return item.to_dict()

	def run_stress_scenario(self, scenario_id: str, tenant_id: str, profile_id: str, scenario_type: str, impact_minor: int, probability_bps: int, mitigation_reference: str) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		scenario_type = normalize_code(scenario_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "run_stress_scenario", "profile_present": profile is not None, "scenario_type_supported": scenario_type in SUPPORTED_SCENARIO_TYPES, "positive_impact": positive_minor(impact_minor), "probability_valid": probability_bps_valid(probability_bps), "mitigation_present": bool(mitigation_reference)})
		item = StressScenario(scenario_id, tenant_id, profile_id, scenario_type, int(impact_minor), int(probability_bps), mitigation_reference)
		self.scenarios[scenario_id] = item
		self._audit(tenant_id, "risk_stress_scenario_recorded", scenario_id)
		return item.to_dict()

	def record_limit_breach(self, breach_id: str, tenant_id: str, exposure_id: str, severity: str, evidence_reference: str, remediation_owner: str) -> dict[str, Any]:
		exposure = self._tenant_exposure_or_none(exposure_id, tenant_id)
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_limit_breach", "exposure_present": exposure is not None, "severity_supported": severity in SUPPORTED_SEVERITIES, "evidence_present": bool(evidence_reference), "owner_present": bool(remediation_owner)})
		item = LimitBreach(breach_id, tenant_id, exposure_id, severity, evidence_reference, remediation_owner, "open")
		self.breaches[breach_id] = item
		self._audit(tenant_id, "risk_limit_breach_recorded", breach_id)
		return item.to_dict()

	def open_risk_event(self, event_id: str, tenant_id: str, profile_id: str, event_type: str, severity: str, evidence_reference: str) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		event_type = normalize_code(event_type)
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "open_risk_event", "profile_present": profile is not None, "event_type_supported": event_type in SUPPORTED_EVENT_TYPES, "severity_supported": severity in SUPPORTED_SEVERITIES, "evidence_present": bool(evidence_reference)})
		item = RiskEvent(event_id, tenant_id, profile_id, event_type, severity, evidence_reference, "open")
		self.events[event_id] = item
		self._audit(tenant_id, "risk_event_opened", event_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": bool(reviewer_id), "evidence_present": bool(evidence_reference)})
		item = RiskReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = item
		self._audit(tenant_id, "risk_review_recorded", review_id)
		return item.to_dict()

	def register_risk_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_risk_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = RiskEvidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self.evidence[agent_id] = item
		self._audit(tenant_id, "risk_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "risk_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "risk_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.risk.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "appetite_count": self._count(self.appetites, tenant_id), "profile_count": self._count(self.profiles, tenant_id), "exposure_count": self._count(self.exposures, tenant_id), "over_limit_count": sum(1 for item in self.exposures.values() if item.tenant_id == tenant_id and item.status == "over_limit"), "control_count": self._count(self.controls, tenant_id), "scenario_count": self._count(self.scenarios, tenant_id), "breach_count": self._count(self.breaches, tenant_id), "event_count": self._count(self.events, tenant_id), "review_count": self._count(self.reviews, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_profile_or_none(self, item_id: str, tenant_id: str) -> RiskProfile | None:
		item = self.profiles.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_exposure_or_none(self, item_id: str, tenant_id: str) -> RiskExposure | None:
		item = self.exposures.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "risk_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "risk_policy_denied")


FintechRiskService = RiskManagementService
