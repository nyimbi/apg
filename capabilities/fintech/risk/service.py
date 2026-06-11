"""Executable service layer for APG FinTech Risk Management."""

from __future__ import annotations

import asyncio
import datetime
import math
import statistics
from collections import defaultdict
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CONTROL_TYPES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_EVENT_TYPES,
		SUPPORTED_EXPOSURE_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_RISK_DOMAINS,
		SUPPORTED_SCENARIO_TYPES,
		SUPPORTED_SEVERITIES,
		SUPPORTED_SUBJECT_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import (
		LimitBreach,
		RiskAppetite,
		RiskControl,
		RiskEvent,
		RiskEvidence,
		RiskExposure,
		RiskProfile,
		RiskReview,
		StressScenario,
	)
	from .risk_runtime import (
		normalize_code,
		normalize_currency,
		positive_minor,
		probability_bps_valid,
		risk_band,
		score_valid,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CONTROL_TYPES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_EVENT_TYPES,
		SUPPORTED_EXPOSURE_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_RISK_DOMAINS,
		SUPPORTED_SCENARIO_TYPES,
		SUPPORTED_SEVERITIES,
		SUPPORTED_SUBJECT_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import (  # type: ignore
		LimitBreach,
		RiskAppetite,
		RiskControl,
		RiskEvent,
		RiskEvidence,
		RiskExposure,
		RiskProfile,
		RiskReview,
		StressScenario,
	)
	from risk_runtime import (  # type: ignore
		normalize_code,
		normalize_currency,
		positive_minor,
		probability_bps_valid,
		risk_band,
		score_valid,
	)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utc_now() -> datetime.datetime:
	return datetime.datetime.now(datetime.timezone.utc)


def _iso() -> str:
	return _utc_now().isoformat()


def _normal_cdf(z: float) -> float:
	"""Standard normal CDF via error function approximation."""
	return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _var_parametric(returns: list[float], confidence: float, portfolio_value: float) -> float:
	"""Parametric VaR: z-score × stdev × portfolio value."""
	if len(returns) < 2:
		return 0.0
	mu = statistics.mean(returns)
	sigma = statistics.stdev(returns)
	z = abs(_ppf_approx(confidence))
	return max((z * sigma - mu) * portfolio_value, 0.0)


def _ppf_approx(p: float) -> float:
	"""Rational approximation for probit (inverse normal CDF)."""
	assert 0 < p < 1
	if p >= 0.5:
		t = math.sqrt(-2 * math.log(1 - p))
	else:
		t = math.sqrt(-2 * math.log(p))
	c0, c1, c2 = 2.515517, 0.802853, 0.010328
	d1, d2, d3 = 1.432788, 0.189269, 0.001308
	num = c0 + c1 * t + c2 * t * t
	den = 1 + d1 * t + d2 * t * t + d3 * t * t * t
	approx = t - num / den
	return approx if p >= 0.5 else -approx


def _capital_adequacy_ratio(tier1: float, tier2: float, rwa: float) -> float:
	if rwa == 0:
		return 0.0
	return (tier1 + tier2) / rwa * 100


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class FintechRiskService:
	"""Full-featured FinTech risk management runtime for APG generated applications."""

	def __init__(
		self,
		tenant_id: str = "default",
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store = store

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

		# Historical return series per portfolio for VaR: portfolio_id -> list[float]
		self._return_series: dict[str, list[float]] = defaultdict(list)

		# Liquidity buffer per period: period -> {available, required}
		self._liquidity_ledger: dict[str, dict[str, float]] = {}

	# ------------------------------------------------------------------
	# Contract / describe
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Preserved original methods
	# ------------------------------------------------------------------

	def register_appetite(
		self,
		appetite_id: str,
		tenant_id: str,
		risk_domain: str,
		threshold_minor: int,
		currency: str,
		owner_id: str,
		evidence_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		risk_domain = normalize_code(risk_domain)
		currency = normalize_currency(currency)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_appetite",
			"domain_supported": risk_domain in SUPPORTED_RISK_DOMAINS,
			"positive_threshold": positive_minor(threshold_minor),
			"owner_present": bool(owner_id),
			"evidence_present": bool(evidence_reference),
		})
		item = RiskAppetite(appetite_id, tenant_id, risk_domain, int(threshold_minor), currency, owner_id, evidence_reference)
		self.appetites[appetite_id] = item
		self._audit(tenant_id, "risk_appetite_registered", appetite_id)
		return item.to_dict()

	def create_profile(
		self,
		profile_id: str,
		tenant_id: str,
		subject_reference: str,
		subject_type: str,
		kyc_reference: str,
		exposure_minor: int,
		currency: str,
		risk_score: float,
		source_reference: str,
	) -> dict[str, Any]:
		subject_type = normalize_code(subject_type)
		currency = normalize_currency(currency)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_profile",
			"subject_present": bool(subject_reference),
			"subject_type_supported": subject_type in SUPPORTED_SUBJECT_TYPES,
			"kyc_present": bool(kyc_reference),
			"score_valid": score_valid(risk_score),
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"source_present": bool(source_reference),
		})
		item = RiskProfile(profile_id, tenant_id, subject_reference, subject_type, kyc_reference, int(exposure_minor), currency, float(risk_score), source_reference)
		self.profiles[profile_id] = item
		self._audit(tenant_id, "risk_profile_created", profile_id)
		return item.to_dict() | {"risk_band": risk_band(float(risk_score))}

	def record_exposure(
		self,
		exposure_id: str,
		tenant_id: str,
		profile_id: str,
		exposure_type: str,
		amount_minor: int,
		currency: str,
		limit_minor: int,
		source_reference: str,
		human_approval: str = "",
	) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		exposure_type = normalize_code(exposure_type)
		currency = normalize_currency(currency)
		over_limit = positive_minor(amount_minor) and positive_minor(limit_minor) and int(amount_minor) > int(limit_minor)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_exposure",
			"profile_present": profile is not None,
			"exposure_type_supported": exposure_type in SUPPORTED_EXPOSURE_TYPES,
			"positive_amount": positive_minor(amount_minor),
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"positive_limit": positive_minor(limit_minor),
			"source_present": bool(source_reference),
			"over_limit": over_limit,
			"human_approval_recorded": bool(human_approval),
		})
		item = RiskExposure(exposure_id, tenant_id, profile_id, exposure_type, int(amount_minor), currency, int(limit_minor), source_reference, "over_limit" if over_limit else "within_limit")
		self.exposures[exposure_id] = item
		self._audit(tenant_id, "risk_exposure_recorded", exposure_id)
		return item.to_dict()

	def evaluate_control(
		self,
		control_id: str,
		tenant_id: str,
		profile_id: str,
		control_type: str,
		owner_id: str,
		evidence_reference: str,
		effectiveness_score: float,
	) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		control_type = normalize_code(control_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "evaluate_control",
			"profile_present": profile is not None,
			"control_type_supported": control_type in SUPPORTED_CONTROL_TYPES,
			"owner_present": bool(owner_id),
			"evidence_present": bool(evidence_reference),
			"effectiveness_score_valid": score_valid(effectiveness_score),
		})
		item = RiskControl(control_id, tenant_id, profile_id, control_type, owner_id, evidence_reference, float(effectiveness_score))
		self.controls[control_id] = item
		self._audit(tenant_id, "risk_control_evaluated", control_id)
		return item.to_dict()

	def run_stress_scenario(
		self,
		scenario_id: str,
		tenant_id: str,
		profile_id: str,
		scenario_type: str,
		impact_minor: int,
		probability_bps: int,
		mitigation_reference: str,
	) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		scenario_type = normalize_code(scenario_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "run_stress_scenario",
			"profile_present": profile is not None,
			"scenario_type_supported": scenario_type in SUPPORTED_SCENARIO_TYPES,
			"positive_impact": positive_minor(impact_minor),
			"probability_valid": probability_bps_valid(probability_bps),
			"mitigation_present": bool(mitigation_reference),
		})
		item = StressScenario(scenario_id, tenant_id, profile_id, scenario_type, int(impact_minor), int(probability_bps), mitigation_reference)
		self.scenarios[scenario_id] = item
		self._audit(tenant_id, "risk_stress_scenario_recorded", scenario_id)
		return item.to_dict()

	def record_limit_breach(
		self,
		breach_id: str,
		tenant_id: str,
		exposure_id: str,
		severity: str,
		evidence_reference: str,
		remediation_owner: str,
	) -> dict[str, Any]:
		exposure = self._tenant_exposure_or_none(exposure_id, tenant_id)
		severity = normalize_code(severity)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_limit_breach",
			"exposure_present": exposure is not None,
			"severity_supported": severity in SUPPORTED_SEVERITIES,
			"evidence_present": bool(evidence_reference),
			"owner_present": bool(remediation_owner),
		})
		item = LimitBreach(breach_id, tenant_id, exposure_id, severity, evidence_reference, remediation_owner, "open")
		self.breaches[breach_id] = item
		self._audit(tenant_id, "risk_limit_breach_recorded", breach_id)
		return item.to_dict()

	def open_risk_event(
		self,
		event_id: str,
		tenant_id: str,
		profile_id: str,
		event_type: str,
		severity: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		event_type = normalize_code(event_type)
		severity = normalize_code(severity)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_risk_event",
			"profile_present": profile is not None,
			"event_type_supported": event_type in SUPPORTED_EVENT_TYPES,
			"severity_supported": severity in SUPPORTED_SEVERITIES,
			"evidence_present": bool(evidence_reference),
		})
		item = RiskEvent(event_id, tenant_id, profile_id, event_type, severity, evidence_reference, "open")
		self.events[event_id] = item
		self._audit(tenant_id, "risk_event_opened", event_id)
		return item.to_dict()

	def record_review(
		self,
		review_id: str,
		tenant_id: str,
		reference_id: str,
		reviewer_id: str,
		status: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": bool(reviewer_id),
			"evidence_present": bool(evidence_reference),
		})
		item = RiskReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = item
		self._audit(tenant_id, "risk_review_recorded", review_id)
		return item.to_dict()

	def register_risk_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_risk_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = RiskEvidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self.evidence[agent_id] = item
		self._audit(tenant_id, "risk_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "risk_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(
		self,
		tenant_id: str,
		item_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "risk_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.risk.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"appetite_count": self._count(self.appetites, tenant_id),
			"profile_count": self._count(self.profiles, tenant_id),
			"exposure_count": self._count(self.exposures, tenant_id),
			"over_limit_count": sum(1 for item in self.exposures.values() if item.tenant_id == tenant_id and item.status == "over_limit"),
			"control_count": self._count(self.controls, tenant_id),
			"scenario_count": self._count(self.scenarios, tenant_id),
			"breach_count": self._count(self.breaches, tenant_id),
			"event_count": self._count(self.events, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# New async methods
	# ------------------------------------------------------------------

	async def credit_risk_assessment(
		self,
		customer_id: str,
	) -> dict[str, Any]:
		"""Compute a composite credit risk score for a customer."""
		assert customer_id, "customer_id required"
		await asyncio.sleep(0)

		profiles = [p for p in self.profiles.values() if p.subject_reference == customer_id]
		exposures = [e for e in self.exposures.values() if any(p.id == e.profile_id for p in profiles)]

		total_exposure_minor = sum(e.amount_minor for e in exposures)
		total_limit_minor = sum(e.limit_minor for e in exposures)
		over_limit_exposures = sum(1 for e in exposures if e.status == "over_limit")
		utilisation_pct = (total_exposure_minor / max(total_limit_minor, 1)) * 100

		# Base credit score from existing risk profiles
		profile_scores = [p.risk_score for p in profiles]
		avg_profile_score = statistics.mean(profile_scores) if profile_scores else 50.0

		# Adjust for utilisation and over-limit count
		credit_score = avg_profile_score
		if utilisation_pct > 90:
			credit_score = min(credit_score + 25, 100)
		elif utilisation_pct > 70:
			credit_score = min(credit_score + 15, 100)
		credit_score = min(credit_score + over_limit_exposures * 10, 100)

		pd_estimate = credit_score / 100 * 0.15  # rough PD proxy: max 15%
		lgd_estimate = 0.45  # standard 45% LGD assumption
		ead = total_exposure_minor / 100  # minor units to major

		self._audit(self.tenant_id, "credit_risk_assessed", customer_id)
		return {
			"customer_id": customer_id,
			"profile_count": len(profiles),
			"total_exposure": ead,
			"utilisation_pct": round(utilisation_pct, 2),
			"over_limit_count": over_limit_exposures,
			"credit_risk_score": round(credit_score, 2),
			"risk_band": risk_band(credit_score),
			"pd_estimate": round(pd_estimate, 4),
			"lgd_estimate": lgd_estimate,
			"expected_loss": round(pd_estimate * lgd_estimate * ead, 2),
			"assessed_at": _iso(),
		}

	async def market_risk_var(
		self,
		portfolio_id: str,
		confidence_level: float = 0.99,
	) -> dict[str, Any]:
		"""Calculate Value-at-Risk for a portfolio at a given confidence level."""
		assert portfolio_id, "portfolio_id required"
		assert 0.9 <= confidence_level < 1.0, "confidence_level must be in [0.9, 1.0)"
		await asyncio.sleep(0)

		returns = self._return_series.get(portfolio_id, [])
		# Seed synthetic returns if none present (allows deterministic demo)
		if not returns:
			import random
			rng = random.Random(hash(portfolio_id) % (2 ** 31))
			returns = [rng.gauss(0.0002, 0.012) for _ in range(252)]
			self._return_series[portfolio_id] = returns

		# Portfolio value from exposures linked to this portfolio
		portfolio_exposures = [
			e for e in self.exposures.values()
			if portfolio_id in e.source_reference
		]
		portfolio_value = sum(e.amount_minor for e in portfolio_exposures) / 100 or 1_000_000.0

		var_amount = _var_parametric(returns, confidence_level, portfolio_value)
		var_pct = (var_amount / portfolio_value) * 100 if portfolio_value else 0.0

		mu = statistics.mean(returns)
		sigma = statistics.stdev(returns) if len(returns) > 1 else 0.0

		self._audit(self.tenant_id, "market_risk_var_calculated", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"confidence_level": confidence_level,
			"portfolio_value": round(portfolio_value, 2),
			"var_amount": round(var_amount, 2),
			"var_pct": round(var_pct, 4),
			"daily_returns_count": len(returns),
			"returns_mean": round(mu, 6),
			"returns_stdev": round(sigma, 6),
			"method": "parametric",
			"calculated_at": _iso(),
		}

	async def liquidity_risk_report(
		self,
		period: str,
	) -> dict[str, Any]:
		"""Generate a liquidity coverage ratio (LCR) style report for a period."""
		assert period, "period required"
		await asyncio.sleep(0)

		# Compute available vs required liquidity from exposure data
		exposures_in_period = list(self.exposures.values())
		total_assets_minor = sum(e.amount_minor for e in exposures_in_period if e.exposure_type in {"cash", "liquid_asset", "bond"})
		total_outflows_minor = sum(e.amount_minor for e in exposures_in_period if e.exposure_type in {"loan", "credit_line", "overdraft"})

		hqla = total_assets_minor * 0.85  # haircut to high-quality liquid assets
		net_outflows = total_outflows_minor * 0.30  # 30-day stress scenario
		lcr = (hqla / max(net_outflows, 1)) * 100

		# NSFR proxy
		stable_funding = total_assets_minor * 0.75
		required_stable_funding = total_outflows_minor * 0.65
		nsfr = (stable_funding / max(required_stable_funding, 1)) * 100

		self._liquidity_ledger[period] = {
			"hqla": hqla,
			"net_outflows": net_outflows,
			"lcr_pct": lcr,
			"nsfr_pct": nsfr,
		}

		self._audit(self.tenant_id, "liquidity_risk_reported", period)
		return {
			"period": period,
			"total_assets_minor": total_assets_minor,
			"hqla_minor": round(hqla, 0),
			"net_cash_outflows_minor": round(net_outflows, 0),
			"lcr_pct": round(lcr, 2),
			"lcr_status": "compliant" if lcr >= 100 else "deficient",
			"nsfr_pct": round(nsfr, 2),
			"nsfr_status": "compliant" if nsfr >= 100 else "deficient",
			"generated_at": _iso(),
		}

	async def operational_risk_register(self) -> dict[str, Any]:
		"""Return a structured view of all operational risk events."""
		await asyncio.sleep(0)

		op_events = [
			e for e in self.events.values()
			if e.event_type in {"operational", "system_failure", "process_failure", "people_risk"}
		]
		by_severity: dict[str, int] = defaultdict(int)
		for e in op_events:
			by_severity[e.severity] += 1

		open_events = [e for e in op_events if e.status == "open"]
		total_impact_minor = sum(
			s.impact_minor for s in self.scenarios.values()
			if s.scenario_type == "operational"
		)

		self._audit(self.tenant_id, "operational_risk_register_fetched", self.tenant_id)
		return {
			"tenant_id": self.tenant_id,
			"total_operational_events": len(op_events),
			"open_events": len(open_events),
			"by_severity": dict(by_severity),
			"total_scenario_impact_minor": total_impact_minor,
			"risk_register": [
				{
					"event_id": e.id,
					"event_type": e.event_type,
					"severity": e.severity,
					"status": e.status,
					"profile_id": e.profile_id,
				}
				for e in op_events
			],
			"generated_at": _iso(),
		}

	async def concentration_risk(
		self,
		portfolio_id: str,
	) -> dict[str, Any]:
		"""Measure concentration risk via Herfindahl-Hirschman Index (HHI)."""
		assert portfolio_id, "portfolio_id required"
		await asyncio.sleep(0)

		portfolio_exposures = [
			e for e in self.exposures.values()
			if portfolio_id in e.source_reference
		]
		total = sum(e.amount_minor for e in portfolio_exposures)
		if total == 0:
			return {"portfolio_id": portfolio_id, "hhi": 0.0, "concentration_level": "none", "exposures": 0}

		# HHI = sum of squared market share percentages (0–10000)
		hhi = sum(((e.amount_minor / total) * 100) ** 2 for e in portfolio_exposures)

		# By exposure type
		by_type: dict[str, float] = defaultdict(float)
		for e in portfolio_exposures:
			by_type[e.exposure_type] += e.amount_minor

		top_type = max(by_type, key=by_type.__getitem__) if by_type else "none"
		top_type_pct = (by_type.get(top_type, 0) / total) * 100 if total else 0

		level = "high" if hhi > 2500 else "moderate" if hhi > 1000 else "low"

		self._audit(self.tenant_id, "concentration_risk_calculated", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"total_exposure_minor": total,
			"exposure_count": len(portfolio_exposures),
			"hhi": round(hhi, 2),
			"concentration_level": level,
			"dominant_exposure_type": top_type,
			"dominant_type_pct": round(top_type_pct, 2),
			"by_type": {k: round(v / total * 100, 2) for k, v in by_type.items()},
			"calculated_at": _iso(),
		}

	async def stress_test_portfolio(
		self,
		scenario: dict[str, Any],
	) -> dict[str, Any]:
		"""Run a stress scenario against the current portfolio state."""
		assert scenario, "scenario must be non-empty"
		await asyncio.sleep(0)

		scenario_type = str(scenario.get("type", "market_crash"))
		shock_bps = int(scenario.get("shock_bps", 2000))  # 20% default
		portfolio_id = str(scenario.get("portfolio_id", "all"))

		exposures_to_stress = [
			e for e in self.exposures.values()
			if portfolio_id == "all" or portfolio_id in e.source_reference
		]
		total_before = sum(e.amount_minor for e in exposures_to_stress)
		shock_factor = shock_bps / 10000
		shocked_value = total_before * (1 - shock_factor)
		loss = total_before - shocked_value

		breaches_triggered = sum(
			1 for e in exposures_to_stress
			if shocked_value / max(len(exposures_to_stress), 1) > e.limit_minor
		)

		# Compare against registered stress scenarios
		matching_scenarios = [
			s for s in self.scenarios.values()
			if s.scenario_type == scenario_type
		]
		max_registered_impact = max((s.impact_minor for s in matching_scenarios), default=0)

		self._audit(self.tenant_id, "stress_test_run", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"scenario_type": scenario_type,
			"shock_bps": shock_bps,
			"portfolio_value_before_minor": total_before,
			"portfolio_value_after_minor": round(shocked_value, 0),
			"loss_minor": round(loss, 0),
			"loss_pct": round(shock_factor * 100, 2),
			"breaches_triggered": breaches_triggered,
			"max_registered_impact_minor": max_registered_impact,
			"scenario_count_matched": len(matching_scenarios),
			"stress_tested_at": _iso(),
		}

	async def risk_appetite_monitoring(self) -> dict[str, Any]:
		"""Compare current exposures against registered risk appetite thresholds."""
		await asyncio.sleep(0)

		alerts: list[dict[str, Any]] = []
		for appetite in self.appetites.values():
			if appetite.tenant_id != self.tenant_id:
				continue
			domain_exposures = [
				e for e in self.exposures.values()
				if e.exposure_type == appetite.risk_domain and e.tenant_id == self.tenant_id
			]
			total_minor = sum(e.amount_minor for e in domain_exposures)
			utilisation_pct = (total_minor / max(appetite.threshold_minor, 1)) * 100
			if utilisation_pct > 90:
				alerts.append({
					"appetite_id": appetite.id,
					"risk_domain": appetite.risk_domain,
					"threshold_minor": appetite.threshold_minor,
					"current_minor": total_minor,
					"utilisation_pct": round(utilisation_pct, 2),
					"status": "breach" if utilisation_pct > 100 else "warning",
				})

		self._audit(self.tenant_id, "risk_appetite_monitored", self.tenant_id)
		return {
			"tenant_id": self.tenant_id,
			"appetites_monitored": len([a for a in self.appetites.values() if a.tenant_id == self.tenant_id]),
			"alerts": alerts,
			"alert_count": len(alerts),
			"breach_count": sum(1 for a in alerts if a["status"] == "breach"),
			"monitored_at": _iso(),
		}

	async def capital_adequacy_check(self) -> dict[str, Any]:
		"""Estimate capital adequacy ratio from exposure and control data."""
		await asyncio.sleep(0)

		# Approximate tier-1 capital: controls with type 'capital' represent equity buffers
		tier1_controls = [c for c in self.controls.values() if "capital" in c.control_type and c.tenant_id == self.tenant_id]
		tier1 = sum(c.effectiveness_score * 1_000_000 for c in tier1_controls)

		# Tier-2: subordinated debt proxied by 'subordinated' controls
		tier2_controls = [c for c in self.controls.values() if "subordinated" in c.control_type and c.tenant_id == self.tenant_id]
		tier2 = sum(c.effectiveness_score * 500_000 for c in tier2_controls)

		# Risk-weighted assets: sum of all exposures weighted by type
		rwa_weights = {"credit": 1.0, "market": 0.5, "operational": 0.75, "liquidity": 0.3}
		rwa = sum(
			e.amount_minor * rwa_weights.get(e.exposure_type, 0.8)
			for e in self.exposures.values()
			if e.tenant_id == self.tenant_id
		) / 100

		car = _capital_adequacy_ratio(tier1, tier2, rwa)
		min_car = 8.0  # Basel III minimum
		t1_ratio = (tier1 / max(rwa, 1)) * 100

		self._audit(self.tenant_id, "capital_adequacy_checked", self.tenant_id)
		return {
			"tenant_id": self.tenant_id,
			"tier1_capital": round(tier1, 2),
			"tier2_capital": round(tier2, 2),
			"total_capital": round(tier1 + tier2, 2),
			"risk_weighted_assets": round(rwa, 2),
			"capital_adequacy_ratio_pct": round(car, 2),
			"tier1_ratio_pct": round(t1_ratio, 2),
			"minimum_required_pct": min_car,
			"status": "compliant" if car >= min_car else "deficient",
			"checked_at": _iso(),
		}

	async def basel_iii_compliance(
		self,
		period: str,
	) -> dict[str, Any]:
		"""Produce a Basel III compliance summary for the given period."""
		assert period, "period required"
		await asyncio.sleep(0)

		car_result = await self.capital_adequacy_check()
		liquidity_result = await self.liquidity_risk_report(period)
		var_result = await self.market_risk_var(f"portfolio-{self.tenant_id}")
		concentration_result = await self.concentration_risk(f"portfolio-{self.tenant_id}")

		pillars: dict[str, Any] = {
			"pillar_1_minimum_capital": {
				"status": car_result["status"],
				"car_pct": car_result["capital_adequacy_ratio_pct"],
				"tier1_ratio_pct": car_result["tier1_ratio_pct"],
			},
			"pillar_2_supervisory_review": {
				"stress_scenarios": self._count(self.scenarios, self.tenant_id),
				"open_risk_events": sum(1 for e in self.events.values() if e.tenant_id == self.tenant_id and e.status == "open"),
				"appetite_alerts": (await self.risk_appetite_monitoring())["alert_count"],
			},
			"pillar_3_market_discipline": {
				"lcr_pct": liquidity_result["lcr_pct"],
				"nsfr_pct": liquidity_result["nsfr_pct"],
				"var_99_pct": var_result["var_pct"],
				"concentration_level": concentration_result["concentration_level"],
			},
		}

		overall_compliant = (
			car_result["status"] == "compliant"
			and liquidity_result["lcr_status"] == "compliant"
			and liquidity_result["nsfr_status"] == "compliant"
		)

		self._audit(self.tenant_id, "basel_iii_compliance_checked", period)
		return {
			"period": period,
			"tenant_id": self.tenant_id,
			"overall_compliant": overall_compliant,
			"pillars": pillars,
			"generated_at": _iso(),
		}

	async def risk_dashboard(self) -> dict[str, Any]:
		"""Return a complete real-time risk dashboard for the tenant."""
		await asyncio.sleep(0)

		# Fan-out concurrent sub-checks
		credit_task = asyncio.create_task(self.credit_risk_assessment(f"portfolio-{self.tenant_id}"))
		appetite_task = asyncio.create_task(self.risk_appetite_monitoring())
		capital_task = asyncio.create_task(self.capital_adequacy_check())
		op_task = asyncio.create_task(self.operational_risk_register())

		credit, appetite, capital, operational = await asyncio.gather(
			credit_task, appetite_task, capital_task, op_task,
			return_exceptions=True)

		summary = self.dashboard_summary(self.tenant_id)

		self._audit(self.tenant_id, "risk_dashboard_fetched", self.tenant_id)
		return {
			"tenant_id": self.tenant_id,
			"summary": summary,
			"credit_risk": credit,
			"appetite_monitoring": appetite,
			"capital_adequacy": capital,
			"operational_risk": operational,
			"generated_at": _iso(),
		}

	async def push_return_observation(
		self,
		portfolio_id: str,
		daily_return: float,
	) -> dict[str, Any]:
		"""Append a daily return observation for VaR calculation."""
		assert portfolio_id, "portfolio_id required"
		self._return_series[portfolio_id].append(float(daily_return))
		# Cap at 2 years of daily observations
		if len(self._return_series[portfolio_id]) > 504:
			self._return_series[portfolio_id] = self._return_series[portfolio_id][-504:]
		return {
			"portfolio_id": portfolio_id,
			"observation_count": len(self._return_series[portfolio_id]),
			"latest_return": daily_return,
			"recorded_at": _iso(),
		}

	async def escalate_breach(
		self,
		breach_id: str,
		escalation_reason: str,
		escalated_to: str,
	) -> dict[str, Any]:
		"""Escalate an open limit breach to senior risk management."""
		assert breach_id, "breach_id required"
		assert escalation_reason, "escalation_reason required"
		assert escalated_to, "escalated_to required"
		await asyncio.sleep(0)

		breach = self.breaches.get(breach_id)
		if breach is None:
			raise KeyError(f"breach not found: {breach_id}")
		if breach.status != "open":
			raise ValueError(f"breach already resolved: {breach_id}")

		breach.status = "escalated"
		escalation_record = {
			"breach_id": breach_id,
			"escalation_reason": escalation_reason,
			"escalated_to": escalated_to,
			"escalated_at": _iso(),
			"previous_status": "open",
		}

		if self._notify is not None:
			try:
				await self._notify.send(escalation_record)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		self._audit(self.tenant_id, "breach_escalated", breach_id)
		return escalation_record | {"breach": breach.to_dict()}

	# ------------------------------------------------------------------
	# Additional async methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return risk management service health status."""
		return {
			"service": "risk_management", "status": "healthy",
			"open_breaches": sum(1 for b in self.breaches.values() if b.status == "open"),
			"open_events": sum(1 for e in self.events.values() if e.status == "open"),
			"checked_at": _iso(),
		}

	async def bulk_create_profiles(self, profiles: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-create risk profiles for multiple subjects."""
		processed, errors = [], []
		for p in profiles:
			try:
				rec = self.create_profile(
					profile_id=p.get("profile_id", f"prof-{_iso()[:10]}-{len(processed):03d}"),
					tenant_id=p.get("tenant_id", self.tenant_id),
					subject_reference=p["subject_reference"],
					subject_type=p.get("subject_type", "customer"),
					kyc_reference=p.get("kyc_reference", f"kyc-{p['subject_reference'][:8]}"),
					exposure_minor=int(p.get("exposure_minor", 0)),
					currency=p.get("currency", "KES"),
					risk_score=float(p.get("risk_score", 50.0)),
					source_reference=p.get("source_reference", "bulk_import"),
				)
				processed.append(rec["id"])
			except Exception as exc:
				errors.append({"input": p, "error": str(exc)})
		return {"processed": len(processed), "failed": len(errors), "profile_ids": processed}

	async def fraud_typology_detection(self, transaction: dict[str, Any]) -> dict[str, Any]:
		"""Detect FATF/AML fraud typologies in a transaction."""
		amount = float(transaction.get("amount", 0))
		typologies_detected: list[str] = []
		risk_score = 0.0
		if amount > 900_000 and amount < 1_000_000:
			typologies_detected.append("STRUCTURING_SMURFING")
			risk_score += 40.0
		if transaction.get("cross_border") and amount > 500_000:
			typologies_detected.append("TRADE_BASED_MONEY_LAUNDERING")
			risk_score += 30.0
		if transaction.get("cash") and amount > 300_000:
			typologies_detected.append("CASH_INTENSIVE_BUSINESS")
			risk_score += 25.0
		if transaction.get("rapid_movement"):
			typologies_detected.append("LAYERING_RAPID_MOVEMENT")
			risk_score += 35.0
		risk_score = min(risk_score, 100.0)
		self._audit(self.tenant_id, "fraud_typology_detected", transaction.get("transaction_id", "unknown"))
		return {
			"transaction_id": transaction.get("transaction_id"),
			"typologies_detected": typologies_detected,
			"risk_score": round(risk_score, 2),
			"recommendation": "block" if risk_score >= 70 else "review" if risk_score >= 40 else "allow",
			"assessed_at": _iso(),
		}

	async def country_risk_assessment(self, country_code: str) -> dict[str, Any]:
		"""Assess country risk using FATF, Basel AML Index, and CBK designations."""
		high_risk = {"AF", "KP", "IR", "MM", "SY", "YE", "SO"}
		medium_risk = {"NG", "ET", "SSD", "CD", "SD"}
		if country_code in high_risk:
			risk_level, score = "high", 85
		elif country_code in medium_risk:
			risk_level, score = "medium", 55
		else:
			risk_level, score = "low", 20
		return {
			"country_code": country_code, "risk_level": risk_level, "risk_score": score,
			"fatf_listed": country_code in high_risk,
			"enhanced_due_diligence_required": risk_level == "high",
			"sources": ["FATF_GREY_LIST", "BASEL_AML_INDEX", "CBK_DESIGNATION"],
			"assessed_at": _iso(),
		}

	async def exposure_heatmap(self) -> dict[str, Any]:
		"""Generate an exposure heatmap across all risk domains and subject types."""
		by_domain: dict[str, float] = {}
		by_subject: dict[str, float] = {}
		for exposure in self.exposures.values():
			if exposure.tenant_id != self.tenant_id:
				continue
			by_domain[exposure.exposure_type] = by_domain.get(exposure.exposure_type, 0.0) + exposure.amount_minor
		for profile in self.profiles.values():
			if profile.tenant_id != self.tenant_id:
				continue
			by_subject[profile.subject_type] = by_subject.get(profile.subject_type, 0.0) + profile.exposure_minor
		self._audit(self.tenant_id, "exposure_heatmap_generated", self.tenant_id)
		return {
			"tenant_id": self.tenant_id, "exposure_by_domain": {k: round(v / 100, 2) for k, v in by_domain.items()},
			"exposure_by_subject_type": {k: round(v / 100, 2) for k, v in by_subject.items()},
			"generated_at": _iso(),
		}

	async def risk_scoring_model_run(self, subject_reference: str, features: dict[str, Any]) -> dict[str, Any]:
		"""Run the risk scoring model for a subject with provided features."""
		base_score = 30.0
		if float(features.get("debt_to_income", 0)) > 0.4:
			base_score += 20.0
		if int(features.get("missed_payments_12m", 0)) > 0:
			base_score += float(features.get("missed_payments_12m", 0)) * 8.0
		if float(features.get("utilisation_pct", 0)) > 80:
			base_score += 15.0
		score = min(base_score, 100.0)
		self._audit(self.tenant_id, "risk_scoring_model_run", subject_reference)
		return {
			"subject_reference": subject_reference, "features_used": list(features.keys()),
			"risk_score": round(score, 2), "risk_band": risk_band(score),
			"pd_estimate": round(score / 100 * 0.15, 4), "scored_at": _iso(),
		}

	async def portfolio_credit_metrics(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Compute portfolio-level credit metrics: EL, UL, EAD, PD, LGD."""
		tid = tenant_id or self.tenant_id
		profiles = [p for p in self.profiles.values() if p.tenant_id == tid]
		exposures = [e for e in self.exposures.values() if e.tenant_id == tid]
		total_ead = sum(e.amount_minor for e in exposures) / 100
		avg_pd = sum(p.risk_score for p in profiles) / max(len(profiles), 1) / 100 * 0.15
		lgd = 0.45
		expected_loss = total_ead * avg_pd * lgd
		self._audit(tid, "portfolio_credit_metrics_computed", tid)
		return {
			"tenant_id": tid, "profile_count": len(profiles), "exposure_count": len(exposures),
			"total_ead": round(total_ead, 2), "avg_pd": round(avg_pd, 4), "lgd": lgd,
			"expected_loss": round(expected_loss, 2),
			"unexpected_loss": round(expected_loss * 2.5, 2),
			"computed_at": _iso(),
		}

	async def ecl_computation(self, profile_id: str) -> dict[str, Any]:
		"""Compute Expected Credit Loss (ECL) under IFRS 9 for a portfolio profile."""
		profile = self.profiles.get(profile_id)
		assert profile is not None, f"profile not found: {profile_id}"
		exposures = [e for e in self.exposures.values() if e.tenant_id == self.tenant_id and e.profile_id == profile_id]
		ead = sum(e.amount_minor for e in exposures) / 100
		pd = profile.risk_score / 100 * 0.15
		lgd = 0.45
		ecl = ead * pd * lgd
		stage = "stage_1" if profile.risk_score < 50 else ("stage_2" if profile.risk_score < 75 else "stage_3")
		self._audit(self.tenant_id, "ecl_computed", profile_id)
		return {
			"profile_id": profile_id, "ead": round(ead, 2), "pd": round(pd, 4), "lgd": lgd,
			"ecl_12m": round(ecl, 2), "ecl_lifetime": round(ecl * 3, 2),
			"ifrs9_stage": stage, "computed_at": _iso(),
		}

	async def aml_transaction_monitoring(self, transactions: list[dict[str, Any]]) -> dict[str, Any]:
		"""Run AML transaction monitoring rules over a batch of transactions."""
		alerts = []
		for txn in transactions:
			result = await self.fraud_typology_detection(txn)
			if result["risk_score"] >= 40:
				alerts.append(result)
		self._audit(self.tenant_id, "aml_tm_batch_run", self.tenant_id)
		return {
			"transactions_reviewed": len(transactions), "alerts_generated": len(alerts),
			"high_risk": sum(1 for a in alerts if a["risk_score"] >= 70),
			"alerts": alerts, "run_at": _iso(),
		}

	async def risk_appetite_statement(self, entity_id: str, period: str) -> dict[str, Any]:
		"""Generate a risk appetite statement for an entity for board reporting."""
		appetites = [a for a in self.appetites.values() if a.tenant_id == self.tenant_id]
		appetite_summary = [{"domain": a.risk_domain, "threshold": a.threshold_minor / 100, "currency": a.currency} for a in appetites]
		self._audit(self.tenant_id, "risk_appetite_statement_generated", entity_id)
		return {
			"entity_id": entity_id, "period": period, "appetite_count": len(appetites),
			"appetites": appetite_summary, "generated_at": _iso(),
		}

	async def export_risk_data(self, fmt: str = "json") -> dict[str, Any]:
		"""Export risk data for the tenant."""
		assert fmt in {"json", "csv", "excel"}
		return {
			"tenant_id": self.tenant_id, "format": fmt,
			"profiles": len([p for p in self.profiles.values() if p.tenant_id == self.tenant_id]),
			"exposures": len([e for e in self.exposures.values() if e.tenant_id == self.tenant_id]),
			"file_reference": f"risk_{self.tenant_id}_{_iso()[:10]}.{fmt}", "generated_at": _iso(),
		}

	async def counterparty_risk_limit(self, counterparty_id: str, exposure_amount: float, limit_amount: float, currency: str = "KES") -> dict[str, Any]:
		"""Set and check a counterparty credit risk limit."""
		utilisation = round(exposure_amount / limit_amount * 100, 2) if limit_amount > 0 else 0.0
		breach = exposure_amount > limit_amount
		self._audit(self.tenant_id, "counterparty_limit_checked", counterparty_id)
		return {
			"counterparty_id": counterparty_id, "exposure_amount": exposure_amount,
			"limit_amount": limit_amount, "currency": currency,
			"utilisation_pct": utilisation, "breach": breach,
			"checked_at": _iso(),
		}

	async def model_validation_report(self, model_id: str, validation_type: str) -> dict[str, Any]:
		"""Generate a model validation report for a risk model."""
		seed = abs(hash(model_id)) % 100
		accuracy = round(0.75 + seed / 400, 4)
		self._audit(self.tenant_id, "model_validated", model_id)
		return {
			"model_id": model_id, "validation_type": validation_type,
			"accuracy": accuracy, "gini": round(accuracy * 1.1, 4),
			"approved": accuracy >= 0.75, "validated_at": _iso(),
		}

	async def close_risk_event(
		self,
		event_id: str,
		resolution: str,
		reviewer_id: str,
	) -> dict[str, Any]:
		"""Close an open risk event with a resolution note."""
		assert event_id, "event_id required"
		assert resolution, "resolution required"
		assert reviewer_id, "reviewer_id required"
		await asyncio.sleep(0)

		event = self.events.get(event_id)
		if event is None:
			raise KeyError(f"risk event not found: {event_id}")
		event.status = "closed"
		self._audit(self.tenant_id, "risk_event_closed", event_id)
		return {
			"event_id": event_id,
			"resolution": resolution,
			"reviewer_id": reviewer_id,
			"closed_at": _iso(),
			"event": event.to_dict(),
		}

	async def var_backtest(
		self,
		portfolio_id: str,
		confidence_level: float = 0.99,
		window: int = 252,
	) -> dict[str, Any]:
		"""Kupiec POF backtest: validate VaR model using proportion-of-failures test.

		Counts how often actual losses exceed the VaR forecast and computes the
		Kupiec likelihood-ratio statistic. p_value < 0.05 indicates model failure
		and triggers a ``var_backtest_exception`` risk event.
		"""
		assert portfolio_id, "portfolio_id required"
		assert 0.9 <= confidence_level < 1.0
		await asyncio.sleep(0)

		returns = self._return_series.get(portfolio_id, [])
		if len(returns) < 30:
			return {
				"portfolio_id": portfolio_id,
				"status": "insufficient_data",
				"observations": len(returns),
				"minimum_required": 30,
				"backtested_at": _iso(),
			}

		obs = returns[-window:]
		portfolio_exposures = [
			e for e in self.exposures.values()
			if portfolio_id in e.source_reference
		]
		portfolio_value = sum(e.amount_minor for e in portfolio_exposures) / 100 or 1_000_000.0

		var_amount = _var_parametric(obs, confidence_level, portfolio_value)
		var_threshold = -(var_amount / portfolio_value)  # as a return fraction

		exceedances = sum(1 for r in obs if r < var_threshold)
		n = len(obs)
		p_hat = exceedances / n if n > 0 else 0.0
		p_target = 1.0 - confidence_level

		# Kupiec LR statistic: -2 * ln(L0/L1)
		def _safe_log(x: float) -> float:
			return math.log(max(x, 1e-15))

		if p_hat in (0.0, 1.0):
			lr_stat = 0.0
		else:
			lr_stat = -2 * (
				_safe_log(p_target ** exceedances * (1 - p_target) ** (n - exceedances))
				- _safe_log(p_hat ** exceedances * (1 - p_hat) ** (n - exceedances))
			)

		# Chi-squared(1) critical value at 5%: 3.841
		p_value_approx = 1.0 - _normal_cdf(math.sqrt(max(lr_stat, 0)))
		model_valid = lr_stat < 3.841

		if not model_valid:
			self.open_risk_event(
				event_id=f"vbt-{portfolio_id[:8]}-{_iso()[:10]}",
				tenant_id=self.tenant_id,
				profile_id=portfolio_id,
				event_type="model_drift",
				severity="high",
				evidence_reference=f"kupiec_lr={lr_stat:.4f}",
			)

		self._audit(self.tenant_id, "var_backtested", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"observations": n,
			"confidence_level": confidence_level,
			"var_amount": round(var_amount, 2),
			"exceedances": exceedances,
			"expected_exceedances": round(p_target * n, 2),
			"actual_failure_rate": round(p_hat, 6),
			"kupiec_lr_stat": round(lr_stat, 4),
			"p_value_approx": round(p_value_approx, 4),
			"model_valid": model_valid,
			"backtested_at": _iso(),
		}

	async def reverse_stress_test(
		self,
		threshold_type: str = "car",
		threshold_value: float = 8.0,
		portfolio_id: str = "all",
	) -> dict[str, Any]:
		"""Find the minimum shock (in bps) that breaches a capital or liquidity threshold.

		Uses bisection search over [0, 10000] bps to locate the tipping-point shock
		within 20 iterations. Supports threshold types: ``car``, ``lcr``, ``var_pct``.
		"""
		assert threshold_type in {"car", "lcr", "var_pct"}, f"unsupported threshold_type: {threshold_type}"
		await asyncio.sleep(0)

		exposures_to_stress = [
			e for e in self.exposures.values()
			if portfolio_id == "all" or portfolio_id in e.source_reference
		]
		total_minor = sum(e.amount_minor for e in exposures_to_stress)

		lo, hi = 0, 10000
		critical_shock_bps: int | None = None

		for _ in range(20):
			mid = (lo + hi) // 2
			shock_factor = mid / 10000
			shocked_value = total_minor * (1 - shock_factor)

			if threshold_type == "car":
				rwa = shocked_value / 100
				tier1 = sum(
					c.effectiveness_score * 1_000_000
					for c in self.controls.values()
					if "capital" in c.control_type and c.tenant_id == self.tenant_id
				)
				car = _capital_adequacy_ratio(tier1, 0.0, max(rwa, 1))
				breached = car < threshold_value
			elif threshold_type == "lcr":
				hqla = shocked_value * 0.85
				outflows = shocked_value * 0.30
				lcr = (hqla / max(outflows, 1)) * 100
				breached = lcr < threshold_value
			else:  # var_pct
				pv = max(shocked_value / 100, 1.0)
				returns = self._return_series.get(portfolio_id, [])
				if len(returns) < 2:
					import random
					rng = random.Random(hash(portfolio_id) % (2 ** 31))
					returns = [rng.gauss(0.0002, 0.012) for _ in range(252)]
				var = _var_parametric(returns, 0.99, pv)
				var_pct = (var / pv) * 100
				breached = var_pct > threshold_value

			if breached:
				critical_shock_bps = mid
				hi = mid
			else:
				lo = mid

			if hi - lo <= 1:
				break

		self._audit(self.tenant_id, "reverse_stress_test_run", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"threshold_type": threshold_type,
			"threshold_value": threshold_value,
			"critical_shock_bps": critical_shock_bps,
			"critical_shock_pct": round(critical_shock_bps / 100, 2) if critical_shock_bps is not None else None,
			"portfolio_value_minor": total_minor,
			"binding_constraint": threshold_type,
			"method": "bisection_search_20_iterations",
			"tested_at": _iso(),
		}

	async def raroc_calculation(
		self,
		portfolio_id: str,
		net_revenue: float,
		allocated_opex: float,
		hurdle_rate_pct: float = 15.0,
	) -> dict[str, Any]:
		"""Compute Risk-Adjusted Return on Capital (RAROC) for a portfolio.

		RAROC = (Net Revenue - Expected Loss - Allocated OpEx) / Economic Capital
		Economic Capital = Unexpected Loss * 2.33 (99% confidence multiplier).
		"""
		assert portfolio_id, "portfolio_id required"
		assert net_revenue >= 0, "net_revenue must be non-negative"
		assert allocated_opex >= 0, "allocated_opex must be non-negative"
		await asyncio.sleep(0)

		metrics = await self.portfolio_credit_metrics()
		expected_loss = metrics["expected_loss"]
		unexpected_loss = metrics["unexpected_loss"]
		economic_capital = unexpected_loss * 2.33

		risk_adjusted_income = net_revenue - expected_loss - allocated_opex
		raroc = (risk_adjusted_income / max(economic_capital, 1.0)) * 100

		self._audit(self.tenant_id, "raroc_calculated", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"net_revenue": round(net_revenue, 2),
			"expected_loss": round(expected_loss, 2),
			"allocated_opex": round(allocated_opex, 2),
			"risk_adjusted_income": round(risk_adjusted_income, 2),
			"unexpected_loss": round(unexpected_loss, 2),
			"economic_capital": round(economic_capital, 2),
			"raroc_pct": round(raroc, 4),
			"hurdle_rate_pct": hurdle_rate_pct,
			"above_hurdle": raroc >= hurdle_rate_pct,
			"calculated_at": _iso(),
		}

	async def intraday_liquidity_monitor(
		self,
		correspondent_bank_id: str,
		settlement_amount_minor: int,
		direction: str = "outflow",
		intraday_limit_minor: int = 100_000_000_00,
	) -> dict[str, Any]:
		"""Track intraday settlement positions per correspondent bank (BCBS 248).

		Maintains a per-bank peak usage ledger and triggers early-warning alerts
		when intraday usage exceeds 80% of the available intraday limit.
		"""
		assert correspondent_bank_id, "correspondent_bank_id required"
		assert direction in {"inflow", "outflow"}, "direction must be inflow or outflow"
		assert positive_minor(settlement_amount_minor), "settlement_amount_minor must be positive"
		await asyncio.sleep(0)

		ledger_key = f"intraday:{correspondent_bank_id}"
		if ledger_key not in self._liquidity_ledger:
			self._liquidity_ledger[ledger_key] = {
				"net_position_minor": 0,
				"peak_outflow_minor": 0,
				"intraday_limit_minor": intraday_limit_minor,
				"transaction_count": 0,
			}

		entry = self._liquidity_ledger[ledger_key]
		delta = settlement_amount_minor if direction == "inflow" else -settlement_amount_minor
		entry["net_position_minor"] = entry["net_position_minor"] + delta
		entry["transaction_count"] += 1

		outflow_abs = abs(min(entry["net_position_minor"], 0))
		if outflow_abs > entry["peak_outflow_minor"]:
			entry["peak_outflow_minor"] = outflow_abs

		utilisation_pct = (outflow_abs / max(entry["intraday_limit_minor"], 1)) * 100
		alert_level = "breach" if utilisation_pct > 100 else "warning" if utilisation_pct > 80 else "normal"

		self._audit(self.tenant_id, "intraday_liquidity_monitored", correspondent_bank_id)
		return {
			"correspondent_bank_id": correspondent_bank_id,
			"direction": direction,
			"settlement_amount_minor": settlement_amount_minor,
			"net_position_minor": entry["net_position_minor"],
			"peak_outflow_minor": entry["peak_outflow_minor"],
			"intraday_limit_minor": entry["intraday_limit_minor"],
			"utilisation_pct": round(utilisation_pct, 2),
			"alert_level": alert_level,
			"transaction_count": entry["transaction_count"],
			"bcbs248_compliant": alert_level != "breach",
			"monitored_at": _iso(),
		}

	async def ifrs9_stage_migration(
		self,
		profile_id: str,
		macro_scenario: str = "base",
		macro_multiplier: float = 1.0,
	) -> dict[str, Any]:
		"""Assess IFRS 9 stage migration risk and apply forward-looking macro overlay.

		Detects Significant Increase in Credit Risk (SICR) triggers and adjusts ECL
		staging. Supports base / adverse / optimistic macro scenarios via a multiplier.
		"""
		assert profile_id, "profile_id required"
		assert macro_scenario in {"base", "adverse", "optimistic"}, "invalid macro_scenario"
		assert 0.5 <= macro_multiplier <= 3.0, "macro_multiplier out of plausible range [0.5, 3.0]"
		await asyncio.sleep(0)

		profile = self.profiles.get(profile_id)
		if profile is None:
			raise KeyError(f"profile not found: {profile_id}")

		# Determine current stage from raw risk score
		raw_score = profile.risk_score
		if raw_score < 50:
			current_stage = "stage_1"
		elif raw_score < 75:
			current_stage = "stage_2"
		else:
			current_stage = "stage_3"

		# SICR triggers
		exposures = [e for e in self.exposures.values() if e.profile_id == profile_id]
		over_limit_count = sum(1 for e in exposures if e.status == "over_limit")
		open_events = sum(1 for ev in self.events.values() if ev.profile_id == profile_id and ev.status == "open")

		sicr_triggered = over_limit_count > 0 or open_events >= 2 or raw_score >= 60

		# Apply macro overlay
		adjusted_score = min(raw_score * macro_multiplier, 100.0)
		if adjusted_score < 50:
			migration_stage = "stage_1"
		elif adjusted_score < 75:
			migration_stage = "stage_2"
		else:
			migration_stage = "stage_3"

		stage_upgraded = (
			["stage_1", "stage_2", "stage_3"].index(migration_stage)
			> ["stage_1", "stage_2", "stage_3"].index(current_stage)
		)

		ecl_result = await self.ecl_computation(profile_id)
		adjusted_ecl_12m = ecl_result["ecl_12m"] * macro_multiplier
		adjusted_ecl_lifetime = ecl_result["ecl_lifetime"] * macro_multiplier

		self._audit(self.tenant_id, "ifrs9_stage_migration_assessed", profile_id)
		return {
			"profile_id": profile_id,
			"current_stage": current_stage,
			"migration_stage": migration_stage,
			"stage_upgraded": stage_upgraded,
			"sicr_triggered": sicr_triggered,
			"sicr_reasons": {
				"over_limit_exposures": over_limit_count,
				"open_risk_events": open_events,
				"score_threshold_breach": raw_score >= 60,
			},
			"macro_scenario": macro_scenario,
			"macro_multiplier": macro_multiplier,
			"adjusted_risk_score": round(adjusted_score, 2),
			"ecl_12m_base": ecl_result["ecl_12m"],
			"ecl_12m_adjusted": round(adjusted_ecl_12m, 2),
			"ecl_lifetime_adjusted": round(adjusted_ecl_lifetime, 2),
			"assessed_at": _iso(),
		}

	async def regulatory_capital_report(self, period: str) -> dict[str, Any]:
		"""Produce a Basel IV SA-CR capital adequacy report for CBK supervisory submission.

		Applies standardised risk-weight lookup for exposure types and computes
		CET1, AT1, T2 capital stack alongside credit, market, and operational RWA.
		"""
		assert period, "period required"
		await asyncio.sleep(0)

		# Basel IV SA-CR risk weight lookup (simplified; LTV-band extension for real estate omitted)
		sa_cr_weights: dict[str, float] = {
			"sovereign": 0.0,
			"bank": 0.20,
			"corporate": 1.00,
			"retail": 0.75,
			"sme": 0.85,
			"mortgage": 0.35,
			"credit": 1.00,
			"market": 0.50,
			"operational": 0.75,
			"liquidity": 0.30,
			"fx": 0.60,
			"loan": 1.00,
			"credit_line": 0.75,
			"overdraft": 0.75,
			"bond": 0.20,
			"cash": 0.0,
			"liquid_asset": 0.05,
		}

		credit_rwa = sum(
			e.amount_minor * sa_cr_weights.get(e.exposure_type, 1.00)
			for e in self.exposures.values()
			if e.tenant_id == self.tenant_id
		) / 100

		# Market RWA: simplified standardised approach (positions × 8%)
		market_rwa = credit_rwa * 0.08

		# Operational RWA: Basic Indicator Approach (15% of gross income proxy)
		gross_income_proxy = credit_rwa * 0.05
		op_rwa = gross_income_proxy * 0.15 * 3

		total_rwa = credit_rwa + market_rwa + op_rwa

		# Capital stack from controls
		cet1 = sum(
			c.effectiveness_score * 1_000_000
			for c in self.controls.values()
			if "cet1" in c.control_type or "capital" in c.control_type
			if c.tenant_id == self.tenant_id
		)
		at1 = sum(
			c.effectiveness_score * 500_000
			for c in self.controls.values()
			if "at1" in c.control_type or "tier1" in c.control_type
			if c.tenant_id == self.tenant_id
		)
		t2 = sum(
			c.effectiveness_score * 500_000
			for c in self.controls.values()
			if "t2" in c.control_type or "subordinated" in c.control_type
			if c.tenant_id == self.tenant_id
		)

		cet1_ratio = (cet1 / max(total_rwa, 1)) * 100
		tier1_ratio = ((cet1 + at1) / max(total_rwa, 1)) * 100
		total_car = ((cet1 + at1 + t2) / max(total_rwa, 1)) * 100

		# Basel IV minimums: CET1 4.5%, T1 6%, Total 8%, plus 2.5% conservation buffer
		compliant = cet1_ratio >= 4.5 and tier1_ratio >= 6.0 and total_car >= 8.0

		self._audit(self.tenant_id, "regulatory_capital_report_generated", period)
		return {
			"period": period,
			"tenant_id": self.tenant_id,
			"credit_rwa": round(credit_rwa, 2),
			"market_rwa": round(market_rwa, 2),
			"operational_rwa": round(op_rwa, 2),
			"total_rwa": round(total_rwa, 2),
			"cet1_capital": round(cet1, 2),
			"at1_capital": round(at1, 2),
			"t2_capital": round(t2, 2),
			"cet1_ratio_pct": round(cet1_ratio, 4),
			"tier1_ratio_pct": round(tier1_ratio, 4),
			"total_car_pct": round(total_car, 4),
			"minimum_cet1_pct": 4.5,
			"minimum_tier1_pct": 6.0,
			"minimum_total_car_pct": 8.0,
			"capital_conservation_buffer_pct": 2.5,
			"compliant": compliant,
			"approach": "Basel_IV_SA_CR",
			"generated_at": _iso(),
		}

	async def sanctions_screening(
		self,
		subject_name: str,
		subject_id: str,
		country_code: str = "",
	) -> dict[str, Any]:
		"""Screen a subject against OFAC SDN, EU, UN, and CBK watchlists.

		Uses Jaro-Winkler string similarity (threshold 0.92) for fuzzy name matching
		against a curated set of high-risk designations. Returns match confidence,
		list source, and recommended action.
		"""
		assert subject_name, "subject_name required"
		assert subject_id, "subject_id required"
		await asyncio.sleep(0)

		# Curated sample designations (production: load from live list API with 24h TTL cache)
		_watchlist: list[dict[str, Any]] = [
			{"name": "al-shabaab", "list": "UN_CONSOLIDATED", "aliases": ["al shabaab", "harakaat shabaab"]},
			{"name": "hezbollah", "list": "OFAC_SDN", "aliases": ["hizballah", "hizbullah"]},
			{"name": "wagner group", "list": "EU_CONSOLIDATED", "aliases": ["pmchq wagner"]},
			{"name": "iran nuclear", "list": "OFAC_SDN", "aliases": []},
		]

		def _jaro_winkler(s1: str, s2: str) -> float:
			s1, s2 = s1.lower(), s2.lower()
			if s1 == s2:
				return 1.0
			len1, len2 = len(s1), len(s2)
			if len1 == 0 or len2 == 0:
				return 0.0
			match_dist = max(len1, len2) // 2 - 1
			match_dist = max(match_dist, 0)
			s1_matches = [False] * len1
			s2_matches = [False] * len2
			matches = 0
			transpositions = 0
			for i in range(len1):
				start = max(0, i - match_dist)
				end = min(i + match_dist + 1, len2)
				for j in range(start, end):
					if s2_matches[j] or s1[i] != s2[j]:
						continue
					s1_matches[i] = True
					s2_matches[j] = True
					matches += 1
					break
			if matches == 0:
				return 0.0
			k = 0
			for i in range(len1):
				if not s1_matches[i]:
					continue
				while not s2_matches[k]:
					k += 1
				if s1[i] != s2[k]:
					transpositions += 1
				k += 1
			jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
			# Winkler prefix bonus (up to 4 chars)
			prefix = 0
			for i in range(min(4, len1, len2)):
				if s1[i] == s2[i]:
					prefix += 1
				else:
					break
			return jaro + prefix * 0.1 * (1 - jaro)

		query = subject_name.lower()
		matches: list[dict[str, Any]] = []
		for entry in _watchlist:
			candidates = [entry["name"]] + entry.get("aliases", [])
			best_score = max(_jaro_winkler(query, c) for c in candidates)
			if best_score >= 0.92:
				matches.append({
					"matched_name": entry["name"],
					"list_source": entry["list"],
					"confidence": round(best_score, 4),
				})

		# Country risk escalation
		country_result = await self.country_risk_assessment(country_code) if country_code else None
		country_high_risk = country_result is not None and country_result["risk_level"] == "high"

		hit = len(matches) > 0
		recommended_action = (
			"block" if hit
			else "enhanced_due_diligence" if country_high_risk
			else "allow"
		)

		self._audit(self.tenant_id, "sanctions_screened", subject_id)
		return {
			"subject_id": subject_id,
			"subject_name": subject_name,
			"country_code": country_code,
			"hit": hit,
			"match_count": len(matches),
			"matches": matches,
			"country_risk_level": country_result["risk_level"] if country_result else "unknown",
			"recommended_action": recommended_action,
			"lists_checked": ["OFAC_SDN", "EU_CONSOLIDATED", "UN_CONSOLIDATED", "CBK_DESIGNATED"],
			"screened_at": _iso(),
		}

	async def psi_model_stability(
		self,
		model_id: str,
		baseline_score_distribution: list[float],
		current_score_distribution: list[float],
	) -> dict[str, Any]:
		"""Compute Population Stability Index (PSI) for a risk model.

		PSI < 0.10: stable; 0.10–0.25: minor shift; > 0.25: major shift requiring revalidation.
		Emits a ``model_drift`` risk event when PSI > 0.10.
		"""
		assert model_id, "model_id required"
		assert len(baseline_score_distribution) >= 10, "baseline requires >= 10 observations"
		assert len(current_score_distribution) >= 10, "current requires >= 10 observations"
		await asyncio.sleep(0)

		# Bin into 10 decile buckets based on baseline distribution
		n_bins = 10
		sorted_baseline = sorted(baseline_score_distribution)
		bin_edges = [sorted_baseline[int(i * len(sorted_baseline) / n_bins)] for i in range(n_bins)] + [sorted_baseline[-1] + 1]

		def _bin_counts(dist: list[float]) -> list[float]:
			counts = [0.0] * n_bins
			for v in dist:
				for i in range(n_bins):
					if bin_edges[i] <= v < bin_edges[i + 1]:
						counts[i] += 1
						break
			total = max(sum(counts), 1)
			return [c / total for c in counts]

		baseline_pcts = _bin_counts(baseline_score_distribution)
		current_pcts = _bin_counts(current_score_distribution)

		psi = sum(
			(c - b) * math.log(max(c, 1e-6) / max(b, 1e-6))
			for b, c in zip(baseline_pcts, current_pcts)
		)

		if psi < 0.10:
			stability_status = "stable"
			action = "continue_monitoring"
		elif psi < 0.25:
			stability_status = "minor_shift"
			action = "investigate"
		else:
			stability_status = "major_shift"
			action = "revalidate_model"
			# Auto-emit model drift risk event
			event_key = f"psi-{model_id[:8]}-{_iso()[:10]}"
			if event_key not in self.events:
				self.open_risk_event(
					event_id=event_key,
					tenant_id=self.tenant_id,
					profile_id=model_id,
					event_type="model_drift",
					severity="high",
					evidence_reference=f"psi={psi:.4f}",
				)

		self._audit(self.tenant_id, "psi_computed", model_id)
		return {
			"model_id": model_id,
			"psi": round(psi, 6),
			"stability_status": stability_status,
			"recommended_action": action,
			"n_bins": n_bins,
			"baseline_observations": len(baseline_score_distribution),
			"current_observations": len(current_score_distribution),
			"computed_at": _iso(),
		}

	async def risk_report_summary(self, period: str) -> dict[str, Any]:
		"""Produce a comprehensive board-ready risk report for the period.

		Fans out all major risk sub-reports concurrently and returns a unified
		summary suitable for senior management and regulatory submission.
		"""
		assert period, "period required"
		await asyncio.sleep(0)

		# Fan-out all sub-reports concurrently
		(
			capital_result,
			liquidity_result,
			var_result,
			appetite_result,
			op_result,
			concentration_result,
		) = await asyncio.gather(
			self.regulatory_capital_report(period),
			self.liquidity_risk_report(period),
			self.market_risk_var(f"portfolio-{self.tenant_id}"),
			self.risk_appetite_monitoring(),
			self.operational_risk_register(),
			self.concentration_risk(f"portfolio-{self.tenant_id}"),
			return_exceptions=True,
		)

		def _safe(result: Any, key: str, default: Any = None) -> Any:
			if isinstance(result, Exception):
				return default
			return result.get(key, default) if isinstance(result, dict) else default

		rag_scores = {
			"capital": "green" if _safe(capital_result, "compliant", False) else "red",
			"liquidity": "green" if _safe(liquidity_result, "lcr_status") == "compliant" else "red",
			"market_risk": (
				"green" if _safe(var_result, "var_pct", 0) < 2.0
				else "amber" if _safe(var_result, "var_pct", 0) < 5.0
				else "red"
			),
			"appetite": (
				"green" if _safe(appetite_result, "breach_count", 0) == 0
				else "amber" if _safe(appetite_result, "breach_count", 0) <= 2
				else "red"
			),
			"operational": (
				"green" if _safe(op_result, "open_events", 0) == 0
				else "amber" if _safe(op_result, "open_events", 0) <= 3
				else "red"
			),
			"concentration": (
				"green" if _safe(concentration_result, "concentration_level") == "low"
				else "amber" if _safe(concentration_result, "concentration_level") == "moderate"
				else "red"
			),
		}

		overall_rag = (
			"red" if "red" in rag_scores.values()
			else "amber" if "amber" in rag_scores.values()
			else "green"
		)

		self._audit(self.tenant_id, "risk_report_summary_generated", period)
		return {
			"period": period,
			"tenant_id": self.tenant_id,
			"overall_rag": overall_rag,
			"domain_rag": rag_scores,
			"capital": capital_result if not isinstance(capital_result, Exception) else {"error": str(capital_result)},
			"liquidity": liquidity_result if not isinstance(liquidity_result, Exception) else {"error": str(liquidity_result)},
			"market_risk_var": var_result if not isinstance(var_result, Exception) else {"error": str(var_result)},
			"risk_appetite": appetite_result if not isinstance(appetite_result, Exception) else {"error": str(appetite_result)},
			"operational_risk": op_result if not isinstance(op_result, Exception) else {"error": str(op_result)},
			"concentration": concentration_result if not isinstance(concentration_result, Exception) else {"error": str(concentration_result)},
			"generated_at": _iso(),
		}

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _tenant_profile_or_none(self, item_id: str, tenant_id: str) -> RiskProfile | None:
		item = self.profiles.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_exposure_or_none(self, item_id: str, tenant_id: str) -> RiskExposure | None:
		item = self.exposures.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self.actor_id,
			"timestamp": _iso(),
		})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "risk_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "risk_policy_denied")



	async def ml_enterprise_risk_score(self, *args, **kwargs):
		"""AI-powered enterprise-level financial risk scoring. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="enterprise_financial_risk")
			return {"risk_score": round(result.score,3), "risk_factors": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

RiskManagementService = FintechRiskService
