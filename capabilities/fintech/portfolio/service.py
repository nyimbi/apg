"""Executable service layer for APG Portfolio Management."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_BREACH_SEVERITIES, SUPPORTED_CORPORATE_ACTIONS, SUPPORTED_CURRENCIES, SUPPORTED_PORTFOLIO_TYPES, SUPPORTED_REVIEW_STATUSES, evaluate_capability_rules, get_capability_contract
	from .models import AllocationPolicy, BenchmarkAssignment, CashMovement, ComplianceBreach, CorporateAction, HoldingRecord, PerformanceAttribution, PortfolioBook, PortfolioEvidence, PortfolioReview, PortfolioValuation, RiskExposure
	from .portfolio_runtime import allocation_totals_100, normalize_code, normalize_currency, positive_minor, positive_quantity
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_BREACH_SEVERITIES, SUPPORTED_CORPORATE_ACTIONS, SUPPORTED_CURRENCIES, SUPPORTED_PORTFOLIO_TYPES, SUPPORTED_REVIEW_STATUSES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import AllocationPolicy, BenchmarkAssignment, CashMovement, ComplianceBreach, CorporateAction, HoldingRecord, PerformanceAttribution, PortfolioBook, PortfolioEvidence, PortfolioReview, PortfolioValuation, RiskExposure  # type: ignore
	from portfolio_runtime import allocation_totals_100, normalize_code, normalize_currency, positive_minor, positive_quantity  # type: ignore


class PortfolioManagementService:
	"""In-memory Portfolio Management runtime for generated APG applications."""

	def __init__(self) -> None:
		self.portfolios: dict[str, PortfolioBook] = {}
		self.holdings: dict[str, HoldingRecord] = {}
		self.allocations: dict[str, AllocationPolicy] = {}
		self.valuations: dict[str, PortfolioValuation] = {}
		self.benchmarks: dict[str, BenchmarkAssignment] = {}
		self.risk: dict[str, RiskExposure] = {}
		self.attribution: dict[str, PerformanceAttribution] = {}
		self.cash: dict[str, CashMovement] = {}
		self.corporate_actions: dict[str, CorporateAction] = {}
		self.compliance: dict[str, ComplianceBreach] = {}
		self.reviews: dict[str, PortfolioReview] = {}
		self.evidence: dict[str, PortfolioEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_portfolio_book(self, portfolio_id: str, tenant_id: str, owner_id: str, name: str, portfolio_type: str, base_currency: str, policy_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		portfolio_type = normalize_code(portfolio_type)
		base_currency = normalize_currency(base_currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "create_portfolio_book", "owner_present": bool(owner_id), "portfolio_type_supported": portfolio_type in SUPPORTED_PORTFOLIO_TYPES, "currency_supported": base_currency in SUPPORTED_CURRENCIES})
		portfolio = PortfolioBook(portfolio_id, tenant_id, owner_id, name, portfolio_type, base_currency, policy_reference)
		self.portfolios[portfolio_id] = portfolio
		self._audit(tenant_id, "portfolio_book_created", portfolio_id)
		return portfolio.to_dict()

	def record_holding(self, holding_id: str, tenant_id: str, portfolio_id: str, instrument_id: str, quantity: float, cost_minor: int, currency: str) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		currency = normalize_currency(currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_holding", "portfolio_present": portfolio is not None, "instrument_present": bool(instrument_id), "positive_quantity": positive_quantity(quantity), "positive_cost": positive_minor(cost_minor)})
		holding = HoldingRecord(holding_id, tenant_id, portfolio_id, instrument_id, float(quantity), int(cost_minor), currency)
		self.holdings[holding_id] = holding
		self._audit(tenant_id, "portfolio_holding_recorded", holding_id)
		return holding.to_dict()

	def activate_allocation_policy(self, allocation_id: str, tenant_id: str, portfolio_id: str, target_allocation: dict[str, float], policy_reference: str) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "activate_allocation_policy", "portfolio_present": portfolio is not None, "allocation_totals_100": allocation_totals_100(target_allocation), "policy_reference_present": bool(policy_reference)})
		allocation = AllocationPolicy(allocation_id, tenant_id, portfolio_id, dict(target_allocation), policy_reference)
		self.allocations[allocation_id] = allocation
		self._audit(tenant_id, "allocation_policy_activated", allocation_id)
		return allocation.to_dict()

	def record_valuation(self, valuation_id: str, tenant_id: str, portfolio_id: str, market_value_minor: int, currency: str, valuation_date: str, source_reference: str) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		currency = normalize_currency(currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_valuation", "portfolio_present": portfolio is not None, "positive_market_value": positive_minor(market_value_minor), "source_present": bool(source_reference), "valuation_date_present": bool(valuation_date)})
		valuation = PortfolioValuation(valuation_id, tenant_id, portfolio_id, int(market_value_minor), currency, valuation_date, source_reference)
		self.valuations[valuation_id] = valuation
		self._audit(tenant_id, "portfolio_valuation_recorded", valuation_id)
		return valuation.to_dict()

	def assign_benchmark(self, benchmark_id: str, tenant_id: str, portfolio_id: str, index_id: str, policy_reference: str) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "assign_benchmark", "portfolio_present": portfolio is not None, "index_present": bool(index_id)})
		benchmark = BenchmarkAssignment(benchmark_id, tenant_id, portfolio_id, index_id, policy_reference)
		self.benchmarks[benchmark_id] = benchmark
		self._audit(tenant_id, "benchmark_assigned", benchmark_id)
		return benchmark.to_dict()

	def record_risk_exposure(self, exposure_id: str, tenant_id: str, portfolio_id: str, metric: str, value: float, as_of_date: str, source_reference: str, limit_reference: str) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_risk_exposure", "portfolio_present": portfolio is not None, "source_present": bool(source_reference), "as_of_date_present": bool(as_of_date)})
		exposure = RiskExposure(exposure_id, tenant_id, portfolio_id, normalize_code(metric), float(value), as_of_date, source_reference, limit_reference)
		self.risk[exposure_id] = exposure
		self._audit(tenant_id, "risk_exposure_recorded", exposure_id)
		return exposure.to_dict()

	def record_attribution(self, attribution_id: str, tenant_id: str, portfolio_id: str, period: str, benchmark_id: str, source_reference: str, contributions: dict[str, float]) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_attribution", "portfolio_present": portfolio is not None, "period_present": bool(period), "source_present": bool(source_reference)})
		attribution = PerformanceAttribution(attribution_id, tenant_id, portfolio_id, period, benchmark_id, source_reference, dict(contributions))
		self.attribution[attribution_id] = attribution
		self._audit(tenant_id, "performance_attribution_recorded", attribution_id)
		return attribution.to_dict()

	def record_cash_movement(self, movement_id: str, tenant_id: str, portfolio_id: str, amount_minor: int, currency: str, reference: str) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		currency = normalize_currency(currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_cash_movement", "portfolio_present": portfolio is not None, "positive_amount": positive_minor(amount_minor), "currency_supported": currency in SUPPORTED_CURRENCIES, "reference_present": bool(reference)})
		movement = CashMovement(movement_id, tenant_id, portfolio_id, int(amount_minor), currency, reference)
		self.cash[movement_id] = movement
		self._audit(tenant_id, "cash_movement_recorded", movement_id)
		return movement.to_dict()

	def record_corporate_action(self, action_id: str, tenant_id: str, instrument_id: str, action_type: str, effective_date: str, evidence_reference: str) -> dict[str, Any]:
		action_type = normalize_code(action_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_corporate_action", "action_type_supported": action_type in SUPPORTED_CORPORATE_ACTIONS, "evidence_present": bool(evidence_reference) and bool(instrument_id) and bool(effective_date)})
		action = CorporateAction(action_id, tenant_id, instrument_id, action_type, effective_date, evidence_reference)
		self.corporate_actions[action_id] = action
		self._audit(tenant_id, "corporate_action_recorded", action_id)
		return action.to_dict()

	def record_compliance_breach(self, breach_id: str, tenant_id: str, portfolio_id: str, severity: str, evidence_reference: str) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_compliance_breach", "portfolio_present": portfolio is not None, "severity_supported": severity in SUPPORTED_BREACH_SEVERITIES, "evidence_present": bool(evidence_reference)})
		breach = ComplianceBreach(breach_id, tenant_id, portfolio_id, severity, evidence_reference)
		self.compliance[breach_id] = breach
		self._audit(tenant_id, "compliance_breach_recorded", breach_id)
		return breach.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "evidence_present": bool(evidence_reference) and bool(reviewer_id)})
		review = PortfolioReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = review
		self._audit(tenant_id, "portfolio_review_recorded", review_id)
		return review.to_dict()

	def register_portfolio_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_portfolio_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = PortfolioEvidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self.evidence[agent_id] = evidence
		self._audit(tenant_id, "portfolio_agent_registered", agent_id)
		return evidence.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "portfolio_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "portfolio_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.portfolio.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "portfolio_count": self._count(self.portfolios, tenant_id), "holding_count": self._count(self.holdings, tenant_id), "allocation_count": self._count(self.allocations, tenant_id), "valuation_count": self._count(self.valuations, tenant_id), "benchmark_count": self._count(self.benchmarks, tenant_id), "risk_count": self._count(self.risk, tenant_id), "attribution_count": self._count(self.attribution, tenant_id), "cash_count": self._count(self.cash, tenant_id), "corporate_action_count": self._count(self.corporate_actions, tenant_id), "compliance_count": self._count(self.compliance, tenant_id), "review_count": self._count(self.reviews, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_portfolio_or_none(self, item_id: str, tenant_id: str) -> PortfolioBook | None:
		item = self.portfolios.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "portfolio_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "portfolio_policy_denied")


PortfolioService = PortfolioManagementService
