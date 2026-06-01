"""Executable service layer for APG Wealth Management."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CURRENCIES, SUPPORTED_HORIZONS, SUPPORTED_MANDATES, SUPPORTED_ORDER_SIDES, SUPPORTED_RISK_PROFILES, SUPPORTED_TOLERANCES, evaluate_capability_rules, get_capability_contract
	from .models import AdvisoryMandate, ClientProfile, FeeSchedule, PerformanceSnapshot, Portfolio, RebalanceProposal, SuitabilityProfile, WealthEvidence, WealthOrder
	from .wealth_runtime import allocation_totals_100, normalize_code, normalize_codes, normalize_currency, percent_bounded
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CURRENCIES, SUPPORTED_HORIZONS, SUPPORTED_MANDATES, SUPPORTED_ORDER_SIDES, SUPPORTED_RISK_PROFILES, SUPPORTED_TOLERANCES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import AdvisoryMandate, ClientProfile, FeeSchedule, PerformanceSnapshot, Portfolio, RebalanceProposal, SuitabilityProfile, WealthEvidence, WealthOrder  # type: ignore
	from wealth_runtime import allocation_totals_100, normalize_code, normalize_codes, normalize_currency, percent_bounded  # type: ignore


class WealthManagementService:
	"""In-memory wealth runtime for generated APG applications."""

	def __init__(self) -> None:
		self.clients: dict[str, ClientProfile] = {}
		self.suitability: dict[str, SuitabilityProfile] = {}
		self.portfolios: dict[str, Portfolio] = {}
		self.mandates: dict[str, AdvisoryMandate] = {}
		self.rebalances: dict[str, RebalanceProposal] = {}
		self.orders: dict[str, WealthOrder] = {}
		self.performance: dict[str, PerformanceSnapshot] = {}
		self.fees: dict[str, FeeSchedule] = {}
		self.evidence: dict[str, WealthEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_client_profile(self, client_id: str, tenant_id: str, name: str, kyc_reference: str, tax_reference: str, risk_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_client_profile", "kyc_present": bool(kyc_reference), "tax_present": bool(tax_reference), "risk_present": bool(risk_reference)})
		client = ClientProfile(client_id, tenant_id, name, kyc_reference, tax_reference, risk_reference)
		self.clients[client_id] = client
		self._audit(tenant_id, "client_profile_registered", client_id)
		return client.to_dict()

	def capture_suitability_profile(self, suitability_id: str, tenant_id: str, client_id: str, risk_profile: str, risk_tolerance: str, horizon: str, goals: list[str], policy_attached: bool = True) -> dict[str, Any]:
		client = self._tenant_client_or_none(client_id, tenant_id)
		risk_profile = normalize_code(risk_profile)
		risk_tolerance = normalize_code(risk_tolerance)
		horizon = normalize_code(horizon)
		goals = normalize_codes(goals)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "capture_suitability_profile", "client_present": client is not None, "risk_profile_supported": risk_profile in SUPPORTED_RISK_PROFILES, "tolerance_supported": risk_tolerance in SUPPORTED_TOLERANCES, "horizon_supported": horizon in SUPPORTED_HORIZONS, "goals_present": bool(goals)})
		profile = SuitabilityProfile(suitability_id, tenant_id, client_id, risk_profile, risk_tolerance, horizon, goals)
		self.suitability[suitability_id] = profile
		self._audit(tenant_id, "suitability_profile_captured", suitability_id)
		return profile.to_dict()

	def create_portfolio(self, portfolio_id: str, tenant_id: str, client_id: str, name: str, base_currency: str, advisor_id: str, policy_reference: str) -> dict[str, Any]:
		client = self._tenant_client_or_none(client_id, tenant_id)
		base_currency = normalize_currency(base_currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "create_portfolio", "client_present": client is not None, "currency_supported": base_currency in SUPPORTED_CURRENCIES, "advisor_present": bool(advisor_id), "policy_present": bool(policy_reference)})
		portfolio = Portfolio(portfolio_id, tenant_id, client_id, name, base_currency, advisor_id, policy_reference)
		self.portfolios[portfolio_id] = portfolio
		self._audit(tenant_id, "portfolio_created", portfolio_id)
		return portfolio.to_dict()

	def create_advisory_mandate(self, mandate_id: str, tenant_id: str, portfolio_id: str, suitability_id: str, mandate_type: str, policy_reference: str) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		suitability = self._tenant_suitability_or_none(suitability_id, tenant_id)
		mandate_type = normalize_code(mandate_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "create_advisory_mandate", "portfolio_present": portfolio is not None, "suitability_present": suitability is not None, "mandate_type_supported": mandate_type in SUPPORTED_MANDATES, "policy_present": bool(policy_reference)})
		mandate = AdvisoryMandate(mandate_id, tenant_id, portfolio_id, suitability_id, mandate_type, policy_reference)
		self.mandates[mandate_id] = mandate
		self._audit(tenant_id, "advisory_mandate_created", mandate_id)
		return mandate.to_dict()

	def propose_rebalance(self, rebalance_id: str, tenant_id: str, portfolio_id: str, mandate_id: str, target_allocation: dict[str, float], analysis_reference: str) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		mandate = self._tenant_mandate_or_none(mandate_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "propose_rebalance", "portfolio_present": portfolio is not None, "mandate_present": mandate is not None, "mandate_matches_portfolio": mandate is not None and mandate.portfolio_id == portfolio_id, "allocation_totals_100": allocation_totals_100(target_allocation), "analysis_present": bool(analysis_reference)})
		rebalance = RebalanceProposal(rebalance_id, tenant_id, portfolio_id, mandate_id, dict(target_allocation), analysis_reference)
		self.rebalances[rebalance_id] = rebalance
		self._audit(tenant_id, "rebalance_proposed", rebalance_id)
		return rebalance.to_dict()

	def stage_order(self, order_id: str, tenant_id: str, portfolio_id: str, instrument_id: str, side: str, quantity: float, notional_minor: int, risk_reference: str, human_approval: str = "") -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		side = normalize_code(side)
		large_order = int(notional_minor) >= get_capability_contract(tenant_id)["configuration"]["orders"]["large_order_threshold_minor"]
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "stage_order", "portfolio_present": portfolio is not None, "side_supported": side in SUPPORTED_ORDER_SIDES, "positive_quantity": float(quantity) > 0, "risk_reference_present": bool(risk_reference), "large_order": large_order, "human_approval_recorded": bool(human_approval)})
		order = WealthOrder(order_id, tenant_id, portfolio_id, instrument_id, side, float(quantity), int(notional_minor), risk_reference, human_approval)
		self.orders[order_id] = order
		self._audit(tenant_id, "order_staged", order_id)
		return order.to_dict()

	def record_performance(self, snapshot_id: str, tenant_id: str, portfolio_id: str, period: str, valuation_reference: str, benchmark_reference: str, return_percent: float) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_performance", "portfolio_present": portfolio is not None, "valuation_present": bool(valuation_reference), "benchmark_present": bool(benchmark_reference)})
		snapshot = PerformanceSnapshot(snapshot_id, tenant_id, portfolio_id, period, valuation_reference, benchmark_reference, float(return_percent))
		self.performance[snapshot_id] = snapshot
		self._audit(tenant_id, "performance_recorded", snapshot_id)
		return snapshot.to_dict()

	def record_fee_schedule(self, fee_id: str, tenant_id: str, portfolio_id: str, advisory_percent: float, performance_percent: float, platform_percent: float, contract_reference: str) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		percent_bounded_all = all(percent_bounded(value) for value in [advisory_percent, performance_percent, platform_percent])
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_fee_schedule", "portfolio_present": portfolio is not None, "percent_bounded": percent_bounded_all, "contract_present": bool(contract_reference)})
		fee = FeeSchedule(fee_id, tenant_id, portfolio_id, float(advisory_percent), float(performance_percent), float(platform_percent), contract_reference)
		self.fees[fee_id] = fee
		self._audit(tenant_id, "fee_schedule_recorded", fee_id)
		return fee.to_dict()

	def register_wealth_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_wealth_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = WealthEvidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self.evidence[agent_id] = evidence
		self._audit(tenant_id, "wealth_agent_registered", agent_id)
		return evidence.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "wealth_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.wealth.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "client_count": self._count(self.clients, tenant_id), "suitability_count": self._count(self.suitability, tenant_id), "portfolio_count": self._count(self.portfolios, tenant_id), "mandate_count": self._count(self.mandates, tenant_id), "rebalance_count": self._count(self.rebalances, tenant_id), "order_count": self._count(self.orders, tenant_id), "performance_count": self._count(self.performance, tenant_id), "fee_count": self._count(self.fees, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_client_or_none(self, item_id: str, tenant_id: str) -> ClientProfile | None:
		item = self.clients.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_suitability_or_none(self, item_id: str, tenant_id: str) -> SuitabilityProfile | None:
		item = self.suitability.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_portfolio_or_none(self, item_id: str, tenant_id: str) -> Portfolio | None:
		item = self.portfolios.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_mandate_or_none(self, item_id: str, tenant_id: str) -> AdvisoryMandate | None:
		item = self.mandates.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "wealth_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "wealth_policy_denied")


WealthService = WealthManagementService
