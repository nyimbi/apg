"""Executable service layer for APG Algorithmic Trading."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_SEVERITIES, SUPPORTED_ASSET_CLASSES, SUPPORTED_ORDER_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_STRATEGY_TYPES, SUPPORTED_VENUES, evaluate_capability_rules, get_capability_contract
	from .models import BacktestRun, ExecutionRecord, OrderIntent, PositionSnapshot, RiskLimit, SignalSource, SurveillanceAlert, TradingEvidence, TradingReview, TradingStrategy
	from .trading_runtime import normalize_code, positive_count, positive_quantity, positive_value
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_SEVERITIES, SUPPORTED_ASSET_CLASSES, SUPPORTED_ORDER_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_STRATEGY_TYPES, SUPPORTED_VENUES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import BacktestRun, ExecutionRecord, OrderIntent, PositionSnapshot, RiskLimit, SignalSource, SurveillanceAlert, TradingEvidence, TradingReview, TradingStrategy  # type: ignore
	from trading_runtime import normalize_code, positive_count, positive_quantity, positive_value  # type: ignore


class AlgorithmicTradingService:
	"""In-memory Algorithmic Trading runtime for generated APG applications."""

	def __init__(self) -> None:
		self.strategies: dict[str, TradingStrategy] = {}
		self.signals: dict[str, SignalSource] = {}
		self.backtests: dict[str, BacktestRun] = {}
		self.risk_limits: dict[str, RiskLimit] = {}
		self.orders: dict[str, OrderIntent] = {}
		self.executions: dict[str, ExecutionRecord] = {}
		self.positions: dict[str, PositionSnapshot] = {}
		self.surveillance: dict[str, SurveillanceAlert] = {}
		self.reviews: dict[str, TradingReview] = {}
		self.evidence: dict[str, TradingEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_strategy(self, strategy_id: str, tenant_id: str, owner_id: str, name: str, strategy_type: str, asset_class: str, policy_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		strategy_type = normalize_code(strategy_type)
		asset_class = normalize_code(asset_class)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_strategy", "owner_present": bool(owner_id), "strategy_type_supported": strategy_type in SUPPORTED_STRATEGY_TYPES, "asset_class_supported": asset_class in SUPPORTED_ASSET_CLASSES, "policy_reference_present": bool(policy_reference)})
		strategy = TradingStrategy(strategy_id, tenant_id, owner_id, name, strategy_type, asset_class, policy_reference)
		self.strategies[strategy_id] = strategy
		self._audit(tenant_id, "trading_strategy_registered", strategy_id)
		return strategy.to_dict()

	def attach_signal_source(self, signal_id: str, tenant_id: str, strategy_id: str, source_reference: str, freshness_sla: str, lineage_reference: str) -> dict[str, Any]:
		strategy = self._tenant_strategy_or_none(strategy_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "attach_signal_source", "strategy_present": strategy is not None, "source_present": bool(source_reference), "freshness_present": bool(freshness_sla)})
		signal = SignalSource(signal_id, tenant_id, strategy_id, source_reference, freshness_sla, lineage_reference)
		self.signals[signal_id] = signal
		self._audit(tenant_id, "signal_source_attached", signal_id)
		return signal.to_dict()

	def record_backtest(self, backtest_id: str, tenant_id: str, strategy_id: str, period: str, trade_count: int, data_source_reference: str, metrics: dict[str, float]) -> dict[str, Any]:
		strategy = self._tenant_strategy_or_none(strategy_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_backtest", "strategy_present": strategy is not None, "period_present": bool(period), "positive_trade_count": positive_count(trade_count), "data_source_present": bool(data_source_reference)})
		backtest = BacktestRun(backtest_id, tenant_id, strategy_id, period, int(trade_count), data_source_reference, dict(metrics))
		self.backtests[backtest_id] = backtest
		self._audit(tenant_id, "backtest_recorded", backtest_id)
		return backtest.to_dict()

	def set_risk_limit(self, limit_id: str, tenant_id: str, strategy_id: str, metric: str, limit_value: float, approval_reference: str) -> dict[str, Any]:
		strategy = self._tenant_strategy_or_none(strategy_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "set_risk_limit", "strategy_present": strategy is not None, "metric_present": bool(metric), "positive_limit": positive_value(limit_value), "approval_present": bool(approval_reference)})
		limit = RiskLimit(limit_id, tenant_id, strategy_id, normalize_code(metric), float(limit_value), approval_reference)
		self.risk_limits[limit_id] = limit
		self._audit(tenant_id, "risk_limit_set", limit_id)
		return limit.to_dict()

	def stage_order_intent(self, order_id: str, tenant_id: str, strategy_id: str, risk_limit_id: str, instrument_id: str, order_type: str, quantity: float, approval_reference: str) -> dict[str, Any]:
		strategy = self._tenant_strategy_or_none(strategy_id, tenant_id)
		limit = self._tenant_risk_limit_or_none(risk_limit_id, tenant_id)
		order_type = normalize_code(order_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "stage_order_intent", "strategy_present": strategy is not None, "risk_limit_present": limit is not None, "order_type_supported": order_type in SUPPORTED_ORDER_TYPES, "instrument_present": bool(instrument_id), "positive_quantity": positive_quantity(quantity), "approval_present": bool(approval_reference)})
		order = OrderIntent(order_id, tenant_id, strategy_id, risk_limit_id, instrument_id, order_type, float(quantity), approval_reference)
		self.orders[order_id] = order
		self._audit(tenant_id, "order_intent_staged", order_id)
		return order.to_dict()

	def record_execution(self, execution_id: str, tenant_id: str, order_id: str, venue: str, filled_quantity: float, source_reference: str) -> dict[str, Any]:
		order = self._tenant_order_or_none(order_id, tenant_id)
		venue = normalize_code(venue)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_execution", "order_present": order is not None, "venue_supported": venue in SUPPORTED_VENUES, "positive_filled_quantity": positive_quantity(filled_quantity), "source_present": bool(source_reference)})
		execution = ExecutionRecord(execution_id, tenant_id, order_id, venue, float(filled_quantity), source_reference)
		self.executions[execution_id] = execution
		self._audit(tenant_id, "execution_recorded", execution_id)
		return execution.to_dict()

	def record_position_snapshot(self, snapshot_id: str, tenant_id: str, strategy_id: str, as_of_date: str, gross_exposure_minor: int, net_exposure_minor: int, source_reference: str) -> dict[str, Any]:
		strategy = self._tenant_strategy_or_none(strategy_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_position_snapshot", "strategy_present": strategy is not None, "as_of_date_present": bool(as_of_date), "source_present": bool(source_reference)})
		snapshot = PositionSnapshot(snapshot_id, tenant_id, strategy_id, as_of_date, int(gross_exposure_minor), int(net_exposure_minor), source_reference)
		self.positions[snapshot_id] = snapshot
		self._audit(tenant_id, "position_snapshot_recorded", snapshot_id)
		return snapshot.to_dict()

	def record_surveillance_alert(self, alert_id: str, tenant_id: str, strategy_id: str, severity: str, evidence_reference: str) -> dict[str, Any]:
		strategy = self._tenant_strategy_or_none(strategy_id, tenant_id)
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_surveillance_alert", "strategy_present": strategy is not None, "severity_supported": severity in SUPPORTED_ALERT_SEVERITIES, "evidence_present": bool(evidence_reference)})
		alert = SurveillanceAlert(alert_id, tenant_id, strategy_id, severity, evidence_reference)
		self.surveillance[alert_id] = alert
		self._audit(tenant_id, "surveillance_alert_recorded", alert_id)
		return alert.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "evidence_present": bool(evidence_reference) and bool(reviewer_id)})
		review = TradingReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = review
		self._audit(tenant_id, "trading_review_recorded", review_id)
		return review.to_dict()

	def register_trading_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_trading_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = TradingEvidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self.evidence[agent_id] = evidence
		self._audit(tenant_id, "trading_agent_registered", agent_id)
		return evidence.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "trading_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "trading_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.trading.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "strategy_count": self._count(self.strategies, tenant_id), "signal_count": self._count(self.signals, tenant_id), "backtest_count": self._count(self.backtests, tenant_id), "risk_limit_count": self._count(self.risk_limits, tenant_id), "order_count": self._count(self.orders, tenant_id), "execution_count": self._count(self.executions, tenant_id), "position_count": self._count(self.positions, tenant_id), "surveillance_count": self._count(self.surveillance, tenant_id), "review_count": self._count(self.reviews, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_strategy_or_none(self, item_id: str, tenant_id: str) -> TradingStrategy | None:
		item = self.strategies.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_risk_limit_or_none(self, item_id: str, tenant_id: str) -> RiskLimit | None:
		item = self.risk_limits.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_order_or_none(self, item_id: str, tenant_id: str) -> OrderIntent | None:
		item = self.orders.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "trading_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "trading_policy_denied")


TradingService = AlgorithmicTradingService
