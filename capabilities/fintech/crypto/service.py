"""Executable service layer for APG Cryptocurrency Services."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSET_TYPES, SUPPORTED_CUSTODY_MODELS, SUPPORTED_ORDER_SIDES, SUPPORTED_ORDER_TYPES, SUPPORTED_PRICE_SOURCES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SCREENING_STATUSES, SUPPORTED_SCREENING_TYPES, SUPPORTED_TRADE_STATUSES, SUPPORTED_TRANSFER_STATUSES, SUPPORTED_TRANSFER_TYPES, evaluate_capability_rules, get_capability_contract
	from .crypto_runtime import non_negative_int, normalize_code, normalize_symbol, positive_int, present
	from .models import ComplianceScreening, CryptoAgent, CryptoAsset, CryptoBalance, CryptoOrder, CryptoReview, CryptoTrade, CryptoTransfer, CustodyAccount, PriceSnapshot
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSET_TYPES, SUPPORTED_CUSTODY_MODELS, SUPPORTED_ORDER_SIDES, SUPPORTED_ORDER_TYPES, SUPPORTED_PRICE_SOURCES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SCREENING_STATUSES, SUPPORTED_SCREENING_TYPES, SUPPORTED_TRADE_STATUSES, SUPPORTED_TRANSFER_STATUSES, SUPPORTED_TRANSFER_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from crypto_runtime import non_negative_int, normalize_code, normalize_symbol, positive_int, present  # type: ignore
	from models import ComplianceScreening, CryptoAgent, CryptoAsset, CryptoBalance, CryptoOrder, CryptoReview, CryptoTrade, CryptoTransfer, CustodyAccount, PriceSnapshot  # type: ignore


class CryptocurrencyServicesService:
	"""Dependency-light cryptocurrency runtime for generated APG applications."""

	def __init__(self) -> None:
		self.assets: dict[str, CryptoAsset] = {}
		self.accounts: dict[str, CustodyAccount] = {}
		self.balances: dict[str, CryptoBalance] = {}
		self.orders: dict[str, CryptoOrder] = {}
		self.trades: dict[str, CryptoTrade] = {}
		self.transfers: dict[str, CryptoTransfer] = {}
		self.screenings: dict[str, ComplianceScreening] = {}
		self.prices: dict[str, PriceSnapshot] = {}
		self.reviews: dict[str, CryptoReview] = {}
		self.agents: dict[str, CryptoAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_asset(self, asset_id: str, tenant_id: str, symbol: str, asset_type: str, network_reference: str, contract_reference: str, precision: int, owner_id: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		symbol = normalize_symbol(symbol)
		asset_type = normalize_code(asset_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_asset", "symbol_present": present(symbol), "asset_type_supported": asset_type in SUPPORTED_ASSET_TYPES, "network_present": present(network_reference), "precision_valid": non_negative_int(precision), "owner_present": present(owner_id), "evidence_present": present(evidence_reference)})
		item = CryptoAsset(asset_id, tenant_id, symbol, asset_type, network_reference, contract_reference, int(precision), owner_id, evidence_reference)
		self.assets[asset_id] = item
		self._audit(tenant_id, "crypto_asset_registered", asset_id)
		return item.to_dict()

	def open_custody_account(self, account_id: str, tenant_id: str, provider_reference: str, custody_model: str, policy_reference: str, owner_id: str, evidence_reference: str) -> dict[str, Any]:
		custody_model = normalize_code(custody_model)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "open_custody_account", "custody_model_supported": custody_model in SUPPORTED_CUSTODY_MODELS, "provider_present": present(provider_reference), "policy_present": present(policy_reference), "owner_present": present(owner_id), "evidence_present": present(evidence_reference)})
		item = CustodyAccount(account_id, tenant_id, provider_reference, custody_model, policy_reference, owner_id, evidence_reference)
		self.accounts[account_id] = item
		self._audit(tenant_id, "crypto_custody_account_opened", account_id)
		return item.to_dict()

	def record_balance(self, balance_id: str, tenant_id: str, account_id: str, asset_id: str, amount_minor: int, valuation_minor: int, valuation_currency: str, evidence_reference: str) -> dict[str, Any]:
		account = self._tenant_account_or_none(account_id, tenant_id)
		asset = self._tenant_asset_or_none(asset_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_balance", "account_present": account is not None, "asset_present": asset is not None, "amount_valid": non_negative_int(amount_minor), "valuation_valid": non_negative_int(valuation_minor), "currency_present": present(valuation_currency), "evidence_present": present(evidence_reference)})
		item = CryptoBalance(balance_id, tenant_id, account_id, asset_id, int(amount_minor), int(valuation_minor), normalize_symbol(valuation_currency), evidence_reference)
		self.balances[balance_id] = item
		self._audit(tenant_id, "crypto_balance_recorded", balance_id)
		return item.to_dict()

	def create_order(self, order_id: str, tenant_id: str, account_id: str, asset_id: str, side: str, order_type: str, quantity_minor: int, limit_price_minor: int, policy_reference: str, requester_id: str, evidence_reference: str) -> dict[str, Any]:
		account = self._tenant_account_or_none(account_id, tenant_id)
		asset = self._tenant_asset_or_none(asset_id, tenant_id)
		side = normalize_code(side)
		order_type = normalize_code(order_type)
		limit_price_required = order_type in {"limit", "stop_limit"}
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "create_order", "account_present": account is not None, "asset_present": asset is not None, "side_supported": side in SUPPORTED_ORDER_SIDES, "order_type_supported": order_type in SUPPORTED_ORDER_TYPES, "quantity_valid": positive_int(quantity_minor), "limit_price_required": limit_price_required, "limit_price_present": positive_int(limit_price_minor), "policy_present": present(policy_reference), "requester_present": present(requester_id), "evidence_present": present(evidence_reference)})
		item = CryptoOrder(order_id, tenant_id, account_id, asset_id, side, order_type, int(quantity_minor), int(limit_price_minor), policy_reference, requester_id, evidence_reference, "requested")
		self.orders[order_id] = item
		self._audit(tenant_id, "crypto_order_created", order_id)
		return item.to_dict()

	def record_trade(self, trade_id: str, tenant_id: str, order_id: str, venue_reference: str, execution_price_minor: int, quantity_minor: int, fee_minor: int, status: str, settlement_reference: str) -> dict[str, Any]:
		order = self._tenant_order_or_none(order_id, tenant_id)
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_trade", "order_present": order is not None, "venue_present": present(venue_reference), "execution_price_valid": non_negative_int(execution_price_minor), "quantity_valid": positive_int(quantity_minor), "fee_valid": non_negative_int(fee_minor), "status_supported": status in SUPPORTED_TRADE_STATUSES, "settlement_present": present(settlement_reference)})
		item = CryptoTrade(trade_id, tenant_id, order_id, venue_reference, int(execution_price_minor), int(quantity_minor), int(fee_minor), status, settlement_reference)
		self.trades[trade_id] = item
		if order is not None:
			order.status = status
		self._audit(tenant_id, "crypto_trade_recorded", trade_id)
		return item.to_dict()

	def request_transfer(self, transfer_id: str, tenant_id: str, account_id: str, asset_id: str, transfer_type: str, destination_reference: str, amount_minor: int, approval_reference: str, evidence_reference: str, status: str = "requested") -> dict[str, Any]:
		account = self._tenant_account_or_none(account_id, tenant_id)
		asset = self._tenant_asset_or_none(asset_id, tenant_id)
		transfer_type = normalize_code(transfer_type)
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "request_transfer", "account_present": account is not None, "asset_present": asset is not None, "transfer_type_supported": transfer_type in SUPPORTED_TRANSFER_TYPES, "destination_present": present(destination_reference), "amount_valid": positive_int(amount_minor), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference), "status_supported": status in SUPPORTED_TRANSFER_STATUSES})
		item = CryptoTransfer(transfer_id, tenant_id, account_id, asset_id, transfer_type, destination_reference, int(amount_minor), approval_reference, evidence_reference, status)
		self.transfers[transfer_id] = item
		self._audit(tenant_id, "crypto_transfer_requested", transfer_id)
		return item.to_dict()

	def record_screening(self, screening_id: str, tenant_id: str, reference_id: str, screening_type: str, status: str, evidence_reference: str, reviewer_id: str = "") -> dict[str, Any]:
		screening_type = normalize_code(screening_type)
		status = normalize_code(status)
		reviewer_required = status != "clear"
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_screening", "reference_present": present(reference_id), "screening_type_supported": screening_type in SUPPORTED_SCREENING_TYPES, "status_supported": status in SUPPORTED_SCREENING_STATUSES, "evidence_present": present(evidence_reference), "reviewer_required": reviewer_required, "reviewer_present": present(reviewer_id)})
		item = ComplianceScreening(screening_id, tenant_id, reference_id, screening_type, status, evidence_reference, reviewer_id)
		self.screenings[screening_id] = item
		self._audit(tenant_id, "crypto_screening_recorded", screening_id)
		return item.to_dict()

	def record_price(self, price_id: str, tenant_id: str, asset_id: str, source: str, price_minor: int, currency: str, observed_at: str, evidence_reference: str) -> dict[str, Any]:
		asset = self._tenant_asset_or_none(asset_id, tenant_id)
		source = normalize_code(source)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_price", "asset_present": asset is not None, "source_supported": source in SUPPORTED_PRICE_SOURCES, "price_valid": non_negative_int(price_minor), "currency_present": present(currency), "observed_at_present": present(observed_at), "evidence_present": present(evidence_reference)})
		item = PriceSnapshot(price_id, tenant_id, asset_id, source, int(price_minor), normalize_symbol(currency), observed_at, evidence_reference)
		self.prices[price_id] = item
		self._audit(tenant_id, "crypto_price_recorded", price_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = CryptoReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = item
		self._audit(tenant_id, "crypto_review_recorded", review_id)
		return item.to_dict()

	def register_crypto_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_crypto_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = CryptoAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[agent_id] = item
		self._audit(tenant_id, "crypto_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "crypto_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "crypto_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.crypto.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "asset_count": self._count(self.assets, tenant_id), "custody_account_count": self._count(self.accounts, tenant_id), "balance_count": self._count(self.balances, tenant_id), "order_count": self._count(self.orders, tenant_id), "open_order_count": sum(1 for item in self.orders.values() if item.tenant_id == tenant_id and item.status in {"requested", "approved"}), "trade_count": self._count(self.trades, tenant_id), "transfer_count": self._count(self.transfers, tenant_id), "blocked_screening_count": sum(1 for item in self.screenings.values() if item.tenant_id == tenant_id and item.status == "blocked"), "price_count": self._count(self.prices, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_asset_or_none(self, item_id: str, tenant_id: str) -> CryptoAsset | None:
		item = self.assets.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_account_or_none(self, item_id: str, tenant_id: str) -> CustodyAccount | None:
		item = self.accounts.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_order_or_none(self, item_id: str, tenant_id: str) -> CryptoOrder | None:
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
		reasons = ", ".join(action.get("reason", "crypto_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "crypto_policy_denied")


FintechCryptoService = CryptocurrencyServicesService
