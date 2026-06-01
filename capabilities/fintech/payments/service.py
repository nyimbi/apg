"""Executable service layer for APG Digital Payments."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_INSTRUMENT_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import PaymentAccount, PaymentEvidence, PaymentInstrument, PaymentOrder, money
	from .payments_runtime import is_high_value, normalize_amount, settlement_variance_detected
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_INSTRUMENT_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import PaymentAccount, PaymentEvidence, PaymentInstrument, PaymentOrder, money  # type: ignore
	from payments_runtime import is_high_value, normalize_amount, settlement_variance_detected  # type: ignore


class DigitalPaymentsService:
	"""Dependency-light payment lifecycle runtime for generated applications."""

	def __init__(self) -> None:
		self.accounts: dict[str, PaymentAccount] = {}
		self.instruments: dict[str, PaymentInstrument] = {}
		self.orders: dict[str, PaymentOrder] = {}
		self.evidence: dict[str, PaymentEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def open_payment_account(
		self,
		account_id: str,
		tenant_id: str,
		owner_reference: str,
		currency: str,
		metadata: dict[str, Any] | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "open_payment_account",
			"owner_present": bool(owner_reference),
			"currency_supported": currency in SUPPORTED_CURRENCIES,
		})
		if account_id in self.accounts:
			raise ValueError(f"payment account already exists: {account_id}")
		account = PaymentAccount(account_id, tenant_id, owner_reference, currency, metadata=dict(metadata or {}))
		self.accounts[account_id] = account
		self._audit(tenant_id, "payment_account_opened", account_id)
		return account.to_dict()

	def register_instrument(
		self,
		instrument_id: str,
		tenant_id: str,
		account_id: str,
		instrument_type: str,
		token_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		account = self.accounts.get(account_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_instrument",
			"account_present": account is not None and account.tenant_id == tenant_id,
			"instrument_type_supported": instrument_type in SUPPORTED_INSTRUMENT_TYPES,
			"token_reference_present": bool(token_reference),
		})
		if instrument_id in self.instruments:
			raise ValueError(f"payment instrument already exists: {instrument_id}")
		instrument = PaymentInstrument(instrument_id, tenant_id, account_id, instrument_type, token_reference)
		self.instruments[instrument_id] = instrument
		self._audit(tenant_id, "payment_instrument_registered", instrument_id)
		return instrument.to_dict()

	def create_payment_order(
		self,
		order_id: str,
		tenant_id: str,
		account_id: str,
		instrument_id: str,
		amount: Decimal | int | str,
		currency: str,
		counterparty_reference: str,
		purpose: str = "payment",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		amount_decimal = normalize_amount(amount)
		account = self.accounts.get(account_id)
		instrument = self.instruments.get(instrument_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_payment_order",
			"amount": amount_decimal,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"account_present": account is not None and account.tenant_id == tenant_id,
			"instrument_present": instrument is not None and instrument.tenant_id == tenant_id,
		})
		if order_id in self.orders:
			raise ValueError(f"payment order already exists: {order_id}")
		order = PaymentOrder(order_id, tenant_id, account_id, instrument_id, amount_decimal, currency, counterparty_reference, purpose)
		self.orders[order_id] = order
		self._audit(tenant_id, "payment_order_created", order_id)
		return order.to_dict()

	def screen_payment_risk(
		self,
		evidence_id: str,
		tenant_id: str,
		order_id: str,
		risk_level: str,
		risk_score: Decimal | int | str,
		reviewer_id: str = "",
	) -> dict[str, Any]:
		order = self._tenant_order(order_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "screen_payment_risk",
			"risk_level": risk_level,
			"review_recorded": bool(reviewer_id),
		})
		order.risk_level = risk_level
		order.risk_score = normalize_amount(risk_score)
		evidence = self._record_evidence(evidence_id, tenant_id, "risk", order_id, "screened", {"risk_level": risk_level, "risk_score": money(order.risk_score), "reviewer_id": reviewer_id})
		self._audit(tenant_id, "payment_risk_screened", order_id)
		return evidence

	def authorize_payment(
		self,
		evidence_id: str,
		tenant_id: str,
		order_id: str,
		provider_reference: str,
		approval_id: str = "",
	) -> dict[str, Any]:
		order = self._tenant_order(order_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "authorize_payment",
			"risk_level": order.risk_level,
			"provider_present": bool(provider_reference),
			"high_value": is_high_value(order.amount),
			"approval_recorded": bool(approval_id),
		})
		order.status = "authorized"
		order.authorized_amount = order.amount
		evidence = self._record_evidence(evidence_id, tenant_id, "authorization", order_id, "authorized", {"provider_reference": provider_reference, "approval_id": approval_id})
		self._audit(tenant_id, "payment_authorized", order_id)
		return evidence

	def capture_payment(self, evidence_id: str, tenant_id: str, order_id: str, amount: Decimal | int | str) -> dict[str, Any]:
		order = self._tenant_order(order_id, tenant_id)
		amount_decimal = normalize_amount(amount)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "capture_payment",
			"authorized_payment_present": order.authorized_amount > 0,
			"overcapture": order.captured_amount + amount_decimal > order.authorized_amount,
		})
		order.captured_amount += amount_decimal
		order.status = "captured"
		evidence = self._record_evidence(evidence_id, tenant_id, "capture", order_id, "captured", {"amount": money(amount_decimal)})
		self._audit(tenant_id, "payment_captured", order_id)
		return evidence

	def refund_payment(self, evidence_id: str, tenant_id: str, order_id: str, amount: Decimal | int | str, reason: str) -> dict[str, Any]:
		order = self._tenant_order(order_id, tenant_id)
		amount_decimal = normalize_amount(amount)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "refund_payment",
			"captured_payment_present": order.captured_amount > 0,
			"overrefund": order.refunded_amount + amount_decimal > order.captured_amount,
		})
		order.refunded_amount += amount_decimal
		order.status = "refunded"
		evidence = self._record_evidence(evidence_id, tenant_id, "refund", order_id, "refunded", {"amount": money(amount_decimal), "reason": reason})
		self._audit(tenant_id, "payment_refunded", order_id)
		return evidence

	def schedule_payout(self, payout_id: str, tenant_id: str, account_id: str, amount: Decimal | int | str, currency: str, destination_reference: str) -> dict[str, Any]:
		account = self.accounts.get(account_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "schedule_payout",
			"account_present": account is not None and account.tenant_id == tenant_id,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"destination_present": bool(destination_reference),
		})
		evidence = self._record_evidence(payout_id, tenant_id, "payout", account_id, "scheduled", {"amount": money(normalize_amount(amount)), "currency": currency, "destination_reference": destination_reference})
		self._audit(tenant_id, "payout_scheduled", account_id)
		return evidence

	def record_settlement(self, settlement_id: str, tenant_id: str, order_id: str, settlement_reference: str, amount: Decimal | int | str, variance_amount: Decimal | int | str = 0, review_id: str = "") -> dict[str, Any]:
		self._tenant_order(order_id, tenant_id)
		variance = normalize_amount(variance_amount)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_settlement",
			"variance_detected": settlement_variance_detected(variance),
			"review_recorded": bool(review_id),
		})
		evidence = self._record_evidence(settlement_id, tenant_id, "settlement", order_id, "settled", {"settlement_reference": settlement_reference, "amount": money(normalize_amount(amount)), "variance_amount": money(variance), "review_id": review_id})
		self._audit(tenant_id, "settlement_recorded", order_id)
		return evidence

	def open_dispute(self, dispute_id: str, tenant_id: str, order_id: str, owner_id: str, reason: str) -> dict[str, Any]:
		self._tenant_order(order_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_dispute",
			"owner_present": bool(owner_id),
		})
		evidence = self._record_evidence(dispute_id, tenant_id, "dispute", order_id, "opened", {"owner_id": owner_id, "reason": reason})
		self._audit(tenant_id, "payment_dispute_opened", order_id)
		return evidence

	def register_payment_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_payment_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "payment_agent_registered", agent_id)
		return evidence

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "payment_batch",
			"event_stream": event_stream,
		})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.payments.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		orders = [order for order in self.orders.values() if order.tenant_id == tenant_id]
		evidence = [item for item in self.evidence.values() if item.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"account_count": sum(1 for item in self.accounts.values() if item.tenant_id == tenant_id),
			"instrument_count": sum(1 for item in self.instruments.values() if item.tenant_id == tenant_id),
			"order_count": len(orders),
			"captured_volume": money(sum((order.captured_amount for order in orders), Decimal("0"))),
			"open_disputes": sum(1 for item in evidence if item.kind == "dispute" and item.status == "opened"),
			"audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	def list_orders(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		orders = self.orders.values()
		if tenant_id is not None:
			orders = [order for order in orders if order.tenant_id == tenant_id]
		return [order.to_dict() for order in sorted(orders, key=lambda item: item.id)]

	def list_evidence(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		evidence = self.evidence.values()
		if tenant_id is not None:
			evidence = [item for item in evidence if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(evidence, key=lambda item: item.id)]

	def _tenant_order(self, order_id: str, tenant_id: str) -> PaymentOrder:
		order = self.orders.get(order_id)
		if order is None or order.tenant_id != tenant_id:
			raise KeyError(f"unknown payment order: {order_id}")
		return order

	def _record_evidence(self, evidence_id: str, tenant_id: str, kind: str, reference_id: str, status: str, metadata: dict[str, Any]) -> dict[str, Any]:
		evidence = PaymentEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "payment_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "payment_policy_denied")


FintechPaymentsService = DigitalPaymentsService
