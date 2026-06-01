"""Executable service layer for APG Cross-Border Remittance."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_FRAUD_DECISIONS, SUPPORTED_PAYOUT_METHODS, SUPPORTED_PURPOSE_CODES, evaluate_capability_rules, get_capability_contract
	from .models import RemittanceEvidence, RemittanceQuote, RemittanceRefund, RemittanceTransfer
	from .remittance_runtime import corridor_key, normalize_amount, normalize_code, normalize_country, normalize_currency, normalize_rate, payout_state, transfer_band
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_FRAUD_DECISIONS, SUPPORTED_PAYOUT_METHODS, SUPPORTED_PURPOSE_CODES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import RemittanceEvidence, RemittanceQuote, RemittanceRefund, RemittanceTransfer  # type: ignore
	from remittance_runtime import corridor_key, normalize_amount, normalize_code, normalize_country, normalize_currency, normalize_rate, payout_state, transfer_band  # type: ignore


class RemittanceService:
	"""Dependency-light remittance runtime for generated applications."""

	def __init__(self) -> None:
		self.quotes: dict[str, RemittanceQuote] = {}
		self.transfers: dict[str, RemittanceTransfer] = {}
		self.refunds: dict[str, RemittanceRefund] = {}
		self.evidence: dict[str, RemittanceEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_quote(self, quote_id: str, tenant_id: str, source_country: str, destination_country: str, source_currency: str, destination_currency: str, send_amount: float | int | str, fx_rate: float | int | str, fee_amount: float | int | str, expiry: str, policy_attached: bool = True) -> dict[str, Any]:
		source_country = normalize_country(source_country)
		destination_country = normalize_country(destination_country)
		source_currency = normalize_currency(source_currency)
		destination_currency = normalize_currency(destination_currency)
		amount = normalize_amount(send_amount)
		rate = normalize_rate(fx_rate)
		fee = normalize_amount(fee_amount)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "quote_transfer", "corridor_supported": source_country in SUPPORTED_COUNTRIES and destination_country in SUPPORTED_COUNTRIES, "same_country": source_country == destination_country, "source_currency_supported": source_currency in SUPPORTED_CURRENCIES, "destination_currency_supported": destination_currency in SUPPORTED_CURRENCIES, "positive_amount": amount > 0, "positive_fx_rate": rate > 0, "fee_non_negative": fee >= 0, "expiry_present": bool(expiry)})
		if quote_id in self.quotes:
			raise ValueError(f"remittance quote already exists: {quote_id}")
		quote = RemittanceQuote(quote_id, tenant_id, source_country, destination_country, source_currency, destination_currency, amount, rate, fee, expiry)
		self.quotes[quote_id] = quote
		self._audit(tenant_id, "remittance_quote_created", quote_id)
		return quote.to_dict() | {"corridor": corridor_key(source_country, destination_country, source_currency, destination_currency), "transfer_band": transfer_band(amount)}

	def create_transfer(self, transfer_id: str, tenant_id: str, quote_id: str, sender_reference: str, beneficiary_reference: str, sender_kyc_id: str, beneficiary_kyc_id: str, funding_reference: str, payout_method: str, purpose_code: str, source_of_funds: str, aml_screen_id: str, fraud_decision: str, aml_review: bool = False, sanctions_hit: bool = False, human_approval: str = "", policy_attached: bool = True) -> dict[str, Any]:
		quote = self._tenant_quote_or_none(quote_id, tenant_id)
		payout_method = normalize_code(payout_method)
		purpose_code = normalize_code(purpose_code)
		fraud_decision = normalize_code(fraud_decision)
		high_value = quote.send_amount >= 100000 if quote else False
		fraud_review = fraud_decision in {"review", "hold"}
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "create_transfer", "quote_present": quote is not None, "sender_present": bool(sender_reference), "beneficiary_present": bool(beneficiary_reference), "sender_kyc_present": bool(sender_kyc_id), "beneficiary_kyc_present": bool(beneficiary_kyc_id), "funding_present": bool(funding_reference), "payout_method_supported": payout_method in SUPPORTED_PAYOUT_METHODS, "purpose_code_supported": purpose_code in SUPPORTED_PURPOSE_CODES, "source_of_funds_present": bool(source_of_funds), "aml_screen_present": bool(aml_screen_id), "sanctions_hit": sanctions_hit, "fraud_decision_supported": fraud_decision in SUPPORTED_FRAUD_DECISIONS, "fraud_blocked": fraud_decision == "block", "aml_review": aml_review, "fraud_review": fraud_review, "high_value": high_value, "human_approval_recorded": bool(human_approval)})
		if transfer_id in self.transfers:
			raise ValueError(f"remittance transfer already exists: {transfer_id}")
		transfer = RemittanceTransfer(transfer_id, tenant_id, quote_id, sender_reference, beneficiary_reference, sender_kyc_id, beneficiary_kyc_id, funding_reference, payout_method, purpose_code, source_of_funds, aml_screen_id, fraud_decision, payout_state(fraud_decision, aml_review), human_approval)
		self.transfers[transfer_id] = transfer
		self._audit(tenant_id, "remittance_transfer_created", transfer_id)
		return transfer.to_dict()

	def release_payout(self, transfer_id: str, tenant_id: str, provider_receipt: str, settlement_reference: str) -> dict[str, Any]:
		transfer = self._tenant_transfer_or_none(transfer_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "release_payout", "transfer_present": transfer is not None, "provider_receipt_present": bool(provider_receipt), "settlement_reference_present": bool(settlement_reference)})
		assert transfer is not None
		transfer.status = "paid"
		transfer.provider_receipt = provider_receipt
		transfer.settlement_reference = settlement_reference
		self._audit(tenant_id, "remittance_payout_released", transfer_id)
		return transfer.to_dict()

	def file_refund(self, refund_id: str, tenant_id: str, transfer_id: str, reason: str, reviewer_id: str) -> dict[str, Any]:
		transfer = self._tenant_transfer_or_none(transfer_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "file_refund", "transfer_present": transfer is not None, "reason_present": bool(reason), "reviewer_present": bool(reviewer_id)})
		if refund_id in self.refunds:
			raise ValueError(f"remittance refund already exists: {refund_id}")
		refund = RemittanceRefund(refund_id, tenant_id, transfer_id, reason, reviewer_id)
		self.refunds[refund_id] = refund
		if transfer is not None:
			transfer.status = "refund_filed"
		self._audit(tenant_id, "remittance_refund_filed", refund_id)
		return refund.to_dict()

	def register_remittance_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_remittance_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "remittance_agent_registered", agent_id)
		return evidence

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "remittance_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.remittance.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		quotes = [item for item in self.quotes.values() if item.tenant_id == tenant_id]
		transfers = [item for item in self.transfers.values() if item.tenant_id == tenant_id]
		refunds = [item for item in self.refunds.values() if item.tenant_id == tenant_id]
		return {"tenant_id": tenant_id, "quote_count": len(quotes), "transfer_count": len(transfers), "paid_count": sum(1 for item in transfers if item.status == "paid"), "review_required_count": sum(1 for item in transfers if item.status == "review_required"), "refund_count": len(refunds), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def list_transfers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		transfers = self.transfers.values()
		if tenant_id is not None:
			transfers = [transfer for transfer in transfers if transfer.tenant_id == tenant_id]
		return [transfer.to_dict() for transfer in sorted(transfers, key=lambda item: item.id)]

	def list_quotes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		quotes = self.quotes.values()
		if tenant_id is not None:
			quotes = [quote for quote in quotes if quote.tenant_id == tenant_id]
		return [quote.to_dict() for quote in sorted(quotes, key=lambda item: item.id)]

	def _tenant_quote_or_none(self, quote_id: str, tenant_id: str) -> RemittanceQuote | None:
		quote = self.quotes.get(quote_id)
		if quote is None or quote.tenant_id != tenant_id:
			return None
		return quote

	def _tenant_transfer_or_none(self, transfer_id: str, tenant_id: str) -> RemittanceTransfer | None:
		transfer = self.transfers.get(transfer_id)
		if transfer is None or transfer.tenant_id != tenant_id:
			return None
		return transfer

	def _record_evidence(self, evidence_id: str, tenant_id: str, kind: str, reference_id: str, status: str, metadata: dict[str, Any]) -> dict[str, Any]:
		evidence = RemittanceEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "remittance_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "remittance_policy_denied")


CrossBorderRemittanceService = RemittanceService
