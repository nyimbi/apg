"""Executable service layer for APG Digital Wallets."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_INSTRUMENT_TYPES,
		SUPPORTED_WALLET_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import Wallet, WalletEvidence, WalletInstrument, WalletLedgerEntry, money
	from .wallets_runtime import exceeds_limit, normalize_amount, normalize_code
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_INSTRUMENT_TYPES,
		SUPPORTED_WALLET_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import Wallet, WalletEvidence, WalletInstrument, WalletLedgerEntry, money  # type: ignore
	from wallets_runtime import exceeds_limit, normalize_amount, normalize_code  # type: ignore


class DigitalWalletsService:
	"""Dependency-light wallet lifecycle runtime for generated applications."""

	def __init__(self) -> None:
		self.wallets: dict[str, Wallet] = {}
		self.instruments: dict[str, WalletInstrument] = {}
		self.ledger: dict[str, WalletLedgerEntry] = {}
		self.evidence: dict[str, WalletEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def open_wallet(self, wallet_id: str, tenant_id: str, owner_reference: str, wallet_type: str, currency: str, initial_balance: Decimal | int | str = 0, metadata: dict[str, Any] | None = None, policy_attached: bool = True) -> dict[str, Any]:
		wallet_type = normalize_code(wallet_type)
		currency = str(currency).upper()
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "open_wallet", "owner_present": bool(owner_reference), "wallet_type_supported": wallet_type in SUPPORTED_WALLET_TYPES, "currency_supported": currency in SUPPORTED_CURRENCIES})
		if wallet_id in self.wallets:
			raise ValueError(f"wallet already exists: {wallet_id}")
		wallet = Wallet(wallet_id, tenant_id, owner_reference, wallet_type, currency, normalize_amount(initial_balance), metadata=dict(metadata or {}))
		if wallet.balance < 0:
			raise PermissionError("negative_balance_blocked")
		self.wallets[wallet_id] = wallet
		self._audit(tenant_id, "wallet_opened", wallet_id)
		return wallet.to_dict()

	def register_instrument(self, instrument_id: str, tenant_id: str, wallet_id: str, instrument_type: str, token_reference: str, verified_by: str) -> dict[str, Any]:
		wallet = self.wallets.get(wallet_id)
		instrument_type = normalize_code(instrument_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_instrument", "wallet_present": wallet is not None and wallet.tenant_id == tenant_id, "instrument_type_supported": instrument_type in SUPPORTED_INSTRUMENT_TYPES, "token_reference_present": bool(token_reference), "verified": bool(verified_by)})
		instrument = WalletInstrument(instrument_id, tenant_id, wallet_id, instrument_type, token_reference, verified_by)
		self.instruments[instrument_id] = instrument
		self._audit(tenant_id, "wallet_instrument_registered", instrument_id)
		return instrument.to_dict()

	def credit_wallet(self, entry_id: str, tenant_id: str, wallet_id: str, amount: Decimal | int | str, description: str, idempotency_key: str) -> dict[str, Any]:
		wallet = self._tenant_wallet(wallet_id, tenant_id)
		amount_decimal = normalize_amount(amount)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "credit_wallet", "amount": amount_decimal})
		wallet.balance += amount_decimal
		entry = self._record_ledger(entry_id, tenant_id, wallet_id, "credit", amount_decimal, wallet.currency, description, idempotency_key)
		self._audit(tenant_id, "wallet_credited", wallet_id)
		return entry

	def debit_wallet(self, entry_id: str, tenant_id: str, wallet_id: str, amount: Decimal | int | str, description: str, idempotency_key: str) -> dict[str, Any]:
		wallet = self._tenant_wallet(wallet_id, tenant_id)
		amount_decimal = normalize_amount(amount)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "debit_wallet", "amount": amount_decimal, "insufficient_available_balance": wallet.available_balance < amount_decimal})
		wallet.balance -= amount_decimal
		entry = self._record_ledger(entry_id, tenant_id, wallet_id, "debit", amount_decimal, wallet.currency, description, idempotency_key)
		self._audit(tenant_id, "wallet_debited", wallet_id)
		return entry

	def transfer(self, transfer_id: str, tenant_id: str, source_wallet_id: str, target_wallet_id: str, amount: Decimal | int | str, review_id: str = "") -> dict[str, Any]:
		source = self._tenant_wallet(source_wallet_id, tenant_id)
		target = self._tenant_wallet(target_wallet_id, tenant_id)
		amount_decimal = normalize_amount(amount)
		limit = Decimal(str(get_capability_contract(tenant_id)["configuration"]["limits"]["single_transfer_limit_minor"])) / Decimal("100")
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "transfer", "same_wallet": source_wallet_id == target_wallet_id, "currency_mismatch": source.currency != target.currency, "limit_exceeded": exceeds_limit(amount_decimal, limit), "review_recorded": bool(review_id)})
		self.debit_wallet(f"{transfer_id}-debit", tenant_id, source.id, amount_decimal, "wallet transfer debit", transfer_id)
		self.credit_wallet(f"{transfer_id}-credit", tenant_id, target.id, amount_decimal, "wallet transfer credit", transfer_id)
		evidence = self._record_evidence(transfer_id, tenant_id, "transfer", source.id, "posted", {"target_wallet_id": target.id, "amount": money(amount_decimal), "review_id": review_id})
		self._audit(tenant_id, "wallet_transfer_posted", transfer_id)
		return evidence

	def place_hold(self, hold_id: str, tenant_id: str, wallet_id: str, amount: Decimal | int | str, reason: str) -> dict[str, Any]:
		wallet = self._tenant_wallet(wallet_id, tenant_id)
		amount_decimal = normalize_amount(amount)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "place_hold", "amount": amount_decimal, "insufficient_available_balance": wallet.available_balance < amount_decimal})
		wallet.held_balance += amount_decimal
		evidence = self._record_evidence(hold_id, tenant_id, "hold", wallet_id, "placed", {"amount": money(amount_decimal), "reason": reason})
		self._audit(tenant_id, "wallet_hold_placed", wallet_id)
		return evidence

	def release_hold(self, hold_id: str, tenant_id: str, wallet_id: str, amount: Decimal | int | str, reason: str) -> dict[str, Any]:
		wallet = self._tenant_wallet(wallet_id, tenant_id)
		amount_decimal = normalize_amount(amount)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "release_hold", "amount": amount_decimal, "release_exceeds_held_balance": amount_decimal > wallet.held_balance})
		wallet.held_balance -= amount_decimal
		evidence = self._record_evidence(hold_id, tenant_id, "hold", wallet_id, "released", {"amount": money(amount_decimal), "reason": reason})
		self._audit(tenant_id, "wallet_hold_released", wallet_id)
		return evidence

	def register_wallet_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_wallet_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "wallet_agent_registered", agent_id)
		return evidence

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "wallet_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.wallets.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		wallets = [wallet for wallet in self.wallets.values() if wallet.tenant_id == tenant_id]
		return {"tenant_id": tenant_id, "wallet_count": len(wallets), "instrument_count": sum(1 for item in self.instruments.values() if item.tenant_id == tenant_id), "ledger_entry_count": sum(1 for item in self.ledger.values() if item.tenant_id == tenant_id), "total_balance": money(sum((wallet.balance for wallet in wallets), Decimal("0"))), "total_available": money(sum((wallet.available_balance for wallet in wallets), Decimal("0"))), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def list_wallets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		wallets = self.wallets.values()
		if tenant_id is not None:
			wallets = [wallet for wallet in wallets if wallet.tenant_id == tenant_id]
		return [wallet.to_dict() for wallet in sorted(wallets, key=lambda item: item.id)]

	def list_ledger(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		entries = self.ledger.values()
		if tenant_id is not None:
			entries = [entry for entry in entries if entry.tenant_id == tenant_id]
		return [entry.to_dict() for entry in sorted(entries, key=lambda item: item.id)]

	def _tenant_wallet(self, wallet_id: str, tenant_id: str) -> Wallet:
		wallet = self.wallets.get(wallet_id)
		if wallet is None or wallet.tenant_id != tenant_id:
			raise KeyError(f"unknown wallet: {wallet_id}")
		return wallet

	def _record_ledger(self, entry_id: str, tenant_id: str, wallet_id: str, entry_type: str, amount: Decimal, currency: str, description: str, idempotency_key: str) -> dict[str, Any]:
		if not idempotency_key:
			raise PermissionError("idempotency_key_required")
		entry = WalletLedgerEntry(entry_id, tenant_id, wallet_id, entry_type, amount, currency, description, idempotency_key)
		self.ledger[entry_id] = entry
		return entry.to_dict()

	def _record_evidence(self, evidence_id: str, tenant_id: str, kind: str, reference_id: str, status: str, metadata: dict[str, Any]) -> dict[str, Any]:
		evidence = WalletEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "wallet_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "wallet_policy_denied")


FintechWalletsService = DigitalWalletsService
