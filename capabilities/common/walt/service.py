"""Service layer for the Wallet and Payment Core capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .wallet_runtime import (
	PaymentInstrumentRecord,
	ReconciliationRecord,
	SettlementBatchRecord,
	TransactionRecord,
	WalletAuditEventRecord,
	WalletRecord,
	money_to_minor_units,
	normalize_currency,
	normalize_instrument_type,
	rule_required_actions,
	stable_id,
	utc_now,
)


class WaltService:
	"""Deterministic wallet, payment, settlement, and reconciliation service."""

	def __init__(self) -> None:
		self.wallets: dict[str, WalletRecord] = {}
		self.instruments: dict[str, PaymentInstrumentRecord] = {}
		self.transactions: dict[str, TransactionRecord] = {}
		self.settlement_batches: dict[str, SettlementBatchRecord] = {}
		self.reconciliations: dict[str, ReconciliationRecord] = {}
		self.audit_events: dict[str, WalletAuditEventRecord] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_wallet(
		self,
		tenant_id: str,
		owner_ref: str,
		currency: str,
		ledger_ref: str,
		compliance_policy_ref: str,
		initial_balance: int | float | str = 0,
		actor: str = "system",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		currency_code = normalize_currency(currency)
		context = {
			"tenant_context_present": True,
			"operation": "create_wallet",
			"wallet_owner_assigned": bool(str(owner_ref or "").strip()),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		if not str(ledger_ref or "").strip():
			raise PermissionError("ledger_integrity_required")
		if not str(compliance_policy_ref or "").strip():
			raise PermissionError("compliance_policy_required")
		balance_minor = money_to_minor_units(initial_balance)
		if balance_minor < 0:
			raise PermissionError("negative_balance_blocked")
		record = WalletRecord(
			id=stable_id("walt_wallet", tenant_id, owner_ref, currency_code, ledger_ref),
			tenant_id=tenant_id,
			owner_ref=owner_ref,
			currency=currency_code,
			ledger_ref=ledger_ref,
			compliance_policy_ref=compliance_policy_ref,
			balance_minor=balance_minor,
		)
		self.wallets[record.id] = record
		self._record_event(tenant_id, "wallet_created", record.id, f"Wallet created for {owner_ref}", actor)
		return record.to_dict()

	def register_instrument(
		self,
		tenant_id: str,
		wallet_id: str,
		instrument_ref: str,
		instrument_type: str,
		token_ref: str,
		encrypted: bool,
		verified_by: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		wallet = self._get_wallet(tenant_id, wallet_id)
		context = {
			"tenant_context_present": True,
			"payment_instrument_present": bool(str(instrument_ref or "").strip()),
			"instrument_encrypted": bool(encrypted),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		if not str(token_ref or "").strip():
			raise PermissionError("instrument_tokenization_required")
		if not str(verified_by or "").strip():
			raise PermissionError("instrument_verification_required")
		record = PaymentInstrumentRecord(
			id=stable_id("walt_instrument", tenant_id, wallet.id, instrument_ref),
			tenant_id=tenant_id,
			wallet_id=wallet.id,
			instrument_ref=instrument_ref,
			instrument_type=normalize_instrument_type(instrument_type),
			token_ref=token_ref,
			encrypted=bool(encrypted),
			verified_by=verified_by,
		)
		self.instruments[record.id] = record
		self._record_event(tenant_id, "instrument_registered", record.id, "Payment instrument registered", verified_by)
		return record.to_dict()

	def authorize_transaction(
		self,
		tenant_id: str,
		wallet_id: str,
		instrument_id: str,
		amount: int | float | str,
		currency: str,
		direction: str = "debit",
		mfa_completed: bool = False,
		risk_score: float = 0.0,
		risk_review_recorded: bool = False,
		idempotency_key: str = "",
		actor: str = "system",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		wallet = self._get_wallet(tenant_id, wallet_id)
		instrument = self._get_instrument(tenant_id, instrument_id)
		if instrument.wallet_id != wallet.id:
			raise PermissionError("instrument_wallet_mismatch")
		direction_value = str(direction or "debit").strip().lower()
		if direction_value not in {"debit", "credit"}:
			raise ValueError(f"unsupported_transaction_direction:{direction}")
		currency_code = normalize_currency(currency)
		if currency_code != wallet.currency:
			raise PermissionError("wallet_currency_mismatch")
		amount_minor = money_to_minor_units(amount)
		if amount_minor <= 0:
			raise ValueError("transaction_amount_must_be_positive")
		context = {
			"tenant_context_present": True,
			"operation": "authorize_transaction",
			"payment_instrument_present": True,
			"instrument_encrypted": bool(instrument.encrypted),
			"transaction_amount": amount_minor / 100,
			"mfa_completed": bool(mfa_completed),
			"transaction_risk_score": float(risk_score),
			"risk_review_recorded": bool(risk_review_recorded),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		if result["decision"] == "allow" and direction_value == "debit" and self._available_balance(wallet) < amount_minor:
			raise PermissionError("insufficient_wallet_balance")
		status = "review_required" if result["decision"] == "require_review" else "authorized"
		if status == "authorized" and direction_value == "debit":
			wallet.hold_minor += amount_minor
			wallet.updated_at = utc_now()
		record = TransactionRecord(
			id=stable_id("walt_txn", tenant_id, wallet.id, instrument.id, idempotency_key or len(self.transactions)),
			tenant_id=tenant_id,
			wallet_id=wallet.id,
			instrument_id=instrument.id,
			direction=direction_value,
			amount_minor=amount_minor,
			currency=currency_code,
			status=status,
			risk_score=float(risk_score),
			mfa_completed=bool(mfa_completed),
			risk_review_recorded=bool(risk_review_recorded),
			idempotency_key=idempotency_key,
			required_actions=rule_required_actions(result),
			matched_rules=list(result["matched_rules"]),
		)
		self.transactions[record.id] = record
		self._record_event(tenant_id, "transaction_authorized", record.id, f"Transaction {status}", actor)
		return record.to_dict()

	def capture_transaction(self, tenant_id: str, transaction_id: str, actor: str = "system") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		transaction = self._get_transaction(tenant_id, transaction_id)
		wallet = self._get_wallet(tenant_id, transaction.wallet_id)
		if transaction.status != "authorized":
			raise PermissionError(f"transaction_not_authorized:{transaction.status}")
		if transaction.direction == "debit":
			if wallet.hold_minor < transaction.amount_minor:
				raise PermissionError("wallet_hold_missing")
			wallet.hold_minor -= transaction.amount_minor
			wallet.balance_minor -= transaction.amount_minor
		else:
			wallet.balance_minor += transaction.amount_minor
		wallet.updated_at = utc_now()
		transaction.status = "captured"
		transaction.captured_at = utc_now()
		self._record_event(tenant_id, "transaction_captured", transaction.id, "Transaction captured", actor)
		return transaction.to_dict()

	def create_settlement_batch(
		self,
		tenant_id: str,
		transaction_ids: list[str],
		settlement_account_ref: str,
		reconciliation_completed: bool,
		created_by: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		context = {
			"tenant_context_present": True,
			"operation": "settle_batch",
			"reconciliation_completed": bool(reconciliation_completed),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		if not transaction_ids:
			raise ValueError("settlement_transactions_required")
		if not str(settlement_account_ref or "").strip():
			raise PermissionError("settlement_account_required")
		transactions = [self._get_transaction(tenant_id, item) for item in transaction_ids]
		if any(transaction.status != "captured" for transaction in transactions):
			raise PermissionError("settlement_requires_captured_transactions")
		currencies = {transaction.currency for transaction in transactions}
		if len(currencies) != 1:
			raise PermissionError("settlement_currency_mismatch")
		total_minor = sum(transaction.amount_minor for transaction in transactions)
		record = SettlementBatchRecord(
			id=stable_id("walt_settlement", tenant_id, settlement_account_ref, len(self.settlement_batches)),
			tenant_id=tenant_id,
			transaction_ids=[transaction.id for transaction in transactions],
			settlement_account_ref=settlement_account_ref,
			total_minor=total_minor,
			currency=next(iter(currencies)),
			reconciliation_completed=bool(reconciliation_completed),
			created_by=created_by,
		)
		self.settlement_batches[record.id] = record
		for transaction in transactions:
			transaction.status = "settled"
			transaction.settled_at = utc_now()
		self._record_event(tenant_id, "settlement_batch_created", record.id, "Settlement batch created", created_by)
		return record.to_dict()

	def record_reconciliation(
		self,
		tenant_id: str,
		settlement_batch_id: str,
		reconciliation_ref: str,
		matched_count: int,
		exception_count: int,
		recorded_by: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		batch = self._get_settlement_batch(tenant_id, settlement_batch_id)
		if not str(reconciliation_ref or "").strip():
			raise PermissionError("reconciliation_evidence_required")
		if matched_count < 0 or exception_count < 0:
			raise ValueError("reconciliation_counts_must_be_non_negative")
		status = "exceptions" if exception_count else "matched"
		batch.status = "exception_review" if exception_count else "reconciled"
		record = ReconciliationRecord(
			id=stable_id("walt_reconciliation", tenant_id, batch.id, reconciliation_ref),
			tenant_id=tenant_id,
			settlement_batch_id=batch.id,
			reconciliation_ref=reconciliation_ref,
			matched_count=int(matched_count),
			exception_count=int(exception_count),
			status=status,
			recorded_by=recorded_by,
		)
		self.reconciliations[record.id] = record
		self._record_event(tenant_id, "reconciliation_recorded", record.id, f"Reconciliation {status}", recorded_by)
		return record.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		record = self.create_wallet(
			tenant_id=tenant_id,
			owner_ref=str(metadata.get("owner_ref") or record_id),
			currency=str(metadata.get("currency") or "USD"),
			ledger_ref=str(metadata.get("ledger_ref") or f"ledger://{record_id}"),
			compliance_policy_ref=str(metadata.get("compliance_policy_ref") or "policy://compatibility"),
			initial_balance=metadata.get("initial_balance", 0),
			actor=str(metadata.get("actor") or "compatibility"),
		)
		if status != "active":
			wallet = self._get_wallet(tenant_id, record["id"])
			wallet.status = status
			wallet.updated_at = utc_now()
			record = wallet.to_dict()
		return record

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_wallets(tenant_id)

	def list_wallets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.wallets, tenant_id)

	def list_instruments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.instruments, tenant_id)

	def list_transactions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.transactions, tenant_id)

	def list_settlement_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.settlement_batches, tenant_id)

	def list_reconciliations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.reconciliations, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		wallets = self.list_wallets(tenant_id)
		transactions = self.list_transactions(tenant_id)
		return {
			"tenant_id": tenant_id,
			"wallet_count": len(wallets),
			"instrument_count": len(self.list_instruments(tenant_id)),
			"transaction_count": len(transactions),
			"authorized_transaction_count": sum(1 for item in transactions if item["status"] == "authorized"),
			"review_required_transaction_count": sum(1 for item in transactions if item["status"] == "review_required"),
			"captured_transaction_count": sum(1 for item in transactions if item["status"] == "captured"),
			"settlement_batch_count": len(self.list_settlement_batches(tenant_id)),
			"reconciliation_count": len(self.list_reconciliations(tenant_id)),
			"total_balance": round(sum(item["balance"] for item in wallets), 2),
			"total_holds": round(sum(item["hold"] for item in wallets), 2),
			"recent_events": self.list_audit_events(tenant_id)[-5:],
		}

	def _require_tenant(self, tenant_id: str) -> None:
		if not str(tenant_id or "").strip():
			self._raise_policy(self.evaluate({"tenant_context_present": False}))

	def _raise_policy(self, result: dict[str, Any]) -> None:
		reasons = ", ".join(action.get("reason", "wallet_policy_blocked") for action in result["actions"])
		raise PermissionError(reasons or "wallet_policy_blocked")

	def _available_balance(self, wallet: WalletRecord) -> int:
		return wallet.balance_minor - wallet.hold_minor

	def _get_wallet(self, tenant_id: str, wallet_id: str) -> WalletRecord:
		wallet = self.wallets.get(wallet_id)
		if wallet is None:
			wallet = next((item for item in self.wallets.values() if item.tenant_id == tenant_id and item.owner_ref == wallet_id), None)
		if wallet is None or wallet.tenant_id != tenant_id:
			raise KeyError(f"wallet_not_found:{wallet_id}")
		return wallet

	def _get_instrument(self, tenant_id: str, instrument_id: str) -> PaymentInstrumentRecord:
		instrument = self.instruments.get(instrument_id)
		if instrument is None:
			instrument = next((item for item in self.instruments.values() if item.tenant_id == tenant_id and item.instrument_ref == instrument_id), None)
		if instrument is None or instrument.tenant_id != tenant_id:
			raise KeyError(f"instrument_not_found:{instrument_id}")
		return instrument

	def _get_transaction(self, tenant_id: str, transaction_id: str) -> TransactionRecord:
		transaction = self.transactions.get(transaction_id)
		if transaction is None or transaction.tenant_id != tenant_id:
			raise KeyError(f"transaction_not_found:{transaction_id}")
		return transaction

	def _get_settlement_batch(self, tenant_id: str, batch_id: str) -> SettlementBatchRecord:
		batch = self.settlement_batches.get(batch_id)
		if batch is None or batch.tenant_id != tenant_id:
			raise KeyError(f"settlement_batch_not_found:{batch_id}")
		return batch

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
		severity: str = "low",
	) -> dict[str, Any]:
		record = WalletAuditEventRecord(
			id=stable_id("walt_event", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])
