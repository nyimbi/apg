"""Executable service layer for APG Digital Wallets."""

from __future__ import annotations

import asyncio
import datetime
import secrets
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
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
except ImportError:  # pragma: no cover
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> datetime.datetime:
	return datetime.datetime.now(datetime.timezone.utc)


def _iso() -> str:
	return _utc_now().isoformat()


def _d(value: Any) -> Decimal:
	return Decimal(str(value))


# Spot exchange rates (approximate, for demo; in production fetch from rate service)
_SPOT_RATES: dict[str, dict[str, Decimal]] = {
	"KES": {"USD": _d("0.00773"), "EUR": _d("0.00712"), "UGX": _d("28.5"), "TZS": _d("19.8")},
	"USD": {"KES": _d("129.4"), "EUR": _d("0.921"), "UGX": _d("3690"), "TZS": _d("2560")},
	"EUR": {"KES": _d("140.5"), "USD": _d("1.086"), "UGX": _d("4010"), "TZS": _d("2780")},
	"UGX": {"KES": _d("0.035"), "USD": _d("0.000271"), "EUR": _d("0.000249"), "TZS": _d("0.694")},
	"TZS": {"KES": _d("0.0505"), "USD": _d("0.000390"), "EUR": _d("0.000360"), "UGX": _d("1.44")},
}


def _convert_amount(amount: Decimal, from_ccy: str, to_ccy: str) -> Decimal:
	if from_ccy == to_ccy:
		return amount
	rates = _SPOT_RATES.get(from_ccy, {})
	rate = rates.get(to_ccy)
	if rate is None:
		raise ValueError(f"no spot rate for {from_ccy}/{to_ccy}")
	return (amount * rate).quantize(_d("0.01"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class DigitalWalletsService:
	"""Full-featured digital wallet lifecycle runtime for APG generated applications."""

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

		self.wallets: dict[str, Wallet] = {}
		self.instruments: dict[str, WalletInstrument] = {}
		self.ledger: dict[str, WalletLedgerEntry] = {}
		self.evidence: dict[str, WalletEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

		# Spending limits store: wallet_id -> {daily_limit, monthly_limit}
		self._limits: dict[str, dict[str, Decimal]] = {}

		# Running spend counters: wallet_id -> {daily, monthly}
		self._spend: dict[str, dict[str, Decimal]] = defaultdict(lambda: {"daily": _d("0"), "monthly": _d("0")})

		# Freeze store: wallet_id -> {frozen: bool, reason: str}
		self._freeze_state: dict[str, dict[str, Any]] = {}

		# Loyalty points store: customer_id -> points
		self._loyalty_points: dict[str, Decimal] = defaultdict(lambda: _d("0"))

		# Statement cache: wallet_id -> list of enriched ledger entries
		self._statement_cache: dict[str, list[dict[str, Any]]] = defaultdict(list)

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

	def open_wallet(
		self,
		wallet_id: str,
		tenant_id: str,
		owner_reference: str,
		wallet_type: str,
		currency: str,
		initial_balance: Decimal | int | str = 0,
		metadata: dict[str, Any] | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		wallet_type = normalize_code(wallet_type)
		currency = str(currency).upper()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "open_wallet",
			"owner_present": bool(owner_reference),
			"wallet_type_supported": wallet_type in SUPPORTED_WALLET_TYPES,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
		})
		if wallet_id in self.wallets:
			raise ValueError(f"wallet already exists: {wallet_id}")
		wallet = Wallet(wallet_id, tenant_id, owner_reference, wallet_type, currency, normalize_amount(initial_balance), metadata=dict(metadata or {}))
		if wallet.balance < 0:
			raise PermissionError("negative_balance_blocked")
		self.wallets[wallet_id] = wallet
		self._audit(tenant_id, "wallet_opened", wallet_id)
		return wallet.to_dict()

	def register_instrument(
		self,
		instrument_id: str,
		tenant_id: str,
		wallet_id: str,
		instrument_type: str,
		token_reference: str,
		verified_by: str,
	) -> dict[str, Any]:
		wallet = self.wallets.get(wallet_id)
		instrument_type = normalize_code(instrument_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_instrument",
			"wallet_present": wallet is not None and wallet.tenant_id == tenant_id,
			"instrument_type_supported": instrument_type in SUPPORTED_INSTRUMENT_TYPES,
			"token_reference_present": bool(token_reference),
			"verified": bool(verified_by),
		})
		instrument = WalletInstrument(instrument_id, tenant_id, wallet_id, instrument_type, token_reference, verified_by)
		self.instruments[instrument_id] = instrument
		self._audit(tenant_id, "wallet_instrument_registered", instrument_id)
		return instrument.to_dict()

	def credit_wallet(
		self,
		entry_id: str,
		tenant_id: str,
		wallet_id: str,
		amount: Decimal | int | str,
		description: str,
		idempotency_key: str,
	) -> dict[str, Any]:
		wallet = self._tenant_wallet(wallet_id, tenant_id)
		amount_decimal = normalize_amount(amount)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "credit_wallet",
			"amount": amount_decimal,
		})
		wallet.balance += amount_decimal
		entry = self._record_ledger(entry_id, tenant_id, wallet_id, "credit", amount_decimal, wallet.currency, description, idempotency_key)
		self._audit(tenant_id, "wallet_credited", wallet_id)
		return entry

	def debit_wallet(
		self,
		entry_id: str,
		tenant_id: str,
		wallet_id: str,
		amount: Decimal | int | str,
		description: str,
		idempotency_key: str,
	) -> dict[str, Any]:
		wallet = self._tenant_wallet(wallet_id, tenant_id)
		amount_decimal = normalize_amount(amount)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "debit_wallet",
			"amount": amount_decimal,
			"insufficient_available_balance": wallet.available_balance < amount_decimal,
		})
		wallet.balance -= amount_decimal
		entry = self._record_ledger(entry_id, tenant_id, wallet_id, "debit", amount_decimal, wallet.currency, description, idempotency_key)
		self._audit(tenant_id, "wallet_debited", wallet_id)
		return entry

	def transfer(
		self,
		transfer_id: str,
		tenant_id: str,
		source_wallet_id: str,
		target_wallet_id: str,
		amount: Decimal | int | str,
		review_id: str = "",
	) -> dict[str, Any]:
		source = self._tenant_wallet(source_wallet_id, tenant_id)
		target = self._tenant_wallet(target_wallet_id, tenant_id)
		amount_decimal = normalize_amount(amount)
		limit = Decimal(str(get_capability_contract(tenant_id)["configuration"]["limits"]["single_transfer_limit_minor"])) / Decimal("100")
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "transfer",
			"same_wallet": source_wallet_id == target_wallet_id,
			"currency_mismatch": source.currency != target.currency,
			"limit_exceeded": exceeds_limit(amount_decimal, limit),
			"review_recorded": bool(review_id),
		})
		self.debit_wallet(f"{transfer_id}-debit", tenant_id, source.id, amount_decimal, "wallet transfer debit", transfer_id)
		self.credit_wallet(f"{transfer_id}-credit", tenant_id, target.id, amount_decimal, "wallet transfer credit", transfer_id)
		evidence = self._record_evidence(transfer_id, tenant_id, "transfer", source.id, "posted", {"target_wallet_id": target.id, "amount": money(amount_decimal), "review_id": review_id})
		self._audit(tenant_id, "wallet_transfer_posted", transfer_id)
		return evidence

	def place_hold(
		self,
		hold_id: str,
		tenant_id: str,
		wallet_id: str,
		amount: Decimal | int | str,
		reason: str,
	) -> dict[str, Any]:
		wallet = self._tenant_wallet(wallet_id, tenant_id)
		amount_decimal = normalize_amount(amount)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "place_hold",
			"amount": amount_decimal,
			"insufficient_available_balance": wallet.available_balance < amount_decimal,
		})
		wallet.held_balance += amount_decimal
		evidence = self._record_evidence(hold_id, tenant_id, "hold", wallet_id, "placed", {"amount": money(amount_decimal), "reason": reason})
		self._audit(tenant_id, "wallet_hold_placed", wallet_id)
		return evidence

	def release_hold(
		self,
		hold_id: str,
		tenant_id: str,
		wallet_id: str,
		amount: Decimal | int | str,
		reason: str,
	) -> dict[str, Any]:
		wallet = self._tenant_wallet(wallet_id, tenant_id)
		amount_decimal = normalize_amount(amount)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "release_hold",
			"amount": amount_decimal,
			"release_exceeds_held_balance": amount_decimal > wallet.held_balance,
		})
		wallet.held_balance -= amount_decimal
		evidence = self._record_evidence(hold_id, tenant_id, "hold", wallet_id, "released", {"amount": money(amount_decimal), "reason": reason})
		self._audit(tenant_id, "wallet_hold_released", wallet_id)
		return evidence

	def register_wallet_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_wallet_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "wallet_agent_registered", agent_id)
		return evidence

	def validate_batch(
		self,
		tenant_id: str,
		item_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "wallet_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.wallets.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		wallets = [w for w in self.wallets.values() if w.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"wallet_count": len(wallets),
			"instrument_count": sum(1 for item in self.instruments.values() if item.tenant_id == tenant_id),
			"ledger_entry_count": sum(1 for item in self.ledger.values() if item.tenant_id == tenant_id),
			"total_balance": money(sum((w.balance for w in wallets), _d("0"))),
			"total_available": money(sum((w.available_balance for w in wallets), _d("0"))),
			"audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	def list_wallets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		wallets = self.wallets.values()
		if tenant_id is not None:
			wallets = [w for w in wallets if w.tenant_id == tenant_id]
		return [w.to_dict() for w in sorted(wallets, key=lambda x: x.id)]

	def list_ledger(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		entries = self.ledger.values()
		if tenant_id is not None:
			entries = [e for e in entries if e.tenant_id == tenant_id]
		return [e.to_dict() for e in sorted(entries, key=lambda x: x.id)]

	# ------------------------------------------------------------------
	# New async methods
	# ------------------------------------------------------------------

	async def create_wallet(
		self,
		customer_id: str,
		currency: str,
		wallet_type: str = "consumer",
	) -> dict[str, Any]:
		"""Create a wallet for a customer with sensible defaults."""
		assert customer_id, "customer_id required"
		assert currency, "currency required"
		await asyncio.sleep(0)

		currency = str(currency).upper()
		wallet_type = normalize_code(wallet_type)
		wallet_id = f"w-{customer_id}-{currency.lower()}-{secrets.token_hex(3)}"

		result = self.open_wallet(
			wallet_id=wallet_id,
			tenant_id=self.tenant_id,
			owner_reference=customer_id,
			wallet_type=wallet_type,
			currency=currency,
			initial_balance=_d("0"),
			metadata={"created_via": "create_wallet", "customer_id": customer_id},
		)
		# Set default spending limits
		self._limits[wallet_id] = {
			"daily_limit": _d("100000"),
			"monthly_limit": _d("500000"),
		}
		return result | {"daily_limit": "100000", "monthly_limit": "500000"}

	async def top_up_wallet(
		self,
		wallet_id: str,
		amount: float | Decimal | str,
		source: str,
	) -> dict[str, Any]:
		"""Top up a wallet from an external source (bank, card, mobile money)."""
		assert wallet_id, "wallet_id required"
		assert source, "source required"
		amount_d = normalize_amount(amount)
		assert amount_d > 0, "amount must be positive"
		await asyncio.sleep(0)

		freeze = self._freeze_state.get(wallet_id, {})
		if freeze.get("frozen"):
			raise PermissionError(f"wallet is frozen: {freeze.get('reason', 'unknown')}")

		idempotency_key = f"topup-{wallet_id}-{source}-{secrets.token_hex(4)}"
		entry = self.credit_wallet(
			entry_id=idempotency_key,
			tenant_id=self.tenant_id,
			wallet_id=wallet_id,
			amount=amount_d,
			description=f"top_up from {source}",
			idempotency_key=idempotency_key,
		)
		self._statement_cache[wallet_id].append({
			**entry,
			"direction": "credit",
			"source": source,
			"topped_up_at": _iso(),
		})
		return entry | {"source": source, "topped_up_at": _iso()}

	async def wallet_to_wallet_transfer(
		self,
		from_wallet: str,
		to_wallet: str,
		amount: float | Decimal | str,
	) -> dict[str, Any]:
		"""Transfer funds between two wallets with spend-limit enforcement."""
		assert from_wallet, "from_wallet required"
		assert to_wallet, "to_wallet required"
		assert from_wallet != to_wallet, "source and target wallets must differ"
		amount_d = normalize_amount(amount)
		assert amount_d > 0, "amount must be positive"
		await asyncio.sleep(0)

		# Check freeze
		for wid in (from_wallet, to_wallet):
			freeze = self._freeze_state.get(wid, {})
			if freeze.get("frozen"):
				raise PermissionError(f"wallet {wid} is frozen: {freeze.get('reason')}")

		# Enforce daily spending limit on source
		limits = self._limits.get(from_wallet, {})
		daily_limit = limits.get("daily_limit", _d("1000000"))
		daily_spent = self._spend[from_wallet]["daily"]
		if daily_spent + amount_d > daily_limit:
			raise PermissionError(f"daily transfer limit exceeded for wallet {from_wallet}")

		transfer_id = f"ww-{from_wallet[:8]}-{to_wallet[:8]}-{secrets.token_hex(4)}"
		result = self.transfer(
			transfer_id=transfer_id,
			tenant_id=self.tenant_id,
			source_wallet_id=from_wallet,
			target_wallet_id=to_wallet,
			amount=amount_d,
		)
		self._spend[from_wallet]["daily"] += amount_d
		self._spend[from_wallet]["monthly"] += amount_d
		return result | {"transfer_id": transfer_id, "transferred_at": _iso()}

	async def withdraw_from_wallet(
		self,
		wallet_id: str,
		amount: float | Decimal | str,
		destination: str,
	) -> dict[str, Any]:
		"""Withdraw funds from a wallet to an external destination."""
		assert wallet_id, "wallet_id required"
		assert destination, "destination required"
		amount_d = normalize_amount(amount)
		assert amount_d > 0, "amount must be positive"
		await asyncio.sleep(0)

		freeze = self._freeze_state.get(wallet_id, {})
		if freeze.get("frozen"):
			raise PermissionError(f"wallet is frozen: {freeze.get('reason')}")

		wallet = self._tenant_wallet(wallet_id, self.tenant_id)
		if wallet.available_balance < amount_d:
			raise PermissionError("insufficient_available_balance")

		idempotency_key = f"withdraw-{wallet_id}-{secrets.token_hex(4)}"
		entry = self.debit_wallet(
			entry_id=idempotency_key,
			tenant_id=self.tenant_id,
			wallet_id=wallet_id,
			amount=amount_d,
			description=f"withdrawal to {destination}",
			idempotency_key=idempotency_key,
		)
		self._spend[wallet_id]["daily"] += amount_d
		self._spend[wallet_id]["monthly"] += amount_d
		self._statement_cache[wallet_id].append({
			**entry,
			"direction": "debit",
			"destination": destination,
			"withdrawn_at": _iso(),
		})
		return entry | {"destination": destination, "withdrawn_at": _iso()}

	async def wallet_balance(self, wallet_id: str) -> dict[str, Any]:
		"""Return current balance, available balance, and held amount."""
		assert wallet_id, "wallet_id required"
		await asyncio.sleep(0)

		wallet = self._tenant_wallet(wallet_id, self.tenant_id)
		limits = self._limits.get(wallet_id, {})
		freeze = self._freeze_state.get(wallet_id, {})
		spent = self._spend.get(wallet_id, {"daily": _d("0"), "monthly": _d("0")})

		return {
			"wallet_id": wallet_id,
			"currency": wallet.currency,
			"balance": money(wallet.balance),
			"available_balance": money(wallet.available_balance),
			"held_balance": money(wallet.held_balance),
			"daily_spent": money(spent["daily"]),
			"monthly_spent": money(spent["monthly"]),
			"daily_limit": money(limits.get("daily_limit", _d("0"))),
			"monthly_limit": money(limits.get("monthly_limit", _d("0"))),
			"frozen": freeze.get("frozen", False),
			"fetched_at": _iso(),
		}

	async def transaction_history(
		self,
		wallet_id: str,
		limit: int = 50,
	) -> dict[str, Any]:
		"""Return the most recent transactions for a wallet."""
		assert wallet_id, "wallet_id required"
		assert limit > 0, "limit must be positive"
		await asyncio.sleep(0)

		# Pull from ledger
		entries = [
			e.to_dict() for e in self.ledger.values()
			if e.wallet_id == wallet_id
		]
		# Sort descending by entry_id (UUID7 ordering approximation)
		entries.sort(key=lambda x: x.get("id", ""), reverse=True)
		page = entries[:limit]

		return {
			"wallet_id": wallet_id,
			"total_entries": len(entries),
			"returned": len(page),
			"limit": limit,
			"transactions": page,
			"fetched_at": _iso(),
		}

	async def freeze_wallet(self, wallet_id: str, reason: str) -> dict[str, Any]:
		"""Freeze a wallet, blocking all debits and transfers."""
		assert wallet_id, "wallet_id required"
		assert reason, "reason required"
		await asyncio.sleep(0)

		if self._freeze_state.get(wallet_id, {}).get("frozen"):
			return {"wallet_id": wallet_id, "frozen": True, "message": "already_frozen"}

		self._freeze_state[wallet_id] = {"frozen": True, "reason": reason, "frozen_at": _iso()}
		# Place a hold for the full available balance to prevent race conditions
		wallet = self.wallets.get(wallet_id)
		if wallet and wallet.available_balance > 0:
			try:
				self.place_hold(f"freeze-hold-{wallet_id}", self.tenant_id, wallet_id, wallet.available_balance, reason)
			except Exception:
				pass

		self._audit(self.tenant_id, "wallet_frozen", wallet_id)
		return {"wallet_id": wallet_id, "frozen": True, "reason": reason, "frozen_at": _iso()}

	async def unfreeze_wallet(self, wallet_id: str, approved_by: str) -> dict[str, Any]:
		"""Unfreeze a previously frozen wallet."""
		assert wallet_id, "wallet_id required"
		assert approved_by, "approved_by required"
		await asyncio.sleep(0)

		if not self._freeze_state.get(wallet_id, {}).get("frozen"):
			raise ValueError(f"wallet is not frozen: {wallet_id}")

		self._freeze_state[wallet_id] = {"frozen": False, "unfrozen_at": _iso(), "approved_by": approved_by}
		self._audit(self.tenant_id, "wallet_unfrozen", wallet_id)
		return {"wallet_id": wallet_id, "frozen": False, "approved_by": approved_by, "unfrozen_at": _iso()}

	async def set_spending_limits(
		self,
		wallet_id: str,
		daily_limit: float | Decimal | str,
		monthly_limit: float | Decimal | str,
	) -> dict[str, Any]:
		"""Set daily and monthly spend limits for a wallet."""
		assert wallet_id, "wallet_id required"
		daily_d = normalize_amount(daily_limit)
		monthly_d = normalize_amount(monthly_limit)
		assert daily_d >= 0, "daily_limit must be non-negative"
		assert monthly_d >= daily_d, "monthly_limit must be >= daily_limit"
		await asyncio.sleep(0)

		self._limits[wallet_id] = {"daily_limit": daily_d, "monthly_limit": monthly_d}
		self._audit(self.tenant_id, "spending_limits_set", wallet_id)
		return {
			"wallet_id": wallet_id,
			"daily_limit": money(daily_d),
			"monthly_limit": money(monthly_d),
			"updated_at": _iso(),
		}

	async def wallet_statement(
		self,
		wallet_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Generate a formatted wallet statement for a period."""
		assert wallet_id, "wallet_id required"
		assert period, "period required"
		await asyncio.sleep(0)

		wallet = self._tenant_wallet(wallet_id, self.tenant_id)
		entries = [
			e.to_dict() for e in self.ledger.values()
			if e.wallet_id == wallet_id
		]
		credits = [e for e in entries if e.get("entry_type") == "credit"]
		debits = [e for e in entries if e.get("entry_type") == "debit"]
		total_credits = sum(_d(e.get("amount", 0)) for e in credits)
		total_debits = sum(_d(e.get("amount", 0)) for e in debits)

		self._audit(self.tenant_id, "wallet_statement_generated", wallet_id)
		return {
			"wallet_id": wallet_id,
			"period": period,
			"currency": wallet.currency,
			"opening_balance": money(wallet.balance + total_debits - total_credits),
			"closing_balance": money(wallet.balance),
			"total_credits": money(total_credits),
			"total_debits": money(total_debits),
			"net_movement": money(total_credits - total_debits),
			"transaction_count": len(entries),
			"credit_count": len(credits),
			"debit_count": len(debits),
			"entries": entries[-100:],  # cap at 100
			"generated_at": _iso(),
		}

	async def loyalty_wallet_credit(
		self,
		customer_id: str,
		points: int,
		reason: str,
	) -> dict[str, Any]:
		"""Credit loyalty points to a customer's loyalty wallet."""
		assert customer_id, "customer_id required"
		assert points > 0, "points must be positive"
		assert reason, "reason required"
		await asyncio.sleep(0)

		self._loyalty_points[customer_id] += _d(str(points))
		new_balance = self._loyalty_points[customer_id]

		# Find a loyalty wallet if it exists
		loyalty_wallets = [
			w for w in self.wallets.values()
			if w.owner_reference == customer_id and w.wallet_type == "loyalty"
		]
		wallet_id = loyalty_wallets[0].id if loyalty_wallets else None

		if wallet_id:
			# Credit 1 point = 0.01 KES equivalent
			kes_equiv = _d(str(points)) * _d("0.01")
			idempotency_key = f"loyalty-{customer_id}-{secrets.token_hex(4)}"
			self.credit_wallet(idempotency_key, self.tenant_id, wallet_id, kes_equiv, reason, idempotency_key)

		self._audit(self.tenant_id, "loyalty_credited", customer_id)
		return {
			"customer_id": customer_id,
			"points_credited": points,
			"reason": reason,
			"new_points_balance": int(new_balance),
			"loyalty_wallet_id": wallet_id,
			"credited_at": _iso(),
		}

	async def currency_conversion_in_wallet(
		self,
		wallet_id: str,
		from_ccy: str,
		to_ccy: str,
		amount: float | Decimal | str,
	) -> dict[str, Any]:
		"""Convert an amount within a wallet from one currency to another."""
		assert wallet_id, "wallet_id required"
		assert from_ccy and to_ccy, "both currencies required"
		amount_d = normalize_amount(amount)
		assert amount_d > 0, "amount must be positive"
		await asyncio.sleep(0)

		from_ccy = from_ccy.upper()
		to_ccy = to_ccy.upper()

		wallet = self._tenant_wallet(wallet_id, self.tenant_id)
		if wallet.currency != from_ccy:
			raise ValueError(f"wallet currency {wallet.currency} does not match from_ccy {from_ccy}")
		if wallet.available_balance < amount_d:
			raise PermissionError("insufficient_available_balance_for_conversion")

		converted = _convert_amount(amount_d, from_ccy, to_ccy)
		spread_pct = _d("0.015")  # 1.5% FX spread
		net_converted = (converted * (1 - spread_pct)).quantize(_d("0.01"))
		spread_amount = (converted - net_converted).quantize(_d("0.01"))

		# Debit source wallet
		idempotency_key = f"fx-{wallet_id}-{from_ccy}-{to_ccy}-{secrets.token_hex(4)}"
		self.debit_wallet(f"{idempotency_key}-debit", self.tenant_id, wallet_id, amount_d, f"FX conversion {from_ccy}->{to_ccy}", idempotency_key)

		self._audit(self.tenant_id, "currency_converted", wallet_id)
		return {
			"wallet_id": wallet_id,
			"from_currency": from_ccy,
			"to_currency": to_ccy,
			"source_amount": money(amount_d),
			"converted_amount": money(converted),
			"net_converted_amount": money(net_converted),
			"spread_amount": money(spread_amount),
			"spread_pct": float(spread_pct * 100),
			"rate_used": str(_convert_amount(_d("1"), from_ccy, to_ccy)),
			"converted_at": _iso(),
		}

	async def wallet_analytics(self, period: str) -> dict[str, Any]:
		"""Aggregate wallet-level analytics for a reporting period."""
		assert period, "period required"
		await asyncio.sleep(0)

		all_wallets = [w for w in self.wallets.values() if w.tenant_id == self.tenant_id]
		total_balance = sum(w.balance for w in all_wallets)
		total_available = sum(w.available_balance for w in all_wallets)
		frozen_count = sum(1 for wid in [w.id for w in all_wallets] if self._freeze_state.get(wid, {}).get("frozen"))

		by_type: dict[str, int] = defaultdict(int)
		by_currency: dict[str, Decimal] = defaultdict(lambda: _d("0"))
		for w in all_wallets:
			by_type[w.wallet_type] += 1
			by_currency[w.currency] += w.balance

		all_entries = [e for e in self.ledger.values() if e.tenant_id == self.tenant_id]
		credit_volume = sum(e.amount for e in all_entries if e.entry_type == "credit")
		debit_volume = sum(e.amount for e in all_entries if e.entry_type == "debit")

		self._audit(self.tenant_id, "wallet_analytics_generated", period)
		return {
			"period": period,
			"tenant_id": self.tenant_id,
			"total_wallets": len(all_wallets),
			"frozen_wallets": frozen_count,
			"total_balance": money(total_balance),
			"total_available": money(total_available),
			"by_wallet_type": dict(by_type),
			"balance_by_currency": {k: money(v) for k, v in by_currency.items()},
			"total_credit_volume": money(credit_volume),
			"total_debit_volume": money(debit_volume),
			"total_ledger_entries": len(all_entries),
			"generated_at": _iso(),
		}

	async def merge_wallets(
		self,
		source_wallet_id: str,
		target_wallet_id: str,
		approved_by: str,
	) -> dict[str, Any]:
		"""Merge source wallet balance into target wallet and close source."""
		assert source_wallet_id and target_wallet_id, "both wallet IDs required"
		assert source_wallet_id != target_wallet_id, "source and target must differ"
		assert approved_by, "approved_by required"
		await asyncio.sleep(0)

		source = self._tenant_wallet(source_wallet_id, self.tenant_id)
		target = self._tenant_wallet(target_wallet_id, self.tenant_id)
		if source.currency != target.currency:
			raise ValueError(f"currency mismatch: {source.currency} vs {target.currency}")

		balance_to_move = source.available_balance
		if balance_to_move > 0:
			idempotency_key = f"merge-{source_wallet_id}-{target_wallet_id}"
			self.debit_wallet(f"{idempotency_key}-d", self.tenant_id, source_wallet_id, balance_to_move, "wallet merge debit", idempotency_key)
			self.credit_wallet(f"{idempotency_key}-c", self.tenant_id, target_wallet_id, balance_to_move, "wallet merge credit", idempotency_key)

		source.status = "closed"  # type: ignore[attr-defined]
		self._audit(self.tenant_id, "wallets_merged", target_wallet_id)
		return {
			"source_wallet_id": source_wallet_id,
			"target_wallet_id": target_wallet_id,
			"amount_transferred": money(balance_to_move),
			"currency": source.currency,
			"approved_by": approved_by,
			"merged_at": _iso(),
		}

	# ------------------------------------------------------------------
	# Additional async methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return digital wallets service health status."""
		return {
			"service": "digital_wallets", "status": "healthy",
			"wallet_count": len(self.wallets), "frozen_wallets": sum(1 for s in self._freeze_state.values() if s.get("frozen")),
			"checked_at": _iso(),
		}

	async def bulk_create_wallets(self, wallets: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-create wallets for multiple customers."""
		processed, errors = [], []
		for w in wallets:
			try:
				rec = await self.create_wallet(customer_id=w["customer_id"], currency=w.get("currency", "KES"), wallet_type=w.get("wallet_type", "consumer"))
				processed.append(rec["id"])
			except Exception as exc:
				errors.append({"input": w, "error": str(exc)})
		return {"processed": len(processed), "failed": len(errors), "wallet_ids": processed}

	async def mpesa_wallet_topup(self, wallet_id: str, amount: float, mpesa_receipt: str) -> dict[str, Any]:
		"""Top up a wallet from an M-Pesa payment."""
		assert mpesa_receipt and amount > 0
		return await self.top_up_wallet(wallet_id, amount, f"mpesa:{mpesa_receipt}")

	async def airtel_money_wallet_topup(self, wallet_id: str, amount: float, transaction_id: str) -> dict[str, Any]:
		"""Top up a wallet from an Airtel Money payment."""
		assert transaction_id and amount > 0
		return await self.top_up_wallet(wallet_id, amount, f"airtel_money:{transaction_id}")

	async def pesalink_wallet_topup(self, wallet_id: str, amount: float, reference: str) -> dict[str, Any]:
		"""Top up a wallet via PesaLink interbank transfer."""
		assert reference and amount > 0
		return await self.top_up_wallet(wallet_id, amount, f"pesalink:{reference}")

	async def wallet_to_mpesa(self, wallet_id: str, amount: float, phone: str) -> dict[str, Any]:
		"""Withdraw from wallet and send to M-Pesa number."""
		if not phone.startswith(("07", "01", "254", "+254")):
			raise ValueError("invalid_mpesa_phone")
		return await self.withdraw_from_wallet(wallet_id, amount, f"mpesa:{phone[-9:]}")

	async def merchant_payment(self, wallet_id: str, merchant_id: str, amount: float, reference: str) -> dict[str, Any]:
		"""Pay a merchant from a wallet (scan-to-pay / merchant QR)."""
		assert merchant_id and reference
		return await self.withdraw_from_wallet(wallet_id, amount, f"merchant:{merchant_id}:{reference}")

	async def refund_to_wallet(self, wallet_id: str, amount: float, reason: str, original_ref: str) -> dict[str, Any]:
		"""Credit a refund to a wallet."""
		assert reason and original_ref
		idempotency = f"refund-{wallet_id}-{original_ref}"
		entry = self.credit_wallet(idempotency, self.tenant_id, wallet_id, _d(str(amount)), f"refund:{reason}", idempotency)
		self._audit(self.tenant_id, "refund_credited", wallet_id)
		return entry | {"reason": reason, "original_ref": original_ref}

	async def wallet_limits_check(self, wallet_id: str, amount: float, operation: str) -> dict[str, Any]:
		"""Check if an operation amount is within wallet limits."""
		limits = self._limits.get(wallet_id, {"daily_limit": _d("100000"), "monthly_limit": _d("500000")})
		spent = self._spend.get(wallet_id, {"daily": _d("0"), "monthly": _d("0")})
		amount_d = _d(str(amount))
		daily_ok = spent["daily"] + amount_d <= limits["daily_limit"]
		monthly_ok = spent["monthly"] + amount_d <= limits["monthly_limit"]
		return {
			"wallet_id": wallet_id, "operation": operation, "amount": str(amount_d),
			"daily_limit": str(limits["daily_limit"]), "daily_spent": str(spent["daily"]),
			"monthly_limit": str(limits["monthly_limit"]), "monthly_spent": str(spent["monthly"]),
			"within_daily_limit": daily_ok, "within_monthly_limit": monthly_ok,
			"allowed": daily_ok and monthly_ok, "checked_at": _iso(),
		}

	async def export_wallet_data(self, wallet_id: str, fmt: str = "json") -> dict[str, Any]:
		"""Export wallet ledger data for reporting or porting."""
		assert fmt in {"json", "csv", "excel"}
		wallet = self._tenant_wallet(wallet_id, self.tenant_id)
		entries = [e.to_dict() for e in self.ledger.values() if e.wallet_id == wallet_id]
		return {
			"wallet_id": wallet_id, "currency": wallet.currency, "format": fmt,
			"entry_count": len(entries),
			"file_reference": f"wallet_{wallet_id}_{fmt}", "generated_at": _iso(),
		}

	async def reward_points_redemption(self, wallet_id: str, points: int, redemption_type: str = "cashback") -> dict[str, Any]:
		"""Redeem loyalty points for cashback, airtime, or vouchers from a wallet."""
		assert points > 0, "points must be positive"
		assert redemption_type in {"cashback", "airtime", "voucher"}, f"unsupported: {redemption_type}"
		kes_value = round(points * 0.01, 2)
		if redemption_type == "cashback":
			idempotency = f"redemption-{wallet_id}-{points}"
			self.credit_wallet(idempotency, self.tenant_id, wallet_id, _d(str(kes_value)), f"points_redemption:{points}", idempotency)
		self._audit(self.tenant_id, "points_redeemed", wallet_id)
		return {"wallet_id": wallet_id, "points_redeemed": points, "kes_value": kes_value, "redemption_type": redemption_type, "redeemed_at": _iso()}

	async def reset_monthly_spend(self, wallet_id: str) -> dict[str, Any]:
		"""Reset monthly spend counters (called by scheduler on month boundary)."""
		assert wallet_id, "wallet_id required"
		await asyncio.sleep(0)
		prev = self._spend[wallet_id]["monthly"]
		self._spend[wallet_id]["monthly"] = _d("0")
		self._audit(self.tenant_id, "monthly_spend_reset", wallet_id)
		return {"wallet_id": wallet_id, "previous_monthly_spend": money(prev), "reset_at": _iso()}

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _tenant_wallet(self, wallet_id: str, tenant_id: str) -> Wallet:
		wallet = self.wallets.get(wallet_id)
		if wallet is None or wallet.tenant_id != tenant_id:
			raise KeyError(f"unknown wallet: {wallet_id}")
		return wallet

	def _record_ledger(
		self,
		entry_id: str,
		tenant_id: str,
		wallet_id: str,
		entry_type: str,
		amount: Decimal,
		currency: str,
		description: str,
		idempotency_key: str,
	) -> dict[str, Any]:
		if not idempotency_key:
			raise PermissionError("idempotency_key_required")
		entry = WalletLedgerEntry(entry_id, tenant_id, wallet_id, entry_type, amount, currency, description, idempotency_key)
		self.ledger[entry_id] = entry
		return entry.to_dict()

	def _record_evidence(
		self,
		evidence_id: str,
		tenant_id: str,
		kind: str,
		reference_id: str,
		status: str,
		metadata: dict[str, Any],
	) -> dict[str, Any]:
		evidence = WalletEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self.actor_id,
			"timestamp": _iso(),
		})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "wallet_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "wallet_policy_denied")


FintechWalletsService = DigitalWalletsService
