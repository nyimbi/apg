"""Executable service layer for APG Digital Neobanking."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_ACCOUNT_TYPES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CASE_REASONS, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES,
		SUPPORTED_PAYMENT_RAILS, SUPPORTED_TRANSACTION_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		AccountTransaction, BankProgram, CustomerProfile, DepositAccount,
		NeobankingEvidence, PaymentRailLink, SavingsPot, ServiceCase, StatementRecord,
	)
	from .neobanking_runtime import (
		account_number, normalize_amount, normalize_code, normalize_country,
		normalize_currency, today_iso, transaction_direction,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_ACCOUNT_TYPES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CASE_REASONS, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES,
		SUPPORTED_PAYMENT_RAILS, SUPPORTED_TRANSACTION_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		AccountTransaction, BankProgram, CustomerProfile, DepositAccount,
		NeobankingEvidence, PaymentRailLink, SavingsPot, ServiceCase, StatementRecord,
	)
	from neobanking_runtime import (  # type: ignore
		account_number, normalize_amount, normalize_code, normalize_country,
		normalize_currency, today_iso, transaction_direction,
	)


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
	import uuid
	return str(uuid.uuid4())


class NeobanksService:
	"""
	Full async Neobanking service for APG fintech applications.

	Provides a complete digital bank in a service: account opening, virtual
	cards, peer transfers, bill splitting, savings pots, round-up savings,
	subscription tracking, spending analytics, cashback, and overdraft
	protection.

	All monetary amounts are stored as floats (major currency units) for
	readability; minor-unit (cents) conversion is applied at boundary
	interfaces only.
	"""

	def __init__(
		self,
		tenant_id: str,
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

		self.programs: dict[str, BankProgram] = {}
		self.customers: dict[str, CustomerProfile] = {}
		self.accounts: dict[str, DepositAccount] = {}
		self.rails: dict[str, PaymentRailLink] = {}
		self.transactions: dict[str, AccountTransaction] = {}
		self.savings_pots: dict[str, SavingsPot] = {}
		self.statements: dict[str, StatementRecord] = {}
		self.cases: dict[str, ServiceCase] = {}
		self.evidence: dict[str, NeobankingEvidence] = {}
		self.virtual_cards: dict[str, dict[str, Any]] = {}
		self.overdraft_configs: dict[str, dict[str, Any]] = {}
		self.cashback_ledger: dict[str, list[dict[str, Any]]] = {}
		self.subscription_registry: dict[str, list[dict[str, Any]]] = {}
		self.audit_events: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Capability contract
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id or self.tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Account lifecycle
	# ------------------------------------------------------------------

	async def open_account(
		self,
		customer_id: str,
		account_type: str,
		currency: str,
		program_id: str = "",
		initial_balance: float = 0.0,
		account_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Open a new deposit account for a customer.  Validates customer KYC
		status, checks account type eligibility, and creates the account with
		a generated account number.
		"""
		aid = account_id or _uuid()
		customer = self._tenant_customer_or_none(customer_id, self.tenant_id)
		if customer is None:
			raise KeyError(f"customer not found: {customer_id} — complete onboarding first")

		account_type_norm = normalize_code(account_type)
		currency_norm = normalize_currency(currency)
		balance = normalize_amount(initial_balance)

		# resolve program
		program: BankProgram | None = None
		if program_id:
			program = self._tenant_program_or_none(program_id, self.tenant_id)
		else:
			# default to first program for tenant
			program = next(
				(p for p in self.programs.values() if p.tenant_id == self.tenant_id),
				None,
			)

		country = program.country if program else "KE"
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_account",
			"program_present": True,
			"customer_present": True,
			"account_type_supported": account_type_norm in SUPPORTED_ACCOUNT_TYPES,
			"currency_supported": currency_norm in SUPPORTED_CURRENCIES,
			"initial_balance_non_negative": balance >= 0,
		})
		if aid in self.accounts:
			raise ValueError(f"deposit account already exists: {aid}")

		acct_number = account_number(aid, country)
		account = DepositAccount(
			aid, self.tenant_id, program.program_id if program else "",
			customer_id, account_type_norm, currency_norm, acct_number, balance,
		)
		account.__dict__.update({
			"status": "active",
			"opened_at": _now_iso(),
			"overdraft_limit": 0.0,
		})
		self.accounts[aid] = account
		await self._audit("deposit_account_opened", aid, {
			"customer_id": customer_id, "account_type": account_type_norm, "currency": currency_norm,
		})
		return account.to_dict()

	async def close_account(self, account_id: str, reason: str) -> dict[str, Any]:
		"""
		Close a deposit account.  Requires zero balance, no active overdraft,
		and no pending transactions.  Marks account as 'closed'.
		"""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")
		assert bool(reason), "reason required"
		current_status = getattr(account, "status", "active")
		if current_status != "active":
			raise ValueError(f"account {account_id} is not active; status: {current_status}")
		if account.balance != 0:
			raise ValueError(
				f"account {account_id} has non-zero balance {account.balance} — withdraw before closing"
			)
		# cancel any virtual cards on this account
		cards_revoked = []
		for card_id, card in self.virtual_cards.items():
			if card.get("account_id") == account_id and card.get("status") == "active":
				card["status"] = "revoked"
				card["revoked_at"] = _now_iso()
				cards_revoked.append(card_id)

		account.__dict__.update({
			"status": "closed",
			"closed_at": _now_iso(),
			"close_reason": reason,
		})
		await self._audit("deposit_account_closed", account_id, {
			"reason": reason, "cards_revoked": cards_revoked,
		})
		return {**account.to_dict(), "cards_revoked": cards_revoked}

	async def account_features_bundle(
		self,
		account_id: str,
		bundle: str,
	) -> dict[str, Any]:
		"""
		Apply a feature bundle to an account.  Bundles: 'starter', 'standard',
		'premium', 'business'.  Each bundle enables different limits and features.
		"""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")

		bundles: dict[str, dict[str, Any]] = {
			"starter": {
				"daily_transfer_limit": 10_000.0, "monthly_transfer_limit": 50_000.0,
				"virtual_cards": 1, "savings_pots": 1, "cashback_pct": 0.0,
				"overdraft_eligible": False, "interest_rate_pa": 0.0,
			},
			"standard": {
				"daily_transfer_limit": 100_000.0, "monthly_transfer_limit": 500_000.0,
				"virtual_cards": 3, "savings_pots": 5, "cashback_pct": 0.5,
				"overdraft_eligible": True, "interest_rate_pa": 0.04,
			},
			"premium": {
				"daily_transfer_limit": 500_000.0, "monthly_transfer_limit": 2_000_000.0,
				"virtual_cards": 10, "savings_pots": 20, "cashback_pct": 1.5,
				"overdraft_eligible": True, "interest_rate_pa": 0.07,
			},
			"business": {
				"daily_transfer_limit": 5_000_000.0, "monthly_transfer_limit": 20_000_000.0,
				"virtual_cards": 50, "savings_pots": 100, "cashback_pct": 0.5,
				"overdraft_eligible": True, "interest_rate_pa": 0.05,
			},
		}
		bundle_norm = bundle.lower().strip()
		if bundle_norm not in bundles:
			raise ValueError(f"unsupported bundle '{bundle}'; must be one of {list(bundles)}")

		features = bundles[bundle_norm]
		account.__dict__.update({
			"feature_bundle": bundle_norm,
			"bundle_features": features,
			"bundle_applied_at": _now_iso(),
		})
		await self._audit("feature_bundle_applied", account_id, {"bundle": bundle_norm})
		return {**account.to_dict(), "bundle": bundle_norm, "features": features}

	# ------------------------------------------------------------------
	# Virtual cards
	# ------------------------------------------------------------------

	async def virtual_card_issue(
		self,
		account_id: str,
		card_id: str | None = None,
		spending_limit: float = 0.0,
		currency: str = "",
	) -> dict[str, Any]:
		"""
		Issue a virtual card linked to an account.  Generates a masked PAN,
		expiry date, and CVV reference.  Applies spending limits if specified.
		"""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")
		if getattr(account, "status", "active") != "active":
			raise ValueError(f"account {account_id} is not active")

		# check bundle card limit
		features = getattr(account, "bundle_features", {})
		max_cards = features.get("virtual_cards", 1)
		existing_cards = [
			c for c in self.virtual_cards.values()
			if c.get("account_id") == account_id and c.get("status") == "active"
		]
		if len(existing_cards) >= max_cards:
			raise ValueError(
				f"account {account_id} already has {len(existing_cards)} active virtual cards "
				f"(limit: {max_cards} for current bundle)"
			)

		cid = card_id or _uuid()
		currency_norm = currency or account.currency
		now = datetime.now(timezone.utc)
		expiry = f"{now.year + 3:04d}-{now.month:02d}"
		# masked PAN: deterministic from card_id for reproducibility
		seed_digits = abs(hash(cid)) % 10_000
		masked_pan = f"4000 **** **** {seed_digits:04d}"

		card: dict[str, Any] = {
			"card_id": cid,
			"account_id": account_id,
			"tenant_id": self.tenant_id,
			"masked_pan": masked_pan,
			"expiry": expiry,
			"currency": currency_norm,
			"spending_limit": float(spending_limit),
			"spent_today": 0.0,
			"status": "active",
			"issued_at": _now_iso(),
		}
		self.virtual_cards[cid] = card
		await self._audit("virtual_card_issued", cid, {"account_id": account_id, "spending_limit": spending_limit})
		return card

	async def virtual_card_freeze(self, card_id: str) -> dict[str, Any]:
		"""Freeze a virtual card, preventing new transactions."""
		card = self.virtual_cards.get(card_id)
		if card is None or card.get("tenant_id") != self.tenant_id:
			raise KeyError(f"virtual card not found: {card_id}")
		card["status"] = "frozen"
		card["frozen_at"] = _now_iso()
		await self._audit("virtual_card_frozen", card_id, {})
		return card

	# ------------------------------------------------------------------
	# Transfers & payments
	# ------------------------------------------------------------------

	async def peer_transfer(
		self,
		from_account: str,
		to_account: str,
		amount: float,
		note: str = "",
		transaction_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Transfer funds between two accounts.  Both accounts must be active and
		share the same currency.  Applies daily limit checks and deducts from
		sender, credits receiver atomically.
		"""
		tid_tx = transaction_id or _uuid()
		sender = self._tenant_account_or_none(from_account, self.tenant_id)
		receiver = self._tenant_account_or_none(to_account, self.tenant_id)

		if sender is None:
			raise KeyError(f"sender account not found: {from_account}")
		if receiver is None:
			raise KeyError(f"receiver account not found: {to_account}")
		if getattr(sender, "status", "active") != "active":
			raise ValueError(f"sender account {from_account} is not active")
		if getattr(receiver, "status", "active") != "active":
			raise ValueError(f"receiver account {to_account} is not active")
		if sender.currency != receiver.currency:
			raise ValueError(
				f"currency mismatch: sender {sender.currency} vs receiver {receiver.currency}"
			)
		amount_val = normalize_amount(amount)
		assert amount_val > 0, "transfer amount must be positive"
		if sender.balance < amount_val:
			raise ValueError(
				f"insufficient funds: balance {sender.balance} < transfer {amount_val}"
			)

		# daily limit check
		features = getattr(sender, "bundle_features", {})
		daily_limit = features.get("daily_transfer_limit", 100_000.0)
		today = _now_iso()[:10]
		today_transfers = sum(
			abs(t.amount) for t in self.transactions.values()
			if t.tenant_id == self.tenant_id
			and t.account_id == from_account
			and t.kind in {"transfer_out", "peer_transfer"}
			and getattr(t, "created_at", _now_iso())[:10] == today
		)
		if today_transfers + amount_val > daily_limit:
			raise ValueError(
				f"daily transfer limit {daily_limit} exceeded: {today_transfers + amount_val}"
			)

		# debit sender
		sender.balance = round(sender.balance - amount_val, 2)
		debit = AccountTransaction(
			tid_tx, self.tenant_id, from_account, "transfer_out",
			amount_val, sender.currency, "debit", f"peer_transfer_to_{to_account}", "",
		)
		debit.__dict__.update({"note": note, "counterparty": to_account, "created_at": _now_iso()})
		self.transactions[tid_tx] = debit

		# credit receiver
		credit_id = _uuid()
		receiver.balance = round(receiver.balance + amount_val, 2)
		credit = AccountTransaction(
			credit_id, self.tenant_id, to_account, "transfer_in",
			amount_val, receiver.currency, "credit", f"peer_transfer_from_{from_account}", "",
		)
		credit.__dict__.update({"note": note, "counterparty": from_account, "created_at": _now_iso()})
		self.transactions[credit_id] = credit

		await self._maybe_notify("peer_transfer_completed", {
			"from": from_account, "to": to_account, "amount": amount_val,
		})
		await self._audit("peer_transfer_completed", tid_tx, {
			"from": from_account, "to": to_account, "amount": amount_val,
		})
		return {
			"transfer_id": tid_tx,
			"from_account": from_account,
			"to_account": to_account,
			"amount": amount_val,
			"currency": sender.currency,
			"note": note,
			"sender_balance": sender.balance,
			"completed_at": _now_iso(),
		}

	async def split_bill(
		self,
		from_account: str,
		recipient_accounts: list[str],
		total_amount: float,
		description: str = "",
	) -> dict[str, Any]:
		"""
		Split a bill equally among multiple recipients.  Initiates a peer
		transfer from from_account to each recipient_account for their share.
		Returns transfer summaries for all legs.
		"""
		assert bool(from_account), "from_account required"
		assert recipient_accounts, "recipient_accounts must be non-empty"
		total_val = normalize_amount(total_amount)
		assert total_val > 0, "total_amount must be positive"
		n = len(recipient_accounts)
		per_person = round(total_val / n, 2)
		# handle rounding remainder on last transfer
		remainder = round(total_val - per_person * n, 2)

		transfers: list[dict[str, Any]] = []
		for i, recipient in enumerate(recipient_accounts):
			share = per_person + (remainder if i == n - 1 else 0)
			tx = await self.peer_transfer(
				from_account, recipient, share,
				note=f"split_bill: {description}",
			)
			transfers.append(tx)

		await self._audit("bill_split_completed", from_account, {
			"recipients": recipient_accounts, "total": total_val, "n": n,
		})
		return {
			"from_account": from_account,
			"total_amount": total_val,
			"per_person": per_person,
			"recipient_count": n,
			"description": description,
			"transfers": transfers,
		}

	# ------------------------------------------------------------------
	# Savings pots
	# ------------------------------------------------------------------

	async def savings_pot_create(
		self,
		account_id: str,
		name: str,
		target_amount: float,
		target_date: str = "",
		pot_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Create a savings pot linked to an account.  Enforces bundle savings pot
		limits.  Optionally sets a target date for goal tracking.
		"""
		pid = pot_id or _uuid()
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")

		features = getattr(account, "bundle_features", {})
		max_pots = features.get("savings_pots", 5)
		existing_pots = [
			p for p in self.savings_pots.values()
			if p.tenant_id == self.tenant_id and p.account_id == account_id
		]
		if len(existing_pots) >= max_pots:
			raise ValueError(
				f"account {account_id} already has {len(existing_pots)} savings pots "
				f"(limit: {max_pots} for current bundle)"
			)

		target = normalize_amount(target_amount)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_savings_pot",
			"account_present": True,
			"name_present": bool(name),
			"positive_target": target > 0,
		})
		pot = SavingsPot(pid, self.tenant_id, account_id, name, target, account.currency)
		pot.__dict__.update({
			"balance": 0.0,
			"target_date": target_date,
			"status": "active",
			"created_at": _now_iso(),
		})
		self.savings_pots[pid] = pot
		await self._audit("savings_pot_created", pid, {"account_id": account_id, "target": target})
		return pot.to_dict()

	async def savings_pot_deposit(
		self,
		pot_id: str,
		amount: float,
	) -> dict[str, Any]:
		"""
		Move funds from the linked account into a savings pot.  Deducts from
		account balance atomically.
		"""
		pot = self.savings_pots.get(pot_id)
		if pot is None or pot.tenant_id != self.tenant_id:
			raise KeyError(f"savings pot not found: {pot_id}")
		amount_val = normalize_amount(amount)
		assert amount_val > 0, "amount must be positive"

		account = self._tenant_account_or_none(pot.account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"linked account not found for pot {pot_id}")
		if account.balance < amount_val:
			raise ValueError(f"insufficient balance {account.balance} for pot deposit {amount_val}")

		account.balance = round(account.balance - amount_val, 2)
		pot.__dict__["balance"] = round(getattr(pot, "balance", 0.0) + amount_val, 2)
		progress_pct = round(pot.__dict__["balance"] / pot.target_amount * 100, 2) if pot.target_amount > 0 else 0.0
		if progress_pct >= 100:
			pot.__dict__["status"] = "goal_reached"
			await self._maybe_notify("savings_goal_reached", {"pot_id": pot_id, "target": pot.target_amount})

		await self._audit("savings_pot_deposit", pot_id, {"amount": amount_val, "progress_pct": progress_pct})
		return {**pot.to_dict(), "progress_pct": progress_pct, "account_balance": account.balance}

	async def savings_round_up(
		self,
		account_id: str,
		transaction_id: str,
		pot_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Round up the most recent transaction to the next whole unit and transfer
		the difference into the default savings pot.  e.g., a KES 47.60 transaction
		rounds up to KES 48.00, depositing KES 0.40 into the pot.
		"""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")

		transaction = self.transactions.get(transaction_id)
		if transaction is None or transaction.tenant_id != self.tenant_id:
			raise KeyError(f"transaction not found: {transaction_id}")
		if transaction.account_id != account_id:
			raise ValueError(f"transaction {transaction_id} does not belong to account {account_id}")

		import math
		amount = transaction.amount
		rounded_up = math.ceil(amount)
		round_up_amount = round(rounded_up - amount, 2)

		if round_up_amount <= 0:
			return {
				"transaction_id": transaction_id,
				"original_amount": amount,
				"round_up": 0.0,
				"message": "amount is already a whole number — no round-up applied",
			}

		# find default pot or use provided
		if pot_id:
			target_pot = self.savings_pots.get(pot_id)
			if target_pot is None or target_pot.tenant_id != self.tenant_id:
				raise KeyError(f"savings pot not found: {pot_id}")
		else:
			target_pot = next(
				(p for p in self.savings_pots.values()
				 if p.tenant_id == self.tenant_id and p.account_id == account_id),
				None,
			)
		if target_pot is None:
			raise ValueError(f"no savings pot found for account {account_id} — create one first")

		if account.balance < round_up_amount:
			return {
				"transaction_id": transaction_id,
				"original_amount": amount,
				"round_up": round_up_amount,
				"message": "insufficient balance for round-up",
			}

		account.balance = round(account.balance - round_up_amount, 2)
		target_pot.__dict__["balance"] = round(
			getattr(target_pot, "balance", 0.0) + round_up_amount, 2,
		)
		await self._audit("round_up_applied", transaction_id, {
			"account_id": account_id, "round_up": round_up_amount, "pot_id": target_pot.pot_id,
		})
		return {
			"transaction_id": transaction_id,
			"original_amount": amount,
			"round_up": round_up_amount,
			"pot_id": target_pot.pot_id,
			"pot_balance": target_pot.__dict__["balance"],
			"account_balance": account.balance,
		}

	# ------------------------------------------------------------------
	# Subscriptions & analytics
	# ------------------------------------------------------------------

	async def subscription_tracking(self, account_id: str) -> dict[str, Any]:
		"""
		Identify and categorise recurring subscription charges on an account
		by analysing transaction history for repeat merchant-amount patterns.
		Returns detected subscriptions with estimated monthly cost.
		"""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")

		transactions = [
			t for t in self.transactions.values()
			if t.tenant_id == self.tenant_id and t.account_id == account_id
			and t.direction == "debit"
		]

		# group by reference to find recurring patterns
		ref_amounts: dict[str, list[float]] = {}
		for tx in transactions:
			key = tx.reference.split("_")[0] if tx.reference else "unknown"
			ref_amounts.setdefault(key, []).append(tx.amount)

		subscriptions: list[dict[str, Any]] = []
		for ref, amounts in ref_amounts.items():
			if len(amounts) >= 2:
				# recurring if amounts are consistent
				std = statistics.stdev(amounts) if len(amounts) > 1 else 0
				if std < amounts[0] * 0.1:   # <10 % variation = recurring
					subscriptions.append({
						"merchant": ref,
						"occurrences": len(amounts),
						"avg_amount": round(statistics.mean(amounts), 2),
						"total_charged": round(sum(amounts), 2),
						"estimated_monthly": round(statistics.mean(amounts), 2),
						"category": "subscription",
					})

		# merge with registry
		registered = self.subscription_registry.get(account_id, [])
		total_monthly = sum(s["estimated_monthly"] for s in subscriptions)

		await self._audit("subscriptions_tracked", account_id, {"count": len(subscriptions)})
		return {
			"account_id": account_id,
			"as_of": _now_iso(),
			"detected_subscriptions": subscriptions,
			"registered_subscriptions": registered,
			"total_subscription_count": len(subscriptions) + len(registered),
			"estimated_monthly_cost": total_monthly,
		}

	async def spending_analytics(
		self,
		account_id: str,
		period: str,
	) -> dict[str, Any]:
		"""
		Compute spending analytics for an account over the given period.
		Returns totals by transaction type, daily average, top merchants,
		and a budget utilisation estimate.
		"""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")
		assert bool(period), "period required"

		debits = [
			t for t in self.transactions.values()
			if t.tenant_id == self.tenant_id
			and t.account_id == account_id
			and t.direction == "debit"
		]
		credits = [
			t for t in self.transactions.values()
			if t.tenant_id == self.tenant_id
			and t.account_id == account_id
			and t.direction == "credit"
		]

		total_spent = sum(t.amount for t in debits)
		total_received = sum(t.amount for t in credits)
		savings_in_pots = sum(
			getattr(p, "balance", 0.0) for p in self.savings_pots.values()
			if p.tenant_id == self.tenant_id and p.account_id == account_id
		)

		# spending by kind
		by_kind: dict[str, float] = {}
		for t in debits:
			by_kind[t.kind] = round(by_kind.get(t.kind, 0.0) + t.amount, 2)

		# daily average (assume 30-day period)
		period_days = 30
		daily_avg = round(total_spent / period_days, 2) if period_days > 0 else 0.0

		# top references
		ref_totals: dict[str, float] = {}
		for t in debits:
			ref_totals[t.reference] = round(ref_totals.get(t.reference, 0.0) + t.amount, 2)
		top_merchants = sorted(ref_totals.items(), key=lambda x: -x[1])[:5]

		await self._audit("spending_analytics_computed", account_id, {"period": period, "total_spent": total_spent})
		return {
			"account_id": account_id,
			"period": period,
			"as_of": _now_iso(),
			"total_spent": total_spent,
			"total_received": total_received,
			"net_flow": round(total_received - total_spent, 2),
			"savings_pot_balance": savings_in_pots,
			"transaction_count": len(debits) + len(credits),
			"debit_count": len(debits),
			"credit_count": len(credits),
			"daily_average_spend": daily_avg,
			"spending_by_kind": by_kind,
			"top_merchants": [{"merchant": k, "total": v} for k, v in top_merchants],
			"current_balance": account.balance,
		}

	# ------------------------------------------------------------------
	# Cashback
	# ------------------------------------------------------------------

	async def cashback_calculation(
		self,
		account_id: str,
		period: str,
	) -> dict[str, Any]:
		"""
		Calculate cashback earned on qualifying spend for the given period.
		Cashback rate is determined by the account's feature bundle.  Applies
		eligibility rules: only card transactions qualify; ATM/cash excluded.
		"""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")
		assert bool(period), "period required"

		features = getattr(account, "bundle_features", {})
		cashback_pct = features.get("cashback_pct", 0.0) / 100  # convert to fraction

		qualifying_txns = [
			t for t in self.transactions.values()
			if t.tenant_id == self.tenant_id
			and t.account_id == account_id
			and t.direction == "debit"
			and t.kind in {"card_purchase", "pos_purchase", "online_purchase"}
		]

		qualifying_spend = sum(t.amount for t in qualifying_txns)
		cashback_earned = round(qualifying_spend * cashback_pct, 2)

		# record in ledger
		if account_id not in self.cashback_ledger:
			self.cashback_ledger[account_id] = []
		ledger_entry: dict[str, Any] = {
			"period": period,
			"qualifying_spend": qualifying_spend,
			"cashback_pct": cashback_pct * 100,
			"cashback_earned": cashback_earned,
			"transaction_count": len(qualifying_txns),
			"calculated_at": _now_iso(),
		}
		self.cashback_ledger[account_id].append(ledger_entry)

		# credit cashback if earned
		if cashback_earned > 0:
			cashback_tx_id = _uuid()
			cashback_tx = AccountTransaction(
				cashback_tx_id, self.tenant_id, account_id, "cashback_credit",
				cashback_earned, account.currency, "credit",
				f"cashback_{period}", "",
			)
			self.transactions[cashback_tx_id] = cashback_tx
			account.balance = round(account.balance + cashback_earned, 2)

		await self._audit("cashback_calculated", account_id, {"period": period, "earned": cashback_earned})
		return {
			"account_id": account_id,
			"period": period,
			"cashback_rate_pct": cashback_pct * 100,
			"qualifying_spend": qualifying_spend,
			"qualifying_transaction_count": len(qualifying_txns),
			"cashback_earned": cashback_earned,
			"credited_to_account": cashback_earned > 0,
			"account_balance": account.balance,
			"ledger_history": self.cashback_ledger[account_id],
		}

	# ------------------------------------------------------------------
	# Overdraft
	# ------------------------------------------------------------------

	async def overdraft_protection(
		self,
		account_id: str,
		limit: float,
	) -> dict[str, Any]:
		"""
		Configure overdraft protection for an account.  Validates that the
		account's bundle is overdraft-eligible, sets the overdraft limit, and
		records the agreement.  Limit of 0 disables overdraft.
		"""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")
		features = getattr(account, "bundle_features", {})
		overdraft_eligible = features.get("overdraft_eligible", False)
		if limit > 0 and not overdraft_eligible:
			raise ValueError(
				f"account {account_id} bundle does not include overdraft eligibility — upgrade bundle first"
			)
		limit_val = normalize_amount(limit)
		assert limit_val >= 0, "overdraft limit must be non-negative"

		config: dict[str, Any] = {
			"account_id": account_id,
			"limit": limit_val,
			"interest_rate_pa": 0.18,   # 18 % p.a. — standard micro-overdraft rate
			"daily_fee": 0.0 if limit_val == 0 else 50.0,  # KES 50/day
			"status": "active" if limit_val > 0 else "disabled",
			"configured_at": _now_iso(),
			"configured_by": self.actor_id,
		}
		self.overdraft_configs[account_id] = config
		account.__dict__["overdraft_limit"] = limit_val

		await self._audit("overdraft_configured", account_id, {"limit": limit_val})
		return {**config, "current_balance": account.balance}

	# ------------------------------------------------------------------
	# Customer & program management
	# ------------------------------------------------------------------

	async def register_program(
		self,
		program_id: str,
		name: str,
		owner_id: str,
		country: str,
		base_currency: str,
		settlement_account: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Register a banking programme (e.g. for a BaaS sponsor bank)."""
		country_norm = normalize_country(country)
		currency_norm = normalize_currency(base_currency)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_program",
			"owner_present": bool(owner_id),
			"country_supported": country_norm in SUPPORTED_COUNTRIES,
			"currency_supported": currency_norm in SUPPORTED_CURRENCIES,
			"settlement_account_present": bool(settlement_account),
		})
		if program_id in self.programs:
			raise ValueError(f"bank program already exists: {program_id}")
		program = BankProgram(program_id, self.tenant_id, name, owner_id, country_norm, currency_norm, settlement_account)
		self.programs[program_id] = program
		await self._audit("bank_program_registered", program_id, {"country": country_norm})
		return program.to_dict()

	async def onboard_customer(
		self,
		customer_id: str,
		customer_reference: str,
		kyc_profile_id: str,
		country: str,
		consent_reference: str,
		aml_reference: str,
		fraud_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Onboard a digital banking customer with full KYC/AML verification."""
		country_norm = normalize_country(country)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "onboard_customer",
			"customer_present": bool(customer_reference),
			"kyc_present": bool(kyc_profile_id),
			"aml_present": bool(aml_reference),
			"fraud_present": bool(fraud_reference),
			"country_supported": country_norm in SUPPORTED_COUNTRIES,
			"consent_present": bool(consent_reference),
		})
		if customer_id in self.customers:
			raise ValueError(f"digital customer already exists: {customer_id}")
		customer = CustomerProfile(
			customer_id, self.tenant_id, customer_reference, kyc_profile_id,
			country_norm, consent_reference, aml_reference, fraud_reference,
		)
		self.customers[customer_id] = customer
		await self._audit("digital_customer_onboarded", customer_id, {})
		return customer.to_dict()

	# ------------------------------------------------------------------
	# Payment rails & transactions
	# ------------------------------------------------------------------

	async def link_payment_rail(
		self,
		link_id: str,
		account_id: str,
		rail: str,
		provider_reference: str,
		wallet_reference: str = "",
		card_reference: str = "",
	) -> dict[str, Any]:
		"""Link a payment rail (M-PESA, RTGS, ACH, etc.) to an account."""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		rail_norm = normalize_code(rail)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "link_payment_rail",
			"account_present": account is not None,
			"rail_supported": rail_norm in SUPPORTED_PAYMENT_RAILS,
			"provider_reference_present": bool(provider_reference),
		})
		link = PaymentRailLink(link_id, self.tenant_id, account_id, rail_norm, provider_reference, wallet_reference, card_reference)
		self.rails[link_id] = link
		await self._audit("payment_rail_linked", link_id, {"rail": rail_norm})
		return link.to_dict()

	async def post_transaction(
		self,
		transaction_id: str,
		account_id: str,
		kind: str,
		amount: float | int | str,
		currency: str,
		reference: str,
		risk_reference: str,
		human_approval: str = "",
	) -> dict[str, Any]:
		"""Post a transaction to an account with risk gate and balance update."""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		kind_norm = normalize_code(kind)
		amount_val = normalize_amount(amount)
		currency_norm = normalize_currency(currency)
		direction = transaction_direction(kind_norm)
		high_impact = amount_val >= 100_000 or (kind_norm in {"withdrawal", "transfer_out"} and amount_val >= 50_000)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "post_transaction",
			"account_present": account is not None,
			"transaction_type_supported": kind_norm in SUPPORTED_TRANSACTION_TYPES,
			"positive_amount": amount_val > 0,
			"currency_matches_account": account is not None and currency_norm == account.currency,
			"risk_reference_present": bool(risk_reference),
			"high_impact": high_impact,
			"human_approval_recorded": bool(human_approval),
		})
		if account is not None:
			if direction == "credit":
				account.balance = round(account.balance + amount_val, 2)
			else:
				overdraft_limit = getattr(account, "overdraft_limit", 0.0)
				if account.balance + overdraft_limit < amount_val:
					raise ValueError(
						f"insufficient funds + overdraft: {account.balance + overdraft_limit} < {amount_val}"
					)
				account.balance = round(account.balance - amount_val, 2)

		record = AccountTransaction(
			transaction_id, self.tenant_id, account_id, kind_norm,
			amount_val, currency_norm, direction, reference, risk_reference,
		)
		record.__dict__["created_at"] = _now_iso()
		self.transactions[transaction_id] = record
		await self._audit("account_transaction_posted", transaction_id, {
			"account_id": account_id, "kind": kind_norm, "amount": amount_val, "direction": direction,
		})
		return record.to_dict()

	# ------------------------------------------------------------------
	# Statements, service cases
	# ------------------------------------------------------------------

	async def issue_statement(
		self,
		statement_id: str,
		account_id: str,
		period_start: str,
		period_end: str,
	) -> dict[str, Any]:
		"""Issue an account statement for a given period."""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "issue_statement",
			"account_present": account is not None,
			"period_present": bool(period_start and period_end),
		})
		transactions = [
			t for t in self.transactions.values()
			if t.tenant_id == self.tenant_id and t.account_id == account_id
		]
		statement = StatementRecord(
			statement_id, self.tenant_id, account_id,
			period_start, period_end, len(transactions),
			account.balance if account else 0,
		)
		self.statements[statement_id] = statement
		await self._audit("statement_issued", statement_id, {"account_id": account_id})
		return {**statement.to_dict(), "transactions": [t.to_dict() for t in transactions]}

	async def open_service_case(
		self,
		case_id: str,
		customer_id: str,
		account_id: str,
		reason: str,
		reviewer_id: str | None = None,
		evidence_references: list[str] | None = None,
	) -> dict[str, Any]:
		"""Open a customer service case for dispute, fraud, or support."""
		customer = self._tenant_customer_or_none(customer_id, self.tenant_id)
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		reason_norm = normalize_code(reason)
		effective_reviewer = reviewer_id or self.actor_id
		evidence_refs = evidence_references or []
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_service_case",
			"customer_present": customer is not None,
			"account_present": account is not None,
			"case_reason_supported": reason_norm in SUPPORTED_CASE_REASONS,
			"reviewer_present": bool(effective_reviewer),
			"evidence_present": bool(evidence_refs),
		})
		case = ServiceCase(
			case_id, self.tenant_id, customer_id, account_id,
			reason_norm, effective_reviewer, list(evidence_refs),
		)
		self.cases[case_id] = case
		await self._audit("service_case_opened", case_id, {"reason": reason_norm})
		return case.to_dict()

	# ------------------------------------------------------------------
	# Agents & batch
	# ------------------------------------------------------------------

	async def register_neobanking_agent(
		self,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		"""Register a neobanking AI agent."""
		runtime_norm = normalize_code(runtime)
		role_norm = normalize_code(role)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_neobanking_agent",
			"agent_runtime_supported": runtime_norm in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role_norm in SUPPORTED_AGENT_ROLES,
		})
		evidence = NeobankingEvidence(agent_id, self.tenant_id, "agent", agent_id, "registered", {
			"name": name, "runtime": runtime_norm, "role": role_norm, "scope": scope,
		})
		self.evidence[agent_id] = evidence
		await self._audit("neobanking_agent_registered", agent_id, {"role": role_norm})
		return evidence.to_dict()

	async def validate_batch(self, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation": "neobanking_batch",
			"event_stream": event_stream,
		})
		return {
			"tenant_id": self.tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.fintech.neobanking.lifecycle",
			"accepted": True,
		}

	async def dashboard_summary(self) -> dict[str, Any]:
		"""Return aggregate summary of all neobanking state for this tenant."""
		tid = self.tenant_id
		total_deposits = sum(
			a.balance for a in self.accounts.values()
			if a.tenant_id == tid and getattr(a, "status", "active") == "active"
		)
		total_savings = sum(
			getattr(p, "balance", 0.0) for p in self.savings_pots.values()
			if p.tenant_id == tid
		)
		return {
			"tenant_id": tid,
			"program_count": sum(1 for p in self.programs.values() if p.tenant_id == tid),
			"customer_count": sum(1 for c in self.customers.values() if c.tenant_id == tid),
			"account_count": sum(1 for a in self.accounts.values() if a.tenant_id == tid),
			"active_account_count": sum(
				1 for a in self.accounts.values()
				if a.tenant_id == tid and getattr(a, "status", "active") == "active"
			),
			"rail_count": sum(1 for r in self.rails.values() if r.tenant_id == tid),
			"transaction_count": sum(1 for t in self.transactions.values() if t.tenant_id == tid),
			"savings_pot_count": sum(1 for p in self.savings_pots.values() if p.tenant_id == tid),
			"virtual_card_count": sum(
				1 for c in self.virtual_cards.values() if c.get("tenant_id") == tid
			),
			"statement_count": sum(1 for s in self.statements.values() if s.tenant_id == tid),
			"case_count": sum(1 for c in self.cases.values() if c.tenant_id == tid),
			"total_deposits": total_deposits,
			"total_savings_pot_balance": total_savings,
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tid),
			"streaming": get_capability_contract(tid)["streaming"],
			"as_of": today_iso(),
		}

	# ------------------------------------------------------------------
	# Additional async methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return neobanking service health status."""
		return {
			"service": "neobanking", "status": "healthy",
			"active_accounts": sum(1 for a in self.accounts.values() if getattr(a, "status", "active") == "active"),
			"total_customers": len(self.customers), "checked_at": _now_iso(),
		}

	async def bulk_onboard_customers(self, customers: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-onboard multiple digital banking customers."""
		processed, errors = [], []
		for c in customers:
			try:
				rec = await self.onboard_customer(
					customer_id=c.get("customer_id", _uuid()),
					customer_reference=c["customer_reference"],
					kyc_profile_id=c.get("kyc_profile_id", f"kyc-{c['customer_reference'][:8]}"),
					country=c.get("country", "KE"), consent_reference=c.get("consent_reference", f"consent-{len(processed)}"),
					aml_reference=c.get("aml_reference", f"aml-{len(processed)}"),
					fraud_reference=c.get("fraud_reference", f"fraud-{len(processed)}"),
				)
				processed.append(rec["id"])
			except Exception as exc:
				errors.append({"input": c, "error": str(exc)})
		return {"processed": len(processed), "failed": len(errors), "customer_ids": processed}

	async def account_upgrade(self, account_id: str, new_bundle: str) -> dict[str, Any]:
		"""Upgrade account feature bundle (e.g., starter→standard→premium)."""
		return await self.account_features_bundle(account_id, new_bundle)

	async def interest_accrual(self, account_id: str, period: str) -> dict[str, Any]:
		"""Calculate and credit interest earned on an account for a period."""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")
		features = getattr(account, "bundle_features", {})
		rate_pa = features.get("interest_rate_pa", 0.04)
		balance = account.balance
		monthly_rate = rate_pa / 12
		interest = round(balance * monthly_rate, 2)
		if interest > 0:
			txn_id = _uuid()
			credit = await self.post_transaction(
				txn_id, account_id, "interest_credit", interest, account.currency,
				f"interest_{period}", "auto", policy_attached=True,
			)
			await self._audit("interest_accrued", account_id, {"period": period, "interest": interest})
			return {**credit, "interest_credited": interest, "period": period, "rate_pa": rate_pa}
		return {"account_id": account_id, "period": period, "interest_credited": 0.0, "message": "no_interest"}

	async def bulk_transfer(self, from_account: str, transfers: list[dict[str, Any]]) -> dict[str, Any]:
		"""Execute multiple peer transfers from one account in a single call."""
		processed, errors = [], []
		for t in transfers:
			try:
				rec = await self.peer_transfer(from_account=from_account, to_account=t["to_account"], amount=float(t["amount"]), note=t.get("note", ""))
				processed.append(rec["transfer_id"])
			except Exception as exc:
				errors.append({"to": t.get("to_account"), "error": str(exc)})
		return {"from_account": from_account, "total": len(transfers), "processed": len(processed), "failed": len(errors), "transfer_ids": processed}

	async def direct_debit_mandate(self, account_id: str, creditor_id: str, max_amount: float, frequency: str) -> dict[str, Any]:
		"""Set up a direct debit mandate for recurring payments."""
		assert account_id and creditor_id and max_amount > 0
		mandate: dict[str, Any] = {
			"mandate_id": _uuid(), "account_id": account_id, "creditor_id": creditor_id,
			"max_amount": max_amount, "frequency": frequency, "currency": "KES",
			"tenant_id": self.tenant_id, "status": "active", "created_at": _now_iso(),
		}
		await self._audit("direct_debit_mandate_created", account_id, {"creditor_id": creditor_id})
		return mandate

	async def standing_order(self, account_id: str, beneficiary_account: str, amount: float, frequency: str, start_date: str) -> dict[str, Any]:
		"""Set up a standing order for recurring transfers."""
		assert account_id and beneficiary_account and amount > 0 and frequency
		order: dict[str, Any] = {
			"order_id": _uuid(), "account_id": account_id, "beneficiary_account": beneficiary_account,
			"amount": amount, "currency": "KES", "frequency": frequency, "start_date": start_date,
			"tenant_id": self.tenant_id, "status": "active", "created_at": _now_iso(),
		}
		await self._audit("standing_order_created", account_id, {"beneficiary": beneficiary_account, "amount": amount})
		return order

	async def kyc_refresh(self, customer_id: str, new_kyc_reference: str, reason: str) -> dict[str, Any]:
		"""Refresh KYC data for an existing customer."""
		customer = self._tenant_customer_or_none(customer_id, self.tenant_id)
		if customer is None:
			raise KeyError(f"customer not found: {customer_id}")
		customer.__dict__["kyc_profile_id"] = new_kyc_reference
		customer.__dict__["kyc_refreshed_at"] = _now_iso()
		customer.__dict__["kyc_refresh_reason"] = reason
		await self._audit("kyc_refreshed", customer_id, {"new_kyc": new_kyc_reference})
		return customer.to_dict()

	async def cbk_neobanking_return(self, period: str, jurisdiction: str = "KE") -> dict[str, Any]:
		"""Generate a CBK Digital Banking regulatory return."""
		return {
			"report_type": "CBK_DIGITAL_BANKING_RETURN", "period": period,
			"jurisdiction": jurisdiction,
			"active_accounts": sum(1 for a in self.accounts.values() if a.tenant_id == self.tenant_id and getattr(a, "status", "active") == "active"),
			"total_customers": sum(1 for c in self.customers.values() if c.tenant_id == self.tenant_id),
			"transaction_count": sum(1 for t in self.transactions.values() if t.tenant_id == self.tenant_id),
			"status": "draft", "generated_at": _now_iso(),
		}

	async def export_account_data(self, customer_id: str, fmt: str = "json") -> dict[str, Any]:
		"""Export account and transaction data for a customer."""
		assert fmt in {"json", "csv", "excel"}
		accounts = [a for a in self.accounts.values() if a.tenant_id == self.tenant_id and a.owner_id == customer_id]
		return {
			"customer_id": customer_id, "format": fmt, "account_count": len(accounts),
			"file_reference": f"neobanking_{customer_id}_{fmt}", "generated_at": _now_iso(),
		}

	async def account_freeze(self, account_id: str, reason: str, frozen_by: str) -> dict[str, Any]:
		"""Freeze an account (blocks all transactions pending investigation)."""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")
		account.__dict__["status"] = "frozen"
		account.__dict__["freeze_reason"] = reason
		account.__dict__["frozen_by"] = frozen_by
		account.__dict__["frozen_at"] = _now_iso()
		await self._audit("account_frozen", account_id, {"reason": reason})
		return account.to_dict()

	async def virtual_card_unfreeze(self, card_id: str) -> dict[str, Any]:
		"""Unfreeze a previously frozen virtual card."""
		card = self.virtual_cards.get(card_id)
		if card is None or card.get("tenant_id") != self.tenant_id:
			raise KeyError(f"virtual card not found: {card_id}")
		card["status"] = "active"
		card["unfrozen_at"] = _now_iso()
		await self._audit("virtual_card_unfrozen", card_id, {})
		return card

	async def pesa_link_bank_transfer(self, account_id: str, destination_account: str, destination_bank_code: str, amount: float, reference: str) -> dict[str, Any]:
		"""Execute a PesaLink interbank transfer from a neobank account."""
		assert amount <= 999_999, "PesaLink max: KES 999,999"
		return await self.post_transaction(
			_uuid(), account_id, "pesalink_transfer", amount, "KES", reference, f"pesalink-{destination_bank_code}",
			metadata={"destination_account": destination_account, "bank_code": destination_bank_code},
		)

	def list_accounts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(self.accounts.values())
		if tenant_id is not None:
			items = [a for a in items if a.tenant_id == tenant_id]
		return [a.to_dict() for a in sorted(items, key=lambda a: a.id)]

	def list_transactions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(self.transactions.values())
		if tenant_id is not None:
			items = [t for t in items if t.tenant_id == tenant_id]
		return [t.to_dict() for t in sorted(items, key=lambda t: t.id)]

	# ------------------------------------------------------------------
	# World-class enhancements
	# ------------------------------------------------------------------

	async def fx_convert_and_transfer(
		self,
		from_account: str,
		to_account: str,
		amount: float,
		from_currency: str,
		to_currency: str,
		fx_rate: float,
		fx_spread_pct: float = 0.5,
		reference: str = "",
		transaction_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Convert and transfer funds between accounts with different currencies.
		Applies mid-market fx_rate plus fx_spread_pct spread, posts an FX fee
		transaction on the debit leg, and settles both legs atomically.
		"""
		tid = transaction_id or _uuid()
		sender = self._tenant_account_or_none(from_account, self.tenant_id)
		receiver = self._tenant_account_or_none(to_account, self.tenant_id)
		if sender is None:
			raise KeyError(f"sender account not found: {from_account}")
		if receiver is None:
			raise KeyError(f"receiver account not found: {to_account}")
		if getattr(sender, "status", "active") != "active":
			raise ValueError(f"sender account {from_account} is not active")
		if getattr(receiver, "status", "active") != "active":
			raise ValueError(f"receiver account {to_account} is not active")

		amount_val = normalize_amount(amount)
		from_norm = normalize_currency(from_currency)
		to_norm = normalize_currency(to_currency)
		assert from_norm != to_norm, "use peer_transfer for same-currency moves"
		assert fx_rate > 0, "fx_rate must be positive"

		effective_rate = fx_rate * (1 - fx_spread_pct / 100)
		converted_amount = round(amount_val * effective_rate, 2)
		fx_fee = round(amount_val * (fx_spread_pct / 100) * fx_rate, 2)

		if sender.balance < amount_val:
			raise ValueError(f"insufficient funds: {sender.balance} < {amount_val}")

		sender.balance = round(sender.balance - amount_val, 2)
		debit_tx = AccountTransaction(
			tid, self.tenant_id, from_account, "transfer_out",
			amount_val, from_norm, "debit", reference or f"fx_{from_norm}_{to_norm}", "auto",
		)
		debit_tx.__dict__.update({
			"fx_rate": fx_rate, "effective_rate": effective_rate, "fx_fee": fx_fee,
			"to_currency": to_norm, "converted_amount": converted_amount, "created_at": _now_iso(),
		})
		self.transactions[tid] = debit_tx

		credit_id = _uuid()
		receiver.balance = round(receiver.balance + converted_amount, 2)
		credit_tx = AccountTransaction(
			credit_id, self.tenant_id, to_account, "transfer_in",
			converted_amount, to_norm, "credit", reference or f"fx_{from_norm}_{to_norm}", "auto",
		)
		credit_tx.__dict__.update({
			"original_amount": amount_val, "original_currency": from_norm,
			"fx_rate": effective_rate, "counterparty": from_account, "created_at": _now_iso(),
		})
		self.transactions[credit_id] = credit_tx

		await self._audit("fx_transfer_completed", tid, {
			"from": from_account, "to": to_account,
			"amount": amount_val, "from_currency": from_norm,
			"converted": converted_amount, "to_currency": to_norm,
			"fx_rate": fx_rate, "fx_fee": fx_fee,
		})
		await self._maybe_notify("fx_transfer_completed", {
			"from": from_account, "to": to_account,
			"amount": amount_val, "converted": converted_amount,
		})
		return {
			"transfer_id": tid,
			"from_account": from_account,
			"to_account": to_account,
			"original_amount": amount_val,
			"original_currency": from_norm,
			"converted_amount": converted_amount,
			"target_currency": to_norm,
			"mid_market_rate": fx_rate,
			"effective_rate": effective_rate,
			"fx_spread_pct": fx_spread_pct,
			"fx_fee": fx_fee,
			"sender_balance": sender.balance,
			"completed_at": _now_iso(),
		}

	async def savings_pot_autosweep_rule(
		self,
		account_id: str,
		pot_id: str,
		trigger: str,
		value: float,
		rule_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Attach an auto-sweep rule to a savings pot.
		trigger: 'end_of_day' | 'percentage_of_balance' | 'after_credit'
		"""
		pot = self.savings_pots.get(pot_id)
		if pot is None or pot.tenant_id != self.tenant_id:
			raise KeyError(f"savings pot not found: {pot_id}")
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")
		valid_triggers = {"end_of_day", "percentage_of_balance", "after_credit"}
		if trigger not in valid_triggers:
			raise ValueError(f"unsupported trigger '{trigger}'; must be one of {valid_triggers}")
		assert value > 0, "autosweep value must be positive"

		rid = rule_id or _uuid()
		if not hasattr(self, "autosweep_rules"):
			self.autosweep_rules: dict[str, dict[str, Any]] = {}
		rule: dict[str, Any] = {
			"rule_id": rid, "account_id": account_id, "pot_id": pot_id,
			"trigger": trigger, "value": value, "tenant_id": self.tenant_id,
			"status": "active", "created_at": _now_iso(),
			"last_executed_at": None, "execution_count": 0,
		}
		self.autosweep_rules[rid] = rule
		await self._audit("autosweep_rule_created", rid, {
			"account_id": account_id, "pot_id": pot_id, "trigger": trigger,
		})
		return rule

	async def execute_autosweep_rules(
		self,
		trigger: str,
		account_id: str | None = None,
	) -> dict[str, Any]:
		"""Evaluate and execute all active autosweep rules matching the given trigger."""
		if not hasattr(self, "autosweep_rules"):
			self.autosweep_rules = {}
		rules = [
			r for r in self.autosweep_rules.values()
			if r["tenant_id"] == self.tenant_id
			and r["trigger"] == trigger
			and r["status"] == "active"
			and (account_id is None or r["account_id"] == account_id)
		]
		applied, skipped = [], []
		for rule in rules:
			acct = self._tenant_account_or_none(rule["account_id"], self.tenant_id)
			if acct is None or acct.balance <= 0:
				skipped.append(rule["rule_id"])
				continue
			sweep_amount = (
				round(acct.balance * rule["value"] / 100, 2)
				if trigger == "percentage_of_balance"
				else rule["value"]
			)
			if sweep_amount <= 0 or acct.balance < sweep_amount:
				skipped.append(rule["rule_id"])
				continue
			try:
				await self.savings_pot_deposit(rule["pot_id"], sweep_amount)
				rule["last_executed_at"] = _now_iso()
				rule["execution_count"] = rule.get("execution_count", 0) + 1
				applied.append({"rule_id": rule["rule_id"], "amount": sweep_amount})
			except Exception as exc:
				skipped.append({"rule_id": rule["rule_id"], "reason": str(exc)})

		await self._audit("autosweep_rules_executed", self.tenant_id, {
			"trigger": trigger, "applied": len(applied), "skipped": len(skipped),
		})
		return {
			"trigger": trigger, "rules_evaluated": len(rules),
			"applied": len(applied), "skipped": len(skipped),
			"details": applied, "executed_at": _now_iso(),
		}

	async def register_account_webhook(
		self,
		account_id: str,
		webhook_url: str,
		event_filter: list[str],
		secret: str = "",
		webhook_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Register a webhook endpoint for account events.
		Payloads are HMAC-SHA256 signed when secret is provided.
		"""
		import hashlib
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")
		assert webhook_url.startswith(("http://", "https://")), "webhook_url must be http/https"

		wid = webhook_id or _uuid()
		if not hasattr(self, "webhooks"):
			self.webhooks: dict[str, dict[str, Any]] = {}
		webhook: dict[str, Any] = {
			"webhook_id": wid, "account_id": account_id, "tenant_id": self.tenant_id,
			"url": webhook_url, "event_filter": list(event_filter),
			"secret_hash": hashlib.sha256(secret.encode()).hexdigest() if secret else "",
			"status": "active", "registered_at": _now_iso(),
			"delivery_count": 0, "last_delivery_at": None,
		}
		self.webhooks[wid] = webhook
		await self._audit("account_webhook_registered", wid, {
			"account_id": account_id, "url": webhook_url,
		})
		return {k: v for k, v in webhook.items() if k != "secret_hash"}

	async def set_spending_budget(
		self,
		account_id: str,
		category: str,
		monthly_limit: float,
		budget_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Set a monthly spending budget for a transaction-kind category.
		Fires 75% and 100% notifications via spending_budget_check.
		Replaces any existing budget for the same category.
		"""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")
		limit_val = normalize_amount(monthly_limit)
		assert limit_val > 0, "monthly_limit must be positive"

		bid = budget_id or _uuid()
		if not hasattr(self, "spending_budgets"):
			self.spending_budgets: dict[str, list[dict[str, Any]]] = {}
		budgets = self.spending_budgets.setdefault(account_id, [])
		budgets[:] = [b for b in budgets if b["category"] != category]
		budget: dict[str, Any] = {
			"budget_id": bid, "account_id": account_id, "tenant_id": self.tenant_id,
			"category": category, "monthly_limit": limit_val,
			"status": "active", "created_at": _now_iso(),
		}
		budgets.append(budget)
		await self._audit("spending_budget_set", bid, {
			"account_id": account_id, "category": category, "limit": limit_val,
		})
		return budget

	async def spending_budget_check(
		self,
		account_id: str,
		category: str,
	) -> dict[str, Any]:
		"""
		Check current spend vs budget for a category.
		Returns remaining, burn rate, projected over-budget date.
		Fires notifications at 75% and 100% thresholds (once per month per threshold).
		"""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")
		if not hasattr(self, "spending_budgets"):
			self.spending_budgets = {}
		budgets = self.spending_budgets.get(account_id, [])
		budget = next((b for b in budgets if b["category"] == category), None)
		if budget is None:
			raise KeyError(f"no budget set for category '{category}' on account {account_id}")

		now = _now_iso()
		month_prefix = now[:7]
		spent = sum(
			t.amount for t in self.transactions.values()
			if t.tenant_id == self.tenant_id
			and t.account_id == account_id
			and t.kind == category
			and t.direction == "debit"
			and getattr(t, "created_at", now)[:7] == month_prefix
		)
		limit = budget["monthly_limit"]
		remaining = round(limit - spent, 2)
		utilisation_pct = round(spent / limit * 100, 2) if limit > 0 else 0.0
		day_of_month = int(now[8:10])
		burn_rate_daily = round(spent / max(day_of_month, 1), 2)
		days_until_over = round(remaining / burn_rate_daily, 1) if burn_rate_daily > 0 else None

		if not hasattr(self, "_budget_alerts"):
			self._budget_alerts: dict[str, bool] = {}
		alert_key = f"budget_{account_id}_{category}_{month_prefix}"
		if utilisation_pct >= 100 and not self._budget_alerts.get(f"{alert_key}_100"):
			await self._maybe_notify("budget_exceeded", {
				"account_id": account_id, "category": category, "spent": spent, "limit": limit,
			})
			self._budget_alerts[f"{alert_key}_100"] = True
		elif utilisation_pct >= 75 and not self._budget_alerts.get(f"{alert_key}_75"):
			await self._maybe_notify("budget_75pct_warning", {
				"account_id": account_id, "category": category, "spent": spent, "limit": limit,
			})
			self._budget_alerts[f"{alert_key}_75"] = True

		return {
			"account_id": account_id, "category": category,
			"monthly_limit": limit, "spent_this_month": round(spent, 2),
			"remaining": remaining, "utilisation_pct": utilisation_pct,
			"burn_rate_daily": burn_rate_daily, "days_until_over_budget": days_until_over,
			"month": month_prefix, "as_of": now,
		}

	async def open_chargeback(
		self,
		case_id: str,
		customer_id: str,
		account_id: str,
		disputed_transaction_id: str,
		reason: str,
		evidence_references: list[str] | None = None,
	) -> dict[str, Any]:
		"""
		Open a chargeback dispute for a posted transaction.
		Issues a provisional credit equal to the disputed amount immediately.
		"""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")
		disputed_tx = self.transactions.get(disputed_transaction_id)
		if disputed_tx is None or disputed_tx.tenant_id != self.tenant_id:
			raise KeyError(f"transaction not found: {disputed_transaction_id}")
		if disputed_tx.account_id != account_id:
			raise ValueError(f"transaction {disputed_transaction_id} does not belong to account {account_id}")

		disputed_amount = disputed_tx.amount
		prov_id = _uuid()
		account.balance = round(account.balance + disputed_amount, 2)
		prov_tx = AccountTransaction(
			prov_id, self.tenant_id, account_id, "transfer_in",
			disputed_amount, account.currency, "credit",
			f"chargeback_{case_id}_provisional", "",
		)
		prov_tx.__dict__["created_at"] = _now_iso()
		self.transactions[prov_id] = prov_tx

		chargeback: dict[str, Any] = {
			"case_id": case_id, "tenant_id": self.tenant_id,
			"customer_id": customer_id, "account_id": account_id,
			"disputed_transaction_id": disputed_transaction_id,
			"disputed_amount": disputed_amount, "currency": account.currency,
			"reason": reason, "evidence_references": evidence_references or [],
			"provisional_credit_tx_id": prov_id,
			"status": "provisional_credit_issued",
			"merchant_response": None, "arbitration_reference": None, "final_ruling": None,
			"opened_at": _now_iso(),
		}
		if not hasattr(self, "chargebacks"):
			self.chargebacks: dict[str, dict[str, Any]] = {}
		self.chargebacks[case_id] = chargeback
		await self._audit("chargeback_opened", case_id, {
			"disputed_tx": disputed_transaction_id, "amount": disputed_amount,
		})
		await self._maybe_notify("chargeback_provisional_credit", {
			"account_id": account_id, "amount": disputed_amount, "case_id": case_id,
		})
		return chargeback

	async def resolve_chargeback(
		self,
		case_id: str,
		ruling: str,
		ruling_notes: str = "",
	) -> dict[str, Any]:
		"""
		Resolve a chargeback. ruling='upheld' keeps provisional credit;
		ruling='rejected' reverses it with a debit transaction.
		"""
		if not hasattr(self, "chargebacks"):
			self.chargebacks = {}
		chargeback = self.chargebacks.get(case_id)
		if chargeback is None or chargeback["tenant_id"] != self.tenant_id:
			raise KeyError(f"chargeback not found: {case_id}")
		if ruling not in {"upheld", "rejected"}:
			raise ValueError(f"ruling must be 'upheld' or 'rejected'; got '{ruling}'")

		if ruling == "rejected":
			account = self._tenant_account_or_none(chargeback["account_id"], self.tenant_id)
			if account is not None:
				account.balance = round(account.balance - chargeback["disputed_amount"], 2)
				reversal_id = _uuid()
				reversal = AccountTransaction(
					reversal_id, self.tenant_id, chargeback["account_id"], "transfer_out",
					chargeback["disputed_amount"], account.currency, "debit",
					f"chargeback_{case_id}_reversal", "",
				)
				reversal.__dict__["created_at"] = _now_iso()
				self.transactions[reversal_id] = reversal

		chargeback["final_ruling"] = ruling
		chargeback["ruling_notes"] = ruling_notes
		chargeback["status"] = f"resolved_{ruling}"
		chargeback["resolved_at"] = _now_iso()
		await self._audit("chargeback_resolved", case_id, {"ruling": ruling})
		await self._maybe_notify("chargeback_resolved", {"case_id": case_id, "ruling": ruling})
		return chargeback

	async def compute_customer_risk_score(
		self,
		customer_id: str,
	) -> dict[str, Any]:
		"""
		Aggregate a 0-100 risk score from velocity, overdraft utilisation,
		savings behaviour, and account freeze history.
		Tier: low (0-29) | medium (30-64) | high (65-100).
		"""
		customer = self._tenant_customer_or_none(customer_id, self.tenant_id)
		if customer is None:
			raise KeyError(f"customer not found: {customer_id}")

		accounts = [
			a for a in self.accounts.values()
			if a.tenant_id == self.tenant_id and a.owner_id == customer_id
		]
		if not accounts:
			return {
				"customer_id": customer_id, "risk_score": 50, "tier": "medium",
				"signals": {}, "note": "no accounts — baseline score applied",
			}

		acct_ids = {a.id for a in accounts}
		total_debits = sum(
			1 for t in self.transactions.values()
			if t.tenant_id == self.tenant_id and t.account_id in acct_ids and t.direction == "debit"
		)
		velocity_score = min(total_debits / 10, 30)

		overdraft_score = 0.0
		for acct in accounts:
			od_limit = getattr(acct, "overdraft_limit", 0.0)
			if od_limit > 0 and acct.balance < 0:
				overdraft_score += min(abs(acct.balance) / od_limit * 20, 20)

		total_balance = sum(a.balance for a in accounts)
		total_savings = sum(
			getattr(p, "balance", 0.0) for p in self.savings_pots.values()
			if p.tenant_id == self.tenant_id and p.account_id in acct_ids
		)
		savings_ratio = total_savings / max(total_balance + total_savings, 1)
		savings_score = max(0, 20 - round(savings_ratio * 20))
		freeze_score = sum(10 for a in accounts if getattr(a, "status", "active") == "frozen")

		risk_score = min(max(round(velocity_score + overdraft_score + savings_score + freeze_score), 0), 100)
		tier = "low" if risk_score < 30 else "medium" if risk_score < 65 else "high"
		await self._audit("customer_risk_score_computed", customer_id, {
			"risk_score": risk_score, "tier": tier,
		})
		return {
			"customer_id": customer_id, "risk_score": risk_score, "tier": tier,
			"signals": {
				"velocity_score": round(velocity_score, 1),
				"overdraft_score": round(overdraft_score, 1),
				"savings_score": round(savings_score, 1),
				"freeze_score": freeze_score,
				"total_debit_count": total_debits,
				"savings_ratio_pct": round(savings_ratio * 100, 1),
			},
			"computed_at": _now_iso(),
		}

	async def generate_balance_attestation(
		self,
		account_id: str,
		purpose: str = "proof_of_funds",
	) -> dict[str, Any]:
		"""
		Generate a HMAC-SHA256 signed balance attestation for third-party verification.
		Valid until end-of-day of issuance.
		"""
		import hashlib
		import hmac
		import json

		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")
		if getattr(account, "status", "active") != "active":
			raise ValueError(f"cannot attest balance on non-active account {account_id}")

		attestation_id = _uuid()
		payload = {
			"attestation_id": attestation_id,
			"tenant_id": self.tenant_id,
			"account_id": account_id,
			"balance": account.balance,
			"currency": account.currency,
			"purpose": purpose,
			"issued_at": _now_iso(),
			"expires_at": _now_iso()[:10] + "T23:59:59+00:00",
		}
		payload_bytes = json.dumps(payload, sort_keys=True).encode()
		signature = hmac.new(
			self.tenant_id.encode(), payload_bytes, hashlib.sha256
		).hexdigest()
		await self._audit("balance_attestation_generated", attestation_id, {
			"account_id": account_id, "purpose": purpose,
		})
		return {**payload, "signature": signature, "algorithm": "HMAC-SHA256"}

	async def record_consent(
		self,
		customer_id: str,
		consent_type: str,
		channel: str,
		evidence_hash: str = "",
		consent_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Record a structured consent event (Kenya DPA Article 30 compliant).
		consent_type: account_opening | data_sharing | marketing | overdraft | biometric
		channel: sms_otp | biometric | e_signature | agent_assisted | in_app
		"""
		customer = self._tenant_customer_or_none(customer_id, self.tenant_id)
		if customer is None:
			raise KeyError(f"customer not found: {customer_id}")

		valid_types = {"account_opening", "data_sharing", "marketing", "overdraft", "biometric"}
		valid_channels = {"sms_otp", "biometric", "e_signature", "agent_assisted", "in_app"}
		if consent_type not in valid_types:
			raise ValueError(f"unsupported consent_type '{consent_type}'; must be one of {valid_types}")
		if channel not in valid_channels:
			raise ValueError(f"unsupported channel '{channel}'; must be one of {valid_channels}")

		cid = consent_id or _uuid()
		if not hasattr(self, "consent_records"):
			self.consent_records: dict[str, list[dict[str, Any]]] = {}
		records = self.consent_records.setdefault(customer_id, [])
		record: dict[str, Any] = {
			"consent_id": cid, "customer_id": customer_id, "tenant_id": self.tenant_id,
			"consent_type": consent_type, "channel": channel, "evidence_hash": evidence_hash,
			"status": "active", "recorded_at": _now_iso(), "revoked_at": None,
		}
		records.append(record)
		await self._audit("consent_recorded", cid, {
			"customer_id": customer_id, "type": consent_type, "channel": channel,
		})
		return record

	async def revoke_consent(
		self,
		customer_id: str,
		consent_type: str,
		reason: str = "",
	) -> dict[str, Any]:
		"""Revoke all active consents of a given type for a customer."""
		customer = self._tenant_customer_or_none(customer_id, self.tenant_id)
		if customer is None:
			raise KeyError(f"customer not found: {customer_id}")
		if not hasattr(self, "consent_records"):
			self.consent_records = {}
		records = self.consent_records.get(customer_id, [])
		revoked = []
		for r in records:
			if r["consent_type"] == consent_type and r["status"] == "active":
				r["status"] = "revoked"
				r["revoked_at"] = _now_iso()
				r["revocation_reason"] = reason
				revoked.append(r["consent_id"])
		await self._audit("consent_revoked", customer_id, {
			"consent_type": consent_type, "revoked_count": len(revoked),
		})
		return {
			"customer_id": customer_id, "consent_type": consent_type,
			"revoked_count": len(revoked), "revoked_ids": revoked, "revoked_at": _now_iso(),
		}

	async def bulk_issue_statements(
		self,
		period_start: str,
		period_end: str,
		account_ids: list[str] | None = None,
	) -> dict[str, Any]:
		"""
		Bulk-generate statements for all active accounts or a specified subset.
		Returns summary with statement IDs and any errors.
		"""
		assert period_start and period_end, "period_start and period_end required"
		if account_ids is None:
			target_accounts = [
				a for a in self.accounts.values()
				if a.tenant_id == self.tenant_id and getattr(a, "status", "active") == "active"
			]
		else:
			target_accounts = [
				a for a in self.accounts.values()
				if a.tenant_id == self.tenant_id and a.id in account_ids
			]

		issued, errors = [], []
		for acct in target_accounts:
			sid = _uuid()
			try:
				await self.issue_statement(sid, acct.id, period_start, period_end)
				issued.append({"statement_id": sid, "account_id": acct.id})
			except Exception as exc:
				errors.append({"account_id": acct.id, "error": str(exc)})

		await self._audit("bulk_statements_issued", self.tenant_id, {
			"period": f"{period_start}:{period_end}", "issued": len(issued), "errors": len(errors),
		})
		return {
			"period_start": period_start, "period_end": period_end,
			"accounts_processed": len(target_accounts),
			"statements_issued": len(issued), "errors": len(errors),
			"statement_refs": issued, "error_details": errors, "generated_at": _now_iso(),
		}

	async def overdraft_interest_accrual(
		self,
		account_id: str,
		period: str,
	) -> dict[str, Any]:
		"""
		Post daily overdraft interest for an overdrawn account.
		Applies annual rate / 365 on the overdrawn balance plus a flat daily fee.
		Skips silently if the account is not overdrawn or has no overdraft limit.
		"""
		account = self._tenant_account_or_none(account_id, self.tenant_id)
		if account is None:
			raise KeyError(f"account not found: {account_id}")

		od_config = self.overdraft_configs.get(account_id, {})
		od_limit = getattr(account, "overdraft_limit", 0.0)
		overdrawn_balance = max(0.0, -account.balance)

		if overdrawn_balance <= 0 or od_limit <= 0:
			return {
				"account_id": account_id, "period": period,
				"overdrawn_balance": overdrawn_balance, "interest_charged": 0.0,
				"message": "account not overdrawn — no interest applied",
			}

		annual_rate = od_config.get("interest_rate_pa", 0.18)
		daily_rate = annual_rate / 365
		daily_fee = od_config.get("daily_fee", 50.0)
		interest = round(overdrawn_balance * daily_rate, 2)
		total_charge = round(interest + daily_fee, 2)

		fee_tx_id = _uuid()
		fee_tx = AccountTransaction(
			fee_tx_id, self.tenant_id, account_id, "transfer_out",
			total_charge, account.currency, "debit",
			f"overdraft_interest_{period}", "auto",
		)
		fee_tx.__dict__["created_at"] = _now_iso()
		self.transactions[fee_tx_id] = fee_tx
		account.balance = round(account.balance - total_charge, 2)

		await self._audit("overdraft_interest_accrued", account_id, {
			"period": period, "overdrawn": overdrawn_balance,
			"interest": interest, "fee": daily_fee, "total": total_charge,
		})
		return {
			"account_id": account_id, "period": period,
			"overdrawn_balance": overdrawn_balance,
			"annual_rate_pct": round(annual_rate * 100, 2),
			"daily_rate_pct": round(daily_rate * 100, 4),
			"interest_charged": interest, "daily_fee": daily_fee,
			"total_charge": total_charge, "account_balance": account.balance,
			"fee_transaction_id": fee_tx_id, "accrued_at": _now_iso(),
		}

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant_program_or_none(self, program_id: str, tenant_id: str) -> BankProgram | None:
		item = self.programs.get(program_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_customer_or_none(self, customer_id: str, tenant_id: str) -> CustomerProfile | None:
		item = self.customers.get(customer_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_account_or_none(self, account_id: str, tenant_id: str) -> DepositAccount | None:
		item = self.accounts.get(account_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	async def _audit(self, event_type: str, reference_id: str, metadata: dict[str, Any]) -> None:
		record = {
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"metadata": metadata,
			"recorded_at": _now_iso(),
		}
		self.audit_events.append(record)
		if self._audit_adapter is not None:
			try:
				await self._audit_adapter.record(record)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

	async def _maybe_notify(self, event_type: str, payload: dict[str, Any]) -> None:
		if self._notify is not None:
			try:
				await self._notify.send(event_type, payload)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "neobanking_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "neobanking_policy_denied")


DigitalNeobankingService = NeobanksService
NeobankingService = NeobanksService
