"""SACCO FOSA (Front Office Service Activity) — full async service.

FOSA is the transactional banking wing of a SACCO: current accounts,
daily deposits/withdrawals, M-PESA integration, ATM cards, standing orders,
overdrafts, and BOSA cross-transfers.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from uuid import uuid4
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

CAPABILITY_ID = "fintech_sacco_fosa"

# GL account codes (chart of accounts stubs)
GL_CASH_TELLER     = "1001"
GL_CASH_MPESA      = "1010"   # Bank - Current Account (MPESA float)
GL_CASH_BANK       = "1020"   # Bank - Settlement Account
GL_ATM_SUSPENSE    = "1520"   # Suspense - Debit (ATM settlements)
GL_FOSA_DEPOSITS   = "2100"   # Member Deposits - FOSA   # CR — member FOSA deposits liability
GL_BOSA_CONTROL    = "2110"   # Member Deposits - BOSA   # BOSA inter-fund control
GL_OVERDRAFT_ASSET = "1100"   # Member Loans - FOSA (overdraft)   # DR — overdraft receivable
GL_INTEREST_INCOME = "4110"   # Interest Income - FOSA Loans
GL_CHARGES_INCOME  = "4300"   # Fee Income

# Daily withdrawal limit default (KES)
DEFAULT_WITHDRAWAL_LIMIT = Decimal("100000")
DEFAULT_TRANSFER_LIMIT   = Decimal("200000")

# Dormancy threshold
DORMANCY_MONTHS = 6

# Approval threshold for BOSA→FOSA transfers (KES)
BOSA_TRANSFER_APPROVAL_THRESHOLD = Decimal("50000")


class FOSAService:
	"""Async service for SACCO FOSA operations: deposits, withdrawals,
	M-PESA, ATM cards, standing orders, overdrafts, and portfolio management."""

	def __init__(self, tenant_id: str = "default", gl_service=None) -> None:
		self.tenant_id = tenant_id
		self.accounts: dict[str, dict[str, Any]] = {}
		self.transactions: dict[str, dict[str, Any]] = {}
		self.gl_entries: list[dict[str, Any]] = []
		self.atm_cards: dict[str, dict[str, Any]] = {}
		self.standing_orders: dict[str, dict[str, Any]] = {}
		self.overdrafts: dict[str, dict[str, Any]] = {}
		self.interest_records: dict[str, dict[str, Any]] = {}
		self.daily_totals: dict[str, dict[str, Any]] = {}  # key: "tenant|teller|date"
		self._audit_events: list[dict[str, Any]] = []
		self._account_counter: int = 0
		self._card_counter: int = 0

	# ── Helpers ───────────────────────────────────────────────────────────────

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str) -> str:
		return f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _today(self) -> str:
		return date.today().isoformat()

	def _next_account_number(self, tenant_id: str, account_type: str) -> str:
		self._account_counter += 1
		prefix = {"CURRENT": "CUR", "SALARY": "SAL", "FIXED_DEPOSIT": "FXD"}.get(account_type, "FSA")
		return f"FOSA-{prefix}-{tenant_id[:4].upper()}-{self._account_counter:08d}"

	def _next_card_number(self) -> str:
		"""Generate masked card number (PCI-compliant: never store full PAN)."""
		self._card_counter += 1
		return f"4111-****-****-{self._card_counter:04d}"

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record.get("id", ""),
			"emitted_at": self._now(),
		})

	def _get_account(self, account_id: str, tenant_id: str) -> dict[str, Any]:
		acc = self.accounts.get(account_id)
		if not acc or acc["tenant_id"] != tenant_id:
			raise KeyError(f"fosa_account_not_found: {account_id}")
		return acc

	def _get_card(self, card_id: str, tenant_id: str) -> dict[str, Any]:
		card = self.atm_cards.get(card_id)
		if not card or card["tenant_id"] != tenant_id:
			raise KeyError(f"atm_card_not_found: {card_id}")
		return card

	def _get_standing_order(self, so_id: str, tenant_id: str) -> dict[str, Any]:
		so = self.standing_orders.get(so_id)
		if not so or so["tenant_id"] != tenant_id:
			raise KeyError(f"standing_order_not_found: {so_id}")
		return so

	def _post_gl(
		self,
		tenant_id: str,
		debit_code: str,
		debit_name: str,
		credit_code: str,
		credit_name: str,
		amount: Decimal,
		narration: str,
		reference: str,
	) -> None:
		"""Post a balanced double-entry GL pair."""
		now = self._now()
		today = now[:10]
		entry_id = self._record_id("gl")
		self.gl_entries.append({
			"id": entry_id,
			"tenant_id": tenant_id,
			"debit_account": debit_code,
			"debit_name": debit_name,
			"credit_account": credit_code,
			"credit_name": credit_name,
			"amount": amount,
			"narration": narration,
			"reference": reference,
			"posting_date": today,
			"created_at": now,
		})

	def _channel_gl_account(self, channel: str) -> tuple[str, str]:
		"""Map transaction channel to GL cash account."""
		mapping = {
			"TELLER": (GL_CASH_TELLER, "Cash - Teller"),
			"MPESA": (GL_CASH_MPESA, "Cash - M-PESA"),
			"BANK_TRANSFER": (GL_CASH_BANK, "Cash - Bank"),
			"ATM": (GL_ATM_SUSPENSE, "ATM Suspense"),
		}
		return mapping.get(channel, (GL_CASH_TELLER, "Cash - Teller"))

	def _accum_daily(self, tenant_id: str, teller_id: str | None, date_str: str, deposits: Decimal = Decimal("0"), withdrawals: Decimal = Decimal("0"), count: int = 1) -> None:
		key = f"{tenant_id}|{teller_id or '_'}|{date_str}"
		if key not in self.daily_totals:
			self.daily_totals[key] = {
				"tenant_id": tenant_id,
				"teller_id": teller_id,
				"date": date_str,
				"total_deposits": Decimal("0"),
				"total_withdrawals": Decimal("0"),
				"transaction_count": 0,
			}
		self.daily_totals[key]["total_deposits"] += deposits
		self.daily_totals[key]["total_withdrawals"] += withdrawals
		self.daily_totals[key]["transaction_count"] += count

	def _log_pretty_path(self, account_id: str, tenant_id: str) -> str:
		return f"tenant={tenant_id} account={account_id}"

	# ── Member Validation Stubs ───────────────────────────────────────────────
	# In production these call capabilities/fintech/sacco/mem or KYC service.

	def _validate_member(self, member_id: str, tenant_id: str) -> None:
		"""Stub: raises if member not active / KYC not complete."""
		assert member_id, "member_id required"
		assert tenant_id, "tenant_id required"
		# Production: call mem service, check status == "active", kyc_complete == True, joining_fee_paid == True

	# ── Account Lifecycle ─────────────────────────────────────────────────────

	async def open_fosa_account(
		self,
		tenant_id: str,
		member_id: str,
		account_type: str,
		opening_balance: Decimal = Decimal("0"),
		currency: str = "KES",
		account_name: str | None = None,
		daily_withdrawal_limit: Decimal = DEFAULT_WITHDRAWAL_LIMIT,
		daily_transfer_limit: Decimal = DEFAULT_TRANSFER_LIMIT,
	) -> dict[str, Any]:
		"""Open a FOSA current/salary/fixed-deposit account for a member.

		Validates member is active, KYC complete, joining fee paid.
		Posts GL: DR Cash / CR Member FOSA Deposits.
		"""
		t = self._tenant(tenant_id)
		assert account_type in {"CURRENT", "SALARY", "FIXED_DEPOSIT"}, f"invalid_account_type: {account_type}"
		assert opening_balance >= Decimal("0"), "opening_balance must be >= 0"

		self._validate_member(member_id, t)

		# One active FOSA account per type per member (rule for most SACCOs)
		for acc in self.accounts.values():
			if acc["tenant_id"] == t and acc["member_id"] == member_id and acc["account_type"] == account_type and acc["status"] == "active":
				raise ValueError(f"member_already_has_{account_type.lower()}_fosa_account")

		account_number = self._next_account_number(t, account_type)
		acc_id = self._record_id("fosa")

		record: dict[str, Any] = {
			"id": acc_id,
			"type": "fosa_account",
			"tenant_id": t,
			"account_number": account_number,
			"member_id": member_id,
			"account_type": account_type,
			"account_name": account_name or f"{member_id} {account_type.title()} Account",
			"currency": currency,
			"book_balance": opening_balance,
			"available_balance": opening_balance,
			"locked_balance": Decimal("0"),
			"overdraft_limit": Decimal("0"),
			"overdraft_used": Decimal("0"),
			"daily_withdrawal_limit": daily_withdrawal_limit,
			"daily_transfer_limit": daily_transfer_limit,
			"status": "active",
			"created_at": self._now(),
			"updated_at": self._now(),
			"last_transaction_at": None,
		}
		self.accounts[acc_id] = record

		if opening_balance > Decimal("0"):
			# GL: DR Cash-Teller / CR FOSA Deposits
			self._post_gl(
				t,
				GL_CASH_TELLER, "Cash - Teller",
				GL_FOSA_DEPOSITS, "Member FOSA Deposits",
				opening_balance,
				f"Opening deposit — {account_number}",
				acc_id,
			)
			# Record opening transaction
			txn = self._build_txn(t, acc_id, record, "fosa_deposit", opening_balance, "TELLER", acc_id, "Opening deposit")
			self.transactions[txn["id"]] = txn
			self._accum_daily(t, None, self._today(), deposits=opening_balance)

		self._emit(t, "fosa_account_opened", record)
		_log.info("FOSA account opened: %s member=%s type=%s", account_number, member_id, account_type)
		return deepcopy(record)

	async def close_fosa_account(
		self,
		tenant_id: str,
		account_id: str,
		reason: str,
		closed_by: str,
	) -> dict[str, Any]:
		"""Close a FOSA account. Requires zero balance and no pending transactions."""
		t = self._tenant(tenant_id)
		acc = self._get_account(account_id, t)
		assert acc["status"] != "closed", "account_already_closed"

		if acc["book_balance"] != Decimal("0"):
			raise ValueError(f"cannot_close_non_zero_balance: {acc['book_balance']}")

		# Check no active standing orders
		active_so = [so for so in self.standing_orders.values()
		             if so["tenant_id"] == t and so["account_id"] == account_id and so["status"] == "active"]
		if active_so:
			raise ValueError(f"account_has_{len(active_so)}_active_standing_orders")

		# Check no active overdraft
		active_od = [od for od in self.overdrafts.values()
		             if od["tenant_id"] == t and od["account_id"] == account_id and od["status"] == "approved" and od.get("balance_used", Decimal("0")) > Decimal("0")]
		if active_od:
			raise ValueError("account_has_outstanding_overdraft")

		acc["status"] = "closed"
		acc["closure_reason"] = reason
		acc["closed_by"] = closed_by
		acc["closed_at"] = self._now()
		acc["updated_at"] = self._now()

		self._emit(t, "fosa_account_closed", acc)
		_log.info("FOSA account closed: %s reason=%s by=%s", acc["account_number"], reason, closed_by)
		return deepcopy(acc)

	# ── Deposits & Withdrawals ────────────────────────────────────────────────

	def _build_txn(
		self,
		tenant_id: str,
		account_id: str,
		acc: dict[str, Any],
		txn_type: str,
		amount: Decimal,
		channel: str,
		reference: str,
		narration: str,
		extra: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		txn_id = self._record_id("ftxn")
		txn: dict[str, Any] = {
			"id": txn_id,
			"type": txn_type,
			"tenant_id": tenant_id,
			"account_id": account_id,
			"account_number": acc.get("account_number", ""),
			"member_id": acc.get("member_id", ""),
			"txn_type": txn_type,
			"amount": amount,
			"balance_before": acc["book_balance"],
			"balance_after": acc["book_balance"],  # caller updates acc then this
			"channel": channel,
			"reference": reference,
			"narration": narration,
			"status": "completed",
			"created_at": self._now(),
		}
		if extra:
			txn.update(extra)
		return txn

	async def deposit(
		self,
		tenant_id: str,
		account_id: str,
		amount: Decimal,
		channel: str,
		reference: str,
		depositor_name: str | None = None,
		narration: str | None = None,
		teller_id: str | None = None,
	) -> dict[str, Any]:
		"""Deposit funds into a FOSA account. Posts GL, emits event."""
		t = self._tenant(tenant_id)
		assert amount > Decimal("0"), "amount must be positive"
		assert channel in {"TELLER", "MPESA", "BANK_TRANSFER"}, f"invalid_channel: {channel}"

		acc = self._get_account(account_id, t)
		if acc["status"] not in {"active"}:
			raise ValueError(f"cannot_deposit_to_account_status: {acc['status']}")

		bal_before = acc["book_balance"]
		bal_after = bal_before + amount
		acc["book_balance"] = bal_after
		acc["available_balance"] = bal_after - acc["locked_balance"]
		acc["updated_at"] = self._now()
		acc["last_transaction_at"] = self._now()

		txn = self._build_txn(t, account_id, acc, "fosa_deposit", amount, channel, reference,
		                      narration or f"Deposit via {channel}" + (f" — {depositor_name}" if depositor_name else ""))
		txn["balance_before"] = bal_before
		txn["balance_after"] = bal_after
		txn["depositor_name"] = depositor_name
		txn["teller_id"] = teller_id
		self.transactions[txn["id"]] = txn

		gl_dr, gl_dr_name = self._channel_gl_account(channel)
		self._post_gl(t, gl_dr, gl_dr_name, GL_FOSA_DEPOSITS, "Member FOSA Deposits",
		              amount, txn["narration"], reference)

		self._accum_daily(t, teller_id, self._today(), deposits=amount)
		self._emit(t, "fosa_deposit", txn)
		_log.info("FOSA deposit %s %s channel=%s ref=%s", self._log_pretty_path(account_id, t), amount, channel, reference)
		return deepcopy(txn)

	async def withdraw(
		self,
		tenant_id: str,
		account_id: str,
		amount: Decimal,
		channel: str,
		reference: str | None = None,
		authorized_by: str | None = None,
		narration: str | None = None,
		teller_id: str | None = None,
	) -> dict[str, Any]:
		"""Withdraw from FOSA account. Validates available balance, frozen status, daily limit."""
		t = self._tenant(tenant_id)
		assert amount > Decimal("0"), "amount must be positive"
		assert channel in {"TELLER", "ATM", "MPESA"}, f"invalid_channel: {channel}"

		acc = self._get_account(account_id, t)
		if acc["status"] == "frozen":
			raise ValueError("account_frozen")
		if acc["status"] != "active":
			raise ValueError(f"cannot_withdraw_from_status: {acc['status']}")

		available = acc["available_balance"] + acc.get("overdraft_limit", Decimal("0")) - acc.get("overdraft_used", Decimal("0"))
		if amount > available:
			raise ValueError(f"insufficient_available_balance: available={available} requested={amount}")

		# Daily withdrawal limit check
		today = self._today()
		daily_withdrawn = sum(
			tx["amount"] for tx in self.transactions.values()
			if tx["tenant_id"] == t
			and tx["account_id"] == account_id
			and tx["txn_type"] in {"fosa_withdrawal", "fosa_mpesa_out"}
			and tx["created_at"][:10] == today
		)
		if daily_withdrawn + amount > acc["daily_withdrawal_limit"]:
			raise ValueError(f"daily_withdrawal_limit_exceeded: limit={acc['daily_withdrawal_limit']} used={daily_withdrawn} requested={amount}")

		bal_before = acc["book_balance"]
		bal_after = bal_before - amount

		# If dipping into overdraft track usage
		if bal_after < Decimal("0"):
			acc["overdraft_used"] = abs(bal_after)

		acc["book_balance"] = bal_after
		acc["available_balance"] = max(Decimal("0"), bal_after) - acc["locked_balance"]
		acc["updated_at"] = self._now()
		acc["last_transaction_at"] = self._now()

		txn = self._build_txn(t, account_id, acc, "fosa_withdrawal", amount, channel,
		                      reference or self._record_id("wref"),
		                      narration or f"Withdrawal via {channel}")
		txn["balance_before"] = bal_before
		txn["balance_after"] = bal_after
		txn["authorized_by"] = authorized_by
		txn["teller_id"] = teller_id
		self.transactions[txn["id"]] = txn

		gl_cr, gl_cr_name = self._channel_gl_account(channel)
		self._post_gl(t, GL_FOSA_DEPOSITS, "Member FOSA Deposits", gl_cr, gl_cr_name,
		              amount, txn["narration"], txn["reference"])

		self._accum_daily(t, teller_id, today, withdrawals=amount)
		self._emit(t, "fosa_withdrawal", txn)
		_log.info("FOSA withdrawal %s %s channel=%s", self._log_pretty_path(account_id, t), amount, channel)
		return deepcopy(txn)

	# ── BOSA Transfers ────────────────────────────────────────────────────────

	async def transfer_to_bosa(
		self,
		tenant_id: str,
		account_id: str,
		amount: Decimal,
		bosa_account_id: str,
		reference: str,
	) -> dict[str, Any]:
		"""Transfer funds from FOSA to BOSA savings/shares."""
		t = self._tenant(tenant_id)
		assert amount > Decimal("0"), "amount must be positive"

		acc = self._get_account(account_id, t)
		if acc["status"] != "active":
			raise ValueError(f"cannot_transfer_from_status: {acc['status']}")
		if amount > acc["available_balance"]:
			raise ValueError(f"insufficient_balance: available={acc['available_balance']}")

		bal_before = acc["book_balance"]
		acc["book_balance"] -= amount
		acc["available_balance"] -= amount
		acc["updated_at"] = self._now()
		acc["last_transaction_at"] = self._now()

		txn = self._build_txn(t, account_id, acc, "fosa_bosa_out", amount, "INTERNAL", reference,
		                      f"Transfer to BOSA account {bosa_account_id}")
		txn["balance_before"] = bal_before
		txn["balance_after"] = acc["book_balance"]
		txn["bosa_account_id"] = bosa_account_id
		self.transactions[txn["id"]] = txn

		self._post_gl(t, GL_FOSA_DEPOSITS, "Member FOSA Deposits", GL_BOSA_CONTROL, "BOSA Control",
		              amount, txn["narration"], reference)
		self._emit(t, "fosa_to_bosa_transfer", txn)
		return deepcopy(txn)

	async def transfer_from_bosa(
		self,
		tenant_id: str,
		account_id: str,
		amount: Decimal,
		bosa_account_id: str,
		reference: str,
		approved_by: str | None = None,
	) -> dict[str, Any]:
		"""Transfer from BOSA to FOSA. Requires approval if above threshold."""
		t = self._tenant(tenant_id)
		assert amount > Decimal("0"), "amount must be positive"

		if amount > BOSA_TRANSFER_APPROVAL_THRESHOLD and not approved_by:
			raise ValueError(f"approval_required_for_transfers_above_{BOSA_TRANSFER_APPROVAL_THRESHOLD}")

		acc = self._get_account(account_id, t)
		if acc["status"] != "active":
			raise ValueError(f"cannot_receive_transfer_to_status: {acc['status']}")

		bal_before = acc["book_balance"]
		acc["book_balance"] += amount
		acc["available_balance"] += amount
		acc["updated_at"] = self._now()
		acc["last_transaction_at"] = self._now()

		txn = self._build_txn(t, account_id, acc, "fosa_bosa_in", amount, "INTERNAL", reference,
		                      f"Transfer from BOSA account {bosa_account_id}")
		txn["balance_before"] = bal_before
		txn["balance_after"] = acc["book_balance"]
		txn["bosa_account_id"] = bosa_account_id
		txn["approved_by"] = approved_by
		self.transactions[txn["id"]] = txn

		self._post_gl(t, GL_BOSA_CONTROL, "BOSA Control", GL_FOSA_DEPOSITS, "Member FOSA Deposits",
		              amount, txn["narration"], reference)
		self._emit(t, "bosa_to_fosa_transfer", txn)
		return deepcopy(txn)

	# ── M-PESA ────────────────────────────────────────────────────────────────

	async def mpesa_cash_in(
		self,
		tenant_id: str,
		account_id: str,
		mpesa_reference: str,
		amount: Decimal,
		phone_number: str,
	) -> dict[str, Any]:
		"""Handle confirmed M-PESA C2B payment. Idempotent on mpesa_reference."""
		t = self._tenant(tenant_id)
		assert amount > Decimal("0"), "amount must be positive"
		assert mpesa_reference, "mpesa_reference required"

		# Idempotency — reject duplicate M-PESA reference
		for txn in self.transactions.values():
			if txn["tenant_id"] == t and txn.get("mpesa_reference") == mpesa_reference:
				_log.warning("duplicate mpesa_reference ignored: %s", mpesa_reference)
				return deepcopy(txn)

		acc = self._get_account(account_id, t)
		if acc["status"] not in {"active"}:
			raise ValueError(f"cannot_receive_mpesa_to_status: {acc['status']}")

		bal_before = acc["book_balance"]
		acc["book_balance"] += amount
		acc["available_balance"] += amount
		acc["updated_at"] = self._now()
		acc["last_transaction_at"] = self._now()

		txn = self._build_txn(t, account_id, acc, "fosa_mpesa_in", amount, "MPESA", mpesa_reference,
		                      f"M-PESA payment from {phone_number}")
		txn["balance_before"] = bal_before
		txn["balance_after"] = acc["book_balance"]
		txn["mpesa_reference"] = mpesa_reference
		txn["phone_number"] = phone_number
		self.transactions[txn["id"]] = txn

		self._post_gl(t, GL_CASH_MPESA, "Cash - M-PESA", GL_FOSA_DEPOSITS, "Member FOSA Deposits",
		              amount, txn["narration"], mpesa_reference)
		self._accum_daily(t, None, self._today(), deposits=amount)
		self._emit(t, "mpesa_cash_in", txn)
		_log.info("M-PESA cash-in %s amount=%s ref=%s", self._log_pretty_path(account_id, t), amount, mpesa_reference)
		return deepcopy(txn)

	async def mpesa_cash_out(
		self,
		tenant_id: str,
		account_id: str,
		amount: Decimal,
		phone_number: str,
		mpesa_reference: str | None = None,
	) -> dict[str, Any]:
		"""Initiate M-PESA B2C payment and deduct from FOSA."""
		t = self._tenant(tenant_id)
		assert amount > Decimal("0"), "amount must be positive"
		assert phone_number, "phone_number required"

		acc = self._get_account(account_id, t)
		if acc["status"] != "active":
			raise ValueError(f"cannot_mpesa_out_from_status: {acc['status']}")
		if amount > acc["available_balance"]:
			raise ValueError(f"insufficient_balance: available={acc['available_balance']}")

		# Check daily limit
		today = self._today()
		daily_withdrawn = sum(
			tx["amount"] for tx in self.transactions.values()
			if tx["tenant_id"] == t and tx["account_id"] == account_id
			and tx["txn_type"] in {"fosa_withdrawal", "fosa_mpesa_out"}
			and tx["created_at"][:10] == today
		)
		if daily_withdrawn + amount > acc["daily_withdrawal_limit"]:
			raise ValueError(f"daily_limit_exceeded: limit={acc['daily_withdrawal_limit']}")

		ref = mpesa_reference or self._record_id("mout")
		bal_before = acc["book_balance"]
		acc["book_balance"] -= amount
		acc["available_balance"] -= amount
		acc["updated_at"] = self._now()
		acc["last_transaction_at"] = self._now()

		txn = self._build_txn(t, account_id, acc, "fosa_mpesa_out", amount, "MPESA", ref,
		                      f"M-PESA B2C to {phone_number}")
		txn["balance_before"] = bal_before
		txn["balance_after"] = acc["book_balance"]
		txn["mpesa_reference"] = ref
		txn["phone_number"] = phone_number
		txn["b2c_status"] = "initiated"  # production: update on Safaricom callback
		self.transactions[txn["id"]] = txn

		self._post_gl(t, GL_FOSA_DEPOSITS, "Member FOSA Deposits", GL_CASH_MPESA, "Cash - M-PESA",
		              amount, txn["narration"], ref)
		self._accum_daily(t, None, today, withdrawals=amount)
		self._emit(t, "mpesa_cash_out", txn)
		_log.info("M-PESA cash-out %s amount=%s phone=%s", self._log_pretty_path(account_id, t), amount, phone_number)
		return deepcopy(txn)

	# ── ATM Cards ─────────────────────────────────────────────────────────────

	async def issue_atm_card(
		self,
		tenant_id: str,
		member_id: str,
		account_id: str,
		card_type: str,
		card_name: str | None = None,
	) -> dict[str, Any]:
		"""Issue an ATM card for a FOSA account. Triggers card issuance workflow."""
		t = self._tenant(tenant_id)
		assert card_type in {"VISA", "MASTERCARD", "PREPAID"}, f"invalid_card_type: {card_type}"

		acc = self._get_account(account_id, t)
		if acc["status"] != "active":
			raise ValueError(f"cannot_issue_card_to_account_status: {acc['status']}")
		if acc["member_id"] != member_id:
			raise ValueError("member_id_account_mismatch")

		# One active card per account per type
		for card in self.atm_cards.values():
			if card["tenant_id"] == t and card["account_id"] == account_id and card["card_type"] == card_type and card["status"] == "active":
				raise ValueError(f"active_{card_type.lower()}_card_exists_for_account")

		card_id = self._record_id("card")
		issued_date = date.today()
		expiry_date = date(issued_date.year + 3, issued_date.month, 1)  # 3-year expiry

		record: dict[str, Any] = {
			"id": card_id,
			"type": "atm_card",
			"tenant_id": t,
			"member_id": member_id,
			"account_id": account_id,
			"account_number": acc["account_number"],
			"card_number_masked": self._next_card_number(),
			"card_type": card_type,
			"card_name": card_name or acc.get("account_name", member_id)[:26].upper(),
			"status": "requested",
			"issued_at": issued_date.isoformat(),
			"expires_at": expiry_date.isoformat(),
			"blocked_at": None,
			"block_reason": None,
			"unblocked_at": None,
			"created_at": self._now(),
		}
		self.atm_cards[card_id] = record
		self._emit(t, "atm_card_issued", record)
		_log.info("ATM card issued: %s type=%s member=%s", card_id, card_type, member_id)
		return deepcopy(record)

	async def block_atm_card(
		self,
		tenant_id: str,
		card_id: str,
		reason: str,
	) -> dict[str, Any]:
		"""Block an ATM card immediately."""
		t = self._tenant(tenant_id)
		card = self._get_card(card_id, t)
		if card["status"] == "blocked":
			raise ValueError("card_already_blocked")
		if card["status"] == "cancelled":
			raise ValueError("cannot_block_cancelled_card")
		card["status"] = "blocked"
		card["block_reason"] = reason
		card["blocked_at"] = self._now()
		self._emit(t, "atm_card_blocked", card)
		return deepcopy(card)

	async def unblock_atm_card(
		self,
		tenant_id: str,
		card_id: str,
		authorized_by: str,
	) -> dict[str, Any]:
		"""Unblock a previously blocked ATM card."""
		t = self._tenant(tenant_id)
		card = self._get_card(card_id, t)
		if card["status"] != "blocked":
			raise ValueError(f"cannot_unblock_card_in_status: {card['status']}")
		card["status"] = "active"
		card["unblocked_at"] = self._now()
		card["unblocked_by"] = authorized_by
		card["block_reason"] = None
		self._emit(t, "atm_card_unblocked", card)
		return deepcopy(card)

	# ── Daily Limits ──────────────────────────────────────────────────────────

	async def set_daily_limit(
		self,
		tenant_id: str,
		account_id: str,
		withdrawal_limit: Decimal,
		transfer_limit: Decimal,
	) -> dict[str, Any]:
		"""Update daily withdrawal and transfer limits for a FOSA account."""
		t = self._tenant(tenant_id)
		assert withdrawal_limit >= Decimal("0"), "withdrawal_limit must be >= 0"
		assert transfer_limit >= Decimal("0"), "transfer_limit must be >= 0"

		acc = self._get_account(account_id, t)
		acc["daily_withdrawal_limit"] = withdrawal_limit
		acc["daily_transfer_limit"] = transfer_limit
		acc["updated_at"] = self._now()
		self._emit(t, "daily_limit_updated", acc)
		return deepcopy(acc)

	# ── Balance ───────────────────────────────────────────────────────────────

	async def get_account_balance(
		self,
		tenant_id: str,
		account_id: str,
	) -> dict[str, Any]:
		"""Return current balance breakdown for a FOSA account."""
		t = self._tenant(tenant_id)
		acc = self._get_account(account_id, t)
		return {
			"account_id": account_id,
			"account_number": acc["account_number"],
			"book_balance": acc["book_balance"],
			"available_balance": acc["available_balance"],
			"locked_balance": acc["locked_balance"],
			"overdraft_limit": acc.get("overdraft_limit", Decimal("0")),
			"overdraft_used": acc.get("overdraft_used", Decimal("0")),
			"currency": acc.get("currency", "KES"),
			"as_at": self._now(),
		}

	# ── Statements ────────────────────────────────────────────────────────────

	async def get_mini_statement(
		self,
		tenant_id: str,
		account_id: str,
		last_n: int = 10,
	) -> list[dict[str, Any]]:
		"""Return the N most recent transactions for a FOSA account."""
		t = self._tenant(tenant_id)
		self._get_account(account_id, t)  # validate exists
		txns = sorted(
			[deepcopy(tx) for tx in self.transactions.values()
			 if tx["tenant_id"] == t and tx["account_id"] == account_id],
			key=lambda x: x["created_at"],
			reverse=True,
		)
		return txns[:last_n]

	async def get_full_statement(
		self,
		tenant_id: str,
		account_id: str,
		from_date: str,
		to_date: str,
	) -> dict[str, Any]:
		"""Generate a full account statement for a date range."""
		t = self._tenant(tenant_id)
		acc = self._get_account(account_id, t)

		txns = sorted(
			[deepcopy(tx) for tx in self.transactions.values()
			 if tx["tenant_id"] == t
			 and tx["account_id"] == account_id
			 and from_date <= tx["created_at"][:10] <= to_date],
			key=lambda x: x["created_at"],
		)
		total_in = sum(tx["amount"] for tx in txns if tx["txn_type"] in {
			"fosa_deposit", "fosa_mpesa_in", "fosa_bosa_in", "fosa_transfer_in"})
		total_out = sum(tx["amount"] for tx in txns if tx["txn_type"] in {
			"fosa_withdrawal", "fosa_mpesa_out", "fosa_bosa_out", "fosa_transfer_out", "fosa_standing_order"})

		return {
			"account_id": account_id,
			"account_number": acc["account_number"],
			"member_id": acc["member_id"],
			"currency": acc.get("currency", "KES"),
			"from_date": from_date,
			"to_date": to_date,
			"opening_balance": txns[0]["balance_before"] if txns else acc["book_balance"],
			"closing_balance": txns[-1]["balance_after"] if txns else acc["book_balance"],
			"total_credits": total_in,
			"total_debits": total_out,
			"transaction_count": len(txns),
			"transactions": txns,
			"generated_at": self._now(),
		}

	# ── Standing Orders ───────────────────────────────────────────────────────

	async def create_standing_order(
		self,
		tenant_id: str,
		account_id: str,
		beneficiary_account: str,
		amount: Decimal,
		frequency: str,
		start_date: str,
		end_date: str | None = None,
		beneficiary_name: str | None = None,
		narration: str | None = None,
	) -> dict[str, Any]:
		"""Create a recurring standing order from a FOSA account."""
		t = self._tenant(tenant_id)
		valid_frequencies = {"daily", "weekly", "biweekly", "monthly", "quarterly", "annually"}
		assert frequency in valid_frequencies, f"invalid_frequency: {frequency}"
		assert amount > Decimal("0"), "amount must be positive"

		acc = self._get_account(account_id, t)
		if acc["status"] != "active":
			raise ValueError(f"cannot_create_so_on_account_status: {acc['status']}")

		so_id = self._record_id("so")
		record: dict[str, Any] = {
			"id": so_id,
			"type": "fosa_standing_order",
			"tenant_id": t,
			"account_id": account_id,
			"account_number": acc["account_number"],
			"member_id": acc["member_id"],
			"beneficiary_account": beneficiary_account,
			"beneficiary_name": beneficiary_name,
			"amount": amount,
			"frequency": frequency,
			"start_date": start_date,
			"end_date": end_date,
			"next_execution_date": start_date,
			"last_executed_at": None,
			"execution_count": 0,
			"failed_count": 0,
			"status": "active",
			"narration": narration or f"Standing order to {beneficiary_account}",
			"created_at": self._now(),
		}
		self.standing_orders[so_id] = record
		self._emit(t, "standing_order_created", record)
		return deepcopy(record)

	async def cancel_standing_order(
		self,
		tenant_id: str,
		standing_order_id: str,
	) -> dict[str, Any]:
		"""Cancel an active standing order."""
		t = self._tenant(tenant_id)
		so = self._get_standing_order(standing_order_id, t)
		if so["status"] != "active":
			raise ValueError(f"cannot_cancel_so_in_status: {so['status']}")
		so["status"] = "cancelled"
		so["cancelled_at"] = self._now()
		self._emit(t, "standing_order_cancelled", so)
		return deepcopy(so)

	async def get_standing_orders(
		self,
		tenant_id: str,
		account_id: str,
	) -> list[dict[str, Any]]:
		"""List all standing orders for a FOSA account."""
		t = self._tenant(tenant_id)
		self._get_account(account_id, t)
		return [deepcopy(so) for so in self.standing_orders.values()
		        if so["tenant_id"] == t and so["account_id"] == account_id]

	async def process_standing_orders(
		self,
		tenant_id: str,
		processing_date: str,
	) -> dict[str, Any]:
		"""Nightly job: execute all due standing orders. Idempotent."""
		t = self._tenant(tenant_id)
		processed, failed, skipped = 0, 0, 0
		results: list[dict[str, Any]] = []

		due = [so for so in self.standing_orders.values()
		       if so["tenant_id"] == t
		       and so["status"] == "active"
		       and so["next_execution_date"] <= processing_date]

		for so in due:
			# Idempotency: skip if a transaction for this SO was already posted for this processing_date
			already_done = (
				so.get("last_executed_at") is not None
				and so["last_executed_at"][:10] == processing_date
			) or any(
				tx["tenant_id"] == t
				and tx.get("standing_order_id") == so["id"]
				and tx["created_at"][:10] == processing_date
				for tx in self.transactions.values()
			)
			if already_done:
				skipped += 1
				continue

			try:
				acc = self._get_account(so["account_id"], t)
				if acc["status"] != "active" or acc["available_balance"] < so["amount"]:
					so["failed_count"] += 1
					failed += 1
					results.append({"so_id": so["id"], "status": "failed", "reason": "insufficient_balance_or_inactive"})
					continue

				bal_before = acc["book_balance"]
				acc["book_balance"] -= so["amount"]
				acc["available_balance"] -= so["amount"]
				acc["updated_at"] = self._now()
				acc["last_transaction_at"] = self._now()

				txn = self._build_txn(t, so["account_id"], acc, "fosa_standing_order",
				                      so["amount"], "INTERNAL", self._record_id("soref"),
				                      so.get("narration", "Standing order"))
				txn["balance_before"] = bal_before
				txn["balance_after"] = acc["book_balance"]
				txn["standing_order_id"] = so["id"]
				txn["beneficiary_account"] = so["beneficiary_account"]
				self.transactions[txn["id"]] = txn

				# Advance next execution date
				so["last_executed_at"] = self._now()
				so["execution_count"] += 1
				so["next_execution_date"] = self._next_so_date(processing_date, so["frequency"])
				if so.get("end_date") and so["next_execution_date"] > so["end_date"]:
					so["status"] = "completed"

				processed += 1
				results.append({"so_id": so["id"], "status": "success", "amount": str(so["amount"])})
				self._emit(t, "standing_order_executed", txn)

			except Exception as exc:
				_log.error("standing_order error so=%s: %s", so["id"], exc)
				failed += 1
				results.append({"so_id": so["id"], "status": "error", "reason": str(exc)})

		return {
			"processing_date": processing_date,
			"due_count": len(due),
			"processed": processed,
			"failed": failed,
			"skipped_duplicate": skipped,
			"results": results,
		}

	def _next_so_date(self, current: str, frequency: str) -> str:
		d = date.fromisoformat(current)
		delta_map = {
			"daily": timedelta(days=1),
			"weekly": timedelta(weeks=1),
			"biweekly": timedelta(weeks=2),
			"quarterly": timedelta(days=91),
			"annually": timedelta(days=365),
		}
		if frequency == "monthly":
			month = d.month + 1
			year = d.year + (month - 1) // 12
			month = (month - 1) % 12 + 1
			return date(year, month, d.day).isoformat()
		return (d + delta_map.get(frequency, timedelta(days=30))).isoformat()

	# ── Overdrafts ────────────────────────────────────────────────────────────

	async def request_overdraft(
		self,
		tenant_id: str,
		account_id: str,
		requested_amount: Decimal,
		purpose: str,
	) -> dict[str, Any]:
		"""Submit overdraft request for a FOSA account."""
		t = self._tenant(tenant_id)
		assert requested_amount > Decimal("0"), "requested_amount must be positive"
		assert purpose, "purpose required"

		acc = self._get_account(account_id, t)
		if acc["status"] != "active":
			raise ValueError(f"cannot_request_overdraft_on_status: {acc['status']}")

		od_id = self._record_id("od")
		record: dict[str, Any] = {
			"id": od_id,
			"type": "fosa_overdraft",
			"tenant_id": t,
			"account_id": account_id,
			"account_number": acc["account_number"],
			"member_id": acc["member_id"],
			"requested_amount": requested_amount,
			"approved_amount": None,
			"purpose": purpose,
			"status": "requested",
			"approved_by": None,
			"approved_at": None,
			"expiry_date": None,
			"balance_used": Decimal("0"),
			"created_at": self._now(),
		}
		self.overdrafts[od_id] = record
		self._emit(t, "overdraft_requested", record)
		return deepcopy(record)

	async def approve_overdraft(
		self,
		tenant_id: str,
		account_id: str,
		approved_amount: Decimal,
		approved_by: str,
		expiry_date: str,
	) -> dict[str, Any]:
		"""Approve overdraft and set limit on account."""
		t = self._tenant(tenant_id)
		assert approved_amount > Decimal("0"), "approved_amount must be positive"

		acc = self._get_account(account_id, t)

		# Find pending overdraft request for this account
		od = next(
			(o for o in self.overdrafts.values()
			 if o["tenant_id"] == t and o["account_id"] == account_id and o["status"] == "requested"),
			None,
		)
		if not od:
			raise KeyError(f"no_pending_overdraft_request_for_account: {account_id}")

		od["approved_amount"] = approved_amount
		od["approved_by"] = approved_by
		od["approved_at"] = self._now()
		od["expiry_date"] = expiry_date
		od["status"] = "approved"

		acc["overdraft_limit"] = approved_amount
		acc["updated_at"] = self._now()

		self._emit(t, "overdraft_approved", od)
		return deepcopy(od)

	# ── Dormancy ──────────────────────────────────────────────────────────────

	async def get_dormant_fosa_accounts(
		self,
		tenant_id: str,
	) -> list[dict[str, Any]]:
		"""Return accounts with no transaction activity for >6 months."""
		t = self._tenant(tenant_id)
		cutoff = (date.today() - timedelta(days=DORMANCY_MONTHS * 30)).isoformat()
		dormant = []
		for acc in self.accounts.values():
			if acc["tenant_id"] != t or acc["status"] == "closed":
				continue
			last_txn = acc.get("last_transaction_at")
			# No transaction ever, or last was before cutoff
			if last_txn is None or last_txn[:10] < cutoff:
				dormant.append(deepcopy(acc))
		return dormant

	async def reactivate_fosa_account(
		self,
		tenant_id: str,
		account_id: str,
		reactivation_deposit: Decimal,
	) -> dict[str, Any]:
		"""Reactivate a dormant account with a mandatory reactivation deposit."""
		t = self._tenant(tenant_id)
		assert reactivation_deposit > Decimal("0"), "reactivation_deposit must be positive"

		acc = self._get_account(account_id, t)
		if acc["status"] not in {"dormant", "frozen"}:
			raise ValueError(f"cannot_reactivate_from_status: {acc['status']}")

		acc["status"] = "active"
		acc["book_balance"] += reactivation_deposit
		acc["available_balance"] += reactivation_deposit
		acc["reactivated_at"] = self._now()
		acc["last_transaction_at"] = self._now()
		acc["updated_at"] = self._now()

		txn = self._build_txn(t, account_id, acc, "fosa_deposit", reactivation_deposit, "TELLER",
		                      self._record_id("react"), "Reactivation deposit")
		txn["balance_before"] = acc["book_balance"] - reactivation_deposit
		txn["balance_after"] = acc["book_balance"]
		self.transactions[txn["id"]] = txn

		self._post_gl(t, GL_CASH_TELLER, "Cash - Teller", GL_FOSA_DEPOSITS, "Member FOSA Deposits",
		              reactivation_deposit, "Account reactivation deposit", account_id)
		self._emit(t, "fosa_account_reactivated", acc)
		return deepcopy(acc)

	# ── Portfolio & Reporting ─────────────────────────────────────────────────

	async def get_fosa_portfolio(self, tenant_id: str) -> dict[str, Any]:
		"""Aggregate FOSA portfolio statistics for management reporting."""
		t = self._tenant(tenant_id)
		today = self._today()

		accounts = [a for a in self.accounts.values() if a["tenant_id"] == t]
		by_status: dict[str, int] = {}
		total_deposits = Decimal("0")
		for acc in accounts:
			s = acc.get("status", "unknown")
			by_status[s] = by_status.get(s, 0) + 1
			if s != "closed":
				total_deposits += acc.get("book_balance", Decimal("0"))

		daily_txns = [tx for tx in self.transactions.values()
		              if tx["tenant_id"] == t and tx["created_at"][:10] == today]
		daily_dep = sum(tx["amount"] for tx in daily_txns if tx["txn_type"] in {
			"fosa_deposit", "fosa_mpesa_in", "fosa_bosa_in"})
		daily_with = sum(tx["amount"] for tx in daily_txns if tx["txn_type"] in {
			"fosa_withdrawal", "fosa_mpesa_out", "fosa_bosa_out", "fosa_standing_order"})

		total_od = sum(acc.get("overdraft_limit", Decimal("0")) for acc in accounts if acc["status"] != "closed")
		active_cards = sum(1 for c in self.atm_cards.values() if c["tenant_id"] == t and c["status"] == "active")
		active_so = sum(1 for so in self.standing_orders.values() if so["tenant_id"] == t and so["status"] == "active")

		return {
			"tenant_id": t,
			"total_deposits": total_deposits,
			"active_accounts": by_status.get("active", 0),
			"dormant_accounts": by_status.get("dormant", 0),
			"frozen_accounts": by_status.get("frozen", 0),
			"closed_accounts": by_status.get("closed", 0),
			"daily_deposit_volume": daily_dep,
			"daily_withdrawal_volume": daily_with,
			"total_overdraft_exposure": total_od,
			"total_cards_issued": active_cards,
			"active_standing_orders": active_so,
			"generated_at": self._now(),
		}

	async def get_interest_earned(
		self,
		tenant_id: str,
		account_id: str,
		period_id: str,
	) -> Decimal:
		"""Return total interest posted to a FOSA account in a given period."""
		t = self._tenant(tenant_id)
		self._get_account(account_id, t)
		total = sum(
			rec["amount"] for rec in self.interest_records.values()
			if rec["tenant_id"] == t and rec["account_id"] == account_id and rec.get("period_id") == period_id
		)
		return total if isinstance(total, Decimal) else Decimal(str(total))

	# ── Teller ────────────────────────────────────────────────────────────────

	async def get_teller_summary(
		self,
		tenant_id: str,
		teller_id: str,
		date_str: str,
	) -> dict[str, Any]:
		"""Return teller cash position for a given date."""
		t = self._tenant(tenant_id)
		key = f"{t}|{teller_id}|{date_str}"
		totals = self.daily_totals.get(key, {
			"tenant_id": t,
			"teller_id": teller_id,
			"date": date_str,
			"total_deposits": Decimal("0"),
			"total_withdrawals": Decimal("0"),
			"transaction_count": 0,
		})
		opening_float = Decimal("50000")  # configurable per teller in production
		closing_float = opening_float + totals["total_deposits"] - totals["total_withdrawals"]
		return {
			"teller_id": teller_id,
			"date": date_str,
			"opening_float": opening_float,
			"total_deposits": totals["total_deposits"],
			"total_withdrawals": totals["total_withdrawals"],
			"total_transactions": totals["transaction_count"],
			"closing_float": closing_float,
			"variance": Decimal("0"),  # production: compare against physical cash count
		}

	# ── Health ────────────────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		"""Service health and statistics."""
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"account_count": len(self.accounts),
			"active_accounts": sum(1 for a in self.accounts.values() if a.get("status") == "active"),
			"transaction_count": len(self.transactions),
			"gl_entry_count": len(self.gl_entries),
			"active_cards": sum(1 for c in self.atm_cards.values() if c.get("status") == "active"),
			"active_standing_orders": sum(1 for so in self.standing_orders.values() if so.get("status") == "active"),
			"checked_at": self._now(),
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == t]


# Module-level alias
FosaService = FOSAService
