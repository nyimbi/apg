"""Async service layer for APG Point of Sale.

Implements the full PointOfSaleService covering:
  - Session management (open, close, summary)
  - Transactions (begin, add/remove items, discounts, payments, void, park)
  - Payment methods (cash, card, M-Pesa, loyalty, receipts)
  - Returns & refunds (initiate, process, exchange, analytics)
  - Inventory integration (stock check, price check, deduction, low-stock alerts)
  - End of day (EOD report, cash reconciliation, sales summary, till variance)
  - Promotions (apply, check active, performance)

Currency: KES (Kenyan Shilling).  VAT: 16% standard rate (KRA).
M-Pesa integration via reference number (actual API call delegated to adapter).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from .models import (
	# Enums
	CashEventType, DiscountType, InventoryMovementType, PaymentMethod,
	PaymentStatus, ReceiptFormat, RefundReason, SessionStatus,
	TerminalStatus, TransactionStatus, TransactionType,
	# Create / Response models
	CashFloatCreate, CashFloatResponse,
	DiscountCreate, DiscountResponse,
	EndOfDayReportCreate, EndOfDayReportResponse,
	InventoryMovementCreate, InventoryMovementResponse,
	LoyaltyTransactionCreate, LoyaltyTransactionResponse,
	OfflineSyncBatch, OfflineSyncResult,
	PaymentCreate, PaymentResponse,
	PosSessionCreate, PosSessionResponse, PosSessionUpdate,
	PosTerminalCreate, PosTerminalResponse, PosTerminalUpdate,
	PriceOverrideCreate, PriceOverrideResponse,
	ReceiptCreate, ReceiptResponse,
	RefundCreate, RefundResponse,
	SaleItemCreate, SaleTransactionCreate, SaleTransactionResponse,
	SupervisorOverrideCreate, SupervisorOverrideResponse,
	uuid7str,
)

# ---------------------------------------------------------------------------
# Backward-compat type aliases for the original PosService method signatures
# (those methods used Pos-prefixed names that don't exist in models.py)
# ---------------------------------------------------------------------------
PosCashEventCreate = CashFloatCreate
PosCashEventResponse = CashFloatResponse
PosReceiptCreate = ReceiptCreate
PosReceiptResponse = ReceiptResponse


class _PosReconciliationCreate:
	"""Minimal compat shim for create_reconciliation legacy method."""
	def __init__(self, **kw: Any) -> None:
		self.__dict__.update(kw)


class _PosReconciliationResponse:
	"""Minimal compat shim for reconciliation response."""
	def __init__(self, **kw: Any) -> None:
		self.__dict__.update(kw)
		self.id = kw.get("id") or uuid7str()

	def model_dump(self) -> dict[str, Any]:
		return dict(self.__dict__)


PosReconciliationCreate = _PosReconciliationCreate
PosReconciliationResponse = _PosReconciliationResponse


class _PosTransactionCreate:
	"""Minimal compat shim for post_transaction legacy method."""
	def __init__(self, **kw: Any) -> None:
		self.__dict__.update(kw)
		self.items = kw.get("items", [])
		self.transaction_type = kw.get("transaction_type", "sale")
		self.payment_method = kw.get("payment_method", "cash")
		self.session_id = kw.get("session_id", "")
		self.tenant_id = kw.get("tenant_id", "default")

	def model_dump(self) -> dict[str, Any]:
		return dict(self.__dict__)


class _PosTransactionResponse:
	"""Minimal compat shim for post_transaction legacy response."""
	def __init__(self, **kw: Any) -> None:
		self.__dict__.update(kw)
		self.id = kw.get("id") or uuid7str()
		self.transaction_number = kw.get("transaction_number") or f"TXN-{self.id[:8].upper()}"

	def model_dump(self) -> dict[str, Any]:
		return dict(self.__dict__)


PosTransactionCreate = _PosTransactionCreate
PosTransactionResponse = _PosTransactionResponse


class _PosVoidCreate:
	"""Minimal compat shim for void_transaction_legacy."""
	def __init__(self, **kw: Any) -> None:
		self.__dict__.update(kw)
		self.original_transaction_id = kw.get("original_transaction_id", "")
		self.terminal_id = kw.get("terminal_id", "")
		self.tenant_id = kw.get("tenant_id", "default")

	def model_dump(self) -> dict[str, Any]:
		return dict(self.__dict__)


class _PosVoidResponse:
	"""Minimal compat shim for void response."""
	def __init__(self, **kw: Any) -> None:
		self.__dict__.update(kw)
		self.id = kw.get("id") or uuid7str()

	def model_dump(self) -> dict[str, Any]:
		return dict(self.__dict__)


PosVoidCreate = _PosVoidCreate
PosVoidResponse = _PosVoidResponse

logger = logging.getLogger(__name__)

_CENTS = Decimal("0.01")
_VAT_RATE = Decimal("0.16")   # Kenya standard VAT
_LOYALTY_EARN_RATE = Decimal("1.0")   # 1 point per KES
_LOYALTY_REDEEM_RATE = Decimal("0.01")  # KES 0.01 per point


def _now() -> datetime:
	return datetime.utcnow()


def _today() -> date:
	return date.today()


def _cents(v: Decimal | float) -> Decimal:
	return Decimal(str(v)).quantize(_CENTS, rounding=ROUND_HALF_UP)


def _log_op(op: str, tenant_id: str, entity_id: str | None = None) -> None:
	logger.info("pos | op=%s tenant=%s entity=%s", op, tenant_id, entity_id or "-")


def _log_warn(msg: str, **kw: Any) -> None:
	logger.warning("pos | %s %s", msg, kw)


def _log_txn(txn_number: str, txn_type: str, total: float) -> None:
	logger.info("pos | txn=%s type=%s total=%.2f", txn_number, txn_type, total)


# ---------------------------------------------------------------------------
# In-process store
# ---------------------------------------------------------------------------

class _Store(dict):
	"""Dict-based in-process store with tenant-scoped lookup."""

	def put(self, tenant_id: str, record_id: str, obj: Any) -> None:
		self[(tenant_id, record_id)] = obj

	def get_item(self, tenant_id: str, record_id: str) -> Any | None:
		return self.get((tenant_id, record_id))

	def tenant_values(self, tenant_id: str) -> list[Any]:
		return [v for (tid, _), v in self.items() if tid == tenant_id]

	def all_values(self) -> list[Any]:
		return list(self.values())


# ---------------------------------------------------------------------------
# Inventory stub (replace with real inventory adapter)
# ---------------------------------------------------------------------------

class _InventoryStore:
	"""Minimal in-process inventory ledger for demo purposes."""

	def __init__(self) -> None:
		self._stock: dict[tuple[str, str], float] = {}  # (store_id, sku) -> qty
		self._prices: dict[tuple[str, str], float] = {}  # (tenant_id, sku) -> unit_price
		self._tier_prices: dict[tuple[str, str, str], float] = {}  # (tenant_id, sku, tier) -> price

	def set_stock(self, store_id: str, sku: str, qty: float) -> None:
		self._stock[(store_id, sku)] = qty

	def get_stock(self, store_id: str, sku: str) -> float:
		return self._stock.get((store_id, sku), 0.0)

	def adjust_stock(self, store_id: str, sku: str, delta: float) -> float:
		current = self._stock.get((store_id, sku), 0.0)
		self._stock[(store_id, sku)] = current + delta
		return self._stock[(store_id, sku)]

	def set_price(self, tenant_id: str, sku: str, price: float, tier: str | None = None) -> None:
		if tier:
			self._tier_prices[(tenant_id, sku, tier)] = price
		else:
			self._prices[(tenant_id, sku)] = price

	def get_price(self, tenant_id: str, sku: str, tier: str | None = None) -> float | None:
		if tier:
			return self._tier_prices.get((tenant_id, sku, tier)) or self._prices.get((tenant_id, sku))
		return self._prices.get((tenant_id, sku))


# ---------------------------------------------------------------------------
# Promotions store
# ---------------------------------------------------------------------------

class _PromotionStore:
	"""In-process promotions registry."""

	def __init__(self) -> None:
		self._promos: dict[str, dict[str, Any]] = {}

	def add(self, promo: dict[str, Any]) -> None:
		self._promos[promo["id"]] = promo

	def get(self, promo_id: str) -> dict[str, Any] | None:
		return self._promos.get(promo_id)

	def active_for_sku(self, tenant_id: str, sku: str, tier: str | None) -> list[dict[str, Any]]:
		now = _now()
		return [
			p for p in self._promos.values()
			if p["tenant_id"] == tenant_id
			and p.get("is_active", True)
			and (not p.get("valid_from") or p["valid_from"] <= now)
			and (not p.get("valid_until") or p["valid_until"] >= now)
			and (not p.get("product_skus") or sku in p["product_skus"])
			and (not tier or not p.get("tier") or p["tier"] == tier)
		]

	def record_use(self, promo_id: str, amount: float) -> None:
		p = self._promos.get(promo_id)
		if p:
			p["times_used"] = p.get("times_used", 0) + 1
			p["total_discount_given"] = p.get("total_discount_given", 0.0) + amount


# ---------------------------------------------------------------------------
# Loyalty store
# ---------------------------------------------------------------------------

class _LoyaltyStore:
	"""In-process loyalty points ledger keyed by (tenant_id, customer_id)."""

	def __init__(self) -> None:
		self._balances: dict[tuple[str, str], int] = {}
		self._history: list[dict[str, Any]] = []

	def balance(self, tenant_id: str, customer_id: str) -> int:
		return self._balances.get((tenant_id, customer_id), 0)

	def earn(self, tenant_id: str, customer_id: str, points: int) -> int:
		self._balances[(tenant_id, customer_id)] = self.balance(tenant_id, customer_id) + points
		return self._balances[(tenant_id, customer_id)]

	def redeem(self, tenant_id: str, customer_id: str, points: int) -> int:
		current = self.balance(tenant_id, customer_id)
		assert current >= points, f"insufficient loyalty points: have {current}, need {points}"
		self._balances[(tenant_id, customer_id)] = current - points
		return self._balances[(tenant_id, customer_id)]

	def record(self, entry: dict[str, Any]) -> None:
		self._history.append(entry)

	def customer_history(self, tenant_id: str, customer_id: str) -> list[dict[str, Any]]:
		return [e for e in self._history if e["tenant_id"] == tenant_id and e["customer_id"] == customer_id]


# ===========================================================================
# Main service
# ===========================================================================


class PointOfSaleService:
	"""Full Point of Sale service.

	All public methods are async to support future DB/adapter integration.
	Internal state uses in-process stores — swap for SQLAlchemy / Redis adapters
	in production.

	Preserves the original PosService methods as well as the expanded interface.
	"""

	def __init__(self) -> None:
		# New stores (typed, _Store)
		self._store_terminals: _Store = _Store()
		self._store_sessions: _Store = _Store()
		self._store_transactions: _Store = _Store()
		self._store_parked: _Store = _Store()
		self._store_payments: _Store = _Store()
		self._store_receipts_v2: _Store = _Store()
		self._store_refunds: _Store = _Store()
		self._store_returns: _Store = _Store()
		self._store_movements: _Store = _Store()
		self._store_price_overrides: _Store = _Store()
		self._store_discounts: _Store = _Store()
		self._store_supervisor_overrides: _Store = _Store()
		self._store_eod_reports: _Store = _Store()
		self._store_reconciliations_v2: _Store = _Store()
		self._store_loyalty_txns: _Store = _Store()

		self._inventory = _InventoryStore()
		self._promotions = _PromotionStore()
		self._loyalty = _LoyaltyStore()

		# Original stores (backward-compat)
		self._terminals: dict[str, dict[str, Any]] = {}
		self._sessions: dict[str, dict[str, Any]] = {}
		self._transactions: dict[str, dict[str, Any]] = {}
		self._cash_events: dict[str, dict[str, Any]] = {}
		self._reconciliations: dict[str, dict[str, Any]] = {}
		self._receipts: dict[str, dict[str, Any]] = {}
		self._voids: dict[str, dict[str, Any]] = {}

	# ======================================================================
	# SESSION MANAGEMENT
	# ======================================================================

	async def open_session(
		self,
		terminal_id: str,
		cashier_id: str,
		opening_float: float,
		*,
		tenant_id: str = "default",
		store_id: str = "default",
		supervisor_id: str | None = None,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Open a new cashier session on a terminal.

		Enforces: no concurrent open session on the same terminal.
		"""
		assert opening_float >= 0, "opening_float must be non-negative"

		# Check no existing open session on terminal
		existing = next(
			(s for s in self._store_sessions.tenant_values(tenant_id)
			 if s.terminal_id == terminal_id and s.status == SessionStatus.OPEN),
			None,
		)
		assert existing is None, f"terminal {terminal_id} already has an open session"

		_log_op("open_session", tenant_id, terminal_id)
		session = PosSessionResponse(
			tenant_id=tenant_id,
			terminal_id=terminal_id,
			store_id=store_id,
			cashier_id=cashier_id,
			opening_float=opening_float,
			supervisor_id=supervisor_id,
			status=SessionStatus.OPEN,
			created_by=created_by,
		)
		self._store_sessions.put(tenant_id, session.id, session)
		return session.model_dump(mode="json")

	async def close_session(
		self,
		session_id: str,
		closing_float: float,
		closing_notes: str | None = None,
		*,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Close an open session, recording the physical cash count."""
		session = self._store_sessions.get_item(tenant_id, session_id)
		assert session is not None, f"session not found: {session_id}"
		assert session.status == SessionStatus.OPEN, "session must be open to close"

		data = session.model_dump()
		data["status"] = SessionStatus.CLOSED.value
		data["closing_cash_counted"] = closing_float
		data["closed_at"] = _now()
		data["notes"] = closing_notes
		data["updated_at"] = _now()

		# Calculate expected cash and variance
		expected = session.opening_float + session.total_cash_sales
		data["expected_cash"] = expected
		data["variance"] = round(closing_float - expected, 2)

		updated = PosSessionResponse(**data)
		self._store_sessions.put(tenant_id, session_id, updated)
		_log_op("close_session", tenant_id, session_id)
		return updated.model_dump(mode="json")

	async def get_session(self, session_id: str, *, tenant_id: str = "default") -> dict[str, Any] | None:
		"""Fetch a session by ID."""
		session = self._store_sessions.get_item(tenant_id, session_id)
		if session is None:
			return None
		return session.model_dump(mode="json")

	async def session_summary(self, session_id: str, *, tenant_id: str = "default") -> dict[str, Any]:
		"""Produce a session summary: sales, refunds, net, variance."""
		session = self._store_sessions.get_item(tenant_id, session_id)
		assert session is not None, f"session not found: {session_id}"

		txns = [
			t for t in self._store_transactions.tenant_values(tenant_id)
			if t.session_id == session_id and not t.is_deleted
		]
		sales = [t for t in txns if t.transaction_type == TransactionType.SALE and t.status == TransactionStatus.COMPLETED]
		refunds = [t for t in txns if t.transaction_type == TransactionType.REFUND]
		voids = [t for t in txns if t.status == TransactionStatus.VOIDED]

		gross_sales = sum(t.grand_total for t in sales)
		total_refunds = sum(t.grand_total for t in refunds)
		total_discounts = sum(t.discount_total for t in sales)
		total_tax = sum(t.tax_total for t in sales)
		net_sales = gross_sales - total_refunds

		expected_cash = session.opening_float + sum(
			t.grand_total for t in sales
			if any(p.payment_method == PaymentMethod.CASH for p in self._get_txn_payments(tenant_id, t.id))
		)
		variance = (session.closing_cash_counted or 0.0) - expected_cash

		# Payment method breakdown
		payment_breakdown: dict[str, float] = defaultdict(float)
		for t in sales:
			for p in self._get_txn_payments(tenant_id, t.id):
				payment_breakdown[p.payment_method.value] += float(p.amount)

		return {
			"session_id": session_id,
			"cashier_id": session.cashier_id,
			"terminal_id": session.terminal_id,
			"store_id": session.store_id,
			"status": session.status.value,
			"opened_at": session.opened_at.isoformat(),
			"closed_at": session.closed_at.isoformat() if session.closed_at else None,
			"transaction_count": len(sales),
			"refund_count": len(refunds),
			"void_count": len(voids),
			"gross_sales": round(gross_sales, 2),
			"total_refunds": round(total_refunds, 2),
			"total_discounts": round(total_discounts, 2),
			"total_tax": round(total_tax, 2),
			"net_sales": round(net_sales, 2),
			"opening_float": session.opening_float,
			"closing_float": session.closing_cash_counted,
			"expected_cash": round(expected_cash, 2),
			"variance": round(variance, 2),
			"payment_breakdown": dict(payment_breakdown),
			"generated_at": _now().isoformat(),
		}

	# ======================================================================
	# TRANSACTIONS
	# ======================================================================

	async def begin_transaction(
		self,
		session_id: str,
		customer_id: str | None = None,
		*,
		tenant_id: str = "default",
		cashier_id: str = "system",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Open a new transaction basket on a session."""
		session = self._store_sessions.get_item(tenant_id, session_id)
		assert session is not None, f"session not found: {session_id}"
		assert session.status == SessionStatus.OPEN, "session must be open"

		txn = SaleTransactionResponse(
			tenant_id=tenant_id,
			session_id=session_id,
			terminal_id=session.terminal_id,
			store_id=session.store_id,
			cashier_id=cashier_id,
			transaction_type=TransactionType.SALE,
			customer_id=customer_id,
			status=TransactionStatus.PENDING,
			items=[],
			created_by=created_by,
		)
		self._store_transactions.put(tenant_id, txn.id, txn)
		_log_op("begin_transaction", tenant_id, txn.id)
		return txn.model_dump(mode="json")

	async def add_item(
		self,
		transaction_id: str,
		sku: str,
		quantity: float,
		price_override: float | None = None,
		*,
		tenant_id: str = "default",
		description: str | None = None,
		tax_rate: float | None = None,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Add a line item to an open transaction."""
		txn = self._store_transactions.get_item(tenant_id, transaction_id)
		assert txn is not None, f"transaction not found: {transaction_id}"
		assert txn.status == TransactionStatus.PENDING, "transaction must be in PENDING state"
		assert quantity > 0, "quantity must be positive"

		unit_price = price_override or self._inventory.get_price(tenant_id, sku) or 0.0
		tr = tax_rate if tax_rate is not None else float(_VAT_RATE)
		tax_inclusive = True
		if tax_inclusive:
			# Extract VAT from inclusive price
			tax_amount = round(unit_price * quantity * tr / (1 + tr), 4)
		else:
			tax_amount = round(unit_price * quantity * tr, 4)

		line_total = round(unit_price * quantity, 4)

		item = SaleItemCreate(
			sku=sku,
			description=description or sku,
			quantity=quantity,
			unit_price=unit_price,
			tax_rate=tr,
			tax_amount=tax_amount,
			tax_inclusive=tax_inclusive,
			discount_amount=0.0,
			line_total=line_total,
		)

		data = txn.model_dump()
		data["items"].append(item.model_dump())
		data["updated_at"] = _now()
		self._recalculate_totals(data)
		updated = SaleTransactionResponse(**data)
		self._store_transactions.put(tenant_id, transaction_id, updated)
		return updated.model_dump(mode="json")

	async def remove_item(
		self,
		transaction_id: str,
		line_id: str,
		*,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Remove an item from an open transaction by SKU or line index.

		``line_id`` is the SKU string (removes first matching line).
		"""
		txn = self._store_transactions.get_item(tenant_id, transaction_id)
		assert txn is not None, f"transaction not found: {transaction_id}"
		assert txn.status == TransactionStatus.PENDING, "transaction must be in PENDING state"

		data = txn.model_dump()
		before = len(data["items"])
		data["items"] = [i for i in data["items"] if i.get("sku") != line_id]
		assert len(data["items"]) < before, f"item not found: {line_id}"
		data["updated_at"] = _now()
		self._recalculate_totals(data)
		updated = SaleTransactionResponse(**data)
		self._store_transactions.put(tenant_id, transaction_id, updated)
		return updated.model_dump(mode="json")

	async def apply_discount(
		self,
		transaction_id: str,
		discount_type: str,
		value: float,
		approved_by: str | None = None,
		*,
		tenant_id: str = "default",
		coupon_code: str | None = None,
	) -> dict[str, Any]:
		"""Apply a discount to the transaction.

		discount_types: percentage | fixed_amount | coupon_code | loyalty_points
		"""
		txn = self._store_transactions.get_item(tenant_id, transaction_id)
		assert txn is not None, f"transaction not found: {transaction_id}"
		assert txn.status == TransactionStatus.PENDING, "transaction must be PENDING"

		data = txn.model_dump()
		subtotal = data.get("subtotal", 0.0) or sum(i.get("line_total", 0) for i in data["items"])

		discount_amount = 0.0
		dt = _normalize(discount_type)

		match dt:
			case "percentage":
				assert 0 < value <= 100, "percentage must be 0-100"
				discount_amount = round(subtotal * value / 100, 2)
			case "fixed_amount":
				assert value > 0, "fixed_amount must be positive"
				discount_amount = min(round(value, 2), subtotal)
			case "coupon_code":
				# Look up coupon in promotion store
				promo = next(
					(p for p in self._promotions._promos.values()
					 if p.get("coupon_code") == (coupon_code or str(value))
					 and p.get("tenant_id") == tenant_id
					 and p.get("is_active", True)),
					None,
				)
				assert promo is not None, f"coupon not found: {coupon_code or value}"
				if promo.get("discount_type") == "percentage":
					discount_amount = round(subtotal * promo["value"] / 100, 2)
				else:
					discount_amount = min(round(promo["value"], 2), subtotal)
				self._promotions.record_use(promo["id"], discount_amount)
			case "loyalty_points":
				# Convert points to currency value
				customer_id = txn.customer_id
				assert customer_id, "customer_id required for loyalty discount"
				points = int(value)
				balance = self._loyalty.balance(tenant_id, customer_id)
				assert balance >= points, f"insufficient points: have {balance}"
				discount_amount = round(points * float(_LOYALTY_REDEEM_RATE), 2)
				data["metadata"] = data.get("metadata", {})
				data["metadata"]["loyalty_points_redeemed"] = points
			case _:
				raise ValueError(f"unknown discount_type: {discount_type}")

		data["discount_total"] = round((data.get("discount_total") or 0.0) + discount_amount, 2)
		data["grand_total"] = round((data.get("subtotal") or subtotal) - data["discount_total"] + data.get("tax_total", 0.0), 2)
		data["supervisor_override_id"] = approved_by
		data["updated_at"] = _now()

		updated = SaleTransactionResponse(**data)
		self._store_transactions.put(tenant_id, transaction_id, updated)
		_log_op("apply_discount", tenant_id, transaction_id)
		return updated.model_dump(mode="json")

	async def split_payment(
		self,
		transaction_id: str,
		payment_splits: list[dict[str, Any]],
		*,
		tenant_id: str = "default",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Record multiple payment tenders against a transaction (e.g. cash + M-Pesa + card).

		Each split dict: {"method": "cash|mpesa|card", "amount": float, "reference": str}
		"""
		txn = self._store_transactions.get_item(tenant_id, transaction_id)
		assert txn is not None, f"transaction not found: {transaction_id}"
		assert txn.status == TransactionStatus.PENDING, "transaction must be PENDING"

		total_tendered = sum(s["amount"] for s in payment_splits)
		grand_total = txn.grand_total
		assert total_tendered >= grand_total, (
			f"insufficient tender: {total_tendered:.2f} < {grand_total:.2f}"
		)

		_pm_map = {
			"cash": PaymentMethod.CASH,
			"card": PaymentMethod.CARD_DEBIT,
			"card_credit": PaymentMethod.CARD_CREDIT,
			"card_debit": PaymentMethod.CARD_DEBIT,
			"mpesa": PaymentMethod.MOBILE_MONEY,
			"mobile_money": PaymentMethod.MOBILE_MONEY,
			"loyalty": PaymentMethod.LOYALTY_POINTS,
			"gift_card": PaymentMethod.GIFT_CARD,
			"store_credit": PaymentMethod.STORE_CREDIT,
		}

		payment_records = []
		for split in payment_splits:
			pm = _pm_map.get(_normalize(split["method"]), PaymentMethod.CASH)
			pay = PaymentResponse(
				tenant_id=tenant_id,
				transaction_id=transaction_id,
				session_id=txn.session_id,
				payment_method=pm,
				amount=round(split["amount"], 2),
				reference=split.get("reference"),
				status=PaymentStatus.AUTHORISED,
				created_by=created_by,
			)
			self._store_payments.put(tenant_id, pay.id, pay)
			payment_records.append(pay)

		# Update transaction totals
		data = txn.model_dump()
		data["amount_tendered"] = round(total_tendered, 2)
		data["change_due"] = round(total_tendered - grand_total, 2)
		data["balance_due"] = 0.0
		data["payments"] = [p.model_dump() for p in payment_records]
		data["updated_at"] = _now()

		updated = SaleTransactionResponse(**data)
		self._store_transactions.put(tenant_id, transaction_id, updated)
		_log_op("split_payment", tenant_id, transaction_id)
		return {
			"transaction_id": transaction_id,
			"grand_total": grand_total,
			"total_tendered": round(total_tendered, 2),
			"change_due": data["change_due"],
			"payments": [p.model_dump(mode="json") for p in payment_records],
		}

	async def complete_transaction(
		self,
		transaction_id: str,
		payments: list[dict[str, Any]] | None = None,
		*,
		tenant_id: str = "default",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Complete a transaction: validate payment, deduct inventory, earn loyalty."""
		txn = self._store_transactions.get_item(tenant_id, transaction_id)
		assert txn is not None, f"transaction not found: {transaction_id}"
		assert txn.status == TransactionStatus.PENDING, "transaction must be PENDING"

		# Apply any additional payments
		if payments:
			await self.split_payment(transaction_id, payments, tenant_id=tenant_id, created_by=created_by)
			txn = self._store_transactions.get_item(tenant_id, transaction_id)

		# Ensure fully paid
		total_paid = sum(p.get("amount", 0) if isinstance(p, dict) else p.amount for p in (txn.payments or []))
		existing_payments = self._get_txn_payments(tenant_id, transaction_id)
		total_paid = sum(float(p.amount) for p in existing_payments)

		assert total_paid >= txn.grand_total - 0.005, (
			f"underpaid: tendered={total_paid:.2f} due={txn.grand_total:.2f}"
		)

		data = txn.model_dump()
		data["status"] = TransactionStatus.COMPLETED.value
		data["posted_at"] = _now()
		data["receipt_number"] = f"REC-{_now().strftime('%Y%m%d')}-{uuid7str()[:6].upper()}"
		data["signature_ref"] = uuid7str()
		data["amount_tendered"] = round(total_paid, 2)
		data["change_due"] = round(total_paid - txn.grand_total, 2)
		data["balance_due"] = 0.0
		data["updated_at"] = _now()

		completed = SaleTransactionResponse(**data)
		self._store_transactions.put(tenant_id, transaction_id, completed)

		# Update session totals
		session = self._store_sessions.get_item(tenant_id, txn.session_id)
		if session:
			sdata = session.model_dump()
			sdata["transaction_count"] += 1
			sdata["total_sales"] = round(sdata["total_sales"] + txn.grand_total, 2)
			sdata["total_discounts"] = round(sdata["total_discounts"] + txn.discount_total, 2)
			sdata["total_tax"] = round(sdata["total_tax"] + txn.tax_total, 2)
			for p in existing_payments:
				match p.payment_method:
					case PaymentMethod.CASH:
						sdata["total_cash_sales"] = round(sdata["total_cash_sales"] + float(p.amount), 2)
					case PaymentMethod.CARD_CREDIT | PaymentMethod.CARD_DEBIT:
						sdata["total_card_sales"] = round(sdata["total_card_sales"] + float(p.amount), 2)
					case PaymentMethod.MOBILE_MONEY:
						sdata["total_mobile_sales"] = round(sdata["total_mobile_sales"] + float(p.amount), 2)
					case PaymentMethod.LOYALTY_POINTS:
						sdata["total_loyalty_sales"] = round(sdata["total_loyalty_sales"] + float(p.amount), 2)
			sdata["updated_at"] = _now()
			self._store_sessions.put(tenant_id, txn.session_id, PosSessionResponse(**sdata))

		# Deduct inventory
		await self.inventory_deduction(transaction_id, tenant_id=tenant_id, created_by=created_by)

		# Earn loyalty points if customer attached
		if txn.customer_id:
			points = int(txn.grand_total * float(_LOYALTY_EARN_RATE))
			self._loyalty.earn(tenant_id, txn.customer_id, points)

		_log_txn(completed.transaction_number, completed.transaction_type.value, completed.grand_total)
		return completed.model_dump(mode="json")

	async def void_transaction(
		self,
		transaction_id: str,
		reason: str,
		supervisor_id: str,
		*,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Void a transaction. Requires supervisor authorisation."""
		txn = self._store_transactions.get_item(tenant_id, transaction_id)
		assert txn is not None, f"transaction not found: {transaction_id}"
		assert txn.status in (TransactionStatus.PENDING, TransactionStatus.COMPLETED), (
			"only PENDING or COMPLETED transactions can be voided"
		)
		assert _present(reason), "void reason required"
		assert _present(supervisor_id), "supervisor_id required"

		data = txn.model_dump()
		data["status"] = TransactionStatus.VOIDED.value
		data["voided_at"] = _now()
		data["notes"] = f"VOID: {reason} | supervisor={supervisor_id}"
		data["supervisor_override_id"] = supervisor_id
		data["updated_at"] = _now()

		voided = SaleTransactionResponse(**data)
		self._store_transactions.put(tenant_id, transaction_id, voided)

		# Reverse inventory if transaction was completed
		if txn.status == TransactionStatus.COMPLETED:
			for item in txn.items:
				self._inventory.adjust_stock(txn.store_id, item.sku, item.quantity)

		# Reverse session totals
		if txn.status == TransactionStatus.COMPLETED:
			session = self._store_sessions.get_item(tenant_id, txn.session_id)
			if session:
				sdata = session.model_dump()
				sdata["total_sales"] = max(0.0, round(sdata["total_sales"] - txn.grand_total, 2))
				sdata["transaction_count"] = max(0, sdata["transaction_count"] - 1)
				sdata["updated_at"] = _now()
				self._store_sessions.put(tenant_id, txn.session_id, PosSessionResponse(**sdata))

		_log_op("void_transaction", tenant_id, transaction_id)
		return voided.model_dump(mode="json")

	async def park_transaction(
		self, transaction_id: str, *, tenant_id: str = "default"
	) -> dict[str, Any]:
		"""Suspend a transaction for later retrieval (parked sale)."""
		txn = self._store_transactions.get_item(tenant_id, transaction_id)
		assert txn is not None, f"transaction not found: {transaction_id}"
		assert txn.status == TransactionStatus.PENDING, "can only park PENDING transactions"

		data = txn.model_dump()
		data["status"] = TransactionStatus.SUSPENDED.value
		data["notes"] = (data.get("notes") or "") + " | PARKED"
		data["updated_at"] = _now()

		parked = SaleTransactionResponse(**data)
		self._store_transactions.put(tenant_id, transaction_id, parked)
		self._store_parked.put(tenant_id, transaction_id, parked)
		_log_op("park_transaction", tenant_id, transaction_id)
		return parked.model_dump(mode="json")

	async def retrieve_parked_transaction(
		self, transaction_id: str, *, tenant_id: str = "default"
	) -> dict[str, Any]:
		"""Retrieve a previously parked transaction and set it back to PENDING."""
		txn = self._store_parked.get_item(tenant_id, transaction_id)
		assert txn is not None, f"parked transaction not found: {transaction_id}"

		data = txn.model_dump()
		data["status"] = TransactionStatus.PENDING.value
		data["notes"] = (data.get("notes") or "").replace(" | PARKED", "") + " | RETRIEVED"
		data["updated_at"] = _now()

		retrieved = SaleTransactionResponse(**data)
		self._store_transactions.put(tenant_id, transaction_id, retrieved)
		del self._store_parked[(tenant_id, transaction_id)]
		return retrieved.model_dump(mode="json")

	# ======================================================================
	# PAYMENT METHODS
	# ======================================================================

	async def process_cash_payment(
		self,
		transaction_id: str,
		amount_tendered: float,
		*,
		tenant_id: str = "default",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Process a cash tender and compute change due."""
		txn = self._store_transactions.get_item(tenant_id, transaction_id)
		assert txn is not None, f"transaction not found: {transaction_id}"
		assert amount_tendered > 0, "amount_tendered must be positive"

		due = txn.grand_total
		assert amount_tendered >= due, f"insufficient cash: tendered={amount_tendered:.2f} due={due:.2f}"
		change = round(amount_tendered - due, 2)

		pay = PaymentResponse(
			tenant_id=tenant_id,
			transaction_id=transaction_id,
			session_id=txn.session_id,
			payment_method=PaymentMethod.CASH,
			amount=round(due, 2),
			status=PaymentStatus.AUTHORISED,
			created_by=created_by,
		)
		self._store_payments.put(tenant_id, pay.id, pay)

		data = txn.model_dump()
		data["amount_tendered"] = round(amount_tendered, 2)
		data["change_due"] = change
		data["balance_due"] = 0.0
		data["updated_at"] = _now()
		self._store_transactions.put(tenant_id, transaction_id, SaleTransactionResponse(**data))

		_log_op("process_cash_payment", tenant_id, transaction_id)
		return {
			"payment_id": pay.id,
			"transaction_id": transaction_id,
			"payment_method": "cash",
			"amount_due": due,
			"amount_tendered": amount_tendered,
			"change_due": change,
		}

	async def process_card_payment(
		self,
		transaction_id: str,
		amount: float,
		card_type: str,
		auth_code: str,
		*,
		tenant_id: str = "default",
		terminal_ref: str | None = None,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Process a card payment (credit or debit)."""
		txn = self._store_transactions.get_item(tenant_id, transaction_id)
		assert txn is not None, f"transaction not found: {transaction_id}"
		assert amount > 0, "amount must be positive"
		assert _present(auth_code), "auth_code required"

		pm = PaymentMethod.CARD_CREDIT if "credit" in card_type.lower() else PaymentMethod.CARD_DEBIT
		pay = PaymentResponse(
			tenant_id=tenant_id,
			transaction_id=transaction_id,
			session_id=txn.session_id,
			payment_method=pm,
			amount=round(amount, 2),
			reference=auth_code,
			terminal_ref=terminal_ref,
			status=PaymentStatus.CAPTURED,
			created_by=created_by,
		)
		self._store_payments.put(tenant_id, pay.id, pay)

		data = txn.model_dump()
		data["amount_tendered"] = round(amount, 2)
		data["balance_due"] = max(0.0, round(txn.grand_total - amount, 2))
		data["updated_at"] = _now()
		self._store_transactions.put(tenant_id, transaction_id, SaleTransactionResponse(**data))

		_log_op("process_card_payment", tenant_id, transaction_id)
		return pay.model_dump(mode="json")

	async def process_mpesa_payment(
		self,
		transaction_id: str,
		phone: str,
		amount: float,
		mpesa_ref: str,
		*,
		tenant_id: str = "default",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Record an M-Pesa payment confirmation.

		In production, this would call the Safaricom Daraja API; here we record
		the mpesa_ref as confirmation.
		"""
		txn = self._store_transactions.get_item(tenant_id, transaction_id)
		assert txn is not None, f"transaction not found: {transaction_id}"
		assert amount > 0, "amount must be positive"
		assert _present(mpesa_ref), "mpesa_ref required"
		assert _present(phone), "phone required"

		pay = PaymentResponse(
			tenant_id=tenant_id,
			transaction_id=transaction_id,
			session_id=txn.session_id,
			payment_method=PaymentMethod.MOBILE_MONEY,
			amount=round(amount, 2),
			reference=mpesa_ref,
			gateway_response={"phone": phone, "mpesa_ref": mpesa_ref},
			status=PaymentStatus.CAPTURED,
			created_by=created_by,
		)
		self._store_payments.put(tenant_id, pay.id, pay)

		data = txn.model_dump()
		data["amount_tendered"] = round(amount, 2)
		data["balance_due"] = max(0.0, round(txn.grand_total - amount, 2))
		data["updated_at"] = _now()
		self._store_transactions.put(tenant_id, transaction_id, SaleTransactionResponse(**data))

		_log_op("process_mpesa_payment", tenant_id, transaction_id)
		return {
			"payment_id": pay.id,
			"transaction_id": transaction_id,
			"payment_method": "mobile_money",
			"mpesa_ref": mpesa_ref,
			"phone": phone,
			"amount": round(amount, 2),
			"status": "captured",
		}

	async def process_loyalty_redemption(
		self,
		transaction_id: str,
		customer_id: str,
		points_to_redeem: int,
		*,
		tenant_id: str = "default",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Redeem loyalty points against a transaction."""
		txn = self._store_transactions.get_item(tenant_id, transaction_id)
		assert txn is not None, f"transaction not found: {transaction_id}"
		assert points_to_redeem > 0, "points_to_redeem must be positive"

		balance = self._loyalty.balance(tenant_id, customer_id)
		assert balance >= points_to_redeem, f"insufficient points: have {balance}"

		cash_value = round(points_to_redeem * float(_LOYALTY_REDEEM_RATE), 2)
		assert cash_value <= txn.grand_total, "redemption cannot exceed transaction total"

		self._loyalty.redeem(tenant_id, customer_id, points_to_redeem)

		pay = PaymentResponse(
			tenant_id=tenant_id,
			transaction_id=transaction_id,
			session_id=txn.session_id,
			payment_method=PaymentMethod.LOYALTY_POINTS,
			amount=cash_value,
			loyalty_points_used=points_to_redeem,
			status=PaymentStatus.AUTHORISED,
			created_by=created_by,
		)
		self._store_payments.put(tenant_id, pay.id, pay)

		# Record loyalty transaction
		before = self._loyalty.balance(tenant_id, customer_id) + points_to_redeem
		after = self._loyalty.balance(tenant_id, customer_id)
		lt = LoyaltyTransactionResponse(
			tenant_id=tenant_id,
			customer_id=customer_id,
			transaction_id=transaction_id,
			points_earned=0,
			points_redeemed=points_to_redeem,
			points_balance_before=before,
			points_balance_after=after,
			redeem_rate=float(_LOYALTY_REDEEM_RATE),
			created_by=created_by,
		)
		self._store_loyalty_txns.put(tenant_id, lt.id, lt)

		_log_op("process_loyalty_redemption", tenant_id, transaction_id)
		return {
			"payment_id": pay.id,
			"points_redeemed": points_to_redeem,
			"cash_value": cash_value,
			"remaining_balance": after,
		}

	async def generate_receipt(
		self,
		transaction_id: str,
		format: str = "thermal",
		*,
		tenant_id: str = "default",
		recipient_email: str | None = None,
		recipient_mobile: str | None = None,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Generate and store a receipt for a transaction."""
		txn = self._store_transactions.get_item(tenant_id, transaction_id)
		assert txn is not None, f"transaction not found: {transaction_id}"

		try:
			fmt = ReceiptFormat(format.lower())
		except ValueError:
			fmt = ReceiptFormat.THERMAL

		lines: list[str] = []
		lines.append("=" * 40)
		lines.append("       DATACRAFT POS RECEIPT")
		lines.append("=" * 40)
		for item in txn.items:
			item_d = item if isinstance(item, dict) else item.model_dump()
			lines.append(
				f"  {item_d.get('sku', '')} x{item_d.get('quantity', 1):<4}"
				f"  KES {item_d.get('line_total', 0):>8.2f}"
			)
		lines.append("-" * 40)
		lines.append(f"  Subtotal:          KES {txn.subtotal:>8.2f}")
		if txn.discount_total:
			lines.append(f"  Discount:         -KES {txn.discount_total:>8.2f}")
		lines.append(f"  VAT (16%):         KES {txn.tax_total:>8.2f}")
		lines.append(f"  TOTAL:             KES {txn.grand_total:>8.2f}")
		lines.append("=" * 40)
		lines.append(f"  Ref: {txn.transaction_number}")
		lines.append(f"  {_now().strftime('%Y-%m-%d %H:%M:%S')}")
		lines.append("  Thank you for shopping with us.")

		receipt = ReceiptResponse(
			tenant_id=tenant_id,
			transaction_id=transaction_id,
			session_id=txn.session_id,
			receipt_format=fmt,
			recipient_email=recipient_email,
			recipient_mobile=recipient_mobile,
			receipt_payload=txn.model_dump(mode="json"),
			rendered_content="\n".join(lines),
			issued_at=_now(),
			created_by=created_by,
		)
		self._store_receipts_v2.put(tenant_id, receipt.id, receipt)
		_log_op("generate_receipt", tenant_id, transaction_id)
		return receipt.model_dump(mode="json")

	# ======================================================================
	# RETURNS & REFUNDS
	# ======================================================================

	async def initiate_return(
		self,
		original_transaction_id: str,
		items_to_return: list[dict[str, Any]],
		reason: str,
		*,
		tenant_id: str = "default",
		manager_auth_id: str | None = None,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Begin a return for one or more items from an original transaction."""
		orig = self._store_transactions.get_item(tenant_id, original_transaction_id)
		assert orig is not None, f"original transaction not found: {original_transaction_id}"
		assert orig.status == TransactionStatus.COMPLETED, "can only return COMPLETED transactions"
		assert items_to_return, "items_to_return cannot be empty"

		try:
			rr = RefundReason(reason.lower())
		except ValueError:
			rr = RefundReason.OTHER

		# Validate returned items exist in original
		orig_skus = {
			(i.get("sku") if isinstance(i, dict) else i.sku)
			for i in orig.items
		}
		for ri in items_to_return:
			assert ri.get("sku") in orig_skus, f"SKU {ri.get('sku')} not in original transaction"

		return_items = [
			SaleItemCreate(
				sku=ri["sku"],
				description=ri.get("description", ri["sku"]),
				quantity=ri.get("quantity", 1),
				unit_price=ri.get("unit_price", 0.0),
				discount_amount=0.0,
				line_total=round(ri.get("quantity", 1) * ri.get("unit_price", 0.0), 4),
			)
			for ri in items_to_return
		]
		refund_amount = sum(i.line_total for i in return_items)

		ret = RefundResponse(
			tenant_id=tenant_id,
			original_transaction_id=original_transaction_id,
			session_id=orig.session_id,
			terminal_id=orig.terminal_id,
			items=return_items,
			reason=rr,
			refund_amount=round(float(refund_amount), 2),
			status=TransactionStatus.PENDING,
			manager_auth_id=manager_auth_id,
			created_by=created_by,
		)
		self._store_returns.put(tenant_id, ret.id, ret)
		_log_op("initiate_return", tenant_id, ret.id)
		return ret.model_dump(mode="json")

	async def process_return_payment(
		self,
		return_id: str,
		refund_method: str,
		*,
		tenant_id: str = "default",
		mpesa_phone: str | None = None,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Process a refund payment: cash / mpesa / credit_note."""
		ret = self._store_returns.get_item(tenant_id, return_id)
		assert ret is not None, f"return not found: {return_id}"
		assert ret.status == TransactionStatus.PENDING, "return must be PENDING"

		_pm_map = {
			"cash": PaymentMethod.CASH,
			"mpesa": PaymentMethod.MOBILE_MONEY,
			"credit_note": PaymentMethod.STORE_CREDIT,
			"card": PaymentMethod.CARD_DEBIT,
		}
		pm = _pm_map.get(_normalize(refund_method), PaymentMethod.CASH)

		data = ret.model_dump()
		data["refund_method"] = pm.value
		data["status"] = TransactionStatus.COMPLETED.value
		data["refunded_at"] = _now()
		data["updated_at"] = _now()

		# Create refund transaction to maintain ledger integrity
		refund_txn = SaleTransactionResponse(
			tenant_id=tenant_id,
			session_id=ret.session_id,
			terminal_id=ret.terminal_id,
			store_id="default",
			cashier_id=created_by,
			transaction_type=TransactionType.REFUND,
			items=ret.items,
			original_transaction_id=ret.original_transaction_id,
			status=TransactionStatus.COMPLETED,
			grand_total=ret.refund_amount,
			subtotal=ret.refund_amount,
			posted_at=_now(),
			created_by=created_by,
		)
		self._store_transactions.put(tenant_id, refund_txn.id, refund_txn)
		data["refund_transaction_id"] = refund_txn.id

		# Restore inventory
		orig = self._store_transactions.get_item(tenant_id, ret.original_transaction_id)
		if orig:
			for item in ret.items:
				item_d = item if isinstance(item, dict) else item.model_dump()
				self._inventory.adjust_stock(orig.store_id, item_d["sku"], item_d["quantity"])

		updated = RefundResponse(**data)
		self._store_returns.put(tenant_id, return_id, updated)

		# Update session totals
		session = self._store_sessions.get_item(tenant_id, ret.session_id)
		if session:
			sdata = session.model_dump()
			sdata["total_refunds"] = round(sdata.get("total_refunds", 0.0) + ret.refund_amount, 2)
			sdata["updated_at"] = _now()
			self._store_sessions.put(tenant_id, ret.session_id, PosSessionResponse(**sdata))

		_log_op("process_return_payment", tenant_id, return_id)
		return updated.model_dump(mode="json")

	async def exchange_item(
		self,
		transaction_id: str,
		return_items: list[dict[str, Any]],
		new_items: list[dict[str, Any]],
		*,
		tenant_id: str = "default",
		supervisor_id: str | None = None,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Process a same-value or top-up exchange.

		Creates a return for return_items and a new sale transaction for new_items.
		If new_items total > return total, a balance payment is required.
		"""
		orig = self._store_transactions.get_item(tenant_id, transaction_id)
		assert orig is not None, f"transaction not found: {transaction_id}"

		# Initiate return for the items being given back
		return_resp = await self.initiate_return(
			transaction_id, return_items, "wrong_item",
			tenant_id=tenant_id, manager_auth_id=supervisor_id, created_by=created_by,
		)
		return_total = return_resp["refund_amount"]

		# Calculate new items total
		new_total = sum(i.get("quantity", 1) * i.get("unit_price", 0.0) for i in new_items)
		balance_due = max(0.0, round(new_total - return_total, 2))
		credit_back = max(0.0, round(return_total - new_total, 2))

		return {
			"transaction_id": transaction_id,
			"return_id": return_resp["id"],
			"return_total": return_total,
			"new_items_total": round(new_total, 2),
			"balance_due": balance_due,
			"credit_back": credit_back,
			"exchange_type": "equal" if balance_due == 0 and credit_back == 0 else (
				"upgrade" if balance_due > 0 else "downgrade"
			),
			"status": "pending_payment" if balance_due > 0 else "completed",
		}

	async def return_analytics(
		self, period: str, *, tenant_id: str = "default"
	) -> dict[str, Any]:
		"""Aggregate return/refund statistics for a reporting period."""
		period_start, period_end = self._parse_period(period)
		all_returns = [
			r for r in self._store_returns.tenant_values(tenant_id)
			if (r.refunded_at and r.refunded_at.date() >= period_start
				and r.refunded_at.date() <= period_end)
			or (not r.refunded_at and r.created_at.date() >= period_start
				and r.created_at.date() <= period_end)
		]
		by_reason: dict[str, int] = defaultdict(int)
		total_refunded = 0.0
		for r in all_returns:
			by_reason[r.reason.value if hasattr(r.reason, "value") else str(r.reason)] += 1
			total_refunded += r.refund_amount

		return {
			"period": period,
			"total_returns": len(all_returns),
			"total_refunded": round(total_refunded, 2),
			"by_reason": dict(by_reason),
			"avg_refund": round(total_refunded / len(all_returns), 2) if all_returns else 0.0,
			"generated_at": _now().isoformat(),
		}

	# ======================================================================
	# INVENTORY INTEGRATION
	# ======================================================================

	async def stock_check(
		self, sku: str, store_id: str, *, tenant_id: str = "default"
	) -> dict[str, Any]:
		"""Check current stock level for a SKU at a given store."""
		qty = self._inventory.get_stock(store_id, sku)
		price = self._inventory.get_price(tenant_id, sku)
		return {
			"sku": sku,
			"store_id": store_id,
			"quantity_on_hand": qty,
			"unit_price": price,
			"in_stock": qty > 0,
			"checked_at": _now().isoformat(),
		}

	async def price_check(
		self,
		sku: str,
		customer_tier: str | None = None,
		*,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Retrieve the selling price including any tier pricing and active promotions."""
		base_price = self._inventory.get_price(tenant_id, sku)
		tier_price = self._inventory.get_price(tenant_id, sku, customer_tier) if customer_tier else None
		effective_price = tier_price or base_price or 0.0

		promos = self._promotions.active_for_sku(tenant_id, sku, customer_tier)
		best_promo_discount = 0.0
		applied_promo = None
		for p in promos:
			if p.get("discount_type") == "percentage":
				disc = effective_price * p["value"] / 100
			else:
				disc = p.get("value", 0.0)
			if disc > best_promo_discount:
				best_promo_discount = disc
				applied_promo = p

		promotional_price = max(0.0, round(effective_price - best_promo_discount, 2))

		return {
			"sku": sku,
			"customer_tier": customer_tier,
			"base_price": base_price,
			"tier_price": tier_price,
			"effective_price": round(effective_price, 2),
			"promotional_price": promotional_price,
			"discount_amount": round(best_promo_discount, 2),
			"applied_promotion": applied_promo["id"] if applied_promo else None,
			"checked_at": _now().isoformat(),
		}

	async def inventory_deduction(
		self,
		transaction_id: str,
		*,
		tenant_id: str = "default",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Deduct inventory for all items in a completed transaction."""
		txn = self._store_transactions.get_item(tenant_id, transaction_id)
		assert txn is not None, f"transaction not found: {transaction_id}"

		movements = []
		for item in txn.items:
			item_d = item if isinstance(item, dict) else item.model_dump()
			sku = item_d["sku"]
			qty = item_d["quantity"]
			delta = -qty  # sale = negative movement

			new_qty = self._inventory.adjust_stock(txn.store_id, sku, delta)
			mov = InventoryMovementResponse(
				tenant_id=tenant_id,
				store_id=txn.store_id,
				terminal_id=txn.terminal_id,
				transaction_id=transaction_id,
				sku=sku,
				movement_type=InventoryMovementType.SALE,
				quantity_delta=delta,
				unit_cost=item_d.get("cost_price"),
				stock_before=new_qty - delta,
				stock_after=new_qty,
				created_by=created_by,
			)
			self._store_movements.put(tenant_id, mov.id, mov)
			movements.append({"sku": sku, "delta": delta, "stock_after": new_qty})

			if new_qty < 0:
				_log_warn("negative_stock", sku=sku, store_id=txn.store_id, qty=new_qty)

		return {"transaction_id": transaction_id, "movements": movements}

	async def low_stock_alerts(
		self,
		store_id: str,
		threshold_days: int = 7,
		*,
		tenant_id: str = "default",
	) -> list[dict[str, Any]]:
		"""Return SKUs whose stock level is below the average daily sales × threshold_days."""
		# Calculate average daily velocity from recent movements
		cutoff = _now() - timedelta(days=30)
		velocity: dict[str, float] = defaultdict(float)
		for mov in self._store_movements.tenant_values(tenant_id):
			if mov.store_id == store_id and mov.movement_type == InventoryMovementType.SALE:
				if mov.occurred_at >= cutoff:
					velocity[mov.sku] += abs(mov.quantity_delta)

		alerts = []
		for (sid, sku), qty in self._inventory._stock.items():
			if sid != store_id:
				continue
			daily_velocity = velocity.get(sku, 0.0) / 30
			days_of_stock = qty / daily_velocity if daily_velocity > 0 else float("inf")
			if days_of_stock < threshold_days:
				alerts.append({
					"sku": sku,
					"store_id": store_id,
					"quantity_on_hand": qty,
					"daily_velocity": round(daily_velocity, 2),
					"days_of_stock": round(days_of_stock, 1),
					"threshold_days": threshold_days,
					"reorder_qty": round(daily_velocity * threshold_days * 2, 0),
				})

		alerts.sort(key=lambda a: a["days_of_stock"])
		return alerts

	# ======================================================================
	# END OF DAY
	# ======================================================================

	async def end_of_day_report(
		self,
		store_id: str,
		report_date: str,
		*,
		tenant_id: str = "default",
		generated_by: str = "system",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Generate the End of Day report for a store."""
		target_date = date.fromisoformat(report_date)

		sessions = [
			s for s in self._store_sessions.tenant_values(tenant_id)
			if s.store_id == store_id
			and s.opened_at.date() == target_date
		]
		session_ids = {s.id for s in sessions}

		txns = [
			t for t in self._store_transactions.tenant_values(tenant_id)
			if t.store_id == store_id
			and t.session_id in session_ids
			and not t.is_deleted
		]
		sales = [t for t in txns if t.transaction_type == TransactionType.SALE and t.status == TransactionStatus.COMPLETED]
		refunds = [t for t in txns if t.transaction_type == TransactionType.REFUND and t.status == TransactionStatus.COMPLETED]
		voids = [t for t in txns if t.status == TransactionStatus.VOIDED]

		gross_sales = sum(t.grand_total for t in sales)
		total_refunds = sum(t.grand_total for t in refunds)
		total_discounts = sum(t.discount_total for t in sales)
		total_tax = sum(t.tax_total for t in sales)
		net_sales = gross_sales - total_refunds

		# Payment method breakdown
		cash_sales = card_sales = mobile_sales = loyalty_sales = other_sales = 0.0
		for t in sales:
			for p in self._get_txn_payments(tenant_id, t.id):
				match p.payment_method:
					case PaymentMethod.CASH:
						cash_sales += float(p.amount)
					case PaymentMethod.CARD_CREDIT | PaymentMethod.CARD_DEBIT:
						card_sales += float(p.amount)
					case PaymentMethod.MOBILE_MONEY:
						mobile_sales += float(p.amount)
					case PaymentMethod.LOYALTY_POINTS:
						loyalty_sales += float(p.amount)
					case _:
						other_sales += float(p.amount)

		# Floats and safe drops
		opening_floats = sum(s.opening_float for s in sessions)
		variances = sum(
			(s.closing_cash_counted or 0) - (s.expected_cash or 0)
			for s in sessions
		)

		# Hourly breakdown
		hourly: dict[int, dict[str, float]] = defaultdict(lambda: {"sales": 0.0, "count": 0})
		for t in sales:
			if t.posted_at:
				h = t.posted_at.hour
				hourly[h]["sales"] += t.grand_total
				hourly[h]["count"] += 1
		hourly_breakdown = [
			{"hour": h, "sales": round(v["sales"], 2), "transactions": int(v["count"])}
			for h, v in sorted(hourly.items())
		]

		# Top-selling SKUs by revenue
		sku_revenue: dict[str, float] = defaultdict(float)
		for t in sales:
			for item in t.items:
				item_d = item if isinstance(item, dict) else item.model_dump()
				sku_revenue[item_d["sku"]] = sku_revenue.get(item_d["sku"], 0.0) + item_d.get("line_total", 0.0)
		top_skus = sorted(
			[{"sku": k, "revenue": round(v, 2)} for k, v in sku_revenue.items()],
			key=lambda x: x["revenue"],
			reverse=True,
		)[:10]

		eod = EndOfDayReportResponse(
			tenant_id=tenant_id,
			store_id=store_id,
			business_date=report_date,
			session_count=len(sessions),
			transaction_count=len(sales),
			gross_sales=round(gross_sales, 2),
			total_refunds=round(total_refunds, 2),
			total_discounts=round(total_discounts, 2),
			total_tax=round(total_tax, 2),
			net_sales=round(net_sales, 2),
			cash_sales=round(cash_sales, 2),
			card_sales=round(card_sales, 2),
			mobile_sales=round(mobile_sales, 2),
			loyalty_sales=round(loyalty_sales, 2),
			other_sales=round(other_sales, 2),
			opening_floats_total=round(opening_floats, 2),
			safe_drops_total=0.0,
			variance_total=round(variances, 2),
			hourly_breakdown=hourly_breakdown,
			top_selling_skus=top_skus,
			status="draft",
			created_by=created_by,
		)
		self._store_eod_reports.put(tenant_id, eod.id, eod)
		_log_op("end_of_day_report", tenant_id, store_id)
		return eod.model_dump(mode="json")

	async def cash_reconciliation(
		self,
		session_ids: list[str],
		report_date: str,
		*,
		tenant_id: str = "default",
		reconciled_by: str = "system",
	) -> dict[str, Any]:
		"""Reconcile cash for a list of sessions on a given date."""
		results = []
		total_system_cash = 0.0
		total_counted_cash = 0.0
		total_variance = 0.0

		for sid in session_ids:
			session = self._store_sessions.get_item(tenant_id, sid)
			if session is None:
				continue
			system_cash = session.opening_float + session.total_cash_sales
			counted_cash = session.closing_cash_counted or 0.0
			variance = round(counted_cash - system_cash, 2)

			results.append({
				"session_id": sid,
				"cashier_id": session.cashier_id,
				"terminal_id": session.terminal_id,
				"opening_float": session.opening_float,
				"cash_sales": session.total_cash_sales,
				"system_cash": round(system_cash, 2),
				"counted_cash": counted_cash,
				"variance": variance,
				"variance_pct": round(abs(variance) / system_cash * 100, 2) if system_cash else 0.0,
				"status": "ok" if abs(variance) <= 50 else "discrepancy",
			})
			total_system_cash += system_cash
			total_counted_cash += counted_cash
			total_variance += variance

		return {
			"date": report_date,
			"session_count": len(results),
			"total_system_cash": round(total_system_cash, 2),
			"total_counted_cash": round(total_counted_cash, 2),
			"total_variance": round(total_variance, 2),
			"sessions": results,
			"reconciled_by": reconciled_by,
			"reconciled_at": _now().isoformat(),
		}

	async def sales_summary_report(
		self,
		store_id: str,
		period: str,
		*,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Produce a sales summary for a store over a period (daily, weekly, monthly)."""
		period_start, period_end = self._parse_period(period)

		txns = [
			t for t in self._store_transactions.tenant_values(tenant_id)
			if t.store_id == store_id
			and t.transaction_type == TransactionType.SALE
			and t.status == TransactionStatus.COMPLETED
			and t.posted_at
			and t.posted_at.date() >= period_start
			and t.posted_at.date() <= period_end
		]
		refunds = [
			r for r in self._store_transactions.tenant_values(tenant_id)
			if r.store_id == store_id
			and r.transaction_type == TransactionType.REFUND
			and r.status == TransactionStatus.COMPLETED
			and r.posted_at
			and r.posted_at.date() >= period_start
			and r.posted_at.date() <= period_end
		]

		gross_sales = sum(t.grand_total for t in txns)
		total_refunds = sum(r.grand_total for r in refunds)
		net_sales = gross_sales - total_refunds
		total_discounts = sum(t.discount_total for t in txns)
		total_tax = sum(t.tax_total for t in txns)

		# Daily breakdown
		daily: dict[str, dict[str, float]] = defaultdict(lambda: {"sales": 0.0, "count": 0, "refunds": 0.0})
		for t in txns:
			d = t.posted_at.date().isoformat()
			daily[d]["sales"] = round(daily[d]["sales"] + t.grand_total, 2)
			daily[d]["count"] += 1
		for r in refunds:
			d = r.posted_at.date().isoformat()
			daily[d]["refunds"] = round(daily[d].get("refunds", 0.0) + r.grand_total, 2)

		daily_breakdown = [
			{"date": k, **v} for k, v in sorted(daily.items())
		]

		return {
			"store_id": store_id,
			"period": period,
			"period_start": period_start.isoformat(),
			"period_end": period_end.isoformat(),
			"transaction_count": len(txns),
			"refund_count": len(refunds),
			"gross_sales": round(gross_sales, 2),
			"total_refunds": round(total_refunds, 2),
			"total_discounts": round(total_discounts, 2),
			"total_tax": round(total_tax, 2),
			"net_sales": round(net_sales, 2),
			"avg_transaction": round(gross_sales / len(txns), 2) if txns else 0.0,
			"daily_breakdown": daily_breakdown,
			"generated_at": _now().isoformat(),
		}

	async def till_variance_report(
		self, period: str, *, tenant_id: str = "default"
	) -> dict[str, Any]:
		"""Report till variance (actual vs expected cash) for all sessions in a period."""
		period_start, period_end = self._parse_period(period)

		sessions = [
			s for s in self._store_sessions.tenant_values(tenant_id)
			if s.opened_at.date() >= period_start
			and s.opened_at.date() <= period_end
			and s.status in (SessionStatus.CLOSED, SessionStatus.RECONCILED)
		]

		rows = []
		total_variance = 0.0
		sessions_over = 0
		sessions_under = 0
		sessions_exact = 0

		for s in sessions:
			expected = s.opening_float + s.total_cash_sales
			counted = s.closing_cash_counted or 0.0
			variance = round(counted - expected, 2)
			total_variance += variance

			if variance > 0:
				sessions_over += 1
			elif variance < 0:
				sessions_under += 1
			else:
				sessions_exact += 1

			rows.append({
				"session_id": s.id,
				"session_number": s.session_number,
				"store_id": s.store_id,
				"cashier_id": s.cashier_id,
				"terminal_id": s.terminal_id,
				"date": s.opened_at.date().isoformat(),
				"opening_float": s.opening_float,
				"cash_sales": s.total_cash_sales,
				"expected_cash": round(expected, 2),
				"counted_cash": counted,
				"variance": variance,
				"variance_flag": "over" if variance > 0 else ("under" if variance < 0 else "exact"),
			})

		return {
			"period": period,
			"total_sessions": len(sessions),
			"sessions_over": sessions_over,
			"sessions_under": sessions_under,
			"sessions_exact": sessions_exact,
			"net_variance": round(total_variance, 2),
			"sessions": rows,
			"generated_at": _now().isoformat(),
		}

	# ======================================================================
	# PROMOTIONS
	# ======================================================================

	async def apply_promotion(
		self,
		transaction_id: str,
		promo_code: str,
		*,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Apply a promotion by code to a transaction."""
		txn = self._store_transactions.get_item(tenant_id, transaction_id)
		assert txn is not None, f"transaction not found: {transaction_id}"

		promo = next(
			(p for p in self._promotions._promos.values()
			 if p.get("tenant_id") == tenant_id
			 and (p.get("coupon_code") == promo_code or p.get("id") == promo_code)
			 and p.get("is_active", True)),
			None,
		)
		assert promo is not None, f"promotion not found: {promo_code}"

		# Check usage limit
		if promo.get("max_uses") is not None:
			assert promo.get("times_used", 0) < promo["max_uses"], "promotion usage limit reached"

		# Check validity window
		now = _now()
		if promo.get("valid_from"):
			assert promo["valid_from"] <= now, "promotion not yet active"
		if promo.get("valid_until"):
			assert promo["valid_until"] >= now, "promotion has expired"

		subtotal = txn.subtotal or sum(
			(i.get("line_total") if isinstance(i, dict) else i.line_total)
			for i in txn.items
		)

		# Check minimum purchase
		if promo.get("min_purchase"):
			assert subtotal >= promo["min_purchase"], (
				f"minimum purchase {promo['min_purchase']:.2f} not met"
			)

		if promo.get("discount_type") == "percentage":
			discount_amount = round(subtotal * promo["value"] / 100, 2)
		else:
			discount_amount = min(round(promo["value"], 2), subtotal)

		self._promotions.record_use(promo["id"], discount_amount)

		data = txn.model_dump()
		data["discount_total"] = round((data.get("discount_total") or 0.0) + discount_amount, 2)
		data["grand_total"] = round(subtotal - data["discount_total"] + (data.get("tax_total") or 0.0), 2)
		data["discount_ids"] = list(set(data.get("discount_ids", []) or [])) + [promo["id"]]
		data["updated_at"] = _now()

		updated = SaleTransactionResponse(**data)
		self._store_transactions.put(tenant_id, transaction_id, updated)
		_log_op("apply_promotion", tenant_id, transaction_id)

		return {
			"transaction_id": transaction_id,
			"promo_id": promo["id"],
			"promo_code": promo_code,
			"discount_amount": discount_amount,
			"new_grand_total": data["grand_total"],
			"times_used": promo.get("times_used", 1),
		}

	async def check_active_promotions(
		self,
		sku: str,
		customer_tier: str | None = None,
		*,
		tenant_id: str = "default",
	) -> list[dict[str, Any]]:
		"""Return all active promotions that apply to the given SKU and customer tier."""
		return self._promotions.active_for_sku(tenant_id, sku, customer_tier)

	async def promotion_performance(
		self, promo_id: str, *, tenant_id: str = "default"
	) -> dict[str, Any]:
		"""Report on the performance of a specific promotion."""
		promo = self._promotions.get(promo_id)
		assert promo is not None, f"promotion not found: {promo_id}"

		return {
			"promo_id": promo_id,
			"name": promo.get("name"),
			"discount_type": promo.get("discount_type"),
			"value": promo.get("value"),
			"times_used": promo.get("times_used", 0),
			"total_discount_given": round(promo.get("total_discount_given", 0.0), 2),
			"max_uses": promo.get("max_uses"),
			"usage_rate": (
				round(promo.get("times_used", 0) / promo["max_uses"] * 100, 1)
				if promo.get("max_uses") else None
			),
			"is_active": promo.get("is_active", True),
			"valid_from": promo["valid_from"].isoformat() if promo.get("valid_from") else None,
			"valid_until": promo["valid_until"].isoformat() if promo.get("valid_until") else None,
			"generated_at": _now().isoformat(),
		}

	# ======================================================================
	# INTERNAL HELPERS
	# ======================================================================

	def _recalculate_totals(self, data: dict[str, Any]) -> None:
		"""Recompute subtotal, tax_total, discount_total, grand_total from items list."""
		items = data.get("items", [])
		subtotal = 0.0
		discount_total = data.get("discount_total") or 0.0
		tax_total = 0.0
		for item in items:
			item_d = item if isinstance(item, dict) else item.model_dump()
			lt = item_d.get("line_total", 0.0) or 0.0
			ta = item_d.get("tax_amount", 0.0) or 0.0
			subtotal += lt
			tax_total += ta
		data["subtotal"] = round(subtotal, 2)
		data["tax_total"] = round(tax_total, 2)
		data["grand_total"] = round(subtotal - (data.get("discount_total") or 0.0), 2)

	def _get_txn_payments(self, tenant_id: str, transaction_id: str) -> list[PaymentResponse]:
		return [
			p for p in self._store_payments.tenant_values(tenant_id)
			if p.transaction_id == transaction_id
		]

	def _parse_period(self, period: str) -> tuple[date, date]:
		"""Parse a period string to (start, end) date tuple."""
		import re, calendar as cal
		p = period.strip()
		if "/" in p:
			parts = p.split("/")
			return date.fromisoformat(parts[0]), date.fromisoformat(parts[1])
		qm = re.match(r"Q([1-4])-(\d{4})", p, re.IGNORECASE)
		if qm:
			q, year = int(qm.group(1)), int(qm.group(2))
			month_start = (q - 1) * 3 + 1
			month_end = month_start + 2
			return date(year, month_start, 1), date(year, month_end, cal.monthrange(year, month_end)[1])
		if re.match(r"^\d{4}-\d{2}$", p):
			year, month = int(p[:4]), int(p[5:7])
			return date(year, month, 1), date(year, month, cal.monthrange(year, month)[1])
		if re.match(r"^\d{4}$", p):
			year = int(p)
			return date(year, 1, 1), date(year, 12, 31)
		try:
			d = date.fromisoformat(p)
			return d, d
		except ValueError:
			raise ValueError(f"Cannot parse period: {period!r}")

	# ======================================================================
	# ORIGINAL PosService INTERFACE (backward-compat)
	# ======================================================================

	async def register_terminal(self, data: PosTerminalCreate) -> PosTerminalResponse:
		_log_op("register_terminal", data.tenant_id)
		rec = PosTerminalResponse(**data.model_dump())
		self._terminals[rec.id] = rec.model_dump()
		return rec

	async def get_terminal(self, tenant_id: str, terminal_id: str) -> PosTerminalResponse | None:
		rec = self._terminals.get(terminal_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		return PosTerminalResponse(**rec)

	async def list_terminals(self, tenant_id: str, store_id: str | None = None) -> list[PosTerminalResponse]:
		result = [v for v in self._terminals.values() if v["tenant_id"] == tenant_id]
		if store_id:
			result = [v for v in result if v["store_id"] == store_id]
		return [PosTerminalResponse(**v) for v in result]

	async def heartbeat_terminal(self, tenant_id: str, terminal_id: str) -> PosTerminalResponse | None:
		rec = self._terminals.get(terminal_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		rec["status"] = "online"
		rec["last_heartbeat_at"] = _now().isoformat()
		rec["updated_at"] = _now().isoformat()
		self._terminals[terminal_id] = rec
		return PosTerminalResponse(**rec)

	async def mark_terminal_offline(self, tenant_id: str, terminal_id: str) -> PosTerminalResponse | None:
		rec = self._terminals.get(terminal_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		rec["status"] = "offline"
		rec["updated_at"] = _now().isoformat()
		self._terminals[terminal_id] = rec
		return PosTerminalResponse(**rec)

	async def open_session_legacy(self, data: PosSessionCreate) -> PosSessionResponse:
		prior = self._get_open_session(data.tenant_id, data.terminal_id)
		assert prior is None, "terminal already has an open session"
		self._assert_no_unreconciled_session(data.tenant_id, data.cashier_id)
		_log_op("open_session", data.tenant_id)
		rec = PosSessionResponse(**data.model_dump())
		self._sessions[rec.id] = rec.model_dump()
		return rec

	def _get_open_session(self, tenant_id: str, terminal_id: str) -> dict[str, Any] | None:
		for s in self._sessions.values():
			if s["tenant_id"] == tenant_id and s["terminal_id"] == terminal_id and s["status"] == "open":
				return s
		return None

	def _assert_no_unreconciled_session(self, tenant_id: str, cashier_id: str) -> None:
		for s in self._sessions.values():
			if (s["tenant_id"] == tenant_id and s["cashier_id"] == cashier_id
					and s["status"] in ("closed",) and s.get("reconciled_at") is None):
				raise AssertionError("previous session must be reconciled before opening a new one")

	async def get_session_legacy(self, tenant_id: str, session_id: str) -> PosSessionResponse | None:
		rec = self._sessions.get(session_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		return PosSessionResponse(**rec)

	async def suspend_session(self, tenant_id: str, session_id: str) -> PosSessionResponse | None:
		return await self._update_session_status(tenant_id, session_id, "suspended")

	async def resume_session(self, tenant_id: str, session_id: str) -> PosSessionResponse | None:
		return await self._update_session_status(tenant_id, session_id, "open")

	async def close_session_legacy(self, tenant_id: str, session_id: str, closing_cash: float) -> PosSessionResponse | None:
		rec = self._sessions.get(session_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		rec["status"] = "closed"
		rec["closing_cash_counted"] = closing_cash
		rec["closed_at"] = _now().isoformat()
		rec["updated_at"] = _now().isoformat()
		self._sessions[session_id] = rec
		return PosSessionResponse(**rec)

	async def _update_session_status(self, tenant_id: str, session_id: str, status: str) -> PosSessionResponse | None:
		rec = self._sessions.get(session_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		rec["status"] = status
		rec["updated_at"] = _now().isoformat()
		self._sessions[session_id] = rec
		return PosSessionResponse(**rec)

	async def list_sessions(self, tenant_id: str, store_id: str | None = None, status: str | None = None) -> list[PosSessionResponse]:
		result = [v for v in self._sessions.values() if v["tenant_id"] == tenant_id]
		if store_id:
			result = [v for v in result if v["store_id"] == store_id]
		if status:
			result = [v for v in result if v["status"] == status]
		return [PosSessionResponse(**v) for v in result]

	async def post_transaction(self, data: PosTransactionCreate) -> PosTransactionResponse:
		session = self._sessions.get(data.session_id)
		assert session and session["tenant_id"] == data.tenant_id, "session not found"
		assert session["status"] == "open", "session must be open to post transactions"
		subtotal = sum(item.line_total for item in data.items)
		discount = sum(getattr(item, "discount_amount", 0.0) for item in data.items)
		tax = sum(item.tax_amount for item in data.items)
		grand_total = subtotal - discount + tax
		_log_op("post_transaction", data.tenant_id)
		rec = PosTransactionResponse(
			**data.model_dump(),
			subtotal=subtotal,
			discount_total=discount,
			tax_total=tax,
			grand_total=grand_total,
			tender_status="authorised",
			transaction_signed=True,
			signature_ref=uuid7str(),
		)
		self._transactions[rec.id] = rec.model_dump()
		if data.transaction_type == "sale":
			session["transaction_count"] = session.get("transaction_count", 0) + 1
			session["total_sales"] = session.get("total_sales", 0.0) + grand_total
			if data.payment_method == "cash":
				session["total_cash_sales"] = session.get("total_cash_sales", 0.0) + grand_total
		elif data.transaction_type == "refund":
			session["total_refunds"] = session.get("total_refunds", 0.0) + grand_total
		session["updated_at"] = _now().isoformat()
		self._sessions[data.session_id] = session
		_log_txn(rec.transaction_number, data.transaction_type, grand_total)
		return rec

	async def get_transaction(self, tenant_id: str, transaction_id: str) -> PosTransactionResponse | None:
		rec = self._transactions.get(transaction_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		return PosTransactionResponse(**rec)

	async def list_transactions(self, tenant_id: str, session_id: str | None = None) -> list[PosTransactionResponse]:
		result = [v for v in self._transactions.values() if v["tenant_id"] == tenant_id]
		if session_id:
			result = [v for v in result if v["session_id"] == session_id]
		return [PosTransactionResponse(**v) for v in result]

	async def void_transaction_legacy(self, data: PosVoidCreate) -> PosVoidResponse:
		orig = self._transactions.get(data.original_transaction_id)
		assert orig and orig["tenant_id"] == data.tenant_id, "original transaction not found"
		assert orig["terminal_id"] == data.terminal_id, "must void on originating terminal"
		_log_op("void_transaction", data.tenant_id, data.original_transaction_id)
		rec = PosVoidResponse(**data.model_dump(), void_transaction_id=uuid7str(), status="completed")
		self._voids[rec.id] = rec.model_dump()
		return rec

	async def list_voids(self, tenant_id: str, session_id: str | None = None) -> list[PosVoidResponse]:
		result = [v for v in self._voids.values() if v["tenant_id"] == tenant_id]
		if session_id:
			result = [v for v in result if v["session_id"] == session_id]
		return [PosVoidResponse(**v) for v in result]

	async def record_cash_event(self, data: PosCashEventCreate) -> PosCashEventResponse:
		session = self._sessions.get(data.session_id)
		assert session and session["tenant_id"] == data.tenant_id, "session not found"
		session_cash = self._compute_session_cash(data.tenant_id, data.session_id)
		balance_after = session_cash + data.amount
		assert balance_after >= 0, "cash event would result in negative till balance"
		_log_op("record_cash_event", data.tenant_id)
		rec = PosCashEventResponse(**data.model_dump(), balance_after=balance_after)
		self._cash_events[rec.id] = rec.model_dump()
		return rec

	def _compute_session_cash(self, tenant_id: str, session_id: str) -> float:
		session = self._sessions.get(session_id)
		base = session.get("opening_float", 0.0) if session else 0.0
		cash_sales = session.get("total_cash_sales", 0.0) if session else 0.0
		events_total = sum(
			v["amount"] for v in self._cash_events.values()
			if v["tenant_id"] == tenant_id and v["session_id"] == session_id
		)
		return base + cash_sales + events_total

	async def list_cash_events(self, tenant_id: str, session_id: str) -> list[PosCashEventResponse]:
		result = [v for v in self._cash_events.values() if v["tenant_id"] == tenant_id and v["session_id"] == session_id]
		return [PosCashEventResponse(**v) for v in result]

	async def create_reconciliation(self, data: PosReconciliationCreate) -> PosReconciliationResponse:
		variance = data.counted_cash_total - data.system_cash_total
		rec = PosReconciliationResponse(**data.model_dump(), variance=variance)
		self._reconciliations[rec.id] = rec.model_dump()
		session = self._sessions.get(data.session_id)
		if session and session["tenant_id"] == data.tenant_id:
			session["reconciled_at"] = _now().isoformat()
			session["updated_at"] = _now().isoformat()
			self._sessions[data.session_id] = session
		_log_op("create_reconciliation", data.tenant_id, rec.id)
		return rec

	async def approve_reconciliation(self, tenant_id: str, reconciliation_id: str, by: str) -> PosReconciliationResponse | None:
		rec = self._reconciliations.get(reconciliation_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		rec["status"] = "approved"
		rec["approved_by"] = by
		rec["approved_at"] = _now().isoformat()
		rec["updated_at"] = _now().isoformat()
		self._reconciliations[reconciliation_id] = rec
		return PosReconciliationResponse(**rec)

	async def get_reconciliation(self, tenant_id: str, reconciliation_id: str) -> PosReconciliationResponse | None:
		rec = self._reconciliations.get(reconciliation_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		return PosReconciliationResponse(**rec)

	async def issue_receipt(self, data: PosReceiptCreate) -> PosReceiptResponse:
		_log_op("issue_receipt", data.tenant_id)
		rec = PosReceiptResponse(**data.model_dump())
		self._receipts[rec.id] = rec.model_dump()
		return rec

	async def list_receipts(self, tenant_id: str, transaction_id: str) -> list[PosReceiptResponse]:
		result = [v for v in self._receipts.values() if v["tenant_id"] == tenant_id and v["transaction_id"] == transaction_id]
		return [PosReceiptResponse(**v) for v in result]

	async def session_summary_legacy(self, tenant_id: str, session_id: str) -> dict[str, Any]:
		session = await self.get_session_legacy(tenant_id, session_id)
		if session is None:
			return {}
		txns = await self.list_transactions(tenant_id, session_id)
		cash_events = await self.list_cash_events(tenant_id, session_id)
		return {
			"session": session.model_dump(),
			"transaction_count": len(txns),
			"total_sales": session.total_sales,
			"total_refunds": session.total_refunds,
			"net_sales": session.total_sales - session.total_refunds,
			"cash_balance": self._compute_session_cash(tenant_id, session_id),
			"cash_events": [e.model_dump() for e in cash_events],
		}


# backward-compat alias
PosService = PointOfSaleService


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _normalize(value: str) -> str:
	return value.strip().lower() if value else ""
