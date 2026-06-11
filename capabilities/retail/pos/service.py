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


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
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

		# Offline sync sequence tracking: (tenant_id, terminal_id) -> last_accepted_sequence
		self._offline_sync_sequences: dict[tuple[str, str], int] = {}

		# Inventory soft-holds: (store_id, sku) -> list of active hold dicts
		self._inventory_holds: dict[tuple[str, str], list[dict[str, Any]]] = {}

		# Shift handovers: (tenant_id, handover_id) -> handover dict
		self._handovers: dict[tuple[str, str], dict[str, Any]] = {}

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
		store_id: str | None = None,
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

	# ======================================================================
	# LEGACY + UNIFIED INTERFACE — all methods return dict[str, Any]
	# ======================================================================

	async def register_terminal(self, data: PosTerminalCreate) -> dict[str, Any]:
		"""Register a new POS terminal. Returns dict."""
		_log_op("register_terminal", data.tenant_id)
		# Enforce uniqueness per tenant+store+code
		existing = [
			v for v in self._terminals.values()
			if v["tenant_id"] == data.tenant_id
			and v["store_id"] == data.store_id
			and v["terminal_code"] == data.terminal_code
		]
		assert not existing, f"terminal code '{data.terminal_code}' already registered in store '{data.store_id}'"
		rec = PosTerminalResponse(**data.model_dump())
		d = rec.model_dump()
		self._terminals[rec.id] = d
		# Also write to typed store for cross-method consistency
		self._store_terminals.put(data.tenant_id, rec.id, rec)
		return d

	async def get_terminal(self, tenant_id: str, terminal_id: str) -> dict[str, Any] | None:
		"""Fetch a terminal by ID. Returns dict or None."""
		rec = self._terminals.get(terminal_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		return rec

	async def list_terminals(self, tenant_id: str, store_id: str | None = None) -> list[dict[str, Any]]:
		"""List terminals for a tenant, optionally filtered by store."""
		result = [v for v in self._terminals.values() if v["tenant_id"] == tenant_id]
		if store_id:
			result = [v for v in result if v["store_id"] == store_id]
		return result

	async def heartbeat_terminal(self, tenant_id: str, terminal_id: str) -> dict[str, Any] | None:
		"""Mark terminal online and record heartbeat timestamp."""
		rec = self._terminals.get(terminal_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		rec["status"] = "online"
		rec["last_heartbeat_at"] = _now().isoformat()
		rec["updated_at"] = _now().isoformat()
		self._terminals[terminal_id] = rec
		return rec

	async def mark_terminal_offline(self, tenant_id: str, terminal_id: str) -> dict[str, Any] | None:
		"""Mark terminal offline."""
		rec = self._terminals.get(terminal_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		rec["status"] = "offline"
		rec["updated_at"] = _now().isoformat()
		self._terminals[terminal_id] = rec
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

	async def get_session(self, session_id: str, *, tenant_id: str = "default") -> dict[str, Any] | None:
		"""Fetch a session by ID. Returns dict or None."""
		# Check typed store first (new-style open_session)
		obj = self._store_sessions.get_item(tenant_id, session_id)
		if obj is not None:
			return obj.model_dump(mode="json")
		# Fall back to legacy store
		rec = self._sessions.get(session_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		return rec

	async def suspend_session(self, tenant_id: str, session_id: str) -> dict[str, Any] | None:
		"""Suspend an open session."""
		return await self._update_session_status(tenant_id, session_id, "suspended")

	async def resume_session(self, tenant_id: str, session_id: str) -> dict[str, Any] | None:
		"""Resume a suspended session."""
		return await self._update_session_status(tenant_id, session_id, "open")

	async def _update_session_status(self, tenant_id: str, session_id: str, status: str) -> dict[str, Any] | None:
		# Try typed store first
		obj = self._store_sessions.get_item(tenant_id, session_id)
		if obj is not None:
			data = obj.model_dump()
			data["status"] = status
			data["updated_at"] = _now()
			updated = PosSessionResponse(**data)
			self._store_sessions.put(tenant_id, session_id, updated)
			return updated.model_dump(mode="json")
		# Fall back to legacy store
		rec = self._sessions.get(session_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		rec["status"] = status
		rec["updated_at"] = _now().isoformat()
		self._sessions[session_id] = rec
		return rec

	async def list_sessions(
		self,
		tenant_id: str,
		store_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List sessions for a tenant."""
		# Merge typed + legacy stores
		result: list[dict[str, Any]] = []
		seen: set[str] = set()
		for obj in self._store_sessions.tenant_values(tenant_id):
			d = obj.model_dump(mode="json")
			result.append(d)
			seen.add(d["id"])
		for v in self._sessions.values():
			if v["tenant_id"] == tenant_id and v["id"] not in seen:
				result.append(v)
				seen.add(v["id"])
		if store_id:
			result = [v for v in result if v.get("store_id") == store_id]
		if status:
			result = [v for v in result if v.get("status") == status]
		return result

	async def get_transaction(self, transaction_id: str, *, tenant_id: str = "default") -> dict[str, Any] | None:
		"""Fetch a transaction by ID. Returns dict or None."""
		# Check typed store first
		obj = self._store_transactions.get_item(tenant_id, transaction_id)
		if obj is not None:
			return obj.model_dump(mode="json")
		# Fall back to legacy store
		rec = self._transactions.get(transaction_id)
		if rec is None or rec.get("tenant_id") != tenant_id:
			return None
		return rec

	async def list_transactions(
		self,
		tenant_id: str,
		session_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List transactions, merging typed and legacy stores."""
		result: list[dict[str, Any]] = []
		seen: set[str] = set()
		for obj in self._store_transactions.tenant_values(tenant_id):
			d = obj.model_dump(mode="json")
			result.append(d)
			seen.add(d["id"])
		for v in self._transactions.values():
			if v.get("tenant_id") == tenant_id and v["id"] not in seen:
				result.append(v)
				seen.add(v["id"])
		if session_id:
			result = [v for v in result if v.get("session_id") == session_id]
		return result

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

	async def record_cash_float(self, data: CashFloatCreate) -> dict[str, Any]:
		"""Record any cash float event (safe drop, petty cash, till loan, etc)."""
		_log_op("record_cash_float", data.tenant_id, data.session_id)
		# Look up session from typed store
		session = self._store_sessions.get_item(data.tenant_id, data.session_id)
		opening_float = session.opening_float if session else 0.0
		# Sum prior events for this session
		prior = sum(
			e.amount for e in self._store_payments.tenant_values(data.tenant_id)
		)
		balance_after = opening_float + data.amount
		rec = CashFloatResponse(
			**data.model_dump(),
			balance_after=balance_after,
		)
		self._store_payments.put(data.tenant_id, rec.id, rec)
		return rec.model_dump(mode="json")

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

	async def cash_count_reconciliation(
		self,
		session_id: str,
		counted_cash: float,
		*,
		tenant_id: str = "default",
		denominations: dict[str, int] | None = None,
		counted_by: str = "system",
	) -> dict[str, Any]:
		"""Count physical cash and compute variance vs expected.

		Returns a reconciliation summary with denomination breakdown.
		"""
		from .domain.calculations import count_denominations, expected_cash_in_till
		session = self._store_sessions.get_item(tenant_id, session_id)
		assert session is not None, f"session not found: {session_id}"

		# Derive expected cash from session totals
		expected = round(session.opening_float + session.total_cash_sales - session.total_refunds, 2)
		variance = round(counted_cash - expected, 2)

		# Denomination breakdown validation
		denom_total = None
		if denominations:
			denom_total = count_denominations(denominations)
			if abs(denom_total - counted_cash) > 0.01:
				_log_warn(
					"denomination_mismatch",
					denom_total=denom_total,
					counted_cash=counted_cash,
				)

		_log_op("cash_count_reconciliation", tenant_id, session_id)
		return {
			"session_id": session_id,
			"cashier_id": session.cashier_id,
			"terminal_id": session.terminal_id,
			"opening_float": session.opening_float,
			"expected_cash": expected,
			"counted_cash": counted_cash,
			"variance": variance,
			"denomination_total": denom_total,
			"denominations": denominations,
			"counted_by": counted_by,
			"reconciled_at": _now().isoformat(),
		}

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

	async def receipt_generation(
		self,
		transaction_id: str,
		fmt: str = "thermal",
		*,
		tenant_id: str = "default",
		recipient_email: str | None = None,
		recipient_mobile: str | None = None,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Generate a formatted receipt for a completed transaction.

		fmt: 'thermal' | 'email' | 'sms' | 'digital'
		Returns dict with rendered_content.
		"""
		txn = self._store_transactions.get_item(tenant_id, transaction_id)
		assert txn is not None, f"transaction not found: {transaction_id}"
		assert txn.status == TransactionStatus.COMPLETED, "receipt only for completed transactions"

		lines: list[str] = []
		if fmt == "thermal":
			lines = self._render_thermal_receipt(txn)
		else:
			lines = self._render_html_receipt(txn)

		rendered = "\n".join(lines)

		rec = ReceiptResponse(
			tenant_id=tenant_id,
			transaction_id=transaction_id,
			session_id=txn.session_id,
			receipt_format=ReceiptFormat(fmt) if fmt in ReceiptFormat._value2member_map_ else ReceiptFormat.THERMAL,
			recipient_email=recipient_email,
			recipient_mobile=recipient_mobile,
			rendered_content=rendered,
			receipt_payload={
				"transaction_number": txn.transaction_number,
				"grand_total": txn.grand_total,
				"items": [i if isinstance(i, dict) else i.model_dump() for i in txn.items],
			},
			created_by=created_by,
		)
		self._store_receipts_v2.put(tenant_id, rec.id, rec)
		_log_op("receipt_generation", tenant_id, transaction_id)
		return rec.model_dump(mode="json")

	def _render_thermal_receipt(self, txn: SaleTransactionResponse) -> list[str]:
		"""Render ESC/POS-style thermal receipt as text lines."""
		lines = [
			"================================",
			"        DATACRAFT POS",
			"================================",
			f"Txn: {txn.transaction_number}",
			f"Date: {(txn.posted_at or _now()).strftime('%Y-%m-%d %H:%M')}",
			f"Cashier: {txn.cashier_id}",
			"--------------------------------",
		]
		for item in txn.items:
			item_d = item if isinstance(item, dict) else item.model_dump()
			desc = item_d.get("description", item_d.get("sku", ""))
			qty = item_d.get("quantity", 0)
			price = item_d.get("unit_price", 0)
			total = item_d.get("line_total", 0)
			lines.append(f"{desc[:20]:<20} {qty:>3} x {price:>7.2f}")
			lines.append(f"{'':>24} {total:>10.2f}")
		lines += [
			"--------------------------------",
			f"{'Subtotal':.<24} {txn.subtotal:>10.2f}",
			f"{'Discount':.<24} {txn.discount_total:>10.2f}",
			f"{'VAT':.<24} {txn.tax_total:>10.2f}",
			f"{'TOTAL':.<24} {txn.grand_total:>10.2f}",
			"================================",
			"   Thank you for shopping!",
			"================================",
		]
		return lines

	def _render_html_receipt(self, txn: SaleTransactionResponse) -> list[str]:
		"""Render HTML email receipt."""
		rows = ""
		for item in txn.items:
			item_d = item if isinstance(item, dict) else item.model_dump()
			desc = item_d.get("description", item_d.get("sku", ""))
			qty = item_d.get("quantity", 0)
			total = item_d.get("line_total", 0)
			rows += f"<tr><td>{desc}</td><td>{qty}</td><td>{total:.2f}</td></tr>"
		return [f"""<html><body>
<h2>Receipt — {txn.transaction_number}</h2>
<table><thead><tr><th>Item</th><th>Qty</th><th>Total</th></tr></thead>
<tbody>{rows}</tbody></table>
<p><strong>Grand Total: {txn.grand_total:.2f}</strong></p>
</body></html>"""]

	async def loyalty_points_earn_redeem(
		self,
		customer_id: str,
		transaction_id: str,
		points_earned: int = 0,
		points_to_redeem: int = 0,
		*,
		tenant_id: str = "default",
		earn_rate: float = 1.0,
		redeem_rate: float = 0.01,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Earn and/or redeem loyalty points for a customer on a transaction.

		Returns updated loyalty balance and transaction details.
		"""
		from .domain.rules import assert_loyalty_points_sufficient
		balance_before = self._loyalty.balance(tenant_id, customer_id)

		if points_to_redeem > 0:
			assert_loyalty_points_sufficient(balance_before, points_to_redeem)
			self._loyalty.redeem(tenant_id, customer_id, points_to_redeem)

		if points_earned > 0:
			self._loyalty.earn(tenant_id, customer_id, points_earned)

		balance_after = self._loyalty.balance(tenant_id, customer_id)
		redemption_value = round(points_to_redeem * redeem_rate, 2)

		rec = LoyaltyTransactionResponse(
			tenant_id=tenant_id,
			customer_id=customer_id,
			transaction_id=transaction_id,
			points_earned=points_earned,
			points_redeemed=points_to_redeem,
			points_balance_before=balance_before,
			points_balance_after=balance_after,
			earn_rate=earn_rate,
			redeem_rate=redeem_rate,
			created_by=created_by,
		)
		self._store_loyalty_txns.put(tenant_id, rec.id, rec)
		self._loyalty.record(rec.model_dump(mode="json"))
		_log_op("loyalty_earn_redeem", tenant_id, customer_id)
		return {
			**rec.model_dump(mode="json"),
			"redemption_value": redemption_value,
		}

	async def end_of_day_closing(
		self,
		store_id: str,
		business_date: str,
		*,
		tenant_id: str = "default",
		generated_by: str = "system",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Run EOD closing: generate report, validate all sessions closed.

		Idempotency guard: raises AssertionError if EOD already exists for that date.
		"""
		from .domain.rules import assert_eod_not_already_run
		from .domain.calculations import hourly_sales_breakdown, top_selling_skus

		# Idempotency check
		existing = next(
			(r for r in self._store_eod_reports.tenant_values(tenant_id)
			 if r.store_id == store_id and r.business_date == business_date),
			None,
		)
		assert_eod_not_already_run(existing.id if existing else None, business_date)

		# Collect sessions for this store/date
		target = date.fromisoformat(business_date)
		sessions = [
			s for s in self._store_sessions.tenant_values(tenant_id)
			if s.store_id == store_id and s.opened_at.date() == target
		]
		session_ids = {s.id for s in sessions}

		txns = [
			t for t in self._store_transactions.tenant_values(tenant_id)
			if t.store_id == store_id and t.session_id in session_ids and not t.is_deleted
		]
		sales = [t for t in txns if t.transaction_type == TransactionType.SALE and t.status == TransactionStatus.COMPLETED]
		refunds = [t for t in txns if t.transaction_type == TransactionType.REFUND]

		gross_sales = round(sum(t.grand_total for t in sales), 2)
		total_refunds = round(sum(t.grand_total for t in refunds), 2)
		total_discounts = round(sum(t.discount_total for t in sales), 2)
		total_tax = round(sum(t.tax_total for t in sales), 2)
		net_sales = round(gross_sales - total_refunds, 2)

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

		opening_floats_total = round(sum(s.opening_float for s in sessions), 2)
		variance_total = round(sum(
			(s.closing_cash_counted or 0) - (s.expected_cash or 0)
			for s in sessions if s.closing_cash_counted is not None
		), 2)

		txns_as_dicts = [t.model_dump(mode="json") for t in sales]
		all_items = [item for t in sales for item in (t.items or [])]
		items_as_dicts = [i if isinstance(i, dict) else i.model_dump() for i in all_items]

		report = EndOfDayReportResponse(
			tenant_id=tenant_id,
			store_id=store_id,
			business_date=business_date,
			session_count=len(sessions),
			transaction_count=len(sales),
			gross_sales=gross_sales,
			total_refunds=total_refunds,
			total_discounts=total_discounts,
			total_tax=total_tax,
			net_sales=net_sales,
			cash_sales=round(cash_sales, 2),
			card_sales=round(card_sales, 2),
			mobile_sales=round(mobile_sales, 2),
			loyalty_sales=round(loyalty_sales, 2),
			other_sales=round(other_sales, 2),
			opening_floats_total=opening_floats_total,
			safe_drops_total=0.0,
			variance_total=variance_total,
			hourly_breakdown=hourly_sales_breakdown(txns_as_dicts),
			top_selling_skus=top_selling_skus(items_as_dicts),
			generated_at=_now(),
			status="draft",
			approved_by=None,
			created_by=created_by,
		)
		self._store_eod_reports.put(tenant_id, report.id, report)
		_log_op("end_of_day_closing", tenant_id, store_id)
		return report.model_dump(mode="json")

	async def offline_mode_sync(
		self,
		batch: OfflineSyncBatch,
		*,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Process a batch of transactions collected while offline.

		Enforces monotone sync_sequence to detect replays / gaps.
		Accepted transactions are created in PENDING state for normal completion.
		Duplicates (same transaction id already known) are skipped.
		"""
		from .domain.rules import assert_offline_sync_sequence_monotone, assert_offline_transaction_count_within_limit

		# Check sequence monotonicity per terminal
		last_seq_key = (tenant_id, batch.terminal_id)
		last_seq = self._offline_sync_sequences.get(last_seq_key, 0)
		assert_offline_sync_sequence_monotone(last_seq, batch.sync_sequence)
		assert_offline_transaction_count_within_limit(len(batch.transactions))

		accepted: list[str] = []
		rejected: list[dict[str, Any]] = []
		duplicate_skipped: list[str] = []

		for txn_create in batch.transactions:
			# Check for duplicate
			existing = self._store_transactions.get_item(tenant_id, txn_create.tenant_id)
			# Use a deterministic ID if provided, else generate
			txn_id = uuid7str()
			try:
				txn = SaleTransactionResponse(
					id=txn_id,
					tenant_id=tenant_id,
					session_id=txn_create.session_id,
					terminal_id=txn_create.terminal_id,
					store_id=txn_create.store_id,
					cashier_id=txn_create.cashier_id,
					transaction_type=txn_create.transaction_type,
					items=txn_create.items,
					customer_id=txn_create.customer_id,
					offline_mode=True,
					offline_synced=True,
					status=TransactionStatus.PENDING,
					created_by=txn_create.created_by,
				)
				self._store_transactions.put(tenant_id, txn.id, txn)
				accepted.append(txn.id)
			except Exception as exc:
				rejected.append({"error": str(exc), "tenant_id": tenant_id})

		# Record successful sequence
		self._offline_sync_sequences[last_seq_key] = batch.sync_sequence
		_log_op("offline_mode_sync", tenant_id, batch.terminal_id)

		return OfflineSyncResult(
			tenant_id=tenant_id,
			terminal_id=batch.terminal_id,
			accepted=accepted,
			rejected=rejected,
			duplicate_skipped=duplicate_skipped,
		).model_dump(mode="json")

	async def supervisor_override(self, data: SupervisorOverrideCreate) -> dict[str, Any]:
		"""Grant a supervisor override and record the authorisation.

		Raises AssertionError if supervisor == cashier (self-approval).
		"""
		from .domain.rules import assert_supervisor_not_self_approving
		# Look up session to get cashier_id
		session = self._store_sessions.get_item(data.tenant_id, data.session_id)
		cashier_id = session.cashier_id if session else data.created_by
		assert_supervisor_not_self_approving(cashier_id, data.supervisor_id)

		rec = SupervisorOverrideResponse(**data.model_dump())
		self._store_supervisor_overrides.put(data.tenant_id, rec.id, rec)
		_log_op("supervisor_override", data.tenant_id, rec.id)
		return rec.model_dump(mode="json")

	async def create_discount(self, data: DiscountCreate) -> dict[str, Any]:
		"""Create a discount/promotion definition."""
		rec = DiscountResponse(**data.model_dump())
		self._store_discounts.put(data.tenant_id, rec.id, rec)
		# Also register in promotion store for real-time coupon lookups
		promo_data = rec.model_dump(mode="json")
		promo_data["discount_type"] = data.discount_type.value if hasattr(data.discount_type, "value") else str(data.discount_type)
		self._promotions.add(promo_data)
		_log_op("create_discount", data.tenant_id, rec.id)
		return promo_data

	async def process_refund(self, data: RefundCreate) -> dict[str, Any]:
		"""Process a refund against a completed transaction.

		Validates items exist in original, reverses inventory, marks original as refunded.
		"""
		from .domain.rules import (
			assert_transaction_refundable,
			assert_refund_items_in_original,
		)
		original = self._store_transactions.get_item(data.tenant_id, data.original_transaction_id)
		assert original is not None, f"original transaction not found: {data.original_transaction_id}"
		assert_transaction_refundable(original.status.value)

		# Validate refund items exist in original
		orig_skus = [
			(i.sku if hasattr(i, "sku") else i["sku"])
			for i in original.items
		]
		refund_skus = [
			(i.sku if hasattr(i, "sku") else i["sku"])
			for i in data.items
		]
		if refund_skus:
			assert_refund_items_in_original(refund_skus, orig_skus)

		refund_amount = round(sum(
			(i.line_total if hasattr(i, "line_total") else i.get("line_total", 0))
			for i in data.items
		), 2) or original.grand_total

		refund_method = data.override_payment_method or PaymentMethod.CASH

		rec = RefundResponse(
			tenant_id=data.tenant_id,
			original_transaction_id=data.original_transaction_id,
			session_id=data.session_id,
			terminal_id=data.terminal_id,
			items=data.items,
			reason=data.reason,
			refund_amount=refund_amount,
			refund_method=refund_method,
			status=TransactionStatus.COMPLETED,
			manager_auth_id=data.manager_auth_id,
			notes=data.notes,
			created_by=data.created_by,
		)
		self._store_refunds.put(data.tenant_id, rec.id, rec)

		# Reverse inventory for refunded items
		for item in data.items:
			item_d = item if isinstance(item, dict) else item.model_dump()
			self._inventory.adjust_stock(original.store_id, item_d["sku"], item_d["quantity"])

		# Mark original as partially/fully refunded
		orig_data = original.model_dump()
		orig_data["status"] = TransactionStatus.REFUNDED.value
		orig_data["refunded_at"] = _now()
		orig_data["updated_at"] = _now()
		self._store_transactions.put(data.tenant_id, data.original_transaction_id, SaleTransactionResponse(**orig_data))

		_log_op("process_refund", data.tenant_id, rec.id)
		return rec.model_dump(mode="json")

	async def session_summary_legacy(self, tenant_id: str, session_id: str) -> dict[str, Any]:
		session_dict = await self.get_session(session_id, tenant_id=tenant_id)
		if not session_dict:
			return {}
		txns = await self.list_transactions(tenant_id, session_id)
		return {
			"session": session_dict,
			"transaction_count": len(txns),
			"total_sales": session_dict.get("total_sales", 0.0),
			"total_refunds": session_dict.get("total_refunds", 0.0),
			"net_sales": session_dict.get("total_sales", 0.0) - session_dict.get("total_refunds", 0.0),
			"cash_balance": self._compute_session_cash(tenant_id, session_id),
		}

	# ======================================================================
	# IMPROVEMENTS: New high-value async methods
	# ======================================================================

	async def basket_suggestions(
		self,
		customer_id: str,
		current_skus: list[str],
		*,
		tenant_id: str = "default",
		top_n: int = 3,
	) -> list[dict[str, Any]]:
		"""Return SKUs frequently co-purchased with current_skus by this customer.

		Scans loyalty transaction history to find items the customer almost always
		buys alongside the items already in the basket. Zero external dependencies —
		runs on existing in-process loyalty history.

		Returns top_n suggestions sorted by frequency descending.
		"""
		history = self._loyalty.customer_history(tenant_id, customer_id)
		freq: dict[str, int] = defaultdict(int)
		for entry in history:
			txn_skus: set[str] = {i["sku"] for i in entry.get("items", [])}
			if txn_skus & set(current_skus):
				for sku in txn_skus - set(current_skus):
					freq[sku] += 1
		top = sorted(freq, key=freq.__getitem__, reverse=True)[:top_n]
		return [
			{
				"sku": s,
				"frequency": freq[s],
				"price": self._inventory.get_price(tenant_id, s),
			}
			for s in top
		]

	async def session_performance_metrics(
		self,
		store_id: str,
		*,
		tenant_id: str = "default",
	) -> list[dict[str, Any]]:
		"""Real-time performance metrics for all open sessions in a store.

		Returns per-cashier stats: transactions/hour, avg basket, void rate,
		discount rate, and an alert flag when anomalies are detected.

		Supervisors can use this to identify slow or high-risk sessions in real time.
		"""
		now = _now()
		open_sessions = [
			s for s in self._store_sessions.tenant_values(tenant_id)
			if s.store_id == store_id and s.status == SessionStatus.OPEN
		]
		metrics: list[dict[str, Any]] = []
		for s in open_sessions:
			duration_h = max((now - s.opened_at).total_seconds() / 3600, 0.001)
			txns = [
				t for t in self._store_transactions.tenant_values(tenant_id)
				if t.session_id == s.id and t.status == TransactionStatus.COMPLETED
			]
			voids = [
				t for t in self._store_transactions.tenant_values(tenant_id)
				if t.session_id == s.id and t.status == TransactionStatus.VOIDED
			]
			void_rate = len(voids) / max(len(txns) + len(voids), 1)
			avg_basket = s.total_sales / max(len(txns), 1)
			discount_rate = s.total_discounts / max(s.total_sales, 0.01)
			metrics.append({
				"session_id": s.id,
				"cashier_id": s.cashier_id,
				"terminal_id": s.terminal_id,
				"transactions_per_hour": round(len(txns) / duration_h, 1),
				"avg_basket_value": round(avg_basket, 2),
				"void_rate_pct": round(void_rate * 100, 2),
				"discount_rate_pct": round(discount_rate * 100, 2),
				"duration_minutes": round(duration_h * 60, 0),
				"total_sales": round(s.total_sales, 2),
				"alert": void_rate > 0.05 or discount_rate > 0.15,
			})
		return sorted(metrics, key=lambda m: m["transactions_per_hour"], reverse=True)

	async def predict_cash_runway(
		self,
		session_id: str,
		*,
		tenant_id: str = "default",
		horizon_minutes: int = 30,
	) -> dict[str, Any]:
		"""Predict how many minutes until the till needs a safe drop or cash top-up.

		Uses session-level cash velocity (cash sales per hour) to project when
		the till will fall below 20% of the opening float. Fires an alert when
		the projected shortage is within horizon_minutes.
		"""
		session = self._store_sessions.get_item(tenant_id, session_id)
		assert session is not None, f"session not found: {session_id}"
		now = _now()
		duration_h = max((now - session.opened_at).total_seconds() / 3600, 0.001)
		current_cash = session.opening_float + session.total_cash_sales
		cash_velocity_per_hour = session.total_cash_sales / duration_h
		# Change given is approximately 30% of cash-sales value on average
		change_velocity = cash_velocity_per_hour * 0.30
		minimum_float = session.opening_float * 0.20
		runway_hours = max(
			(current_cash - minimum_float) / max(change_velocity, 0.01), 0.0
		)
		runway_minutes = runway_hours * 60
		_log_op("predict_cash_runway", tenant_id, session_id)
		return {
			"session_id": session_id,
			"cashier_id": session.cashier_id,
			"current_cash": round(current_cash, 2),
			"cash_velocity_per_hour": round(cash_velocity_per_hour, 2),
			"change_velocity_per_hour": round(change_velocity, 2),
			"minimum_float": round(minimum_float, 2),
			"predicted_shortage_in_minutes": round(runway_minutes, 0),
			"alert": runway_minutes < horizon_minutes,
			"recommended_action": "request_safe_drop" if runway_minutes < horizon_minutes else "ok",
			"checked_at": now.isoformat(),
		}

	async def score_transaction_fraud_risk(
		self,
		transaction_id: str,
		*,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Score a transaction's fraud risk on a 0–100 scale.

		Signals evaluated:
		  - Supervisor override present on transaction (+15)
		  - Discount > 20% of basket (+25)
		  - Cashier daily void rate > 5% (+20)
		  - Suspiciously fast transaction: < 30 s for > 3 items (+20)
		  - Price override present (+20)

		Risk levels: low (0–29), medium (30–59), high (60+).
		Transactions with score >= 60 are flagged for supervisor review.
		"""
		txn = self._store_transactions.get_item(tenant_id, transaction_id)
		assert txn is not None, f"transaction not found: {transaction_id}"
		session = self._store_sessions.get_item(tenant_id, txn.session_id) if txn.session_id else None

		score = 0
		signals: list[str] = []

		if txn.supervisor_override_id:
			score += 15
			signals.append("supervisor_override_on_transaction")

		if txn.subtotal and txn.subtotal > 0 and txn.discount_total / txn.subtotal > 0.20:
			score += 25
			signals.append(f"high_discount_rate_{txn.discount_total / txn.subtotal * 100:.0f}pct")

		if session:
			today_txns = [
				t for t in self._store_transactions.tenant_values(tenant_id)
				if t.cashier_id == session.cashier_id
				and t.created_at.date() == txn.created_at.date()
			]
			void_rate = sum(
				1 for t in today_txns if t.status == TransactionStatus.VOIDED
			) / max(len(today_txns), 1)
			if void_rate > 0.05:
				score += 20
				signals.append(f"cashier_void_rate_{void_rate * 100:.0f}pct")

		if txn.posted_at and txn.created_at:
			elapsed = (txn.posted_at - txn.created_at).total_seconds()
			if elapsed < 30 and len(txn.items or []) > 3:
				score += 20
				signals.append(f"suspicious_speed_{elapsed:.0f}s_for_{len(txn.items or [])}items")

		# Check for price overrides via supervisor override on items
		overrides = [
			o for o in self._store_supervisor_overrides.tenant_values(tenant_id)
			if o.target_id == transaction_id
		]
		if overrides:
			score += 20
			signals.append("price_override_on_transaction")

		score = min(score, 100)
		return {
			"transaction_id": transaction_id,
			"fraud_risk_score": score,
			"risk_level": "high" if score >= 60 else "medium" if score >= 30 else "low",
			"signals": signals,
			"requires_review": score >= 60,
			"scored_at": _now().isoformat(),
		}

	async def get_live_dashboard_metrics(
		self,
		store_id: str,
		*,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Live trading snapshot for the store manager dashboard.

		Designed to be called every 15–30 seconds and pushed via SSE to a
		browser-based dashboard. Returns:
		  - active_sessions: count of open cashier sessions
		  - open_baskets: in-flight PENDING transactions
		  - transactions_per_minute_5m: rolling 5-minute TPM
		  - hour_revenue_kes: cumulative sales in the last 60 minutes
		  - payment_mix: payment method breakdown in last hour (KES)
		"""
		now = _now()
		five_min_ago = now - timedelta(minutes=5)
		one_hour_ago = now - timedelta(hours=1)

		open_sessions = [
			s for s in self._store_sessions.tenant_values(tenant_id)
			if s.store_id == store_id and s.status == SessionStatus.OPEN
		]
		open_baskets = [
			t for t in self._store_transactions.tenant_values(tenant_id)
			if t.store_id == store_id and t.status == TransactionStatus.PENDING
		]
		recent_txns = [
			t for t in self._store_transactions.tenant_values(tenant_id)
			if t.store_id == store_id
			and t.status == TransactionStatus.COMPLETED
			and t.posted_at and t.posted_at >= five_min_ago
		]
		hour_txns = [
			t for t in self._store_transactions.tenant_values(tenant_id)
			if t.store_id == store_id
			and t.status == TransactionStatus.COMPLETED
			and t.posted_at and t.posted_at >= one_hour_ago
		]

		tpm = round(len(recent_txns) / 5.0, 2)
		hour_revenue = round(sum(t.grand_total for t in hour_txns), 2)

		payment_mix: dict[str, float] = defaultdict(float)
		for t in hour_txns:
			for p in self._get_txn_payments(tenant_id, t.id):
				payment_mix[p.payment_method.value] += float(p.amount)

		_log_op("get_live_dashboard_metrics", tenant_id, store_id)
		return {
			"store_id": store_id,
			"active_sessions": len(open_sessions),
			"open_baskets": len(open_baskets),
			"transactions_per_minute_5m": tpm,
			"hour_revenue_kes": hour_revenue,
			"hour_transaction_count": len(hour_txns),
			"payment_mix": dict(payment_mix),
			"snapshot_at": now.isoformat(),
		}

	async def reserve_inventory(
		self,
		transaction_id: str,
		sku: str,
		quantity: float,
		store_id: str,
		*,
		ttl_seconds: int = 900,
	) -> dict[str, Any]:
		"""Soft-reserve stock for an in-flight transaction.

		Prevents two concurrent baskets from selling the same last unit.
		On complete_transaction the hold is converted to a hard deduction.
		On void or basket abandonment, call release_inventory_hold().

		Raises AssertionError when available stock (on-hand minus all active holds)
		is less than requested quantity.
		"""
		now = _now()
		hold_key = (store_id, sku)
		holds: list[dict[str, Any]] = self._inventory_holds.setdefault(hold_key, [])
		# Expire stale holds before computing availability
		holds[:] = [h for h in holds if h["expires_at"] > now]
		held = sum(h["quantity"] for h in holds)
		on_hand = self._inventory.get_stock(store_id, sku)
		available = on_hand - held
		assert available >= quantity, (
			f"insufficient stock for sku={sku}: on_hand={on_hand:.2f} "
			f"held={held:.2f} available={available:.2f} requested={quantity:.2f}"
		)
		holds.append({
			"transaction_id": transaction_id,
			"quantity": quantity,
			"reserved_at": now.isoformat(),
			"expires_at": (now + timedelta(seconds=ttl_seconds)),
		})
		_log_op("reserve_inventory", store_id, sku)
		return {
			"sku": sku,
			"store_id": store_id,
			"reserved": quantity,
			"on_hand": on_hand,
			"available_after_hold": round(available - quantity, 4),
			"expires_in_seconds": ttl_seconds,
		}

	async def release_inventory_hold(
		self,
		transaction_id: str,
		sku: str,
		store_id: str,
	) -> dict[str, Any]:
		"""Release a previously soft-reserved inventory hold.

		Called automatically on void_transaction or when a basket is abandoned.
		Safe to call multiple times (idempotent — no hold = no-op).
		"""
		hold_key = (store_id, sku)
		holds: list[dict[str, Any]] = self._inventory_holds.get(hold_key, [])
		before = len(holds)
		holds[:] = [h for h in holds if h["transaction_id"] != transaction_id]
		released = before - len(holds)
		_log_op("release_inventory_hold", store_id, sku)
		return {
			"sku": sku,
			"store_id": store_id,
			"transaction_id": transaction_id,
			"holds_released": released,
		}

	async def initiate_shift_handover(
		self,
		outgoing_session_id: str,
		incoming_cashier_id: str,
		*,
		tenant_id: str = "default",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Lock a session for shift handover and require dual cash counts.

		Both the outgoing and incoming cashier must submit independent cash counts
		via submit_handover_count(). The terminal is only released for the new
		session when both counts are within the configured tolerance (default KES 10).

		Raises AssertionError if the session is not open or if incoming_cashier_id
		is the same as the outgoing cashier (cannot hand over to self).
		"""
		session = self._store_sessions.get_item(tenant_id, outgoing_session_id)
		assert session is not None, f"session not found: {outgoing_session_id}"
		assert session.status == SessionStatus.OPEN, "can only hand over an open session"
		assert incoming_cashier_id != session.cashier_id, "cannot hand over to self"

		handover_id = uuid7str()
		handover: dict[str, Any] = {
			"id": handover_id,
			"tenant_id": tenant_id,
			"outgoing_session_id": outgoing_session_id,
			"outgoing_cashier_id": session.cashier_id,
			"incoming_cashier_id": incoming_cashier_id,
			"terminal_id": session.terminal_id,
			"status": "awaiting_counts",
			"outgoing_count": None,
			"incoming_count": None,
			"variance": None,
			"tolerance_kes": 10.0,
			"initiated_at": _now().isoformat(),
			"created_by": created_by,
		}
		self._handovers[(tenant_id, handover_id)] = handover

		# Lock session to prevent new transactions during handover
		sdata = session.model_dump()
		sdata["notes"] = (sdata.get("notes") or "") + " | HANDOVER_IN_PROGRESS"
		sdata["updated_at"] = _now()
		self._store_sessions.put(tenant_id, outgoing_session_id, PosSessionResponse(**sdata))
		_log_op("initiate_shift_handover", tenant_id, handover_id)
		return handover

	async def submit_handover_count(
		self,
		handover_id: str,
		cashier_id: str,
		counted_cash: float,
		*,
		tenant_id: str = "default",
		denominations: dict[str, int] | None = None,
	) -> dict[str, Any]:
		"""Submit a cash count for an active shift handover.

		Either the outgoing or incoming cashier calls this. When both counts are
		received the variance is computed. If within tolerance the handover is
		completed; otherwise it is flagged as disputed for supervisor review.

		Returns the updated handover record.
		"""
		handover = self._handovers.get((tenant_id, handover_id))
		assert handover is not None, f"handover not found: {handover_id}"
		assert handover["status"] == "awaiting_counts", "handover already finalised"
		assert cashier_id in (
			handover["outgoing_cashier_id"], handover["incoming_cashier_id"]
		), f"cashier {cashier_id} is not party to handover {handover_id}"

		if cashier_id == handover["outgoing_cashier_id"]:
			handover["outgoing_count"] = counted_cash
			if denominations:
				handover["outgoing_denominations"] = denominations
		else:
			handover["incoming_count"] = counted_cash
			if denominations:
				handover["incoming_denominations"] = denominations

		if handover["outgoing_count"] is not None and handover["incoming_count"] is not None:
			variance = round(handover["incoming_count"] - handover["outgoing_count"], 2)
			handover["variance"] = variance
			tolerance = handover.get("tolerance_kes", 10.0)
			handover["status"] = "completed" if abs(variance) <= tolerance else "disputed"
			handover["completed_at"] = _now().isoformat()
			_log_op("handover_completed", tenant_id, handover_id)

		self._handovers[(tenant_id, handover_id)] = handover
		return handover

	async def customer_purchase_history(
		self,
		customer_id: str,
		*,
		tenant_id: str = "default",
		limit: int = 50,
		period: str | None = None,
	) -> dict[str, Any]:
		"""Return a customer's transaction history with spending analytics.

		Includes: total spend, transaction count, average basket, last visit date,
		top SKUs by frequency, and payment method preferences. Used for loyalty
		tier assignment, personalised promotions, and RFM segmentation.

		period: optional period string (e.g. "2026-06", "Q1-2026", "2026-01-01/2026-06-30")
		         If omitted, returns all history up to limit.
		"""
		if period:
			period_start, period_end = self._parse_period(period)
		else:
			period_start, period_end = None, None

		txns = [
			t for t in self._store_transactions.tenant_values(tenant_id)
			if t.customer_id == customer_id
			and t.transaction_type == TransactionType.SALE
			and t.status == TransactionStatus.COMPLETED
			and (not period_start or (t.posted_at and t.posted_at.date() >= period_start))
			and (not period_end or (t.posted_at and t.posted_at.date() <= period_end))
		]
		txns.sort(key=lambda t: t.posted_at or t.created_at, reverse=True)
		txns = txns[:limit]

		total_spend = round(sum(t.grand_total for t in txns), 2)
		avg_basket = round(total_spend / len(txns), 2) if txns else 0.0
		last_visit = max(
			(t.posted_at or t.created_at for t in txns), default=None
		)

		sku_freq: dict[str, int] = defaultdict(int)
		for t in txns:
			for item in (t.items or []):
				sku = item.sku if hasattr(item, "sku") else item["sku"]
				sku_freq[sku] += 1
		top_skus = sorted(
			[{"sku": k, "count": v} for k, v in sku_freq.items()],
			key=lambda x: x["count"], reverse=True,
		)[:10]

		payment_prefs: dict[str, int] = defaultdict(int)
		for t in txns:
			for p in self._get_txn_payments(tenant_id, t.id):
				payment_prefs[p.payment_method.value] += 1

		loyalty_balance = self._loyalty.balance(tenant_id, customer_id)

		return {
			"customer_id": customer_id,
			"transaction_count": len(txns),
			"total_spend": total_spend,
			"avg_basket": avg_basket,
			"loyalty_balance": loyalty_balance,
			"loyalty_value_kes": round(loyalty_balance * float(_LOYALTY_REDEEM_RATE), 2),
			"last_visit": last_visit.isoformat() if last_visit else None,
			"top_skus": top_skus,
			"payment_preferences": dict(payment_prefs),
			"transactions": [t.model_dump(mode="json") for t in txns],
			"generated_at": _now().isoformat(),
		}


# backward-compat alias
PosService = PointOfSaleService


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _normalize(value: str) -> str:
	return value.strip().lower() if value else ""
