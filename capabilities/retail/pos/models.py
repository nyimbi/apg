"""Pydantic v2 models for APG Point of Sale.

Covers the complete entity lifecycle:
  POSTerminal · POSSession · SaleTransaction · SaleItem · Payment
  Refund · CashFloat · EndOfDayReport · InventoryMovement
  PriceOverride · Discount · Receipt · CustomerDisplay · LoyaltyAccount
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Annotated
from pydantic.functional_validators import AfterValidator
from uuid6 import uuid7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def uuid7str() -> str:
	return str(uuid7())


def _non_empty(v: str) -> str:
	assert v and v.strip(), "must be non-empty"
	return v.strip()


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]
_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
_CFG_OPEN = ConfigDict(extra="ignore", validate_by_name=True, validate_by_alias=True)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TerminalType(str, Enum):
	FIXED_COUNTER = "fixed_counter"
	MOBILE = "mobile"
	SELF_SERVICE = "self_service"
	KIOSK = "kiosk"
	MPOS = "mpos"


class TerminalStatus(str, Enum):
	OFFLINE = "offline"
	ONLINE = "online"
	IN_SESSION = "in_session"
	MAINTENANCE = "maintenance"
	SUSPENDED = "suspended"


class SessionStatus(str, Enum):
	OPEN = "open"
	SUSPENDED = "suspended"
	CLOSED = "closed"
	RECONCILED = "reconciled"
	FORCE_CLOSED = "force_closed"


class TransactionType(str, Enum):
	SALE = "sale"
	REFUND = "refund"
	EXCHANGE = "exchange"
	VOID = "void"
	LAYAWAY = "layaway"
	LAYAWAY_PICKUP = "layaway_pickup"
	NO_SALE = "no_sale"


class TransactionStatus(str, Enum):
	PENDING = "pending"
	AUTHORISED = "authorised"
	PARTIALLY_PAID = "partially_paid"
	COMPLETED = "completed"
	VOIDED = "voided"
	SUSPENDED = "suspended"
	REFUNDED = "refunded"
	PARTIALLY_REFUNDED = "partially_refunded"


class PaymentMethod(str, Enum):
	CASH = "cash"
	CARD_CREDIT = "card_credit"
	CARD_DEBIT = "card_debit"
	MOBILE_MONEY = "mobile_money"
	LOYALTY_POINTS = "loyalty_points"
	GIFT_CARD = "gift_card"
	STORE_CREDIT = "store_credit"
	CHEQUE = "cheque"
	BANK_TRANSFER = "bank_transfer"


class PaymentStatus(str, Enum):
	PENDING = "pending"
	AUTHORISED = "authorised"
	CAPTURED = "captured"
	DECLINED = "declined"
	REVERSED = "reversed"
	REFUNDED = "refunded"


class DiscountType(str, Enum):
	PERCENTAGE = "percentage"
	FIXED_AMOUNT = "fixed_amount"
	BUY_X_GET_Y = "buy_x_get_y"
	BUNDLE = "bundle"
	LOYALTY = "loyalty"
	COUPON = "coupon"
	STAFF = "staff"
	MANAGER = "manager"


class CashEventType(str, Enum):
	OPENING_FLOAT = "opening_float"
	PETTY_CASH_OUT = "petty_cash_out"
	PETTY_CASH_IN = "petty_cash_in"
	SAFE_DROP = "safe_drop"
	SAFE_PICKUP = "safe_pickup"
	TILL_LOAN = "till_loan"
	CORRECTION = "correction"


class RefundReason(str, Enum):
	DEFECTIVE = "defective"
	WRONG_ITEM = "wrong_item"
	CUSTOMER_CHANGE_MIND = "customer_change_mind"
	OVERCHARGE = "overcharge"
	DUPLICATE = "duplicate"
	NOT_AS_DESCRIBED = "not_as_described"
	OTHER = "other"


class ReceiptFormat(str, Enum):
	THERMAL = "thermal"
	EMAIL = "email"
	SMS = "sms"
	DIGITAL = "digital"
	BOTH = "both"


class PriceOverrideReason(str, Enum):
	PRICE_MATCH = "price_match"
	DAMAGE = "damage"
	CLEARANCE = "clearance"
	CUSTOMER_COMPLAINT = "customer_complaint"
	MANAGER_DISCRETION = "manager_discretion"


class InventoryMovementType(str, Enum):
	SALE = "sale"
	REFUND = "refund"
	ADJUSTMENT = "adjustment"
	TRANSFER = "transfer"
	WRITE_OFF = "write_off"


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class PosBase(BaseModel):
	"""Common fields shared by all POS entities."""
	model_config = _CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr
	is_deleted: bool = False


# ---------------------------------------------------------------------------
# POSTerminal
# ---------------------------------------------------------------------------

class PosTerminalCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	store_id: NonEmptyStr
	terminal_code: NonEmptyStr
	terminal_type: TerminalType = TerminalType.FIXED_COUNTER
	serial_number: str | None = None
	hardware_model: str | None = None
	offline_capable: bool = True
	floor_limit: float = 5000.0
	tax_profile_id: str | None = None
	default_currency: str = "KES"
	created_by: NonEmptyStr


class PosTerminalUpdate(BaseModel):
	model_config = _CFG
	terminal_type: TerminalType | None = None
	hardware_model: str | None = None
	offline_capable: bool | None = None
	floor_limit: float | None = None
	tax_profile_id: str | None = None
	status: TerminalStatus | None = None
	updated_by: NonEmptyStr


class PosTerminalResponse(PosTerminalCreate, PosBase):
	status: TerminalStatus = TerminalStatus.OFFLINE
	last_heartbeat_at: datetime | None = None
	current_session_id: str | None = None
	firmware_version: str | None = None


# ---------------------------------------------------------------------------
# POSSession
# ---------------------------------------------------------------------------

class PosSessionCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	terminal_id: NonEmptyStr
	store_id: NonEmptyStr
	cashier_id: NonEmptyStr
	opening_float: float
	supervisor_id: str | None = None
	created_by: NonEmptyStr

	@field_validator("opening_float")
	@classmethod
	def float_non_negative(cls, v: float) -> float:
		assert v >= 0, "opening_float must be non-negative"
		return v


class PosSessionUpdate(BaseModel):
	model_config = _CFG
	status: SessionStatus | None = None
	closing_cash_counted: float | None = None
	supervisor_id: str | None = None
	notes: str | None = None
	updated_by: NonEmptyStr


class PosSessionResponse(PosBase):
	terminal_id: str
	store_id: str
	cashier_id: str
	opening_float: float
	supervisor_id: str | None = None
	session_number: str = Field(default_factory=lambda: f"SES-{uuid7str()[:8].upper()}")
	status: SessionStatus = SessionStatus.OPEN
	transaction_count: int = 0
	total_sales: float = 0.0
	total_refunds: float = 0.0
	total_cash_sales: float = 0.0
	total_card_sales: float = 0.0
	total_mobile_sales: float = 0.0
	total_loyalty_sales: float = 0.0
	total_discounts: float = 0.0
	total_tax: float = 0.0
	closing_cash_counted: float | None = None
	expected_cash: float | None = None
	variance: float | None = None
	opened_at: datetime = Field(default_factory=datetime.utcnow)
	closed_at: datetime | None = None
	reconciled_at: datetime | None = None
	notes: str | None = None


# ---------------------------------------------------------------------------
# SaleItem
# ---------------------------------------------------------------------------

class SaleItemCreate(BaseModel):
	model_config = _CFG
	sku: NonEmptyStr
	barcode: str | None = None
	description: str
	quantity: float
	unit_price: float
	original_price: float | None = None  # tracks price overrides
	cost_price: float | None = None
	tax_code: str | None = None
	tax_rate: float = 0.0
	tax_amount: float = 0.0
	tax_inclusive: bool = True
	discount_amount: float = 0.0
	discount_type: DiscountType | None = None
	discount_ref: str | None = None
	line_total: float = 0.0
	promotion_ids: list[str] = Field(default_factory=list)
	weight_item: bool = False
	serialised: bool = False
	serial_numbers: list[str] = Field(default_factory=list)
	department: str | None = None
	category: str | None = None

	@model_validator(mode="after")
	def compute_line_total(self) -> "SaleItemCreate":
		if self.line_total == 0.0:
			base = round(self.unit_price * self.quantity, 4)
			self.line_total = round(base - self.discount_amount, 4)
		return self


# ---------------------------------------------------------------------------
# Discount
# ---------------------------------------------------------------------------

class DiscountCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	discount_type: DiscountType
	name: str
	value: float  # percentage or fixed amount
	max_uses: int | None = None
	min_purchase: float | None = None
	coupon_code: str | None = None
	valid_from: datetime | None = None
	valid_until: datetime | None = None
	product_skus: list[str] = Field(default_factory=list)
	category_ids: list[str] = Field(default_factory=list)
	requires_supervisor: bool = False
	created_by: NonEmptyStr


class DiscountResponse(DiscountCreate, PosBase):
	times_used: int = 0
	total_discount_given: float = 0.0
	is_active: bool = True


# ---------------------------------------------------------------------------
# Payment
# ---------------------------------------------------------------------------

class PaymentCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	transaction_id: NonEmptyStr
	session_id: NonEmptyStr
	payment_method: PaymentMethod
	amount: float
	currency: str = "KES"
	exchange_rate: float = 1.0
	reference: str | None = None      # card auth code, M-Pesa ref, etc.
	terminal_ref: str | None = None   # PED/card terminal reference
	loyalty_points_used: int | None = None
	gift_card_number: str | None = None
	created_by: NonEmptyStr

	@field_validator("amount")
	@classmethod
	def amount_positive(cls, v: float) -> float:
		assert v > 0, "payment amount must be positive"
		return v


class PaymentResponse(PaymentCreate, PosBase):
	status: PaymentStatus = PaymentStatus.AUTHORISED
	authorised_at: datetime | None = Field(default_factory=datetime.utcnow)
	gateway_response: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# SaleTransaction
# ---------------------------------------------------------------------------

class SaleTransactionCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	session_id: NonEmptyStr
	terminal_id: NonEmptyStr
	store_id: NonEmptyStr
	cashier_id: NonEmptyStr
	transaction_type: TransactionType = TransactionType.SALE
	items: list[SaleItemCreate] = Field(default_factory=list)
	customer_id: str | None = None
	customer_display_ref: str | None = None
	original_transaction_id: str | None = None  # for refunds
	notes: str | None = None
	offline_mode: bool = False
	offline_synced: bool = False
	tax_exempt: bool = False
	tax_exempt_ref: str | None = None
	discount_ids: list[str] = Field(default_factory=list)
	supervisor_override_id: str | None = None
	created_by: NonEmptyStr


class SaleTransactionResponse(PosBase):
	session_id: str
	terminal_id: str
	store_id: str
	cashier_id: str
	transaction_type: TransactionType
	items: list[SaleItemCreate] = Field(default_factory=list)
	customer_id: str | None = None
	original_transaction_id: str | None = None
	transaction_number: str = Field(default_factory=lambda: f"TXN-{uuid7str()[:8].upper()}")
	status: TransactionStatus = TransactionStatus.PENDING
	subtotal: float = 0.0
	discount_total: float = 0.0
	tax_total: float = 0.0
	grand_total: float = 0.0
	amount_tendered: float = 0.0
	change_due: float = 0.0
	balance_due: float = 0.0
	payments: list[PaymentResponse] = Field(default_factory=list)
	offline_mode: bool = False
	offline_synced: bool = False
	tax_exempt: bool = False
	tax_exempt_ref: str | None = None
	receipt_number: str | None = None
	signature_ref: str | None = None
	notes: str | None = None
	discount_ids: list[str] = Field(default_factory=list)
	supervisor_override_id: str | None = None
	posted_at: datetime | None = None
	voided_at: datetime | None = None
	refunded_at: datetime | None = None


# ---------------------------------------------------------------------------
# Refund
# ---------------------------------------------------------------------------

class RefundCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	original_transaction_id: NonEmptyStr
	session_id: NonEmptyStr
	terminal_id: NonEmptyStr
	items: list[SaleItemCreate] = Field(default_factory=list)  # subset of originals
	reason: RefundReason
	notes: str | None = None
	refund_to_original_method: bool = True
	override_payment_method: PaymentMethod | None = None
	manager_auth_id: str | None = None
	created_by: NonEmptyStr


class RefundResponse(PosBase):
	original_transaction_id: str
	session_id: str
	terminal_id: str
	items: list[SaleItemCreate] = Field(default_factory=list)
	reason: RefundReason
	refund_transaction_id: str | None = None
	refund_amount: float = 0.0
	refund_method: PaymentMethod | None = None
	status: TransactionStatus = TransactionStatus.PENDING
	manager_auth_id: str | None = None
	notes: str | None = None
	refunded_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# CashFloat
# ---------------------------------------------------------------------------

class CashFloatCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	session_id: NonEmptyStr
	terminal_id: NonEmptyStr
	store_id: NonEmptyStr
	cashier_id: NonEmptyStr
	event_type: CashEventType
	amount: float
	reason: str | None = None
	authorised_by: str | None = None
	denominations: dict[str, int] | None = None  # {"1000": 2, "500": 5, ...}
	created_by: NonEmptyStr


class CashFloatResponse(CashFloatCreate, PosBase):
	balance_after: float = 0.0
	occurred_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# PriceOverride
# ---------------------------------------------------------------------------

class PriceOverrideCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	transaction_id: NonEmptyStr
	session_id: NonEmptyStr
	sku: NonEmptyStr
	original_price: float
	override_price: float
	reason: PriceOverrideReason
	notes: str | None = None
	supervisor_id: NonEmptyStr
	created_by: NonEmptyStr

	@model_validator(mode="after")
	def validate_override(self) -> "PriceOverrideCreate":
		assert self.override_price >= 0, "override price must be non-negative"
		assert self.override_price != self.original_price, "override price must differ from original"
		return self


class PriceOverrideResponse(PriceOverrideCreate, PosBase):
	approved_at: datetime = Field(default_factory=datetime.utcnow)
	variance: float = 0.0

	@model_validator(mode="after")
	def set_variance(self) -> "PriceOverrideResponse":
		self.variance = round(self.override_price - self.original_price, 4)
		return self


# ---------------------------------------------------------------------------
# InventoryMovement
# ---------------------------------------------------------------------------

class InventoryMovementCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	store_id: NonEmptyStr
	terminal_id: NonEmptyStr
	transaction_id: NonEmptyStr
	sku: NonEmptyStr
	movement_type: InventoryMovementType
	quantity_delta: float  # negative for sales, positive for refunds
	unit_cost: float | None = None
	notes: str | None = None
	created_by: NonEmptyStr


class InventoryMovementResponse(InventoryMovementCreate, PosBase):
	stock_before: float | None = None
	stock_after: float | None = None
	occurred_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------

class ReceiptCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	transaction_id: NonEmptyStr
	session_id: NonEmptyStr
	receipt_format: ReceiptFormat = ReceiptFormat.THERMAL
	recipient_email: str | None = None
	recipient_mobile: str | None = None
	header_lines: list[str] = Field(default_factory=list)
	footer_lines: list[str] = Field(default_factory=list)
	logo_url: str | None = None
	receipt_payload: dict[str, Any] = Field(default_factory=dict)
	created_by: NonEmptyStr


class ReceiptResponse(ReceiptCreate, PosBase):
	receipt_number: str = Field(default_factory=lambda: f"REC-{uuid7str()[:8].upper()}")
	rendered_content: str | None = None  # thermal ESC/POS or HTML
	delivered_at: datetime | None = None
	issued_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# CustomerDisplay
# ---------------------------------------------------------------------------

class CustomerDisplayMessage(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	terminal_id: NonEmptyStr
	session_id: NonEmptyStr
	lines: list[str] = Field(default_factory=list, max_length=4)
	subtotal: float | None = None
	item_count: int | None = None
	promotional_message: str | None = None
	created_by: NonEmptyStr


# ---------------------------------------------------------------------------
# LoyaltyAccount
# ---------------------------------------------------------------------------

class LoyaltyTransactionCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	customer_id: NonEmptyStr
	transaction_id: NonEmptyStr
	points_earned: int = 0
	points_redeemed: int = 0
	points_balance_before: int = 0
	points_balance_after: int = 0
	earn_rate: float = 1.0      # points per currency unit
	redeem_rate: float = 0.01   # currency value per point
	created_by: NonEmptyStr

	@model_validator(mode="after")
	def validate_points(self) -> "LoyaltyTransactionCreate":
		assert self.points_earned >= 0, "points_earned must be non-negative"
		assert self.points_redeemed >= 0, "points_redeemed must be non-negative"
		return self


class LoyaltyTransactionResponse(LoyaltyTransactionCreate, PosBase):
	occurred_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# EndOfDayReport
# ---------------------------------------------------------------------------

class EndOfDayReportCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	store_id: NonEmptyStr
	business_date: str  # ISO date YYYY-MM-DD
	generated_by: NonEmptyStr
	created_by: NonEmptyStr


class EndOfDayReportResponse(PosBase):
	store_id: str
	business_date: str
	session_count: int = 0
	transaction_count: int = 0
	gross_sales: float = 0.0
	total_refunds: float = 0.0
	total_discounts: float = 0.0
	total_tax: float = 0.0
	net_sales: float = 0.0
	cash_sales: float = 0.0
	card_sales: float = 0.0
	mobile_sales: float = 0.0
	loyalty_sales: float = 0.0
	other_sales: float = 0.0
	opening_floats_total: float = 0.0
	safe_drops_total: float = 0.0
	variance_total: float = 0.0
	hourly_breakdown: list[dict[str, Any]] = Field(default_factory=list)
	top_selling_skus: list[dict[str, Any]] = Field(default_factory=list)
	generated_at: datetime = Field(default_factory=datetime.utcnow)
	status: str = "draft"
	approved_by: str | None = None
	approved_at: datetime | None = None


# ---------------------------------------------------------------------------
# Offline sync
# ---------------------------------------------------------------------------

class OfflineSyncBatch(BaseModel):
	"""Batch of transactions collected offline, submitted for sync."""
	model_config = _CFG
	tenant_id: NonEmptyStr
	terminal_id: NonEmptyStr
	session_id: NonEmptyStr
	transactions: list[SaleTransactionCreate]
	cash_events: list[CashFloatCreate] = Field(default_factory=list)
	sync_sequence: int  # monotone counter to detect gaps
	checksum: str | None = None
	created_by: NonEmptyStr


class OfflineSyncResult(BaseModel):
	model_config = _CFG_OPEN
	tenant_id: str
	terminal_id: str
	accepted: list[str] = Field(default_factory=list)
	rejected: list[dict[str, Any]] = Field(default_factory=list)
	duplicate_skipped: list[str] = Field(default_factory=list)
	sync_completed_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Supervisor Override
# ---------------------------------------------------------------------------

class SupervisorOverrideCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	session_id: NonEmptyStr
	terminal_id: NonEmptyStr
	supervisor_id: NonEmptyStr
	override_type: str   # "price_override" | "discount_override" | "void" | "refund" | "close_session"
	target_id: str | None = None  # transaction/item id being overridden
	notes: str | None = None
	created_by: NonEmptyStr


class SupervisorOverrideResponse(SupervisorOverrideCreate, PosBase):
	granted_at: datetime = Field(default_factory=datetime.utcnow)
	expires_at: datetime | None = None


# ---------------------------------------------------------------------------
# Pagination & filtering
# ---------------------------------------------------------------------------

class PaginatedResponse(BaseModel):
	model_config = _CFG_OPEN
	items: list[Any]
	total: int
	page: int
	page_size: int
	pages: int

	@classmethod
	def build(cls, items: list[Any], total: int, page: int, page_size: int) -> "PaginatedResponse":
		pages = max(1, -(-total // page_size))  # ceiling division
		return cls(items=items, total=total, page=page, page_size=page_size, pages=pages)


class TransactionFilter(BaseModel):
	model_config = _CFG_OPEN
	session_id: str | None = None
	terminal_id: str | None = None
	store_id: str | None = None
	cashier_id: str | None = None
	transaction_type: TransactionType | None = None
	status: TransactionStatus | None = None
	customer_id: str | None = None
	date_from: datetime | None = None
	date_to: datetime | None = None
	page: int = 1
	page_size: int = 50
	sort_by: str = "created_at"
	sort_dir: str = "desc"
