"""
Accounts Payable — Pydantic v2 domain models.

Every lifecycle state, edge case, and financial instrument in AP is represented
here. No external dependencies beyond pydantic, uuid6, and stdlib.

© 2025 Datacraft. All rights reserved.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from uuid6 import uuid7


# ---------------------------------------------------------------------------
# UUID helper
# ---------------------------------------------------------------------------

def uuid7str() -> str:
	return str(uuid7())


# ---------------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------------

_CFG = ConfigDict(
	extra="forbid",
	validate_by_name=True,
	validate_by_alias=True,
	str_strip_whitespace=True,
)


class APBase(BaseModel):
	"""All AP entities share this base."""
	model_config = _CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str | None = None
	is_deleted: bool = False


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SupplierStatus(str, Enum):
	ACTIVE = "active"
	INACTIVE = "inactive"
	ON_HOLD = "on_hold"
	BLOCKED = "blocked"
	PENDING_APPROVAL = "pending_approval"
	SUSPENDED = "suspended"


class SupplierType(str, Enum):
	GOODS = "goods"
	SERVICES = "services"
	UTILITY = "utility"
	LANDLORD = "landlord"
	CONTRACTOR = "contractor"
	GOVERNMENT = "government"
	INTERCOMPANY = "intercompany"
	EMPLOYEE = "employee"


class InvoiceStatus(str, Enum):
	RECEIVED = "received"
	VALIDATED = "validated"
	MATCHED = "matched"
	APPROVED = "approved"
	POSTED = "posted"
	PAID = "paid"
	PARTIALLY_PAID = "partially_paid"
	DISPUTED = "disputed"
	ON_HOLD = "on_hold"
	CANCELLED = "cancelled"
	DUPLICATE = "duplicate"
	REJECTED = "rejected"


class InvoiceType(str, Enum):
	STANDARD = "standard"
	CREDIT_NOTE = "credit_note"
	DEBIT_NOTE = "debit_note"
	PREPAYMENT = "prepayment"
	SELF_BILLED = "self_billed"
	RECURRING = "recurring"
	PRO_FORMA = "pro_forma"
	RETENTION = "retention"


class POMatchingType(str, Enum):
	NONE = "none"
	TWO_WAY = "two_way"     # invoice vs PO only
	THREE_WAY = "three_way" # invoice vs PO vs GRN


class MatchStatus(str, Enum):
	PENDING = "pending"
	MATCHED = "matched"
	PARTIAL = "partial"
	TOLERANCE_EXCEEDED = "tolerance_exceeded"
	FAILED = "failed"
	WAIVED = "waived"


class PaymentRunStatus(str, Enum):
	DRAFT = "draft"
	APPROVED = "approved"
	PROCESSING = "processing"
	PROCESSED = "processed"
	POSTED = "posted"
	FAILED = "failed"
	CANCELLED = "cancelled"


class PaymentStatus(str, Enum):
	SCHEDULED = "scheduled"
	PROCESSING = "processing"
	COMPLETED = "completed"
	FAILED = "failed"
	RETURNED = "returned"
	CANCELLED = "cancelled"
	VOIDED = "voided"
	CLEARED = "cleared"


class PaymentMethod(str, Enum):
	ACH = "ach"
	WIRE = "wire"
	CHECK = "check"
	VIRTUAL_CARD = "virtual_card"
	RTP = "rtp"
	FEDNOW = "fednow"
	SEPA = "sepa"
	SWIFT = "swift"
	MPESA = "mpesa"
	CASH = "cash"


class DisputeType(str, Enum):
	PRICE_DISCREPANCY = "price_discrepancy"
	QUANTITY_DISCREPANCY = "quantity_discrepancy"
	DUPLICATE = "duplicate"
	GOODS_NOT_RECEIVED = "goods_not_received"
	QUALITY_ISSUE = "quality_issue"
	INCORRECT_SUPPLIER = "incorrect_supplier"
	TERMS_DISPUTE = "terms_dispute"
	TAX_ERROR = "tax_error"
	OTHER = "other"


class DisputeStatus(str, Enum):
	OPEN = "open"
	UNDER_REVIEW = "under_review"
	AWAITING_SUPPLIER = "awaiting_supplier"
	RESOLVED_ACCEPTED = "resolved_accepted"
	RESOLVED_REJECTED = "resolved_rejected"
	ESCALATED = "escalated"
	CLOSED = "closed"


class AccrualType(str, Enum):
	GOODS_RECEIPT = "goods_receipt"
	SERVICE_ACCRUAL = "service_accrual"
	YEAR_END = "year_end"
	PREPAYMENT = "prepayment"
	RETENTION = "retention"
	INTERCOMPANY = "intercompany"


class CapexOpex(str, Enum):
	CAPEX = "capex"
	OPEX = "opex"
	MIXED = "mixed"


class StatementFrequency(str, Enum):
	MONTHLY = "monthly"
	QUARTERLY = "quarterly"
	ANNUAL = "annual"
	ON_DEMAND = "on_demand"


class InvoiceFormat(str, Enum):
	PDF = "pdf"
	XML = "xml"
	EDI = "edi"
	UBL = "ubl_2_1"
	PEPPOL = "peppol_bis"
	CSV = "csv"
	PAPER = "paper"


class HoldReason(str, Enum):
	DISPUTE = "dispute"
	PENDING_GRN = "pending_grn"
	PRICE_MISMATCH = "price_mismatch"
	QUANTITY_MISMATCH = "quantity_mismatch"
	APPROVAL_PENDING = "approval_pending"
	DUPLICATE_SUSPECTED = "duplicate_suspected"
	TAX_REVIEW = "tax_review"
	LEGAL_REVIEW = "legal_review"
	CASH_FLOW = "cash_flow"
	SUPPLIER_ON_HOLD = "supplier_on_hold"
	MANUAL = "manual"


# ---------------------------------------------------------------------------
# Value objects (not APBase — they're embedded)
# ---------------------------------------------------------------------------

class BankAccount(BaseModel):
	model_config = _CFG

	id: str = Field(default_factory=uuid7str)
	bank_name: str
	account_name: str
	account_number: str
	routing_number: str | None = None   # ACH/ABA
	sort_code: str | None = None        # UK
	iban: str | None = None
	swift_bic: str | None = None
	currency: str = "USD"
	payment_method: PaymentMethod = PaymentMethod.ACH
	is_primary: bool = False
	is_active: bool = True
	country_code: str = "US"
	verified: bool = False
	verified_at: datetime | None = None
	verified_by: str | None = None


class Address(BaseModel):
	model_config = _CFG

	address_type: str = "billing"
	line1: str
	line2: str | None = None
	city: str
	state_province: str | None = None
	postal_code: str | None = None
	country_code: str
	is_primary: bool = False


class PaymentTerms(BaseModel):
	"""Encodes terms like 2/10 Net30: 2% discount if paid within 10 days, else net 30."""
	model_config = _CFG

	code: str
	description: str
	net_days: int = Field(ge=0, le=365)
	discount_days: int = Field(default=0, ge=0)
	discount_pct: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0"), le=Decimal("100"))
	penalty_pct_per_month: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0"))
	# Some terms are end-of-month: "net 30 EOM"
	end_of_month: bool = False


class WithholdingTaxConfig(BaseModel):
	model_config = _CFG

	applicable: bool = False
	rate_pct: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0"), le=Decimal("100"))
	tax_type: str | None = None          # WHT, VAT, GST, etc.
	certificate_number: str | None = None
	exemption_reason: str | None = None


# ---------------------------------------------------------------------------
# APSupplier
# ---------------------------------------------------------------------------

class APSupplierCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	created_by: str
	supplier_number: str
	legal_name: str
	trade_name: str | None = None
	supplier_type: SupplierType = SupplierType.GOODS
	tax_registration: str | None = None
	tax_country: str | None = None
	vat_number: str | None = None
	payment_terms: PaymentTerms
	currency: str = "USD"
	bank_accounts: list[BankAccount] = Field(default_factory=list)
	payment_method: PaymentMethod = PaymentMethod.ACH
	credit_limit: Decimal | None = Field(default=None, ge=Decimal("0"))
	statement_frequency: StatementFrequency = StatementFrequency.MONTHLY
	on_hold: bool = False
	withholding_tax: WithholdingTaxConfig = Field(default_factory=WithholdingTaxConfig)
	preferred_invoice_format: InvoiceFormat = InvoiceFormat.PDF
	po_required: bool = False
	self_billing_enabled: bool = False
	intercompany_entity_id: str | None = None
	addresses: list[Address] = Field(default_factory=list)
	# Peppol/e-invoicing endpoint
	peppol_id: str | None = None
	contact_email: str | None = None
	contact_phone: str | None = None
	notes: str | None = None

	@field_validator("supplier_number")
	@classmethod
	def _validate_supplier_number(cls, v: str) -> str:
		assert len(v) >= 2, "supplier_number must be at least 2 chars"
		return v.upper()


class APSupplierUpdate(BaseModel):
	model_config = _CFG

	updated_by: str
	trade_name: str | None = None
	payment_terms: PaymentTerms | None = None
	currency: str | None = None
	bank_accounts: list[BankAccount] | None = None
	payment_method: PaymentMethod | None = None
	credit_limit: Decimal | None = None
	statement_frequency: StatementFrequency | None = None
	on_hold: bool | None = None
	withholding_tax: WithholdingTaxConfig | None = None
	preferred_invoice_format: InvoiceFormat | None = None
	po_required: bool | None = None
	self_billing_enabled: bool | None = None
	peppol_id: str | None = None
	contact_email: str | None = None
	notes: str | None = None
	status: SupplierStatus | None = None


class APSupplier(APBase):
	supplier_number: str
	legal_name: str
	trade_name: str | None = None
	supplier_type: SupplierType = SupplierType.GOODS
	status: SupplierStatus = SupplierStatus.PENDING_APPROVAL
	tax_registration: str | None = None
	tax_country: str | None = None
	vat_number: str | None = None
	payment_terms: PaymentTerms
	currency: str = "USD"
	bank_accounts: list[BankAccount] = Field(default_factory=list)
	payment_method: PaymentMethod = PaymentMethod.ACH
	credit_limit: Decimal | None = None
	statement_frequency: StatementFrequency = StatementFrequency.MONTHLY
	on_hold: bool = False
	hold_reason: str | None = None
	hold_placed_at: datetime | None = None
	withholding_tax: WithholdingTaxConfig = Field(default_factory=WithholdingTaxConfig)
	preferred_invoice_format: InvoiceFormat = InvoiceFormat.PDF
	po_required: bool = False
	self_billing_enabled: bool = False
	intercompany_entity_id: str | None = None
	addresses: list[Address] = Field(default_factory=list)
	peppol_id: str | None = None
	contact_email: str | None = None
	contact_phone: str | None = None
	notes: str | None = None
	# Lifetime metrics (denormalised for fast display)
	total_invoiced_ytd: Decimal = Field(default=Decimal("0.00"))
	total_paid_ytd: Decimal = Field(default=Decimal("0.00"))
	invoice_count: int = 0
	dispute_count: int = 0
	avg_payment_days: Decimal | None = None


# ---------------------------------------------------------------------------
# APInvoiceLine
# ---------------------------------------------------------------------------

class APInvoiceLineCreate(BaseModel):
	model_config = _CFG

	line_number: int = Field(ge=1)
	description: str
	quantity: Decimal = Field(ge=Decimal("0"))
	unit_price: Decimal
	tax_code: str | None = None
	tax_rate_pct: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0"))
	gl_account: str
	cost_center: str | None = None
	project_code: str | None = None
	asset_ref: str | None = None
	capex_opex: CapexOpex = CapexOpex.OPEX
	po_line_ref: str | None = None
	grn_line_ref: str | None = None
	# Retention (construction): percentage withheld until practical completion
	retention_pct: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0"), le=Decimal("100"))


class APInvoiceLine(BaseModel):
	model_config = _CFG

	id: str = Field(default_factory=uuid7str)
	invoice_id: str
	line_number: int
	description: str
	quantity: Decimal
	unit_price: Decimal
	line_subtotal: Decimal      # quantity * unit_price
	tax_code: str | None = None
	tax_rate_pct: Decimal = Decimal("0.00")
	tax_amount: Decimal = Decimal("0.00")
	line_total: Decimal         # line_subtotal + tax_amount − retention_amount
	retention_pct: Decimal = Decimal("0.00")
	retention_amount: Decimal = Decimal("0.00")
	gl_account: str
	cost_center: str | None = None
	project_code: str | None = None
	asset_ref: str | None = None
	capex_opex: CapexOpex = CapexOpex.OPEX
	po_line_ref: str | None = None
	grn_line_ref: str | None = None


# ---------------------------------------------------------------------------
# APInvoice
# ---------------------------------------------------------------------------

class APInvoiceCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	created_by: str
	supplier_id: str
	invoice_type: InvoiceType = InvoiceType.STANDARD
	supplier_invoice_ref: str
	invoice_date: date
	received_date: date = Field(default_factory=date.today)
	# posting_date / accounting_date can differ from invoice_date
	posting_date: date | None = None
	accounting_period: str | None = None  # e.g. "2025-06"
	due_date: date
	currency: str = "USD"
	exchange_rate: Decimal = Field(default=Decimal("1.000000"), gt=Decimal("0"))
	payment_terms: PaymentTerms
	lines: list[APInvoiceLineCreate]
	po_matching: POMatchingType = POMatchingType.NONE
	po_refs: list[str] = Field(default_factory=list)
	grn_refs: list[str] = Field(default_factory=list)
	# Self-billing: buyer generates the invoice on behalf of supplier
	self_billed: bool = False
	# Document storage reference (e.g. S3 key, DMS ID)
	document_ref: str | None = None
	ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
	notes: str | None = None

	@model_validator(mode="after")
	def _validate_dates(self) -> APInvoiceCreate:
		assert self.due_date >= self.invoice_date, "due_date must be >= invoice_date"
		return self

	@model_validator(mode="after")
	def _validate_lines(self) -> APInvoiceCreate:
		assert len(self.lines) >= 1, "invoice must have at least one line"
		return self


class APInvoiceUpdate(BaseModel):
	model_config = _CFG

	updated_by: str
	posting_date: date | None = None
	accounting_period: str | None = None
	due_date: date | None = None
	exchange_rate: Decimal | None = None
	notes: str | None = None
	document_ref: str | None = None


class APInvoice(APBase):
	invoice_number: str      # system-generated sequential
	supplier_id: str
	invoice_type: InvoiceType = InvoiceType.STANDARD
	supplier_invoice_ref: str
	invoice_date: date
	received_date: date
	posting_date: date | None = None
	accounting_period: str | None = None
	due_date: date
	currency: str
	exchange_rate: Decimal = Decimal("1.000000")
	# Amounts in invoice currency
	subtotal: Decimal
	tax_amount: Decimal = Decimal("0.00")
	retention_amount: Decimal = Decimal("0.00")
	total: Decimal
	# Amounts in functional (base) currency
	subtotal_base: Decimal
	total_base: Decimal
	paid_amount: Decimal = Decimal("0.00")
	outstanding: Decimal       # total − paid_amount (in invoice currency)
	status: InvoiceStatus = InvoiceStatus.RECEIVED
	payment_terms: PaymentTerms
	lines: list[APInvoiceLine] = Field(default_factory=list)
	po_matching: POMatchingType = POMatchingType.NONE
	po_refs: list[str] = Field(default_factory=list)
	grn_refs: list[str] = Field(default_factory=list)
	match_status: MatchStatus = MatchStatus.PENDING
	self_billed: bool = False
	document_ref: str | None = None
	ocr_confidence: float | None = None
	# Hold state
	on_hold: bool = False
	hold_reason: HoldReason | None = None
	hold_notes: str | None = None
	hold_placed_by: str | None = None
	hold_placed_at: datetime | None = None
	hold_released_by: str | None = None
	hold_released_at: datetime | None = None
	# Approval
	approved_by: str | None = None
	approved_at: datetime | None = None
	rejected_by: str | None = None
	rejected_at: datetime | None = None
	rejection_reason: str | None = None
	# Posting
	posted_by: str | None = None
	posted_at: datetime | None = None
	gl_journal_ref: str | None = None
	# Duplicate detection
	duplicate_of: str | None = None
	duplicate_score: float | None = None
	notes: str | None = None
	# Early payment discount window
	discount_due_date: date | None = None
	discount_amount: Decimal | None = None


# ---------------------------------------------------------------------------
# APThreeWayMatch
# ---------------------------------------------------------------------------

class APThreeWayMatchCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	created_by: str
	invoice_id: str
	po_id: str
	grn_id: str | None = None   # None for 2-way match
	price_tolerance_pct: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0"))
	qty_tolerance_pct: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0"))


class APThreeWayMatch(APBase):
	invoice_id: str
	po_id: str
	grn_id: str | None = None
	matching_type: POMatchingType
	# Line-level results
	quantity_match: bool = False
	price_match: bool = False
	quantity_variance_pct: Decimal = Decimal("0.0000")
	price_variance_pct: Decimal = Decimal("0.0000")
	price_tolerance_pct: Decimal = Decimal("0.00")
	qty_tolerance_pct: Decimal = Decimal("0.00")
	within_tolerance: bool = False
	match_status: MatchStatus = MatchStatus.PENDING
	discrepancies: list[dict[str, Any]] = Field(default_factory=list)
	auto_approved: bool = False
	reviewed_by: str | None = None
	reviewed_at: datetime | None = None


# ---------------------------------------------------------------------------
# APPaymentRun
# ---------------------------------------------------------------------------

class APPaymentRunCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	created_by: str
	run_name: str
	payment_date: date
	payment_method: PaymentMethod
	bank_account_id: str
	currency: str = "USD"
	# Selection criteria
	due_date_from: date | None = None
	due_date_to: date | None = None
	supplier_ids: list[str] = Field(default_factory=list)   # empty = all
	capture_early_discount: bool = True
	exclude_disputed: bool = True
	exclude_on_hold: bool = True
	max_run_amount: Decimal | None = None
	notes: str | None = None


class APPaymentRunUpdate(BaseModel):
	model_config = _CFG

	updated_by: str
	notes: str | None = None


class APPaymentRun(APBase):
	run_number: str
	run_name: str
	run_date: date = Field(default_factory=date.today)
	payment_date: date
	payment_method: PaymentMethod
	bank_account_id: str
	currency: str
	exchange_rate: Decimal = Decimal("1.000000")
	# Selection criteria stored for audit
	due_date_from: date | None = None
	due_date_to: date | None = None
	supplier_ids: list[str] = Field(default_factory=list)
	capture_early_discount: bool = True
	exclude_disputed: bool = True
	exclude_on_hold: bool = True
	max_run_amount: Decimal | None = None
	# Results
	invoices_selected: list[str] = Field(default_factory=list)
	invoice_count: int = 0
	total_amount: Decimal = Decimal("0.00")
	total_discount: Decimal = Decimal("0.00")
	net_payment: Decimal = Decimal("0.00")
	status: PaymentRunStatus = PaymentRunStatus.DRAFT
	approved_by: str | None = None
	approved_at: datetime | None = None
	processed_by: str | None = None
	processed_at: datetime | None = None
	bank_file_ref: str | None = None
	notes: str | None = None


# ---------------------------------------------------------------------------
# APPayment
# ---------------------------------------------------------------------------

class APPaymentAllocation(BaseModel):
	"""Links a payment to an invoice with partial-pay support."""
	model_config = _CFG

	invoice_id: str
	invoice_number: str
	allocated_amount: Decimal
	discount_taken: Decimal = Decimal("0.00")
	is_credit_note: bool = False


class APPaymentCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	created_by: str
	payment_run_id: str | None = None
	supplier_id: str
	payment_date: date
	payment_method: PaymentMethod
	bank_account_from: str
	bank_account_to: str
	currency: str = "USD"
	exchange_rate: Decimal = Field(default=Decimal("1.000000"), gt=Decimal("0"))
	allocations: list[APPaymentAllocation]
	reference: str | None = None
	remittance_email: str | None = None

	@model_validator(mode="after")
	def _validate_allocations(self) -> APPaymentCreate:
		assert len(self.allocations) >= 1, "payment must allocate to at least one invoice"
		return self


class APPayment(APBase):
	payment_ref: str
	payment_run_id: str | None = None
	supplier_id: str
	payment_date: date
	cleared_date: date | None = None
	payment_method: PaymentMethod
	bank_account_from: str
	bank_account_to: str
	currency: str
	exchange_rate: Decimal = Decimal("1.000000")
	amount: Decimal
	amount_base: Decimal
	status: PaymentStatus = PaymentStatus.SCHEDULED
	allocations: list[APPaymentAllocation] = Field(default_factory=list)
	reference: str | None = None
	check_number: str | None = None
	bank_transaction_id: str | None = None
	remittance_sent: bool = False
	remittance_sent_at: datetime | None = None
	voided_by: str | None = None
	voided_at: datetime | None = None
	void_reason: str | None = None


# ---------------------------------------------------------------------------
# APDispute
# ---------------------------------------------------------------------------

class APDisputeCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	created_by: str
	invoice_id: str
	dispute_type: DisputeType
	disputed_amount: Decimal = Field(gt=Decimal("0"))
	description: str
	expected_resolution_date: date | None = None
	evidence_refs: list[str] = Field(default_factory=list)


class APDisputeUpdate(BaseModel):
	model_config = _CFG

	updated_by: str
	status: DisputeStatus | None = None
	resolution_notes: str | None = None
	resolved_amount: Decimal | None = None
	evidence_refs: list[str] | None = None


class APDispute(APBase):
	invoice_id: str
	dispute_type: DisputeType
	disputed_amount: Decimal
	resolved_amount: Decimal | None = None
	description: str
	status: DisputeStatus = DisputeStatus.OPEN
	opened_at: datetime = Field(default_factory=datetime.utcnow)
	resolved_at: datetime | None = None
	resolution_notes: str | None = None
	expected_resolution_date: date | None = None
	evidence_refs: list[str] = Field(default_factory=list)
	supplier_response: str | None = None
	supplier_responded_at: datetime | None = None
	escalated_to: str | None = None
	escalated_at: datetime | None = None


# ---------------------------------------------------------------------------
# APAccrual
# ---------------------------------------------------------------------------

class APAccrualCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	created_by: str
	accrual_type: AccrualType
	accounting_period: str   # "2025-12"
	supplier_id: str | None = None
	gl_account: str
	cost_center: str | None = None
	amount: Decimal = Field(gt=Decimal("0"))
	description: str
	reversal_date: date | None = None   # None = manual reversal
	po_ref: str | None = None
	grn_ref: str | None = None
	invoice_id: str | None = None      # if later invoice arrives


class APAccrual(APBase):
	accrual_number: str
	accrual_type: AccrualType
	accounting_period: str
	supplier_id: str | None = None
	gl_account: str
	cost_center: str | None = None
	amount: Decimal
	description: str
	posted: bool = False
	posted_at: datetime | None = None
	posted_by: str | None = None
	journal_ref: str | None = None
	reversed: bool = False
	reversal_date: date | None = None
	reversal_journal_ref: str | None = None
	reversed_at: datetime | None = None
	reversed_by: str | None = None
	po_ref: str | None = None
	grn_ref: str | None = None
	invoice_id: str | None = None


# ---------------------------------------------------------------------------
# APCreditNote  (credit notes are a distinct entity in AP)
# ---------------------------------------------------------------------------

class APCreditNoteCreate(BaseModel):
	model_config = _CFG

	tenant_id: str
	created_by: str
	supplier_id: str
	supplier_credit_ref: str
	credit_date: date
	currency: str
	exchange_rate: Decimal = Decimal("1.000000")
	amount: Decimal = Field(gt=Decimal("0"))
	tax_amount: Decimal = Decimal("0.00")
	reason: str
	gl_account: str
	# Which invoice(s) this credit relates to (optional)
	invoice_refs: list[str] = Field(default_factory=list)


class APCreditNote(APBase):
	credit_number: str
	supplier_id: str
	supplier_credit_ref: str
	credit_date: date
	currency: str
	exchange_rate: Decimal
	amount: Decimal
	tax_amount: Decimal
	reason: str
	gl_account: str
	invoice_refs: list[str] = Field(default_factory=list)
	applied_amount: Decimal = Decimal("0.00")
	remaining_amount: Decimal
	posted: bool = False
	posted_at: datetime | None = None


# ---------------------------------------------------------------------------
# APStatementReconciliation
# ---------------------------------------------------------------------------

class APStatementLine(BaseModel):
	model_config = _CFG

	supplier_ref: str
	statement_date: date
	description: str
	amount: Decimal
	# Matched AP invoice id (None if unmatched)
	matched_invoice_id: str | None = None
	discrepancy: Decimal | None = None
	status: str = "unmatched"  # matched, unmatched, discrepancy


class APStatementRecon(APBase):
	supplier_id: str
	statement_date: date
	statement_total: Decimal
	ap_balance: Decimal
	variance: Decimal
	lines: list[APStatementLine] = Field(default_factory=list)
	unmatched_count: int = 0
	discrepancy_count: int = 0
	reconciled: bool = False
	reconciled_by: str | None = None
	reconciled_at: datetime | None = None


# ---------------------------------------------------------------------------
# AP Aging report models
# ---------------------------------------------------------------------------

class APAgingBucket(BaseModel):
	model_config = _CFG

	supplier_id: str
	supplier_name: str
	currency: str
	current: Decimal = Decimal("0.00")       # not yet due
	days_1_30: Decimal = Decimal("0.00")
	days_31_60: Decimal = Decimal("0.00")
	days_61_90: Decimal = Decimal("0.00")
	days_91_120: Decimal = Decimal("0.00")
	over_120: Decimal = Decimal("0.00")
	total_outstanding: Decimal = Decimal("0.00")
	invoice_count: int = 0


class APAgingReport(BaseModel):
	model_config = _CFG

	tenant_id: str
	as_of_date: date
	currency: str
	buckets: list[APAgingBucket] = Field(default_factory=list)
	grand_total: Decimal = Decimal("0.00")
	total_current: Decimal = Decimal("0.00")
	total_overdue: Decimal = Decimal("0.00")
	generated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# AP KPI / Dashboard
# ---------------------------------------------------------------------------

class APDashboard(BaseModel):
	model_config = _CFG

	tenant_id: str
	as_of: datetime = Field(default_factory=datetime.utcnow)
	# Volumes
	invoices_received_mtd: int = 0
	invoices_pending_approval: int = 0
	invoices_on_hold: int = 0
	invoices_overdue: int = 0
	invoices_disputed: int = 0
	# Amounts
	total_outstanding: Decimal = Decimal("0.00")
	total_overdue: Decimal = Decimal("0.00")
	total_due_this_week: Decimal = Decimal("0.00")
	total_due_next_30_days: Decimal = Decimal("0.00")
	# DPO (Days Payable Outstanding)
	dpo_days: Decimal | None = None
	# Automation
	auto_match_rate_pct: Decimal = Decimal("0.00")
	straight_through_rate_pct: Decimal = Decimal("0.00")
	# Cash
	projected_cash_out_30d: Decimal = Decimal("0.00")
	early_discount_captured_mtd: Decimal = Decimal("0.00")
	early_discount_available: Decimal = Decimal("0.00")


# ---------------------------------------------------------------------------
# Duplicate detection result
# ---------------------------------------------------------------------------

class DuplicateCheckResult(BaseModel):
	model_config = _CFG

	invoice_id: str
	is_duplicate: bool = False
	exact_match_ids: list[str] = Field(default_factory=list)
	fuzzy_match_ids: list[str] = Field(default_factory=list)
	confidence: float = 0.0
	reason: str | None = None


# ---------------------------------------------------------------------------
# Early-payment discount opportunity
# ---------------------------------------------------------------------------

class EarlyDiscountOpportunity(BaseModel):
	model_config = _CFG

	invoice_id: str
	invoice_number: str
	supplier_id: str
	due_date: date
	discount_due_date: date
	invoice_amount: Decimal
	discount_amount: Decimal
	discount_pct: Decimal
	# Annualised return of taking the discount  (discount_pct / days_saved * 365)
	annualised_return_pct: Decimal
	days_remaining: int
	recommended: bool


__all__ = [
	"uuid7str",
	# Enums
	"SupplierStatus", "SupplierType", "InvoiceStatus", "InvoiceType",
	"POMatchingType", "MatchStatus", "PaymentRunStatus", "PaymentStatus",
	"PaymentMethod", "DisputeType", "DisputeStatus", "AccrualType",
	"CapexOpex", "StatementFrequency", "InvoiceFormat", "HoldReason",
	# Value objects
	"BankAccount", "Address", "PaymentTerms", "WithholdingTaxConfig",
	# Entities
	"APSupplierCreate", "APSupplierUpdate", "APSupplier",
	"APInvoiceLineCreate", "APInvoiceLine",
	"APInvoiceCreate", "APInvoiceUpdate", "APInvoice",
	"APThreeWayMatchCreate", "APThreeWayMatch",
	"APPaymentRunCreate", "APPaymentRunUpdate", "APPaymentRun",
	"APPaymentAllocation", "APPaymentCreate", "APPayment",
	"APDisputeCreate", "APDisputeUpdate", "APDispute",
	"APAccrualCreate", "APAccrual",
	"APCreditNoteCreate", "APCreditNote",
	"APStatementLine", "APStatementRecon",
	# Reports
	"APAgingBucket", "APAgingReport", "APDashboard",
	"DuplicateCheckResult", "EarlyDiscountOpportunity",
]
