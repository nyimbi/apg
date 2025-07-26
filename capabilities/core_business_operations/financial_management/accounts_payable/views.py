"""
APG Core Financials - Accounts Payable View Models

CLAUDE.md compliant view models with ConfigDict validation for Flask-AppBuilder
UI integration within the APG platform ecosystem.

© 2025 Datacraft. All rights reserved.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field
from uuid_extensions import uuid7str

from .models import (
	APVendor, APInvoice, APPayment, APApprovalWorkflow, APExpenseReport,
	VendorStatus, VendorType, InvoiceStatus, PaymentStatus, PaymentMethod,
	ApprovalStatus, MatchingStatus
)


# Base Configuration for all view models per APG standards
VIEW_MODEL_CONFIG = ConfigDict(
	extra='forbid',
	validate_by_name=True,
	validate_by_alias=True,
	str_strip_whitespace=True,
	validate_default=True,
	populate_by_name=True
)


# Dashboard and Analytics View Models

class APDashboardSummary(BaseModel):
	"""Executive dashboard summary with key AP metrics"""
	total_vendors: int = 0
	active_vendors: int = 0
	pending_invoices: int = 0
	pending_invoice_amount: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	overdue_invoices: int = 0
	overdue_amount: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	
	# Payment metrics
	pending_payments: int = 0
	pending_payment_amount: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	payments_this_month: int = 0
	payment_amount_this_month: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	
	# Efficiency metrics
	touchless_processing_rate: float = 0.0
	average_processing_time_hours: float = 0.0
	approval_cycle_time_days: float = 0.0
	early_payment_discount_captured: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	
	# Compliance metrics
	compliance_score: float = 100.0
	audit_findings_open: int = 0
	
	tenant_id: str
	generated_at: datetime = Field(default_factory=datetime.utcnow)
	
	model_config = VIEW_MODEL_CONFIG


class APPerformanceMetrics(BaseModel):
	"""Operational performance metrics view model"""
	metric_name: str
	current_value: Decimal = Field(max_digits=15, decimal_places=6)
	target_value: Decimal | None = Field(default=None, max_digits=15, decimal_places=6)
	previous_period_value: Decimal | None = Field(default=None, max_digits=15, decimal_places=6)
	
	unit_of_measure: str  # percentage, dollars, hours, count, etc.
	trend_direction: str  # up, down, stable
	performance_status: str  # excellent, good, warning, critical
	
	period_start: date
	period_end: date
	
	model_config = VIEW_MODEL_CONFIG
	
	@computed_field
	@property
	def variance_from_target(self) -> Decimal | None:
		"""Calculate variance from target"""
		if self.target_value is None:
			return None
		return self.current_value - self.target_value
	
	@computed_field
	@property
	def period_over_period_change(self) -> Decimal | None:
		"""Calculate change from previous period"""
		if self.previous_period_value is None or self.previous_period_value == 0:
			return None
		return ((self.current_value - self.previous_period_value) / self.previous_period_value) * 100


# Vendor View Models

class APVendorSummaryView(BaseModel):
	"""Vendor summary for list views with performance indicators"""
	id: str
	vendor_code: str
	legal_name: str
	trade_name: str | None = None
	vendor_type: VendorType
	status: VendorStatus
	
	# Financial summary
	total_outstanding: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	current_amount: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	overdue_amount: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	
	# Contact information
	primary_contact_name: str
	primary_contact_email: str | None = None
	primary_contact_phone: str | None = None
	
	# Performance metrics
	performance_score: float = Field(default=0.0, ge=0.0, le=100.0)
	on_time_delivery_rate: float = Field(default=0.0, ge=0.0, le=100.0)
	invoice_accuracy_rate: float = Field(default=0.0, ge=0.0, le=100.0)
	
	# Recent activity
	last_invoice_date: date | None = None
	last_payment_date: date | None = None
	total_invoices_ytd: int = 0
	total_payments_ytd: int = 0
	
	# Payment terms
	payment_terms_display: str
	credit_limit: Decimal | None = Field(default=None, max_digits=15, decimal_places=2)
	
	created_at: datetime
	
	model_config = VIEW_MODEL_CONFIG
	
	@computed_field
	@property
	def risk_indicator(self) -> str:
		"""Calculate vendor risk indicator"""
		if self.overdue_amount > Decimal('10000'):
			return 'high'
		elif self.performance_score < 70:
			return 'medium'
		else:
			return 'low'
	
	@classmethod
	def from_vendor(cls, vendor: APVendor, financial_summary: Dict[str, Any] | None = None) -> "APVendorSummaryView":
		"""Convert APVendor to summary view model"""
		summary = financial_summary or {}
		
		return cls(
			id=vendor.id,
			vendor_code=vendor.vendor_code,
			legal_name=vendor.legal_name,
			trade_name=vendor.trade_name,
			vendor_type=vendor.vendor_type,
			status=vendor.status,
			total_outstanding=Decimal(str(summary.get('total_outstanding', '0.00'))),
			current_amount=Decimal(str(summary.get('current_amount', '0.00'))),
			overdue_amount=Decimal(str(summary.get('overdue_amount', '0.00'))),
			primary_contact_name=vendor.primary_contact.name,
			primary_contact_email=vendor.primary_contact.email,
			primary_contact_phone=vendor.primary_contact.phone,
			performance_score=vendor.performance_metrics.overall_score,
			on_time_delivery_rate=vendor.performance_metrics.on_time_delivery_rate,
			invoice_accuracy_rate=vendor.performance_metrics.quality_score,
			last_invoice_date=summary.get('last_invoice_date'),
			last_payment_date=summary.get('last_payment_date'),
			total_invoices_ytd=summary.get('total_invoices_ytd', 0),
			total_payments_ytd=summary.get('total_payments_ytd', 0),
			payment_terms_display=f"{vendor.payment_terms.name} ({vendor.payment_terms.net_days} days)",
			credit_limit=vendor.credit_limit,
			created_at=vendor.created_at
		)


class APVendorDetailView(BaseModel):
	"""Comprehensive vendor detail view for vendor management screen"""
	id: str
	vendor_code: str
	legal_name: str
	trade_name: str | None = None
	vendor_type: VendorType
	status: VendorStatus
	
	# Contact information
	primary_contact: Dict[str, Any]
	addresses: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Financial information
	payment_terms: Dict[str, Any]
	banking_details: List[Dict[str, Any]] = Field(default_factory=list)
	credit_limit: Decimal | None = Field(default=None, max_digits=15, decimal_places=2)
	
	# Tax and compliance
	tax_information: Dict[str, Any]
	
	# Performance metrics
	performance_metrics: Dict[str, Any]
	
	# Financial summary
	total_outstanding: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	current_amount: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	past_due_1_30: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	past_due_31_60: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	past_due_61_90: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	past_due_over_90: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	
	# Activity summary
	ytd_invoice_count: int = 0
	ytd_invoice_amount: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	ytd_payment_count: int = 0
	ytd_payment_amount: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	
	# Recent transactions
	recent_invoices: List[Dict[str, Any]] = Field(default_factory=list)
	recent_payments: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Audit information
	created_at: datetime
	created_by: str
	updated_at: datetime
	updated_by: str | None = None
	
	model_config = VIEW_MODEL_CONFIG
	
	@classmethod
	def from_vendor(
		cls, 
		vendor: APVendor, 
		financial_summary: Dict[str, Any] | None = None,
		activity_summary: Dict[str, Any] | None = None,
		recent_transactions: Dict[str, Any] | None = None
	) -> "APVendorDetailView":
		"""Convert APVendor to detailed view model"""
		financial = financial_summary or {}
		activity = activity_summary or {}
		transactions = recent_transactions or {}
		
		return cls(
			id=vendor.id,
			vendor_code=vendor.vendor_code,
			legal_name=vendor.legal_name,
			trade_name=vendor.trade_name,
			vendor_type=vendor.vendor_type,
			status=vendor.status,
			primary_contact=vendor.primary_contact.model_dump(),
			addresses=[addr.model_dump() for addr in vendor.addresses],
			payment_terms=vendor.payment_terms.model_dump(),
			banking_details=[bank.model_dump() for bank in vendor.banking_details],
			credit_limit=vendor.credit_limit,
			tax_information=vendor.tax_information.model_dump(),
			performance_metrics=vendor.performance_metrics.model_dump(),
			total_outstanding=Decimal(str(financial.get('total_outstanding', '0.00'))),
			current_amount=Decimal(str(financial.get('current_amount', '0.00'))),
			past_due_1_30=Decimal(str(financial.get('past_due_1_30', '0.00'))),
			past_due_31_60=Decimal(str(financial.get('past_due_31_60', '0.00'))),
			past_due_61_90=Decimal(str(financial.get('past_due_61_90', '0.00'))),
			past_due_over_90=Decimal(str(financial.get('past_due_over_90', '0.00'))),
			ytd_invoice_count=activity.get('ytd_invoice_count', 0),
			ytd_invoice_amount=Decimal(str(activity.get('ytd_invoice_amount', '0.00'))),
			ytd_payment_count=activity.get('ytd_payment_count', 0),
			ytd_payment_amount=Decimal(str(activity.get('ytd_payment_amount', '0.00'))),
			recent_invoices=transactions.get('recent_invoices', []),
			recent_payments=transactions.get('recent_payments', []),
			created_at=vendor.created_at,
			created_by=vendor.created_by,
			updated_at=vendor.updated_at,
			updated_by=vendor.updated_by
		)


# Invoice View Models

class APInvoiceSummaryView(BaseModel):
	"""Invoice summary for list views with processing status"""
	id: str
	invoice_number: str
	vendor_id: str
	vendor_name: str
	vendor_invoice_number: str
	
	# Dates
	invoice_date: date
	due_date: date
	received_date: date
	
	# Amounts
	total_amount: Decimal = Field(max_digits=15, decimal_places=2)
	currency_code: str = "USD"
	
	# Status information
	status: InvoiceStatus
	matching_status: MatchingStatus
	approval_status: ApprovalStatus | None = None
	
	# Processing information
	processing_method: str  # manual, ai_processed, edi, etc.
	ocr_confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
	exception_count: int = 0
	
	# Workflow information
	current_approver: str | None = None
	days_in_workflow: int = 0
	
	# Payment information
	payment_status: str  # unpaid, partially_paid, fully_paid
	paid_amount: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	
	created_at: datetime
	
	model_config = VIEW_MODEL_CONFIG
	
	@computed_field
	@property
	def days_past_due(self) -> int:
		"""Calculate days past due"""
		today = date.today()
		if today > self.due_date:
			return (today - self.due_date).days
		return 0
	
	@computed_field
	@property
	def priority_indicator(self) -> str:
		"""Calculate priority indicator"""
		if self.days_past_due > 30 or self.total_amount > Decimal('50000'):
			return 'high'
		elif self.days_past_due > 0 or self.exception_count > 0:
			return 'medium'
		else:
			return 'normal'
	
	@classmethod
	def from_invoice(
		cls, 
		invoice: APInvoice, 
		vendor_name: str,
		workflow_info: Dict[str, Any] | None = None,
		payment_info: Dict[str, Any] | None = None
	) -> "APInvoiceSummaryView":
		"""Convert APInvoice to summary view model"""
		workflow = workflow_info or {}
		payment = payment_info or {}
		
		return cls(
			id=invoice.id,
			invoice_number=invoice.invoice_number,
			vendor_id=invoice.vendor_id,
			vendor_name=vendor_name,
			vendor_invoice_number=invoice.vendor_invoice_number,
			invoice_date=invoice.invoice_date,
			due_date=invoice.due_date,
			received_date=invoice.received_date,
			total_amount=invoice.total_amount,
			currency_code=invoice.currency_code,
			status=invoice.status,
			matching_status=invoice.matching_status,
			approval_status=workflow.get('approval_status'),
			processing_method=workflow.get('processing_method', 'manual'),
			ocr_confidence_score=invoice.ocr_confidence_score,
			exception_count=workflow.get('exception_count', 0),
			current_approver=workflow.get('current_approver'),
			days_in_workflow=workflow.get('days_in_workflow', 0),
			payment_status=payment.get('payment_status', 'unpaid'),
			paid_amount=Decimal(str(payment.get('paid_amount', '0.00'))),
			created_at=invoice.created_at
		)


class APInvoiceDetailView(BaseModel):
	"""Comprehensive invoice detail view for processing screen"""
	id: str
	invoice_number: str
	vendor_id: str
	vendor_name: str
	vendor_invoice_number: str
	
	# Dates
	invoice_date: date
	due_date: date
	received_date: date
	
	# Amounts
	subtotal_amount: Decimal = Field(max_digits=15, decimal_places=2)
	tax_amount: Decimal = Field(max_digits=15, decimal_places=2)
	total_amount: Decimal = Field(max_digits=15, decimal_places=2)
	currency_code: str = "USD"
	exchange_rate: Decimal = Field(default=Decimal('1.00'), max_digits=10, decimal_places=6)
	
	# Status and workflow
	status: InvoiceStatus
	matching_status: MatchingStatus
	approval_workflow_id: str | None = None
	
	# Processing information
	payment_terms: Dict[str, Any]
	line_items: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Document information
	document_id: str | None = None
	ocr_confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
	processing_notes: List[str] = Field(default_factory=list)
	
	# Three-way matching
	purchase_order_number: str | None = None
	matching_exceptions: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Approval workflow
	approval_history: List[Dict[str, Any]] = Field(default_factory=list)
	current_approval_step: Dict[str, Any] | None = None
	
	# Payment information
	payment_allocations: List[Dict[str, Any]] = Field(default_factory=list)
	total_paid: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	remaining_balance: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	
	# Audit information
	created_at: datetime
	created_by: str
	updated_at: datetime
	updated_by: str | None = None
	
	model_config = VIEW_MODEL_CONFIG
	
	@classmethod
	def from_invoice(
		cls,
		invoice: APInvoice,
		vendor_name: str,
		extended_data: Dict[str, Any] | None = None
	) -> "APInvoiceDetailView":
		"""Convert APInvoice to detailed view model"""
		extended = extended_data or {}
		
		return cls(
			id=invoice.id,
			invoice_number=invoice.invoice_number,
			vendor_id=invoice.vendor_id,
			vendor_name=vendor_name,
			vendor_invoice_number=invoice.vendor_invoice_number,
			invoice_date=invoice.invoice_date,
			due_date=invoice.due_date,
			received_date=invoice.received_date,
			subtotal_amount=invoice.subtotal_amount,
			tax_amount=invoice.tax_amount,
			total_amount=invoice.total_amount,
			currency_code=invoice.currency_code,
			exchange_rate=invoice.exchange_rate,
			status=invoice.status,
			matching_status=invoice.matching_status,
			approval_workflow_id=invoice.approval_workflow_id,
			payment_terms=invoice.payment_terms.model_dump(),
			line_items=[line.model_dump() for line in invoice.line_items],
			document_id=invoice.document_id,
			ocr_confidence_score=invoice.ocr_confidence_score,
			processing_notes=extended.get('processing_notes', []),
			purchase_order_number=invoice.purchase_order_number,
			matching_exceptions=extended.get('matching_exceptions', []),
			approval_history=extended.get('approval_history', []),
			current_approval_step=extended.get('current_approval_step'),
			payment_allocations=extended.get('payment_allocations', []),
			total_paid=Decimal(str(extended.get('total_paid', '0.00'))),
			remaining_balance=invoice.total_amount - Decimal(str(extended.get('total_paid', '0.00'))),
			created_at=invoice.created_at,
			created_by=invoice.created_by,
			updated_at=invoice.updated_at,
			updated_by=invoice.updated_by
		)


# Payment View Models

class APPaymentSummaryView(BaseModel):
	"""Payment summary for list views with processing status"""
	id: str
	payment_number: str
	vendor_id: str
	vendor_name: str
	
	# Payment details
	payment_method: PaymentMethod
	payment_amount: Decimal = Field(max_digits=15, decimal_places=2)
	currency_code: str = "USD"
	
	# Dates
	payment_date: date
	scheduled_date: date | None = None
	cleared_date: date | None = None
	
	# Status
	status: PaymentStatus
	bank_account: str | None = None
	reference_number: str | None = None
	
	# Invoice allocation
	invoice_count: int = 0
	discount_taken: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	
	created_at: datetime
	
	model_config = VIEW_MODEL_CONFIG
	
	@computed_field
	@property
	def status_indicator(self) -> str:
		"""Get status indicator color"""
		status_colors = {
			PaymentStatus.SCHEDULED: 'info',
			PaymentStatus.PROCESSING: 'warning',
			PaymentStatus.COMPLETED: 'success',
			PaymentStatus.FAILED: 'danger',
			PaymentStatus.CANCELLED: 'secondary',
			PaymentStatus.RETURNED: 'danger'
		}
		return status_colors.get(self.status, 'secondary')
	
	@classmethod
	def from_payment(
		cls,
		payment: APPayment,
		vendor_name: str,
		invoice_info: Dict[str, Any] | None = None
	) -> "APPaymentSummaryView":
		"""Convert APPayment to summary view model"""
		invoice = invoice_info or {}
		
		return cls(
			id=payment.id,
			payment_number=payment.payment_number,
			vendor_id=payment.vendor_id,
			vendor_name=vendor_name,
			payment_method=payment.payment_method,
			payment_amount=payment.payment_amount,
			currency_code=payment.currency_code,
			payment_date=payment.payment_date,
			scheduled_date=payment.scheduled_date,
			cleared_date=payment.cleared_date,
			status=payment.status,
			bank_account=payment.bank_account_id,
			reference_number=payment.reference_number,
			invoice_count=len(payment.payment_lines),
			discount_taken=sum(line.discount_taken for line in payment.payment_lines),
			created_at=payment.created_at
		)


# Approval Workflow View Models

class APApprovalWorkflowView(BaseModel):
	"""Approval workflow view for workflow management screen"""
	id: str
	workflow_type: str
	entity_id: str
	entity_number: str
	entity_description: str
	
	# Workflow status
	status: ApprovalStatus
	priority: str = "normal"
	current_step: int = 1
	total_steps: int = 1
	
	# Timing information
	initiated_at: datetime
	due_date: datetime | None = None
	completed_at: datetime | None = None
	
	# Approval steps
	approval_steps: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Current step details
	current_approver_name: str | None = None
	current_step_due_date: datetime | None = None
	days_pending: int = 0
	
	# Entity details (invoice/payment specific)
	amount: Decimal | None = Field(default=None, max_digits=15, decimal_places=2)
	vendor_name: str | None = None
	
	model_config = VIEW_MODEL_CONFIG
	
	@computed_field
	@property
	def progress_percentage(self) -> float:
		"""Calculate workflow progress percentage"""
		if self.total_steps == 0:
			return 0.0
		return (self.current_step - 1) / self.total_steps * 100
	
	@computed_field
	@property
	def urgency_indicator(self) -> str:
		"""Calculate urgency indicator"""
		if self.days_pending > 7 or self.priority == 'urgent':
			return 'high'
		elif self.days_pending > 3 or self.priority == 'high':
			return 'medium'
		else:
			return 'normal'
	
	@classmethod
	def from_workflow(
		cls,
		workflow: APApprovalWorkflow,
		entity_details: Dict[str, Any] | None = None
	) -> "APApprovalWorkflowView":
		"""Convert APApprovalWorkflow to view model"""
		details = entity_details or {}
		
		# Calculate days pending
		days_pending = (datetime.utcnow() - workflow.initiated_at).days
		
		return cls(
			id=workflow.id,
			workflow_type=workflow.workflow_type,
			entity_id=workflow.entity_id,
			entity_number=workflow.entity_number,
			entity_description=details.get('description', workflow.entity_number),
			status=workflow.status,
			priority=workflow.priority,
			current_step=workflow.current_step,
			total_steps=len(workflow.approval_steps),
			initiated_at=workflow.initiated_at,
			due_date=workflow.due_date,
			completed_at=workflow.completed_at,
			approval_steps=[step.model_dump() for step in workflow.approval_steps],
			current_approver_name=details.get('current_approver_name'),
			current_step_due_date=details.get('current_step_due_date'),
			days_pending=days_pending,
			amount=details.get('amount'),
			vendor_name=details.get('vendor_name')
		)


# Analytics and Reporting View Models

class APAgingReportView(BaseModel):
	"""Accounts payable aging report view model"""
	vendor_id: str
	vendor_name: str
	vendor_code: str
	
	# Aging buckets
	current_amount: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	past_due_1_30: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	past_due_31_60: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	past_due_61_90: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	past_due_over_90: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	total_outstanding: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	
	# Analysis date
	as_of_date: date = Field(default_factory=date.today)
	
	# Contact information for collections
	primary_contact_name: str | None = None
	primary_contact_email: str | None = None
	primary_contact_phone: str | None = None
	
	model_config = VIEW_MODEL_CONFIG
	
	@computed_field
	@property
	def risk_score(self) -> str:
		"""Calculate collection risk score"""
		if self.past_due_over_90 > Decimal('10000'):
			return 'high'
		elif self.past_due_31_60 + self.past_due_61_90 > Decimal('5000'):
			return 'medium'
		else:
			return 'low'
	
	@computed_field
	@property
	def weighted_days_outstanding(self) -> float:
		"""Calculate weighted average days outstanding"""
		if self.total_outstanding == 0:
			return 0.0
		
		weighted_sum = (
			float(self.current_amount) * 0 +
			float(self.past_due_1_30) * 15 +
			float(self.past_due_31_60) * 45 +
			float(self.past_due_61_90) * 75 +
			float(self.past_due_over_90) * 120
		)
		
		return weighted_sum / float(self.total_outstanding)


class APCashFlowForecastView(BaseModel):
	"""Cash flow forecast view model for dashboard display"""
	forecast_date: date
	projected_outflow: Decimal = Field(max_digits=15, decimal_places=2)
	confidence_level: float = Field(ge=0.0, le=1.0)
	
	# Breakdown by category
	scheduled_payments: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	early_payments: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	urgent_payments: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	
	# Optimization opportunities
	discount_opportunities: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	potential_savings: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	
	model_config = VIEW_MODEL_CONFIG


# Form and Input View Models

class APVendorCreateForm(BaseModel):
	"""Vendor creation form model"""
	vendor_code: str = Field(min_length=3, max_length=20)
	legal_name: str = Field(min_length=1, max_length=200)
	trade_name: str | None = Field(default=None, max_length=200)
	vendor_type: VendorType
	
	# Contact information
	contact_name: str = Field(min_length=1, max_length=100)
	contact_title: str | None = Field(default=None, max_length=100)
	contact_email: str | None = Field(default=None, max_length=255)
	contact_phone: str | None = Field(default=None, max_length=50)
	
	# Address information
	address_line1: str = Field(min_length=1, max_length=100)
	address_line2: str | None = Field(default=None, max_length=100)
	city: str = Field(min_length=1, max_length=50)
	state_province: str = Field(min_length=1, max_length=50)
	postal_code: str = Field(min_length=1, max_length=20)
	country_code: str = Field(min_length=2, max_length=3)
	
	# Payment terms
	payment_terms_code: str = Field(min_length=1, max_length=20)
	net_days: int = Field(ge=0, le=365)
	discount_days: int = Field(default=0, ge=0, le=365)
	discount_percent: Decimal = Field(default=Decimal('0.00'), ge=0, le=100, max_digits=5, decimal_places=2)
	
	# Tax information
	tax_id: str | None = Field(default=None, max_length=50)
	tax_id_type: str | None = Field(default=None, max_length=20)
	is_1099_vendor: bool = False
	
	# Banking information
	bank_name: str | None = Field(default=None, max_length=100)
	routing_number: str | None = Field(default=None, max_length=20)
	account_number: str | None = Field(default=None, max_length=50)
	account_holder_name: str | None = Field(default=None, max_length=100)
	
	model_config = VIEW_MODEL_CONFIG


class APInvoiceProcessingForm(BaseModel):
	"""Invoice processing form model"""
	vendor_id: str
	processing_method: str = "manual"  # manual, upload, ai_processing
	
	# Basic invoice information
	vendor_invoice_number: str = Field(min_length=1, max_length=50)
	invoice_date: date
	due_date: date | None = None
	
	# Amounts
	subtotal_amount: Decimal = Field(gt=0, max_digits=15, decimal_places=2)
	tax_amount: Decimal = Field(default=Decimal('0.00'), ge=0, max_digits=15, decimal_places=2)
	total_amount: Decimal = Field(gt=0, max_digits=15, decimal_places=2)
	currency_code: str = "USD"
	
	# Purchase order matching
	purchase_order_number: str | None = Field(default=None, max_length=50)
	require_three_way_matching: bool = True
	
	# Processing notes
	processing_notes: str | None = Field(default=None, max_length=1000)
	
	model_config = VIEW_MODEL_CONFIG


class APPaymentCreateForm(BaseModel):
	"""Payment creation form model"""
	vendor_id: str
	payment_method: PaymentMethod
	payment_date: date = Field(default_factory=date.today)
	
	# Payment details
	payment_amount: Decimal = Field(gt=0, max_digits=15, decimal_places=2)
	currency_code: str = "USD"
	bank_account_id: str | None = None
	reference_number: str | None = Field(default=None, max_length=50)
	
	# Invoice allocation
	invoice_allocations: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Payment options
	take_early_discount: bool = True
	payment_memo: str | None = Field(default=None, max_length=500)
	
	model_config = VIEW_MODEL_CONFIG


# Export all view models
__all__ = [
	# Dashboard models
	'APDashboardSummary', 'APPerformanceMetrics',
	
	# Vendor models
	'APVendorSummaryView', 'APVendorDetailView',
	
	# Invoice models
	'APInvoiceSummaryView', 'APInvoiceDetailView',
	
	# Payment models
	'APPaymentSummaryView',
	
	# Workflow models
	'APApprovalWorkflowView',
	
	# Analytics models
	'APAgingReportView', 'APCashFlowForecastView',
	
	# Form models
	'APVendorCreateForm', 'APInvoiceProcessingForm', 'APPaymentCreateForm'
]