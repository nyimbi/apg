"""
Accounts Payable Service

Business logic for Accounts Payable operations including vendor management,
invoice processing, payment processing, expense reports, and aging analysis.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func

from .models import (
	CFAPVendor, CFAPInvoice, CFAPInvoiceLine, CFAPPayment, CFAPPaymentLine,
	CFAPExpenseReport, CFAPExpenseLine, CFAPPurchaseOrder, CFAPTaxCode, CFAPAging
)
from ..general_ledger.models import CFGLJournalEntry, CFGLJournalLine, CFGLAccount
from ...auth_rbac.models import db


class AccountsPayableService:
	"""Service class for Accounts Payable operations"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
	
	# Vendor Management
	
	def create_vendor(self, vendor_data: Dict[str, Any]) -> CFAPVendor:
		"""Create a new vendor"""
		vendor = CFAPVendor(
			tenant_id=self.tenant_id,
			vendor_number=vendor_data['vendor_number'],
			vendor_name=vendor_data['vendor_name'],
			vendor_type=vendor_data.get('vendor_type', 'SUPPLIER'),
			contact_name=vendor_data.get('contact_name'),
			email=vendor_data.get('email'),
			phone=vendor_data.get('phone'),
			address_line1=vendor_data.get('address_line1'),
			address_line2=vendor_data.get('address_line2'),
			city=vendor_data.get('city'),
			state_province=vendor_data.get('state_province'),
			postal_code=vendor_data.get('postal_code'),
			country=vendor_data.get('country'),
			payment_terms_code=vendor_data.get('payment_terms_code', 'NET_30'),
			payment_method=vendor_data.get('payment_method', 'CHECK'),
			currency_code=vendor_data.get('currency_code', 'USD'),
			bank_name=vendor_data.get('bank_name'),
			bank_account_number=vendor_data.get('bank_account_number'),
			bank_routing_number=vendor_data.get('bank_routing_number'),
			tax_id=vendor_data.get('tax_id'),
			tax_exempt=vendor_data.get('tax_exempt', False),
			is_active=vendor_data.get('is_active', True),
			is_employee=vendor_data.get('is_employee', False),
			credit_limit=vendor_data.get('credit_limit', 0.00),
			require_po=vendor_data.get('require_po', False),
			is_1099_vendor=vendor_data.get('is_1099_vendor', False),
			notes=vendor_data.get('notes')
		)
		
		db.session.add(vendor)
		db.session.commit()
		return vendor
	
	def get_vendor(self, vendor_id: str) -> Optional[CFAPVendor]:
		"""Get vendor by ID"""
		return CFAPVendor.query.filter_by(
			tenant_id=self.tenant_id,
			vendor_id=vendor_id
		).first()
	
	def get_vendor_by_number(self, vendor_number: str) -> Optional[CFAPVendor]:
		"""Get vendor by vendor number"""
		return CFAPVendor.query.filter_by(
			tenant_id=self.tenant_id,
			vendor_number=vendor_number
		).first()
	
	def get_vendors(self, include_inactive: bool = False) -> List[CFAPVendor]:
		"""Get all vendors"""
		query = CFAPVendor.query.filter_by(tenant_id=self.tenant_id)
		
		if not include_inactive:
			query = query.filter_by(is_active=True)
		
		return query.order_by(CFAPVendor.vendor_name).all()
	
	def update_vendor_balance(self, vendor_id: str):
		"""Update vendor's current balance"""
		vendor = self.get_vendor(vendor_id)
		if not vendor:
			return
		
		# Calculate outstanding balance
		outstanding = vendor.get_outstanding_balance()
		vendor.current_balance = outstanding
		
		# Calculate YTD purchases
		ytd_invoices = CFAPInvoice.query.filter(
			CFAPInvoice.tenant_id == self.tenant_id,
			CFAPInvoice.vendor_id == vendor_id,
			CFAPInvoice.status == 'Posted',
			func.extract('year', CFAPInvoice.invoice_date) == date.today().year
		).all()
		
		vendor.ytd_purchases = sum(inv.total_amount for inv in ytd_invoices)
		
		db.session.commit()
	
	# Invoice Management
	
	def create_invoice(self, invoice_data: Dict[str, Any]) -> CFAPInvoice:
		"""Create a new vendor invoice"""
		invoice = CFAPInvoice(
			tenant_id=self.tenant_id,
			invoice_number=invoice_data['invoice_number'],
			vendor_invoice_number=invoice_data['vendor_invoice_number'],
			vendor_id=invoice_data['vendor_id'],
			invoice_date=invoice_data['invoice_date'],
			due_date=invoice_data.get('due_date') or self._calculate_due_date(
				invoice_data['invoice_date'], 
				invoice_data['vendor_id']
			),
			description=invoice_data.get('description'),
			purchase_order_id=invoice_data.get('purchase_order_id'),
			subtotal_amount=invoice_data.get('subtotal_amount', 0.00),
			tax_amount=invoice_data.get('tax_amount', 0.00),
			freight_amount=invoice_data.get('freight_amount', 0.00),
			misc_amount=invoice_data.get('misc_amount', 0.00),
			discount_amount=invoice_data.get('discount_amount', 0.00),
			currency_code=invoice_data.get('currency_code', 'USD'),
			exchange_rate=invoice_data.get('exchange_rate', 1.000000),
			requires_approval=invoice_data.get('requires_approval', True),
			notes=invoice_data.get('notes')
		)
		
		# Calculate total
		invoice.total_amount = (invoice.subtotal_amount + invoice.tax_amount + 
								invoice.freight_amount + invoice.misc_amount - 
								invoice.discount_amount)
		invoice.outstanding_amount = invoice.total_amount
		
		db.session.add(invoice)
		db.session.commit()
		return invoice
	
	def add_invoice_line(self, invoice_id: str, line_data: Dict[str, Any]) -> CFAPInvoiceLine:
		"""Add a line to an invoice"""
		invoice = self.get_invoice(invoice_id)
		if not invoice or invoice.status != 'Draft':
			raise ValueError("Cannot add lines to non-draft invoice")
		
		line = CFAPInvoiceLine(
			tenant_id=self.tenant_id,
			invoice_id=invoice_id,
			line_number=line_data['line_number'],
			description=line_data.get('description'),
			item_code=line_data.get('item_code'),
			quantity=line_data.get('quantity', 1.0000),
			unit_price=line_data.get('unit_price', 0.0000),
			line_amount=line_data['line_amount'],
			gl_account_id=line_data['gl_account_id'],
			tax_code=line_data.get('tax_code'),
			tax_rate=line_data.get('tax_rate', 0.00),
			is_tax_inclusive=line_data.get('is_tax_inclusive', False),
			cost_center=line_data.get('cost_center'),
			department=line_data.get('department'),
			project=line_data.get('project'),
			is_asset=line_data.get('is_asset', False),
			asset_id=line_data.get('asset_id')
		)
		
		# Calculate tax
		line.calculate_tax()
		
		db.session.add(line)
		
		# Recalculate invoice totals
		invoice.calculate_totals()
		
		db.session.commit()
		return line
	
	def get_invoice(self, invoice_id: str) -> Optional[CFAPInvoice]:
		"""Get invoice by ID"""
		return CFAPInvoice.query.filter_by(
			tenant_id=self.tenant_id,
			invoice_id=invoice_id
		).first()
	
	def get_invoices(self, status: Optional[str] = None, vendor_id: Optional[str] = None) -> List[CFAPInvoice]:
		"""Get invoices with optional filters"""
		query = CFAPInvoice.query.filter_by(tenant_id=self.tenant_id)
		
		if status:
			query = query.filter_by(status=status)
		
		if vendor_id:
			query = query.filter_by(vendor_id=vendor_id)
		
		return query.order_by(desc(CFAPInvoice.invoice_date)).all()
	
	def approve_invoice(self, invoice_id: str, user_id: str) -> bool:
		"""Approve an invoice"""
		invoice = self.get_invoice(invoice_id)
		if not invoice or not invoice.can_approve():
			return False
		
		invoice.approve_invoice(user_id)
		
		# Update vendor balance
		self.update_vendor_balance(invoice.vendor_id)
		
		db.session.commit()
		return True
	
	def post_invoice(self, invoice_id: str, user_id: str) -> bool:
		"""Post invoice to General Ledger"""
		invoice = self.get_invoice(invoice_id)
		if not invoice or not invoice.can_post():
			return False
		
		# Create GL journal entry
		journal_entry = self._create_invoice_journal_entry(invoice, user_id)
		
		# Post the invoice
		invoice.post_invoice(user_id)
		invoice.gl_batch_id = journal_entry.journal_id
		
		# Update vendor balance
		self.update_vendor_balance(invoice.vendor_id)
		
		db.session.commit()
		return True
	
	def _calculate_due_date(self, invoice_date: date, vendor_id: str) -> date:
		"""Calculate due date based on payment terms"""
		vendor = self.get_vendor(vendor_id)
		if not vendor:
			return invoice_date + timedelta(days=30)
		
		# Simplified payment terms calculation
		payment_terms = vendor.payment_terms_code or 'NET_30'
		
		if payment_terms == 'NET_15':
			return invoice_date + timedelta(days=15)
		elif payment_terms == 'NET_30':
			return invoice_date + timedelta(days=30)
		elif payment_terms == 'NET_60':
			return invoice_date + timedelta(days=60)
		elif payment_terms == 'NET_90':
			return invoice_date + timedelta(days=90)
		elif payment_terms == 'DUE_ON_RECEIPT':
			return invoice_date
		else:
			# Extract days from terms like '2_10_NET_30'
			parts = payment_terms.split('_')
			if len(parts) >= 3 and parts[2] == 'NET':
				try:
					days = int(parts[3])
					return invoice_date + timedelta(days=days)
				except (ValueError, IndexError):
					pass
		
		return invoice_date + timedelta(days=30)
	
	def _create_invoice_journal_entry(self, invoice: CFAPInvoice, user_id: str) -> CFGLJournalEntry:
		"""Create GL journal entry for invoice posting"""
		from ..general_ledger.service import GeneralLedgerService
		
		gl_service = GeneralLedgerService(self.tenant_id)
		
		# Get AP control account
		ap_account = CFGLAccount.query.filter_by(
			tenant_id=self.tenant_id,
			account_code='2110'  # Accounts Payable
		).first()
		
		if not ap_account:
			raise ValueError("Accounts Payable GL account not found")
		
		# Create journal entry
		journal_data = {
			'description': f'AP Invoice {invoice.invoice_number} - {invoice.vendor.vendor_name}',
			'reference': invoice.vendor_invoice_number,
			'entry_date': invoice.invoice_date,
			'posting_date': date.today(),
			'source': 'AP'
		}
		
		journal_entry = gl_service.create_journal_entry(journal_data)
		
		# Create expense/asset lines
		line_number = 1
		for inv_line in invoice.lines:
			gl_service.add_journal_line(journal_entry.journal_id, {
				'line_number': line_number,
				'account_id': inv_line.gl_account_id,
				'debit_amount': inv_line.line_amount,
				'credit_amount': 0.00,
				'description': inv_line.description or f'Invoice {invoice.invoice_number}',
				'reference_type': 'AP_Invoice',
				'reference_id': invoice.invoice_id,
				'cost_center': inv_line.cost_center,
				'department': inv_line.department,
				'project': inv_line.project
			})
			line_number += 1
		
		# Create tax line if applicable
		if invoice.tax_amount > 0:
			# Find tax payable account
			tax_account = CFGLAccount.query.filter_by(
				tenant_id=self.tenant_id,
				account_code='2300'  # Tax Payable
			).first()
			
			if tax_account:
				gl_service.add_journal_line(journal_entry.journal_id, {
					'line_number': line_number,
					'account_id': tax_account.account_id,
					'debit_amount': invoice.tax_amount,
					'credit_amount': 0.00,
					'description': f'Tax on Invoice {invoice.invoice_number}',
					'reference_type': 'AP_Invoice',
					'reference_id': invoice.invoice_id
				})
				line_number += 1
		
		# Create AP credit line
		gl_service.add_journal_line(journal_entry.journal_id, {
			'line_number': line_number,
			'account_id': ap_account.account_id,
			'debit_amount': 0.00,
			'credit_amount': invoice.total_amount,
			'description': f'AP Liability - {invoice.vendor.vendor_name}',
			'reference_type': 'AP_Invoice',
			'reference_id': invoice.invoice_id
		})
		
		# Post the journal entry
		gl_service.post_journal_entry(journal_entry.journal_id, user_id)
		
		return journal_entry
	
	# Payment Management
	
	def create_payment(self, payment_data: Dict[str, Any]) -> CFAPPayment:
		"""Create a new payment"""
		payment = CFAPPayment(
			tenant_id=self.tenant_id,
			payment_number=payment_data['payment_number'],
			vendor_id=payment_data['vendor_id'],
			payment_date=payment_data['payment_date'],
			payment_method=payment_data.get('payment_method', 'CHECK'),
			check_number=payment_data.get('check_number'),
			bank_account_id=payment_data.get('bank_account_id'),
			description=payment_data.get('description'),
			currency_code=payment_data.get('currency_code', 'USD'),
			exchange_rate=payment_data.get('exchange_rate', 1.000000),
			requires_approval=payment_data.get('requires_approval', True),
			notes=payment_data.get('notes')
		)
		
		db.session.add(payment)
		db.session.commit()
		return payment
	
	def add_payment_line(self, payment_id: str, line_data: Dict[str, Any]) -> CFAPPaymentLine:
		"""Add an invoice allocation line to a payment"""
		payment = self.get_payment(payment_id)
		if not payment or payment.status != 'Draft':
			raise ValueError("Cannot add lines to non-draft payment")
		
		invoice = self.get_invoice(line_data['invoice_id'])
		if not invoice or not invoice.can_pay():
			raise ValueError("Invoice cannot be paid")
		
		line = CFAPPaymentLine(
			tenant_id=self.tenant_id,
			payment_id=payment_id,
			line_number=line_data['line_number'],
			invoice_id=line_data['invoice_id'],
			invoice_amount=invoice.total_amount,
			payment_amount=line_data['payment_amount'],
			discount_taken=line_data.get('discount_taken', 0.00),
			notes=line_data.get('notes')
		)
		
		line.remaining_amount = line.invoice_amount - line.payment_amount - line.discount_taken
		
		db.session.add(line)
		
		# Recalculate payment totals
		payment.calculate_totals()
		
		db.session.commit()
		return line
	
	def get_payment(self, payment_id: str) -> Optional[CFAPPayment]:
		"""Get payment by ID"""
		return CFAPPayment.query.filter_by(
			tenant_id=self.tenant_id,
			payment_id=payment_id
		).first()
	
	def get_payments(self, status: Optional[str] = None, vendor_id: Optional[str] = None) -> List[CFAPPayment]:
		"""Get payments with optional filters"""
		query = CFAPPayment.query.filter_by(tenant_id=self.tenant_id)
		
		if status:
			query = query.filter_by(status=status)
		
		if vendor_id:
			query = query.filter_by(vendor_id=vendor_id)
		
		return query.order_by(desc(CFAPPayment.payment_date)).all()
	
	def approve_payment(self, payment_id: str, user_id: str) -> bool:
		"""Approve a payment"""
		payment = self.get_payment(payment_id)
		if not payment or not payment.can_approve():
			return False
		
		payment.approve_payment(user_id)
		db.session.commit()
		return True
	
	def post_payment(self, payment_id: str, user_id: str) -> bool:
		"""Post payment to General Ledger"""
		payment = self.get_payment(payment_id)
		if not payment or not payment.can_post():
			return False
		
		# Create GL journal entry
		journal_entry = self._create_payment_journal_entry(payment, user_id)
		
		# Post the payment
		payment.post_payment(user_id)
		payment.gl_batch_id = journal_entry.journal_id
		
		# Update invoice payment status
		for line in payment.payment_lines:
			invoice = line.invoice
			invoice.paid_amount += line.payment_amount
			invoice.outstanding_amount = invoice.total_amount - invoice.paid_amount
			
			if invoice.outstanding_amount <= 0.01:
				invoice.payment_status = 'Paid'
			else:
				invoice.payment_status = 'Partial'
		
		# Update vendor balance
		self.update_vendor_balance(payment.vendor_id)
		
		db.session.commit()
		return True
	
	def _create_payment_journal_entry(self, payment: CFAPPayment, user_id: str) -> CFGLJournalEntry:
		"""Create GL journal entry for payment posting"""
		from ..general_ledger.service import GeneralLedgerService
		
		gl_service = GeneralLedgerService(self.tenant_id)
		
		# Get required GL accounts
		ap_account = CFGLAccount.query.filter_by(
			tenant_id=self.tenant_id,
			account_code='2110'  # Accounts Payable
		).first()
		
		cash_account = CFGLAccount.query.filter_by(
			tenant_id=self.tenant_id,
			account_code='1105'  # Cash Clearing
		).first()
		
		discount_account = CFGLAccount.query.filter_by(
			tenant_id=self.tenant_id,
			account_code='4200'  # Purchase Discounts
		).first()
		
		if not ap_account or not cash_account:
			raise ValueError("Required GL accounts not found")
		
		# Create journal entry
		journal_data = {
			'description': f'AP Payment {payment.payment_number} - {payment.vendor.vendor_name}',
			'reference': payment.check_number or payment.payment_number,
			'entry_date': payment.payment_date,
			'posting_date': payment.payment_date,
			'source': 'AP'
		}
		
		journal_entry = gl_service.create_journal_entry(journal_data)
		
		# Debit AP for total amount being paid
		gl_service.add_journal_line(journal_entry.journal_id, {
			'line_number': 1,
			'account_id': ap_account.account_id,
			'debit_amount': payment.total_amount,
			'credit_amount': 0.00,
			'description': f'Payment to {payment.vendor.vendor_name}',
			'reference_type': 'AP_Payment',
			'reference_id': payment.payment_id
		})
		
		# Credit discount account if discount taken
		line_number = 2
		if payment.discount_taken > 0 and discount_account:
			gl_service.add_journal_line(journal_entry.journal_id, {
				'line_number': line_number,
				'account_id': discount_account.account_id,
				'debit_amount': 0.00,
				'credit_amount': payment.discount_taken,
				'description': f'Discount taken - {payment.vendor.vendor_name}',
				'reference_type': 'AP_Payment',
				'reference_id': payment.payment_id
			})
			line_number += 1
		
		# Credit cash for payment amount
		gl_service.add_journal_line(journal_entry.journal_id, {
			'line_number': line_number,
			'account_id': cash_account.account_id,
			'debit_amount': 0.00,
			'credit_amount': payment.payment_amount,
			'description': f'Payment via {payment.payment_method}',
			'reference_type': 'AP_Payment',
			'reference_id': payment.payment_id
		})
		
		# Post the journal entry
		gl_service.post_journal_entry(journal_entry.journal_id, user_id)
		
		return journal_entry
	
	# Expense Report Management
	
	def create_expense_report(self, report_data: Dict[str, Any]) -> CFAPExpenseReport:
		"""Create a new expense report"""
		expense_report = CFAPExpenseReport(
			tenant_id=self.tenant_id,
			report_number=report_data['report_number'],
			report_name=report_data['report_name'],
			employee_id=report_data['employee_id'],
			employee_name=report_data['employee_name'],
			department=report_data.get('department'),
			vendor_id=report_data.get('vendor_id'),
			report_date=report_data['report_date'],
			period_start=report_data['period_start'],
			period_end=report_data['period_end'],
			description=report_data.get('description'),
			currency_code=report_data.get('currency_code', 'USD'),
			requires_approval=report_data.get('requires_approval', True),
			notes=report_data.get('notes')
		)
		
		db.session.add(expense_report)
		db.session.commit()
		return expense_report
	
	def add_expense_line(self, report_id: str, line_data: Dict[str, Any]) -> CFAPExpenseLine:
		"""Add a line to an expense report"""
		report = self.get_expense_report(report_id)
		if not report or report.status != 'Draft':
			raise ValueError("Cannot add lines to non-draft expense report")
		
		line = CFAPExpenseLine(
			tenant_id=self.tenant_id,
			expense_report_id=report_id,
			line_number=line_data['line_number'],
			description=line_data['description'],
			expense_date=line_data['expense_date'],
			expense_category=line_data['expense_category'],
			merchant_name=line_data.get('merchant_name'),
			location=line_data.get('location'),
			amount=line_data['amount'],
			currency_code=line_data.get('currency_code', 'USD'),
			exchange_rate=line_data.get('exchange_rate', 1.000000),
			tax_code=line_data.get('tax_code'),
			tax_amount=line_data.get('tax_amount', 0.00),
			is_tax_inclusive=line_data.get('is_tax_inclusive', True),
			gl_account_id=line_data['gl_account_id'],
			is_reimbursable=line_data.get('is_reimbursable', True),
			reimbursement_rate=line_data.get('reimbursement_rate', 100.00),
			is_mileage=line_data.get('is_mileage', False),
			mileage_distance=line_data.get('mileage_distance', 0.00),
			mileage_rate=line_data.get('mileage_rate', 0.0000),
			has_receipt=line_data.get('has_receipt', False),
			receipt_path=line_data.get('receipt_path'),
			is_personal=line_data.get('is_personal', False),
			business_percentage=line_data.get('business_percentage', 100.00),
			cost_center=line_data.get('cost_center'),
			project=line_data.get('project'),
			notes=line_data.get('notes')
		)
		
		# Calculate home currency amount
		line.home_currency_amount = line.amount * line.exchange_rate
		
		db.session.add(line)
		
		# Recalculate report totals
		report.calculate_totals()
		
		db.session.commit()
		return line
	
	def get_expense_report(self, report_id: str) -> Optional[CFAPExpenseReport]:
		"""Get expense report by ID"""
		return CFAPExpenseReport.query.filter_by(
			tenant_id=self.tenant_id,
			expense_report_id=report_id
		).first()
	
	def get_expense_reports(self, status: Optional[str] = None, employee_id: Optional[str] = None) -> List[CFAPExpenseReport]:
		"""Get expense reports with optional filters"""
		query = CFAPExpenseReport.query.filter_by(tenant_id=self.tenant_id)
		
		if status:
			query = query.filter_by(status=status)
		
		if employee_id:
			query = query.filter_by(employee_id=employee_id)
		
		return query.order_by(desc(CFAPExpenseReport.report_date)).all()
	
	def submit_expense_report(self, report_id: str) -> bool:
		"""Submit expense report for approval"""
		report = self.get_expense_report(report_id)
		if not report or not report.can_submit():
			return False
		
		report.submit_report()
		db.session.commit()
		return True
	
	def approve_expense_report(self, report_id: str, user_id: str) -> bool:
		"""Approve an expense report"""
		report = self.get_expense_report(report_id)
		if not report or not report.can_approve():
			return False
		
		report.approve_report(user_id)
		db.session.commit()
		return True
	
	def reject_expense_report(self, report_id: str, user_id: str, reason: str) -> bool:
		"""Reject an expense report"""
		report = self.get_expense_report(report_id)
		if not report or not report.can_approve():
			return False
		
		report.reject_report(user_id, reason)
		db.session.commit()
		return True
	
	# Aging Analysis
	
	def generate_aging_report(self, as_of_date: Optional[date] = None, vendor_id: Optional[str] = None) -> List[CFAPAging]:
		"""Generate AP aging analysis"""
		if as_of_date is None:
			as_of_date = date.today()
		
		# Get vendors to analyze
		vendor_query = CFAPVendor.query.filter_by(tenant_id=self.tenant_id, is_active=True)
		if vendor_id:
			vendor_query = vendor_query.filter_by(vendor_id=vendor_id)
		
		vendors = vendor_query.all()
		aging_records = []
		
		for vendor in vendors:
			aging = self._calculate_vendor_aging(vendor, as_of_date)
			if aging.total_outstanding > 0:
				aging_records.append(aging)
		
		return aging_records
	
	def _calculate_vendor_aging(self, vendor: CFAPVendor, as_of_date: date) -> CFAPAging:
		"""Calculate aging for a specific vendor"""
		# Get outstanding invoices
		outstanding_invoices = CFAPInvoice.query.filter(
			CFAPInvoice.tenant_id == self.tenant_id,
			CFAPInvoice.vendor_id == vendor.vendor_id,
			CFAPInvoice.status == 'Posted',
			CFAPInvoice.outstanding_amount > 0,
			CFAPInvoice.invoice_date <= as_of_date
		).all()
		
		aging = CFAPAging(
			tenant_id=self.tenant_id,
			as_of_date=as_of_date,
			vendor_id=vendor.vendor_id,
			generated_by='system'  # Should be actual user_id
		)
		
		for invoice in outstanding_invoices:
			days_outstanding = (as_of_date - invoice.due_date).days
			amount = invoice.outstanding_amount
			
			if days_outstanding <= 0:
				aging.current_amount += amount
			elif days_outstanding <= 30:
				aging.current_amount += amount
			elif days_outstanding <= 60:
				aging.days_31_60 += amount
			elif days_outstanding <= 90:
				aging.days_61_90 += amount
			elif days_outstanding <= 120:
				aging.days_91_120 += amount
			else:
				aging.over_120_days += amount
		
		aging.total_outstanding = (aging.current_amount + aging.days_31_60 +
								   aging.days_61_90 + aging.days_91_120 + aging.over_120_days)
		
		return aging
	
	def save_aging_snapshot(self, aging_records: List[CFAPAging], user_id: str):
		"""Save aging analysis snapshot to database"""
		for aging in aging_records:
			aging.generated_by = user_id
			aging.generated_date = datetime.utcnow()
			
			# Check if record already exists for this date/vendor
			existing = CFAPAging.query.filter_by(
				tenant_id=self.tenant_id,
				as_of_date=aging.as_of_date,
				vendor_id=aging.vendor_id
			).first()
			
			if existing:
				# Update existing record
				existing.current_amount = aging.current_amount
				existing.days_31_60 = aging.days_31_60
				existing.days_61_90 = aging.days_61_90
				existing.days_91_120 = aging.days_91_120
				existing.over_120_days = aging.over_120_days
				existing.total_outstanding = aging.total_outstanding
				existing.generated_date = aging.generated_date
				existing.generated_by = aging.generated_by
			else:
				db.session.add(aging)
		
		db.session.commit()
	
	# Reporting and Analytics
	
	def get_ap_summary(self) -> Dict[str, Any]:
		"""Get AP summary statistics"""
		# Outstanding invoices
		outstanding_invoices = CFAPInvoice.query.filter(
			CFAPInvoice.tenant_id == self.tenant_id,
			CFAPInvoice.status == 'Posted',
			CFAPInvoice.outstanding_amount > 0
		).all()
		
		# Pending approvals
		pending_invoices = CFAPInvoice.query.filter(
			CFAPInvoice.tenant_id == self.tenant_id,
			CFAPInvoice.status.in_(['Draft', 'Pending'])
		).count()
		
		pending_payments = CFAPPayment.query.filter(
			CFAPPayment.tenant_id == self.tenant_id,
			CFAPPayment.status.in_(['Draft', 'Pending'])
		).count()
		
		# Calculate totals
		total_outstanding = sum(inv.outstanding_amount for inv in outstanding_invoices)
		overdue_amount = sum(
			inv.outstanding_amount for inv in outstanding_invoices
			if inv.due_date < date.today()
		)
		
		# Active vendors
		active_vendors = CFAPVendor.query.filter_by(
			tenant_id=self.tenant_id,
			is_active=True
		).count()
		
		return {
			'total_outstanding': float(total_outstanding),
			'overdue_amount': float(overdue_amount),
			'pending_invoices': pending_invoices,
			'pending_payments': pending_payments,
			'active_vendors': active_vendors,
			'outstanding_invoice_count': len(outstanding_invoices)
		}
	
	def get_cash_requirements(self, days_forward: int = 30) -> List[Dict[str, Any]]:
		"""Get cash requirements for upcoming payments"""
		end_date = date.today() + timedelta(days=days_forward)
		
		due_invoices = CFAPInvoice.query.filter(
			CFAPInvoice.tenant_id == self.tenant_id,
			CFAPInvoice.status == 'Posted',
			CFAPInvoice.outstanding_amount > 0,
			CFAPInvoice.due_date <= end_date
		).order_by(CFAPInvoice.due_date).all()
		
		cash_requirements = []
		for invoice in due_invoices:
			days_until_due = (invoice.due_date - date.today()).days
			
			cash_requirements.append({
				'invoice_id': invoice.invoice_id,
				'invoice_number': invoice.invoice_number,
				'vendor_name': invoice.vendor.vendor_name,
				'due_date': invoice.due_date.isoformat(),
				'days_until_due': days_until_due,
				'amount': float(invoice.outstanding_amount),
				'is_overdue': days_until_due < 0
			})
		
		return cash_requirements
	
	# Utility Methods
	
	def _log_vendor_name(self, vendor: CFAPVendor) -> str:
		"""Format vendor name for logging"""
		return f"{vendor.vendor_number} - {vendor.vendor_name}"
	
	def _log_invoice_details(self, invoice: CFAPInvoice) -> str:
		"""Format invoice details for logging"""
		return f"Invoice {invoice.invoice_number} (${invoice.total_amount}) - {invoice.vendor.vendor_name}"
	
	def _log_payment_details(self, payment: CFAPPayment) -> str:
		"""Format payment details for logging"""
		return f"Payment {payment.payment_number} (${payment.total_amount}) - {payment.vendor.vendor_name}"