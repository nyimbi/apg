"""
Accounts Receivable Service

Business logic for Accounts Receivable operations including customer management,
invoice processing, payment processing, collections management, and aging analysis.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func

from .models import (
	CFARCustomer, CFARInvoice, CFARInvoiceLine, CFARPayment, CFARPaymentLine,
	CFARCreditMemo, CFARCreditMemoLine, CFARStatement, CFARCollection,
	CFARAging, CFARTaxCode, CFARRecurringBilling
)
from ..general_ledger.models import CFGLJournalEntry, CFGLJournalLine, CFGLAccount
from ...auth_rbac.models import db


class AccountsReceivableService:
	"""Service class for Accounts Receivable operations"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
	
	# Customer Management
	
	def create_customer(self, customer_data: Dict[str, Any]) -> CFARCustomer:
		"""Create a new customer"""
		customer = CFARCustomer(
			tenant_id=self.tenant_id,
			customer_number=customer_data['customer_number'],
			customer_name=customer_data['customer_name'],
			customer_type=customer_data.get('customer_type', 'RETAIL'),
			contact_name=customer_data.get('contact_name'),
			email=customer_data.get('email'),
			phone=customer_data.get('phone'),
			billing_address_line1=customer_data.get('billing_address_line1'),
			billing_address_line2=customer_data.get('billing_address_line2'),
			billing_city=customer_data.get('billing_city'),
			billing_state_province=customer_data.get('billing_state_province'),
			billing_postal_code=customer_data.get('billing_postal_code'),
			billing_country=customer_data.get('billing_country'),
			shipping_address_line1=customer_data.get('shipping_address_line1'),
			shipping_address_line2=customer_data.get('shipping_address_line2'),
			shipping_city=customer_data.get('shipping_city'),
			shipping_state_province=customer_data.get('shipping_state_province'),
			shipping_postal_code=customer_data.get('shipping_postal_code'),
			shipping_country=customer_data.get('shipping_country'),
			payment_terms_code=customer_data.get('payment_terms_code', 'NET_30'),
			payment_method=customer_data.get('payment_method', 'CHECK'),
			currency_code=customer_data.get('currency_code', 'USD'),
			credit_limit=customer_data.get('credit_limit', 0.00),
			credit_hold=customer_data.get('credit_hold', False),
			tax_exempt=customer_data.get('tax_exempt', False),
			tax_exempt_number=customer_data.get('tax_exempt_number'),
			default_tax_code=customer_data.get('default_tax_code'),
			is_active=customer_data.get('is_active', True),
			sales_rep_id=customer_data.get('sales_rep_id'),
			territory_id=customer_data.get('territory_id'),
			price_level=customer_data.get('price_level', 'STANDARD'),
			notes=customer_data.get('notes')
		)
		
		db.session.add(customer)
		db.session.commit()
		return customer
	
	def get_customer(self, customer_id: str) -> Optional[CFARCustomer]:
		"""Get customer by ID"""
		return CFARCustomer.query.filter_by(
			tenant_id=self.tenant_id,
			customer_id=customer_id
		).first()
	
	def get_customer_by_number(self, customer_number: str) -> Optional[CFARCustomer]:
		"""Get customer by customer number"""
		return CFARCustomer.query.filter_by(
			tenant_id=self.tenant_id,
			customer_number=customer_number
		).first()
	
	def get_customers(self, include_inactive: bool = False) -> List[CFARCustomer]:
		"""Get all customers"""
		query = CFARCustomer.query.filter_by(tenant_id=self.tenant_id)
		
		if not include_inactive:
			query = query.filter_by(is_active=True)
		
		return query.order_by(CFARCustomer.customer_name).all()
	
	def update_customer_balance(self, customer_id: str):
		"""Update customer's current balance"""
		customer = self.get_customer(customer_id)
		if not customer:
			return
		
		# Calculate outstanding balance
		outstanding = customer.get_outstanding_balance()
		customer.current_balance = outstanding
		
		# Calculate YTD sales
		ytd_invoices = CFARInvoice.query.filter(
			and_(
				CFARInvoice.tenant_id == self.tenant_id,
				CFARInvoice.customer_id == customer_id,
				CFARInvoice.status == 'Posted',
				func.extract('year', CFARInvoice.invoice_date) == date.today().year
			)
		).all()
		
		customer.ytd_sales = sum(inv.total_amount for inv in ytd_invoices)
		
		# Update last payment info
		last_payment = CFARPayment.query.filter_by(
			tenant_id=self.tenant_id,
			customer_id=customer_id,
			status='Posted'
		).order_by(desc(CFARPayment.payment_date)).first()
		
		if last_payment:
			customer.last_payment_date = last_payment.payment_date
			customer.last_payment_amount = last_payment.total_amount
		
		db.session.commit()
	
	def place_customer_on_hold(self, customer_id: str, reason: str, user_id: str):
		"""Place customer on credit hold"""
		customer = self.get_customer(customer_id)
		if customer:
			customer.credit_hold = True
			customer.notes = (customer.notes or '') + f"\nCredit hold placed by {user_id}: {reason}"
			db.session.commit()
	
	def release_customer_hold(self, customer_id: str, user_id: str):
		"""Release customer from credit hold"""
		customer = self.get_customer(customer_id)
		if customer:
			customer.credit_hold = False
			customer.notes = (customer.notes or '') + f"\nCredit hold released by {user_id}"
			db.session.commit()
	
	# Invoice Management
	
	def create_invoice(self, invoice_data: Dict[str, Any]) -> CFARInvoice:
		"""Create a new customer invoice"""
		# Generate invoice number if not provided
		if 'invoice_number' not in invoice_data:
			invoice_data['invoice_number'] = self._generate_invoice_number()
		
		# Calculate due date if not provided
		if 'due_date' not in invoice_data and 'customer_id' in invoice_data:
			customer = self.get_customer(invoice_data['customer_id'])
			if customer and customer.payment_terms_code:
				due_date = self._calculate_due_date(
					invoice_data.get('invoice_date', date.today()),
					customer.payment_terms_code
				)
				invoice_data['due_date'] = due_date
		
		invoice = CFARInvoice(
			tenant_id=self.tenant_id,
			invoice_number=invoice_data['invoice_number'],
			customer_id=invoice_data['customer_id'],
			invoice_date=invoice_data.get('invoice_date', date.today()),
			due_date=invoice_data['due_date'],
			description=invoice_data.get('description'),
			sales_order_id=invoice_data.get('sales_order_id'),
			customer_po_number=invoice_data.get('customer_po_number'),
			payment_terms_code=invoice_data.get('payment_terms_code'),
			currency_code=invoice_data.get('currency_code', 'USD'),
			sales_rep_id=invoice_data.get('sales_rep_id'),
			territory_id=invoice_data.get('territory_id'),
			freight_amount=invoice_data.get('freight_amount', 0.00),
			misc_amount=invoice_data.get('misc_amount', 0.00),
			discount_amount=invoice_data.get('discount_amount', 0.00),
			notes=invoice_data.get('notes')
		)
		
		db.session.add(invoice)
		db.session.flush()  # Get invoice ID
		
		# Add invoice lines if provided
		if 'lines' in invoice_data:
			for line_number, line_data in enumerate(invoice_data['lines'], 1):
				self._add_invoice_line(invoice.invoice_id, line_number, line_data)
		
		# Calculate totals
		invoice.calculate_totals()
		db.session.commit()
		
		return invoice
	
	def _add_invoice_line(self, invoice_id: str, line_number: int, line_data: Dict[str, Any]):
		"""Add a line to an invoice"""
		line = CFARInvoiceLine(
			tenant_id=self.tenant_id,
			invoice_id=invoice_id,
			line_number=line_number,
			description=line_data.get('description'),
			item_code=line_data.get('item_code'),
			item_description=line_data.get('item_description'),
			item_type=line_data.get('item_type', 'PRODUCT'),
			quantity=line_data.get('quantity', 1.0000),
			unit_price=line_data.get('unit_price', 0.0000),
			line_amount=line_data.get('line_amount', 0.00),
			gl_account_id=line_data['gl_account_id'],
			tax_code=line_data.get('tax_code'),
			tax_rate=line_data.get('tax_rate', 0.00),
			discount_percentage=line_data.get('discount_percentage', 0.00),
			discount_amount=line_data.get('discount_amount', 0.00),
			cost_center=line_data.get('cost_center'),
			department=line_data.get('department'),
			project=line_data.get('project')
		)
		
		# Calculate line amount if not provided
		if line.line_amount == 0:
			line.line_amount = line.quantity * line.unit_price
			if line.discount_percentage > 0:
				line.discount_amount = line.line_amount * (line.discount_percentage / 100)
				line.line_amount -= line.discount_amount
		
		# Calculate tax
		line.calculate_tax()
		
		db.session.add(line)
	
	def get_invoice(self, invoice_id: str) -> Optional[CFARInvoice]:
		"""Get invoice by ID"""
		return CFARInvoice.query.filter_by(
			tenant_id=self.tenant_id,
			invoice_id=invoice_id
		).first()
	
	def get_invoices_by_customer(self, customer_id: str, include_paid: bool = False) -> List[CFARInvoice]:
		"""Get invoices for a customer"""
		query = CFARInvoice.query.filter_by(
			tenant_id=self.tenant_id,
			customer_id=customer_id
		)
		
		if not include_paid:
			query = query.filter(CFARInvoice.payment_status.in_(['Unpaid', 'Partial']))
		
		return query.order_by(desc(CFARInvoice.invoice_date)).all()
	
	def post_invoice(self, invoice_id: str, user_id: str) -> bool:
		"""Post invoice to GL"""
		invoice = self.get_invoice(invoice_id)
		if not invoice or not invoice.can_post():
			return False
		
		# Create GL journal entry
		self._create_invoice_gl_entry(invoice, user_id)
		
		# Update invoice status
		invoice.post_invoice(user_id)
		
		# Update customer balance
		self.update_customer_balance(invoice.customer_id)
		
		db.session.commit()
		return True
	
	def _create_invoice_gl_entry(self, invoice: CFARInvoice, user_id: str):
		"""Create GL journal entry for invoice posting"""
		from ..general_ledger.service import GeneralLedgerService
		
		gl_service = GeneralLedgerService(self.tenant_id)
		
		# Prepare journal entry data
		journal_data = {
			'description': f'Customer Invoice {invoice.invoice_number}',
			'reference': invoice.invoice_number,
			'entry_date': invoice.invoice_date,
			'source': 'AR_INVOICE',
			'lines': []
		}
		
		# AR debit (increase receivables)
		journal_data['lines'].append({
			'account_id': self._get_ar_account_id(invoice.customer_id),
			'debit_amount': invoice.total_amount,
			'credit_amount': 0.00,
			'description': f'Invoice {invoice.invoice_number} - {invoice.customer.customer_name}',
			'reference_type': 'AR_INVOICE',
			'reference_id': invoice.invoice_id
		})
		
		# Revenue credits (by line)
		for line in invoice.lines:
			if line.line_amount > 0:
				journal_data['lines'].append({
					'account_id': line.gl_account_id,
					'debit_amount': 0.00,
					'credit_amount': line.line_amount,
					'description': line.description or f'Invoice line {line.line_number}',
					'cost_center': line.cost_center,
					'department': line.department,
					'project': line.project
				})
		
		# Tax credits
		tax_total = sum(line.tax_amount for line in invoice.lines if line.tax_amount > 0)
		if tax_total > 0:
			journal_data['lines'].append({
				'account_id': self._get_tax_payable_account_id(),
				'debit_amount': 0.00,
				'credit_amount': tax_total,
				'description': f'Sales tax on invoice {invoice.invoice_number}'
			})
		
		# Create and post journal entry
		journal_entry = gl_service.create_journal_entry(journal_data)
		gl_service.post_journal_entry(journal_entry.journal_id, user_id)
		
		invoice.gl_batch_id = journal_entry.journal_id
	
	# Payment Management
	
	def create_payment(self, payment_data: Dict[str, Any]) -> CFARPayment:
		"""Create a new customer payment"""
		# Generate payment number if not provided
		if 'payment_number' not in payment_data:
			payment_data['payment_number'] = self._generate_payment_number()
		
		payment = CFARPayment(
			tenant_id=self.tenant_id,
			payment_number=payment_data['payment_number'],
			customer_id=payment_data['customer_id'],
			payment_date=payment_data.get('payment_date', date.today()),
			payment_method=payment_data['payment_method'],
			payment_amount=payment_data['payment_amount'],
			check_number=payment_data.get('check_number'),
			reference_number=payment_data.get('reference_number'),
			bank_account_id=payment_data.get('bank_account_id'),
			currency_code=payment_data.get('currency_code', 'USD'),
			exchange_rate=payment_data.get('exchange_rate', 1.000000),
			notes=payment_data.get('notes')
		)
		
		db.session.add(payment)
		db.session.flush()  # Get payment ID
		
		# Add payment lines (applications) if provided
		if 'applications' in payment_data:
			for line_number, app_data in enumerate(payment_data['applications'], 1):
				self._add_payment_line(payment.payment_id, line_number, app_data)
		
		# Calculate totals
		payment.calculate_totals()
		db.session.commit()
		
		return payment
	
	def _add_payment_line(self, payment_id: str, line_number: int, app_data: Dict[str, Any]):
		"""Add a payment application line"""
		line = CFARPaymentLine(
			tenant_id=self.tenant_id,
			payment_id=payment_id,
			line_number=line_number,
			invoice_id=app_data.get('invoice_id'),
			credit_memo_id=app_data.get('credit_memo_id'),
			original_amount=app_data.get('original_amount', 0.00),
			payment_amount=app_data['payment_amount'],
			discount_taken=app_data.get('discount_taken', 0.00),
			writeoff_amount=app_data.get('writeoff_amount', 0.00),
			writeoff_reason=app_data.get('writeoff_reason'),
			notes=app_data.get('notes')
		)
		
		# Calculate remaining amount
		line.remaining_amount = line.original_amount - line.payment_amount - line.discount_taken - line.writeoff_amount
		
		db.session.add(line)
	
	def post_payment(self, payment_id: str, user_id: str) -> bool:
		"""Post payment to GL and apply to invoices"""
		payment = self.get_payment(payment_id)
		if not payment or not payment.can_post():
			return False
		
		# Create GL journal entry
		self._create_payment_gl_entry(payment, user_id)
		
		# Update payment status
		payment.post_payment(user_id)
		
		# Apply payment to invoices
		self._apply_payment_to_invoices(payment)
		
		# Update customer balance
		self.update_customer_balance(payment.customer_id)
		
		db.session.commit()
		return True
	
	def _create_payment_gl_entry(self, payment: CFARPayment, user_id: str):
		"""Create GL journal entry for payment posting"""
		from ..general_ledger.service import GeneralLedgerService
		
		gl_service = GeneralLedgerService(self.tenant_id)
		
		# Prepare journal entry data
		journal_data = {
			'description': f'Customer Payment {payment.payment_number}',
			'reference': payment.payment_number,
			'entry_date': payment.payment_date,
			'source': 'AR_PAYMENT',
			'lines': []
		}
		
		# Cash debit (increase cash)
		cash_account_id = payment.bank_account_id or self._get_cash_account_id()
		journal_data['lines'].append({
			'account_id': cash_account_id,
			'debit_amount': payment.payment_amount,
			'credit_amount': 0.00,
			'description': f'Payment {payment.payment_number} - {payment.customer.customer_name}',
			'reference_type': 'AR_PAYMENT',
			'reference_id': payment.payment_id
		})
		
		# AR credit (decrease receivables)
		applied_amount = sum(line.payment_amount for line in payment.payment_lines)
		if applied_amount > 0:
			journal_data['lines'].append({
				'account_id': self._get_ar_account_id(payment.customer_id),
				'debit_amount': 0.00,
				'credit_amount': applied_amount,
				'description': f'Payment application - {payment.customer.customer_name}'
			})
		
		# Discount account for early payment discounts
		discount_total = sum(line.discount_taken for line in payment.payment_lines)
		if discount_total > 0:
			journal_data['lines'].append({
				'account_id': self._get_sales_discount_account_id(),
				'debit_amount': discount_total,
				'credit_amount': 0.00,
				'description': f'Early payment discount - {payment.customer.customer_name}'
			})
		
		# Write-off account for bad debt write-offs
		writeoff_total = sum(line.writeoff_amount for line in payment.payment_lines)
		if writeoff_total > 0:
			journal_data['lines'].append({
				'account_id': self._get_bad_debt_account_id(),
				'debit_amount': writeoff_total,
				'credit_amount': 0.00,
				'description': f'Bad debt write-off - {payment.customer.customer_name}'
			})
		
		# Unapplied cash (if any)
		if payment.unapplied_amount > 0:
			journal_data['lines'].append({
				'account_id': self._get_unapplied_cash_account_id(),
				'debit_amount': 0.00,
				'credit_amount': payment.unapplied_amount,
				'description': f'Unapplied payment - {payment.customer.customer_name}'
			})
		
		# Create and post journal entry
		journal_entry = gl_service.create_journal_entry(journal_data)
		gl_service.post_journal_entry(journal_entry.journal_id, user_id)
		
		payment.gl_batch_id = journal_entry.journal_id
	
	def _apply_payment_to_invoices(self, payment: CFARPayment):
		"""Apply payment lines to invoices"""
		for line in payment.payment_lines:
			if line.invoice_id:
				invoice = self.get_invoice(line.invoice_id)
				if invoice:
					invoice.paid_amount += line.payment_amount + line.discount_taken + line.writeoff_amount
					invoice.outstanding_amount = max(0, invoice.total_amount - invoice.paid_amount)
					
					# Update payment status
					if invoice.outstanding_amount == 0:
						invoice.payment_status = 'Paid'
					elif invoice.paid_amount > 0:
						invoice.payment_status = 'Partial'
	
	def get_payment(self, payment_id: str) -> Optional[CFARPayment]:
		"""Get payment by ID"""
		return CFARPayment.query.filter_by(
			tenant_id=self.tenant_id,
			payment_id=payment_id
		).first()
	
	def auto_apply_payment(self, payment_id: str) -> bool:
		"""Automatically apply payment to oldest invoices"""
		payment = self.get_payment(payment_id)
		if not payment or payment.status != 'Draft':
			return False
		
		# Get outstanding invoices for customer, ordered by due date
		invoices = CFARInvoice.query.filter_by(
			tenant_id=self.tenant_id,
			customer_id=payment.customer_id,
			status='Posted'
		).filter(CFARInvoice.outstanding_amount > 0).order_by(CFARInvoice.due_date).all()
		
		remaining_payment = payment.payment_amount
		line_number = 1
		
		for invoice in invoices:
			if remaining_payment <= 0:
				break
			
			# Calculate payment amount for this invoice
			payment_amount = min(remaining_payment, invoice.outstanding_amount)
			
			# Create payment line
			self._add_payment_line(payment.payment_id, line_number, {
				'invoice_id': invoice.invoice_id,
				'original_amount': invoice.total_amount,
				'payment_amount': payment_amount
			})
			
			remaining_payment -= payment_amount
			line_number += 1
		
		# Recalculate payment totals
		payment.calculate_totals()
		db.session.commit()
		
		return True
	
	# Credit Memo Management
	
	def create_credit_memo(self, credit_memo_data: Dict[str, Any]) -> CFARCreditMemo:
		"""Create a new credit memo"""
		# Generate credit memo number if not provided
		if 'credit_memo_number' not in credit_memo_data:
			credit_memo_data['credit_memo_number'] = self._generate_credit_memo_number()
		
		credit_memo = CFARCreditMemo(
			tenant_id=self.tenant_id,
			credit_memo_number=credit_memo_data['credit_memo_number'],
			customer_id=credit_memo_data['customer_id'],
			credit_date=credit_memo_data.get('credit_date', date.today()),
			reference_invoice_id=credit_memo_data.get('reference_invoice_id'),
			reason_code=credit_memo_data.get('reason_code'),
			description=credit_memo_data.get('description'),
			currency_code=credit_memo_data.get('currency_code', 'USD'),
			return_authorization=credit_memo_data.get('return_authorization'),
			notes=credit_memo_data.get('notes')
		)
		
		db.session.add(credit_memo)
		db.session.flush()  # Get credit memo ID
		
		# Add credit memo lines if provided
		if 'lines' in credit_memo_data:
			for line_number, line_data in enumerate(credit_memo_data['lines'], 1):
				self._add_credit_memo_line(credit_memo.credit_memo_id, line_number, line_data)
		
		# Calculate totals
		credit_memo.calculate_totals()
		db.session.commit()
		
		return credit_memo
	
	def _add_credit_memo_line(self, credit_memo_id: str, line_number: int, line_data: Dict[str, Any]):
		"""Add a line to a credit memo"""
		line = CFARCreditMemoLine(
			tenant_id=self.tenant_id,
			credit_memo_id=credit_memo_id,
			line_number=line_number,
			description=line_data.get('description'),
			item_code=line_data.get('item_code'),
			item_description=line_data.get('item_description'),
			original_invoice_line_id=line_data.get('original_invoice_line_id'),
			quantity=line_data.get('quantity', 1.0000),
			unit_price=line_data.get('unit_price', 0.0000),
			line_amount=line_data.get('line_amount', 0.00),
			gl_account_id=line_data['gl_account_id'],
			tax_code=line_data.get('tax_code'),
			tax_rate=line_data.get('tax_rate', 0.00),
			cost_center=line_data.get('cost_center'),
			department=line_data.get('department'),
			project=line_data.get('project')
		)
		
		# Calculate line amount if not provided
		if line.line_amount == 0:
			line.line_amount = line.quantity * line.unit_price
		
		# Calculate tax
		line.calculate_tax()
		
		db.session.add(line)
	
	def post_credit_memo(self, credit_memo_id: str, user_id: str) -> bool:
		"""Post credit memo to GL"""
		credit_memo = self.get_credit_memo(credit_memo_id)
		if not credit_memo or not credit_memo.can_post():
			return False
		
		# Create GL journal entry
		self._create_credit_memo_gl_entry(credit_memo, user_id)
		
		# Update credit memo status
		credit_memo.post_credit_memo(user_id)
		
		# Update customer balance
		self.update_customer_balance(credit_memo.customer_id)
		
		db.session.commit()
		return True
	
	def _create_credit_memo_gl_entry(self, credit_memo: CFARCreditMemo, user_id: str):
		"""Create GL journal entry for credit memo posting"""
		from ..general_ledger.service import GeneralLedgerService
		
		gl_service = GeneralLedgerService(self.tenant_id)
		
		# Prepare journal entry data
		journal_data = {
			'description': f'Customer Credit Memo {credit_memo.credit_memo_number}',
			'reference': credit_memo.credit_memo_number,
			'entry_date': credit_memo.credit_date,
			'source': 'AR_CREDIT_MEMO',
			'lines': []
		}
		
		# AR credit (decrease receivables)
		journal_data['lines'].append({
			'account_id': self._get_ar_account_id(credit_memo.customer_id),
			'debit_amount': 0.00,
			'credit_amount': credit_memo.total_amount,
			'description': f'Credit Memo {credit_memo.credit_memo_number} - {credit_memo.customer.customer_name}',
			'reference_type': 'AR_CREDIT_MEMO',
			'reference_id': credit_memo.credit_memo_id
		})
		
		# Revenue debits (by line) - reverse revenue
		for line in credit_memo.lines:
			if line.line_amount > 0:
				journal_data['lines'].append({
					'account_id': line.gl_account_id,
					'debit_amount': line.line_amount,
					'credit_amount': 0.00,
					'description': line.description or f'Credit memo line {line.line_number}',
					'cost_center': line.cost_center,
					'department': line.department,
					'project': line.project
				})
		
		# Tax debits - reverse tax
		tax_total = sum(line.tax_amount for line in credit_memo.lines if line.tax_amount > 0)
		if tax_total > 0:
			journal_data['lines'].append({
				'account_id': self._get_tax_payable_account_id(),
				'debit_amount': tax_total,
				'credit_amount': 0.00,
				'description': f'Sales tax reversal on credit memo {credit_memo.credit_memo_number}'
			})
		
		# Create and post journal entry
		journal_entry = gl_service.create_journal_entry(journal_data)
		gl_service.post_journal_entry(journal_entry.journal_id, user_id)
		
		credit_memo.gl_batch_id = journal_entry.journal_id
	
	def get_credit_memo(self, credit_memo_id: str) -> Optional[CFARCreditMemo]:
		"""Get credit memo by ID"""
		return CFARCreditMemo.query.filter_by(
			tenant_id=self.tenant_id,
			credit_memo_id=credit_memo_id
		).first()
	
	# Collections Management
	
	def create_collection_activity(self, collection_data: Dict[str, Any]) -> CFARCollection:
		"""Create a collection activity record"""
		collection = CFARCollection(
			tenant_id=self.tenant_id,
			customer_id=collection_data['customer_id'],
			collection_date=collection_data.get('collection_date', date.today()),
			collection_type=collection_data['collection_type'],
			collector_id=collection_data['collector_id'],
			dunning_level=collection_data.get('dunning_level', 1),
			days_past_due=collection_data.get('days_past_due', 0),
			amount_past_due=collection_data.get('amount_past_due', 0.00),
			subject=collection_data['subject'],
			notes=collection_data.get('notes'),
			outcome=collection_data.get('outcome'),
			follow_up_date=collection_data.get('follow_up_date'),
			follow_up_required=collection_data.get('follow_up_required', False),
			promised_amount=collection_data.get('promised_amount', 0.00),
			promised_date=collection_data.get('promised_date'),
			document_path=collection_data.get('document_path')
		)
		
		# Set related invoice IDs if provided
		if 'related_invoice_ids' in collection_data:
			collection.set_related_invoices(collection_data['related_invoice_ids'])
		
		db.session.add(collection)
		db.session.commit()
		
		return collection
	
	def get_customers_for_collections(self, days_past_due: int = 30) -> List[CFARCustomer]:
		"""Get customers with past due balances for collections"""
		cutoff_date = date.today() - timedelta(days=days_past_due)
		
		# Get customers with past due invoices
		customers_with_past_due = db.session.query(CFARCustomer).join(CFARInvoice).filter(
			and_(
				CFARCustomer.tenant_id == self.tenant_id,
				CFARCustomer.is_active == True,
				CFARCustomer.send_dunning_letters == True,
				CFARInvoice.status == 'Posted',
				CFARInvoice.due_date < cutoff_date,
				CFARInvoice.outstanding_amount > 0
			)
		).distinct().all()
		
		return customers_with_past_due
	
	def generate_dunning_letters(self, customer_ids: List[str], user_id: str) -> List[CFARCollection]:
		"""Generate dunning letters for customers"""
		dunning_activities = []
		
		for customer_id in customer_ids:
			customer = self.get_customer(customer_id)
			if not customer:
				continue
			
			# Calculate current dunning level
			dunning_level = self._calculate_dunning_level(customer)
			
			# Get past due amount
			past_due_amount = self._calculate_past_due_amount(customer)
			days_past_due = self._calculate_days_past_due(customer)
			
			if past_due_amount > 0:
				collection_data = {
					'customer_id': customer_id,
					'collection_type': 'LETTER',
					'collector_id': user_id,
					'dunning_level': dunning_level,
					'days_past_due': days_past_due,
					'amount_past_due': past_due_amount,
					'subject': f'Dunning Letter Level {dunning_level}',
					'notes': f'Automated dunning letter generation for {past_due_amount} past due',
					'follow_up_required': True,
					'follow_up_date': date.today() + timedelta(days=7)
				}
				
				collection = self.create_collection_activity(collection_data)
				dunning_activities.append(collection)
		
		return dunning_activities
	
	def _calculate_dunning_level(self, customer: CFARCustomer) -> int:
		"""Calculate appropriate dunning level for customer"""
		# Get last dunning activity
		last_dunning = CFARCollection.query.filter_by(
			tenant_id=self.tenant_id,
			customer_id=customer.customer_id,
			collection_type='LETTER'
		).order_by(desc(CFARCollection.collection_date)).first()
		
		if not last_dunning:
			return 1
		
		# Increment level, max 5
		return min(last_dunning.dunning_level + 1, 5)
	
	def _calculate_past_due_amount(self, customer: CFARCustomer) -> Decimal:
		"""Calculate total past due amount for customer"""
		past_due_invoices = CFARInvoice.query.filter(
			and_(
				CFARInvoice.tenant_id == self.tenant_id,
				CFARInvoice.customer_id == customer.customer_id,
				CFARInvoice.status == 'Posted',
				CFARInvoice.due_date < date.today(),
				CFARInvoice.outstanding_amount > 0
			)
		).all()
		
		return sum(inv.outstanding_amount for inv in past_due_invoices)
	
	def _calculate_days_past_due(self, customer: CFARCustomer) -> int:
		"""Calculate average days past due for customer"""
		past_due_invoices = CFARInvoice.query.filter(
			and_(
				CFARInvoice.tenant_id == self.tenant_id,
				CFARInvoice.customer_id == customer.customer_id,
				CFARInvoice.status == 'Posted',
				CFARInvoice.due_date < date.today(),
				CFARInvoice.outstanding_amount > 0
			)
		).all()
		
		if not past_due_invoices:
			return 0
		
		total_days = sum(inv.days_past_due() for inv in past_due_invoices)
		return total_days // len(past_due_invoices)
	
	# Statement Management
	
	def generate_statement(self, customer_id: str, statement_date: date, user_id: str) -> CFARStatement:
		"""Generate customer statement"""
		customer = self.get_customer(customer_id)
		if not customer:
			raise ValueError("Customer not found")
		
		# Calculate statement period (typically monthly)
		period_end = statement_date
		period_start = date(period_end.year, period_end.month, 1)
		
		statement = CFARStatement(
			tenant_id=self.tenant_id,
			statement_number=self._generate_statement_number(),
			statement_date=statement_date,
			statement_period_start=period_start,
			statement_period_end=period_end,
			customer_id=customer_id,
			statement_type='MONTHLY',
			currency_code=customer.currency_code,
			delivery_method='PRINT',
			email_address=customer.email
		)
		
		# Calculate statement balances
		statement.beginning_balance = self._get_customer_balance_as_of(customer_id, period_start)
		statement.charges = self._get_customer_charges_for_period(customer_id, period_start, period_end)
		statement.payments = self._get_customer_payments_for_period(customer_id, period_start, period_end)
		statement.adjustments = self._get_customer_adjustments_for_period(customer_id, period_start, period_end)
		statement.ending_balance = customer.get_outstanding_balance(statement_date)
		
		# Generate aging
		statement.generate_statement(user_id)
		
		db.session.add(statement)
		db.session.commit()
		
		return statement
	
	def _get_customer_balance_as_of(self, customer_id: str, as_of_date: date) -> Decimal:
		"""Get customer balance as of a specific date"""
		customer = self.get_customer(customer_id)
		return customer.get_outstanding_balance(as_of_date) if customer else Decimal('0.00')
	
	def _get_customer_charges_for_period(self, customer_id: str, start_date: date, end_date: date) -> Decimal:
		"""Get customer charges for a period"""
		invoices = CFARInvoice.query.filter(
			and_(
				CFARInvoice.tenant_id == self.tenant_id,
				CFARInvoice.customer_id == customer_id,
				CFARInvoice.status == 'Posted',
				CFARInvoice.invoice_date >= start_date,
				CFARInvoice.invoice_date <= end_date
			)
		).all()
		
		return sum(inv.total_amount for inv in invoices)
	
	def _get_customer_payments_for_period(self, customer_id: str, start_date: date, end_date: date) -> Decimal:
		"""Get customer payments for a period"""
		payments = CFARPayment.query.filter(
			and_(
				CFARPayment.tenant_id == self.tenant_id,
				CFARPayment.customer_id == customer_id,
				CFARPayment.status == 'Posted',
				CFARPayment.payment_date >= start_date,
				CFARPayment.payment_date <= end_date
			)
		).all()
		
		return sum(pay.total_amount for pay in payments)
	
	def _get_customer_adjustments_for_period(self, customer_id: str, start_date: date, end_date: date) -> Decimal:
		"""Get customer adjustments (credit memos) for a period"""
		credit_memos = CFARCreditMemo.query.filter(
			and_(
				CFARCreditMemo.tenant_id == self.tenant_id,
				CFARCreditMemo.customer_id == customer_id,
				CFARCreditMemo.status == 'Posted',
				CFARCreditMemo.credit_date >= start_date,
				CFARCreditMemo.credit_date <= end_date
			)
		).all()
		
		return sum(cm.total_amount for cm in credit_memos)
	
	# Aging Analysis
	
	def generate_aging_report(self, as_of_date: Optional[date] = None, user_id: str = None) -> List[CFARAging]:
		"""Generate aging report for all customers"""
		if as_of_date is None:
			as_of_date = date.today()
		
		aging_records = []
		customers = self.get_customers(include_inactive=False)
		
		for customer in customers:
			aging = self._calculate_customer_aging(customer.customer_id, as_of_date, user_id)
			if aging.total_outstanding > 0:  # Only include customers with balances
				aging_records.append(aging)
		
		return aging_records
	
	def _calculate_customer_aging(self, customer_id: str, as_of_date: date, user_id: str) -> CFARAging:
		"""Calculate aging for a specific customer"""
		# Check if aging already exists for this date
		existing_aging = CFARAging.query.filter_by(
			tenant_id=self.tenant_id,
			customer_id=customer_id,
			as_of_date=as_of_date
		).first()
		
		if existing_aging:
			return existing_aging
		
		# Create new aging record
		aging = CFARAging(
			tenant_id=self.tenant_id,
			customer_id=customer_id,
			as_of_date=as_of_date,
			generated_by=user_id or 'system'
		)
		
		# Get outstanding invoices
		invoices = CFARInvoice.query.filter(
			and_(
				CFARInvoice.tenant_id == self.tenant_id,
				CFARInvoice.customer_id == customer_id,
				CFARInvoice.status == 'Posted',
				CFARInvoice.outstanding_amount > 0,
				CFARInvoice.invoice_date <= as_of_date
			)
		).all()
		
		# Calculate aging buckets
		for invoice in invoices:
			days_past_due = invoice.days_past_due(as_of_date)
			
			if days_past_due <= 30:
				aging.current_amount += invoice.outstanding_amount
			elif days_past_due <= 60:
				aging.days_31_60 += invoice.outstanding_amount
			elif days_past_due <= 90:
				aging.days_61_90 += invoice.outstanding_amount
			elif days_past_due <= 120:
				aging.days_91_120 += invoice.outstanding_amount
			else:
				aging.over_120_days += invoice.outstanding_amount
		
		aging.total_outstanding = (
			aging.current_amount + aging.days_31_60 + 
			aging.days_61_90 + aging.days_91_120 + aging.over_120_days
		)
		
		# Set collection status
		if aging.over_120_days > 0:
			aging.collection_status = 'Collections'
		elif aging.days_61_90 + aging.days_91_120 > 0:
			aging.collection_status = 'Past_Due'
		else:
			aging.collection_status = 'Current'
		
		db.session.add(aging)
		db.session.commit()
		
		return aging
	
	# Recurring Billing
	
	def process_recurring_billing(self, as_of_date: Optional[date] = None, user_id: str = None) -> List[CFARInvoice]:
		"""Process recurring billing for all active subscriptions"""
		if as_of_date is None:
			as_of_date = date.today()
		
		# Get recurring billings ready for processing
		recurring_billings = CFARRecurringBilling.query.filter(
			and_(
				CFARRecurringBilling.tenant_id == self.tenant_id,
				CFARRecurringBilling.is_active == True,
				CFARRecurringBilling.is_paused == False,
				CFARRecurringBilling.next_billing_date <= as_of_date
			)
		).all()
		
		generated_invoices = []
		
		for recurring in recurring_billings:
			if recurring.is_ready_for_billing(as_of_date):
				invoice = self._generate_recurring_invoice(recurring, user_id)
				if invoice:
					generated_invoices.append(invoice)
		
		return generated_invoices
	
	def _generate_recurring_invoice(self, recurring: CFARRecurringBilling, user_id: str) -> Optional[CFARInvoice]:
		"""Generate invoice from recurring billing setup"""
		try:
			# Create invoice data
			invoice_data = {
				'customer_id': recurring.customer_id,
				'invoice_date': recurring.next_billing_date,
				'description': recurring.invoice_description_template or recurring.billing_name,
				'payment_terms_code': recurring.payment_terms_code or recurring.customer.payment_terms_code,
				'lines': [{
					'description': recurring.billing_name,
					'quantity': 1.0000,
					'unit_price': float(recurring.billing_amount),
					'line_amount': float(recurring.billing_amount),
					'gl_account_id': recurring.gl_account_id,
					'tax_code': recurring.tax_code
				}]
			}
			
			# Create invoice
			invoice = self.create_invoice(invoice_data)
			
			# Update recurring billing
			recurring.last_processed_date = date.today()
			recurring.invoices_generated += 1
			recurring.calculate_next_billing_date()
			
			# Auto-post if configured
			if recurring.auto_process and user_id:
				self.post_invoice(invoice.invoice_id, user_id)
			
			db.session.commit()
			return invoice
			
		except Exception as e:
			db.session.rollback()
			print(f"Error generating recurring invoice for {recurring.billing_name}: {e}")
			return None
	
	# Helper Methods
	
	def _generate_invoice_number(self) -> str:
		"""Generate next invoice number"""
		last_invoice = CFARInvoice.query.filter_by(tenant_id=self.tenant_id)\
			.order_by(desc(CFARInvoice.invoice_number)).first()
		
		if last_invoice and last_invoice.invoice_number.startswith('INV-'):
			try:
				last_num = int(last_invoice.invoice_number.split('-')[1])
				return f"INV-{last_num + 1:06d}"
			except (ValueError, IndexError):
				pass
		
		return "INV-000001"
	
	def _generate_payment_number(self) -> str:
		"""Generate next payment number"""
		last_payment = CFARPayment.query.filter_by(tenant_id=self.tenant_id)\
			.order_by(desc(CFARPayment.payment_number)).first()
		
		if last_payment and last_payment.payment_number.startswith('PMT-'):
			try:
				last_num = int(last_payment.payment_number.split('-')[1])
				return f"PMT-{last_num + 1:06d}"
			except (ValueError, IndexError):
				pass
		
		return "PMT-000001"
	
	def _generate_credit_memo_number(self) -> str:
		"""Generate next credit memo number"""
		last_cm = CFARCreditMemo.query.filter_by(tenant_id=self.tenant_id)\
			.order_by(desc(CFARCreditMemo.credit_memo_number)).first()
		
		if last_cm and last_cm.credit_memo_number.startswith('CM-'):
			try:
				last_num = int(last_cm.credit_memo_number.split('-')[1])
				return f"CM-{last_num + 1:06d}"
			except (ValueError, IndexError):
				pass
		
		return "CM-000001"
	
	def _generate_statement_number(self) -> str:
		"""Generate next statement number"""
		last_statement = CFARStatement.query.filter_by(tenant_id=self.tenant_id)\
			.order_by(desc(CFARStatement.statement_number)).first()
		
		if last_statement and last_statement.statement_number.startswith('STMT-'):
			try:
				last_num = int(last_statement.statement_number.split('-')[1])
				return f"STMT-{last_num + 1:06d}"
			except (ValueError, IndexError):
				pass
		
		return "STMT-000001"
	
	def _calculate_due_date(self, invoice_date: date, payment_terms: str) -> date:
		"""Calculate due date based on payment terms"""
		if payment_terms == 'NET_10':
			return invoice_date + timedelta(days=10)
		elif payment_terms == 'NET_15':
			return invoice_date + timedelta(days=15)
		elif payment_terms == 'NET_30':
			return invoice_date + timedelta(days=30)
		elif payment_terms == 'NET_60':
			return invoice_date + timedelta(days=60)
		elif payment_terms in ['2_10_NET_30', '1_10_NET_30']:
			return invoice_date + timedelta(days=30)
		elif payment_terms == 'DUE_ON_RECEIPT':
			return invoice_date
		else:
			return invoice_date + timedelta(days=30)  # Default
	
	def _get_ar_account_id(self, customer_id: str) -> str:
		"""Get AR control account ID for customer"""
		# In practice, this might vary by customer or use a default
		account = CFGLAccount.query.filter_by(
			tenant_id=self.tenant_id,
			account_code='1120'  # Accounts Receivable
		).first()
		return account.account_id if account else None
	
	def _get_cash_account_id(self) -> str:
		"""Get default cash account ID"""
		account = CFGLAccount.query.filter_by(
			tenant_id=self.tenant_id,
			account_code='1110'  # Cash
		).first()
		return account.account_id if account else None
	
	def _get_tax_payable_account_id(self) -> str:
		"""Get sales tax payable account ID"""
		account = CFGLAccount.query.filter_by(
			tenant_id=self.tenant_id,
			account_code='2130'  # Sales Tax Payable
		).first()
		return account.account_id if account else None
	
	def _get_sales_discount_account_id(self) -> str:
		"""Get sales discount account ID"""
		account = CFGLAccount.query.filter_by(
			tenant_id=self.tenant_id,
			account_code='4200'  # Sales Discounts
		).first()
		return account.account_id if account else None
	
	def _get_bad_debt_account_id(self) -> str:
		"""Get bad debt expense account ID"""
		account = CFGLAccount.query.filter_by(
			tenant_id=self.tenant_id,
			account_code='5250'  # Bad Debt Expense
		).first()
		return account.account_id if account else None
	
	def _get_unapplied_cash_account_id(self) -> str:
		"""Get unapplied cash account ID"""
		account = CFGLAccount.query.filter_by(
			tenant_id=self.tenant_id,
			account_code='2140'  # Unapplied Cash/Unearned Revenue
		).first()
		return account.account_id if account else None
	
	# Utility Methods
	
	def _log_pretty_amount(self, amount: Decimal) -> str:
		"""Format amount for logging"""
		return f"${amount:,.2f}"
	
	def _log_pretty_date(self, date_value: date) -> str:
		"""Format date for logging"""
		return date_value.strftime('%Y-%m-%d')
	
	def _log_pretty_customer(self, customer: CFARCustomer) -> str:
		"""Format customer for logging"""
		return f"{customer.customer_number} - {customer.customer_name}"