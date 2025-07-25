"""
Financial Management Views

Flask-AppBuilder views for comprehensive financial management and accounting
with CRUD operations, reporting, approval workflows, and multi-currency support.
"""

from flask import request, jsonify, flash, redirect, url_for, render_template
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from flask_appbuilder.widgets import FormWidget, ListWidget, SearchWidget
from flask_appbuilder.forms import DynamicForm
from wtforms import StringField, TextAreaField, SelectField, BooleanField, DecimalField, DateTimeField, IntegerField, validators
from wtforms.validators import DataRequired, Length, Optional, NumberRange
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal
import json

from .models import (
	FMChartOfAccounts, FMJournalEntry, FMJournalEntryLine, FMBudget,
	FMInvoice, FMInvoiceLine, FMPayment, FMFinancialReport, FMDocument
)


class FinancialManagementBaseView(BaseView):
	"""Base view for financial management functionality"""
	
	def __init__(self):
		super().__init__()
		self.default_view = 'dashboard'
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID from security context"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _format_currency(self, amount: Decimal, currency: str = 'USD') -> str:
		"""Format currency amount for display"""
		if amount is None:
			return f"0.00 {currency}"
		return f"{amount:,.2f} {currency}"
	
	def _validate_accounting_period(self, date: datetime) -> bool:
		"""Validate if accounting period is open for transactions"""
		# Implementation would check if the period is closed
		return True


class FMChartOfAccountsModelView(ModelView):
	"""Chart of Accounts management view"""
	
	datamodel = SQLAInterface(FMChartOfAccounts)
	
	# List view configuration
	list_columns = [
		'account_code', 'account_name', 'account_type', 'account_subtype',
		'parent_account', 'current_balance', 'is_active'
	]
	show_columns = [
		'account_code', 'account_name', 'description', 'account_type', 'account_subtype',
		'normal_balance', 'parent_account', 'level', 'base_currency', 'is_active',
		'is_system_account', 'current_balance', 'opening_balance', 'last_transaction_date'
	]
	edit_columns = [
		'account_code', 'account_name', 'description', 'account_type', 'account_subtype',
		'normal_balance', 'parent_account', 'base_currency', 'is_active',
		'allow_manual_entries', 'tax_code', 'cost_center', 'department'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['account_code', 'account_name', 'description']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('account_code', 'asc')
	
	# Form validation
	validators_columns = {
		'account_code': [DataRequired(), Length(min=1, max=50)],
		'account_name': [DataRequired(), Length(min=1, max=200)],
		'account_type': [DataRequired()]
	}
	
	# Custom labels
	label_columns = {
		'account_code': 'Account Code',
		'account_name': 'Account Name',
		'account_type': 'Account Type',
		'account_subtype': 'Subtype',
		'normal_balance': 'Normal Balance',
		'parent_account': 'Parent Account',
		'current_balance': 'Current Balance',
		'is_active': 'Active'
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new account"""
		item.tenant_id = self._get_tenant_id()
		if item.parent_account:
			item.level = item.parent_account.level + 1
			item.path = f"{item.parent_account.path}/{item.account_code}" if item.parent_account.path else item.account_code
		else:
			item.level = 0
			item.path = item.account_code
	
	def post_add(self, item):
		"""Post-process after adding new account"""
		flash(f'Account {item.account_code} - {item.account_name} created successfully', 'success')
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		# Implementation would get tenant from current user context
		return "default_tenant"


class FMJournalEntryModelView(ModelView):
	"""Journal Entry management view"""
	
	datamodel = SQLAInterface(FMJournalEntry)
	
	# List view configuration
	list_columns = [
		'entry_number', 'entry_date', 'description', 'transaction_type',
		'total_debits', 'total_credits', 'status'
	]
	show_columns = [
		'entry_number', 'entry_date', 'posting_date', 'description', 'memo',
		'transaction_type', 'reference_number', 'total_debits', 'total_credits',
		'status', 'fiscal_period', 'fiscal_year', 'is_reversed', 'journal_lines'
	]
	edit_columns = [
		'entry_date', 'description', 'memo', 'transaction_type',
		'reference_number', 'cost_center', 'department', 'tags'
	]
	add_columns = edit_columns
	
	# Related views
	related_views = [FMChartOfAccountsModelView]
	
	# Search and filtering
	search_columns = ['entry_number', 'description', 'reference_number']
	base_filters = [['status', lambda: 'draft', lambda: True]]
	
	# Ordering
	base_order = ('entry_date', 'desc')
	
	# Form validation
	validators_columns = {
		'description': [DataRequired(), Length(min=1, max=500)],
		'entry_date': [DataRequired()],
		'transaction_type': [DataRequired()]
	}
	
	# Custom labels
	label_columns = {
		'entry_number': 'Entry Number',
		'entry_date': 'Entry Date',
		'posting_date': 'Posting Date',
		'transaction_type': 'Transaction Type',
		'total_debits': 'Total Debits',
		'total_credits': 'Total Credits',
		'fiscal_period': 'Fiscal Period',
		'is_reversed': 'Reversed'
	}
	
	@expose('/post/<int:pk>')
	@has_access
	def post_entry(self, pk):
		"""Post journal entry to general ledger"""
		entry = self.datamodel.get(pk)
		if not entry:
			flash('Journal entry not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if entry.post_entry(self._get_current_user_id()):
				flash(f'Journal entry {entry.entry_number} posted successfully', 'success')
			else:
				flash('Cannot post journal entry - check balance and status', 'error')
		except Exception as e:
			flash(f'Error posting journal entry: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/reverse/<int:pk>')
	@has_access
	def reverse_entry(self, pk):
		"""Reverse posted journal entry"""
		entry = self.datamodel.get(pk)
		if not entry:
			flash('Journal entry not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			reason = request.args.get('reason', 'Manual reversal')
			reversal = entry.reverse_entry(reason, self._get_current_user_id())
			self.datamodel.add(reversal)
			flash(f'Journal entry {entry.entry_number} reversed successfully', 'success')
		except Exception as e:
			flash(f'Error reversing journal entry: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new journal entry"""
		item.tenant_id = self._get_tenant_id()
		item.fiscal_period = item.entry_date.strftime('%Y-%m')
		item.fiscal_year = item.entry_date.year
		# Generate entry number
		import uuid
		item.entry_number = f"JE-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class FMBudgetModelView(ModelView):
	"""Budget management view"""
	
	datamodel = SQLAInterface(FMBudget)
	
	# List view configuration
	list_columns = [
		'budget_name', 'account', 'fiscal_year', 'budget_period',
		'budgeted_amount', 'actual_amount', 'variance_percentage', 'status'
	]
	show_columns = [
		'budget_name', 'description', 'budget_type', 'account', 'fiscal_year',
		'budget_period', 'start_date', 'end_date', 'budgeted_amount',
		'actual_amount', 'variance_amount', 'variance_percentage', 'status'
	]
	edit_columns = [
		'budget_name', 'description', 'budget_type', 'account', 'fiscal_year',
		'budget_period', 'start_date', 'end_date', 'budgeted_amount',
		'cost_center', 'department', 'enable_budget_control'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['budget_name', 'description']
	base_filters = [['status', lambda: 'draft', lambda: True]]
	
	# Ordering
	base_order = ('fiscal_year', 'desc')
	
	# Form validation
	validators_columns = {
		'budget_name': [DataRequired(), Length(min=1, max=200)],
		'budgeted_amount': [DataRequired(), NumberRange(min=0)],
		'fiscal_year': [DataRequired()],
		'start_date': [DataRequired()],
		'end_date': [DataRequired()]
	}
	
	# Custom labels
	label_columns = {
		'budget_name': 'Budget Name',
		'budget_type': 'Budget Type',
		'fiscal_year': 'Fiscal Year',
		'budget_period': 'Budget Period',
		'budgeted_amount': 'Budgeted Amount',
		'actual_amount': 'Actual Amount',
		'variance_amount': 'Variance Amount',
		'variance_percentage': 'Variance %'
	}
	
	@expose('/calculate_variance/<int:pk>')
	@has_access
	def calculate_variance(self, pk):
		"""Calculate budget variance"""
		budget = self.datamodel.get(pk)
		if not budget:
			flash('Budget not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			budget.calculate_variance()
			self.datamodel.edit(budget)
			flash(f'Variance calculated for budget {budget.budget_name}', 'success')
		except Exception as e:
			flash(f'Error calculating variance: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new budget"""
		item.tenant_id = self._get_tenant_id()
		item.created_by = self._get_current_user_id()
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class FMInvoiceModelView(ModelView):
	"""Invoice management view"""
	
	datamodel = SQLAInterface(FMInvoice)
	
	# List view configuration
	list_columns = [
		'invoice_number', 'invoice_date', 'customer_name', 'total_amount',
		'amount_due', 'status', 'payment_status', 'due_date'
	]
	show_columns = [
		'invoice_number', 'invoice_date', 'due_date', 'customer_id', 'customer_name',
		'subtotal', 'tax_amount', 'total_amount', 'amount_paid', 'amount_due',
		'status', 'payment_status', 'payment_terms', 'aging_bucket', 'invoice_lines'
	]
	edit_columns = [
		'invoice_date', 'due_date', 'customer_id', 'customer_name',
		'billing_address', 'payment_terms', 'notes'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['invoice_number', 'customer_name', 'purchase_order']
	base_filters = [['status', lambda: 'draft', lambda: True]]
	
	# Ordering
	base_order = ('invoice_date', 'desc')
	
	# Form validation
	validators_columns = {
		'invoice_date': [DataRequired()],
		'due_date': [DataRequired()],
		'customer_name': [DataRequired(), Length(min=1, max=200)],
		'total_amount': [DataRequired(), NumberRange(min=0)]
	}
	
	# Custom labels
	label_columns = {
		'invoice_number': 'Invoice Number',
		'invoice_date': 'Invoice Date',
		'due_date': 'Due Date',
		'customer_name': 'Customer Name',
		'total_amount': 'Total Amount',
		'amount_paid': 'Amount Paid',
		'amount_due': 'Amount Due',
		'payment_status': 'Payment Status',
		'aging_bucket': 'Aging Bucket'
	}
	
	@expose('/send/<int:pk>')
	@has_access
	def send_invoice(self, pk):
		"""Send invoice to customer"""
		invoice = self.datamodel.get(pk)
		if not invoice:
			flash('Invoice not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would send invoice via email
			invoice.status = 'sent'
			invoice.sent_date = datetime.utcnow()
			self.datamodel.edit(invoice)
			flash(f'Invoice {invoice.invoice_number} sent successfully', 'success')
		except Exception as e:
			flash(f'Error sending invoice: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/calculate_aging')
	@has_access
	def calculate_aging(self):
		"""Calculate aging for all invoices"""
		try:
			invoices = self.datamodel.get_all()
			for invoice in invoices:
				invoice.calculate_days_outstanding()
			flash('Aging calculated for all invoices', 'success')
		except Exception as e:
			flash(f'Error calculating aging: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new invoice"""
		item.tenant_id = self._get_tenant_id()
		# Generate invoice number
		import uuid
		item.invoice_number = f"INV-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class FMPaymentModelView(ModelView):
	"""Payment management view"""
	
	datamodel = SQLAInterface(FMPayment)
	
	# List view configuration
	list_columns = [
		'payment_number', 'payment_date', 'customer_id', 'invoice',
		'payment_amount', 'payment_method', 'status'
	]
	show_columns = [
		'payment_number', 'payment_date', 'customer_id', 'invoice',
		'payment_amount', 'currency', 'payment_method', 'payment_reference',
		'status', 'is_deposited', 'is_cleared', 'net_amount'
	]
	edit_columns = [
		'payment_date', 'customer_id', 'invoice', 'payment_amount',
		'payment_method', 'payment_reference', 'notes'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['payment_number', 'payment_reference']
	base_filters = [['status', lambda: 'received', lambda: True]]
	
	# Ordering
	base_order = ('payment_date', 'desc')
	
	# Form validation
	validators_columns = {
		'payment_date': [DataRequired()],
		'payment_amount': [DataRequired(), NumberRange(min=0)],
		'payment_method': [DataRequired()]
	}
	
	# Custom labels
	label_columns = {
		'payment_number': 'Payment Number',
		'payment_date': 'Payment Date',
		'payment_amount': 'Payment Amount',
		'payment_method': 'Payment Method',
		'payment_reference': 'Reference',
		'is_deposited': 'Deposited',
		'is_cleared': 'Cleared',
		'net_amount': 'Net Amount'
	}
	
	@expose('/deposit/<int:pk>')
	@has_access
	def deposit_payment(self, pk):
		"""Mark payment as deposited"""
		payment = self.datamodel.get(pk)
		if not payment:
			flash('Payment not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			payment.deposit_payment()
			self.datamodel.edit(payment)
			flash(f'Payment {payment.payment_number} marked as deposited', 'success')
		except Exception as e:
			flash(f'Error depositing payment: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new payment"""
		item.tenant_id = self._get_tenant_id()
		item.calculate_net_amount()
		# Generate payment number
		import uuid
		item.payment_number = f"PAY-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
		
		# Apply payment to invoice if specified
		if item.invoice:
			item.invoice.apply_payment(item.payment_amount)
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class FinancialReportsView(FinancialManagementBaseView):
	"""Financial reports generation and viewing"""
	
	route_base = "/financial_reports"
	default_view = "dashboard"
	
	@expose('/')
	@has_access
	def dashboard(self):
		"""Financial reports dashboard"""
		return render_template('financial_management/reports_dashboard.html',
							   page_title="Financial Reports")
	
	@expose('/balance_sheet/')
	@has_access
	def balance_sheet(self):
		"""Generate balance sheet report"""
		try:
			as_of_date = request.args.get('as_of_date', datetime.now().strftime('%Y-%m-%d'))
			
			# Implementation would generate balance sheet
			report_data = self._generate_balance_sheet(as_of_date)
			
			return render_template('financial_management/balance_sheet.html',
								   report_data=report_data,
								   as_of_date=as_of_date,
								   page_title="Balance Sheet")
		except Exception as e:
			flash(f'Error generating balance sheet: {str(e)}', 'error')
			return redirect(url_for('FinancialReportsView.dashboard'))
	
	@expose('/income_statement/')
	@has_access
	def income_statement(self):
		"""Generate income statement report"""
		try:
			start_date = request.args.get('start_date', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
			end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
			
			# Implementation would generate income statement
			report_data = self._generate_income_statement(start_date, end_date)
			
			return render_template('financial_management/income_statement.html',
								   report_data=report_data,
								   start_date=start_date,
								   end_date=end_date,
								   page_title="Income Statement")
		except Exception as e:
			flash(f'Error generating income statement: {str(e)}', 'error')
			return redirect(url_for('FinancialReportsView.dashboard'))
	
	@expose('/trial_balance/')
	@has_access
	def trial_balance(self):
		"""Generate trial balance report"""
		try:
			as_of_date = request.args.get('as_of_date', datetime.now().strftime('%Y-%m-%d'))
			
			# Implementation would generate trial balance
			report_data = self._generate_trial_balance(as_of_date)
			
			return render_template('financial_management/trial_balance.html',
								   report_data=report_data,
								   as_of_date=as_of_date,
								   page_title="Trial Balance")
		except Exception as e:
			flash(f'Error generating trial balance: {str(e)}', 'error')
			return redirect(url_for('FinancialReportsView.dashboard'))
	
	@expose('/aged_receivables/')
	@has_access
	def aged_receivables(self):
		"""Generate aged receivables report"""
		try:
			as_of_date = request.args.get('as_of_date', datetime.now().strftime('%Y-%m-%d'))
			
			# Implementation would generate aged receivables
			report_data = self._generate_aged_receivables(as_of_date)
			
			return render_template('financial_management/aged_receivables.html',
								   report_data=report_data,
								   as_of_date=as_of_date,
								   page_title="Aged Receivables")
		except Exception as e:
			flash(f'Error generating aged receivables: {str(e)}', 'error')
			return redirect(url_for('FinancialReportsView.dashboard'))
	
	def _generate_balance_sheet(self, as_of_date: str) -> Dict[str, Any]:
		"""Generate balance sheet data"""
		# Implementation would query accounts and calculate balances
		return {
			'assets': {'current': [], 'fixed': [], 'total': 0},
			'liabilities': {'current': [], 'long_term': [], 'total': 0},
			'equity': {'items': [], 'total': 0},
			'as_of_date': as_of_date
		}
	
	def _generate_income_statement(self, start_date: str, end_date: str) -> Dict[str, Any]:
		"""Generate income statement data"""
		# Implementation would query revenue and expense accounts
		return {
			'revenue': {'items': [], 'total': 0},
			'expenses': {'items': [], 'total': 0},
			'net_income': 0,
			'period': f"{start_date} to {end_date}"
		}
	
	def _generate_trial_balance(self, as_of_date: str) -> Dict[str, Any]:
		"""Generate trial balance data"""
		# Implementation would query all accounts with balances
		return {
			'accounts': [],
			'total_debits': 0,
			'total_credits': 0,
			'as_of_date': as_of_date
		}
	
	def _generate_aged_receivables(self, as_of_date: str) -> Dict[str, Any]:
		"""Generate aged receivables data"""
		# Implementation would query outstanding invoices
		return {
			'customers': [],
			'aging_buckets': ['Current', '1-30', '31-60', '61-90', '90+'],
			'totals': {'current': 0, '1_30': 0, '31_60': 0, '61_90': 0, '90_plus': 0},
			'as_of_date': as_of_date
		}


class FinancialDashboardView(FinancialManagementBaseView):
	"""Main financial management dashboard"""
	
	route_base = "/financial_dashboard"
	default_view = "index"
	
	@expose('/')
	@has_access
	def index(self):
		"""Financial dashboard main page"""
		try:
			# Get key financial metrics
			metrics = self._get_dashboard_metrics()
			
			return render_template('financial_management/dashboard.html',
								   metrics=metrics,
								   page_title="Financial Dashboard")
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return render_template('financial_management/dashboard.html',
								   metrics={},
								   page_title="Financial Dashboard")
	
	@expose('/cash_flow/')
	@has_access
	def cash_flow(self):
		"""Cash flow analysis"""
		try:
			period_days = int(request.args.get('period', 30))
			cash_flow_data = self._get_cash_flow_data(period_days)
			
			return render_template('financial_management/cash_flow.html',
								   cash_flow_data=cash_flow_data,
								   period_days=period_days,
								   page_title="Cash Flow Analysis")
		except Exception as e:
			flash(f'Error loading cash flow data: {str(e)}', 'error')
			return redirect(url_for('FinancialDashboardView.index'))
	
	def _get_dashboard_metrics(self) -> Dict[str, Any]:
		"""Get key financial metrics for dashboard"""
		# Implementation would calculate real metrics from database
		return {
			'total_revenue': {'current': 150000, 'previous': 140000, 'change': 7.1},
			'total_expenses': {'current': 120000, 'previous': 115000, 'change': 4.3},
			'net_income': {'current': 30000, 'previous': 25000, 'change': 20.0},
			'accounts_receivable': {'current': 85000, 'overdue': 12000},
			'accounts_payable': {'current': 45000, 'overdue': 3000},
			'cash_balance': 75000,
			'budget_variance': {'favorable': 5000, 'unfavorable': 2000}
		}
	
	def _get_cash_flow_data(self, period_days: int) -> Dict[str, Any]:
		"""Get cash flow data for specified period"""
		# Implementation would calculate cash flow from transactions
		return {
			'inflows': {'operations': 100000, 'investing': 5000, 'financing': 20000},
			'outflows': {'operations': 80000, 'investing': 15000, 'financing': 10000},
			'net_cash_flow': 20000,
			'period_days': period_days
		}


# Register views with AppBuilder
def register_views(appbuilder):
	"""Register all financial management views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		FMChartOfAccountsModelView,
		"Chart of Accounts",
		icon="fa-list",
		category="Financial Management",
		category_icon="fa-money"
	)
	
	appbuilder.add_view(
		FMJournalEntryModelView,
		"Journal Entries",
		icon="fa-book",
		category="Financial Management"
	)
	
	appbuilder.add_view(
		FMBudgetModelView,
		"Budgets",
		icon="fa-target",
		category="Financial Management"
	)
	
	appbuilder.add_view(
		FMInvoiceModelView,
		"Invoices",
		icon="fa-file-text-o",
		category="Financial Management"
	)
	
	appbuilder.add_view(
		FMPaymentModelView,
		"Payments",
		icon="fa-credit-card",
		category="Financial Management"
	)
	
	# Dashboard and reports views
	appbuilder.add_view_no_menu(FinancialDashboardView)
	appbuilder.add_view_no_menu(FinancialReportsView)
	
	# Menu links
	appbuilder.add_link(
		"Financial Dashboard",
		href="/financial_dashboard/",
		icon="fa-dashboard",
		category="Financial Management"
	)
	
	appbuilder.add_link(
		"Financial Reports",
		href="/financial_reports/",
		icon="fa-bar-chart",
		category="Financial Management"
	)