"""
Cash Management Views

Flask-AppBuilder views for Cash Management functionality including
bank accounts, transactions, reconciliations, and cash forecasting.
"""

from flask import flash, redirect, request, url_for, jsonify, render_template
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface  
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget
from flask_appbuilder.actions import action
from wtforms import Form, StringField, SelectField, DecimalField, DateField, TextAreaField, BooleanField
from wtforms.validators import DataRequired, NumberRange, Optional
from datetime import date, datetime, timedelta
from typing import Dict, List, Any
from decimal import Decimal

from .models import (
	CFCMBankAccount, CFCMBankTransaction, CFCMReconciliation, CFCMReconciliationItem,
	CFCMCashForecast, CFCMCashPosition, CFCMInvestment, CFCMCurrencyRate,
	CFCMCashTransfer, CFCMDeposit, CFCMCheckRegister
)
from .service import CashManagementService
from ...auth_rbac.models import db


class CFCMBankAccountModelView(ModelView):
	"""Bank Account Management View"""
	
	datamodel = SQLAInterface(CFCMBankAccount)
	
	list_title = "Bank Accounts"
	show_title = "Bank Account Details"
	add_title = "Add Bank Account"
	edit_title = "Edit Bank Account"
	
	list_columns = [
		'account_number', 'account_name', 'bank_name', 'account_type',
		'currency_code', 'current_balance', 'is_active', 'is_primary'
	]
	
	show_columns = [
		'account_number', 'account_name', 'account_type', 'bank_name',
		'bank_code', 'routing_number', 'branch_name',
		'currency_code', 'current_balance', 'available_balance', 'ledger_balance',
		'is_active', 'is_primary', 'allow_overdraft', 'overdraft_limit',
		'requires_reconciliation', 'last_reconciliation_date', 'last_statement_date',
		'interest_rate', 'minimum_balance', 'monthly_fee',
		'online_banking_enabled', 'last_import_date',
		'bank_contact_name', 'bank_contact_phone', 'bank_contact_email',
		'notes', 'created_on', 'updated_on'
	]
	
	add_columns = [
		'account_number', 'account_name', 'account_type', 'bank_name',
		'bank_code', 'routing_number', 'branch_code', 'branch_name',
		'bank_address_line1', 'bank_city', 'bank_state', 'bank_country',
		'currency_code', 'is_active', 'is_primary', 'allow_overdraft', 'overdraft_limit',
		'gl_account_id', 'requires_reconciliation', 'auto_reconciliation',
		'reconciliation_tolerance', 'interest_rate', 'minimum_balance', 'monthly_fee',
		'online_banking_enabled', 'bank_contact_name', 'bank_contact_phone', 
		'bank_contact_email', 'notes'
	]
	
	edit_columns = [
		'account_name', 'bank_name', 'branch_name', 'is_active', 'is_primary',
		'allow_overdraft', 'overdraft_limit', 'requires_reconciliation',
		'auto_reconciliation', 'reconciliation_tolerance', 'interest_rate',
		'minimum_balance', 'monthly_fee', 'online_banking_enabled',
		'bank_contact_name', 'bank_contact_phone', 'bank_contact_email', 'notes'
	]
	
	search_columns = ['account_number', 'account_name', 'bank_name']
	order_columns = ['account_number', 'account_name', 'bank_name', 'current_balance']
	base_order = ('account_number', 'asc')
	
	formatters_columns = {
		'current_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'available_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'ledger_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'overdraft_limit': lambda x: f"${x:,.2f}" if x else "$0.00",
		'reconciliation_tolerance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'interest_rate': lambda x: f"{x:.4f}%" if x else "0.0000%",
		'minimum_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'monthly_fee': lambda x: f"${x:,.2f}" if x else "$0.00"
	}
	
	def pre_add(self, item):
		"""Set tenant_id and initialize balances"""
		item.tenant_id = self.get_tenant_id()
		if not item.current_balance:
			item.current_balance = Decimal('0.00')
		if not item.available_balance:
			item.available_balance = item.current_balance
		if not item.ledger_balance:
			item.ledger_balance = item.current_balance
	
	@action("reconcile", "Reconcile", "Start reconciliation for selected accounts", "fa-check-square")
	def reconcile_accounts(self, items):
		"""Start reconciliation for selected accounts"""
		if not items:
			flash("No accounts selected", "warning")
			return redirect(self.get_redirect())
		
		for account in items:
			if not account.requires_reconciliation:
				continue
			
			# Redirect to reconciliation creation
			return redirect(url_for('CFCMReconciliationModelView.add', 
								  bank_account_id=account.bank_account_id))
		
		flash("Reconciliation initiated", "success")
		return redirect(self.get_redirect())
	
	@expose('/balance_summary')
	@has_access
	def balance_summary(self):
		"""Show account balance summary"""
		service = CashManagementService(db.session)
		tenant_id = self.get_tenant_id()
		
		summary = service.get_bank_account_summary(tenant_id)
		
		return self.render_template('cash_management/balance_summary.html',
								   summary=summary,
								   title="Account Balance Summary")


class CFCMBankTransactionModelView(ModelView):
	"""Bank Transaction Management View"""
	
	datamodel = SQLAInterface(CFCMBankTransaction)
	
	list_title = "Bank Transactions"
	show_title = "Transaction Details"
	add_title = "Add Transaction"
	edit_title = "Edit Transaction"
	
	list_columns = [
		'transaction_date', 'bank_account.account_name', 'description',
		'transaction_type', 'amount', 'is_debit', 'status', 'is_reconciled'
	]
	
	show_columns = [
		'transaction_number', 'bank_reference', 'transaction_date', 'value_date',
		'posting_date', 'bank_account', 'description', 'amount', 'is_debit',
		'transaction_type', 'transaction_code', 'status', 'running_balance',
		'is_reconciled', 'reconciled_date', 'source_type', 'check_number',
		'counterparty_name', 'counterparty_account', 'counterparty_bank',
		'gl_posted', 'currency_code', 'exchange_rate', 'imported',
		'import_date', 'memo', 'created_on', 'updated_on'
	]
	
	add_columns = [
		'bank_account', 'transaction_date', 'description', 'amount',
		'is_debit', 'transaction_type', 'value_date', 'check_number',
		'counterparty_name', 'currency_code', 'memo'
	]
	
	edit_columns = [
		'description', 'counterparty_name', 'memo'
	]
	
	search_columns = ['description', 'counterparty_name', 'check_number', 'bank_reference']
	order_columns = ['transaction_date', 'amount', 'transaction_type', 'status']
	base_order = ('transaction_date', 'desc')
	
	formatters_columns = {
		'amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'running_balance': lambda x: f"${x:,.2f}" if x else None,
		'exchange_rate': lambda x: f"{x:.6f}" if x else "1.000000",
		'is_debit': lambda x: "Debit" if x else "Credit"
	}
	
	def pre_add(self, item):
		"""Set tenant_id and update account balance"""
		item.tenant_id = self.get_tenant_id()
		item.status = 'Posted'
		
		# Update bank account balance
		if item.bank_account:
			item.bank_account.update_balance(item.amount, item.transaction_type)
	
	@action("reconcile", "Mark Reconciled", "Mark selected transactions as reconciled", "fa-check")
	def mark_reconciled(self, items):
		"""Mark selected transactions as reconciled"""
		count = 0
		for transaction in items:
			if transaction.can_reconcile():
				transaction.is_reconciled = True
				transaction.reconciled_date = date.today()
				count += 1
		
		if count > 0:
			db.session.commit()
			flash(f"Marked {count} transactions as reconciled", "success")
		else:
			flash("No transactions could be reconciled", "warning")
		
		return redirect(self.get_redirect())
	
	@expose('/import_form')
	@has_access
	def import_form(self):
		"""Show transaction import form"""
		return self.render_template('cash_management/import_transactions.html',
								   title="Import Bank Transactions")


class CFCMReconciliationModelView(ModelView):
	"""Bank Reconciliation Management View"""
	
	datamodel = SQLAInterface(CFCMReconciliation)
	
	list_title = "Bank Reconciliations"
	show_title = "Reconciliation Details"
	add_title = "Create Reconciliation"
	edit_title = "Edit Reconciliation"
	
	list_columns = [
		'reconciliation_number', 'bank_account.account_name', 'statement_date',
		'statement_ending_balance', 'book_balance', 'variance_amount', 'status'
	]
	
	show_columns = [
		'reconciliation_number', 'reconciliation_name', 'bank_account',
		'statement_date', 'statement_number', 'statement_beginning_balance',
		'statement_ending_balance', 'period_start_date', 'period_end_date',
		'status', 'reconciliation_type', 'book_balance', 'adjusted_book_balance',
		'adjusted_bank_balance', 'variance_amount', 'total_deposits', 'total_withdrawals',
		'matched_transactions', 'unmatched_bank_items', 'unmatched_book_items',
		'reconciled_by', 'reconciled_date', 'approved_by', 'approved_date',
		'notes', 'created_on', 'updated_on'
	]
	
	add_columns = [
		'bank_account', 'reconciliation_name', 'statement_date', 'statement_number',
		'statement_beginning_balance', 'statement_ending_balance',
		'period_start_date', 'period_end_date', 'description'
	]
	
	edit_columns = [
		'reconciliation_name', 'statement_number', 'description', 'notes'
	]
	
	search_columns = ['reconciliation_number', 'reconciliation_name', 'statement_number']
	order_columns = ['statement_date', 'reconciliation_number', 'status', 'variance_amount']
	base_order = ('statement_date', 'desc')
	
	formatters_columns = {
		'statement_beginning_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'statement_ending_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'book_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'adjusted_book_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'adjusted_bank_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'variance_amount': lambda x: f"${x:,.2f}" if x else "$0.00"
	}
	
	def pre_add(self, item):
		"""Set tenant_id and generate reconciliation number"""
		item.tenant_id = self.get_tenant_id()
		
		# Generate reconciliation number
		service = CashManagementService(db.session)
		item.reconciliation_number = service._generate_reconciliation_number(item.tenant_id)
		
		# Set initial balances
		if item.bank_account:
			item.book_balance = item.bank_account.ledger_balance
			item.adjusted_book_balance = item.book_balance
			item.adjusted_bank_balance = item.statement_ending_balance
	
	@action("auto_match", "Auto Match", "Perform automatic matching", "fa-magic")
	def auto_match_transactions(self, items):
		"""Perform automatic transaction matching"""
		service = CashManagementService(db.session)
		
		for reconciliation in items:
			if reconciliation.status in ['Draft', 'In Progress']:
				result = service.perform_auto_matching(reconciliation.reconciliation_id)
				flash(f"Auto-matched {result['matched_count']} transactions for {reconciliation.reconciliation_number}", "info")
		
		return redirect(self.get_redirect())
	
	@action("complete", "Complete", "Complete reconciliation", "fa-check-circle")
	def complete_reconciliation(self, items):
		"""Complete selected reconciliations"""
		service = CashManagementService(db.session)
		user_id = self.get_user_id()
		
		completed_count = 0
		for reconciliation in items:
			if service.complete_reconciliation(reconciliation.reconciliation_id, user_id):
				completed_count += 1
		
		if completed_count > 0:
			flash(f"Completed {completed_count} reconciliations", "success")
		else:
			flash("No reconciliations could be completed", "warning")
		
		return redirect(self.get_redirect())
	
	@expose('/reconcile/<reconciliation_id>')
	@has_access
	def reconcile_interface(self, reconciliation_id):
		"""Show reconciliation interface"""
		reconciliation = self.datamodel.get(reconciliation_id)
		if not reconciliation:
			flash("Reconciliation not found", "error")
			return redirect(self.get_redirect())
		
		return self.render_template('cash_management/reconciliation_interface.html',
								   reconciliation=reconciliation,
								   title=f"Reconcile {reconciliation.reconciliation_number}")


class CFCMCashForecastModelView(ModelView):
	"""Cash Forecast Management View"""
	
	datamodel = SQLAInterface(CFCMCashForecast)
	
	list_title = "Cash Forecasts"
	show_title = "Forecast Details"
	add_title = "Add Forecast"
	edit_title = "Edit Forecast"
	
	list_columns = [
		'forecast_date', 'category_name', 'category_type', 'forecast_amount',
		'actual_amount', 'variance_amount', 'confidence_level', 'status'
	]
	
	show_columns = [
		'forecast_name', 'forecast_type', 'forecast_date', 'forecast_horizon',
		'period_start_date', 'period_end_date', 'bank_account', 'category_code',
		'category_name', 'category_type', 'forecast_amount', 'actual_amount',
		'variance_amount', 'variance_percentage', 'confidence_level',
		'forecast_method', 'data_source', 'is_recurring', 'status',
		'last_updated_date', 'notes', 'assumptions'
	]
	
	add_columns = [
		'forecast_name', 'forecast_type', 'forecast_date', 'forecast_horizon',
		'bank_account', 'category_code', 'category_name', 'category_type',
		'forecast_amount', 'confidence_level', 'forecast_method',
		'is_recurring', 'notes', 'assumptions'
	]
	
	edit_columns = [
		'forecast_name', 'forecast_amount', 'confidence_level',
		'status', 'notes', 'assumptions'
	]
	
	search_columns = ['forecast_name', 'category_name', 'category_code']
	order_columns = ['forecast_date', 'category_name', 'forecast_amount', 'variance_amount']
	base_order = ('forecast_date', 'asc')
	
	formatters_columns = {
		'forecast_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'actual_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'variance_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'variance_percentage': lambda x: f"{x:.2f}%" if x else "0.00%",
		'confidence_level': lambda x: f"{x:.1f}%" if x else "0.0%",
		'category_type': lambda x: "Inflow" if x == 'INFLOW' else "Outflow" if x == 'OUTFLOW' else x
	}
	
	def pre_add(self, item):
		"""Set tenant_id and default values"""
		item.tenant_id = self.get_tenant_id()
		if not item.period_start_date:
			item.period_start_date = date.today()
		if not item.period_end_date and item.forecast_horizon:
			item.period_end_date = item.period_start_date + timedelta(days=item.forecast_horizon)
	
	@expose('/generate_forecast')
	@has_access
	def generate_forecast(self):
		"""Show forecast generation form"""
		return self.render_template('cash_management/generate_forecast.html',
								   title="Generate Cash Forecast")
	
	@expose('/forecast_accuracy')
	@has_access
	def forecast_accuracy(self):
		"""Show forecast accuracy analysis"""
		return self.render_template('cash_management/forecast_accuracy.html',
								   title="Forecast Accuracy Analysis")


class CFCMCashPositionModelView(ModelView):
	"""Cash Position Management View"""
	
	datamodel = SQLAInterface(CFCMCashPosition)
	
	list_title = "Cash Positions"
	show_title = "Position Details"
	add_title = "Add Position"
	edit_title = "Edit Position"
	
	list_columns = [
		'position_date', 'bank_account.account_name', 'opening_balance',
		'closing_balance', 'net_change', 'transaction_count', 'is_reconciled'
	]
	
	show_columns = [
		'position_date', 'bank_account', 'opening_balance', 'closing_balance',
		'average_balance', 'minimum_balance', 'maximum_balance',
		'total_inflows', 'total_outflows', 'net_change', 'transaction_count',
		'operating_inflows', 'operating_outflows', 'investing_inflows',
		'investing_outflows', 'financing_inflows', 'financing_outflows',
		'interest_earned', 'fees_charged', 'currency_code', 'exchange_rate',
		'is_reconciled', 'generated_date', 'data_source'
	]
	
	search_columns = ['bank_account.account_name']
	order_columns = ['position_date', 'closing_balance', 'net_change']
	base_order = ('position_date', 'desc')
	
	formatters_columns = {
		'opening_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'closing_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'average_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'minimum_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'maximum_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'total_inflows': lambda x: f"${x:,.2f}" if x else "$0.00",
		'total_outflows': lambda x: f"${x:,.2f}" if x else "$0.00",
		'net_change': lambda x: f"${x:,.2f}" if x else "$0.00",
	}
	
	@expose('/position_summary')
	@has_access
	def position_summary(self):
		"""Show cash position summary"""
		service = CashManagementService(db.session)
		tenant_id = self.get_tenant_id()
		
		summary = service.get_cash_position_summary(tenant_id)
		
		return self.render_template('cash_management/position_summary.html',
								   summary=summary,
								   title="Cash Position Summary")


class CFCMInvestmentModelView(ModelView):
	"""Investment Management View"""
	
	datamodel = SQLAInterface(CFCMInvestment)
	
	list_title = "Investments"
	show_title = "Investment Details"
	add_title = "Add Investment"
	edit_title = "Edit Investment"
	
	list_columns = [
		'investment_number', 'investment_name', 'investment_type',
		'purchase_amount', 'current_value', 'interest_rate', 'maturity_date', 'status'
	]
	
	show_columns = [
		'investment_number', 'investment_name', 'description', 'investment_type',
		'investment_category', 'risk_rating', 'bank_account', 'custodian_name',
		'purchase_date', 'maturity_date', 'purchase_amount', 'current_value',
		'face_value', 'interest_rate', 'yield_to_maturity', 'accrued_interest',
		'status', 'is_liquid', 'liquidation_penalty', 'unrealized_gain_loss',
		'total_interest_earned', 'auto_rollover', 'currency_code', 'notes'
	]
	
	add_columns = [
		'investment_name', 'description', 'investment_type', 'investment_category',
		'risk_rating', 'bank_account', 'custodian_name', 'account_number',
		'purchase_date', 'maturity_date', 'purchase_amount', 'face_value',
		'interest_rate', 'interest_payment_frequency', 'is_liquid',
		'liquidation_penalty', 'auto_rollover', 'currency_code', 'notes'
	]
	
	edit_columns = [
		'investment_name', 'description', 'custodian_name', 'current_value',
		'accrued_interest', 'status', 'auto_rollover', 'notes'
	]
	
	search_columns = ['investment_number', 'investment_name', 'custodian_name']
	order_columns = ['maturity_date', 'current_value', 'interest_rate', 'purchase_date']
	base_order = ('maturity_date', 'asc')
	
	formatters_columns = {
		'purchase_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'current_value': lambda x: f"${x:,.2f}" if x else "$0.00",
		'face_value': lambda x: f"${x:,.2f}" if x else "$0.00",
		'accrued_interest': lambda x: f"${x:,.2f}" if x else "$0.00",
		'unrealized_gain_loss': lambda x: f"${x:,.2f}" if x else "$0.00",
		'total_interest_earned': lambda x: f"${x:,.2f}" if x else "$0.00",
		'interest_rate': lambda x: f"{x:.4f}%" if x else "0.0000%",
		'yield_to_maturity': lambda x: f"{x:.4f}%" if x else None,
		'liquidation_penalty': lambda x: f"{x:.4f}%" if x else "0.0000%"
	}
	
	def pre_add(self, item):
		"""Set tenant_id and generate investment number"""
		item.tenant_id = self.get_tenant_id()
		
		service = CashManagementService(db.session)
		item.investment_number = service._generate_investment_number(item.tenant_id)
		
		# Set initial current value to purchase amount
		item.current_value = item.purchase_amount
	
	@expose('/maturing_investments')
	@has_access
	def maturing_investments(self):
		"""Show investments maturing soon"""
		service = CashManagementService(db.session)
		tenant_id = self.get_tenant_id()
		
		investments = service.get_maturing_investments(tenant_id, 30)
		
		return self.render_template('cash_management/maturing_investments.html',
								   investments=investments,
								   title="Maturing Investments")


class CFCMCashTransferModelView(ModelView):
	"""Cash Transfer Management View"""
	
	datamodel = SQLAInterface(CFCMCashTransfer)
	
	list_title = "Cash Transfers"
	show_title = "Transfer Details"
	add_title = "Create Transfer"
	edit_title = "Edit Transfer"
	
	list_columns = [
		'transfer_number', 'transfer_date', 'from_account.account_name',
		'to_account.account_name', 'transfer_amount', 'transfer_method', 'status'
	]
	
	show_columns = [
		'transfer_number', 'description', 'reference', 'from_account', 'to_account',
		'transfer_date', 'value_date', 'transfer_amount', 'transfer_fee',
		'total_amount', 'transfer_method', 'transfer_type', 'status',
		'approved', 'approved_by', 'approved_date', 'submitted', 'submitted_date',
		'completed', 'completed_date', 'confirmation_number', 'currency_code',
		'tracking_number', 'notes'
	]
	
	add_columns = [
		'description', 'from_account', 'to_account', 'transfer_date',
		'transfer_amount', 'transfer_fee', 'transfer_method', 'transfer_type',
		'reference', 'notes'
	]
	
	edit_columns = [
		'description', 'transfer_date', 'transfer_amount', 'transfer_fee',
		'reference', 'notes'
	]
	
	search_columns = ['transfer_number', 'description', 'reference']
	order_columns = ['transfer_date', 'transfer_amount', 'status']
	base_order = ('transfer_date', 'desc')
	
	formatters_columns = {
		'transfer_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'transfer_fee': lambda x: f"${x:,.2f}" if x else "$0.00",
		'total_amount': lambda x: f"${x:,.2f}" if x else "$0.00"
	}
	
	def pre_add(self, item):
		"""Set tenant_id and generate transfer number"""
		item.tenant_id = self.get_tenant_id()
		
		service = CashManagementService(db.session)
		item.transfer_number = service._generate_transfer_number(item.tenant_id)
		
		# Calculate total amount
		item.calculate_total_amount()
		
		# Set requires approval based on amount
		item.requires_approval = item.total_amount > Decimal('10000.00')
	
	@action("approve", "Approve", "Approve selected transfers", "fa-check")
	def approve_transfers(self, items):
		"""Approve selected transfers"""
		user_id = self.get_user_id()
		approved_count = 0
		
		for transfer in items:
			if transfer.can_approve():
				transfer.approve_transfer(user_id)
				approved_count += 1
		
		if approved_count > 0:
			db.session.commit()
			flash(f"Approved {approved_count} transfers", "success")
		else:
			flash("No transfers could be approved", "warning")
		
		return redirect(self.get_redirect())
	
	@action("submit", "Submit", "Submit approved transfers", "fa-paper-plane")
	def submit_transfers(self, items):
		"""Submit approved transfers"""
		user_id = self.get_user_id()
		submitted_count = 0
		
		for transfer in items:
			if transfer.can_submit():
				transfer.submit_transfer(user_id)
				submitted_count += 1
		
		if submitted_count > 0:
			db.session.commit()
			flash(f"Submitted {submitted_count} transfers", "success")
		else:
			flash("No transfers could be submitted", "warning")
		
		return redirect(self.get_redirect())


class CFCMCheckRegisterModelView(ModelView):
	"""Check Register Management View"""
	
	datamodel = SQLAInterface(CFCMCheckRegister)
	
	list_title = "Check Register"
	show_title = "Check Details"
	add_title = "Issue Check"
	edit_title = "Edit Check"
	
	list_columns = [
		'check_number', 'check_date', 'payee_name', 'check_amount',
		'status', 'is_cleared', 'days_outstanding'
	]
	
	show_columns = [
		'check_number', 'bank_account', 'check_date', 'issue_date',
		'payee_name', 'check_amount', 'status', 'cleared_date',
		'payment_id', 'invoice_number', 'description', 'memo',
		'is_cleared', 'cleared_amount', 'is_voided', 'void_date',
		'void_reason', 'stop_payment', 'stop_payment_date',
		'days_outstanding', 'is_stale_dated', 'notes'
	]
	
	add_columns = [
		'check_number', 'bank_account', 'check_date', 'payee_name',
		'check_amount', 'payment_id', 'invoice_number', 'description', 'memo'
	]
	
	edit_columns = [
		'payee_name', 'description', 'memo', 'notes'
	]
	
	search_columns = ['check_number', 'payee_name', 'invoice_number']
	order_columns = ['check_date', 'check_number', 'check_amount', 'days_outstanding']
	base_order = ('check_date', 'desc')
	
	formatters_columns = {
		'check_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'cleared_amount': lambda x: f"${x:,.2f}" if x else None,
		'stop_payment_fee': lambda x: f"${x:,.2f}" if x else "$0.00"
	}
	
	def pre_add(self, item):
		"""Set tenant_id and update account balance"""
		item.tenant_id = self.get_tenant_id()
		
		# Update bank account balance for issued check
		if item.bank_account:
			item.bank_account.update_balance(item.check_amount, 'CHECK')
	
	@action("void", "Void", "Void selected checks", "fa-times-circle")
	def void_checks(self, items):
		"""Void selected checks"""
		user_id = self.get_user_id()
		voided_count = 0
		
		for check in items:
			if check.can_void():
				check.void_check(user_id, "Voided via bulk action")
				voided_count += 1
		
		if voided_count > 0:
			db.session.commit()
			flash(f"Voided {voided_count} checks", "success")
		else:
			flash("No checks could be voided", "warning")
		
		return redirect(self.get_redirect())
	
	@action("stop_payment", "Stop Payment", "Place stop payment on selected checks", "fa-ban")
	def stop_payment_checks(self, items):
		"""Place stop payment on selected checks"""
		stopped_count = 0
		
		for check in items:
			if check.can_stop_payment():
				check.place_stop_payment("Stop payment via bulk action", Decimal('30.00'))
				stopped_count += 1
		
		if stopped_count > 0:
			db.session.commit()
			flash(f"Placed stop payment on {stopped_count} checks", "success")
		else:
			flash("No checks could have stop payment placed", "warning")
		
		return redirect(self.get_redirect())
	
	@expose('/outstanding_checks')
	@has_access
	def outstanding_checks(self):
		"""Show outstanding checks report"""
		query = db.session.query(CFCMCheckRegister).filter(
			CFCMCheckRegister.tenant_id == self.get_tenant_id(),
			CFCMCheckRegister.is_cleared == False,
			CFCMCheckRegister.is_voided == False
		).order_by(CFCMCheckRegister.check_date)
		
		checks = query.all()
		
		return self.render_template('cash_management/outstanding_checks.html',
								   checks=checks,
								   title="Outstanding Checks")


class CFCMDashboardView(BaseView):
	"""Cash Management Dashboard"""
	
	route_base = "/cash_management"
	default_view = "dashboard"
	
	@expose('/')
	@expose('/dashboard')
	@has_access
	def dashboard(self):
		"""Show cash management dashboard"""
		service = CashManagementService(db.session)
		tenant_id = self.get_tenant_id()
		
		# Get summary data
		account_summary = service.get_bank_account_summary(tenant_id)
		cash_position = service.get_cash_position_summary(tenant_id)
		reconciliation_status = service.get_reconciliation_status(tenant_id)
		maturing_investments = service.get_maturing_investments(tenant_id, 30)
		
		# Get recent transactions
		recent_transactions = []
		for account in account_summary['accounts']:
			transactions = service.get_transaction_history(account['account_id'], limit=5)
			recent_transactions.extend(transactions)
		
		# Sort by date and take latest 20
		recent_transactions.sort(key=lambda x: x['transaction_date'], reverse=True)
		recent_transactions = recent_transactions[:20]
		
		return self.render_template(
			'cash_management/dashboard.html',
			account_summary=account_summary,
			cash_position=cash_position,
			reconciliation_status=reconciliation_status,
			maturing_investments=maturing_investments,
			recent_transactions=recent_transactions,
			title="Cash Management Dashboard"
		)
	
	@expose('/cash_flow_chart')
	@has_access
	def cash_flow_chart(self):
		"""Generate cash flow chart data"""
		service = CashManagementService(db.session)
		tenant_id = self.get_tenant_id()
		
		# Get last 30 days of cash flow
		end_date = date.today()
		start_date = end_date - timedelta(days=30)
		
		report = service.generate_cash_flow_report(tenant_id, start_date, end_date)
		
		return jsonify(report)
	
	@expose('/liquidity_analysis')
	@has_access
	def liquidity_analysis(self):
		"""Show liquidity analysis"""
		return self.render_template('cash_management/liquidity_analysis.html',
								   title="Liquidity Analysis")
	
	def get_tenant_id(self):
		"""Get current tenant ID (implement based on your auth system)"""
		# This would integrate with your actual tenant resolution logic
		return "default_tenant"
	
	def get_user_id(self):
		"""Get current user ID (implement based on your auth system)"""
		# This would integrate with your actual user resolution logic
		return "current_user"