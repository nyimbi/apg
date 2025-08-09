"""
APG Financial Management General Ledger - Flask-AppBuilder Views

Comprehensive UI components for enterprise general ledger operations including:
- Chart of Accounts management with hierarchical visualization
- Journal Entry creation and management with approval workflows
- Financial Reporting dashboards with real-time analytics
- Period management and closing procedures
- Multi-currency transaction handling
- Audit trail and compliance reporting

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from flask import render_template, request, flash, redirect, url_for, jsonify, session
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget, EditWidget
from flask_appbuilder.actions import action
from flask_appbuilder.security.decorators import protect
from flask_babel import lazy_gettext as _
from wtforms import Form, StringField, TextAreaField, SelectField, DecimalField, DateField, HiddenField
from wtforms.validators import DataRequired, Length, NumberRange, Optional
from decimal import Decimal
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
import json
import logging

from .models import (
	GLTenant, GLAccountType, GLAccount, GLAccountGroup, GLAccountBalance,
	GLAccountBudget, GLPeriod, GLCurrencyRate, GLJournalEntry, GLJournalLine,
	GLPosting, AccountTypeEnum, BalanceTypeEnum, PeriodStatusEnum, 
	JournalStatusEnum, JournalSourceEnum, CurrencyEnum, ReportingFrameworkEnum
)
from .service import (
	GeneralLedgerService, AccountCreationRequest, JournalEntryRequest,
	TrialBalanceParams, FinancialReportingResult, GLServiceException
)

# Configure logging
logger = logging.getLogger(__name__)


# =====================================
# CUSTOM WIDGETS AND FORMS
# =====================================

class ChartOfAccountsWidget(ListWidget):
	"""Custom widget for hierarchical chart of accounts display"""
	template = 'general_ledger/widgets/chart_of_accounts.html'


class JournalEntryWidget(EditWidget):
	"""Custom widget for journal entry creation with line items"""
	template = 'general_ledger/widgets/journal_entry.html'


class FinancialReportWidget(ShowWidget):
	"""Custom widget for financial report display"""
	template = 'general_ledger/widgets/financial_report.html'


class AccountCreationForm(Form):
	"""Form for creating new GL accounts"""
	account_code = StringField(
		_('Account Code'),
		validators=[DataRequired(), Length(min=3, max=20)],
		render_kw={'placeholder': 'e.g., 1001, 2001, 3001'}
	)
	account_name = StringField(
		_('Account Name'),
		validators=[DataRequired(), Length(min=3, max=100)],
		render_kw={'placeholder': 'e.g., Cash in Bank, Accounts Payable'}
	)
	description = TextAreaField(
		_('Description'),
		validators=[Optional(), Length(max=500)],
		render_kw={'rows': 3, 'placeholder': 'Optional detailed description'}
	)
	account_type_id = SelectField(
		_('Account Type'),
		validators=[DataRequired()],
		coerce=str
	)
	parent_account_id = SelectField(
		_('Parent Account'),
		validators=[Optional()],
		coerce=str
	)
	currency = SelectField(
		_('Primary Currency'),
		choices=[(c.value, c.value) for c in CurrencyEnum],
		default=CurrencyEnum.USD.value
	)
	is_header = SelectField(
		_('Account Type'),
		choices=[('false', 'Detail Account'), ('true', 'Header Account')],
		default='false'
	)


class JournalEntryForm(Form):
	"""Form for creating journal entries"""
	description = StringField(
		_('Description'),
		validators=[DataRequired(), Length(min=5, max=200)],
		render_kw={'placeholder': 'Describe the transaction purpose'}
	)
	reference = StringField(
		_('Reference'),
		validators=[Optional(), Length(max=50)],
		render_kw={'placeholder': 'External reference number'}
	)
	entry_date = DateField(
		_('Entry Date'),
		validators=[DataRequired()],
		default=datetime.now().date()
	)
	posting_date = DateField(
		_('Posting Date'),
		validators=[DataRequired()],
		default=datetime.now().date()
	)
	source = SelectField(
		_('Source'),
		choices=[(s.value, s.value) for s in JournalSourceEnum],
		default=JournalSourceEnum.MANUAL.value
	)
	requires_approval = SelectField(
		_('Requires Approval'),
		choices=[('false', 'No'), ('true', 'Yes')],
		default='false'
	)


class PeriodManagementForm(Form):
	"""Form for period management operations"""
	period_id = HiddenField()
	action = SelectField(
		_('Action'),
		choices=[
			('open', 'Open Period'),
			('close', 'Close Period'),
			('reopen', 'Reopen Period')
		],
		validators=[DataRequired()]
	)
	force_close = SelectField(
		_('Force Close'),
		choices=[('false', 'No'), ('true', 'Yes')],
		default='false'
	)


# =====================================
# CHART OF ACCOUNTS VIEWS
# =====================================

class GLAccountTypeView(ModelView):
	"""Account Types management view"""
	datamodel = SQLAInterface(GLAccountType)
	
	list_title = _('Account Types')
	show_title = _('Account Type Details')
	add_title = _('Add Account Type')
	edit_title = _('Edit Account Type')
	
	list_columns = ['type_code', 'type_name', 'normal_balance', 'is_balance_sheet', 'is_income_statement']
	show_columns = ['type_code', 'type_name', 'description', 'normal_balance', 
					'is_balance_sheet', 'is_income_statement', 'balance_sheet_section',
					'income_statement_section', 'reporting_sequence']
	edit_columns = ['type_code', 'type_name', 'description', 'normal_balance',
					'is_balance_sheet', 'is_income_statement', 'balance_sheet_section',
					'income_statement_section', 'reporting_sequence']
	add_columns = edit_columns
	
	base_order = ('reporting_sequence', 'asc')
	base_filters = [['tenant_id', lambda: session.get('tenant_id')]]


class GLAccountView(ModelView):
	"""Chart of Accounts management view"""
	datamodel = SQLAInterface(GLAccount)
	
	list_title = _('Chart of Accounts')
	show_title = _('Account Details')
	add_title = _('Add Account')
	edit_title = _('Edit Account')
	
	list_columns = ['account_code', 'account_name', 'account_type.type_name', 
					'current_balance', 'primary_currency', 'is_active', 'is_header']
	show_columns = ['account_code', 'account_name', 'description', 'account_type.type_name',
					'parent_account.account_name', 'current_balance', 'opening_balance',
					'primary_currency', 'is_active', 'is_header', 'allow_posting',
					'level', 'path', 'created_on', 'updated_on']
	edit_columns = ['account_code', 'account_name', 'description', 'account_type',
					'parent_account', 'primary_currency', 'is_active', 'is_header',
					'allow_posting', 'opening_balance']
	add_columns = edit_columns
	
	search_columns = ['account_code', 'account_name', 'description']
	list_widget = ChartOfAccountsWidget
	base_order = ('account_code', 'asc')
	base_filters = [['tenant_id', lambda: session.get('tenant_id')]]
	
	formatters_columns = {
		'current_balance': lambda x: f"{x:,.2f}" if x else "0.00",
		'opening_balance': lambda x: f"{x:,.2f}" if x else "0.00"
	}


class ChartOfAccountsView(BaseView):
	"""Hierarchical Chart of Accounts view"""
	
	@expose('/')
	@has_access
	def index(self):
		"""Display hierarchical chart of accounts"""
		tenant_id = session.get('tenant_id')
		if not tenant_id:
			flash(_('Tenant not selected'), 'error')
			return redirect(url_for('GLAccountView.list'))
		
		try:
			gl_service = GeneralLedgerService(tenant_id, session.get('user_id'))
			accounts = gl_service.get_chart_of_accounts(include_inactive=False)
			
			# Build hierarchy tree
			account_tree = self._build_account_tree(accounts)
			
			return self.render_template(
				'general_ledger/chart_of_accounts.html',
				account_tree=account_tree,
				title=_('Chart of Accounts')
			)
		
		except Exception as e:
			logger.error(f"Error loading chart of accounts: {e}")
			flash(_('Error loading chart of accounts'), 'error')
			return redirect(url_for('GLAccountView.list'))
	
	@expose('/create_account', methods=['GET', 'POST'])
	@has_access
	def create_account(self):
		"""Create new account with AJAX support"""
		tenant_id = session.get('tenant_id')
		if not tenant_id:
			return jsonify({'error': 'Tenant not selected'}), 400
		
		form = AccountCreationForm(request.form)
		
		# Populate form choices
		gl_service = GeneralLedgerService(tenant_id, session.get('user_id'))
		
		try:
			# Get account types
			account_types = gl_service.get_account_types()
			form.account_type_id.choices = [(at.type_id, at.type_name) for at in account_types]
			
			# Get header accounts for parent selection
			header_accounts = gl_service.get_chart_of_accounts()
			header_accounts = [acc for acc in header_accounts if acc.is_header]
			form.parent_account_id.choices = [('', 'None')] + [
				(acc.account_id, f"{acc.account_code} - {acc.account_name}") 
				for acc in header_accounts
			]
			
			if request.method == 'POST' and form.validate():
				# Create account
				request_data = AccountCreationRequest(
					account_code=form.account_code.data,
					account_name=form.account_name.data,
					account_type_id=form.account_type_id.data,
					description=form.description.data,
					parent_account_id=form.parent_account_id.data or None,
					currency=CurrencyEnum(form.currency.data),
					is_header=form.is_header.data == 'true'
				)
				
				account = gl_service.create_account(request_data)
				
				if request.is_json:
					return jsonify({
						'success': True,
						'account_id': account.account_id,
						'message': _('Account created successfully')
					})
				else:
					flash(_('Account created successfully'), 'success')
					return redirect(url_for('ChartOfAccountsView.index'))
			
			if request.is_json and not form.validate():
				return jsonify({'error': 'Form validation failed', 'errors': form.errors}), 400
			
			return self.render_template(
				'general_ledger/create_account.html',
				form=form,
				title=_('Create Account')
			)
		
		except GLServiceException as e:
			error_msg = str(e)
			if request.is_json:
				return jsonify({'error': error_msg}), 400
			else:
				flash(error_msg, 'error')
				return self.render_template('general_ledger/create_account.html', form=form)
	
	def _build_account_tree(self, accounts: List) -> List[Dict]:
		"""Build hierarchical account tree structure"""
		account_dict = {acc.account_id: acc for acc in accounts}
		tree = []
		
		# Find root accounts (no parent)
		for account in accounts:
			if not account.parent_account_id:
				tree.append(self._build_account_node(account, account_dict))
		
		return tree
	
	def _build_account_node(self, account, account_dict: Dict) -> Dict:
		"""Build individual account node with children"""
		node = {
			'account_id': account.account_id,
			'account_code': account.account_code,
			'account_name': account.account_name,
			'account_type': account.account_type.type_name if account.account_type else '',
			'current_balance': float(account.current_balance or 0),
			'currency': account.primary_currency.value if account.primary_currency else 'USD',
			'is_header': account.is_header,
			'is_active': account.is_active,
			'level': account.level or 0,
			'children': []
		}
		
		# Find and add children
		for acc in account_dict.values():
			if acc.parent_account_id == account.account_id:
				node['children'].append(self._build_account_node(acc, account_dict))
		
		return node


# =====================================
# JOURNAL ENTRY VIEWS
# =====================================

class GLJournalEntryView(ModelView):
	"""Journal Entries management view"""
	datamodel = SQLAInterface(GLJournalEntry)
	
	list_title = _('Journal Entries')
	show_title = _('Journal Entry Details')
	add_title = _('Create Journal Entry')
	edit_title = _('Edit Journal Entry')
	
	list_columns = ['journal_number', 'description', 'posting_date', 
					'total_debits', 'total_credits', 'status', 'source']
	show_columns = ['journal_number', 'description', 'reference', 'entry_date',
					'posting_date', 'period.period_name', 'total_debits', 'total_credits',
					'line_count', 'status', 'source', 'requires_approval',
					'created_by', 'created_on', 'posted_by', 'posted_on']
	
	search_columns = ['journal_number', 'description', 'reference']
	base_order = ('journal_number', 'desc')
	base_filters = [['tenant_id', lambda: session.get('tenant_id')]]
	
	formatters_columns = {
		'total_debits': lambda x: f"{x:,.2f}" if x else "0.00",
		'total_credits': lambda x: f"{x:,.2f}" if x else "0.00"
	}
	
	@action('post_entries', _('Post Entries'), _('Post selected journal entries'))
	def post_entries(self, ids):
		"""Post selected journal entries"""
		tenant_id = session.get('tenant_id')
		gl_service = GeneralLedgerService(tenant_id, session.get('user_id'))
		
		posted_count = 0
		errors = []
		
		for journal_id in ids:
			try:
				success = gl_service.post_journal_entry(journal_id)
				if success:
					posted_count += 1
			except GLServiceException as e:
				errors.append(f"Journal {journal_id}: {str(e)}")
		
		if posted_count > 0:
			flash(f"Successfully posted {posted_count} journal entries", 'success')
		
		if errors:
			flash(f"Errors: {'; '.join(errors)}", 'error')
		
		return redirect(url_for('GLJournalEntryView.list'))


# =====================================
# FINANCIAL REPORTING VIEWS
# =====================================

class FinancialReportsView(BaseView):
	"""Financial reports dashboard"""
	
	@expose('/')
	@has_access
	def index(self):
		"""Financial reports dashboard"""
		return self.render_template(
			'general_ledger/financial_reports.html',
			title=_('Financial Reports')
		)
	
	@expose('/trial_balance')
	@has_access
	def trial_balance(self):
		"""Generate and display trial balance"""
		tenant_id = session.get('tenant_id')
		if not tenant_id:
			flash(_('Tenant not selected'), 'error')
			return redirect(url_for('FinancialReportsView.index'))
		
		# Get parameters
		as_of_date = request.args.get('as_of_date')
		account_type = request.args.get('account_type', '')
		include_zero = request.args.get('include_zero', 'false') == 'true'
		currency = request.args.get('currency', 'USD')
		
		try:
			gl_service = GeneralLedgerService(tenant_id, session.get('user_id'))
			
			# Set default date if not provided
			if not as_of_date:
				as_of_date = date.today()
			else:
				as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
			
			# Create parameters
			params = TrialBalanceParams(
				as_of_date=as_of_date,
				account_type_filter=AccountTypeEnum(account_type) if account_type else None,
				include_zero_balances=include_zero,
				currency=CurrencyEnum(currency)
			)
			
			# Generate trial balance
			trial_balance = gl_service.generate_trial_balance(params)
			
			if request.is_json:
				return jsonify({
					'success': True,
					'data': trial_balance.data,
					'metadata': trial_balance.metadata
				})
			
			return self.render_template(
				'general_ledger/trial_balance.html',
				trial_balance=trial_balance,
				as_of_date=as_of_date.isoformat(),
				account_type=account_type,
				include_zero=include_zero,
				currency=currency,
				title=_('Trial Balance')
			)
		
		except Exception as e:
			logger.error(f"Error generating trial balance: {e}")
			if request.is_json:
				return jsonify({'error': str(e)}), 400
			else:
				flash(_('Error generating trial balance'), 'error')
				return self.render_template('general_ledger/trial_balance.html')


# =====================================
# DASHBOARD AND ANALYTICS VIEWS
# =====================================

class GLDashboardView(BaseView):
	"""General Ledger main dashboard"""
	
	@expose('/')
	@has_access
	def index(self):
		"""Main General Ledger dashboard"""
		tenant_id = session.get('tenant_id')
		if not tenant_id:
			flash(_('Tenant not selected'), 'error')
			return redirect(url_for('index'))
		
		try:
			gl_service = GeneralLedgerService(tenant_id, session.get('user_id'))
			
			# Get dashboard metrics
			dashboard_data = self._get_dashboard_metrics(gl_service)
			
			return self.render_template(
				'general_ledger/dashboard.html',
				dashboard_data=dashboard_data,
				title=_('General Ledger Dashboard')
			)
		
		except Exception as e:
			logger.error(f"Error loading GL dashboard: {e}")
			flash(_('Error loading dashboard'), 'error')
			return self.render_template('general_ledger/dashboard.html', dashboard_data={})
	
	def _get_dashboard_metrics(self, gl_service: GeneralLedgerService) -> Dict[str, Any]:
		"""Get comprehensive dashboard metrics"""
		today = date.today()
		
		try:
			# Account summary
			accounts = gl_service.get_chart_of_accounts()
			active_accounts = len([acc for acc in accounts if acc.is_active])
			header_accounts = len([acc for acc in accounts if acc.is_header])
			
			# Period information
			current_period = gl_service.get_current_period()
			open_periods = gl_service.get_open_periods()
			
			# Trial balance summary
			trial_balance_params = TrialBalanceParams(as_of_date=today)
			trial_balance = gl_service.generate_trial_balance(trial_balance_params)
			
			return {
				'accounts': {
					'total_accounts': len(accounts),
					'active_accounts': active_accounts,
					'header_accounts': header_accounts,
					'detail_accounts': active_accounts - header_accounts
				},
				'periods': {
					'current_period': current_period.period_name if current_period else 'None',
					'current_status': current_period.status.value if current_period else 'N/A',
					'open_periods_count': len(open_periods)
				},
				'trial_balance': {
					'total_debits': trial_balance.data['totals']['total_debits'],
					'total_credits': trial_balance.data['totals']['total_credits'],
					'balanced': trial_balance.metadata['balanced'],
					'variance': trial_balance.metadata['variance']
				},
				'last_updated': datetime.now().isoformat()
			}
		
		except Exception as e:
			logger.error(f"Error calculating dashboard metrics: {e}")
			return {
				'error': str(e),
				'last_updated': datetime.now().isoformat()
			}


# Export all view classes for blueprint registration
__all__ = [
	'GLAccountTypeView',
	'GLAccountView', 
	'ChartOfAccountsView',
	'GLJournalEntryView',
	'FinancialReportsView',
	'GLDashboardView'
]