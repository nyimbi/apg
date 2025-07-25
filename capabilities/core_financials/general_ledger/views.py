"""
General Ledger Views

Flask-AppBuilder views for General Ledger functionality including
chart of accounts, journal entries, and financial reporting.
"""

from flask import flash, redirect, request, url_for, jsonify
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface  
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget
from wtforms import Form, StringField, SelectField, DecimalField, DateField, TextAreaField, BooleanField
from wtforms.validators import DataRequired, NumberRange
from datetime import date, datetime
from typing import Dict, List, Any

from .models import (
	CFGLAccount, CFGLAccountType, CFGLPeriod, CFGLJournalEntry, 
	CFGLJournalLine, CFGLPosting, CFGLTrialBalance
)
from .service import GeneralLedgerService
from ...auth_rbac.models import db


class GLAccountModelView(ModelView):
	"""General Ledger Chart of Accounts View"""
	
	datamodel = SQLAInterface(CFGLAccount)
	
	list_title = "Chart of Accounts"
	show_title = "Account Details"
	add_title = "Add Account"
	edit_title = "Edit Account"
	
	list_columns = [
		'account_code', 'account_name', 'account_type.type_name', 
		'parent_account.account_name', 'current_balance', 'is_active'
	]
	
	show_columns = [
		'account_code', 'account_name', 'description', 'account_type',
		'parent_account', 'is_active', 'is_header', 'allow_posting',
		'current_balance', 'ytd_balance', 'opening_balance',
		'currency_code', 'tax_code', 'cost_center_required', 'department_required',
		'level', 'path', 'created_on', 'updated_on'
	]
	
	add_columns = [
		'account_code', 'account_name', 'description', 'account_type',
		'parent_account', 'is_active', 'is_header', 'allow_posting',
		'opening_balance', 'currency_code', 'tax_code', 
		'cost_center_required', 'department_required'
	]
	
	edit_columns = [
		'account_name', 'description', 'is_active', 'is_header', 'allow_posting',
		'currency_code', 'tax_code', 'cost_center_required', 'department_required'
	]
	
	search_columns = ['account_code', 'account_name', 'description']
	
	order_columns = ['account_code', 'account_name', 'account_type', 'current_balance']
	
	base_order = ('account_code', 'asc')
	
	formatters_columns = {
		'current_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'ytd_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'opening_balance': lambda x: f"${x:,.2f}" if x else "$0.00"
	}
	
	def pre_add(self, item):
		"""Set tenant_id before adding account"""
		item.tenant_id = self.get_tenant_id()
		
		# Calculate hierarchy information
		if item.parent_account_id:
			parent = CFGLAccount.query.get(item.parent_account_id)
			if parent:
				item.level = parent.level + 1
				item.path = f"{parent.path}/{item.account_code}" if parent.path else item.account_code
		else:
			item.level = 0
			item.path = item.account_code
	
	def pre_update(self, item):
		"""Update hierarchy information if needed"""
		if item.parent_account_id:
			parent = CFGLAccount.query.get(item.parent_account_id)
			if parent:
				item.level = parent.level + 1
				item.path = f"{parent.path}/{item.account_code}" if parent.path else item.account_code
		else:
			item.level = 0
			item.path = item.account_code
	
	@expose('/hierarchy')
	@has_access
	def hierarchy(self):
		"""Show chart of accounts in hierarchical view"""
		gl_service = GeneralLedgerService(self.get_tenant_id())
		accounts = gl_service.get_chart_of_accounts()
		
		# Build hierarchy tree
		account_tree = self._build_account_tree(accounts)
		
		return self.render_template(
			'gl_account_hierarchy.html',
			accounts=account_tree,
			title="Chart of Accounts Hierarchy"
		)
	
	def _build_account_tree(self, accounts: List[CFGLAccount]) -> List[Dict]:
		"""Build hierarchical account tree"""
		account_dict = {acc.account_id: acc for acc in accounts}
		tree = []
		
		for account in accounts:
			if not account.parent_account_id:
				tree.append(self._build_account_node(account, account_dict))
		
		return tree
	
	def _build_account_node(self, account: CFGLAccount, account_dict: Dict) -> Dict:
		"""Build single account node with children"""
		node = {
			'account': account,
			'children': []
		}
		
		# Find children
		for acc_id, acc in account_dict.items():
			if acc.parent_account_id == account.account_id:
				node['children'].append(self._build_account_node(acc, account_dict))
		
		return node


class GLAccountTypeModelView(ModelView):
	"""Account Type Management View"""
	
	datamodel = SQLAInterface(CFGLAccountType)
	
	list_title = "Account Types"
	show_title = "Account Type Details"
	
	list_columns = ['type_code', 'type_name', 'normal_balance', 'is_balance_sheet', 'sort_order']
	show_columns = ['type_code', 'type_name', 'description', 'normal_balance', 'is_balance_sheet', 'sort_order']
	add_columns = ['type_code', 'type_name', 'description', 'normal_balance', 'is_balance_sheet', 'sort_order']
	edit_columns = ['type_name', 'description', 'normal_balance', 'is_balance_sheet', 'sort_order']
	
	def pre_add(self, item):
		item.tenant_id = self.get_tenant_id()


class GLPeriodModelView(ModelView):
	"""Accounting Period Management View"""
	
	datamodel = SQLAInterface(CFGLPeriod)
	
	list_title = "Accounting Periods"
	show_title = "Period Details"
	
	list_columns = [
		'fiscal_year', 'period_number', 'period_name', 
		'start_date', 'end_date', 'status'
	]
	
	show_columns = [
		'fiscal_year', 'period_number', 'period_name',
		'start_date', 'end_date', 'status', 'is_adjustment_period',
		'closed_date', 'closed_by', 'closing_entries_count'
	]
	
	add_columns = [
		'fiscal_year', 'period_number', 'period_name',
		'start_date', 'end_date', 'is_adjustment_period'
	]
	
	edit_columns = ['period_name', 'start_date', 'end_date']
	
	base_order = ('fiscal_year', 'desc'), ('period_number', 'asc')
	
	def pre_add(self, item):
		item.tenant_id = self.get_tenant_id()
	
	@expose('/close/<period_id>')
	@has_access
	def close_period(self, period_id):
		"""Close accounting period"""
		period = CFGLPeriod.query.get(period_id)
		if period and period.tenant_id == self.get_tenant_id():
			if period.status == 'Open':
				period.close_period(self.get_user_id())
				db.session.commit()
				flash(f"Period {period.period_name} has been closed", "success")
			else:
				flash("Period is already closed", "warning")
		else:
			flash("Period not found", "error")
		
		return redirect(url_for('GLPeriodModelView.list'))


class GLJournalEntryModelView(ModelView):
	"""Journal Entry Management View"""
	
	datamodel = SQLAInterface(CFGLJournalEntry)
	
	list_title = "Journal Entries"
	show_title = "Journal Entry Details"
	add_title = "Add Journal Entry"
	edit_title = "Edit Journal Entry"
	
	list_columns = [
		'journal_number', 'description', 'entry_date', 'posting_date',
		'status', 'total_debits', 'total_credits', 'posted'
	]
	
	show_columns = [
		'journal_number', 'description', 'reference', 'entry_date', 'posting_date',
		'period', 'status', 'source', 'total_debits', 'total_credits', 'line_count',
		'requires_approval', 'approved', 'approved_by', 'approved_date',
		'posted', 'posted_by', 'posted_date', 'reversed'
	]
	
	add_columns = [
		'description', 'reference', 'entry_date', 'posting_date',
		'source', 'requires_approval'
	]
	
	edit_columns = ['description', 'reference', 'entry_date', 'posting_date']
	
	search_columns = ['journal_number', 'description', 'reference']
	order_columns = ['journal_number', 'entry_date', 'posting_date', 'total_debits']
	
	base_order = ('posting_date', 'desc'), ('journal_number', 'desc')
	
	formatters_columns = {
		'total_debits': lambda x: f"${x:,.2f}" if x else "$0.00",
		'total_credits': lambda x: f"${x:,.2f}" if x else "$0.00"
	}
	
	def pre_add(self, item):
		item.tenant_id = self.get_tenant_id()
		
		# Set period based on posting date
		gl_service = GeneralLedgerService(self.get_tenant_id())
		period = gl_service.get_current_period(item.posting_date)
		if period:
			item.period_id = period.period_id
		
		# Generate journal number if not provided
		if not item.journal_number:
			item.journal_number = gl_service._generate_journal_number()
	
	@expose('/post/<journal_id>')
	@has_access
	def post_entry(self, journal_id):
		"""Post journal entry"""
		try:
			gl_service = GeneralLedgerService(self.get_tenant_id())
			gl_service.post_journal_entry(journal_id, self.get_user_id())
			flash("Journal entry posted successfully", "success")
		except Exception as e:
			flash(f"Error posting journal entry: {str(e)}", "error")
		
		return redirect(url_for('GLJournalEntryModelView.show', pk=journal_id))
	
	@expose('/lines/<journal_id>')
	@has_access
	def show_lines(self, journal_id):
		"""Show journal entry lines"""
		journal = CFGLJournalEntry.query.get(journal_id)
		if not journal or journal.tenant_id != self.get_tenant_id():
			flash("Journal entry not found", "error")
			return redirect(url_for('GLJournalEntryModelView.list'))
		
		return self.render_template(
			'gl_journal_lines.html',
			journal=journal,
			title=f"Journal Entry Lines - {journal.journal_number}"
		)


class GLJournalLineModelView(ModelView):
	"""Journal Line Management View"""
	
	datamodel = SQLAInterface(CFGLJournalLine)
	
	list_title = "Journal Lines"
	
	list_columns = [
		'journal_entry.journal_number', 'line_number', 'account.account_code',
		'account.account_name', 'debit_amount', 'credit_amount', 'description'
	]
	
	show_columns = [
		'journal_entry', 'line_number', 'account', 'description',
		'debit_amount', 'credit_amount', 'cost_center', 'department',
		'project', 'reference_type', 'reference_number', 'tax_code', 'tax_amount'
	]
	
	formatters_columns = {
		'debit_amount': lambda x: f"${x:,.2f}" if x and x > 0 else "",
		'credit_amount': lambda x: f"${x:,.2f}" if x and x > 0 else "",
		'tax_amount': lambda x: f"${x:,.2f}" if x and x > 0 else ""
	}
	
	base_order = ('journal_entry.posting_date', 'desc'), ('line_number', 'asc')


class GLPostingModelView(ModelView):
	"""Posted Transactions View"""
	
	datamodel = SQLAInterface(CFGLPosting)
	
	list_title = "Posted Transactions"
	show_title = "Posting Details"
	
	list_columns = [
		'posting_date', 'account.account_code', 'account.account_name',
		'description', 'debit_amount', 'credit_amount', 'reference'
	]
	
	show_columns = [
		'posting_date', 'account', 'description', 'reference',
		'debit_amount', 'credit_amount', 'journal_entry', 'period',
		'is_posted', 'is_reversed', 'posted_by', 'posted_date'
	]
	
	search_columns = ['description', 'reference', 'account.account_code']
	order_columns = ['posting_date', 'account.account_code', 'debit_amount', 'credit_amount']
	
	base_order = ('posting_date', 'desc'), ('account.account_code', 'asc')
	
	formatters_columns = {
		'debit_amount': lambda x: f"${x:,.2f}" if x and x > 0 else "",
		'credit_amount': lambda x: f"${x:,.2f}" if x and x > 0 else ""
	}


class GLTrialBalanceView(BaseView):
	"""Trial Balance Report View"""
	
	route_base = "/gl/trial_balance"
	default_view = 'index'
	
	@expose('/')
	@has_access 
	def index(self):
		"""Display trial balance form and results"""
		as_of_date = request.args.get('as_of_date')
		account_type = request.args.get('account_type')
		
		if as_of_date:
			try:
				as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
			except ValueError:
				as_of_date = date.today()
		else:
			as_of_date = date.today()
		
		gl_service = GeneralLedgerService(self.get_tenant_id())
		trial_balance = gl_service.generate_trial_balance(as_of_date, account_type)
		account_types = gl_service.get_account_types()
		
		return self.render_template(
			'gl_trial_balance.html',
			trial_balance=trial_balance,
			account_types=account_types,
			as_of_date=as_of_date,
			selected_type=account_type,
			title="Trial Balance"
		)
	
	@expose('/export')
	@has_access
	def export(self):
		"""Export trial balance to CSV"""
		as_of_date = request.args.get('as_of_date', date.today().isoformat())
		account_type = request.args.get('account_type')
		
		try:
			as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
		except ValueError:
			as_of_date = date.today()
		
		gl_service = GeneralLedgerService(self.get_tenant_id())
		trial_balance = gl_service.generate_trial_balance(as_of_date, account_type)
		
		# Create CSV response
		import io
		import csv
		from flask import Response
		
		output = io.StringIO()
		writer = csv.writer(output)
		
		# Headers
		writer.writerow(['Account Code', 'Account Name', 'Account Type', 'Debit Balance', 'Credit Balance'])
		
		# Data
		for account in trial_balance['accounts']:
			writer.writerow([
				account['account_code'],
				account['account_name'],
				account['account_type'],
				float(account['debit_balance']),
				float(account['credit_balance'])
			])
		
		# Totals
		writer.writerow(['', '', 'TOTALS', float(trial_balance['total_debits']), float(trial_balance['total_credits'])])
		
		output.seek(0)
		
		return Response(
			output.getvalue(),
			mimetype='text/csv',
			headers={
				'Content-Disposition': f'attachment; filename=trial_balance_{as_of_date}.csv'
			}
		)


class GLDashboardView(BaseView):
	"""General Ledger Dashboard"""
	
	route_base = "/gl/dashboard"
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Display GL dashboard"""
		gl_service = GeneralLedgerService(self.get_tenant_id())
		
		# Get key metrics
		current_period = gl_service.get_current_period()
		open_periods = gl_service.get_open_periods()
		unposted_journals = gl_service.get_journal_entries(status='Draft', limit=50)
		
		# Get account type balances
		trial_balance = gl_service.generate_trial_balance()
		account_balances = {}
		
		for account in trial_balance['accounts']:
			acc_type = account['account_type']
			if acc_type not in account_balances:
				account_balances[acc_type] = {'debit': 0, 'credit': 0}
			
			account_balances[acc_type]['debit'] += float(account['debit_balance'])
			account_balances[acc_type]['credit'] += float(account['credit_balance'])
		
		return self.render_template(
			'gl_dashboard.html',
			current_period=current_period,
			open_periods=open_periods,
			unposted_journals=unposted_journals[:10],  # Show only first 10
			account_balances=account_balances,
			trial_balance=trial_balance,
			title="General Ledger Dashboard"
		)
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID - implement based on your auth system"""
		# TODO: Implement tenant resolution logic
		return "default_tenant"
	
	def get_user_id(self) -> str:
		"""Get current user ID - implement based on your auth system"""
		# TODO: Implement user resolution logic  
		return "default_user"


# Add common methods to all views
for view_class in [GLAccountModelView, GLAccountTypeModelView, GLPeriodModelView, 
				   GLJournalEntryModelView, GLJournalLineModelView, GLPostingModelView,
				   GLTrialBalanceView, GLDashboardView]:
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID - implement based on your auth system"""
		# TODO: Implement tenant resolution logic
		return "default_tenant"
	
	def get_user_id(self) -> str:
		"""Get current user ID - implement based on your auth system"""
		# TODO: Implement user resolution logic  
		return "default_user"
	
	view_class.get_tenant_id = get_tenant_id
	view_class.get_user_id = get_user_id