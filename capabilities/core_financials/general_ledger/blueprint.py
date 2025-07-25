"""
General Ledger Blueprint

Flask blueprint registration for General Ledger sub-capability.
Registers all views, API endpoints, and URL routes.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

from .views import (
	GLAccountModelView, GLAccountTypeModelView, GLPeriodModelView,
	GLJournalEntryModelView, GLJournalLineModelView, GLPostingModelView,
	GLTrialBalanceView, GLDashboardView
)


def register_views(appbuilder: AppBuilder):
	"""Register General Ledger views with Flask-AppBuilder"""
	
	# Chart of Accounts Management
	appbuilder.add_view(
		GLAccountModelView,
		"Chart of Accounts",
		icon="fa-list-alt",
		category="General Ledger",
		category_icon="fa-book"
	)
	
	appbuilder.add_view(
		GLAccountTypeModelView,
		"Account Types",
		icon="fa-tags",
		category="General Ledger"
	)
	
	# Period Management
	appbuilder.add_view(
		GLPeriodModelView,
		"Accounting Periods",
		icon="fa-calendar",
		category="General Ledger"
	)
	
	# Transaction Management
	appbuilder.add_view(
		GLJournalEntryModelView,
		"Journal Entries",
		icon="fa-book",
		category="General Ledger"
	)
	
	appbuilder.add_view(
		GLJournalLineModelView,
		"Journal Lines",
		icon="fa-list",
		category="General Ledger"
	)
	
	appbuilder.add_view(
		GLPostingModelView,
		"Posted Transactions",
		icon="fa-check-circle",
		category="General Ledger"
	)
	
	# Reporting
	appbuilder.add_view_no_menu(GLTrialBalanceView())
	appbuilder.add_link(
		"Trial Balance",
		href="/gl/trial_balance/",
		icon="fa-balance-scale",
		category="General Ledger"
	)
	
	# Dashboard
	appbuilder.add_view_no_menu(GLDashboardView())
	appbuilder.add_link(
		"GL Dashboard",
		href="/gl/dashboard/",
		icon="fa-dashboard",
		category="General Ledger"
	)


def create_blueprint() -> Blueprint:
	"""Create Flask blueprint for General Ledger"""
	
	gl_bp = Blueprint(
		'general_ledger',
		__name__,
		url_prefix='/gl',
		template_folder='templates',
		static_folder='static'
	)
	
	return gl_bp


def register_permissions(appbuilder: AppBuilder):
	"""Register General Ledger permissions"""
	
	permissions = [
		# View permissions
		('can_list', 'GLAccountModelView'),
		('can_show', 'GLAccountModelView'),
		('can_add', 'GLAccountModelView'),
		('can_edit', 'GLAccountModelView'),
		('can_delete', 'GLAccountModelView'),
		
		('can_list', 'GLJournalEntryModelView'),
		('can_show', 'GLJournalEntryModelView'),
		('can_add', 'GLJournalEntryModelView'),
		('can_edit', 'GLJournalEntryModelView'),
		('can_delete', 'GLJournalEntryModelView'),
		('can_post_entry', 'GLJournalEntryModelView'),
		
		('can_list', 'GLPostingModelView'),
		('can_show', 'GLPostingModelView'),
		
		('can_list', 'GLPeriodModelView'),
		('can_show', 'GLPeriodModelView'),
		('can_add', 'GLPeriodModelView'),
		('can_edit', 'GLPeriodModelView'),
		('can_close_period', 'GLPeriodModelView'),
		
		# Report permissions
		('can_index', 'GLTrialBalanceView'),
		('can_export', 'GLTrialBalanceView'),
		('can_index', 'GLDashboardView'),
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
		if not perm:
			appbuilder.sm.add_permission_view_menu(permission_name, view_name)


def get_menu_structure():
	"""Get menu structure for General Ledger"""
	
	return {
		'name': 'General Ledger',
		'icon': 'fa-book',
		'items': [
			{
				'name': 'GL Dashboard',
				'href': '/gl/dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'can_index on GLDashboardView'
			},
			{
				'name': 'Chart of Accounts',
				'href': '/glaccountmodelview/list/',
				'icon': 'fa-list-alt',
				'permission': 'can_list on GLAccountModelView'
			},
			{
				'name': 'Journal Entries',
				'href': '/gljournalentrymodelview/list/',
				'icon': 'fa-book',
				'permission': 'can_list on GLJournalEntryModelView'
			},
			{
				'name': 'Posted Transactions',
				'href': '/glpostingmodelview/list/',
				'icon': 'fa-check-circle',
				'permission': 'can_list on GLPostingModelView'
			},
			{
				'name': 'Trial Balance',
				'href': '/gl/trial_balance/',
				'icon': 'fa-balance-scale',
				'permission': 'can_index on GLTrialBalanceView'
			},
			{
				'name': 'Accounting Periods',
				'href': '/glperiodmodelview/list/',
				'icon': 'fa-calendar',
				'permission': 'can_list on GLPeriodModelView'
			}
		]
	}


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize General Ledger sub-capability"""
	
	# Register views
	register_views(appbuilder)
	
	# Register permissions
	register_permissions(appbuilder)
	
	# Initialize default data if needed
	_init_default_data(appbuilder)


def _init_default_data(appbuilder: AppBuilder):
	"""Initialize default GL data if needed"""
	
	from .models import CFGLAccountType
	from ...auth_rbac.models import db
	
	# Create default account types if they don't exist
	default_types = [
		{
			'type_code': 'A',
			'type_name': 'Asset',
			'description': 'Assets - resources owned by the company',
			'normal_balance': 'Debit',
			'is_balance_sheet': True,
			'sort_order': 1
		},
		{
			'type_code': 'L',
			'type_name': 'Liability',
			'description': 'Liabilities - debts and obligations',
			'normal_balance': 'Credit',
			'is_balance_sheet': True,
			'sort_order': 2
		},
		{
			'type_code': 'E',
			'type_name': 'Equity',
			'description': 'Equity - ownership interest',
			'normal_balance': 'Credit',
			'is_balance_sheet': True,
			'sort_order': 3
		},
		{
			'type_code': 'R',
			'type_name': 'Revenue',
			'description': 'Revenue - income from operations',
			'normal_balance': 'Credit',
			'is_balance_sheet': False,
			'sort_order': 4
		},
		{
			'type_code': 'X',
			'type_name': 'Expense',
			'description': 'Expenses - costs of operations',
			'normal_balance': 'Debit',
			'is_balance_sheet': False,
			'sort_order': 5
		}
	]
	
	try:
		# Check if account types already exist (use a default tenant for now)
		existing_types = CFGLAccountType.query.filter_by(tenant_id='default_tenant').count()
		
		if existing_types == 0:
			for type_data in default_types:
				account_type = CFGLAccountType(
					tenant_id='default_tenant',
					**type_data
				)
				db.session.add(account_type)
			
			db.session.commit()
			print("Default GL account types created")
			
	except Exception as e:
		print(f"Error initializing default GL data: {e}")
		db.session.rollback()